"""
Portfolio Monitor Agent
────────────────────────
Challenge overcome: Knowing WHEN to exit is as hard as knowing when to enter.
Common mistakes:
- Holding losers too long (belief that AI estimate is still right)
- Exiting winners too early (risk aversion)
- Not adjusting for new information that invalidates the thesis

This agent:
1. Continuously monitors all open positions
2. Updates probability estimates as markets move and news arrives
3. Applies exit rules: stop-loss, take-profit, thesis-invalidation
4. Uses AI to detect when the original rationale is no longer valid
"""
import asyncio
import json
from dataclasses import dataclass
from agents.base_agent import BaseAgent
from core.kalshi_client import KalshiClient
from core.database import get_open_positions
from config.settings import TOTAL_CAPITAL_USD
import structlog

log = structlog.get_logger()


MONITOR_SYSTEM = """You are a position monitor for a prediction market hedge fund.

Given an open position and current market data, determine if we should EXIT.

Exit criteria:
1. STOP LOSS: Current price has moved 40%+ against us (we were wrong)
2. TAKE PROFIT: We've captured 70%+ of the max possible edge
3. THESIS BROKEN: New information invalidates our original reasoning
4. TIME DECAY: Less than 24 hours to resolution, uncertainty remains high
5. LIQUIDITY: Spread has widened dramatically (hard to exit)

Return JSON:
{
  "action": "hold" | "exit" | "reduce",
  "urgency": "immediate" | "soon" | "monitor",
  "reason": "...",
  "confidence": 0.0-1.0
}
"""


@dataclass
class PositionStatus:
    ticker: str
    side: str
    contracts: int
    entry_price: int
    current_price: int
    unrealized_pnl_usd: float
    pnl_pct: float
    action: str        # "hold" | "exit" | "reduce"
    urgency: str
    reason: str


class PortfolioMonitorAgent(BaseAgent):
    def __init__(self, kalshi: KalshiClient, executor=None):
        super().__init__("PortfolioMonitor")
        self.kalshi = kalshi
        self.executor = executor   # TradeExecutorAgent for closing positions

    async def run(self, check_interval: int = 120):
        """Continuously monitor all open positions."""
        log.info("monitor_start")
        while True:
            try:
                await self.check_all_positions()
            except Exception as e:
                log.error("monitor_error", error=str(e))
            await asyncio.sleep(check_interval)

    async def check_all_positions(self) -> list[PositionStatus]:
        """Check every open position and take action if needed."""
        positions = await get_open_positions()
        if not positions:
            return []

        log.info("monitor_checking", count=len(positions))
        tasks = [self._check_position(p) for p in positions]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        statuses = []
        for pos, status in zip(positions, results):
            if isinstance(status, Exception):
                log.error("monitor_position_error", ticker=pos["ticker"], error=str(status))
                continue
            statuses.append(status)

            if status.action == "exit" and status.urgency in ("immediate", "soon"):
                await self._exit_position(status)
            elif status.action == "reduce":
                await self._reduce_position(status)

        return statuses

    async def _check_position(self, position: dict) -> PositionStatus:
        ticker = position["ticker"]
        side = position["side"]
        contracts = position["contracts"]
        entry_price = position["avg_price_cents"]

        # Get current market
        try:
            market_data = await self.kalshi.get_market(ticker)
            market = market_data.get("market", {})
        except Exception as e:
            log.warning("monitor_market_fetch_failed", ticker=ticker, error=str(e))
            return PositionStatus(
                ticker=ticker, side=side, contracts=contracts,
                entry_price=entry_price, current_price=entry_price,
                unrealized_pnl_usd=0, pnl_pct=0,
                action="hold", urgency="monitor", reason="Failed to fetch market data"
            )

        # Current value
        if side == "yes":
            current_price = market.get("yes_bid", entry_price)  # sell at bid
        else:
            current_price = market.get("no_bid", entry_price)

        pnl_cents = (current_price - entry_price) * contracts
        pnl_usd = pnl_cents / 100
        pnl_pct = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0

        # Quick rule-based checks (no AI needed)
        quick_action = self._apply_exit_rules(
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            pnl_pct=pnl_pct,
            market=market,
        )

        if quick_action:
            return PositionStatus(
                ticker=ticker, side=side, contracts=contracts,
                entry_price=entry_price, current_price=current_price,
                unrealized_pnl_usd=pnl_usd, pnl_pct=pnl_pct,
                **quick_action,
            )

        # AI review for borderline cases (PnL between -20% and +50%)
        ai_decision = await self._ai_review(position, market, pnl_pct, pnl_usd)

        return PositionStatus(
            ticker=ticker, side=side, contracts=contracts,
            entry_price=entry_price, current_price=current_price,
            unrealized_pnl_usd=pnl_usd, pnl_pct=pnl_pct,
            action=ai_decision.get("action", "hold"),
            urgency=ai_decision.get("urgency", "monitor"),
            reason=ai_decision.get("reason", ""),
        )

    def _apply_exit_rules(
        self,
        side: str,
        entry_price: int,
        current_price: int,
        pnl_pct: float,
        market: dict,
    ) -> dict | None:
        """Fast rule-based exit checks."""

        # Stop loss: -40%
        if pnl_pct < -40:
            return {"action": "exit", "urgency": "immediate", "reason": f"Stop loss hit ({pnl_pct:.1f}%)"}

        # Take profit: YES moved from 30 to 80 — we've captured most of the edge
        if pnl_pct > 70:
            return {"action": "exit", "urgency": "soon", "reason": f"Take profit ({pnl_pct:.1f}%)"}

        # Market about to close
        close_time = market.get("close_time", "")
        if close_time:
            from datetime import datetime, timezone
            try:
                close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                hours_remaining = (close_dt - now).total_seconds() / 3600
                if hours_remaining < 2 and abs(pnl_pct) < 10:
                    return {"action": "exit", "urgency": "soon",
                            "reason": f"Near resolution with small edge ({hours_remaining:.1f}h remaining)"}
            except ValueError:
                pass

        return None

    async def _ai_review(
        self,
        position: dict,
        market: dict,
        pnl_pct: float,
        pnl_usd: float,
    ) -> dict:
        prompt = f"""Open position review:

Ticker: {position['ticker']}
Title: {market.get('title', position['ticker'])}
Side: {position['side'].upper()}
Contracts: {position['contracts']}
Entry price: {position['avg_price_cents']}¢
Current price: {market.get(position['side'] + '_bid', position['avg_price_cents'])}¢
Unrealized P&L: ${pnl_usd:.2f} ({pnl_pct:.1f}%)
Close time: {market.get('close_time', 'unknown')}
Market volume today: ${market.get('volume', 0)/100:.2f}
Open interest: ${market.get('open_interest', 0)/100:.2f}

Should we hold, exit, or reduce this position?"""

        result = await self.analyze(prompt, system=MONITOR_SYSTEM, max_turns=1)

        try:
            return self._extract_json(result)
        except (json.JSONDecodeError, AttributeError):
            pass
        return {"action": "hold", "urgency": "monitor", "reason": "AI review inconclusive"}

    async def _exit_position(self, status: PositionStatus):
        log.info("monitor_exit", ticker=status.ticker, reason=status.reason)
        if self.executor:
            await self.executor.close_position(
                ticker=status.ticker,
                side=status.side,
                contracts=status.contracts,
                reason=status.reason,
            )

    async def _reduce_position(self, status: PositionStatus):
        reduce_contracts = max(1, status.contracts // 2)
        log.info("monitor_reduce", ticker=status.ticker, contracts=reduce_contracts)
        if self.executor:
            await self.executor.close_position(
                ticker=status.ticker,
                side=status.side,
                contracts=reduce_contracts,
                reason=f"Reduce: {status.reason}",
            )

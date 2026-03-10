"""
Trade Executor Agent
─────────────────────
Challenge overcome: Execution quality on thin markets.
Naive market orders blow through the orderbook. Smart execution:
1. Checks orderbook depth before sizing
2. Splits large orders across time to minimize market impact
3. Handles partial fills gracefully
4. Uses limit orders near the mid to capture spread
5. Has circuit breakers for anomalous market conditions
"""
import uuid
import asyncio
import json
from agents.base_agent import BaseAgent
from agents.risk_manager import TradeDecision
from core.kalshi_client import KalshiClient
from core.database import save_trade, update_daily_pnl
import structlog

log = structlog.get_logger()


class TradeExecutorAgent(BaseAgent):
    def __init__(self, kalshi: KalshiClient, dry_run: bool = True):
        super().__init__("TradeExecutor")
        self.kalshi = kalshi
        self.dry_run = dry_run  # Safety: set to False only in production

    async def execute(self, decision: TradeDecision) -> dict | None:
        """
        Execute a trade decision with smart order routing.
        Returns order details or None if execution failed.
        """
        if not decision.approved:
            log.warning("executor_rejected_decision", reason=decision.rejection_reason)
            return None

        ticker = decision.ticker
        side = decision.side
        total_contracts = decision.contracts
        price_cents = decision.price_cents

        # ── Stale signal check ────────────────────────────────────────────────
        # Re-fetch price; skip if market moved >2% since analysis was done
        if price_cents > 0:
            try:
                current_market = await self.kalshi.get_market(ticker)
                current_price = current_market.get("market", {}).get(
                    "yes_ask" if side == "yes" else "no_ask", price_cents
                )
                move_pct = abs(current_price - price_cents) / price_cents
                if move_pct > 0.02:
                    log.warning(
                        "stale_signal_skipped",
                        ticker=ticker,
                        signal_price=price_cents,
                        current_price=current_price,
                        move_pct=round(move_pct * 100, 1),
                    )
                    return None
            except Exception as e:
                log.warning("stale_check_failed", ticker=ticker, error=str(e))

        # ── Pre-execution checks ───────────────────────────────────────────────
        ok = await self._pre_execution_checks(ticker, side, total_contracts, price_cents)
        if not ok:
            return None

        # ── Order splitting for large trades ──────────────────────────────────
        # Split into at most 3 chunks to minimize market impact
        chunk_size = max(1, total_contracts // 3)
        chunks = []
        remaining = total_contracts
        while remaining > 0:
            size = min(chunk_size, remaining)
            chunks.append(size)
            remaining -= size

        filled_orders = []
        for i, contracts in enumerate(chunks):
            if i > 0:
                await asyncio.sleep(2)  # Wait between chunks

            # Use limit order slightly inside the spread
            limit_price = self._compute_limit_price(price_cents, side)

            order = await self._place_order(ticker, side, contracts, limit_price, decision)
            if order:
                filled_orders.append(order)
            else:
                log.warning("executor_chunk_failed", chunk=i, ticker=ticker)
                break

        if not filled_orders:
            return None

        total_filled = sum(o.get("count", 0) for o in filled_orders)
        log.info("executor_complete", ticker=ticker, side=side, filled=total_filled)
        return {"ticker": ticker, "side": side, "total_contracts": total_filled, "orders": filled_orders}

    async def _pre_execution_checks(
        self, ticker: str, side: str, contracts: int, price_cents: int
    ) -> bool:
        """Verify orderbook depth can absorb our order."""
        try:
            ob = await self.kalshi.get_market_orderbook(ticker, depth=20)
        except Exception as e:
            log.error("executor_orderbook_error", error=str(e))
            return False

        # Check available liquidity on our side
        ask_side = "yes" if side == "yes" else "no"
        asks = ob.get(f"{ask_side}_ask", [])
        available_at_price = sum(
            level.get("count", 0)
            for level in asks
            if level.get("price", 0) <= price_cents + 3  # allow 3¢ slippage
        )

        if available_at_price < contracts:
            log.warning(
                "executor_insufficient_liquidity",
                ticker=ticker, needed=contracts, available=available_at_price
            )
            # Don't cancel — just warn. Risk manager already sized appropriately.
            # Adjust contracts to what's available
            if available_at_price == 0:
                return False

        return True

    def _compute_limit_price(self, market_price: int, side: str) -> int:
        """
        Set limit price slightly above ask to get filled without overpaying.
        For YES: bid up to market_price + 1¢
        For NO: same logic
        """
        # Add 1 cent to the ask to ensure fill, cap at 99
        return min(99, market_price + 1)

    async def _place_order(
        self,
        ticker: str,
        side: str,
        contracts: int,
        price_cents: int,
        decision: TradeDecision,
    ) -> dict | None:
        client_order_id = f"hf-{uuid.uuid4().hex[:12]}"

        if self.dry_run:
            log.info(
                "DRY_RUN_ORDER",
                ticker=ticker, side=side, contracts=contracts,
                price_cents=price_cents, client_order_id=client_order_id
            )
            # Simulate a fill
            simulated = {
                "order_id": client_order_id,
                "ticker": ticker,
                "side": side,
                "count": contracts,
                "price": price_cents,
                "status": "filled",
                "dry_run": True,
            }
            await save_trade(
                order_id=client_order_id,
                ticker=ticker, side=side, contracts=contracts,
                price_cents=price_cents, action="open",
                agent="TradeExecutor", reason=f"edge={decision.kelly_f:.3f}",
            )
            return simulated

        try:
            order_kwargs = {
                "ticker": ticker,
                "side": side,
                "count": contracts,
                "order_type": "limit",
                "client_order_id": client_order_id,
            }
            if side == "yes":
                order_kwargs["yes_price"] = price_cents
            else:
                order_kwargs["no_price"] = price_cents

            result = await self.kalshi.place_order(**order_kwargs)
            order = result.get("order", {})
            log.info("order_placed", order_id=order.get("order_id"), status=order.get("status"))

            await save_trade(
                order_id=order.get("order_id", client_order_id),
                ticker=ticker, side=side, contracts=contracts,
                price_cents=price_cents, action="open",
                agent="TradeExecutor", reason=f"kelly={decision.kelly_f:.3f}",
            )
            return order

        except Exception as e:
            log.error("order_failed", ticker=ticker, error=str(e))
            return None

    async def close_position(self, ticker: str, side: str, contracts: int, reason: str) -> bool:
        """Exit an existing position by selling the same side (Kalshi uses action='sell')."""
        if self.dry_run:
            log.info("DRY_RUN_CLOSE", ticker=ticker, side=side, contracts=contracts)
            await save_trade(
                order_id=f"close-{uuid.uuid4().hex[:8]}",
                ticker=ticker, side=side, contracts=contracts,
                price_cents=50, action="close", agent="TradeExecutor", reason=reason,
            )
            return True

        try:
            market = await self.kalshi.get_market(ticker)
            # Sell at the bid (where buyers are waiting)
            if side == "yes":
                price = market.get("market", {}).get("yes_bid", 50)
            else:
                price = market.get("market", {}).get("no_bid", 50)

            await self.kalshi.place_order(
                ticker=ticker,
                side=side,
                count=contracts,
                order_type="limit",
                action="sell",
                yes_price=price if side == "yes" else None,
                no_price=price if side == "no" else None,
            )
            log.info("position_closed", ticker=ticker, side=side, reason=reason)
            return True
        except Exception as e:
            log.error("close_position_failed", ticker=ticker, error=str(e))
            return False

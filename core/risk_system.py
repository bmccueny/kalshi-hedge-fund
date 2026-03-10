"""
Risk Management System
─────────────────────
Comprehensive risk management including:
- Position sizing (Kelly criterion, fixed fraction)
- Portfolio-level risk limits
- Drawdown protection
- Exposure management
- Volatility-based sizing
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import numpy as np
import structlog
from core.kalshi_client import KalshiClient
from core.database import get_open_positions

log = structlog.get_logger()


@dataclass
class RiskLimits:
    max_position_size_pct: float = 0.2
    max_portfolio_risk_pct: float = 0.1
    max_daily_loss_pct: float = 0.05
    max_positions: int = 10
    max_correlation: float = 0.7
    max_leverage: float = 1.0


@dataclass
class RiskMetrics:
    portfolio_value: float
    daily_pnl: float
    daily_return_pct: float
    current_drawdown_pct: float
    max_drawdown_pct: float
    position_count: int
    largest_position_pct: float
    volatility_20d: float
    var_95: float
    exposure: float


@dataclass
class PositionRisk:
    ticker: str
    contracts: int
    entry_price: float
    current_price: float
    side: str
    pnl: float
    pnl_pct: float
    risk_pct: float
    position_value: float
    position_value_pct: float
    should_close: bool
    close_reason: str | None


class RiskManager:
    """
    Comprehensive risk management system.
    """

    def __init__(
        self,
        portfolio_value: float = 100000.0,
        limits: RiskLimits | None = None,
    ):
        self.portfolio_value = portfolio_value
        self.limits = limits or RiskLimits()
        self.daily_pnl = 0.0
        self.daily_high = portfolio_value
        self.peak_value = portfolio_value
        self.max_drawdown = 0.0
        self.positions: dict[str, dict] = {}
        self.risk_history: list[RiskMetrics] = []

    def calculate_position_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        confidence: float = 0.5,
        method: str = "kelly",
    ) -> float:
        """
        Calculate optimal position size based on edge and win rate.
        
        Methods:
        - kelly: Kelly criterion
        - half_kelly: Conservative Kelly (half)
        - fixed: Fixed fraction
        """
        if method == "kelly":
            if avg_loss == 0 or win_rate == 0:
                return 0.0
            win_loss_ratio = avg_win / avg_loss
            kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
            kelly = max(0, min(kelly, self.limits.max_position_size_pct))
            return kelly * confidence
        
        elif method == "half_kelly":
            return self.calculate_position_size(win_rate, avg_win, avg_loss, confidence, "kelly") * 0.5
        
        elif method == "fixed":
            return self.limits.max_position_size_pct * confidence
        
        return self.limits.max_position_size_pct * 0.1

    def calculate_kelly_size(
        self,
        win_rate: float,
        odds: float,
        fraction: float = 0.25,
    ) -> float:
        """
        Calculate Kelly criterion position size.
        
        K% = W - (1-W)/Odds
        """
        if odds <= 0:
            return 0.0
        
        kelly = win_rate - (1 - win_rate) / odds
        kelly = max(0, kelly)
        
        fractional_kelly = kelly * fraction
        
        return min(fractional_kelly, self.limits.max_position_size_pct)

    def calculate_volatility_size(
        self,
        price: float,
        volatility: float,
        target_risk_pct: float = 0.02,
    ) -> int:
        """
        Calculate position size based on volatility.
        
        Size = (Portfolio * Target Risk) / (Price * Volatility)
        """
        if volatility <= 0:
            volatility = 0.1
        
        position_value = (self.portfolio_value * target_risk_pct) / volatility
        contracts = int(position_value / (price / 100))
        
        return max(1, contracts)

    def check_risk_limits(self, new_position_pct: float) -> tuple[bool, str]:
        """
        Check if adding a new position would violate risk limits.
        """
        if new_position_pct > self.limits.max_position_size_pct:
            return False, f"Position size {new_position_pct:.1%} exceeds max {self.limits.max_position_size_pct:.1%}"

        total_exposure = sum(p.get("position_value_pct", 0) for p in self.positions.values())
        
        if total_exposure + new_position_pct > self.limits.max_leverage:
            return False, f"Total exposure {total_exposure + new_position_pct:.1%} exceeds max {self.limits.max_leverage:.1%}"

        if len(self.positions) >= self.limits.max_positions:
            return False, f"Max positions {self.limits.max_positions} reached"

        if self.daily_pnl < -self.portfolio_value * self.limits.max_daily_loss_pct:
            return False, f"Daily loss limit reached: {self.daily_pnl:.2f}"

        return True, "OK"

    def update_position(
        self,
        ticker: str,
        contracts: int,
        entry_price: float,
        current_price: float,
        side: str,
    ) -> None:
        """Update position tracking."""
        if side == "yes":
            pnl = (current_price - entry_price) * contracts
        else:
            pnl = (entry_price - current_price) * contracts

        position_value = entry_price * contracts / 100
        pnl_pct = (pnl / position_value) * 100 if position_value > 0 else 0
        position_value_pct = position_value / self.portfolio_value

        self.positions[ticker] = {
            "contracts": contracts,
            "entry_price": entry_price,
            "current_price": current_price,
            "side": side,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "position_value": position_value,
            "position_value_pct": position_value_pct,
        }

    def remove_position(self, ticker: str) -> float:
        """Remove position and return PnL."""
        if ticker in self.positions:
            pnl = self.positions[ticker]["pnl"]
            del self.positions[ticker]
            return pnl
        return 0.0

    def calculate_risk_metrics(self) -> RiskMetrics:
        """Calculate current portfolio risk metrics."""
        total_pnl = sum(p.get("pnl", 0) for p in self.positions.values())
        self.daily_pnl = total_pnl
        
        self.daily_high = max(self.daily_high, self.portfolio_value + total_pnl)
        current_value = self.portfolio_value + total_pnl
        
        if self.daily_high > 0:
            drawdown = (self.daily_high - current_value) / self.daily_high * 100
            self.max_drawdown = max(self.max_drawdown, drawdown)

        daily_return = total_pnl / self.portfolio_value * 100 if self.portfolio_value > 0 else 0

        position_values = [p.get("position_value_pct", 0) for p in self.positions.values()]
        largest_position = max(position_values) if position_values else 0

        exposure = sum(position_values)

        var_95 = 1.65 * 0.02 * self.portfolio_value

        return RiskMetrics(
            portfolio_value=current_value,
            daily_pnl=total_pnl,
            daily_return_pct=daily_return,
            current_drawdown_pct=drawdown if 'drawdown' in locals() else 0,
            max_drawdown_pct=self.max_drawdown,
            position_count=len(self.positions),
            largest_position_pct=largest_position,
            volatility_20d=0.02,
            var_95=var_95,
            exposure=exposure,
        )

    def get_positions_to_close(self) -> list[PositionRisk]:
        """Identify positions that should be closed based on risk rules."""
        to_close = []
        metrics = self.calculate_risk_metrics()

        for ticker, pos in self.positions.items():
            should_close = False
            reason = None

            if pos.get("pnl_pct", 0) < -10:
                should_close = True
                reason = "stop_loss_10pct"
            elif pos.get("position_value_pct", 0) > self.limits.max_position_size_pct:
                should_close = True
                reason = "position_size_limit"
            elif metrics.current_drawdown_pct > self.limits.max_daily_loss_pct * 2:
                should_close = True
                reason = "drawdown_protection"

            if should_close:
                to_close.append(PositionRisk(
                    ticker=ticker,
                    contracts=pos.get("contracts", 0),
                    entry_price=pos.get("entry_price", 0),
                    current_price=pos.get("current_price", 0),
                    side=pos.get("side", "yes"),
                    pnl=pos.get("pnl", 0),
                    pnl_pct=pos.get("pnl_pct", 0),
                    risk_pct=pos.get("position_value_pct", 0),
                    position_value=pos.get("position_value", 0),
                    position_value_pct=pos.get("position_value_pct", 0),
                    should_close=should_close,
                    close_reason=reason,
                ))

        return to_close


class PortfolioRiskMonitor:
    """
    Monitor and manage portfolio-level risk.
    """

    def __init__(self, kalshi: KalshiClient, risk_manager: RiskManager):
        self.kalshi = kalshi
        self.risk_manager = risk_manager
        self.alerts: list[dict] = []

    async def check_portfolio_risk(self) -> RiskMetrics:
        """Check current portfolio risk status."""
        try:
            positions_data = await self.kalshi.get_positions()
            positions = positions_data.get("positions", [])

            balance_data = await self.kalshi.get_balance()
            portfolio_value = balance_data.get("balance", 0)

            self.risk_manager.portfolio_value = portfolio_value

            for pos in positions:
                ticker = pos.get("ticker")
                if not ticker:
                    continue

                try:
                    market = await self.kalshi.get_market(ticker)
                    current_price = market.get("market", {}).get("yes_ask", 50)
                    
                    self.risk_manager.update_position(
                        ticker=ticker,
                        contracts=pos.get("count", 0),
                        entry_price=pos.get("avg_price", 50),
                        current_price=current_price,
                        side=pos.get("side", "yes"),
                    )
                except Exception as e:
                    log.debug("position_update_failed", ticker=ticker, error=str(e))

        except Exception as e:
            log.error("portfolio_risk_check_failed", error=str(e))

        return self.risk_manager.calculate_risk_metrics()

    async def run_monitoring(self, check_interval: int = 60):
        """Run continuous risk monitoring."""
        while True:
            try:
                metrics = await self.check_portfolio_risk()

                if metrics.daily_return_pct < -self.risk_manager.limits.max_daily_loss_pct * 100:
                    alert = {
                        "time": datetime.utcnow().isoformat(),
                        "type": "daily_loss_limit",
                        "details": f"Loss: {metrics.daily_return_pct:.2f}%",
                    }
                    self.alerts.append(alert)
                    log.warning("risk_alert", alert_type=alert["type"],
                                details=alert["details"], time=alert["time"])

                if metrics.current_drawdown_pct > 10:
                    alert = {
                        "time": datetime.utcnow().isoformat(),
                        "type": "high_drawdown",
                        "details": f"Drawdown: {metrics.current_drawdown_pct:.2f}%",
                    }
                    self.alerts.append(alert)
                    log.warning("risk_alert", alert_type=alert["type"],
                                details=alert["details"], time=alert["time"])

                to_close = self.risk_manager.get_positions_to_close()
                if to_close:
                    log.warning("positions_to_close", count=len(to_close), positions=[p.ticker for p in to_close])

                log.debug(
                    "risk_metrics",
                    portfolio=metrics.portfolio_value,
                    daily_pnl=metrics.daily_pnl,
                    positions=metrics.position_count,
                    drawdown=metrics.current_drawdown_pct,
                )

            except Exception as e:
                log.error("monitoring_error", error=str(e))

            await asyncio.sleep(check_interval)

    def get_alerts(self, since: datetime | None = None) -> list[dict]:
        """Get risk alerts."""
        if since is None:
            return self.alerts
        return [a for a in self.alerts if datetime.fromisoformat(a["time"]) > since]

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self.alerts = []

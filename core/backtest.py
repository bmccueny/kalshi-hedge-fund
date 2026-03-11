"""
Backtesting Engine
──────────────────
Simulates trading strategies using historical data.
Evaluates performance with: Sharpe ratio, max drawdown, cumulative returns, win rate, etc.
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import numpy as np
import structlog
from core.kalshi_client import KalshiClient
from core.data_pipeline import DataCollector, DataCleaner, DataPreprocessor
from core.models import TrendDetector, AnomalyDetector

log = structlog.get_logger()


@dataclass
class Trade:
    ticker: str
    entry_time: int
    exit_time: int | None
    entry_price: float
    exit_price: float | None
    side: str
    contracts: int
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class BacktestResult:
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    profit_factor: float
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    daily_returns: list[float] = field(default_factory=list)


class BacktestEngine:
    """
    Backtesting engine for trading strategies.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.0,
        slippage: float = 0.0,
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.trades: list[Trade] = []
        self.equity_curve: list[float] = []
        self.current_capital = initial_capital

    def reset(self) -> None:
        """Reset engine state."""
        self.trades = []
        self.equity_curve = []
        self.current_capital = self.initial_capital

    def add_trade(self, trade: Trade) -> None:
        """Record a trade."""
        self.trades.append(trade)
        if trade.pnl != 0:
            self.current_capital += trade.pnl
        self.equity_curve.append(self.current_capital)

    def compute_metrics(self, strategy_name: str = "Strategy") -> BacktestResult:
        """Compute performance metrics from recorded trades."""
        if not self.equity_curve:
            return BacktestResult(
                strategy_name=strategy_name,
                start_date="",
                end_date="",
                initial_capital=self.initial_capital,
                final_capital=self.current_capital,
                total_return_pct=0.0,
                sharpe_ratio=0.0,
                max_drawdown_pct=0.0,
                win_rate=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
            )

        equity = np.array(self.equity_curve)
        
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital * 100
        
        cummax = np.maximum.accumulate(equity)
        drawdowns = (equity - cummax) / cummax * 100
        max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
        
        daily_returns = np.diff(equity) / equity[:-1] * 100 if len(equity) > 1 else np.array([0.0])
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0.0
        
        completed_trades = [t for t in self.trades if t.exit_price is not None]
        winning = [t for t in completed_trades if t.pnl > 0]
        losing = [t for t in completed_trades if t.pnl <= 0]
        
        win_rate = len(winning) / len(completed_trades) * 100 if completed_trades else 0.0
        avg_win = np.mean([t.pnl for t in winning]) if winning else 0.0
        avg_loss = abs(np.mean([t.pnl for t in losing])) if losing else 0.0
        
        total_wins = sum(t.pnl for t in winning)
        total_losses = abs(sum(t.pnl for t in losing))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        start_time = completed_trades[0].entry_time if completed_trades else 0
        end_time = completed_trades[-1].exit_time if completed_trades else 0

        def _safe_timestamp(ts: int) -> str:
            """Convert timestamp or array index to date string."""
            if ts and ts > 1000000000:  # Reasonable timestamp (after year 2001)
                return datetime.fromtimestamp(ts).isoformat()
            return ""

        return BacktestResult(
            strategy_name=strategy_name,
            start_date=_safe_timestamp(start_time),
            end_date=_safe_timestamp(end_time),
            initial_capital=self.initial_capital,
            final_capital=self.current_capital,
            total_return_pct=total_return,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_drawdown,
            win_rate=win_rate,
            total_trades=len(completed_trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            trades=completed_trades,
            equity_curve=self.equity_curve,
            daily_returns=daily_returns.tolist(),
        )


class MomentumBacktester:
    """
    Backtest momentum trading strategy.
    """

    def __init__(
        self,
        kalshi: KalshiClient,
        initial_capital: float = 10000.0,
        min_confidence: float = 0.6,
        position_size_pct: float = 0.1,
    ):
        self.kalshi = kalshi
        self.min_confidence = min_confidence
        self.position_size_pct = position_size_pct
        self.engine = BacktestEngine(initial_capital=initial_capital)
        self.slippage = self.engine.slippage
        self.commission = self.engine.commission
        self.collector = DataCollector(kalshi)
        self.cleaner = DataCleaner()
        self.preprocessor = DataPreprocessor()
        self.trend_detector = TrendDetector()

    async def run(
        self,
        tickers: list[str] | None = None,
        lookback: int = 50,
    ) -> BacktestResult:
        """Run momentum backtest on given tickers."""
        self.engine.reset()
        
        if tickers is None:
            markets = await self.kalshi.get_all_open_markets()
            tickers = [m.get("ticker") for m in markets if m.get("ticker") and m.get("open_interest", 0) > 10000]
            tickers = tickers[:50]

        log.info("momentum_backtest_start", tickers=len(tickers))

        for ticker in tickers:
            try:
                await self._backtest_ticker(ticker, lookback)
            except Exception as e:
                log.debug("backtest_ticker_error", ticker=ticker, error=str(e))

        return self.engine.compute_metrics("Momentum")

    async def _backtest_ticker(self, ticker: str, lookback: int) -> None:
        """Backtest a single ticker with momentum strategy."""
        data = await self.collector.collect_market_data(ticker, lookback=lookback)
        history = data.get("history", [])
        
        if len(history) < 20:
            return

        cleaned = self.cleaner.clean_price_history(history)
        arrays = self.preprocessor.process_price_history(cleaned)
        
        yes_prices = arrays.get("yes_prices")
        if yes_prices is None or len(yes_prices) < 20:
            return

        prices = yes_prices.tolist()

        for i in range(20, len(prices)):
            window_prices = prices[:i]
            signal = self.trend_detector.analyze(window_prices)

            if signal.confidence >= self.min_confidence and signal.strength in ("strong", "moderate"):
                current_price = prices[i]
                
                has_open = any(
                    t.ticker == ticker and t.exit_price is None 
                    for t in self.engine.trades
                )
                
                if has_open:
                    continue

                if signal.direction == "bullish":
                    side = "yes"
                elif signal.direction == "bearish":
                    side = "no"
                else:
                    continue

                position_size = (self.engine.current_capital * self.position_size_pct) / (current_price / 100)
                contracts = int(position_size)

                if contracts < 1:
                    continue

                entry_price = current_price * (1 + self.slippage / 100)
                
                trade = Trade(
                    ticker=ticker,
                    entry_time=history[i].get("ts", 0),
                    exit_time=None,
                    entry_price=entry_price,
                    exit_price=None,
                    side=side,
                    contracts=contracts,
                )
                self.engine.add_trade(trade)

                exit_idx = min(i + 10, len(prices) - 1)
                exit_price = prices[exit_idx] * (1 - self.slippage / 100)

                if side == "yes":
                    pnl = (exit_price - entry_price) * contracts
                else:
                    pnl = (entry_price - exit_price) * contracts
                
                pnl -= self.commission

                trade.exit_price = exit_price
                trade.exit_time = history[exit_idx].get("ts", 0)
                trade.pnl = pnl
                trade.pnl_pct = (pnl / (entry_price * contracts)) * 100 if entry_price * contracts > 0 else 0


class MeanReversionBacktester:
    """
    Backtest mean reversion strategy.
    """

    def __init__(
        self,
        kalshi: KalshiClient,
        initial_capital: float = 10000.0,
        zscore_threshold: float = 2.0,
        position_size_pct: float = 0.1,
    ):
        self.kalshi = kalshi
        self.zscore_threshold = zscore_threshold
        self.position_size_pct = position_size_pct
        self.engine = BacktestEngine(initial_capital=initial_capital)
        self.slippage = self.engine.slippage
        self.commission = self.engine.commission
        self.collector = DataCollector(kalshi)
        self.cleaner = DataCleaner()
        self.preprocessor = DataPreprocessor()

    async def run(
        self,
        tickers: list[str] | None = None,
        lookback: int = 50,
    ) -> BacktestResult:
        """Run mean reversion backtest."""
        self.engine.reset()
        
        if tickers is None:
            markets = await self.kalshi.get_all_open_markets()
            tickers = [m.get("ticker") for m in markets if m.get("ticker") and m.get("open_interest", 0) > 10000]
            tickers = tickers[:50]

        log.info("mean_reversion_backtest_start", tickers=len(tickers))

        for ticker in tickers:
            try:
                await self._backtest_ticker(ticker, lookback)
            except Exception as e:
                log.debug("backtest_ticker_error", ticker=ticker, error=str(e))

        return self.engine.compute_metrics("MeanReversion")

    def _compute_zscore(self, prices: list[float], current: float) -> float:
        """Compute z-score for current price."""
        if len(prices) < 10:
            return 0.0
        mean = np.mean(prices)
        std = np.std(prices)
        if std == 0:
            return 0.0
        return (current - mean) / std

    async def _backtest_ticker(self, ticker: str, lookback: int) -> None:
        """Backtest a single ticker with mean reversion strategy."""
        data = await self.collector.collect_market_data(ticker, lookback=lookback)
        history = data.get("history", [])
        
        if len(history) < 20:
            return

        cleaned = self.cleaner.clean_price_history(history)
        arrays = self.preprocessor.process_price_history(cleaned)
        
        yes_prices = arrays.get("yes_prices")
        if yes_prices is None or len(yes_prices) < 20:
            return

        prices = yes_prices.tolist()

        for i in range(20, len(prices)):
            window = prices[max(0, i-20):i]
            current = prices[i]
            
            zscore = self._compute_zscore(window, current)

            has_open = any(
                t.ticker == ticker and t.exit_price is None 
                for t in self.engine.trades
            )

            if has_open:
                # Check if zscore returned to mean -> close position
                if abs(zscore) < 0.5:
                    open_trades = [t for t in self.engine.trades if t.ticker == ticker and t.exit_price is None]
                    for trade in open_trades:
                        exit_price = current * (1 - self.slippage / 100)
                        pnl = (exit_price - trade.entry_price) * trade.contracts if trade.side == "yes" else (trade.entry_price - exit_price) * trade.contracts
                        trade.exit_price = exit_price
                        trade.exit_time = history[i].get("ts", 0)
                        trade.pnl = pnl - self.commission
                        trade.pnl_pct = (pnl / (trade.entry_price * trade.contracts)) * 100 if trade.entry_price * trade.contracts > 0 else 0
                continue

            if abs(zscore) < self.zscore_threshold:
                continue

            position_size = (self.engine.current_capital * self.position_size_pct) / (current / 100)
            contracts = int(position_size)

            if contracts < 1:
                continue

            side = "no" if zscore > 0 else "yes"
            entry_price = current * (1 + self.slippage / 100)

            trade = Trade(
                ticker=ticker,
                entry_time=history[i].get("ts", 0),
                exit_time=None,
                entry_price=entry_price,
                exit_price=None,
                side=side,
                contracts=contracts,
            )
            self.engine.add_trade(trade)


class EventDrivenBacktester:
    """
    Backtest event-driven strategy.
    """

    def __init__(
        self,
        kalshi: KalshiClient,
        initial_capital: float = 10000.0,
        hours_before_event: int = 24,
        position_size_pct: float = 0.15,
    ):
        self.kalshi = kalshi
        self.hours_before_event = hours_before_event
        self.position_size_pct = position_size_pct
        self.engine = BacktestEngine(initial_capital=initial_capital)
        self.slippage = self.engine.slippage
        self.commission = self.engine.commission
        self.collector = DataCollector(kalshi)
        self.cleaner = DataCleaner()
        self.preprocessor = DataPreprocessor()

    async def run(
        self,
        tickers: list[str] | None = None,
    ) -> BacktestResult:
        """Run event-driven backtest."""
        self.engine.reset()
        
        if tickers is None:
            markets = await self.kalshi.get_all_open_markets()
            tickers = [m.get("ticker") for m in markets if m.get("ticker")]
            tickers = tickers[:30]

        log.info("event_backtest_start", tickers=len(tickers))

        for ticker in tickers:
            try:
                await self._backtest_ticker(ticker)
            except Exception as e:
                log.debug("backtest_ticker_error", ticker=ticker, error=str(e))

        return self.engine.compute_metrics("EventDriven")

    async def _backtest_ticker(self, ticker: str) -> None:
        """Backtest a single ticker with event-driven strategy."""
        data = await self.collector.collect_market_data(ticker, lookback=50)
        market = data.get("market", {})
        history = data.get("history", [])
        
        if not history or len(history) < 10:
            return

        cleaned = self.cleaner.clean_price_history(history)
        arrays = self.preprocessor.process_price_history(cleaned)
        
        yes_prices = arrays.get("yes_prices")
        if yes_prices is None or len(yes_prices) < 10:
            return

        prices = yes_prices.tolist()

        event_keywords = ["fed", "election", "CPI", "GDP", "earnings", "jobs", "unemployment"]
        is_event = any(k.lower() in market.get("title", "").lower() for k in event_keywords)

        if not is_event:
            return

        mid_idx = len(prices) // 2
        entry_price = prices[mid_idx]
        exit_price = prices[-1]

        position_size = (self.engine.current_capital * self.position_size_pct) / (entry_price / 100)
        contracts = int(position_size)

        if contracts < 1:
            return

        trade = Trade(
            ticker=ticker,
            entry_time=history[mid_idx].get("ts", 0),
            exit_time=history[-1].get("ts", 0),
            entry_price=entry_price,
            exit_price=exit_price,
            side="yes",
            contracts=contracts,
        )

        pnl = (exit_price - entry_price) * contracts
        trade.pnl = pnl - self.commission
        trade.pnl_pct = (pnl / (entry_price * contracts)) * 100 if entry_price * contracts > 0 else 0
        
        self.engine.add_trade(trade)


async def run_full_backtest(
    kalshi: KalshiClient,
    initial_capital: float = 10000.0,
) -> dict[str, BacktestResult]:
    """Run all strategies and compare."""
    results = {}

    momentum = MomentumBacktester(kalshi, initial_capital=initial_capital)
    results["momentum"] = await momentum.run()

    mean_reversion = MeanReversionBacktester(kalshi, initial_capital=initial_capital)
    results["mean_reversion"] = await mean_reversion.run()

    event_driven = EventDrivenBacktester(kalshi, initial_capital=initial_capital)
    results["event_driven"] = await event_driven.run()

    return results

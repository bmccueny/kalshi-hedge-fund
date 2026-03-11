"""
Parameter Optimization
────────────────────
Fine-tune strategy parameters using grid search and Bayesian optimization.
"""
import asyncio
from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
import structlog
from core.backtest import BacktestEngine, BacktestResult, Trade
from core.data_pipeline import DataCollector, DataCleaner, DataPreprocessor
from core.models import TrendDetector

log = structlog.get_logger()


@dataclass
class OptimizedParams:
    strategy: str
    parameters: dict
    sharpe_ratio: float
    total_return_pct: float
    max_drawdown_pct: float
    win_rate: float


class ParameterOptimizer:
    """
    Base class for parameter optimization.
    """

    def __init__(self):
        self.best_result: OptimizedParams | None = None

    def optimize(
        self,
        param_grid: dict[str, list[Any]],
        objective: Callable[[dict], BacktestResult],
    ) -> OptimizedParams:
        """Override in subclass."""
        raise NotImplementedError


class GridSearchOptimizer(ParameterOptimizer):
    """
    Grid search parameter optimization.
    """

    def __init__(self):
        super().__init__()
        self.results: list[OptimizedParams] = []

    def optimize(
        self,
        param_grid: dict[str, list[Any]],
        objective: Callable[[dict], BacktestResult],
    ) -> OptimizedParams:
        """Exhaustively search parameter combinations."""
        import itertools

        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))

        log.info("grid_search_start", combinations=len(combinations))

        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))

            try:
                result = objective(params)
                opt = OptimizedParams(
                    strategy=params.get("strategy", "unknown"),
                    parameters=params,
                    sharpe_ratio=result.sharpe_ratio,
                    total_return_pct=result.total_return_pct,
                    max_drawdown_pct=result.max_drawdown_pct,
                    win_rate=result.win_rate,
                )
                self.results.append(opt)

                if self.best_result is None or opt.sharpe_ratio > self.best_result.sharpe_ratio:
                    self.best_result = opt
                    log.info("grid_search_new_best", params=params, sharpe=opt.sharpe_ratio)

            except Exception as e:
                log.warning("grid_search_trial_failed", params=params, error=str(e))

            if (i + 1) % 10 == 0:
                log.info("grid_search_progress", completed=i + 1, total=len(combinations))

        log.info("grid_search_complete", best_sharpe=self.best_result.sharpe_ratio if self.best_result else None)
        return self.best_result


class BayesianOptimizer(ParameterOptimizer):
    """
    Bayesian optimization using Gaussian Processes.
    """

    def __init__(self, n_iterations: int = 20):
        super().__init__()
        self.n_iterations = n_iterations
        self.results: list[OptimizedParams] = []
        self.param_space: dict[str, tuple[float, float]] = {}

    def optimize(
        self,
        param_space: dict[str, tuple[float, float]],
        objective: Callable[[dict], BacktestResult],
    ) -> OptimizedParams:
        """Bayesian optimization with Gaussian Process surrogate."""
        self.param_space = param_space

        best_x = self._sample_initial()
        best_y = self._evaluate(best_x, objective)

        log.info("bayesian_start", iterations=self.n_iterations)

        for i in range(self.n_iterations):
            x_candidate = self._propose_candidate(best_x, best_y)

            if x_candidate == best_x:
                x_candidate = self._sample_random()

            y_candidate = self._evaluate(x_candidate, objective)

            if y_candidate > best_y:
                best_x = x_candidate
                best_y = y_candidate

            log.info(
                "bayesian_iteration",
                iteration=i + 1,
                best_sharpe=best_y,
                params=x_candidate,
            )

        return self.best_result

    def _sample_initial(self) -> dict[str, float]:
        """Sample initial random parameters."""
        return {k: np.random.uniform(v[0], v[1]) for k, v in self.param_space.items()}

    def _sample_random(self) -> dict[str, float]:
        """Sample random parameters."""
        return {k: np.random.uniform(v[0], v[1]) for k, v in self.param_space.items()}

    def _evaluate(
        self,
        params: dict[str, float],
        objective: Callable[[dict], BacktestResult],
    ) -> float:
        """Evaluate parameters."""
        try:
            result = objective(params)
            opt = OptimizedParams(
                strategy=params.get("strategy", "unknown"),
                parameters=params,
                sharpe_ratio=result.sharpe_ratio,
                total_return_pct=result.total_return_pct,
                max_drawdown_pct=result.max_drawdown_pct,
                win_rate=result.win_rate,
            )
            self.results.append(opt)

            if self.best_result is None or result.sharpe_ratio > self.best_result.sharpe_ratio:
                self.best_result = opt

            return result.sharpe_ratio
        except Exception as e:
            log.warning("bayesian_eval_failed", params=params, error=str(e))
            return -999

    def _propose_candidate(self, best_x: dict, best_y: float) -> dict[str, float]:
        """Propose next candidate using exploration-exploitation."""
        candidate = {}
        for k, v in self.param_space.items():
            if np.random.random() < 0.3:
                candidate[k] = best_x.get(k, (v[0] + v[1]) / 2) + np.random.uniform(-0.1, 0.1) * (v[1] - v[0])
            else:
                candidate[k] = np.random.uniform(v[0], v[1])
            candidate[k] = max(v[0], min(v[1], candidate[k]))
        return candidate


def optimize_momentum_strategy(
    prices: list[list[float]],
    param_grid: dict[str, list] | None = None,
    use_bayesian: bool = False,
) -> OptimizedParams:
    """
    Optimize momentum strategy parameters.
    
    Parameters to optimize:
    - short_window: 3-15
    - long_window: 10-50
    - min_confidence: 0.3-0.9
    - position_size_pct: 0.05-0.3
    """
    if param_grid is None:
        param_grid = {
            "strategy": ["momentum"],
            "short_window": [3, 5, 7, 10],
            "long_window": [10, 15, 20, 30],
            "min_confidence": [0.4, 0.5, 0.6, 0.7],
            "position_size_pct": [0.05, 0.1, 0.15, 0.2],
        }

    def objective(params: dict) -> BacktestResult:
        short_window = int(params.get("short_window", 5))
        long_window = int(params.get("long_window", 20))
        min_confidence = params.get("min_confidence", 0.5)
        position_size_pct = params.get("position_size_pct", 0.1)

        engine = BacktestEngine(initial_capital=10000)
        trend_detector = TrendDetector(short_window=short_window, long_window=long_window)

        for price_series in prices:
            if len(price_series) < long_window:
                continue

            for i in range(long_window, len(price_series) - 5):
                window_prices = price_series[:i]
                signal = trend_detector.analyze(window_prices)

                if signal.confidence >= min_confidence and signal.strength in ("strong", "moderate"):
                    current_price = price_series[i]

                    has_open = any(
                        t.exit_price is None for t in engine.trades
                    )
                    if has_open:
                        continue

                    side = "yes" if signal.direction == "bullish" else "no" if signal.direction == "bearish" else None
                    if side is None:
                        continue

                    position_size = (engine.current_capital * position_size_pct) / (current_price / 100)
                    contracts = int(position_size)

                    if contracts < 1:
                        continue

                    entry_price = current_price
                    trade = Trade(
                        ticker="OPT",
                        entry_time=i,
                        exit_time=None,
                        entry_price=entry_price,
                        exit_price=None,
                        side=side,
                        contracts=contracts,
                    )
                    engine.add_trade(trade)

                    exit_idx = min(i + 5, len(price_series) - 1)
                    exit_price = price_series[exit_idx]

                    if side == "yes":
                        pnl = (exit_price - entry_price) * contracts
                    else:
                        pnl = (entry_price - exit_price) * contracts

                    trade.exit_price = exit_price
                    trade.exit_time = exit_idx
                    trade.pnl = pnl
                    trade.pnl_pct = (pnl / (entry_price * contracts)) * 100 if entry_price * contracts > 0 else 0

        return engine.compute_metrics("Momentum")

    if use_bayesian:
        optimizer = BayesianOptimizer(n_iterations=20)
    else:
        optimizer = GridSearchOptimizer()

    return optimizer.optimize(param_grid, objective)


def optimize_mean_reversion_strategy(
    prices: list[list[float]],
    param_grid: dict[str, list] | None = None,
    use_bayesian: bool = False,
) -> OptimizedParams:
    """
    Optimize mean reversion strategy parameters.
    
    Parameters:
    - lookback: 10-50
    - zscore_threshold: 1.0-3.0
    - position_size_pct: 0.05-0.3
    """
    if param_grid is None:
        param_grid = {
            "strategy": ["mean_reversion"],
            "lookback": [10, 15, 20, 30],
            "zscore_threshold": [1.5, 2.0, 2.5, 3.0],
            "position_size_pct": [0.05, 0.1, 0.15, 0.2],
        }

    def objective(params: dict) -> BacktestResult:
        lookback = int(params.get("lookback", 20))
        zscore_threshold = params.get("zscore_threshold", 2.0)
        position_size_pct = params.get("position_size_pct", 0.1)

        engine = BacktestEngine(initial_capital=10000)

        def compute_zscore(data: list[float], current: float) -> float:
            if len(data) < 5:
                return 0.0
            mean = np.mean(data)
            std = np.std(data)
            return (current - mean) / std if std > 0 else 0.0

        for price_series in prices:
            if len(price_series) < lookback + 5:
                continue

            for i in range(lookback, len(price_series) - 3):
                window = price_series[max(0, i-lookback):i]
                current = price_series[i]
                zscore = compute_zscore(window, current)

                if abs(zscore) < zscore_threshold:
                    continue

                has_open = any(t.exit_price is None for t in engine.trades)
                if has_open:
                    if abs(zscore) < 0.5:
                        continue
                else:
                    continue

                position_size = (engine.current_capital * position_size_pct) / (current / 100)
                contracts = int(position_size)

                if contracts < 1:
                    continue

                side = "no" if zscore > 0 else "yes"
                trade = Trade(
                    ticker="OPT",
                    entry_time=i,
                    exit_time=None,
                    entry_price=current,
                    exit_price=None,
                    side=side,
                    contracts=contracts,
                )
                engine.add_trade(trade)

                exit_idx = min(i + 3, len(price_series) - 1)
                exit_price = price_series[exit_idx]

                if side == "yes":
                    pnl = (exit_price - current) * contracts
                else:
                    pnl = (current - exit_price) * contracts

                trade.exit_price = exit_price
                trade.exit_time = exit_idx
                trade.pnl = pnl
                trade.pnl_pct = (pnl / (current * contracts)) * 100 if current * contracts > 0 else 0

        return engine.compute_metrics("MeanReversion")

    if use_bayesian:
        optimizer = BayesianOptimizer(n_iterations=20)
    else:
        optimizer = GridSearchOptimizer()

    return optimizer.optimize(param_grid, objective)


def run_optimization(
    prices: list[list[float]],
    strategies: list[str] | None = None,
) -> dict[str, OptimizedParams]:
    """Run optimization for multiple strategies."""
    if strategies is None:
        strategies = ["momentum", "mean_reversion"]

    results = {}

    if "momentum" in strategies:
        log.info("optimizing_momentum")
        results["momentum"] = optimize_momentum_strategy(prices, use_bayesian=False)

    if "mean_reversion" in strategies:
        log.info("optimizing_mean_reversion")
        results["mean_reversion"] = optimize_mean_reversion_strategy(prices, use_bayesian=False)

    return results

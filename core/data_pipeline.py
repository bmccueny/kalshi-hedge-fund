"""
Data Collection and Preprocessing
────────────────────────────────
Collects and preprocesses market data for model input.
Handles:
- Historical price/volume data collection from Kalshi API
- Data cleaning (missing values, outliers, inconsistencies)
- Normalization and standardization
- Feature engineering
- Time series preparation
"""
import asyncio
from collections import deque
from datetime import datetime
from typing import Any
import numpy as np
import structlog
from core.kalshi_client import KalshiClient

log = structlog.get_logger()


class DataCollector:
    """
    Collects historical market data from Kalshi API.
    """

    def __init__(self, kalshi: KalshiClient):
        self.kalshi = kalshi
        self._cache: dict[str, dict] = {}

    async def collect_market_data(
        self,
        ticker: str,
        lookback: int = 100,
    ) -> dict[str, Any]:
        """
        Collect comprehensive historical market data for a ticker.
        Fetches max available history for meaningful backtesting.
        """
        if ticker in self._cache:
            cached = self._cache[ticker]
            if datetime.utcnow().timestamp() - cached.get("_cached_at", 0) < 60:
                return cached

        try:
            market = await self.kalshi.get_market(ticker)
            history = await self.kalshi.get_market_history(ticker, limit=lookback)
            orderbook = await self.kalshi.get_market_orderbook(ticker, depth=10)

            data = {
                "ticker": ticker,
                "market": market.get("market", {}),
                "history": history.get("history", []),
                "orderbook": orderbook,
                "_cached_at": datetime.utcnow().timestamp(),
                "_data_points": len(history.get("history", [])),
            }

            self._cache[ticker] = data
            log.debug("data_collected", ticker=ticker, points=data["_data_points"])
            return data
        except Exception as e:
            log.error("data_collection_failed", ticker=ticker, error=str(e))
            return {}

    async def collect_historical_data(
        self,
        ticker: str,
        min_data_points: int = 50,
    ) -> list[dict]:
        """
        Ensure sufficient historical data for meaningful analysis.
        Attempts to get at least min_data_points of history.
        """
        lookback = min_data_points
        data = await self.collect_market_data(ticker, lookback=lookback)
        
        history = data.get("history", [])
        attempts = 0
        max_attempts = 3
        
        while len(history) < min_data_points and attempts < max_attempts:
            lookback *= 2
            data = await self.collect_market_data(ticker, lookback=lookback)
            history = data.get("history", [])
            attempts += 1
        
        return history

    async def collect_batch(
        self,
        tickers: list[str],
        lookback: int = 50,
    ) -> dict[str, dict]:
        """Collect data for multiple tickers concurrently."""
        sem = asyncio.Semaphore(5)

        async def fetch(ticker: str) -> tuple[str, dict]:
            async with sem:
                return ticker, await self.collect_market_data(ticker, lookback)

        results = await asyncio.gather(*[fetch(t) for t in tickers], return_exceptions=True)
        collected = {}
        for r in results:
            if isinstance(r, Exception):
                continue
            if isinstance(r, tuple) and len(r) == 2:
                t, d = r
                if t and d:
                    collected[t] = d
        log.info("batch_collected", tickers=len(collected), failed=len(tickers) - len(collected))
        return collected

    async def collect_all_markets_data(
        self,
        max_markets: int = 100,
        min_data_points: int = 20,
    ) -> dict[str, list[dict]]:
        """Collect historical data for all open markets."""
        markets = await self.kalshi.get_all_open_markets()
        markets = [m for m in markets if m.get("open_interest", 0) > 5000][:max_markets]
        
        tickers = [m.get("ticker") for m in markets if m.get("ticker")]
        tickers = [t for t in tickers if t]
        batch_data = await self.collect_batch(tickers, lookback=min_data_points)
        
        return {
            ticker: data.get("history", [])
            for ticker, data in batch_data.items()
            if data.get("history")
        }


class DataCleaner:
    """
    Cleans raw market data: handles missing values, outliers, inconsistencies.
    """

    def __init__(
        self,
        max_gap_fill: int = 3,
        outlier_std_multiplier: float = 3.0,
    ):
        self.max_gap_fill = max_gap_fill
        self.outlier_std_multiplier = outlier_std_multiplier

    def clean_price_history(self, history: list[dict]) -> list[dict]:
        """
        Clean price history data:
        - Removes duplicate timestamps
        - Fills missing values via interpolation
        - Removes outliers
        - Ensures chronological order
        """
        if not history:
            return []

        cleaned = []
        seen_ts = set()

        for h in history:
            ts = h.get("ts")
            if ts is None or ts in seen_ts:
                continue
            seen_ts.add(ts)
            cleaned.append(h)

        cleaned.sort(key=lambda x: x.get("ts", 0))

        cleaned = self._fill_missing_values(cleaned)
        cleaned = self._remove_outliers(cleaned)
        cleaned = self._fix_inconsistencies(cleaned)

        log.debug("data_cleaned", 
                  input=len(history), 
                  output=len(cleaned),
                  removed=len(history) - len(cleaned))
        return cleaned

    def _fill_missing_values(self, data: list[dict]) -> list[dict]:
        """Interpolate missing price/volume values using linear interpolation."""
        if len(data) < 2:
            return data

        yes_prices = [d.get("yes_price") or 50.0 for d in data]
        no_prices = [d.get("no_price") or 50.0 for d in data]
        volumes = [d.get("volume") or 0 for d in data]

        # Interpolate missing values using pandas for simplicity
        import pandas as pd
        
        # Create a DataFrame for interpolation
        df = pd.DataFrame({
            "yes_price": yes_prices,
            "no_price": no_prices,
            "volume": volumes,
        })
        
        # Interpolate missing values (linear interpolation)
        df = df.interpolate(method="linear", limit_direction="both")
        
        # Fill any remaining NaN at edges with forward/backward fill
        df = df.ffill().bfill()
        
        # Write back to data
        for i, d in enumerate(data):
            d["yes_price"] = df.iloc[i]["yes_price"]
            d["no_price"] = df.iloc[i]["no_price"]
            d["volume"] = int(df.iloc[i]["volume"])
        
        return data

    def _remove_outliers(self, data: list[dict]) -> list[dict]:
        """Remove data points with extreme price values."""
        if len(data) < 10:
            return data

        prices = [float(d.get("yes_price") or 50) for d in data]
        mean = np.mean(prices)
        std = np.std(prices)

        threshold = self.outlier_std_multiplier * std

        filtered = []
        for d in data:
            price = d.get("yes_price", 50)
            if abs(price - mean) <= threshold:
                filtered.append(d)
            else:
                log.debug("outlier_removed", 
                         price=price, 
                         mean=mean, 
                         deviation=abs(price - mean) / std if std > 0 else 0)

        return filtered

    def _fix_inconsistencies(self, data: list[dict]) -> list[dict]:
        """Fix inconsistent data points."""
        if len(data) < 2:
            return data

        for i, d in enumerate(data):
            yes_price = float(d.get("yes_price") or 50)
            no_price = float(d.get("no_price") or 50)

            if yes_price < 0:
                yes_price = 0
            if yes_price > 100:
                yes_price = 100
            if no_price < 0:
                no_price = 0
            if no_price > 100:
                no_price = 100

            if yes_price + no_price < 95 or yes_price + no_price > 105:
                no_price = 100 - yes_price

            if yes_price == 0 and no_price == 0:
                if i > 0:
                    yes_price = data[i-1].get("yes_price", 50)
                    no_price = data[i-1].get("no_price", 50)
                else:
                    yes_price = 50
                    no_price = 50

            d["yes_price"] = yes_price
            d["no_price"] = no_price

        return data


class DataPreprocessor:
    """
    Preprocesses raw market data into features for models.
    """

    def __init__(self):
        self.price_window = deque(maxlen=100)
        self.volume_window = deque(maxlen=100)

    def process_price_history(
        self,
        history: list[dict],
    ) -> dict[str, np.ndarray]:
        """Convert raw price history to feature arrays."""
        if not history:
            return {}

        yes_prices = []
        no_prices = []
        volumes = []
        timestamps = []

        for h in history:
            yes_prices.append(h.get("yes_price", 50))
            no_prices.append(h.get("no_price", 50))
            volumes.append(h.get("volume", 0))
            timestamps.append(h.get("ts", 0))

        return {
            "yes_prices": np.array(yes_prices, dtype=np.float64),
            "no_prices": np.array(no_prices, dtype=np.float64),
            "volumes": np.array(volumes, dtype=np.float64),
            "timestamps": np.array(timestamps, dtype=np.int64),
        }

    def compute_features(
        self,
        prices: np.ndarray,
        volumes: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute technical features from price series."""
        if len(prices) < 5:
            return {}

        features = {}

        features["mean"] = float(np.mean(prices))
        features["std"] = float(np.std(prices))
        features["min"] = float(np.min(prices))
        features["max"] = float(np.max(prices))
        features["range"] = features["max"] - features["min"]

        if len(prices) >= 5:
            features["sma_5"] = float(np.mean(prices[-5:]))
        if len(prices) >= 10:
            features["sma_10"] = float(np.mean(prices[-10:]))
        if len(prices) >= 20:
            features["sma_20"] = float(np.mean(prices[-20:]))

        if len(prices) >= 2:
            returns = np.diff(prices) / (prices[:-1] + 1e-10)
            features["momentum"] = float(np.mean(returns))
            features["volatility"] = float(np.std(returns))

        if len(prices) >= 10:
            recent = prices[-5:]
            older = prices[-10:-5]
            features["trend_strength"] = float(np.mean(recent) - np.mean(older))

        if volumes is not None and len(volumes) > 0:
            features["avg_volume"] = float(np.mean(volumes))
            features["volume_std"] = float(np.std(volumes))
            if len(volumes) > 0:
                features["volume_ratio"] = float(volumes[-1] / (features["avg_volume"] + 1))

        return features

    def normalize_features(
        self,
        features: dict[str, float],
        reference: dict[str, tuple[float, float]],
    ) -> dict[str, float]:
        """Normalize features using reference min/max values (0-1 scaling)."""
        normalized = {}
        for key, value in features.items():
            if key in reference:
                min_val, max_val = reference[key]
                if max_val > min_val:
                    normalized[key] = (value - min_val) / (max_val - min_val)
                else:
                    normalized[key] = 0.5
            else:
                normalized[key] = value
        return normalized

    def standardize_features(
        self,
        features: dict[str, float],
        reference: dict[str, tuple[float, float]],
    ) -> dict[str, float]:
        """Standardize features using z-score (mean=0, std=1)."""
        standardized = {}
        for key, value in features.items():
            if key in reference:
                mean, std = reference[key]
                if std > 0:
                    standardized[key] = (value - mean) / std
                else:
                    standardized[key] = 0.0
            else:
                standardized[key] = value
        return standardized

    def prepare_time_series(
        self,
        prices: list[float],
        window: int = 20,
    ) -> np.ndarray:
        """Prepare time series data for model input (z-score normalized)."""
        if len(prices) < window:
            window = len(prices)
        if window < 2:
            return np.array([])

        arr = np.array(prices[-window:], dtype=np.float64)
        mean = np.mean(arr)
        std = np.std(arr)
        if std > 0:
            arr = (arr - mean) / std
        return arr

    def detect_regime(
        self,
        prices: np.ndarray,
    ) -> str:
        """Detect market regime (trending, ranging, volatile)."""
        if len(prices) < 10:
            return "unknown"

        returns = np.diff(prices) / (prices[:-1] + 1e-10)
        volatility = np.std(returns)
        trend = np.mean(returns)

        if volatility > 0.1:
            return "volatile"
        if abs(trend) > 0.02:
            return "trending_up" if trend > 0 else "trending_down"
        return "ranging"

    def compute_normalization_reference(
        self,
        feature_sets: list[dict[str, float]],
    ) -> dict[str, tuple[float, float]]:
        """Compute min/max reference for normalization from multiple feature sets."""
        if not feature_sets:
            return {}

        all_keys = set()
        for fs in feature_sets:
            all_keys.update(fs.keys())

        reference = {}
        for key in all_keys:
            values = [fs.get(key, 0) for fs in feature_sets if key in fs]
            if values:
                reference[key] = (min(values), max(values))

        return reference

    def compute_standardization_reference(
        self,
        feature_sets: list[dict[str, float]],
    ) -> dict[str, tuple[float, float]]:
        """Compute mean/std reference for standardization from multiple feature sets."""
        if not feature_sets:
            return {}

        all_keys = set()
        for fs in feature_sets:
            all_keys.update(fs.keys())

        reference = {}
        for key in all_keys:
            values = [fs.get(key, 0) for fs in feature_sets if key in fs]
            if values:
                reference[key] = (np.mean(values), np.std(values))

        return reference


class FeatureStore:
    """
    Stores computed features for quick lookup.
    """

    def __init__(self):
        self._features: dict[str, dict[str, float]] = {}
        self._last_update: dict[str, float] = {}

    def set(self, ticker: str, features: dict[str, float]) -> None:
        """Store features for a ticker."""
        self._features[ticker] = features
        self._last_update[ticker] = datetime.utcnow().timestamp()

    def get(self, ticker: str, max_age_seconds: int = 300) -> dict[str, float] | None:
        """Retrieve features if fresh enough."""
        if ticker not in self._features:
            return None

        age = datetime.utcnow().timestamp() - self._last_update.get(ticker, 0)
        if age > max_age_seconds:
            return None

        return self._features[ticker]

    def get_all(self) -> dict[str, dict[str, float]]:
        """Get all stored features."""
        return self._features.copy()

    def get_feature_list(self) -> list[dict[str, float]]:
        """Get all features as a list for computing references."""
        return list(self._features.values())

    def clear(self) -> None:
        """Clear all stored features."""
        self._features.clear()
        self._last_update.clear()

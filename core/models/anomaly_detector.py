"""
Anomaly Detection Model
-----------------------
Identifies unusual price movements, spread anomalies, and liquidity issues
using statistical methods and deviation from historical norms.
"""
from dataclasses import dataclass
from collections import deque
import structlog

log = structlog.get_logger()


@dataclass
class AnomalyResult:
    is_anomaly: bool
    anomaly_type: str | None
    severity: float
    deviation_score: float
    details: dict


class AnomalyDetector:
    """
    Detects anomalies in market data using:
    - Z-score based price deviation
    - Spread anomaly detection
    - Volume spike detection
    - Mean reversion signals
    """

    def __init__(
        self,
        lookback_window: int = 50,
        zscore_threshold: float = 2.5,
        spread_zscore_threshold: float = 2.0,
    ):
        self.lookback_window = lookback_window
        self.zscore_threshold = zscore_threshold
        self.spread_zscore_threshold = spread_zscore_threshold
        self.price_history: deque[float] = deque(maxlen=lookback_window)
        self.spread_history: deque[float] = deque(maxlen=lookback_window)
        self.volume_history: deque[float] = deque(maxlen=lookback_window)

    def add_price(self, price: float) -> None:
        self.price_history.append(price)

    def add_spread(self, spread: float) -> None:
        self.spread_history.append(spread)

    def add_volume(self, volume: float) -> None:
        self.volume_history.append(volume)

    def analyze(
        self,
        current_price: float | None = None,
        current_spread: float | None = None,
        current_volume: float | None = None,
    ) -> AnomalyResult:
        """
        Analyze current market state for anomalies.
        Call add_price/add_spread/add_volume first to feed historical data.
        """
        if current_price is not None:
            self.add_price(current_price)
        if current_spread is not None:
            self.add_spread(current_spread)
        if current_volume is not None:
            self.add_volume(current_volume)

        if len(self.price_history) < 10:
            return AnomalyResult(
                is_anomaly=False,
                anomaly_type=None,
                severity=0.0,
                deviation_score=0.0,
                details={"reason": "insufficient_data"},
            )

        price_zscore = self._zscore(current_price or self.price_history[-1])
        spread_zscore = self._zscore_spread(current_spread or (self.spread_history[-1] if self.spread_history else 0))
        volume_zscore = self._zscore_volume(current_volume or (self.volume_history[-1] if self.volume_history else 0))

        anomalies = []
        max_severity = 0.0
        max_deviation = 0.0

        if abs(price_zscore) > self.zscore_threshold:
            anomalies.append("price_spike")
            max_severity = max(max_severity, min(1.0, abs(price_zscore) / 5))
            max_deviation = max(max_deviation, abs(price_zscore))

        if abs(spread_zscore) > self.spread_zscore_threshold:
            anomalies.append("spread_anomaly")
            max_severity = max(max_severity, min(1.0, abs(spread_zscore) / 4))
            max_deviation = max(max_deviation, abs(spread_zscore))

        if volume_zscore > self.zscore_threshold:
            anomalies.append("volume_spike")
            max_severity = max(max_severity, min(1.0, volume_zscore / 5))
            max_deviation = max(max_deviation, volume_zscore)

        if len(self.price_history) >= 20:
            mean_reversion_score = self._mean_reversion_potential()
            if mean_reversion_score > 0.7:
                anomalies.append("mean_reversion_setup")
                max_severity = max(max_severity, mean_reversion_score * 0.5)

        is_anomaly = len(anomalies) > 0

        details = {
            "price_zscore": price_zscore,
            "spread_zscore": spread_zscore,
            "volume_zscore": volume_zscore,
            "anomalies_found": anomalies,
            "history_size": len(self.price_history),
        }

        log.debug(
            "anomaly_analyzed",
            is_anomaly=is_anomaly,
            anomalies=anomalies,
            severity=max_severity,
            **details,
        )

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_type=anomalies[0] if anomalies else None,
            severity=max_severity,
            deviation_score=max_deviation,
            details=details,
        )

    def _zscore(self, value: float) -> float:
        if len(self.price_history) < 2:
            return 0.0
        prices = list(self.price_history)
        mean = sum(prices) / len(prices)
        variance = sum((p - mean) ** 2 for p in prices) / len(prices)
        std = variance ** 0.5
        if std == 0:
            return 0.0
        return (value - mean) / std

    def _zscore_spread(self, spread: float) -> float:
        if len(self.spread_history) < 2:
            return 0.0
        spreads = list(self.spread_history)
        mean = sum(spreads) / len(spreads)
        variance = sum((s - mean) ** 2 for s in spreads) / len(spreads)
        std = variance ** 0.5
        if std == 0:
            return 0.0
        return (spread - mean) / std

    def _zscore_volume(self, volume: float) -> float:
        if len(self.volume_history) < 2:
            return 0.0
        volumes = list(self.volume_history)
        mean = sum(volumes) / len(volumes)
        variance = sum((v - mean) ** 2 for v in volumes) / len(volumes)
        std = variance ** 0.5
        if std == 0:
            return 0.0
        return (volume - mean) / std

    def _mean_reversion_potential(self) -> float:
        if len(self.price_history) < 20:
            return 0.0
        prices = list(self.price_history)
        recent = prices[-10:]
        older = prices[-20:-10]
        recent_mean = sum(recent) / len(recent)
        older_mean = sum(older) / len(older)
        deviation = abs(recent_mean - older_mean) / older_mean if older_mean else 0.0
        return min(1.0, deviation * 10)

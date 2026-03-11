"""
Trend Detection Model
---------------------
Identifies directional trends in market prices using multiple timeframes
and technical indicators.
"""
from dataclasses import dataclass
from typing import Literal
import structlog

log = structlog.get_logger()

TrendDirection = Literal["bullish", "bearish", "neutral"]
TrendStrength = Literal["strong", "moderate", "weak"]


@dataclass
class TrendSignal:
    direction: TrendDirection
    strength: TrendStrength
    confidence: float
    indicators: dict


class TrendDetector:
    """
    Detects trends in market price data using multiple indicators:
    - SMA crossover (short vs long term)
    - Price momentum (rate of change)
    - Volume-weighted price position
    """

    def __init__(
        self,
        short_window: int = 5,
        long_window: int = 20,
        momentum_period: int = 10,
    ):
        self.short_window = short_window
        self.long_window = long_window
        self.momentum_period = momentum_period

    def analyze(self, price_history: list[float]) -> TrendSignal:
        """
        Analyze price history and return trend signal.
        Requires at least `long_window` data points.
        """
        if len(price_history) < self.long_window:
            return TrendSignal(
                direction="neutral",
                strength="weak",
                confidence=0.0,
                indicators={},
            )

        prices = price_history
        short_ma = self._sma(prices, self.short_window)
        long_ma = self._sma(prices, self.long_window)
        momentum = self._roc(prices, self.momentum_period)
        volatility = self._volatility(prices)

        ma_signal = self._ma_crossover_signal(short_ma, long_ma)
        momentum_signal = self._momentum_signal(momentum, volatility)

        combined_score = (ma_signal * 0.6) + (momentum_signal * 0.4)

        direction: TrendDirection
        if combined_score > 0.3:
            direction = "bullish"
        elif combined_score < -0.3:
            direction = "bearish"
        else:
            direction = "neutral"

        strength: TrendStrength
        if abs(combined_score) > 0.7:
            strength = "strong"
        elif abs(combined_score) > 0.4:
            strength = "moderate"
        else:
            strength = "weak"

        confidence = min(1.0, abs(combined_score))

        indicators = {
            "short_ma": short_ma,
            "long_ma": long_ma,
            "momentum": momentum,
            "volatility": volatility,
            "ma_signal": ma_signal,
            "momentum_signal": momentum_signal,
            "combined_score": combined_score,
        }

        log.debug(
            "trend_analyzed",
            direction=direction,
            strength=strength,
            confidence=confidence,
            **indicators,
        )

        return TrendSignal(
            direction=direction,
            strength=strength,
            confidence=confidence,
            indicators=indicators,
        )

    def _sma(self, prices: list[float], window: int) -> float:
        return sum(prices[-window:]) / window

    def _roc(self, prices: list[float], period: int) -> float:
        if len(prices) < period + 1:
            return 0.0
        current = prices[-1]
        past = prices[-period - 1]
        if past == 0:
            return 0.0
        return ((current - past) / past) * 100

    def _volatility(self, prices: list[float], window: int = 20) -> float:
        if len(prices) < window:
            window = len(prices)
        if window < 2:
            return 0.0
        recent = prices[-window:]
        mean = sum(recent) / window
        variance = sum((p - mean) ** 2 for p in recent) / window
        return variance ** 0.5

    def _ma_crossover_signal(self, short_ma: float, long_ma: float) -> float:
        if long_ma == 0:
            return 0.0
        return (short_ma - long_ma) / long_ma * 10

    def _momentum_signal(self, momentum: float, volatility: float) -> float:
        if volatility == 0:
            return 0.0
        return max(-1.0, min(1.0, momentum / (volatility * 2)))

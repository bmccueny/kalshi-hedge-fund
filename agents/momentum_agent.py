"""
Momentum Trading Agent
─────────────────────
Capitalizes on strong trends by entering trades in the direction of momentum.
Uses trend detection models to identify and ride trending markets.
"""
import asyncio
from dataclasses import dataclass
from agents.base_agent import BaseAgent, FAST_MODEL
from core.kalshi_client import KalshiClient
from core.models import TrendDetector
import structlog

log = structlog.get_logger()


@dataclass
class MomentumSignal:
    ticker: str
    direction: str
    strength: str
    confidence: float
    entry_price: float
    stop_loss: float
    reason: str


MOMENTUM_SYSTEM = """You are a momentum trading analyst for prediction markets.

Identify markets with strong trending behavior where momentum is likely to continue.
Consider:
- Trend strength and consistency
- Volume confirmation
- Time to resolution (prefer longer duration for momentum plays)
- Risk/reward ratio

Return JSON array:
[{"ticker": "...", "direction": "long|short", "entry_reason": "...", "stop_loss_pct": N}]
"""


class MomentumAgent(BaseAgent):
    """
    Momentum trading agent that identifies and trades with strong trends.
    """

    def __init__(self, kalshi: KalshiClient):
        super().__init__("Momentum", model=FAST_MODEL)
        self.kalshi = kalshi
        self.trend_detector = TrendDetector()
        self._price_cache: dict[str, list[float]] = {}

    async def find_momentum_opportunities(
        self, 
        min_confidence: float = 0.6,
        max_markets: int = 100,
    ) -> list[MomentumSignal]:
        """Find markets with strong momentum signals."""
        log.info("momentum_scan_start")

        markets = await self.kalshi.get_all_open_markets()
        markets = [m for m in markets if m.get("open_interest", 0) > 10000][:max_markets]
        log.info("momentum_markets_filtered", count=len(markets))

        signals = []
        for market in markets:
            try:
                signal = await self._analyze_market(market)
                if signal and signal.confidence >= min_confidence:
                    signals.append(signal)
            except Exception as e:
                log.debug("momentum_analysis_error", ticker=market.get("ticker"), error=str(e))

        signals.sort(key=lambda x: x.confidence, reverse=True)
        log.info("momentum_signals_found", count=len(signals))
        return signals[:10]

    async def _analyze_market(self, market: dict) -> MomentumSignal | None:
        """Analyze a single market for momentum."""
        ticker = market.get("ticker")
        if not ticker:
            return None

        history = await self.kalshi.get_market_history(ticker, limit=30)
        price_points = history.get("history", [])

        if len(price_points) < 10:
            return None

        yes_prices = [float(p.get("yes_price", 50)) for p in price_points if p.get("yes_price")]
        if len(yes_prices) < 10:
            return None

        signal = self.trend_detector.analyze(yes_prices)

        if signal.strength not in ("strong", "moderate"):
            return None

        current_price = yes_prices[-1]
        spread = market.get("yes_ask", 0) - market.get("yes_bid", 0)

        if signal.direction == "bullish":
            direction = "long"
            stop_loss = max(5, current_price - (current_price * 0.15))
        elif signal.direction == "bearish":
            direction = "short"
            stop_loss = min(95, current_price + (current_price * 0.15))
        else:
            return None

        return MomentumSignal(
            ticker=ticker,
            direction=direction,
            strength=signal.strength,
            confidence=signal.confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            reason=f"{signal.direction} trend, {signal.strength} strength, momentum score: {signal.indicators.get('combined_score', 0):.2f}",
        )

    async def run(self, min_confidence: float = 0.6) -> list[dict]:
        """Main entry point for the agent."""
        signals = await self.find_momentum_opportunities(min_confidence=min_confidence)
        return [
            {
                "ticker": s.ticker,
                "direction": s.direction,
                "strength": s.strength,
                "confidence": round(s.confidence, 2),
                "entry_price": s.entry_price,
                "stop_loss": round(s.stop_loss, 2),
                "reason": s.reason,
            }
            for s in signals
        ]

"""
Event-Driven Trading Agent
─────────────────────────
Base trades on specific events or announcements that are likely to impact market prices.
Monitors upcoming events and analyzes their potential market impact.
"""
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from agents.base_agent import BaseAgent, ANALYST_MODEL
from core.kalshi_client import KalshiClient
import structlog

log = structlog.get_logger()


@dataclass
class EventSignal:
    ticker: str
    event_type: str
    event_time: datetime | None
    market_price: float
    expected_move: float
    confidence: float
    rationale: str


EVENT_ANALYSIS_SYSTEM = """You are an event-driven trading analyst.

Analyze how upcoming events will impact prediction market prices.
Consider:
- Historical event impact patterns
- Market expectations vs reality
- Time decay (theta) approaching event
- Sentiment shifts around events

Return JSON array:
[{"ticker": "...", "event_type": "...", "expected_move_pct": N, "confidence": N, "rationale": "..."}]
"""


class EventDrivenAgent(BaseAgent):
    """
    Event-driven trading agent that identifies and trades based on upcoming events.
    """

    def __init__(self, kalshi: KalshiClient):
        super().__init__("EventDriven", model=ANALYST_MODEL)
        self.kalshi = kalshi

    async def find_event_opportunities(
        self,
        hours_ahead: int = 72,
        max_markets: int = 100,
    ) -> list[EventSignal]:
        """Find markets with upcoming events that may impact prices."""
        log.info("event_scan_start", hours_ahead=hours_ahead)

        markets = await self.kalshi.get_all_open_markets()
        markets = [m for m in markets if m.get("open_interest", 0) > 5000][:max_markets]

        upcoming = self._filter_upcoming_events(markets, hours_ahead)
        log.info("event_markets_upcoming", count=len(upcoming))

        signals = []
        for market in upcoming[:20]:
            try:
                signal = await self._analyze_event(market)
                if signal:
                    signals.append(signal)
            except Exception as e:
                log.debug("event_analysis_error", ticker=market.get("ticker"), error=str(e))

        signals.sort(key=lambda x: x.confidence * x.expected_move, reverse=True)
        log.info("event_signals_found", count=len(signals))
        return signals[:10]

    def _filter_upcoming_events(
        self, 
        markets: list[dict], 
        hours_ahead: int,
    ) -> list[dict]:
        """Filter markets with events happening within the time window."""
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours_ahead)
        upcoming = []

        for m in markets:
            close_time = m.get("close_time")
            if not close_time:
                continue

            try:
                if isinstance(close_time, str):
                    close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                else:
                    continue

                if now <= close_dt <= cutoff:
                    upcoming.append(m)
            except Exception:
                continue

        return upcoming

    async def _analyze_event(self, market: dict) -> EventSignal | None:
        """Analyze a market's upcoming event for trading opportunity."""
        ticker = market.get("ticker")
        title = market.get("title", "")
        yes_price = market.get("yes_ask", market.get("last_price", 50))
        close_time = market.get("close_time")

        close_dt = None
        if close_time:
            try:
                close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
            except Exception:
                pass

        event_keywords = {
            "earnings": ["earnings", "revenue", "EPS", "quarterly", "Q1", "Q2", "Q3", "Q4", "profit", "loss"],
            "fed": ["fed", "FOMC", "rate", "interest", "meeting", " Powell"],
            "election": ["election", "vote", "primary", "runoff", "ballot"],
            "economic": ["CPI", "GDP", "jobs", "unemployment", "PPI", "retail"],
            "sports": ["game", "match", "final", "championship", "playoff", "Super Bowl"],
            "weather": ["hurricane", "storm", "snow", "tornado", "forecast"],
        }

        event_type = "general"
        for etype, keywords in event_keywords.items():
            if any(k.lower() in title.lower() for k in keywords):
                event_type = etype
                break

        await asyncio.sleep(0.2)

        prompt = f"""Analyze event-driven opportunity for {ticker}:
- Title: {title}
- Current YES price: {yes_price} cents
- Close time: {close_time}

What price movement is expected based on this event type? Consider:
1. Historical price movements around similar events
2. Market sentiment
3. Time until event (volatility likely increases as event approaches)

Return JSON with expected move percentage and confidence (0-1)."""

        try:
            response = await self.analyze(prompt, system=EVENT_ANALYSIS_SYSTEM)
            parsed = self._extract_json(response)
            
            return EventSignal(
                ticker=ticker,
                event_type=event_type,
                event_time=close_dt,
                market_price=yes_price,
                expected_move=parsed.get("expected_move_pct", 5.0),
                confidence=parsed.get("confidence", 0.5),
                rationale=parsed.get("rationale", f"Event type: {event_type}"),
            )
        except Exception as e:
            log.debug("event_ai_analysis_failed", ticker=ticker, error=str(e))

            expected_move = 5.0
            if event_type in ("fed", "election"):
                expected_move = 15.0
            elif event_type in ("earnings", "economic"):
                expected_move = 10.0

            return EventSignal(
                ticker=ticker,
                event_type=event_type,
                event_time=close_dt,
                market_price=yes_price,
                expected_move=expected_move,
                confidence=0.4,
                rationale=f"Event type: {event_type} - using default estimate",
            )

    async def run(self, hours_ahead: int = 72) -> list[dict]:
        """Main entry point for the agent."""
        signals = await self.find_event_opportunities(hours_ahead=hours_ahead)
        return [
            {
                "ticker": s.ticker,
                "event_type": s.event_type,
                "event_time": s.event_time.isoformat() if s.event_time else None,
                "market_price": s.market_price,
                "expected_move_pct": s.expected_move,
                "confidence": round(s.confidence, 2),
                "rationale": s.rationale,
            }
            for s in signals
        ]

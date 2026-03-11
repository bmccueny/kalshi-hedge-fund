"""
Arbitrage Agent
────────────────
Exploits price differences between related markets or instruments.
Looks for:
- Intra-market arbitrage (yes/no spread inconsistencies)
- Cross-market arbitrage (same event different markets)
- Temporal arbitrage (price movements before events)
"""
import asyncio
from dataclasses import dataclass
from agents.base_agent import BaseAgent, FAST_MODEL
from core.kalshi_client import KalshiClient
import structlog

log = structlog.get_logger()


@dataclass
class ArbitrageOpportunity:
    ticker: str
    arbitrage_type: str
    expected_edge: float
    yes_price: float
    no_price: float
    fair_price: float
    volume: float
    reason: str


class ArbitrageAgent(BaseAgent):
    """
    Identifies and exploits arbitrage opportunities in prediction markets.
    """

    def __init__(self, kalshi: KalshiClient):
        super().__init__("Arbitrage", model=FAST_MODEL)
        self.kalshi = kalshi
        self._event_cache: dict[str, list[dict]] = {}

    async def find_opportunities(
        self,
        min_edge: float = 2.0,
        max_markets: int = 200,
    ) -> list[ArbitrageOpportunity]:
        """Find arbitrage opportunities across markets."""
        log.info("arbitrage_scan_start")

        markets = await self.kalshi.get_all_open_markets()
        markets = [m for m in markets if m.get("open_interest", 0) > 5000][:max_markets]
        log.info("arbitrage_markets_filtered", count=len(markets))

        opportunities = []

        spread_opps = self._find_spread_anomalies(markets)
        opportunities.extend(spread_opps)

        event_groups = self._group_by_event(markets)
        cross_opps = await self._find_cross_market_arbitrage(event_groups)
        opportunities.extend(cross_opps)

        opportunities.sort(key=lambda x: x.expected_edge, reverse=True)
        
        log.info("arbitrage_opportunities_found", count=len(opportunities))
        return [o for o in opportunities if o.expected_edge >= min_edge][:10]

    def _find_spread_anomalies(self, markets: list[dict]) -> list[ArbitrageOpportunity]:
        """Find markets where yes+no spread is inconsistent with 100."""
        opportunities = []

        for m in markets:
            yes_price = m.get("yes_ask", m.get("last_price", 50))
            no_price = m.get("no_ask", m.get("last_price", 50))

            if yes_price is None or no_price is None:
                continue

            total = yes_price + no_price
            deviation = abs(100 - total)

            if deviation > 3:
                fair_yes = 100 - no_price
                edge = abs(yes_price - fair_yes)

                opportunities.append(ArbitrageOpportunity(
                    ticker=m.get("ticker", ""),
                    arbitrage_type="spread_anomaly",
                    expected_edge=edge,
                    yes_price=yes_price,
                    no_price=no_price,
                    fair_price=fair_yes,
                    volume=m.get("volume", 0) / 100,
                    reason=f"Yes+No={total:.1f}, expected 100, deviation={deviation:.1f}",
                ))

        return opportunities

    def _group_by_event(self, markets: list[dict]) -> dict[str, list[dict]]:
        """Group markets by event ticker prefix."""
        groups = {}

        for m in markets:
            ticker = m.get("ticker", "")
            parts = ticker.split("-")
            if len(parts) >= 2:
                event_key = "-".join(parts[:2])
                if event_key not in groups:
                    groups[event_key] = []
                groups[event_key].append(m)

        return {k: v for k, v in groups.items() if len(v) >= 2}

    async def _find_cross_market_arbitrage(
        self, 
        event_groups: dict[str, list[dict]],
    ) -> list[ArbitrageOpportunity]:
        """Find arbitrage between related markets in the same event."""
        opportunities = []

        for event, markets in event_groups.items():
            if len(markets) < 2:
                continue

            prices = [(m.get("ticker"), m.get("yes_ask", 50), m.get("volume", 0)) for m in markets]
            prices.sort(key=lambda x: x[1])

            for i in range(len(prices) - 1):
                low_ticker, low_price, low_vol = prices[i]
                high_ticker, high_price, high_vol = prices[i + 1]

                spread = high_price - low_price
                if spread > 10:
                    opportunities.append(ArbitrageOpportunity(
                        ticker=f"{low_ticker} vs {high_ticker}",
                        arbitrage_type="cross_market",
                        expected_edge=spread,
                        yes_price=low_price,
                        no_price=high_price,
                        fair_price=(low_price + high_price) / 2,
                        volume=min(low_vol, high_vol) / 100,
                        reason=f"Same event price divergence: {spread:.1f} cents",
                    ))

        return opportunities

    async def run(self, min_edge: float = 2.0) -> list[dict]:
        """Main entry point for the agent."""
        opps = await self.find_opportunities(min_edge=min_edge)
        return [
            {
                "ticker": o.ticker,
                "type": o.arbitrage_type,
                "expected_edge_pct": round(o.expected_edge, 2),
                "yes_price": o.yes_price,
                "no_price": o.no_price,
                "volume_usd": round(o.volume, 2),
                "reason": o.reason,
            }
            for o in opps
        ]

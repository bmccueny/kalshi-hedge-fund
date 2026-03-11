"""
Market Research Agent
────────────────────
Researches and understands the types of prediction markets available on Kalshi.
Identifies trends and anomalies across the market landscape.

Provides:
- Market categorization by type (economic, political, sports, weather, etc.)
- Cross-market trend analysis
- Anomaly detection across categories
- Market landscape insights
"""
import asyncio
import json
from collections import defaultdict
from agents.base_agent import BaseAgent, GROK_ANALYST_MODEL
from core.kalshi_client import KalshiClient
from core.models import TrendDetector, AnomalyDetector
import structlog

log = structlog.get_logger()

RESEARCH_SYSTEM = """You are a market research analyst for a quantitative hedge fund.

Your job: Analyze a snapshot of prediction markets and provide insights about:
1. Market categories and their distribution
2. Notable trends (e.g., high conviction markets, converging/diverging probabilities)
3. Anomalies (e.g., mispricings relative to fundamentals, unusual volume)
4. Actionable observations

Return a JSON object with this structure:
{
  "categories": {"category_name": {"count": N, "total_volume_usd": N, "avg_price": N}},
  "trends": [{"ticker": "...", "observation": "...", "significance": "high|medium|low"}],
  "anomalies": [{"ticker": "...", "issue": "...", "potential_edge": N}],
  "summary": "2-3 sentence market overview"
}
"""


class MarketResearchAgent(BaseAgent):
    """
    Researches Kalshi's market landscape.
    Categorizes markets, detects cross-market trends, and identifies anomalies.
    """

    def __init__(self, kalshi: KalshiClient):
        super().__init__("MarketResearch", model=GROK_ANALYST_MODEL, provider="grok")
        self.kalshi = kalshi
        self._category_cache: dict | None = None
        self._ticker_category: dict[str, str] = {}

    async def research(self, max_markets: int = 200) -> dict:
        """
        Fetch markets, categorize them, and run trend/anomaly analysis.
        """
        log.info("research_start", max_markets=max_markets)

        markets = await self.kalshi.get_all_open_markets()
        markets = markets[:max_markets]
        log.info("research_fetched_markets", count=len(markets))

        categorized = self._categorize_markets(markets)
        self._category_cache = categorized

        price_by_category = self._aggregate_by_category(markets)
        volume_by_category = self._volume_by_category(markets)

        cross_market_trends = await self._analyze_cross_market_trends(markets)

        anomalies = await self._find_market_anomalies(markets)

        market_summaries = self._build_market_summaries(markets)

        analysis = await self.analyze(
            prompt=f"Analyze these prediction markets:\n\n{json.dumps(market_summaries, indent=2)}",
            system=RESEARCH_SYSTEM,
        )

        research_result = self._extract_json(analysis)

        research_result["categories"] = price_by_category
        research_result["volume_by_category"] = volume_by_category
        research_result["cross_market_trends"] = cross_market_trends
        research_result["detected_anomalies"] = anomalies
        research_result["market_count"] = len(markets)

        log.info("research_complete", categories=len(price_by_category), anomalies=len(anomalies))
        return research_result

    def _categorize_markets(self, markets: list[dict]) -> dict[str, list[dict]]:
        """Categorize markets by their title/category."""
        categories = defaultdict(list)
        ticker_to_cat: dict[str, str] = {}

        keywords = {
            "economic": ["GDP", "inflation", "CPI", "Fed", "rate", "unemployment", "PPI", "jobs", "GDP", "recession"],
            "political": ["election", "president", "congress", "senate", "governor", "vote", "ballot", "primary"],
            "weather": ["hurricane", "tornado", "snow", "rain", "temperature", "heat", "storm", "flood", "climate"],
            "sports": ["NFL", "NBA", "MLB", "NHL", "Super Bowl", "World Series", "championship", "game", "score"],
            "crypto": ["Bitcoin", "BTC", "Ethereum", "ETH", "crypto", "blockchain"],
            "stock": ["S&P", "Dow", "NASDAQ", "stock", "market", "rally", "crash"],
            "entertainment": ["Oscars", "Emmy", "Grammy", "movie", "box office", "TV show"],
            "science": ["space", "NASA", "Mars", "moon", "asteroid", "climate"],
        }

        for m in markets:
            title = m.get("title", "").lower()
            category = m.get("category", "").lower()
            ticker = m.get("ticker", "").lower()

            assigned = False
            for cat, words in keywords.items():
                if any(w.lower() in title or w.lower() in category or w.lower() in ticker for w in words):
                    categories[cat].append(m)
                    if ticker:
                        ticker_to_cat[ticker] = cat
                    assigned = True
                    break

            if not assigned:
                categories["other"].append(m)
                if ticker:
                    ticker_to_cat[ticker] = "other"

        self._ticker_category = ticker_to_cat
        return dict(categories)

    def _aggregate_by_category(self, markets: list[dict]) -> dict:
        """Calculate average price by category."""
        category_prices = defaultdict(list)

        for m in markets:
            cat = self._get_market_category(m)
            price = m.get("yes_ask", m.get("last_price", 50))
            if price:
                category_prices[cat].append(price)

        return {
            cat: {
                "count": len(prices),
                "avg_price": round(sum(prices) / len(prices), 1) if prices else 0,
            }
            for cat, prices in category_prices.items()
        }

    def _volume_by_category(self, markets: list[dict]) -> dict:
        """Calculate total volume by category."""
        category_volumes = defaultdict(float)

        for m in markets:
            cat = self._get_market_category(m)
            volume = m.get("volume", 0) / 100
            category_volumes[cat] += volume

        return {cat: round(vol, 2) for cat, vol in category_volumes.items()}

    def _get_market_category(self, market: dict) -> str:
        """Get the primary category for a market."""
        ticker = market.get("ticker", "")
        if ticker and self._ticker_category:
            return self._ticker_category.get(ticker, "other")
        if self._category_cache:
            for cat, markets in self._category_cache.items():
                if market in markets:
                    return cat
        return "other"

    async def _analyze_cross_market_trends(self, markets: list[dict]) -> list[dict]:
        """Analyze trends across related markets."""
        trends = []
        tickers = [m.get("ticker") for m in markets if m.get("ticker")]

        sem = asyncio.Semaphore(5)

        async def fetch_and_analyze(ticker: str) -> dict | None:
            async with sem:
                try:
                    history = await self.kalshi.get_market_history(ticker, limit=20)
                    price_points = history.get("history", [])
                    if len(price_points) < 10:
                        return None

                    yes_prices = [float(p.get("yes_price", 50)) for p in price_points if p.get("yes_price")]
                    if len(yes_prices) < 10:
                        return None

                    detector = TrendDetector()
                    signal = detector.analyze(yes_prices)

                    if signal.strength == "strong":
                        return {
                            "ticker": ticker,
                            "direction": signal.direction,
                            "strength": signal.strength,
                            "confidence": round(signal.confidence, 2),
                        }
                except Exception as e:
                    log.debug("trend_analysis_failed", ticker=ticker, error=str(e))
                return None

        results = await asyncio.gather(
            *[fetch_and_analyze(t) for t in tickers[:50]],
            return_exceptions=True
        )

        trends = [r for r in results if r and not isinstance(r, Exception)]
        trends.sort(key=lambda x: x.get("confidence", 0), reverse=True)

        return trends[:10]

    async def _find_market_anomalies(self, markets: list[dict]) -> list[dict]:
        """Find anomalous markets using statistical detection."""
        anomalies = []

        high_volume = [m for m in markets if m.get("volume", 0) > 50000]
        wide_spread = [m for m in markets if (m.get("yes_ask", 0) - m.get("yes_bid", 0)) > 10]

        extreme_prices = []
        for m in markets:
            price = m.get("yes_ask", m.get("last_price", 50))
            if price and (price < 10 or price > 90):
                extreme_prices.append(m)

        for m in high_volume[:5]:
            anomalies.append({
                "ticker": m.get("ticker"),
                "issue": "high_volume",
                "details": f"Volume: ${m.get('volume', 0) / 100:.0f}",
                "severity": "medium",
            })

        for m in wide_spread[:5]:
            spread = m.get("yes_ask", 0) - m.get("yes_bid", 0)
            anomalies.append({
                "ticker": m.get("ticker"),
                "issue": "wide_spread",
                "details": f"Spread: {spread} cents",
                "severity": "low",
            })

        for m in extreme_prices[:5]:
            price = m.get("yes_ask", m.get("last_price", 50))
            anomalies.append({
                "ticker": m.get("ticker"),
                "issue": "extreme_price",
                "details": f"Price: {price} cents",
                "severity": "medium" if price < 5 or price > 95 else "low",
            })

        return anomalies

    def _build_market_summaries(self, markets: list[dict]) -> list[dict]:
        """Build simplified market summaries for AI analysis."""
        summaries = []
        for m in markets:
            yes_price = m.get("yes_ask", m.get("last_price", 50))
            summaries.append({
                "ticker": m.get("ticker"),
                "title": m.get("title", "")[:80],
                "yes_price": yes_price,
                "volume_usd": round(m.get("volume", 0) / 100, 0),
                "open_interest_usd": round(m.get("open_interest", 0) / 100, 0),
                "category": self._get_market_category(m),
            })
        return summaries[:100]

    async def get_category_breakdown(self) -> dict:
        """Return cached category breakdown without re-fetching."""
        if self._category_cache:
            return {
                cat: len(markets) for cat, markets in self._category_cache.items()
            }
        return {}

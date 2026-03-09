"""
Market Scanner Agent
─────────────────────
Challenge overcome: Kalshi has thousands of markets. This agent rapidly
screens them to identify candidates worth deep analysis — filtering by
liquidity, spread quality, category interest, and time-to-resolution.

Uses claude-haiku-4-5 for fast batch scoring (cheaper, higher throughput).
"""
import json
import asyncio
from agents.base_agent import BaseAgent, FAST_MODEL
from core.kalshi_client import KalshiClient
from config.settings import MIN_LIQUIDITY_USD
import structlog

log = structlog.get_logger()

SCANNER_SYSTEM = """You are a prediction market opportunity scanner for a quantitative hedge fund.

Your job: given a batch of Kalshi markets, score each one for "worthiness of deep analysis".

Score each market 0–10 on:
- Information edge potential (can we know something the market doesn't?)
- Liquidity (enough to trade without moving the market)
- Time-to-resolution fit (prefer 1 day to 3 months)
- Category (prefer: macro/finance, elections, weather, sports outcomes — avoid trivia)

Return ONLY a JSON array of objects (no prose, no markdown, just the JSON):
[{"ticker": "...", "score": 8.2, "reason": "...", "category": "..."}]

Only include markets with score >= 6.
"""

# Limit concurrent AI calls to avoid overwhelming the CLI
_SCAN_SEMAPHORE = asyncio.Semaphore(4)


class MarketScannerAgent(BaseAgent):
    def __init__(self, kalshi: KalshiClient):
        super().__init__("MarketScanner", model=FAST_MODEL)
        self.kalshi = kalshi

    async def scan(self) -> list[dict]:
        """
        Fetch all open markets, batch-screen with AI, return high-potential candidates.
        """
        log.info("scanner_start")
        all_markets = await self.kalshi.get_all_open_markets()
        log.info("scanner_fetched_markets", count=len(all_markets))

        # Filter by basic liquidity first (fast, no AI needed)
        liquid_markets = [m for m in all_markets if self._passes_liquidity_filter(m)]
        log.info("scanner_after_liquidity_filter", count=len(liquid_markets))

        # Batch AI scoring in chunks of 10 (smaller batches = faster)
        candidates = []
        chunks = [liquid_markets[i:i+10] for i in range(0, len(liquid_markets), 10)]
        tasks = [self._score_batch(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                log.error("scanner_batch_error", error=str(result))
                continue
            if isinstance(result, list):
                candidates.extend(result)

        candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
        log.info("scanner_candidates", count=len(candidates))
        return candidates

    async def _score_batch(self, markets: list[dict]) -> list[dict]:
        """Send a batch of markets to the AI for scoring."""
        async with _SCAN_SEMAPHORE:
            market_summaries = []
            for m in markets:
                yes_price = m.get("yes_ask", m.get("last_price", 50))
                market_summaries.append({
                    "ticker": m.get("ticker"),
                    "title": m.get("title"),
                    "yes_price_cents": yes_price,
                    "no_price_cents": 100 - yes_price,
                    "volume_usd": m.get("volume", 0) / 100,
                    "open_interest_usd": m.get("open_interest", 0) / 100,
                    "close_time": m.get("close_time"),
                    "category": m.get("category", ""),
                })

            prompt = f"Score these Kalshi markets:\n\n{json.dumps(market_summaries, indent=2)}"
            response = await self.analyze(prompt, system=SCANNER_SYSTEM)

            scored = self._extract_json_array(response)

            # Enrich with full market data
            ticker_map = {m["ticker"]: m for m in markets}
            results = []
            for s in scored:
                if s.get("score", 0) >= 6 and s["ticker"] in ticker_map:
                    results.append({**ticker_map[s["ticker"]], **s})
            return results

    def _passes_liquidity_filter(self, market: dict) -> bool:
        """Fast pre-filter without AI."""
        # open_interest is in cents; MIN_LIQUIDITY_USD * 100 = min cents
        if market.get("open_interest", 0) < MIN_LIQUIDITY_USD * 100:
            return False
        yes_ask = market.get("yes_ask", 0)
        yes_bid = market.get("yes_bid", 0)
        if yes_ask == 0:  # no ask = no market
            return False
        if yes_ask - yes_bid > 15:  # > 15 cent spread = illiquid
            return False
        return True

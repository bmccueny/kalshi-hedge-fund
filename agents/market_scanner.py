"""
Market Scanner Agent
────────────────────
Challenge overcome: Kalshi has thousands of markets. This agent rapidly
screens them to identify candidates worth deep analysis — filtering by
liquidity, spread quality, category interest, and time-to-resolution.

Uses xAI Grok for fast batch scoring (cheaper, higher throughput).

Integrates trend detection and anomaly detection models for enhanced analysis.

Result caching: Scores are cached per-ticker with a fingerprint of key fields
(price, volume, open_interest). If a market hasn't materially changed since the
last scan, the cached score is reused — cutting AI calls by 50-80%.
"""
import json
import time
import asyncio
from agents.base_agent import BaseAgent, GROK_FAST_MODEL
from core.kalshi_client import KalshiClient
from core.models import TrendDetector, AnomalyDetector
from config.settings import MIN_LIQUIDITY_USD
import structlog

log = structlog.get_logger()

SCANNER_SYSTEM = """You are a prediction market opportunity scanner for a quantitative hedge fund.

Your job: given a batch of Kalshi markets, score each one for "worthiness of deep analysis".

Score each market 0–10 on:
- Information edge potential (can we know something the market doesn't?)
- Liquidity (enough to trade without moving the market)
- Time-to-resolution fit (prefer 1 day to 3 months, but all timeframes are valid)
- Trend signals (bullish/bearish trends may indicate momentum or reversal opportunities)
- Anomaly signals (price spikes, spread anomalies may indicate mispricing)

ALL categories are equally valid opportunities:
  weather, economics/finance, politics/elections, sports, crypto, tech, entertainment,
  science, legal, geopolitics, daily events, corporate actions, IPOs, and more.
DO NOT penalize or prefer any category over another. Judge purely on edge potential.

Return ONLY a JSON array of objects (no prose, no markdown, just the JSON):
[{"ticker": "...", "score": 8.2, "reason": "...", "category": "...", "trend": "...", "anomaly": "..."}]

Only include markets with score >= 6.
"""

# Limit concurrent AI calls to avoid overwhelming the CLI
_SCAN_SEMAPHORE = asyncio.Semaphore(4)


class MarketScannerAgent(BaseAgent):
    # Cache TTL: how long a score stays valid before requiring re-scoring
    _CACHE_TTL_SECONDS = 300  # 5 minutes

    def __init__(self, kalshi: KalshiClient):
        super().__init__("MarketScanner", model=GROK_FAST_MODEL, provider="grok")
        self.kalshi = kalshi
        # Score cache: ticker -> {"score_data": dict, "fingerprint": str, "cached_at": float}
        self._score_cache: dict[str, dict] = {}

    def _market_fingerprint(self, market: dict) -> str:
        """Create a fingerprint from fields that matter for scoring.
        If these haven't changed, the AI score won't change either."""
        return f"{market.get('yes_ask', 0)}|{market.get('yes_bid', 0)}|{market.get('volume', 0)}|{market.get('open_interest', 0)}|{market.get('close_time', '')}"

    def _get_cached_score(self, market: dict) -> dict | None:
        """Return cached score if the market hasn't materially changed."""
        ticker = market.get("ticker")
        if not ticker or ticker not in self._score_cache:
            return None
        entry = self._score_cache[ticker]
        # Expired?
        if time.monotonic() - entry["cached_at"] > self._CACHE_TTL_SECONDS:
            del self._score_cache[ticker]
            return None
        # Market changed?
        if self._market_fingerprint(market) != entry["fingerprint"]:
            del self._score_cache[ticker]
            return None
        return entry["score_data"]

    def _set_cached_score(self, market: dict, score_data: dict) -> None:
        """Cache a score result keyed by ticker + fingerprint."""
        ticker = market.get("ticker")
        if not ticker:
            return
        self._score_cache[ticker] = {
            "score_data": score_data,
            "fingerprint": self._market_fingerprint(market),
            "cached_at": time.monotonic(),
        }

    async def scan(self, categories: list[str] | None = None) -> list[dict]:
        """
        Fetch all open markets, batch-screen with AI, return high-potential candidates.
        Uses per-ticker caching to skip re-scoring markets that haven't changed.

        Args:
            categories: Optional list of category filters (e.g. ["weather", "sports"]).
                        None or ["all"] means scan everything.
        """
        # Normalize: None or ["all"] means no filtering
        if categories and "all" in [c.lower() for c in categories]:
            categories = None

        log.info("scanner_start", categories=categories)
        all_markets = await self.kalshi.get_all_open_markets()
        log.info("scanner_fetched_markets", count=len(all_markets))

        # Filter by basic liquidity first (fast, no AI needed)
        liquid_markets = []
        for m in all_markets:
            if self._passes_liquidity_filter(m, categories=categories):
                liquid_markets.append(m)
                log.info(
                    "market_passes_filter",
                    ticker=m.get("ticker", "?"),
                    title=m.get("title", "")[:50],
                    open_interest_usd=m.get("open_interest", 0) // 100,
                    spread=m.get("yes_ask", 0) - m.get("yes_bid", 0),
                )
        log.info("scanner_after_liquidity_filter", count=len(liquid_markets))

        # Separate cached hits from markets that need AI scoring
        candidates = []
        needs_scoring = []
        cache_hits = 0
        for m in liquid_markets:
            cached = self._get_cached_score(m)
            if cached is not None:
                candidates.append(cached)
                cache_hits += 1
            else:
                needs_scoring.append(m)

        log.info("scanner_cache_stats",
                 total=len(liquid_markets), cache_hits=cache_hits,
                 needs_scoring=len(needs_scoring))

        # Batch AI scoring only for uncached markets
        if needs_scoring:
            chunks = [needs_scoring[i:i+10] for i in range(0, len(needs_scoring), 10)]
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
            # Fetch historical data for trend/anomaly detection
            await self._enrich_with_market_data(markets)

            market_summaries = []
            for m in markets:
                yes_price = m.get("yes_ask", m.get("last_price", 50))
                summary = {
                    "ticker": m.get("ticker"),
                    "title": m.get("title"),
                    "yes_price_cents": yes_price,
                    "no_price_cents": 100 - yes_price,
                    "volume_usd": m.get("volume", 0) / 100,
                    "open_interest_usd": m.get("open_interest", 0) / 100,
                    "close_time": m.get("close_time"),
                    "category": m.get("category", ""),
                }

                # Add trend analysis if available
                if "trend_signal" in m:
                    ts = m["trend_signal"]
                    summary["trend_direction"] = ts.direction
                    summary["trend_strength"] = ts.strength
                    summary["trend_confidence"] = round(ts.confidence, 2)

                # Add anomaly analysis if available
                if "anomaly_result" in m:
                    ar = m["anomaly_result"]
                    summary["anomaly_detected"] = ar.is_anomaly
                    summary["anomaly_type"] = ar.anomaly_type
                    summary["anomaly_severity"] = round(ar.severity, 2)

                market_summaries.append(summary)

            prompt = f"Score these Kalshi markets:\n\n{json.dumps(market_summaries, indent=2)}"
            response = await self.analyze(prompt, system=SCANNER_SYSTEM)

            scored = self._extract_json_array(response)

            # Enrich with full market data and cache results
            ticker_map = {m["ticker"]: m for m in markets}
            results = []
            for s in scored:
                if s.get("score", 0) >= 6 and s["ticker"] in ticker_map:
                    enriched = {**ticker_map[s["ticker"]], **s}
                    results.append(enriched)
                    # Cache the enriched score for this market
                    self._set_cached_score(ticker_map[s["ticker"]], enriched)
            return results

    async def _enrich_with_market_data(self, markets: list[dict]) -> None:
        """Fetch historical data and run trend/anomaly detection on each market."""
        trend_detector = TrendDetector()
        anomaly_detector = AnomalyDetector()

        for market in markets:
            ticker = market.get("ticker")
            if not ticker:
                continue

            try:
                history = await self.kalshi.get_market_history(ticker, limit=30)
                price_points = history.get("history", [])

                if len(price_points) >= 10:
                    yes_prices = [float(p.get("yes_price", 50)) for p in price_points if p.get("yes_price")]

                    if len(yes_prices) >= 10:
                        trend_signal = trend_detector.analyze(yes_prices)
                        market["trend_signal"] = trend_signal

                        anomaly_result = anomaly_detector.analyze(
                            current_price=yes_prices[-1],
                            current_spread=market.get("yes_ask", 0) - market.get("yes_bid", 0),
                            current_volume=market.get("volume", 0),
                        )
                        market["anomaly_result"] = anomaly_result

            except Exception as e:
                log.debug("market_data_enrichment_failed", ticker=ticker, error=str(e))

    def _passes_liquidity_filter(self, market: dict, categories: list[str] | None = None) -> bool:
        """Fast pre-filter without AI. Optionally filter by category."""
        # Category filter (if specified)
        if categories:
            market_cat = (market.get("category", "") or "").lower()
            market_title = (market.get("title", "") or "").lower()
            market_ticker = (market.get("ticker", "") or "").lower()
            if not self._matches_category(market_cat, market_title, market_ticker, categories):
                return False

        # open_interest is in cents; MIN_LIQUIDITY_USD * 100 = min cents
        if market.get("open_interest", 0) < MIN_LIQUIDITY_USD * 100:
            return False
        yes_ask = market.get("yes_ask", 0)
        yes_bid = market.get("yes_bid", 0)
        if yes_ask == 0:  # no ask = no market
            return False
        if yes_ask - yes_bid > 25:  # > 25 cent spread = illiquid (widened from 15 for niche markets)
            return False
        return True

    @staticmethod
    def _matches_category(market_cat: str, title: str, ticker: str, categories: list[str]) -> bool:
        """Check if a market matches any of the requested category filters."""
        # Keyword map: category filter name -> keywords to match in title/ticker/category
        _CATEGORY_KEYWORDS: dict[str, list[str]] = {
            "weather": ["weather", "temperature", "rain", "snow", "hurricane", "tornado", "climate", "heat", "cold", "flood"],
            "politics": ["president", "election", "democrat", "republican", "senate", "congress", "governor", "vote", "impeach", "nominee", "political", "trump", "biden"],
            "economics": ["gdp", "inflation", "unemployment", "fed ", "interest rate", "recession", "cpi", "jobs report", "tariff", "trade war", "stock", "s&p", "nasdaq", "dow"],
            "sports": ["nba", "nfl", "mlb", "nhl", "soccer", "football", "basketball", "baseball", "hockey", "championship", "super bowl", "world series", "match", "game", "playoff"],
            "crypto": ["bitcoin", "ethereum", "crypto", "btc", "eth", "solana", "dogecoin", "token"],
            "tech": ["ai ", "artificial intelligence", "spacex", "ipo", "tesla", "apple", "google", "meta", "openai", "startup", "launch"],
            "entertainment": ["oscar", "grammy", "movie", "film", "album", "tv show", "emmy", "box office", "streaming", "celebrity"],
            "geopolitics": ["war", "nato", "china", "russia", "ukraine", "taiwan", "greenland", "territory", "sanction", "military", "canal"],
            "science": ["space", "nasa", "mars", "moon", "vaccine", "fda", "drug", "clinical trial", "disease", "pandemic"],
            "daily": ["today", "this week", "tomorrow", "daily", "weekly"],
        }
        combined = f"{market_cat} {title} {ticker}"
        for cat_filter in categories:
            cat_lower = cat_filter.lower()
            if cat_lower == "all":
                return True
            keywords = _CATEGORY_KEYWORDS.get(cat_lower, [cat_lower])
            if any(kw in combined for kw in keywords):
                return True
        return False

"""
Probability Analyst Agent
──────────────────────────
Challenge overcome: The hardest problem in prediction market trading is
accurately estimating "true" probabilities independent of the market price.
Biases (overconfidence, recency, narrative bias) systematically misprice events.

This agent:
1. Pre-fetches market history, news, and related markets in Python
2. Embeds all context in a single comprehensive prompt
3. Uses Claude Opus with deep reasoning to produce a calibrated probability estimate
4. Returns a confidence-weighted ProbabilityEstimate with edge vs market price
"""
import json
import time
import asyncio
from dataclasses import dataclass
from datetime import datetime
from agents.base_agent import BaseAgent
from core.kalshi_client import KalshiClient
from data.news_feed import NewsFeed
from config.settings import DAILY_AI_COST_LIMIT_USD
import structlog

log = structlog.get_logger()


@dataclass
class ProbabilityEstimate:
    ticker: str
    market_title: str
    market_price_pct: float          # Market's implied probability (0-100)
    true_prob_pct: float             # Agent's estimate (0-100)
    edge_pct: float                  # true_prob - market_price (positive = YES underpriced)
    confidence: float                # 0.0 (low) to 1.0 (high)
    rationale: str
    key_risks: list[str]
    information_needs: list[str]     # What would update this estimate most
    recommended_side: str | None     # "yes" | "no" | None (no trade)
    recommended_size_pct: float      # % of max position size to use


ANALYST_SYSTEM = """You are a senior quantitative analyst at a prediction market hedge fund.

Your task: estimate the TRUE probability of a prediction market event resolving YES,
using the market data, price history, recent news, and related markets provided below.

METHODOLOGY:
1. Base rates: What fraction of similar historical events resolved YES?
2. Reference class: Choose the most appropriate comparison group (avoid narrow framing)
3. Current evidence: Weight recent data, expert forecasts, and leading indicators
4. Decomposition: Break complex events into sub-probabilities if possible
5. Calibration: LLMs tend to be overconfident; apply a 10-15% pull toward 50%
   for uncertain events. For well-defined events with clear data, less pullback.
6. Uncertainty bounds: Report a range, not just a point estimate

CALIBRATION REMINDERS:
- Rare events (<5%) are often overestimated by models
- Common events (>95%) are often underestimated
- "Last year it happened" is weak base rate evidence — check 10+ year rates
- Breaking news often moves markets before full information is available — be skeptical
- Political/election markets: polls are noisy, weight aggregate forecasts

OUTPUT FORMAT (valid JSON, nothing else outside the JSON block):
{
  "true_prob_pct": <float 0-100>,
  "confidence": <float 0.0-1.0>,
  "rationale": "<detailed reasoning, 200-400 words>",
  "base_rate": "<what comparable historical base rate suggests>",
  "key_risks": ["<risk1>", "<risk2>", ...],
  "information_needs": ["<what data would most update this>", ...],
  "recommended_side": "yes" | "no" | null,
  "kelly_fraction": <float 0-1>
}
"""


class ProbabilityAnalystAgent(BaseAgent):
    # Approximate cost per AI call: ~2000 input + ~500 output tokens with Opus
    # $15/M input + $75/M output ≈ $0.068 per call
    _COST_PER_CALL_USD = 0.068

    def __init__(self, kalshi: KalshiClient, news: "NewsFeed | None" = None):
        super().__init__("ProbabilityAnalyst")
        self.kalshi = kalshi
        self.news = news
        # In-memory cache: ticker -> (ProbabilityEstimate, cached_at timestamp)
        self._cache: dict[str, tuple["ProbabilityEstimate", float]] = {}
        self._cache_ttl = 600  # 10 minutes
        # Daily cost tracking
        self._daily_cost_usd = 0.0
        self._cost_date = ""  # YYYY-MM-DD of current spend bucket

    def _get_cached(self, ticker: str) -> "ProbabilityEstimate | None":
        if ticker in self._cache:
            estimate, cached_at = self._cache[ticker]
            if time.monotonic() - cached_at < self._cache_ttl:
                return estimate
            del self._cache[ticker]
        return None

    def _set_cached(self, ticker: str, estimate: "ProbabilityEstimate") -> None:
        self._cache[ticker] = (estimate, time.monotonic())

    def _check_cost_limit(self) -> bool:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if today != self._cost_date:
            self._daily_cost_usd = 0.0
            self._cost_date = today
        return self._daily_cost_usd < DAILY_AI_COST_LIMIT_USD

    async def analyze(self, market: dict) -> "ProbabilityEstimate | None":
        ticker = market["ticker"]
        yes_ask = market.get("yes_ask", 50)
        market_prob = yes_ask  # cents = implied probability %

        # Return cached result if fresh (avoids redundant AI calls)
        cached = self._get_cached(ticker)
        if cached is not None:
            log.info("analyst_cache_hit", ticker=ticker,
                     age_seconds=round(time.monotonic() - self._cache[ticker][1]))
            return cached

        # Enforce daily AI spend cap
        if not self._check_cost_limit():
            log.warning(
                "analyst_daily_cost_limit_reached",
                daily_cost_usd=round(self._daily_cost_usd, 3),
                limit_usd=DAILY_AI_COST_LIMIT_USD,
            )
            return None

        log.info("analyst_start", ticker=ticker, market_prob=market_prob)

        # Pre-fetch all context in parallel (no AI tool calls needed)
        history_task = self.kalshi.get_market_history(ticker, limit=100)
        news_task = (
            self.news.search(market.get("title", ticker), days_back=7)
            if self.news else asyncio.sleep(0, result=[])
        )
        related_task = self.kalshi.get_markets(
            series_ticker=market.get("series_ticker", "")
        ) if market.get("series_ticker") else asyncio.sleep(0, result={"markets": []})

        history, news_articles, related_data = await asyncio.gather(
            history_task, news_task, related_task, return_exceptions=True
        )

        # Handle exceptions and wrong types
        if isinstance(news_articles, Exception):
            news_articles = []
        if isinstance(news_articles, dict):
            news_articles = news_articles.get("articles", [])
        if not isinstance(news_articles, list):
            news_articles = []

        # Build comprehensive prompt with all context embedded
        prompt = f"""Analyze this Kalshi prediction market:

## Market Details
Ticker: {ticker}
Title: {market.get('title', ticker)}
Description: {market.get('description', 'N/A')}
Current YES price: {yes_ask}¢ (market implies {yes_ask}% probability of YES)
Close time: {market.get('close_time', 'N/A')}
Volume (total): ${market.get('volume', 0)/100:.2f}
Open interest: ${market.get('open_interest', 0)/100:.2f}

## Price History (most recent 50 data points)
{json.dumps(history[:50] if isinstance(history, list) else [], indent=2)}

## Recent News (last 7 days)
{json.dumps([{"title": a.get("title", ""), "source": a.get("source", ""), "body": a.get("description", "")[:300], "published": a.get("publishedAt", "")} for a in news_articles[:8]], indent=2)}

## Related Markets
{json.dumps((related_data.get("markets", []) if isinstance(related_data, dict) else [])[:10], indent=2)}

Based on all of the above, provide your calibrated probability estimate in the required JSON format."""

        log.info("analyst_calling_ai", ticker=ticker)
        result = await super().analyze(prompt, system=ANALYST_SYSTEM, max_turns=2)

        if not result:
            log.warning("analyst_empty_result", ticker=ticker)
            return None

        # Charge estimated cost for this call
        self._daily_cost_usd += self._COST_PER_CALL_USD
        log.debug("analyst_cost_update",
                  call_cost=self._COST_PER_CALL_USD,
                  daily_total=round(self._daily_cost_usd, 3))

        try:
            data = self._extract_json(result)
            true_prob = float(data["true_prob_pct"])
            edge = true_prob - market_prob

            estimate = ProbabilityEstimate(
                ticker=ticker,
                market_title=market.get("title", ticker),
                market_price_pct=market_prob,
                true_prob_pct=true_prob,
                edge_pct=edge,
                confidence=float(data["confidence"]),
                rationale=data["rationale"],
                key_risks=data.get("key_risks", []),
                information_needs=data.get("information_needs", []),
                recommended_side=data.get("recommended_side"),
                recommended_size_pct=float(data.get("kelly_fraction", 0.25)),
            )
            self._set_cached(ticker, estimate)
            return estimate
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            log.error("analyst_parse_error", ticker=ticker, error=str(e), result=result[:500])
            return None

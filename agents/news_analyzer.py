"""
News Analyzer Agent
────────────────────
Challenge overcome: Real-time news can move prediction markets significantly,
but most news is noise. The key is identifying HIGH-SIGNAL breaking news that
actually changes event probabilities, then acting before the market reprices.

This agent:
1. Continuously monitors news feeds for relevant developments
2. Fast-triages each story: noise vs signal
3. Deep-analyzes signals: which markets are affected and by how much?
4. Emits alerts for the risk manager and trade executor
"""
import json
import asyncio
from dataclasses import dataclass
from datetime import datetime
from agents.base_agent import BaseAgent, FAST_MODEL, ANALYST_MODEL
from data.news_feed import NewsFeed, Article
import structlog

log = structlog.get_logger()


@dataclass
class NewsSignal:
    article: Article
    affected_tickers: list[str]
    direction: str              # "yes_up" | "yes_down" | "neutral"
    prob_shift_pct: float       # estimated change in YES probability
    urgency: str                # "immediate" | "hours" | "days" | "noise"
    confidence: float
    analysis: str


TRIAGE_SYSTEM = """You are a news triage engine for a prediction market trading firm.

Given a news article, determine:
1. Is this SIGNAL (actually changes event probabilities) or NOISE (irrelevant)?
2. Which market categories are affected? (politics, finance, weather, sports, tech, etc.)
3. What is the urgency? immediate/hours/days/noise

Return ONLY valid JSON (no prose):
{"signal": true/false, "categories": ["..."], "urgency": "immediate|hours|days|noise", "summary": "..."}
"""

IMPACT_SYSTEM = """You are a quantitative analyst assessing news impact on prediction market prices.

Given a news article and a list of relevant Kalshi markets, determine:
1. Which specific tickers are directly affected?
2. For each: does this push YES probability UP or DOWN, by how many percentage points?
3. Confidence in your assessment (0-1)

Be conservative: only flag markets where the news is clearly relevant.
Typical news moves markets by 2-15 percentage points. Major surprises: 15-40%.

Return ONLY valid JSON:
{
  "affected": [
    {"ticker": "...", "direction": "yes_up|yes_down", "prob_shift_pct": 8.0, "confidence": 0.7}
  ],
  "analysis": "..."
}
"""

_TRIAGE_SEMAPHORE = asyncio.Semaphore(6)
_ANALYZE_SEMAPHORE = asyncio.Semaphore(2)


class NewsAnalyzerAgent(BaseAgent):
    def __init__(self, news: NewsFeed, open_markets: list[dict] | None = None):
        super().__init__("NewsAnalyzer", model=FAST_MODEL)
        self.news = news
        self.open_markets = open_markets or []
        self._seen_article_ids: set[str] = set()
        self._signal_callbacks: list = []

    def on_signal(self, callback):
        """Register a callback for when a news signal is detected."""
        self._signal_callbacks.append(callback)
        return callback

    async def run(self, poll_interval: int = 60):
        """Continuously monitor news and emit signals."""
        log.info("news_analyzer_start")
        while True:
            try:
                articles = await self.news.get_latest(limit=50)
                new_articles = [a for a in articles if a.id not in self._seen_article_ids]
                if new_articles:
                    log.info("news_new_articles", count=len(new_articles))
                    await self._process_articles(new_articles)
                    self._seen_article_ids.update(a.id for a in new_articles)
            except Exception as e:
                log.error("news_analyzer_error", error=str(e))
            await asyncio.sleep(poll_interval)

    async def _process_articles(self, articles: list[Article]):
        """Triage all articles in parallel, then deep-analyze signals."""
        triage_tasks = [self._triage(a) for a in articles]
        triage_results = await asyncio.gather(*triage_tasks, return_exceptions=True)

        signals_to_analyze = []
        for article, triage in zip(articles, triage_results):
            if isinstance(triage, Exception):
                continue
            if triage.get("signal") and triage.get("urgency") in ("immediate", "hours"):
                signals_to_analyze.append((article, triage))

        if signals_to_analyze:
            log.info("news_signals_found", count=len(signals_to_analyze))
            analysis_tasks = [self._deep_analyze(a, t) for a, t in signals_to_analyze]
            signals = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            for signal in signals:
                if isinstance(signal, NewsSignal):
                    await self._emit_signal(signal)

    async def _triage(self, article: Article) -> dict:
        """Fast triage: is this news signal-worthy?"""
        async with _TRIAGE_SEMAPHORE:
            prompt = f"Headline: {article.title}\n\nBody: {article.body[:500]}"
            response = await self.analyze(prompt, system=TRIAGE_SYSTEM, max_turns=1)
            return self._extract_json(response)

    async def _deep_analyze(self, article: Article, triage: dict) -> "NewsSignal | None":
        """Deep analysis: which markets does this affect and by how much?"""
        async with _ANALYZE_SEMAPHORE:
            categories = triage.get("categories", [])
            relevant_markets = [
                {"ticker": m["ticker"], "title": m.get("title", ""), "category": m.get("category", "")}
                for m in self.open_markets
                if any(c.lower() in str(m.get("category", "")).lower() for c in categories)
            ][:50]

            prompt = f"""News article:
Title: {article.title}
Source: {article.source}
Published: {article.published_at}
Body: {article.body[:1500]}

Relevant open markets to consider:
{json.dumps(relevant_markets, indent=2)}

Triage summary: {triage.get('summary', '')}"""

            # Use opus model for deep analysis
            old_model = self.model
            self.model = ANALYST_MODEL
            response = await self.analyze(prompt, system=IMPACT_SYSTEM, max_turns=2)
            self.model = old_model

            data = self._extract_json(response)
            affected = data.get("affected", [])
            if not affected:
                return None

            return NewsSignal(
                article=article,
                affected_tickers=[a["ticker"] for a in affected],
                direction=affected[0]["direction"] if affected else "neutral",
                prob_shift_pct=max(abs(a.get("prob_shift_pct", 0)) for a in affected),
                urgency=triage.get("urgency", "hours"),
                confidence=max(a.get("confidence", 0) for a in affected),
                analysis=data.get("analysis", ""),
            )

    async def _emit_signal(self, signal: NewsSignal):
        log.info(
            "news_signal",
            tickers=signal.affected_tickers,
            direction=signal.direction,
            shift_pct=signal.prob_shift_pct,
            urgency=signal.urgency,
        )
        for cb in self._signal_callbacks:
            try:
                await cb(signal)
            except Exception as e:
                log.error("signal_callback_error", error=str(e))

"""
News feed abstraction — fetches articles from NewsAPI or Polygon.io.
Swap in any news provider by implementing the NewsFeeed interface.
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import httpx
from config.settings import NEWS_API_KEY, POLYGON_API_KEY
import structlog

log = structlog.get_logger()


@dataclass
class Article:
    id: str
    title: str
    body: str
    source: str
    published_at: str
    url: str
    keywords: list[str] = field(default_factory=list)


class NewsFeeed:
    """Aggregates from multiple news sources."""

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=10.0)
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    async def get_latest(self, limit: int = 50) -> list[Article]:
        """Fetch latest news across all configured sources."""
        sources = []
        tasks = []

        if NEWS_API_KEY:
            tasks.append(self._fetch_newsapi(limit))
        if POLYGON_API_KEY:
            tasks.append(self._fetch_polygon(limit))

        if not tasks:
            log.warning("no_news_sources_configured")
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, list):
                sources.extend(r)

        # Deduplicate and sort by recency
        seen = set()
        unique = []
        for a in sources:
            if a.id not in seen:
                seen.add(a.id)
                unique.append(a)

        unique.sort(key=lambda a: a.published_at, reverse=True)
        return unique[:limit]

    async def search(self, query: str, days_back: int = 7) -> list[dict]:
        """Search for articles relevant to a query."""
        if not NEWS_API_KEY:
            return []

        from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        try:
            r = await self._client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "from": from_date,
                    "sortBy": "relevancy",
                    "pageSize": 20,
                    "apiKey": NEWS_API_KEY,
                },
            )
            r.raise_for_status()
            articles = r.json().get("articles", [])
            return [
                {
                    "title": a.get("title", ""),
                    "description": a.get("description", ""),
                    "content": a.get("content", ""),
                    "source": a.get("source", {}).get("name", ""),
                    "publishedAt": a.get("publishedAt", ""),
                    "url": a.get("url", ""),
                }
                for a in articles
            ]
        except Exception as e:
            log.error("newsapi_search_error", error=str(e))
            return []

    async def _fetch_newsapi(self, limit: int) -> list[Article]:
        try:
            r = await self._client.get(
                "https://newsapi.org/v2/top-headlines",
                params={
                    "language": "en",
                    "pageSize": min(limit, 100),
                    "apiKey": NEWS_API_KEY,
                    "category": "business,politics,science,technology",
                },
            )
            r.raise_for_status()
            articles = r.json().get("articles", [])
            return [
                Article(
                    id=f"newsapi-{hash(a.get('url', ''))}",
                    title=a.get("title", ""),
                    body=a.get("content") or a.get("description") or "",
                    source=a.get("source", {}).get("name", "NewsAPI"),
                    published_at=a.get("publishedAt", ""),
                    url=a.get("url", ""),
                )
                for a in articles if a.get("title")
            ]
        except Exception as e:
            log.error("newsapi_fetch_error", error=str(e))
            return []

    async def _fetch_polygon(self, limit: int) -> list[Article]:
        try:
            r = await self._client.get(
                "https://api.polygon.io/v2/reference/news",
                params={
                    "limit": min(limit, 50),
                    "sort": "published_utc",
                    "order": "desc",
                    "apiKey": POLYGON_API_KEY,
                },
            )
            r.raise_for_status()
            results = r.json().get("results", [])
            return [
                Article(
                    id=f"polygon-{a.get('id', hash(a.get('article_url', '')))}",
                    title=a.get("title", ""),
                    body=a.get("description", ""),
                    source=a.get("publisher", {}).get("name", "Polygon"),
                    published_at=a.get("published_utc", ""),
                    url=a.get("article_url", ""),
                    keywords=a.get("keywords", []),
                )
                for a in results if a.get("title")
            ]
        except Exception as e:
            log.error("polygon_fetch_error", error=str(e))
            return []

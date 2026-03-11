"""
News feed abstraction — fetches articles from NewsAPI or Polygon.io.
Swap in any news provider by implementing the NewsFeed interface.
Includes fallback to web scraper when rate limited.
"""
import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import time
import httpx
from config.settings import NEWS_API_KEY, POLYGON_API_KEY
import structlog
from data.news_scraper import NewsScraper, RSSFeeds

log = structlog.get_logger()

NEWSAPI_RATE_LIMIT = 100
NEWSAPI_RATE_WINDOW = 86400


@dataclass
class Article:
    id: str
    title: str
    body: str
    source: str
    published_at: str
    url: str
    keywords: list[str] = field(default_factory=list)


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, max_calls: int, window_seconds: int):
        self.max_calls = max_calls
        self.window = window_seconds
        self.calls: list[float] = []

    async def acquire(self):
        now = time.time()
        self.calls = [c for c in self.calls if now - c < self.window]

        if len(self.calls) >= self.max_calls:
            sleep_time = self.window - (now - self.calls[0])
            if sleep_time > 0:
                log.info("rate_limit_waiting", seconds=sleep_time)
                await asyncio.sleep(sleep_time)
                self.calls = []

        self.calls.append(time.time())


class NewsFeed:
    """Aggregates from multiple news sources."""

    def __init__(self, save_to_file: str | None = None, use_fallback: bool = True):
        """
        Args:
            save_to_file: Optional path to save news findings after each scrape (JSONL format)
            use_fallback: Use scraper fallback when API is rate limited
        """
        self._client: httpx.AsyncClient | None = None
        self._newsapi_limiter = RateLimiter(NEWSAPI_RATE_LIMIT, NEWSAPI_RATE_WINDOW)
        self._cache: dict[str, tuple[list, float]] = {}
        self._cache_ttl = 300
        self._save_to_file = save_to_file
        self._use_fallback = use_fallback
        self._scraper: NewsScraper | None = None
        self._use_scraper = False

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=10.0)
        if self._use_fallback:
            self._scraper = NewsScraper()
            await self._scraper.__aenter__()
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()
        if self._scraper:
            await self._scraper.__aexit__(*args)

    async def _get_fallback_news(self, limit: int) -> list[Article]:
        """Get news from scraper as fallback."""
        if not self._scraper:
            return []
        
        try:
            articles = await self._scraper.get_business_news(limit)
            return [
                Article(
                    id=a.id,
                    title=a.title,
                    body=a.body,
                    source=a.source,
                    published_at=a.published_at,
                    url=a.url,
                    keywords=a.keywords,
                )
                for a in articles
            ]
        except Exception as e:
            log.error("fallback_scraper_failed", error=str(e))
            return []

    async def _search_fallback(self, query: str, days_back: int) -> list[dict]:
        """Search using scraper as fallback."""
        if not self._scraper:
            return []
        
        try:
            articles = await self._scraper.search_news(query)
            return [
                {
                    "title": a.title,
                    "description": a.body,
                    "content": a.body,
                    "source": a.source,
                    "publishedAt": a.published_at,
                    "url": a.url,
                }
                for a in articles
            ]
        except Exception as e:
            log.error("fallback_search_failed", error=str(e))
            return []

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        max_retries: int = 3,
        **kwargs,
    ) -> httpx.Response | None:
        """Make HTTP request with retry and rate limiting."""
        for attempt in range(max_retries):
            try:
                r = await self._client.request(method, url, **kwargs)
                
                if r.status_code == 429:
                    retry_after = float(r.headers.get("Retry-After", 60))
                    log.warning("newsapi_rate_limited", attempt=attempt, wait=retry_after)
                    await asyncio.sleep(retry_after)
                    continue
                    
                r.raise_for_status()
                return r
                
            except httpx.HTTPStatusError as e:
                if attempt < max_retries - 1 and e.response.status_code >= 500:
                    wait = 2 ** attempt
                    log.warning("newsapi_server_error", attempt=attempt, wait=wait)
                    await asyncio.sleep(wait)
                    continue
                log.error("newsapi_request_failed", error=str(e))
                return None
            except Exception as e:
                log.error("newsapi_request_error", error=str(e))
                return None
                
        return None

    def _get_cached(self, key: str) -> list | None:
        """Get cached result if fresh."""
        if key in self._cache:
            data, cached_at = self._cache[key]
            if time.time() - cached_at < self._cache_ttl:
                return data
        return None

    def _set_cached(self, key: str, data: list) -> None:
        """Cache result."""
        self._cache[key] = (data, time.time())

    async def _save_findings(self, articles: list[Article], source: str = "latest") -> None:
        """Save news findings to file if configured."""
        if not self._save_to_file:
            return
            
        import json
        
        timestamp = datetime.now(timezone.utc).isoformat()
        
        try:
            with open(self._save_to_file, "a") as f:
                for article in articles:
                    record = {
                        "timestamp": timestamp,
                        "source": source,
                        "id": article.id,
                        "title": article.title,
                        "body": article.body[:500] if article.body else "",
                        "article_source": article.source,
                        "published_at": article.published_at,
                        "url": article.url,
                    }
                    f.write(json.dumps(record) + "\n")
                    
            log.info("news_findings_saved", count=len(articles), file=self._save_to_file)
        except Exception as e:
            log.error("news_save_failed", error=str(e))

    async def get_latest(self, limit: int = 50) -> list[Article]:
        """Fetch latest news across all configured sources."""
        cache_key = f"latest_{limit}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        sources = []
        tasks = []

        if NEWS_API_KEY and NEWS_API_KEY.strip():
            tasks.append(self._fetch_newsapi(limit))
        if POLYGON_API_KEY:
            tasks.append(self._fetch_polygon(limit))

        if not tasks:
            log.warning("no_news_sources_configured")
            if self._use_fallback and self._scraper:
                log.info("using_fallback_scraper")
                return await self._get_fallback_news(limit)
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, list):
                sources.extend(r)

        seen = set()
        unique = []
        for a in sources:
            if a.id not in seen:
                seen.add(a.id)
                unique.append(a)

        unique.sort(key=lambda a: a.published_at, reverse=True)
        result = unique[:limit]
        
        if not result and self._use_fallback:
            log.info("api_returned_empty_using_fallback")
            result = await self._get_fallback_news(limit)
        
        self._set_cached(cache_key, result)
        await self._save_findings(result, "get_latest")
        return result

    async def search(self, query: str, days_back: int = 7) -> list[dict]:
        """Search for articles relevant to a query."""
        if not NEWS_API_KEY or not NEWS_API_KEY.strip():
            if self._use_fallback:
                return await self._search_fallback(query, days_back)
            return []

        cache_key = f"search_{query}_{days_back}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        await self._newsapi_limiter.acquire()
        
        from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        try:
            r = await self._request_with_retry(
                "GET",
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "from": from_date,
                    "sortBy": "relevancy",
                    "pageSize": 20,
                    "apiKey": NEWS_API_KEY,
                },
            )
            
            if r is None:
                if self._use_fallback:
                    return await self._search_fallback(query, days_back)
                return []
                
            articles = r.json().get("articles", [])
            result = [
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
            
            self._set_cached(cache_key, result)
            
            # Convert to Article objects for saving
            article_objs = [
                Article(
                    id=f"search-{hashlib.md5(a.get('url', '').encode()).hexdigest()[:12]}",
                    title=a.get("title", ""),
                    body=a.get("content") or a.get("description") or "",
                    source=a.get("source", {}).get("name", "NewsAPI"),
                    published_at=a.get("publishedAt", ""),
                    url=a.get("url", ""),
                )
                for a in articles if a.get("title")
            ]
            await self._save_findings(article_objs, f"search_{query}")
            
            return result
            
        except Exception as e:
            log.error("newsapi_search_error", error=str(e))
            return []

    async def _fetch_newsapi(self, limit: int) -> list[Article]:
        await self._newsapi_limiter.acquire()
        
        try:
            r = await self._request_with_retry(
                "GET",
                "https://newsapi.org/v2/top-headlines",
                params={
                    "language": "en",
                    "pageSize": min(limit, 100),
                    "apiKey": NEWS_API_KEY,
                    "category": "business,politics,science,technology",
                },
            )
            
            if r is None:
                return []
                
            articles = r.json().get("articles", [])
            return [
                Article(
                    id=f"newsapi-{hashlib.md5(a.get('url', '').encode()).hexdigest()[:12]}",
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
            r = await self._request_with_retry(
                "GET",
                "https://api.polygon.io/v2/reference/news",
                params={
                    "limit": min(limit, 50),
                    "sort": "published_utc",
                    "order": "desc",
                    "apiKey": POLYGON_API_KEY,
                },
            )
            
            if r is None:
                return []
                
            results = r.json().get("results", [])
            return [
                Article(
                    id=f"polygon-{a.get('id', hashlib.md5(a.get('article_url', '').encode()).hexdigest()[:12])}",
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

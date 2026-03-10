"""
Alternative News Scraper
──────────────────────
Scrapes news from various free sources when NewsAPI is rate limited.
Sources: RSS feeds, Yahoo News, Google News, Bing News
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from typing import Any
import httpx
import structlog

log = structlog.get_logger()


@dataclass
class ScrapedArticle:
    id: str
    title: str
    body: str
    source: str
    published_at: str
    url: str
    keywords: list[str] = field(default_factory=list)


class NewsScraper:
    """
    Alternative news scraper using web scraping.
    Used as fallback when NewsAPI is rate limited.
    """

    def __init__(self):
        self._client: httpx.AsyncClient | None = None
        self._cache: dict[str, tuple[list[ScrapedArticle], float]] = {}
        self._cache_ttl = 300

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            timeout=15.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    async def get_business_news(self, limit: int = 20) -> list[ScrapedArticle]:
        """Fetch business/finance news from multiple sources."""
        cache_key = f"business_{limit}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        tasks = [
            self._scrape_yahoo_finance(limit // 2),
            self._scrape_reuters(limit // 2),
            self._scrape_bloomberg(limit // 2),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        articles = []
        for r in results:
            if isinstance(r, list):
                articles.extend(r)

        articles = self._dedupe(articles)
        articles.sort(key=lambda a: a.published_at, reverse=True)

        result = articles[:limit]
        self._set_cached(cache_key, result)
        return result

    async def search_news(self, query: str, limit: int = 10) -> list[ScrapedArticle]:
        """Search for news articles matching a query."""
        cache_key = f"search_{query}_{limit}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        tasks = [
            self._search_yahoo(query, limit // 2),
            self._search_bing(query, limit // 2),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        articles = []
        for r in results:
            if isinstance(r, list):
                articles.extend(r)

        articles = self._dedupe(articles)
        result = articles[:limit]
        
        self._set_cached(cache_key, result)
        return result

    async def _scrape_yahoo_finance(self, limit: int) -> list[ScrapedArticle]:
        """Scrape Yahoo Finance."""
        try:
            r = await self._client.get(
                "https://finance.yahoo.com/news",
                params={"count": limit},
            )
            r.raise_for_status()

            articles = []
            # Yahoo Finance pattern
            pattern = r'<a[^>]+href="(/news/[^"]+)"[^>]*>\s*<([^>]*)>([^<]+)</'
            matches = re.findall(pattern, r.text, re.IGNORECASE)

            for match in matches[:limit]:
                url, _, title = match
                if title and len(title) > 10:
                    articles.append(ScrapedArticle(
                        id=f"yahoo-{hash(url)}",
                        title=title.strip(),
                        body="",
                        source="Yahoo Finance",
                        published_at=datetime.now(timezone.utc).isoformat(),
                        url=f"https://finance.yahoo.com{url}",
                    ))

            return articles
        except Exception as e:
            log.warning("yahoo_scrape_failed", error=str(e))
            return []

    async def _scrape_reuters(self, limit: int) -> list[ScrapedArticle]:
        """Scrape Reuters business news."""
        try:
            r = await self._client.get(
                "https://www.reuters.com/business",
                params={"limit": limit},
            )
            r.raise_for_status()

            articles = []
            # Reuters pattern
            pattern = r'<a[^>]+href="(https://www\.reuters\.com/[^"]+)"[^>]*>([^<]+)</a>'
            matches = re.findall(pattern, r.text, re.IGNORECASE)

            seen = set()
            for url, title in matches:
                if title.strip() and url not in seen and len(title) > 15:
                    seen.add(url)
                    articles.append(ScrapedArticle(
                        id=f"reuters-{hash(url)}",
                        title=title.strip(),
                        body="",
                        source="Reuters",
                        published_at=datetime.now(timezone.utc).isoformat(),
                        url=url,
                    ))

            return articles[:limit]
        except Exception as e:
            log.warning("reuters_scrape_failed", error=str(e))
            return []

    async def _scrape_bloomberg(self, limit: int) -> list[ScrapedArticle]:
        """Scrape Bloomberg."""
        try:
            r = await self._client.get("https://www.bloomberg.com/markets")
            r.raise_for_status()

            articles = []
            pattern = r'<a[^>]+href="(https://www\.bloomberg\.com/news/[^"]+)"[^>]*>([^<]+)</a>'
            matches = re.findall(pattern, r.text, re.IGNORECASE)

            seen = set()
            for url, title in matches:
                if title.strip() and url not in seen and len(title) > 15:
                    seen.add(url)
                    articles.append(ScrapedArticle(
                        id=f"bloomberg-{hash(url)}",
                        title=title.strip(),
                        body="",
                        source="Bloomberg",
                        published_at=datetime.now(timezone.utc).isoformat(),
                        url=url,
                    ))

            return articles[:limit]
        except Exception as e:
            log.warning("bloomberg_scrape_failed", error=str(e))
            return []

    async def _search_yahoo(self, query: str, limit: int) -> list[ScrapedArticle]:
        """Search Yahoo Finance."""
        try:
            r = await self._client.get(
                f"https://finance.yahoo.com/search",
                params={"q": query, "newsTab": "mostRelevant"},
            )
            r.raise_for_status()

            articles = []
            pattern = r'<a[^>]+href="(/news/[^"]+)"[^>]*>\s*<([^>]*)>([^<]+)</'
            matches = re.findall(pattern, r.text, re.IGNORECASE)

            for url, _, title in matches[:limit]:
                if title and len(title) > 10:
                    articles.append(ScrapedArticle(
                        id=f"yahoo-search-{hash(url)}",
                        title=title.strip(),
                        body=f"Search: {query}",
                        source="Yahoo Finance",
                        published_at=datetime.now(timezone.utc).isoformat(),
                        url=f"https://finance.yahoo.com{url}",
                    ))

            return articles
        except Exception as e:
            log.warning("yahoo_search_failed", error=str(e))
            return []

    async def _search_bing(self, query: str, limit: int) -> list[ScrapedArticle]:
        """Search Bing News."""
        try:
            r = await self._client.get(
                "https://www.bing.com/news/search",
                params={"q": query, "setlang": "en"},
            )
            r.raise_for_status()

            articles = []
            pattern = r'<a[^>]+href="(https://[^"]+)"[^>]+class="[^"]*title[^"]*"[^>]*>([^<]+)</a>'
            matches = re.findall(pattern, r.text, re.IGNORECASE)

            seen = set()
            for url, title in matches:
                if title.strip() and url not in seen and len(title) > 15:
                    seen.add(url)
                    articles.append(ScrapedArticle(
                        id=f"bing-{hash(url)}",
                        title=title.strip(),
                        body=f"Search: {query}",
                        source="Bing News",
                        published_at=datetime.now(timezone.utc).isoformat(),
                        url=url,
                    ))

            return articles[:limit]
        except Exception as e:
            log.warning("bing_search_failed", error=str(e))
            return []

    async def scrape_rss_feeds(self, feeds: list[str], limit: int = 20) -> list[ScrapedArticle]:
        """Scrape RSS feeds."""
        tasks = [self._scrape_rss(feed, limit) for feed in feeds]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        articles = []
        for r in results:
            if isinstance(r, list):
                articles.extend(r)

        articles = self._dedupe(articles)
        return articles[:limit]

    async def _scrape_rss(self, feed_url: str, limit: int) -> list[ScrapedArticle]:
        """Scrape a single RSS feed."""
        try:
            r = await self._client.get(feed_url)
            r.raise_for_status()

            articles = []
            import xml.etree.ElementTree as ET
            
            root = ET.fromstring(r.text)
            
            # Try RSS 2.0
            items = root.findall(".//item")
            if not items:
                # Try Atom
                items = root.findall(".//entry")

            for item in items[:limit]:
                title = item.findtext("title", "").strip()
                link = item.findtext("link", "")
                if not link:
                    link = item.findtext("guid", "")
                
                pub_date = item.findtext("pubDate") or item.findtext("published") or ""

                if title and link:
                    articles.append(ScrapedArticle(
                        id=f"rss-{hash(link)}",
                        title=title,
                        body=item.findtext("description", "")[:500] or "",
                        source=feed_url.split("//")[-1].split("/")[0],
                        published_at=pub_date,
                        url=link,
                    ))

            return articles
        except Exception as e:
            log.warning("rss_scrape_failed", feed=feed_url, error=str(e))
            return []

    def _get_cached(self, key: str) -> list[ScrapedArticle] | None:
        """Get cached results."""
        if key in self._cache:
            data, cached_at = self._cache[key]
            import time
            if time.time() - cached_at < self._cache_ttl:
                return data
        return None

    def _set_cached(self, key: str, data: list[ScrapedArticle]) -> None:
        """Cache results."""
        import time
        self._cache[key] = (data, time.time())

    def _dedupe(self, articles: list[ScrapedArticle]) -> list[ScrapedArticle]:
        """Deduplicate articles by title."""
        seen = set()
        unique = []
        for a in articles:
            title_lower = a.title.lower()
            if title_lower not in seen:
                seen.add(title_lower)
                unique.append(a)
        return unique


class RSSFeeds:
    """Pre-configured RSS feed URLs."""

    BUSINESS = [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://feeds.bloomberg.com/markets/news.rss",
        "https://www.alphavantage.co/query?function=NEWS_SENTIMENT&topics=financial_markets&apikey=demo",
    ]

    ECONOMY = [
        "https://feeds.reuters.com/reuters/economicNews",
    ]

    POLITICS = [
        "https://feeds.reuters.com/reuters/politicsNews",
    ]

    TECH = [
        "https://feeds.reuters.com/reuters/technologyNews",
    ]


class ScraperNewsFeed:
    """
    Drop-in replacement for NewsFeed using scraper.
    Use when NewsAPI is rate limited.
    """

    def __init__(self):
        self._scraper: NewsScraper | None = None

    async def __aenter__(self):
        self._scraper = NewsScraper()
        await self._scraper.__aenter__()
        return self

    async def __aexit__(self, *args):
        if self._scraper:
            await self._scraper.__aexit__(*args)

    async def get_latest(self, limit: int = 20) -> list:
        """Get latest business news."""
        if not self._scraper:
            return []
        
        articles = await self._scraper.get_business_news(limit)
        
        # Convert to same format as NewsFeed
        return [
            {
                "id": a.id,
                "title": a.title,
                "body": a.body,
                "source": a.source,
                "published_at": a.published_at,
                "url": a.url,
            }
            for a in articles
        ]

    async def search(self, query: str, days_back: int = 7) -> list[dict]:
        """Search news by query."""
        if not self._scraper:
            return []
        
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

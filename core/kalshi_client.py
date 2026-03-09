"""
Kalshi REST API client.
Docs: https://trading-api.kalshi.com/trade-api/v2/redoc
"""
import asyncio
from typing import Any
import httpx
from config.settings import KALSHI_API_KEY, KALSHI_API_SECRET, KALSHI_BASE_URL
import structlog

log = structlog.get_logger()


class KalshiClient:
    def __init__(self):
        self.base_url = KALSHI_BASE_URL
        self.api_key = KALSHI_API_KEY
        self.api_secret = KALSHI_API_SECRET
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=15.0)
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    def _auth_headers(self, content_type: bool = False) -> dict[str, str]:
        """Bearer token auth headers for Kalshi API."""
        h = {"Authorization": f"Bearer {self.api_key}"}
        if content_type:
            h["Content-Type"] = "application/json"
        return h

    async def _request_with_retry(self, coro_fn, max_retries: int = 5):
        """Execute a request coroutine with exponential backoff on 429/5xx."""
        delay = 1.0
        for attempt in range(max_retries):
            r = await coro_fn()
            if r.status_code == 429:
                retry_after = float(r.headers.get("Retry-After", delay))
                log.warning("kalshi_rate_limited", attempt=attempt, wait=retry_after)
                await asyncio.sleep(retry_after)
                delay = min(delay * 2, 30)
                continue
            if r.status_code >= 500:
                log.warning("kalshi_server_error", status=r.status_code, attempt=attempt)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"Kalshi request failed after {max_retries} retries")

    async def _get(self, path: str, params: dict | None = None) -> Any:
        headers = self._auth_headers()
        return await self._request_with_retry(
            lambda: self._client.get(path, headers=headers, params=params)
        )

    async def _post(self, path: str, body: dict) -> Any:
        import json
        payload = json.dumps(body)
        headers = self._auth_headers(content_type=True)
        return await self._request_with_retry(
            lambda: self._client.post(path, headers=headers, content=payload)
        )

    async def _delete(self, path: str) -> Any:
        headers = self._auth_headers()
        return await self._request_with_retry(
            lambda: self._client.delete(path, headers=headers)
        )

    # ── Markets ────────────────────────────────────────────────────────────────

    async def get_markets(
        self,
        status: str = "open",
        limit: int = 100,
        cursor: str | None = None,
        series_ticker: str | None = None,
    ) -> dict:
        params = {"status": status, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        if series_ticker:
            params["series_ticker"] = series_ticker
        return await self._get("/markets", params)

    async def get_market(self, ticker: str) -> dict:
        return await self._get(f"/markets/{ticker}")

    async def get_market_orderbook(self, ticker: str, depth: int = 10) -> dict:
        return await self._get(f"/markets/{ticker}/orderbook", {"depth": depth})

    async def get_market_history(self, ticker: str, limit: int = 100) -> dict:
        return await self._get(f"/markets/{ticker}/history", {"limit": limit})

    async def get_series(self, series_ticker: str) -> dict:
        return await self._get(f"/series/{series_ticker}")

    # ── Orders ─────────────────────────────────────────────────────────────────

    async def place_order(
        self,
        ticker: str,
        side: str,           # "yes" | "no"
        count: int,          # number of contracts
        order_type: str,     # "limit" | "market"
        yes_price: int | None = None,   # cents (0-100)
        no_price: int | None = None,
        client_order_id: str | None = None,
    ) -> dict:
        body: dict = {
            "ticker": ticker,
            "side": side,
            "count": count,
            "action": "buy",
            "type": order_type,
        }
        if yes_price is not None:
            body["yes_price"] = yes_price
        if no_price is not None:
            body["no_price"] = no_price
        if client_order_id:
            body["client_order_id"] = client_order_id
        log.info("placing_order", **body)
        return await self._post("/portfolio/orders", body)

    async def cancel_order(self, order_id: str) -> dict:
        return await self._delete(f"/portfolio/orders/{order_id}")

    async def get_orders(self, status: str | None = None) -> dict:
        params = {}
        if status:
            params["status"] = status
        return await self._get("/portfolio/orders", params)

    # ── Portfolio ──────────────────────────────────────────────────────────────

    async def get_positions(self) -> dict:
        return await self._get("/portfolio/positions")

    async def get_balance(self) -> dict:
        return await self._get("/portfolio/balance")

    async def get_fills(self, ticker: str | None = None) -> dict:
        params = {}
        if ticker:
            params["ticker"] = ticker
        return await self._get("/portfolio/fills", params)

    # ── Events ─────────────────────────────────────────────────────────────────

    async def get_events(self, status: str = "open", limit: int = 100, cursor: str | None = None) -> dict:
        params: dict = {"status": status, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        return await self._get("/events", params)

    async def get_event(self, event_ticker: str) -> dict:
        return await self._get(f"/events/{event_ticker}")

    # ── Paginated helpers ──────────────────────────────────────────────────────

    async def get_all_open_markets(self, max_event_pages: int = 3) -> list[dict]:
        """
        Fetch real single-event markets via the events endpoint.
        The /markets endpoint is dominated by low-liquidity parlay markets;
        real prediction markets come from /events.
        """
        # 1. Collect event tickers across pages
        event_tickers = []
        cursor = None
        for _ in range(max_event_pages):
            resp = await self.get_events(status="open", limit=200)
            events = resp.get("events", [])
            event_tickers.extend(e["event_ticker"] for e in events)
            cursor = resp.get("cursor")
            if not cursor:
                break
            await asyncio.sleep(0.3)

        log.info("kalshi_events_collected", count=len(event_tickers))

        # 2. Fetch event details concurrently (markets embedded in response)
        sem = asyncio.Semaphore(4)

        async def fetch_event_markets(ticker: str) -> list[dict]:
            async with sem:
                try:
                    data = await self.get_event(ticker)
                    return data.get("markets", [])
                except Exception as e:
                    log.debug("kalshi_event_fetch_skip", ticker=ticker, error=str(e))
                    return []

        results = await asyncio.gather(
            *[fetch_event_markets(t) for t in event_tickers],
            return_exceptions=True,
        )

        markets = []
        for r in results:
            if isinstance(r, list):
                markets.extend(r)

        log.info("kalshi_markets_from_events", count=len(markets))
        return markets

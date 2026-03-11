import asyncio, httpx, os
from dotenv import load_dotenv
load_dotenv()

KALSHI_KEY = os.environ["KALSHI_API_KEY"]
NEWS_KEY = os.environ["NEWS_API_KEY"]
POLYGON_KEY = os.environ["POLYGON_API_KEY"]
ANTHROPIC_KEY = os.environ["ANTHROPIC_API_KEY"]

async def test_all():
    async with httpx.AsyncClient(timeout=10) as client:

        # Anthropic
        try:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={"model": "claude-haiku-4-5", "max_tokens": 10,
                      "messages": [{"role": "user", "content": "hi"}]}
            )
            status = "OK" if r.status_code == 200 else f"FAIL {r.status_code}: {r.text[:80]}"
            print(f"Anthropic:  {status}")
        except Exception as e:
            print(f"Anthropic:  ERROR {e}")

        # NewsAPI
        try:
            r = await client.get(
                "https://newsapi.org/v2/top-headlines",
                params={"country": "us", "pageSize": 1, "apiKey": NEWS_KEY}
            )
            status = "OK" if r.status_code == 200 else f"FAIL {r.status_code}: {r.json().get('message','')}"
            print(f"NewsAPI:    {status}")
        except Exception as e:
            print(f"NewsAPI:    ERROR {e}")

        # Polygon
        try:
            r = await client.get(
                "https://api.polygon.io/v2/reference/news",
                params={"limit": 1, "apiKey": POLYGON_KEY}
            )
            status = "OK" if r.status_code == 200 else f"FAIL {r.status_code}: {r.json().get('error','')}"
            print(f"Polygon:    {status}")
        except Exception as e:
            print(f"Polygon:    ERROR {e}")

        # Kalshi — try public endpoint, then authenticated
        try:
            r = await client.get(
                "https://api.elections.kalshi.com/trade-api/v2/markets",
                params={"limit": 1}
            )
            if r.status_code == 200:
                print("Kalshi:     OK (public markets endpoint)")
            else:
                print(f"Kalshi public: FAIL {r.status_code}: {r.text[:80]}")

            r2 = await client.get(
                "https://api.elections.kalshi.com/trade-api/v2/portfolio/balance",
                headers={"Authorization": f"Bearer {KALSHI_KEY}"}
            )
            status = "OK (authenticated)" if r2.status_code == 200 else f"FAIL {r2.status_code}: {r2.text[:80]}"
            print(f"Kalshi auth: {status}")
        except Exception as e:
            print(f"Kalshi:     ERROR {e}")

asyncio.run(test_all())

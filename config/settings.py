import os
from dotenv import load_dotenv

load_dotenv()

# Kalshi
KALSHI_API_KEY = os.environ["KALSHI_API_KEY"]
KALSHI_API_SECRET = os.environ.get("KALSHI_API_SECRET", "")
KALSHI_BASE_URL = os.environ.get("KALSHI_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2")

# News
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")

# Risk limits
# Total account capital — drives all position sizing
TOTAL_CAPITAL_USD = float(os.environ.get("TOTAL_CAPITAL_USD", "1000"))
# Maximum USD at risk in any single market position
MAX_POSITION_SIZE_USD = float(os.environ.get("MAX_POSITION_SIZE_USD", "50"))
# Maximum fraction of capital deployed across all open positions simultaneously
MAX_PORTFOLIO_HEAT_PCT = float(os.environ.get("MAX_PORTFOLIO_HEAT_PCT", "0.20"))
# Maximum fraction of capital in any single market (concentration cap)
MAX_SINGLE_MARKET_PCT = float(os.environ.get("MAX_SINGLE_MARKET_PCT", "0.05"))
# Stop all trading for the day if realized losses exceed this amount
DAILY_LOSS_LIMIT_USD = float(os.environ.get("DAILY_LOSS_LIMIT_USD", "50"))

# AI cost controls
# Approximate USD cap on AI API calls per day (based on token estimates)
DAILY_AI_COST_LIMIT_USD = float(os.environ.get("DAILY_AI_COST_LIMIT_USD", "10.0"))

# Trading mode - set to False to enable live trading
DRY_RUN = os.environ.get("DRY_RUN", "true").lower() == "false"

# DB
DB_PATH = os.environ.get("DB_PATH", "./data/trading.db")

# Agent models
ANALYST_MODEL = "claude-opus-4-6"       # Deep reasoning — probability analysis, strategy
FAST_MODEL = "claude-haiku-4-5"         # High-throughput — news triage, quick classification

# Scanning
SCAN_INTERVAL_SECONDS = 60
MIN_EDGE_THRESHOLD = 0.04               # Minimum probability edge to consider trading (4%)
MIN_LIQUIDITY_USD = 500                 # Minimum open interest to trade into (higher = fewer markets = faster)
MIN_SPREAD_PCT = 0.02                   # Skip markets with spread > 2% (bad liquidity)

# Kalshi AI Hedge Fund

An autonomous prediction market trading bot that uses AI agents to find, analyze, and trade mispriced markets on [Kalshi](https://kalshi.com). Includes a Streamlit web dashboard for monitoring and control.

## How It Works

The bot runs a continuous loop with specialized AI agents, each handling a different part of the trading pipeline:

1. **Market Scanner** — Fetches all open Kalshi markets, filters by liquidity, and batch-scores them with AI for trading potential (runs on Grok for speed/cost)
2. **Probability Analyst** — Estimates the true probability of each candidate market using news, price history, and related markets (runs on Claude Opus for reasoning quality)
3. **Risk Manager** — Sizes positions using Kelly criterion, enforces portfolio limits, and runs AI review on large trades (rules + Claude Opus)
4. **Trade Executor** — Places limit orders with smart execution: checks orderbook depth, splits large orders, handles partial fills, has circuit breakers
5. **Portfolio Monitor** — Watches open positions and recommends hold/exit/reduce actions (runs on Grok)
6. **News Analyzer** — Monitors news feeds in real-time, triages headlines for market relevance, triggers re-analysis when breaking news affects open positions (triage on Grok, deep analysis on Claude Opus)
7. **Market Research** — Hourly broad market analysis looking for cross-market trends and anomalies (runs on Grok)

### Cost Optimization

The bot splits AI workload across providers to minimize costs:

- **High-volume/simple tasks** (scanner, news triage, portfolio monitor) → xAI Grok (~$0.01-0.02/call)
- **Complex reasoning** (probability estimation, risk review) → Claude Opus (~$0.08-0.12/call)
- **Result caching** on the scanner (5-min TTL) cuts repeat calls 50-80%
- **Batch processing** groups up to 10 news articles or markets per AI call
- **Prompt trimming** filters API responses to only relevant fields before sending to AI
- **Centralized cost tracking** across all agents with configurable daily limits

## Screenshots

The web dashboard shows:
- Live trading logs
- Scan results with colored edge/confidence/status indicators
- Open positions and P&L
- Trade history
- Configuration sidebar (API keys, risk limits, category filters)

## Setup

### Prerequisites

- Python 3.12+
- A [Kalshi](https://kalshi.com) account with API access
- An [xAI](https://console.x.ai) API key (required — used for most AI calls)
- An [Anthropic](https://console.anthropic.com) API key (recommended — used for probability analysis)

### Installation

```bash
git clone https://github.com/bmccueny/kalshi-hedge-fund.git
cd kalshi-hedge-fund
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
# Required
KALSHI_API_KEY=your-kalshi-api-key-uuid
KALSHI_RSA_KEY_PATH=./kalshi_rsa_key.pem
XAI_API_KEY=xai-...

# Recommended (falls back to Grok if not set)
ANTHROPIC_API_KEY=sk-ant-...

# Optional
NEWS_API_KEY=              # newsapi.org
POLYGON_API_KEY=           # polygon.io
```

#### Kalshi RSA Key

Kalshi uses RSA key signing for API authentication:

1. Go to [kalshi.com](https://kalshi.com) → Account → API Keys
2. Generate a new API key — this creates an RSA keypair
3. Save the private key as `kalshi_rsa_key.pem` in the project root
4. Copy the API key UUID into your `.env` as `KALSHI_API_KEY`

### Risk Configuration

All risk parameters are configurable in `.env`:

```env
TOTAL_CAPITAL_USD=200          # Total capital to trade with
MAX_POSITION_SIZE_USD=100      # Max USD in any single trade
MAX_PORTFOLIO_HEAT_PCT=0.20    # Max 20% of capital at risk simultaneously
MAX_SINGLE_MARKET_PCT=0.05     # Max 5% of capital in any one market
DAILY_LOSS_LIMIT_USD=50        # Stop trading after $50 daily loss
MIN_EDGE_THRESHOLD=0.04        # Minimum 4% probability edge to trade
DRY_RUN=true                   # Log trades without placing them
```

## Usage

### Web Dashboard (recommended)

```bash
streamlit run gui/app.py
```

Opens at `http://localhost:8501`. The dashboard lets you:
- **Start/Stop Trading** — runs the full autonomous trading loop
- **Scan Markets** — one-shot scan to see what the bot would find
- **Configure** — API keys, risk limits, and market categories via the sidebar
- **Filter by category** — Weather, Sports, Crypto, Tech, Politics, Economics, etc.

### CLI

```bash
# One-shot market scan (prints opportunities table)
python main.py --scan

# Start trading bot (dry run — no real trades)
python main.py

# Start trading bot (live — real money, requires confirmation)
python main.py --live

# Terminal dashboard
python main.py --dashboard
```

## Project Structure

```
kalshi-hedge-fund/
├── agents/                    # AI agents
│   ├── base_agent.py          # Base class with provider routing + cost tracking
│   ├── market_scanner.py      # Batch market scoring with caching
│   ├── probability_analyst.py # Calibrated probability estimation
│   ├── risk_manager.py        # Position sizing + trade approval
│   ├── trade_executor.py      # Smart order execution
│   ├── portfolio_monitor.py   # Open position monitoring
│   ├── news_analyzer.py       # News triage + deep analysis
│   ├── market_research.py     # Hourly market research
│   ├── arbitrage_agent.py     # Cross-market arbitrage detection
│   ├── momentum_agent.py      # Price momentum strategies
│   └── event_driven_agent.py  # Event-driven trading signals
├── core/
│   ├── orchestrator.py        # Main trading loop coordinator
│   ├── kalshi_client.py       # Kalshi API client with RSA signing
│   ├── database.py            # SQLite persistence (signals, trades, P&L)
│   ├── data_pipeline.py       # Data collection and processing
│   ├── models/                # Trend detection + anomaly detection
│   ├── backtest.py            # Backtesting framework
│   └── optimize.py            # Strategy optimization
├── config/
│   └── settings.py            # All configuration from .env
├── data/
│   ├── news_feed.py           # Async news feed aggregator
│   └── news_scraper.py        # Web scraping for news
├── gui/
│   └── app.py                 # Streamlit web dashboard
├── risk/
│   └── kelly.py               # Kelly criterion position sizing
├── monitoring/
│   └── dashboard.py           # Terminal dashboard
├── main.py                    # CLI entry point
├── .env.example               # Configuration template
└── requirements.txt           # Python dependencies
```

## AI Provider Fallback Chain

Each agent has a multi-tier fallback to ensure the bot never stops due to a single provider outage:

**Claude-primary agents** (ProbabilityAnalyst, RiskManager):
```
Claude Code CLI → Anthropic API → xAI Grok → Heuristic fallback
```

**Grok-primary agents** (Scanner, News triage, Portfolio monitor):
```
xAI Grok → Anthropic API → Heuristic fallback
```

## Estimated Costs

| Component | Per cycle (~60s) | Per hour |
|-----------|-----------------|----------|
| Market Scanner (Grok) | $0.06-0.20 | $3.60-12.00 |
| Probability Analyst (Opus) | $0.40-0.60 | $24.00-36.00 |
| News triage (Grok) | — | $0.60 |
| Portfolio monitor (Grok) | — | $0.45 |
| **Total** | **~$0.50-0.80** | **~$29-49** |

The `DAILY_AI_COST_LIMIT_USD` setting (default $10) stops the Probability Analyst when the daily AI budget is exhausted. Adjust based on your budget.

## License

Private repository. All rights reserved.

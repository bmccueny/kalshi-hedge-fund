"""
SQLite persistence layer for positions, trades, and signals.
"""
import json
from datetime import datetime
import aiosqlite
from config.settings import DB_PATH
import structlog

log = structlog.get_logger()

SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,          -- 'yes' | 'no'
    contracts INTEGER NOT NULL,
    avg_price_cents INTEGER NOT NULL,
    current_price_cents INTEGER,
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    pnl_usd REAL,
    status TEXT DEFAULT 'open'   -- 'open' | 'closed'
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT UNIQUE,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    contracts INTEGER NOT NULL,
    price_cents INTEGER NOT NULL,
    action TEXT NOT NULL,        -- 'open' | 'close'
    agent TEXT,
    reason TEXT,
    executed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    agent TEXT NOT NULL,
    true_prob REAL NOT NULL,     -- Agent's estimated probability
    market_prob REAL NOT NULL,   -- Market-implied probability
    edge REAL NOT NULL,          -- true_prob - market_prob
    confidence REAL NOT NULL,    -- 0.0-1.0
    rationale TEXT,
    acted_on INTEGER DEFAULT 0,  -- 1 if trade was placed
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_pnl (
    date TEXT PRIMARY KEY,
    realized_pnl_usd REAL DEFAULT 0,
    unrealized_pnl_usd REAL DEFAULT 0,
    num_trades INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_positions_ticker ON positions(ticker);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker);
"""


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(SCHEMA)
        await db.commit()
    log.info("db_initialized", path=DB_PATH)


async def save_signal(
    ticker: str,
    agent: str,
    true_prob: float,
    market_prob: float,
    edge: float,
    confidence: float,
    rationale: str,
) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """INSERT INTO signals (ticker, agent, true_prob, market_prob, edge, confidence, rationale, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (ticker, agent, true_prob, market_prob, edge, confidence, rationale,
             datetime.utcnow().isoformat()),
        )
        await db.commit()
        return cursor.lastrowid


async def save_trade(
    order_id: str,
    ticker: str,
    side: str,
    contracts: int,
    price_cents: int,
    action: str,
    agent: str,
    reason: str,
) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT OR IGNORE INTO trades
               (order_id, ticker, side, contracts, price_cents, action, agent, reason, executed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (order_id, ticker, side, contracts, price_cents, action, agent, reason,
             datetime.utcnow().isoformat()),
        )
        await db.commit()


async def get_open_positions() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM positions WHERE status = 'open' ORDER BY opened_at DESC"
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def get_daily_pnl(date: str | None = None) -> dict:
    if date is None:
        date = datetime.utcnow().strftime("%Y-%m-%d")
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM daily_pnl WHERE date = ?", (date,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else {"date": date, "realized_pnl_usd": 0, "num_trades": 0}


async def reconcile(kalshi_positions: list[dict], kalshi_balance: float) -> dict:
    """
    Sync local positions table to match Kalshi's actual state.
    Called at startup so local DB reflects reality before trading begins.
    """
    local_positions = await get_open_positions()
    local_by_ticker = {p["ticker"]: p for p in local_positions}
    kalshi_by_ticker = {p["ticker"]: p for p in kalshi_positions if p.get("ticker")}

    discrepancies = []

    # Positions on Kalshi that are missing locally → insert them
    for ticker, kpos in kalshi_by_ticker.items():
        if ticker not in local_by_ticker:
            log.warning("reconcile_missing_local", ticker=ticker,
                        kalshi_contracts=abs(kpos.get("position", 0)))
            discrepancies.append({"type": "missing_local", "ticker": ticker})
            net = kpos.get("position", 0)
            side = "yes" if net > 0 else "no"
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    """INSERT INTO positions
                       (ticker, side, contracts, avg_price_cents, opened_at, status)
                       VALUES (?, ?, ?, ?, ?, 'open')""",
                    (ticker, side, abs(net),
                     kpos.get("avg_price", 50),
                     datetime.utcnow().isoformat()),
                )
                await db.commit()

    # Positions open locally but absent from Kalshi → mark closed
    for ticker, lpos in local_by_ticker.items():
        if ticker not in kalshi_by_ticker:
            log.warning("reconcile_stale_local", ticker=ticker)
            discrepancies.append({"type": "stale_local", "ticker": ticker})
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    "UPDATE positions SET status='closed', closed_at=? WHERE ticker=? AND status='open'",
                    (datetime.utcnow().isoformat(), ticker),
                )
                await db.commit()

    log.info(
        "reconcile_complete",
        discrepancies=len(discrepancies),
        kalshi_positions=len(kalshi_by_ticker),
        local_positions=len(local_by_ticker),
        kalshi_balance_cents=kalshi_balance,
    )
    return {
        "discrepancies": discrepancies,
        "kalshi_balance": kalshi_balance,
        "local_count": len(local_by_ticker),
        "kalshi_count": len(kalshi_by_ticker),
    }


async def update_daily_pnl(realized_delta: float, num_trades_delta: int = 1) -> None:
    date = datetime.utcnow().strftime("%Y-%m-%d")
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO daily_pnl (date, realized_pnl_usd, num_trades)
               VALUES (?, ?, ?)
               ON CONFLICT(date) DO UPDATE SET
                 realized_pnl_usd = realized_pnl_usd + excluded.realized_pnl_usd,
                 num_trades = num_trades + excluded.num_trades""",
            (date, realized_delta, num_trades_delta),
        )
        await db.commit()

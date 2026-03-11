#!/usr/bin/env python3
"""
Kalshi AI Hedge Fund - Web Interface
Run with: streamlit run gui/app.py
"""
import os
import re
import sys
import json
import asyncio
import subprocess
import threading
import queue
from datetime import datetime

_ANSI_ESCAPE = re.compile(r'\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

import streamlit as st
import pandas as pd
import dotenv

# Thread-safe queue for background threads to write log lines into.
# Must be cache_resource so Streamlit script reruns don't create a new empty
# queue and lose items the background thread already put into the old one.
@st.cache_resource
def _get_log_queue() -> queue.Queue:
    return queue.Queue()

_log_queue = _get_log_queue()

# Load .env
dotenv.load_dotenv()

# Page config
st.set_page_config(
    page_title="Kalshi AI Hedge Fund",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { background-color: #0e1117; }
    .status-running { color: #22c55e; font-weight: bold; }
    .status-stopped { color: #ef4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

ENV_FILE = ".env"
CONFIG_FILE = "gui_config.json"

def load_env():
    env = {}
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    env[key] = val
    return env

def save_env(env):
    with open(ENV_FILE, "w") as f:
        for key, val in env.items():
            f.write(f"{key}={val}\n")

# ── Kalshi data fetching ────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def fetch_kalshi_balance():
    """Fetch real account balance from Kalshi. Cached 30s."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from core.kalshi_client import KalshiClient
        async def _get():
            async with KalshiClient() as k:
                return await k.get_balance()
        data = asyncio.run(_get())
        return data.get("balance", 0) / 100
    except Exception as e:
        return None

@st.cache_data(ttl=30)
def fetch_kalshi_positions():
    """Fetch open positions from Kalshi. Cached 30s."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from core.kalshi_client import KalshiClient
        async def _get():
            async with KalshiClient() as k:
                return await k.get_positions()
        data = asyncio.run(_get())
        return data.get("market_positions", data.get("positions", []))
    except Exception as e:
        return None

@st.cache_data(ttl=60)
def fetch_kalshi_fills():
    """Fetch trade fill history from Kalshi. Cached 60s."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from core.kalshi_client import KalshiClient
        async def _get():
            async with KalshiClient() as k:
                return await k.get_fills()
        data = asyncio.run(_get())
        return data.get("fills", [])
    except Exception as e:
        return []


def run_market_scan():
    """Run market scan in-process and return structured results."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from core.kalshi_client import KalshiClient
        from agents.market_scanner import MarketScannerAgent
        from agents.probability_analyst import ProbabilityAnalystAgent
        from data.news_feed import NewsFeed

        async def _scan():
            async with KalshiClient() as kalshi, NewsFeed() as news:
                scanner = MarketScannerAgent(kalshi)
                analyst = ProbabilityAnalystAgent(kalshi, news)

                _log_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] Scanning markets...")
                candidates = await scanner.scan()
                top = candidates[:10]
                _log_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] Found {len(candidates)} candidates — analysing top {len(top)}...")

                estimates = await asyncio.gather(
                    *[analyst.analyze(m) for m in top],
                    return_exceptions=True,
                )

                results = []
                for est in estimates:
                    if isinstance(est, Exception) or est is None:
                        continue
                    results.append({
                        "ticker": est.ticker,
                        "title": est.market_title,
                        "market_pct": est.market_price_pct,
                        "true_pct": est.true_prob_pct,
                        "edge_pct": est.edge_pct,
                        "confidence": est.confidence,
                        "side": est.recommended_side,
                        "risks": est.key_risks,
                        "rationale": est.rationale,
                    })

                _log_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] Scan complete — {len(results)} estimates produced")
                return results

        return asyncio.run(_scan())
    except Exception as e:
        _log_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] Scan error: {e}")
        return None

# ── Session state ───────────────────────────────────────────────────────────

if "trading_process" not in st.session_state:
    st.session_state.trading_process = None
if "scan_results" not in st.session_state:
    st.session_state.scan_results = None
if "scan_running" not in st.session_state:
    st.session_state.scan_running = False
if "logs" not in st.session_state:
    st.session_state.logs = []

def start_trading(dry_run: bool):
    if st.session_state.trading_process is None:
        cmd = [sys.executable, "main.py"]
        if not dry_run:
            cmd.append("--live")
        env_unbuf = {**os.environ, "PYTHONUNBUFFERED": "1", "NO_COLOR": "1"}
        st.session_state.trading_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env_unbuf,
        )
        proc = st.session_state.trading_process
        def read_output(p=proc):
            for line in p.stdout:
                clean = _ANSI_ESCAPE.sub('', line.rstrip())
                if clean:
                    _log_queue.put(f"[{datetime.now().strftime('%H:%M:%S')}] {clean}")
        threading.Thread(target=read_output, daemon=True).start()

def stop_trading():
    if st.session_state.trading_process:
        st.session_state.trading_process.terminate()
        st.session_state.trading_process = None

def is_trading():
    return st.session_state.trading_process is not None and st.session_state.trading_process.poll() is None

def is_scanning():
    return st.session_state.scan_running

# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuration")
    env = load_env()

    st.subheader("API Keys")
    kalshi_key = st.text_input("Kalshi API Key", value=env.get("KALSHI_API_KEY", ""), type="password")
    anthropic_key = st.text_input("Anthropic API Key", value=env.get("ANTHROPIC_API_KEY", ""), type="password")

    if st.button("💾 Save API Keys", type="primary"):
        env["KALSHI_API_KEY"] = kalshi_key
        if anthropic_key:
            env["ANTHROPIC_API_KEY"] = anthropic_key
        save_env(env)
        fetch_kalshi_balance.clear()
        fetch_kalshi_positions.clear()
        st.success("Saved!")

    st.divider()
    st.subheader("Trading Settings")

    total_capital = st.number_input(
        "Total Capital ($)",
        value=float(env.get("TOTAL_CAPITAL_USD", "200")),
        min_value=0.0,
        step=50.0
    )
    max_position_pct = st.slider(
        "Max Position Size (%)",
        value=float(env.get("MAX_SINGLE_MARKET_PCT", "0.05")) * 100,
        min_value=1.0, max_value=25.0, step=1.0
    ) / 100
    min_edge = st.slider(
        "Minimum Edge Threshold (%)",
        value=float(env.get("MIN_EDGE_THRESHOLD", "0.04")) * 100,
        min_value=1.0, max_value=20.0, step=1.0
    ) / 100
    dry_run_toggle = st.toggle("Dry Run Mode", value=env.get("DRY_RUN", "true") == "true")

    if st.button("💾 Save Settings", type="primary"):
        env["TOTAL_CAPITAL_USD"] = str(int(total_capital))
        env["MAX_SINGLE_MARKET_PCT"] = str(max_position_pct)
        env["MIN_EDGE_THRESHOLD"] = str(min_edge)
        env["DRY_RUN"] = "true" if dry_run_toggle else "false"
        save_env(env)
        st.success("Saved!")

    st.divider()
    if st.button("🔄 Refresh Data", use_container_width=True):
        fetch_kalshi_balance.clear()
        fetch_kalshi_positions.clear()
        fetch_kalshi_fills.clear()
        st.rerun()

# ── Main ────────────────────────────────────────────────────────────────────

st.title("📈 Kalshi AI Hedge Fund")

# Fetch live data
balance = fetch_kalshi_balance()
positions = fetch_kalshi_positions()
fills = fetch_kalshi_fills()

# ── Status Banner ───────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)

with col1:
    status = "🟢 Running" if is_trading() else "🔴 Stopped"
    st.metric("Bot Status", status)

with col2:
    if balance is None:
        st.metric("Kalshi Balance", "⚠️ Auth Error")
    else:
        configured = float(env.get("TOTAL_CAPITAL_USD", "200"))
        delta = f"Config: ${configured:,.0f}" if abs(balance - configured) > 0.5 else None
        st.metric("Kalshi Balance", f"${balance:,.2f}", delta=delta, delta_color="off")

with col3:
    n_positions = len(positions) if positions is not None else "—"
    st.metric("Open Positions", n_positions if positions is not None else "⚠️ Error")

with col4:
    dry_run_display = "Yes" if env.get("DRY_RUN", "true") == "true" else "No (LIVE)"
    st.metric("Dry Run", dry_run_display)

st.divider()

# ── Control Panel ───────────────────────────────────────────────────────────

st.subheader("🎮 Control Panel")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("▶️ Start Trading", type="primary", disabled=is_trading()):
        is_dry = env.get("DRY_RUN", "true") == "true"
        start_trading(is_dry)
        st.rerun()

with col2:
    if st.button("⏹️ Stop Trading", type="secondary", disabled=not is_trading()):
        stop_trading()
        st.rerun()

with col3:
    if st.button("🔄 Scan Markets", type="secondary", disabled=is_scanning()):
        st.session_state.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] --- Starting market scan ---")
        st.session_state.scan_running = True
        with st.spinner("Scanning markets & running AI analysis..."):
            results = run_market_scan()
        st.session_state.scan_running = False
        if results is not None:
            st.session_state.scan_results = results
        st.rerun()

# ── Live Logs ───────────────────────────────────────────────────────────────

st.divider()
st.subheader("📋 Live Logs")
if st.button("🗑️ Clear Logs"):
    st.session_state.logs = []

@st.fragment(run_every=0.5)
def live_logs_fragment():
    # Drain queue on every fragment refresh
    drained = False
    while not _log_queue.empty():
        st.session_state.logs.append(_log_queue.get_nowait())
        drained = True

    log_container = st.container(height=300)
    with log_container:
        for line in st.session_state.logs[-100:]:
            st.text(line)

    active = is_trading() or is_scanning()
    status_label = "🟢 Bot running" if is_trading() else ("🔵 Scanning..." if is_scanning() else "")
    if status_label:
        st.caption(status_label)

live_logs_fragment()

# ── Scan Results ────────────────────────────────────────────────────────────

st.divider()
st.subheader("🔍 Scan Results")

if st.session_state.scan_results is None:
    st.info("No scan results yet — click **Scan Markets** above to find opportunities.")
elif len(st.session_state.scan_results) == 0:
    st.warning("Scan completed but found no tradeable opportunities.")
else:
    scan_data = st.session_state.scan_results

    # Summary metrics row
    sc1, sc2, sc3, sc4 = st.columns(4)
    avg_edge = sum(abs(r["edge_pct"]) for r in scan_data) / len(scan_data)
    avg_conf = sum(r["confidence"] for r in scan_data) / len(scan_data)
    buy_yes = sum(1 for r in scan_data if r.get("side") == "yes")
    buy_no = sum(1 for r in scan_data if r.get("side") == "no")
    with sc1:
        st.metric("Opportunities", len(scan_data))
    with sc2:
        st.metric("Avg Edge", f"{avg_edge:+.1f}%")
    with sc3:
        st.metric("Avg Confidence", f"{avg_conf:.0%}")
    with sc4:
        st.metric("YES / NO", f"{buy_yes} / {buy_no}")

    # Build table rows
    scan_rows = []
    for r in scan_data:
        side = (r.get("side") or "—").upper()
        scan_rows.append({
            "Ticker": r["ticker"],
            "Title": r["title"][:45],
            "Side": side,
            "Market %": f"{r['market_pct']:.1f}%",
            "True %": f"{r['true_pct']:.1f}%",
            "Edge": f"{r['edge_pct']:+.1f}%",
            "Confidence": f"{r['confidence']:.0%}",
        })

    scan_df = pd.DataFrame(scan_rows)

    def color_edge(val):
        """Green for positive edge, red for negative."""
        try:
            num = float(val.replace("%", "").replace("+", ""))
        except (ValueError, AttributeError):
            return ""
        if num > 5:
            return "color: #22c55e; font-weight: bold"
        elif num > 0:
            return "color: #4ade80"
        elif num < -5:
            return "color: #ef4444; font-weight: bold"
        elif num < 0:
            return "color: #f87171"
        return ""

    def color_confidence(val):
        """Green gradient based on confidence level."""
        try:
            num = float(val.replace("%", "")) / 100
        except (ValueError, AttributeError):
            return ""
        if num >= 0.7:
            return "color: #22c55e; font-weight: bold"
        elif num >= 0.5:
            return "color: #facc15"
        else:
            return "color: #f87171"

    def color_side(val):
        if val == "YES":
            return "color: #22c55e; font-weight: bold"
        elif val == "NO":
            return "color: #ef4444; font-weight: bold"
        return "color: #6b7280"

    styled = scan_df.style.map(color_edge, subset=["Edge"]) \
                          .map(color_confidence, subset=["Confidence"]) \
                          .map(color_side, subset=["Side"])

    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Expandable rationale details
    with st.expander("📝 Detailed Analysis", expanded=False):
        for r in scan_data:
            side_label = (r.get("side") or "none").upper()
            edge_sign = "+" if r["edge_pct"] > 0 else ""
            st.markdown(f"**{r['ticker']}** — {r['title']}")
            st.markdown(
                f"&nbsp;&nbsp;&nbsp;&nbsp;Edge: `{edge_sign}{r['edge_pct']:.1f}%` · "
                f"Confidence: `{r['confidence']:.0%}` · "
                f"Side: `{side_label}`"
            )
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*{r.get('rationale', 'N/A')[:300]}*")
            risks = r.get("risks", [])
            if risks:
                st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;Risks: " + ", ".join(f"`{rk}`" for rk in risks[:3]))
            st.markdown("---")

# ── Open Positions ──────────────────────────────────────────────────────────

st.divider()
st.subheader("📊 Open Positions")

if positions is None:
    st.error("Could not fetch positions — check Kalshi API key")
elif len(positions) == 0:
    st.info("No open positions")
else:
    rows = []
    total_value = 0
    total_pl = 0
    for p in positions:
        ticker = p.get("ticker", "")
        yes_count = p.get("position", p.get("yes_position", 0))
        no_count = p.get("no_position", 0)
        side = "YES" if yes_count > 0 else "NO"
        contracts = yes_count if yes_count > 0 else no_count
        # Market value in cents
        market_val = p.get("market_exposure", p.get("value", 0)) / 100
        resting_orders_val = p.get("resting_orders_count", 0)
        realized_pnl = p.get("realized_pnl", 0) / 100
        unrealized_pnl = p.get("unrealized_pnl", 0) / 100
        total_pnl = realized_pnl + unrealized_pnl
        total_value += market_val
        total_pl += total_pnl
        rows.append({
            "Ticker": ticker,
            "Side": side,
            "Contracts": contracts,
            "Market Value": f"${market_val:.2f}",
            "Realized P/L": f"${realized_pnl:+.2f}",
            "Unrealized P/L": f"${unrealized_pnl:+.2f}",
            "Total P/L": f"${total_pnl:+.2f}",
        })

    df = pd.DataFrame(rows)

    def color_pl(val):
        s = str(val)
        if s.startswith("+") or (s.startswith("$+") if "$" in s else False):
            return "color: #22c55e"
        elif "-" in s:
            return "color: #ef4444"
        return ""

    st.dataframe(
        df.style.map(color_pl, subset=["Realized P/L", "Unrealized P/L", "Total P/L"]),
        use_container_width=True,
        hide_index=True,
    )

# ── Performance ─────────────────────────────────────────────────────────────

st.divider()
st.subheader("📈 Performance")

col1, col2, col3, col4 = st.columns(4)

# Compute from fills
today = datetime.utcnow().date()
today_fills = [f for f in fills if f.get("created_time", "")[:10] == str(today)] if fills else []
today_pl = sum(f.get("profit_loss", 0) for f in today_fills) / 100 if today_fills else 0
total_fills = len(fills)

# Compute from positions
n_open = len(positions) if positions else 0
total_unrealized = sum(p.get("unrealized_pnl", 0) for p in (positions or [])) / 100

with col1:
    pl_str = f"${today_pl:+.2f}" if today_fills else "—"
    st.metric("Today's P/L", pl_str)
with col2:
    st.metric("Total Trades", total_fills if fills else "—")
with col3:
    st.metric("Unrealized P/L", f"${total_unrealized:+.2f}" if positions else "—")
with col4:
    st.metric("Active Positions", n_open)

st.divider()
st.caption("Kalshi AI Hedge Fund | Use at your own risk | Data refreshes every 30s")

#!/usr/bin/env python3
"""
Kalshi AI Hedge Fund - Web Interface
Run with: streamlit run gui/app.py
"""
import os
import sys
import json
import subprocess
import threading
from datetime import datetime

import streamlit as st
import pandas as pd
import dotenv

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
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #334155;
    }
    .status-running {
        color: #22c55e;
        font-weight: bold;
    }
    .status-stopped {
        color: #ef4444;
        font-weight: bold;
    }
    .trade-buy {
        color: #22c55e;
    }
    .trade-sell {
        color: #ef4444;
    }
</style>
""", unsafe_allow_html=True)

# Config file path
CONFIG_FILE = "gui_config.json"
ENV_FILE = ".env"

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}

def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

def load_env():
    env = {}
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line and "=" in line:
                    key, val = line.split("=", 1)
                    env[key] = val
    return env

def save_env(env):
    with open(ENV_FILE, "w") as f:
        for key, val in env.items():
            f.write(f"{key}={val}\n")

# Session state
if "trading_process" not in st.session_state:
    st.session_state.trading_process = None
if "logs" not in st.session_state:
    st.session_state.logs = []

def start_trading(dry_run: bool):
    if st.session_state.trading_process is None:
        cmd = [sys.executable, "main.py"]
        if not dry_run:
            cmd.append("--live")
        
        st.session_state.trading_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        def read_output():
            for line in st.session_state.trading_process.stdout:
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.logs.append(f"[{timestamp}] {line}")
        
        thread = threading.Thread(target=read_output, daemon=True)
        thread.start()

def stop_trading():
    if st.session_state.trading_process:
        st.session_state.trading_process.terminate()
        st.session_state.trading_process = None

def is_trading():
    return st.session_state.trading_process is not None and st.session_state.trading_process.poll() is None

# Sidebar - Configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # API Keys Section
    st.subheader("API Keys")
    
    config = load_config()
    env = load_env()
    
    kalshi_key = st.text_input(
        "Kalshi API Key",
        value=env.get("KALSHI_API_KEY", ""),
        type="password",
        help="Get from https://elections.kalshi.com/developers"
    )
    
    anthropic_key = st.text_input(
        "Anthropic API Key",
        value=env.get("ANTHROPIC_API_KEY", ""),
        type="password",
        help="For AI analysis (recommended)"
    )
    
    gemini_key = st.text_input(
        "Gemini API Key",
        value=env.get("GEMINI_API_KEY", ""),
        type="password",
        help="Fallback AI provider"
    )
    
    if st.button("💾 Save API Keys", type="primary"):
        env["KALSHI_API_KEY"] = kalshi_key
        if anthropic_key:
            env["ANTHROPIC_API_KEY"] = anthropic_key
        if gemini_key:
            env["GEMINI_API_KEY"] = gemini_key
        save_env(env)
        st.success("API keys saved!")
    
    st.divider()
    
    # Trading Settings
    st.subheader("Trading Settings")
    
    total_capital = st.number_input(
        "Total Capital ($)",
        value=float(env.get("TOTAL_CAPITAL_USD", "200")),
        min_value=100.0,
        step=100.0
    )
    
    max_position_pct = st.slider(
        "Max Position Size (%)",
        value=float(env.get("MAX_SINGLE_MARKET_PCT", "0.05")) * 100,
        min_value=1.0,
        max_value=25.0,
        step=1.0
    ) / 100
    
    min_edge = st.slider(
        "Minimum Edge Threshold (%)",
        value=float(env.get("MIN_EDGE_THRESHOLD", "0.04")) * 100,
        min_value=1.0,
        max_value=20.0,
        step=1.0
    ) / 100
    
    if st.button("💾 Save Settings", type="primary"):
        env["TOTAL_CAPITAL_USD"] = str(int(total_capital))
        env["MAX_SINGLE_MARKET_PCT"] = str(max_position_pct)
        env["MIN_EDGE_THRESHOLD"] = str(min_edge)
        save_env(env)
        st.success("Settings saved!")
    
    st.divider()
    
    # AI Provider
    st.subheader("AI Provider")
    ai_provider = st.selectbox(
        "Select AI Provider",
        ["anthropic", "gemini"],
        index=0 if env.get("AI_PROVIDER") == "anthropic" else 1
    )
    
    if st.button("💾 Save AI Provider", type="primary"):
        env["AI_PROVIDER"] = ai_provider
        save_env(env)
        st.success("AI provider saved!")

# Main Content
st.title("📈 Kalshi AI Hedge Fund")

# Status Banner
col1, col2, col3 = st.columns(3)

with col1:
    status = "🟢 Running" if is_trading() else "🔴 Stopped"
    st.metric("Status", status)

with col2:
    st.metric("Capital", f"${total_capital:,.0f}")

with col3:
    dry_run = "Yes" if env.get("DRY_RUN", "true") == "true" else "No"
    st.metric("Dry Run", dry_run)

st.divider()

# Control Panel
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
    if st.button("🔄 Scan Markets", type="secondary"):
        with st.spinner("Scanning markets..."):
            result = subprocess.run(
                [sys.executable, "main.py", "--scan"],
                capture_output=True,
                text=True,
                timeout=60
            )
            st.text_area("Scan Results", result.stdout, height=200)

# Real-time Logs
st.divider()
st.subheader("📋 Live Logs")

if st.button("🗑️ Clear Logs"):
    st.session_state.logs = []

log_container = st.container(height=400)
with log_container:
    for log in st.session_state.logs[-100:]:
        st.text(log)

# Auto-refresh
if is_trading():
    import time
    time.sleep(2)
    st.rerun()

# Market Overview (placeholder - would connect to DB)
st.divider()
st.subheader("📊 Open Positions")

positions_data = {
    "Ticker": ["KXPRESPERSON-28-KHAR", "KXMISC-28-A"],
    "Side": ["YES", "NO"],
    "Contracts": [10, 5],
    "Entry Price": ["$52", "$48"],
    "Current P/L": ["+$$20", "-$10"]
}

df = pd.DataFrame(positions_data)

# Color code the P/L
def color_pl(val):
    if "+" in str(val):
        return "color: #22c55e"
    elif "-" in str(val):
        return "color: #ef4444"
    return ""

st.dataframe(
    df.style.applymap(color_pl, subset=["Current P/L"]),
    use_container_width=True,
    height=200
)

# Performance Stats
st.divider()
st.subheader("📈 Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Today's P/L", "+$15.00", "+7.5%")
with col2:
    st.metric("Total Trades", "12")
with col3:
    st.metric("Win Rate", "75%")
with col4:
    st.metric("Active Positions", "3")

# Footer
st.divider()
st.caption("Kalshi AI Hedge Fund | Use at your own risk")

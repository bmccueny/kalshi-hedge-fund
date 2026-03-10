#!/usr/bin/env python3
"""
Kalshi AI Hedge Fund — Entry Point

Usage:
  python main.py                    # Live trading (dry_run=True by default)
  python main.py --live             # Real trading (requires explicit flag)
  python main.py --dashboard        # View-only dashboard
  python main.py --scan             # One-shot scan, print opportunities
"""
import os
import asyncio
import argparse
import structlog
import logging

# Allow running from inside a Claude Code terminal (claude-agent-sdk blocks nested sessions)
os.environ.pop("CLAUDECODE", None)

# Configure structured logging (respect NO_COLOR for GUI subprocess)
_use_colors = os.environ.get("NO_COLOR", "") == "" and os.isatty(1)
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    processors=[
        structlog.dev.ConsoleRenderer(colors=_use_colors),
    ],
)
log = structlog.get_logger()


def parse_args():
    p = argparse.ArgumentParser(description="Kalshi AI Hedge Fund")
    p.add_argument("--live", action="store_true", help="Enable live trading (default: dry run)")
    p.add_argument("--dashboard", action="store_true", help="Show dashboard only")
    p.add_argument("--scan", action="store_true", help="One-shot opportunity scan")
    return p.parse_args()


async def run_dashboard():
    from core.kalshi_client import KalshiClient
    from monitoring.dashboard import render_dashboard
    async with KalshiClient() as kalshi:
        await render_dashboard(kalshi)


async def run_scan():
    from core.kalshi_client import KalshiClient
    from agents.market_scanner import MarketScannerAgent
    from agents.probability_analyst import ProbabilityAnalystAgent
    from data.news_feed import NewsFeed

    async with KalshiClient() as kalshi, NewsFeed() as news:
        scanner = MarketScannerAgent(kalshi)
        analyst = ProbabilityAnalystAgent(kalshi, news)

        print("Scanning markets...")
        candidates = await scanner.scan()
        top = candidates[:10]
        print(f"Found {len(candidates)} candidates — analysing {len(top)} in parallel...")

        estimates = await asyncio.gather(
            *[analyst.analyze(m) for m in top],
            return_exceptions=True,
        )

        print("")
        print(f"{'Ticker':<25} {'Title':<35} {'Mkt%':>6} {'True%':>6} {'Edge':>6} {'Conf':>5}")
        print("-" * 90)
        for estimate in estimates:
            if isinstance(estimate, Exception) or estimate is None:
                continue
            print(
                f"{estimate.ticker:<25} "
                f"{estimate.market_title[:34]:<35} "
                f"{estimate.market_price_pct:>5.1f}% "
                f"{estimate.true_prob_pct:>5.1f}% "
                f"{estimate.edge_pct:>+5.1f}% "
                f"{estimate.confidence:>5.2f}"
            )
        print("")


async def run_trading(dry_run: bool):
    from core.orchestrator import Orchestrator
    mode = "DRY RUN" if dry_run else "LIVE TRADING"
    log.info(f"starting_{mode.lower().replace(' ', '_')}", dry_run=dry_run)

    if not dry_run:
        confirm = input(f"\n⚠️  LIVE TRADING MODE — real money will be used. Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            log.info("live_trading_cancelled")
            return

    orchestrator = Orchestrator(dry_run=dry_run)
    await orchestrator.run()


def main():
    args = parse_args()

    # Check DRY_RUN env var, default to True (dry run)
    dry_run = os.environ.get("DRY_RUN", "true").lower() != "false"
    
    # CLI --live flag overrides env var
    if args.live:
        dry_run = False

    if args.dashboard:
        asyncio.run(run_dashboard())
    elif args.scan:
        asyncio.run(run_scan())
    else:
        asyncio.run(run_trading(dry_run=dry_run))


if __name__ == "__main__":
    main()

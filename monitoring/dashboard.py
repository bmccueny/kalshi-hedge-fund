"""
Rich terminal dashboard showing portfolio status in real time.
Run standalone: python -m monitoring.dashboard
"""
import asyncio
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from core.database import get_open_positions, get_daily_pnl
from core.kalshi_client import KalshiClient
from config.settings import TOTAL_CAPITAL_USD

console = Console()


async def render_dashboard(kalshi: KalshiClient):
    """Render a live dashboard with portfolio stats."""
    with Live(console=console, refresh_per_second=0.5) as live:
        while True:
            layout = await build_layout(kalshi)
            live.update(layout)
            await asyncio.sleep(10)


async def build_layout(kalshi: KalshiClient) -> Panel:
    positions = await get_open_positions()
    daily = await get_daily_pnl()

    # ── Header stats ──────────────────────────────────────────────────────────
    try:
        balance_data = await kalshi.get_balance()
        balance = balance_data.get("balance", 0) / 100
    except Exception:
        balance = TOTAL_CAPITAL_USD

    daily_pnl = daily.get("realized_pnl_usd", 0)
    pnl_color = "green" if daily_pnl >= 0 else "red"

    header = Table.grid(expand=True)
    header.add_column(justify="left")
    header.add_column(justify="center")
    header.add_column(justify="right")
    header.add_row(
        f"[bold]Kalshi Hedge Fund[/bold]",
        f"[dim]{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}[/dim]",
        f"Balance: [bold]${balance:,.2f}[/bold]  Daily P&L: [{pnl_color}]{daily_pnl:+.2f}[/{pnl_color}]",
    )

    # ── Positions table ───────────────────────────────────────────────────────
    pos_table = Table(title="Open Positions", expand=True, show_lines=True)
    pos_table.add_column("Ticker", style="cyan", no_wrap=True)
    pos_table.add_column("Side", justify="center")
    pos_table.add_column("Qty", justify="right")
    pos_table.add_column("Entry ¢", justify="right")
    pos_table.add_column("Current ¢", justify="right")
    pos_table.add_column("P&L", justify="right")
    pos_table.add_column("Opened", style="dim")

    for p in positions:
        entry = p["avg_price_cents"]
        current = p.get("current_price_cents", entry)
        pnl = (current - entry) * p["contracts"] / 100
        pnl_str = f"[green]+${pnl:.2f}[/green]" if pnl >= 0 else f"[red]-${abs(pnl):.2f}[/red]"
        side_str = "[green]YES[/green]" if p["side"] == "yes" else "[red]NO[/red]"
        pos_table.add_row(
            p["ticker"][:30],
            side_str,
            str(p["contracts"]),
            str(entry),
            str(current),
            pnl_str,
            p["opened_at"][:16],
        )

    if not positions:
        pos_table.add_row("[dim]No open positions[/dim]", "", "", "", "", "", "")

    # ── Stats footer ──────────────────────────────────────────────────────────
    total_at_risk = sum(p["contracts"] * p["avg_price_cents"] / 100 for p in positions)
    heat_pct = total_at_risk / TOTAL_CAPITAL_USD * 100

    stats = Table.grid(expand=True)
    stats.add_column()
    stats.add_column()
    stats.add_column()
    stats.add_row(
        f"Open Positions: [bold]{len(positions)}[/bold]",
        f"Capital at Risk: [bold]${total_at_risk:,.2f}[/bold] ({heat_pct:.1f}%)",
        f"Trades Today: [bold]{daily.get('num_trades', 0)}[/bold]",
    )

    from rich.columns import Columns
    return Panel(
        Columns([pos_table]),
        title="[bold blue]Kalshi AI Hedge Fund[/bold blue]",
        subtitle=f"Daily P&L: [{'green' if daily_pnl >= 0 else 'red'}]{daily_pnl:+.2f}[/{'green' if daily_pnl >= 0 else 'red'}]",
    )


if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    load_dotenv()

    async def main():
        async with KalshiClient() as kalshi:
            await render_dashboard(kalshi)

    asyncio.run(main())

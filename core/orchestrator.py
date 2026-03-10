"""
Orchestrator — coordinates all agents in the trading loop.

Main loop:
  1. Market Scanner finds candidates
  2. Probability Analyst estimates true probabilities
  3. Risk Manager sizes and approves trades
  4. Trade Executor places orders
  5. Portfolio Monitor watches open positions
  6. News Analyzer runs in parallel, updating estimates when news breaks

Design: all agents run concurrently with asyncio. News analyzer and
portfolio monitor are background tasks; scanning loop is the main loop.
"""
import asyncio
import signal
import structlog
from core.kalshi_client import KalshiClient
from core.database import init_db, save_signal, reconcile
from agents.market_scanner import MarketScannerAgent
from agents.market_research import MarketResearchAgent
from agents.probability_analyst import ProbabilityAnalystAgent
from agents.risk_manager import RiskManagerAgent
from agents.trade_executor import TradeExecutorAgent
from agents.portfolio_monitor import PortfolioMonitorAgent
from agents.news_analyzer import NewsAnalyzerAgent
from data.news_feed import NewsFeed
from config.settings import SCAN_INTERVAL_SECONDS, MIN_EDGE_THRESHOLD

log = structlog.get_logger()


class Orchestrator:
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self._open_markets: list[dict] = []
        self._tasks: list[asyncio.Task] = []

    async def _handle_shutdown(self, sig: signal.Signals) -> None:
        log.warning("shutdown_signal_received", signal=sig.name)
        for task in self._tasks:
            task.cancel()
        log.info("shutdown_tasks_cancelled", count=len(self._tasks))
        # Give tasks a moment to cancel cleanly
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        log.info("shutdown_complete")

    async def run(self):
        await init_db()
        log.info("orchestrator_start", dry_run=self.dry_run)

        # Install signal handlers for graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(self._handle_shutdown(s)),
            )

        async with KalshiClient() as kalshi, NewsFeed() as news:
            # ── Startup reconciliation ─────────────────────────────────────
            try:
                positions_data = await kalshi.get_positions()
                balance_data = await kalshi.get_balance()
                kalshi_positions = positions_data.get("positions", [])
                kalshi_balance = balance_data.get("balance", 0)
                recon = await reconcile(kalshi_positions, kalshi_balance)
                if recon["discrepancies"]:
                    log.warning(
                        "startup_reconciliation_discrepancies",
                        discrepancies=recon["discrepancies"],
                    )
                else:
                    log.info("startup_reconciliation_clean",
                             positions=recon["kalshi_count"],
                             balance_cents=kalshi_balance)
            except Exception as e:
                log.error("startup_reconciliation_failed", error=str(e))
            executor = TradeExecutorAgent(kalshi, dry_run=self.dry_run)
            scanner = MarketScannerAgent(kalshi)
            researcher = MarketResearchAgent(kalshi)
            analyst = ProbabilityAnalystAgent(kalshi, news)
            risk_mgr = RiskManagerAgent()
            monitor = PortfolioMonitorAgent(kalshi, executor)
            news_agent = NewsAnalyzerAgent(news)

            # Wire news signals into re-analysis pipeline
            @news_agent.on_signal
            async def on_news_signal(news_signal):
                log.info("news_triggered_reanalysis", tickers=news_signal.affected_tickers)
                for ticker in news_signal.affected_tickers[:3]:  # limit concurrent re-analyses
                    try:
                        market_data = await kalshi.get_market(ticker)
                        market = market_data.get("market", {})
                        est = await analyst.analyze(market)
                        if est and abs(est.edge_pct) >= MIN_EDGE_THRESHOLD * 100:
                            decision = await risk_mgr.evaluate(est, market)
                            if decision.approved:
                                await executor.execute(decision)
                    except Exception as e:
                        log.error("news_reanalysis_error", ticker=ticker, error=str(e))

            # Start background tasks (stored on self so shutdown handler can cancel them)
            self._tasks = [
                asyncio.create_task(monitor.run(check_interval=120)),
                asyncio.create_task(news_agent.run(poll_interval=60)),
                asyncio.create_task(self._research_loop(kalshi, researcher)),
                asyncio.create_task(self._main_scan_loop(
                    kalshi, scanner, analyst, risk_mgr, executor
                )),
            ]

            log.info("all_agents_started")
            try:
                await asyncio.gather(*self._tasks)
            except asyncio.CancelledError:
                log.info("orchestrator_tasks_cancelled")

    async def _main_scan_loop(
        self,
        kalshi: KalshiClient,
        scanner: MarketScannerAgent,
        analyst: ProbabilityAnalystAgent,
        risk_mgr: RiskManagerAgent,
        executor: TradeExecutorAgent,
    ):
        """Periodically scan all markets and trade on edge."""
        while True:
            try:
                await self._scan_and_trade(kalshi, scanner, analyst, risk_mgr, executor)
            except Exception as e:
                log.error("scan_loop_error", error=str(e))

            log.info("scan_sleeping", seconds=SCAN_INTERVAL_SECONDS)
            await asyncio.sleep(SCAN_INTERVAL_SECONDS)

    async def _scan_and_trade(
        self,
        kalshi: KalshiClient,
        scanner: MarketScannerAgent,
        analyst: ProbabilityAnalystAgent,
        risk_mgr: RiskManagerAgent,
        executor: TradeExecutorAgent,
    ):
        # Step 1: Scan for candidates
        candidates = await scanner.scan()
        if not candidates:
            log.info("no_candidates_found")
            return

        # Step 2: Analyze top candidates in parallel (max 5 at once)
        top = candidates[:5]
        log.info("analyzing_candidates", count=len(top))
        analysis_tasks = [analyst.analyze(m) for m in top]
        estimates = await asyncio.gather(*analysis_tasks, return_exceptions=True)

        # Step 3: Risk-check and execute trades for good estimates
        for market, estimate in zip(top, estimates):
            if isinstance(estimate, Exception):
                log.error("analysis_error", error=str(estimate))
                continue
            if estimate is None:
                continue

            # Save signal to DB
            await save_signal(
                ticker=estimate.ticker,
                agent="ProbabilityAnalyst",
                true_prob=estimate.true_prob_pct / 100,
                market_prob=estimate.market_price_pct / 100,
                edge=estimate.edge_pct / 100,
                confidence=estimate.confidence,
                rationale=estimate.rationale[:1000],
            )

            if abs(estimate.edge_pct) < MIN_EDGE_THRESHOLD * 100:
                log.info("edge_too_small", ticker=estimate.ticker, edge=estimate.edge_pct)
                continue

            decision = await risk_mgr.evaluate(estimate, market)
            if decision.approved:
                await executor.execute(decision)

    async def _research_loop(
        self,
        kalshi: KalshiClient,
        researcher: MarketResearchAgent,
    ):
        """Periodically run market research."""
        import time
        RESEARCH_INTERVAL = 3600  # every hour
        
        while True:
            try:
                log.info("research_loop_start")
                result = await researcher.research(max_markets=200)
                log.info(
                    "research_complete",
                    categories=len(result.get("categories", {})),
                    anomalies=len(result.get("detected_anomalies", [])),
                    trends=len(result.get("cross_market_trends", [])),
                )
            except Exception as e:
                log.error("research_loop_error", error=str(e))
            
            await asyncio.sleep(RESEARCH_INTERVAL)

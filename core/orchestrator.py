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
import structlog
from core.kalshi_client import KalshiClient
from core.database import init_db, save_signal
from agents.market_scanner import MarketScannerAgent
from agents.probability_analyst import ProbabilityAnalystAgent
from agents.risk_manager import RiskManagerAgent
from agents.trade_executor import TradeExecutorAgent
from agents.portfolio_monitor import PortfolioMonitorAgent
from agents.news_analyzer import NewsAnalyzerAgent
from data.news_feed import NewsFeeed
from config.settings import SCAN_INTERVAL_SECONDS, MIN_EDGE_THRESHOLD

log = structlog.get_logger()


class Orchestrator:
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self._open_markets: list[dict] = []

    async def run(self):
        await init_db()
        log.info("orchestrator_start", dry_run=self.dry_run)

        async with KalshiClient() as kalshi, NewsFeeed() as news:
            executor = TradeExecutorAgent(kalshi, dry_run=self.dry_run)
            scanner = MarketScannerAgent(kalshi)
            analyst = ProbabilityAnalystAgent(kalshi, news)
            risk_mgr = RiskManagerAgent()
            monitor = PortfolioMonitorAgent(kalshi, executor)
            news_agent = NewsAnalyzerAgent(news)

            # Wire news signals into re-analysis pipeline
            @news_agent.on_signal
            async def on_news_signal(signal):
                log.info("news_triggered_reanalysis", tickers=signal.affected_tickers)
                for ticker in signal.affected_tickers[:3]:  # limit concurrent re-analyses
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

            # Start background tasks
            tasks = [
                asyncio.create_task(monitor.run(check_interval=120)),
                asyncio.create_task(news_agent.run(poll_interval=60)),
                asyncio.create_task(self._main_scan_loop(
                    kalshi, scanner, analyst, risk_mgr, executor
                )),
            ]

            log.info("all_agents_started")
            await asyncio.gather(*tasks)

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

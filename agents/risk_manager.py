"""
Risk Manager Agent
───────────────────
Challenge overcome: The biggest risk in prediction market trading is:
1. Overconcentration — too much in correlated events
2. Adverse selection — trading against informed participants
3. Liquidity risk — unable to exit positions
4. Model risk — AI probability estimates are wrong
5. Daily loss limits being blown through

This agent approves/rejects/resizes every trade before execution.
It uses rule-based checks first, then AI reasoning for edge cases.
"""
import json
from dataclasses import dataclass
from agents.base_agent import BaseAgent
from agents.probability_analyst import ProbabilityEstimate
from core.database import get_open_positions, get_daily_pnl
from risk.kelly import kelly_fraction, no_kelly_fraction, compute_position_size, contracts_from_usd
from config.settings import (
    MAX_POSITION_SIZE_USD, MAX_PORTFOLIO_HEAT_PCT, MAX_SINGLE_MARKET_PCT,
    DAILY_LOSS_LIMIT_USD, TOTAL_CAPITAL_USD, MIN_EDGE_THRESHOLD
)
import structlog

log = structlog.get_logger()


@dataclass
class TradeDecision:
    approved: bool
    ticker: str
    side: str               # "yes" | "no"
    contracts: int
    price_cents: int
    position_size_usd: float
    kelly_f: float
    rejection_reason: str | None = None
    ai_notes: str | None = None


RISK_SYSTEM = """You are the chief risk officer of a prediction market hedge fund.

A quantitative model has flagged a potential trade. Your job: apply holistic risk judgment.

Consider:
1. Is the edge genuine or could it be due to information asymmetry (we might be the dumb money)?
2. Are there correlated positions that increase our aggregate exposure?
3. Is the market thin enough that our trade would move the price against us?
4. Is the AI probability estimate well-reasoned or superficial?
5. Regulatory/compliance: Are position limits respected?

Output JSON:
{
  "approved": true/false,
  "contracts": <int, adjusted if needed>,
  "notes": "...",
  "concerns": ["...", ...]
}
"""


class RiskManagerAgent(BaseAgent):
    def __init__(self):
        super().__init__("RiskManager")

    async def evaluate(
        self,
        estimate: ProbabilityEstimate,
        market: dict,
    ) -> TradeDecision:
        """
        Multi-layer risk evaluation:
        Layer 1: Hard rule checks (fast, no AI)
        Layer 2: Portfolio-level checks
        Layer 3: AI holistic review (for borderline/large trades)
        """
        ticker = estimate.ticker
        edge = estimate.edge_pct

        # ── Layer 1: Hard rules ────────────────────────────────────────────────

        if abs(edge) < MIN_EDGE_THRESHOLD * 100:  # edge_pct is 0-100
            return TradeDecision(
                approved=False, ticker=ticker, side="", contracts=0,
                price_cents=0, position_size_usd=0, kelly_f=0,
                rejection_reason=f"Edge {edge:.1f}% below minimum {MIN_EDGE_THRESHOLD*100:.1f}%"
            )

        if estimate.confidence < 0.4:
            return TradeDecision(
                approved=False, ticker=ticker, side="", contracts=0,
                price_cents=0, position_size_usd=0, kelly_f=0,
                rejection_reason=f"Confidence {estimate.confidence:.2f} too low"
            )

        daily = await get_daily_pnl()
        if daily["realized_pnl_usd"] < -DAILY_LOSS_LIMIT_USD:
            return TradeDecision(
                approved=False, ticker=ticker, side="", contracts=0,
                price_cents=0, position_size_usd=0, kelly_f=0,
                rejection_reason=f"Daily loss limit hit: ${daily['realized_pnl_usd']:.2f}"
            )

        # ── Layer 2: Position sizing ───────────────────────────────────────────

        side = "yes" if edge > 0 else "no"
        market_prob = estimate.market_price_pct / 100
        true_prob = estimate.true_prob_pct / 100

        if side == "yes":
            kf = kelly_fraction(true_prob, market_prob, max_fraction=0.25, confidence=estimate.confidence)
            price_cents = market.get("yes_ask", int(market_prob * 100))
        else:
            kf = no_kelly_fraction(true_prob, market_prob, max_fraction=0.25, confidence=estimate.confidence)
            price_cents = market.get("no_ask", int((1 - market_prob) * 100))

        position_usd = compute_position_size(TOTAL_CAPITAL_USD, kf, MAX_POSITION_SIZE_USD)

        # Portfolio heat check
        open_positions = await get_open_positions()
        total_at_risk = sum(
            p["contracts"] * p["avg_price_cents"] / 100 for p in open_positions
        )
        heat_pct = total_at_risk / TOTAL_CAPITAL_USD
        if heat_pct + (position_usd / TOTAL_CAPITAL_USD) > MAX_PORTFOLIO_HEAT_PCT:
            position_usd *= 0.5  # halve the position to stay within heat limit
            log.warning("risk_portfolio_heat_reduction", heat_pct=heat_pct)

        # Single market concentration check
        if position_usd / TOTAL_CAPITAL_USD > MAX_SINGLE_MARKET_PCT:
            position_usd = TOTAL_CAPITAL_USD * MAX_SINGLE_MARKET_PCT
            log.info("risk_concentration_cap", ticker=ticker)

        contracts = contracts_from_usd(position_usd, price_cents)
        if contracts < 1:
            return TradeDecision(
                approved=False, ticker=ticker, side=side, contracts=0,
                price_cents=price_cents, position_size_usd=0, kelly_f=kf,
                rejection_reason="Position size rounds to 0 contracts"
            )

        # ── Layer 3: AI holistic review (for positions > $100) ─────────────────

        if position_usd > 100:
            ai_notes = await self._ai_review(estimate, market, position_usd, side)
            if ai_notes and '"approved": false' in ai_notes.lower():
                return TradeDecision(
                    approved=False, ticker=ticker, side=side, contracts=contracts,
                    price_cents=price_cents, position_size_usd=position_usd, kelly_f=kf,
                    rejection_reason="AI risk review rejected trade",
                    ai_notes=ai_notes,
                )
        else:
            ai_notes = None

        log.info(
            "risk_approved",
            ticker=ticker, side=side, contracts=contracts,
            position_usd=position_usd, kelly_f=kf, edge=edge
        )
        return TradeDecision(
            approved=True, ticker=ticker, side=side, contracts=contracts,
            price_cents=price_cents, position_size_usd=position_usd, kelly_f=kf,
            ai_notes=ai_notes,
        )

    async def _ai_review(
        self,
        estimate: ProbabilityEstimate,
        market: dict,
        position_usd: float,
        side: str,
    ) -> str:
        prompt = f"""Reviewing trade for approval:

Market: {estimate.market_title} ({estimate.ticker})
Side: {side.upper()}
Size: ${position_usd:.2f}
Market price: {estimate.market_price_pct:.1f}%
Our estimate: {estimate.true_prob_pct:.1f}%
Edge: {estimate.edge_pct:.1f}%
Confidence: {estimate.confidence:.2f}

Analyst rationale: {estimate.rationale}

Key risks identified: {json.dumps(estimate.key_risks)}
Information needs: {json.dumps(estimate.information_needs)}

Market open interest: ${market.get('open_interest', 0)/100:.2f}
Market volume today: ${market.get('volume', 0)/100:.2f}

Should we approve this trade? Consider adverse selection and model risk."""

        return await self.analyze(prompt, system=RISK_SYSTEM, max_turns=2)

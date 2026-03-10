# Strategy Abstraction

This directory will hold concrete trading strategies that plug into the orchestrator.

## Planned Interface

```python
from abc import ABC, abstractmethod
from agents.probability_analyst import ProbabilityEstimate

class BaseStrategy(ABC):
    name: str
    description: str

    @abstractmethod
    async def analyze(self, market: dict) -> ProbabilityEstimate | None:
        """
        Given a market dict from the Kalshi API, return a probability estimate
        or None if this strategy has no view on the market.
        """
        ...
```

## Planned Strategies

| Strategy | Description | Status |
|----------|-------------|--------|
| `momentum.py` | Trade markets where price has been moving consistently in one direction | Prototype in `agents/momentum_agent.py` |
| `mean_reversion.py` | Fade large short-term price moves with insufficient news backing | Planned |
| `event_driven.py` | Trade on news signals before the market fully reprices | Prototype in `agents/event_driven_agent.py` |
| `arbitrage.py` | Exploit pricing inconsistencies across related markets | Prototype in `agents/arbitrage_agent.py` |
| `base_rate.py` | Pure statistical base-rate trading using historical resolution data | Planned |

## Wiring Strategies into the Orchestrator

The intent is for the `Orchestrator` to iterate over registered strategies in
`strategies/__init__.py` and run each one's `analyze()` instead of going
directly to `ProbabilityAnalystAgent`. This separates "what to trade" (strategy)
from "how much to trade" (risk manager) and "how to trade" (executor).

## Adding a New Strategy

1. Create `strategies/my_strategy.py` implementing `BaseStrategy`
2. Import and register it in `strategies/__init__.py`
3. The orchestrator will pick it up automatically on next start

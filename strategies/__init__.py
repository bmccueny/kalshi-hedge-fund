"""
Strategy abstraction layer for the Kalshi hedge fund.

Each strategy is a self-contained module that:
- Implements the BaseStrategy interface (see README.md for spec)
- Exposes a name, description, and analyze(market) coroutine
- Is registered here for the orchestrator to discover

Current strategies:
  (none yet — see README.md for planned abstractions)
"""

"""
Kelly Criterion position sizing with fractional Kelly and confidence scaling.

Full Kelly is theoretically optimal but causes extreme volatility.
We use fractional Kelly (typically 1/4 to 1/2) scaled by confidence.
"""


def kelly_fraction(
    true_prob: float,
    market_prob: float,
    max_fraction: float = 0.25,
    confidence: float = 1.0,
) -> float:
    """
    Compute the Kelly fraction of bankroll to bet.

    For binary YES/NO markets:
      p = true probability of YES
      b = net odds (win $1 for every $q invested when q is the NO cost)

    For Kalshi contracts paying $1:
      YES cost = yes_price_cents / 100
      If YES wins: profit = (1 - yes_price) per dollar risked
      If YES loses: lose yes_price per dollar risked

    Kelly fraction f* = (p * b - q) / b
    where b = (1 - yes_price) / yes_price, q = 1 - p

    Args:
        true_prob: Estimated true probability of YES (0-1)
        market_prob: Market-implied probability of YES (0-1)
        max_fraction: Hard cap on fraction (fractional Kelly)
        confidence: Agent confidence scalar (0-1); reduces position size

    Returns:
        Fraction of capital to deploy (0-1)
    """
    p = true_prob
    q = 1 - p
    yes_price = market_prob

    if yes_price <= 0 or yes_price >= 1:
        return 0.0

    # Net odds: win/risk ratio
    b = (1 - yes_price) / yes_price

    kelly_f = (p * b - q) / b

    if kelly_f <= 0:
        return 0.0  # Negative edge — don't bet

    # Apply fractional Kelly + confidence scaling
    adjusted = kelly_f * max_fraction * confidence

    # Hard cap
    return min(adjusted, max_fraction)


def no_kelly_fraction(
    true_prob: float,
    market_prob: float,
    max_fraction: float = 0.25,
    confidence: float = 1.0,
) -> float:
    """Same as kelly_fraction but for betting NO side."""
    return kelly_fraction(
        true_prob=1 - true_prob,
        market_prob=1 - market_prob,
        max_fraction=max_fraction,
        confidence=confidence,
    )


def compute_position_size(
    capital_usd: float,
    kelly_f: float,
    max_position_usd: float,
) -> float:
    """Translate Kelly fraction to dollar position size with hard cap."""
    raw = capital_usd * kelly_f
    return min(raw, max_position_usd)


def contracts_from_usd(amount_usd: float, price_cents: int) -> int:
    """Convert dollar amount to number of Kalshi contracts."""
    if price_cents <= 0:
        return 0
    price_dollars = price_cents / 100
    return max(1, int(amount_usd / price_dollars))

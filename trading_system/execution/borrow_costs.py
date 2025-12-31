"""Short borrow cost modeling for short positions."""

from typing import Literal, Optional

import pandas as pd


def compute_borrow_cost_bps(
    asset_class: Literal["equity", "crypto"], symbol: Optional[str] = None, date: Optional[pd.Timestamp] = None
) -> float:
    """
    Compute short borrow cost in basis points per day.

    For equity: Typically 0-50 bps/day depending on availability
    For crypto: Typically 0-10 bps/day (lower due to easier borrowing)

    Args:
        asset_class: "equity" or "crypto"
        symbol: Optional symbol (for future symbol-specific rates)
        date: Optional date (for future time-varying rates)

    Returns:
        Borrow cost in basis points per day

    Example:
        >>> cost = compute_borrow_cost_bps("equity")
        >>> # Returns ~5-10 bps/day for typical equity
        >>> cost = compute_borrow_cost_bps("crypto")
        >>> # Returns ~1-2 bps/day for typical crypto
    """
    if asset_class == "equity":
        # Equity: typical range 0-50 bps/day
        # Hard-to-borrow stocks can be 100+ bps/day, but we use average
        # For MVP, use a fixed rate (can be made symbol-specific later)
        return 5.0  # 5 bps/day = ~1.8% annualized
    elif asset_class == "crypto":
        # Crypto: typically lower, 0-10 bps/day
        # Easier to borrow, lower costs
        return 1.0  # 1 bps/day = ~0.36% annualized
    else:
        raise ValueError(f"Invalid asset_class: {asset_class}, must be 'equity' or 'crypto'")


def compute_borrow_cost_dollars(
    notional: float, asset_class: Literal["equity", "crypto"], days_held: float, symbol: Optional[str] = None
) -> float:
    """
    Compute total borrow cost in dollars for a short position.

    Args:
        notional: Position notional value (entry_price * quantity)
        asset_class: "equity" or "crypto"
        days_held: Number of days position was held
        symbol: Optional symbol (for future symbol-specific rates)

    Returns:
        Total borrow cost in dollars

    Example:
        >>> cost = compute_borrow_cost_dollars(
        ...     notional=100_000,
        ...     asset_class="equity",
        ...     days_held=5.0
        ... )
        >>> # Returns 100_000 * (5 / 10000) * 5 = 250.0
    """
    if days_held <= 0:
        return 0.0

    borrow_bps = compute_borrow_cost_bps(asset_class, symbol=symbol)
    cost_per_day = notional * (borrow_bps / 10000.0)
    total_cost = cost_per_day * days_held

    return total_cost


def is_hard_to_borrow(symbol: str, asset_class: Literal["equity", "crypto"], adv20: Optional[float] = None) -> bool:
    """
    Check if a symbol is hard to borrow (high borrow cost or unavailable).

    For MVP, we use a simple heuristic:
    - Equity: Low liquidity (ADV20 < threshold) may indicate hard-to-borrow
    - Crypto: Generally easier to borrow, rarely hard-to-borrow

    Args:
        symbol: Symbol to check
        asset_class: "equity" or "crypto"
        adv20: Optional 20-day average dollar volume

    Returns:
        True if symbol is likely hard to borrow

    Example:
        >>> is_htb = is_hard_to_borrow("XYZ", "equity", adv20=1_000_000)
        >>> # Returns True if ADV20 is very low
    """
    if asset_class == "crypto":
        # Crypto is generally easy to borrow
        return False

    # Equity: use ADV20 as proxy for borrow availability
    # Very low liquidity may indicate hard-to-borrow
    if adv20 is not None:
        # Threshold: $5M ADV20 (very low liquidity)
        return adv20 < 5_000_000

    # Default: assume not hard-to-borrow (conservative)
    return False

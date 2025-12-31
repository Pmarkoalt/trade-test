"""Execution fees calculation for equity and crypto."""

from typing import Literal


def compute_fee_bps(asset_class: Literal["equity", "crypto"]) -> int:
    """
    Compute fee in basis points per side.
    
    Args:
        asset_class: "equity" or "crypto"
    
    Returns:
        Fee in basis points (1 for equity, 8 for crypto)
    
    Example:
        >>> fee = compute_fee_bps("equity")
        >>> # Returns 1
        >>> fee = compute_fee_bps("crypto")
        >>> # Returns 8
    """
    if asset_class == "equity":
        return 1
    elif asset_class == "crypto":
        return 8
    else:
        raise ValueError(f"Invalid asset_class: {asset_class}, must be 'equity' or 'crypto'")


def compute_fee_cost(
    notional: float,
    asset_class: Literal["equity", "crypto"]
) -> float:
    """
    Compute total fee cost in dollars.
    
    Args:
        notional: Order notional (fill_price * quantity)
        asset_class: "equity" or "crypto"
    
    Returns:
        Fee cost in dollars
    
    Example:
        >>> cost = compute_fee_cost(100_000, "equity")
        >>> # Returns 100_000 * (1 / 10000) = 10.0
    """
    fee_bps = compute_fee_bps(asset_class)
    return notional * (fee_bps / 10000.0)


"""Capacity constraint checks for order execution."""

from typing import Optional, Tuple


def check_capacity_constraint(
    order_notional: float,
    adv20: float,
    asset_class: str,
    max_order_pct_adv: Optional[float] = None
) -> Tuple[bool, str]:
    """
    Check if order passes capacity constraint.
    
    Capacity constraint: order_notional <= max_pct * ADV20
    
    Default thresholds:
    - Equity: 0.5% of ADV20 (0.005)
    - Crypto: 0.25% of ADV20 (0.0025)
    
    Args:
        order_notional: Order size in dollars
        adv20: 20-day average dollar volume
        asset_class: "equity" or "crypto"
        max_order_pct_adv: Optional override for max percentage (if None, uses defaults)
    
    Returns:
        Tuple of (passed: bool, reason: str)
        - passed: True if order passes capacity constraint
        - reason: Empty string if passed, rejection reason if failed
    
    Example:
        >>> passed, reason = check_capacity_constraint(
        ...     order_notional=50_000,
        ...     adv20=10_000_000,
        ...     asset_class="equity"
        ... )
        >>> # Returns (True, "") since 50k < 0.5% of 10M = 50k (at threshold)
    """
    if adv20 <= 0 or order_notional <= 0:
        return False, "Invalid ADV20 or order_notional"
    
    # Default thresholds
    if max_order_pct_adv is None:
        if asset_class == "equity":
            max_order_pct_adv = 0.005  # 0.5%
        elif asset_class == "crypto":
            max_order_pct_adv = 0.0025  # 0.25%
        else:
            return False, f"Invalid asset_class: {asset_class}"
    
    # Compute maximum allowed order size
    max_order_notional = max_order_pct_adv * adv20
    
    # Check constraint
    if order_notional > max_order_notional:
        return False, (
            f"Order notional {order_notional:,.0f} exceeds capacity limit "
            f"{max_order_notional:,.0f} ({max_order_pct_adv*100:.2f}% of ADV20)"
        )
    
    return True, ""


def check_capacity_constraint_with_quantity(
    quantity: int,
    price: float,
    adv20: float,
    asset_class: str,
    max_order_pct_adv: Optional[float] = None
) -> Tuple[bool, str]:
    """
    Check capacity constraint using quantity and price.
    
    Convenience wrapper that computes order_notional from quantity and price.
    
    Args:
        quantity: Order quantity (shares/units)
        price: Expected fill price (for notional calculation)
        adv20: 20-day average dollar volume
        asset_class: "equity" or "crypto"
        max_order_pct_adv: Optional override for max percentage
    
    Returns:
        Tuple of (passed: bool, reason: str)
    """
    order_notional = quantity * price
    return check_capacity_constraint(
        order_notional=order_notional,
        adv20=adv20,
        asset_class=asset_class,
        max_order_pct_adv=max_order_pct_adv
    )


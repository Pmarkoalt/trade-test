"""Position sizing calculations with risk-based sizing and constraints."""

import math
from typing import Optional


def calculate_position_size(
    equity: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
    max_position_notional: float,
    max_exposure: float,
    available_cash: float,
    risk_multiplier: float = 1.0,
) -> int:
    """Calculate position size with risk-based sizing and constraints.

    Steps:
    1. Calculate desired quantity from risk (with volatility scaling)
    2. Check max position notional constraint (15% of equity)
    3. Check max exposure constraint (80% of equity)
    4. Check available cash constraint
    5. Return minimum of all constraints

    Args:
        equity: Current portfolio equity
        risk_pct: Base risk per trade (e.g., 0.0075 for 0.75%)
        entry_price: Entry price for the position
        stop_price: Stop loss price (must be < entry_price for long, > entry_price for short)
        max_position_notional: Maximum position notional as fraction of equity (e.g., 0.15 for 15%)
        max_exposure: Maximum total exposure as fraction of equity (e.g., 0.80 for 80%)
        available_cash: Available cash for the position
        risk_multiplier: Volatility scaling multiplier (0.33 to 1.0, default: 1.0)

    Returns:
        Position size (quantity) as integer, or 0 if cannot afford even 1 share
    """
    if equity <= 0:
        return 0

    if entry_price <= 0:
        return 0

    if stop_price <= 0:
        return 0

    # Determine position side from stop price relationship to entry
    # Long: stop_price < entry_price
    # Short: stop_price > entry_price
    is_long = stop_price < entry_price
    is_short = stop_price > entry_price

    if not (is_long or is_short):
        return 0  # Invalid: stop_price == entry_price

    # Apply volatility scaling to risk percentage
    adjusted_risk_pct = risk_pct * risk_multiplier

    # Risk-based sizing
    risk_dollars = equity * adjusted_risk_pct
    # Calculate stop distance (always positive for risk calculation)
    stop_distance = abs(entry_price - stop_price)

    if stop_distance <= 0:
        return 0  # Invalid stop

    qty_risk = int(risk_dollars / stop_distance)

    # Max position notional constraint (15% of equity)
    max_notional_dollars = equity * max_position_notional
    max_qty_notional = int(max_notional_dollars / entry_price)

    # Max exposure constraint (80% of equity)
    # Note: This is checked at portfolio level, but we can estimate here
    max_exposure_dollars = equity * max_exposure
    max_qty_exposure = int(max_exposure_dollars / entry_price)

    # Cash constraint
    # For longs: need cash to buy
    # For shorts: margin requirement (simplified, use gross exposure limits instead)
    # Note: Shorts receive proceeds, so cash constraint is less relevant, but we check gross exposure
    if is_long:
        max_qty_cash = int(available_cash / entry_price)
    else:  # Short
        # For shorts, we receive proceeds, so cash constraint doesn't apply the same way
        # But we still need to respect margin (handled via exposure limits)
        # Use a large number here since cash isn't the constraint for shorts
        max_qty_cash = int(available_cash * 10)  # Allow more for shorts (margin-based)

    # Take minimum of all constraints
    qty = min(qty_risk, max_qty_notional, max_qty_exposure, max_qty_cash)

    # Minimum position size check
    if qty < 1:
        return 0  # Cannot afford even 1 share

    return qty


def estimate_position_size(
    equity: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
    max_position_notional: float,
    risk_multiplier: float = 1.0,
) -> int:
    """Estimate position size for capacity checks (without cash/exposure constraints).

    This is used during signal generation to estimate order size for capacity checks.
    It does not check cash or exposure constraints since those are checked later.

    Args:
        equity: Current portfolio equity
        risk_pct: Base risk per trade (e.g., 0.0075 for 0.75%)
        entry_price: Entry price for the position
        stop_price: Stop loss price (must be < entry_price for long, > entry_price for short)
        max_position_notional: Maximum position notional as fraction of equity (e.g., 0.15 for 15%)
        risk_multiplier: Volatility scaling multiplier (0.33 to 1.0, default: 1.0)

    Returns:
        Estimated position size (quantity) as integer
    """
    if equity <= 0 or entry_price <= 0:
        return 0

    if stop_price <= 0:
        return 0

    # Determine position side from stop price relationship to entry
    # Long: stop_price < entry_price
    # Short: stop_price > entry_price
    is_long = stop_price < entry_price
    is_short = stop_price > entry_price

    if not (is_long or is_short):
        return 0  # Invalid: stop_price == entry_price

    # Apply volatility scaling to risk percentage
    adjusted_risk_pct = risk_pct * risk_multiplier

    # Risk-based sizing
    risk_dollars = equity * adjusted_risk_pct
    # Calculate stop distance (always positive for risk calculation)
    stop_distance = abs(entry_price - stop_price)

    if stop_distance <= 0:
        return 0

    qty_risk = int(risk_dollars / stop_distance)

    # Max position notional constraint
    max_notional_dollars = equity * max_position_notional
    max_qty_notional = int(max_notional_dollars / entry_price)

    # Take minimum (no cash/exposure check for estimation)
    qty = min(qty_risk, max_qty_notional)

    return max(0, qty)

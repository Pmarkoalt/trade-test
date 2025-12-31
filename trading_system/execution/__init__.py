"""Execution simulation with realistic slippage and fees."""

from .borrow_costs import compute_borrow_cost_bps, compute_borrow_cost_dollars, is_hard_to_borrow
from .capacity import check_capacity_constraint, check_capacity_constraint_with_quantity
from .fees import compute_fee_bps, compute_fee_cost
from .fill_simulator import reject_order_missing_data, simulate_fill
from .slippage import (
    compute_size_penalty,
    compute_slippage_bps,
    compute_slippage_components,
    compute_stress_multiplier,
    compute_volatility_multiplier,
    compute_weekend_penalty,
)
from .weekly_return import compute_weekly_return

__all__ = [
    # Slippage
    "compute_volatility_multiplier",
    "compute_size_penalty",
    "compute_weekend_penalty",
    "compute_stress_multiplier",
    "compute_slippage_bps",
    "compute_slippage_components",
    # Fees
    "compute_fee_bps",
    "compute_fee_cost",
    # Weekly return
    "compute_weekly_return",
    # Fill simulator
    "simulate_fill",
    "reject_order_missing_data",
    # Capacity
    "check_capacity_constraint",
    "check_capacity_constraint_with_quantity",
    # Borrow costs
    "compute_borrow_cost_bps",
    "compute_borrow_cost_dollars",
    "is_hard_to_borrow",
]

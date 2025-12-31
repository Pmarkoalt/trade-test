"""Execution simulation with realistic slippage and fees."""

from .slippage import (
    compute_volatility_multiplier,
    compute_size_penalty,
    compute_weekend_penalty,
    compute_stress_multiplier,
    compute_slippage_bps,
    compute_slippage_components
)
from .fees import compute_fee_bps, compute_fee_cost
from .weekly_return import compute_weekly_return
from .fill_simulator import simulate_fill, reject_order_missing_data
from .capacity import check_capacity_constraint, check_capacity_constraint_with_quantity

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
]


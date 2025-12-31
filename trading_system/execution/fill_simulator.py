"""Fill simulation with realistic slippage and fees."""

import uuid
from typing import Optional
import pandas as pd
import numpy as np

from ..models.orders import Order, Fill, SignalSide
from ..models.bar import Bar
from .slippage import compute_slippage_components
from .fees import compute_fee_bps, compute_fee_cost
from .weekly_return import compute_weekly_return


def simulate_fill(
    order: Order,
    open_bar: Bar,
    atr14: float,
    atr14_history: pd.Series,
    adv20: float,
    benchmark_bars: pd.DataFrame,
    base_slippage_bps: float,
    rng: Optional[np.random.Generator] = None
) -> Fill:
    """
    Simulate order fill with realistic slippage and fees.
    
    Fill price calculation:
    - BUY: fill_price = open_price * (1 + slippage_bps/10000)
    - SELL: fill_price = open_price * (1 - slippage_bps/10000)
    
    Args:
        order: Order to fill
        open_bar: Bar for execution date (contains open price)
        atr14: Current ATR14 value
        atr14_history: Series of ATR14 values (for volatility multiplier)
        adv20: 20-day average dollar volume
        benchmark_bars: DataFrame with benchmark data (SPY or BTC) for stress calculation
        base_slippage_bps: Base slippage (8 for equity, 10 for crypto)
        rng: Optional random number generator for reproducibility
    
    Returns:
        Fill object with execution details
    
    Raises:
        ValueError: If required data is missing or invalid
    
    Example:
        >>> rng = np.random.default_rng(seed=42)
        >>> fill = simulate_fill(
        ...     order=my_order,
        ...     open_bar=bar,
        ...     atr14=2.5,
        ...     atr14_history=atr_series,
        ...     adv20=10_000_000,
        ...     benchmark_bars=spy_bars,
        ...     base_slippage_bps=8,
        ...     rng=rng
        ... )
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Validate inputs
    if atr14 <= 0 or pd.isna(atr14):
        raise ValueError(f"Invalid ATR14: {atr14}")
    
    if adv20 <= 0 or pd.isna(adv20):
        raise ValueError(f"Invalid ADV20: {adv20}")
    
    if open_bar.date != order.execution_date:
        raise ValueError(
            f"Bar date {open_bar.date} does not match order execution_date {order.execution_date}"
        )
    
    open_price = open_bar.open
    if open_price <= 0:
        raise ValueError(f"Invalid open_price: {open_price}")
    
    # Compute weekly return for stress multiplier
    weekly_return = compute_weekly_return(
        benchmark_bars=benchmark_bars,
        current_date=open_bar.date,
        asset_class=order.asset_class
    )
    
    # Compute order notional
    order_notional = open_price * order.quantity
    
    # Compute slippage components
    slippage_bps, vol_mult, size_penalty, weekend_penalty, stress_mult = compute_slippage_components(
        order_notional=order_notional,
        atr14=atr14,
        atr14_history=atr14_history,
        adv20=adv20,
        date=open_bar.date,
        asset_class=order.asset_class,
        weekly_return=weekly_return,
        base_bps=base_slippage_bps,
        rng=rng
    )
    
    # Compute fill price (apply slippage)
    if order.side == SignalSide.BUY:
        # BUY: pay more (add slippage)
        fill_price = open_price * (1 + slippage_bps / 10000.0)
    else:  # SELL
        # SELL: receive less (subtract slippage)
        fill_price = open_price * (1 - slippage_bps / 10000.0)
    
    # Ensure fill_price is positive
    fill_price = max(fill_price, 0.01)
    
    # Compute fee
    fee_bps = compute_fee_bps(order.asset_class)
    notional = fill_price * order.quantity
    fee_cost = compute_fee_cost(notional, order.asset_class)
    
    # Compute slippage cost
    slippage_cost = notional * (slippage_bps / 10000.0)
    total_cost = slippage_cost + fee_cost
    
    # Create fill
    fill = Fill(
        fill_id=str(uuid.uuid4()),
        order_id=order.order_id,
        symbol=order.symbol,
        asset_class=order.asset_class,
        date=open_bar.date,
        side=order.side,
        quantity=order.quantity,
        fill_price=fill_price,
        open_price=open_price,
        slippage_bps=slippage_bps,
        fee_bps=fee_bps,
        total_cost=total_cost,
        vol_mult=vol_mult,
        size_penalty=size_penalty,
        weekend_penalty=weekend_penalty,
        stress_mult=stress_mult,
        notional=notional
    )
    
    return fill


def reject_order_missing_data(
    order: Order,
    reason: str
) -> Fill:
    """
    Create a rejection fill for orders that cannot be executed due to missing data.
    
    Args:
        order: Order to reject
        reason: Reason for rejection
    
    Returns:
        Fill object with rejection details (all costs set to 0)
    """
    fill = Fill(
        fill_id=str(uuid.uuid4()),
        order_id=order.order_id,
        symbol=order.symbol,
        asset_class=order.asset_class,
        date=order.execution_date,
        side=order.side,
        quantity=0,  # Not filled
        fill_price=0.0,
        open_price=0.0,
        slippage_bps=0.0,
        fee_bps=0.0,
        total_cost=0.0,
        vol_mult=1.0,
        size_penalty=1.0,
        weekend_penalty=1.0,
        stress_mult=1.0,
        notional=0.0
    )
    
    # Note: Fill model doesn't have rejection_reason, but Order.status should be set to REJECTED
    return fill


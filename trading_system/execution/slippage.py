"""Dynamic slippage calculation with volatility, size, weekend, and stress multipliers."""

import numpy as np
from typing import Optional, Tuple
import pandas as pd


def compute_volatility_multiplier(
    atr14: float,
    atr14_history: pd.Series
) -> float:
    """
    Compute volatility multiplier: ATR14 / mean(ATR14 last 60 days).
    
    Args:
        atr14: Current ATR14 value
        atr14_history: Series of ATR14 values (last 60 days or more)
    
    Returns:
        Volatility multiplier, clipped to [0.5, 3.0]
    
    Example:
        >>> atr_history = pd.Series([1.0, 1.1, 1.2, ...])  # 60+ values
        >>> vol_mult = compute_volatility_multiplier(1.5, atr_history)
        >>> # Returns ATR14 / mean(last 60) clipped to [0.5, 3.0]
    """
    if len(atr14_history) < 60:
        # Insufficient history: use current value as baseline
        mean_atr = atr14
    else:
        # Use last 60 days
        mean_atr = atr14_history.iloc[-60:].mean()
    
    if mean_atr <= 0:
        return 1.0  # Default to 1.0 if invalid mean
    
    vol_mult = atr14 / mean_atr
    
    # Clip to [0.5, 3.0]
    return np.clip(vol_mult, 0.5, 3.0)


def compute_size_penalty(
    order_notional: float,
    adv20: float
) -> float:
    """
    Compute size penalty: order_notional / (0.01 * ADV20).
    
    Args:
        order_notional: Order size in dollars
        adv20: 20-day average dollar volume
    
    Returns:
        Size penalty, clipped to [0.5, 2.0]
    
    Example:
        >>> penalty = compute_size_penalty(100_000, 10_000_000)
        >>> # Returns 100k / (0.01 * 10M) = 1.0, clipped to [0.5, 2.0]
    """
    if adv20 <= 0:
        return 1.0  # Default if invalid ADV20
    
    # 1% ADV20 threshold
    threshold = 0.01 * adv20
    
    if threshold <= 0:
        return 1.0
    
    size_penalty = order_notional / threshold
    
    # Clip to [0.5, 2.0]
    return np.clip(size_penalty, 0.5, 2.0)


def compute_weekend_penalty(
    date: pd.Timestamp,
    asset_class: str
) -> float:
    """
    Compute weekend penalty for crypto (1.5x on Sat/Sun UTC).
    
    Args:
        date: Execution date
        asset_class: "equity" or "crypto"
    
    Returns:
        Weekend penalty multiplier (1.0 for equity, 1.5 for crypto weekends, 1.0 otherwise)
    
    Example:
        >>> # Saturday UTC
        >>> penalty = compute_weekend_penalty(pd.Timestamp("2024-01-06", tz="UTC"), "crypto")
        >>> # Returns 1.5
    """
    if asset_class != "crypto":
        return 1.0  # No weekend penalty for equity
    
    # Crypto: 1.5x on Saturday (5) or Sunday (6) UTC
    # pandas Timestamp.weekday() returns 0=Monday, 6=Sunday
    weekday = date.weekday()
    
    if weekday == 5 or weekday == 6:  # Saturday or Sunday
        return 1.5
    
    return 1.0


def compute_stress_multiplier(
    weekly_return: float,
    asset_class: str
) -> float:
    """
    Compute stress multiplier based on benchmark weekly return.
    
    Args:
        weekly_return: Benchmark weekly return (SPY for equity, BTC for crypto)
        asset_class: "equity" or "crypto"
    
    Returns:
        Stress multiplier (2.0 if stress threshold exceeded, 1.0 otherwise)
    
    Example:
        >>> # SPY down 4% (below -3% threshold)
        >>> mult = compute_stress_multiplier(-0.04, "equity")
        >>> # Returns 2.0
    """
    if asset_class == "equity":
        threshold = -0.03  # -3%
    else:  # crypto
        threshold = -0.05  # -5%
    
    if weekly_return < threshold:
        return 2.0
    
    return 1.0


def compute_slippage_bps(
    base_bps: float,
    vol_mult: float,
    size_penalty: float,
    weekend_penalty: float,
    stress_mult: float,
    rng: Optional[np.random.Generator] = None
) -> Tuple[float, float, float]:
    """
    Compute slippage in basis points with variance.
    
    Formula:
        slippage_mean = base * vol_mult * size_penalty * weekend_penalty * stress_mult
        slippage_std = slippage_mean * 0.75 (or * 1.5 during stress)
        slippage_bps = N(mean, std) clipped to [0, 500]
    
    Args:
        base_bps: Base slippage (8 for equity, 10 for crypto)
        vol_mult: Volatility multiplier
        size_penalty: Size penalty multiplier
        weekend_penalty: Weekend penalty multiplier (1.0 for equity, 1.5 for crypto weekends)
        stress_mult: Stress multiplier (1.0 or 2.0)
        rng: Optional random number generator for reproducibility
    
    Returns:
        Tuple of (slippage_bps, slippage_mean, slippage_std)
        - slippage_bps: Final slippage in bps (clipped to [0, 500])
        - slippage_mean: Mean slippage before variance
        - slippage_std: Standard deviation used
    
    Example:
        >>> rng = np.random.default_rng(seed=42)
        >>> bps, mean, std = compute_slippage_bps(8, 1.2, 1.0, 1.0, 1.0, rng)
        >>> # Returns (slippage_bps, mean, std)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Compute mean slippage
    slippage_mean = base_bps * vol_mult * size_penalty * weekend_penalty * stress_mult
    
    # Compute standard deviation
    slippage_std = slippage_mean * 0.75
    
    # If stress multiplier is active, increase std by 1.5x (fatter tails)
    if stress_mult == 2.0:
        slippage_std *= 1.5
    
    # Sample from normal distribution
    slippage_draw = rng.normal(slippage_mean, slippage_std)
    
    # Clip to valid range [0, 500 bps]
    slippage_bps = np.clip(slippage_draw, 0.0, 500.0)
    
    return slippage_bps, slippage_mean, slippage_std


def compute_slippage_components(
    order_notional: float,
    atr14: float,
    atr14_history: pd.Series,
    adv20: float,
    date: pd.Timestamp,
    asset_class: str,
    weekly_return: float,
    base_bps: float,
    rng: Optional[np.random.Generator] = None
) -> Tuple[float, float, float, float, float]:
    """
    Compute all slippage components and final slippage.
    
    Args:
        order_notional: Order size in dollars
        atr14: Current ATR14 value
        atr14_history: Series of ATR14 values (for volatility multiplier)
        adv20: 20-day average dollar volume
        date: Execution date (for weekend penalty)
        asset_class: "equity" or "crypto"
        weekly_return: Benchmark weekly return (for stress multiplier)
        base_bps: Base slippage (8 for equity, 10 for crypto)
        rng: Optional random number generator
    
    Returns:
        Tuple of (slippage_bps, vol_mult, size_penalty, weekend_penalty, stress_mult)
    
    Example:
        >>> rng = np.random.default_rng(seed=42)
        >>> bps, vol, size, weekend, stress = compute_slippage_components(
        ...     order_notional=100_000,
        ...     atr14=2.5,
        ...     atr14_history=atr_series,
        ...     adv20=10_000_000,
        ...     date=pd.Timestamp("2024-01-15"),
        ...     asset_class="equity",
        ...     weekly_return=-0.02,
        ...     base_bps=8,
        ...     rng=rng
        ... )
    """
    # Compute multipliers
    vol_mult = compute_volatility_multiplier(atr14, atr14_history)
    size_penalty = compute_size_penalty(order_notional, adv20)
    weekend_penalty = compute_weekend_penalty(date, asset_class)
    stress_mult = compute_stress_multiplier(weekly_return, asset_class)
    
    # Compute final slippage
    slippage_bps, _, _ = compute_slippage_bps(
        base_bps=base_bps,
        vol_mult=vol_mult,
        size_penalty=size_penalty,
        weekend_penalty=weekend_penalty,
        stress_mult=stress_mult,
        rng=rng
    )
    
    return slippage_bps, vol_mult, size_penalty, weekend_penalty, stress_mult


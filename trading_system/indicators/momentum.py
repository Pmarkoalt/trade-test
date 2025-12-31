"""Momentum indicators (Rate of Change)."""

from typing import Optional

import numpy as np
import pandas as pd

from .cache import create_cache_key_for_series, get_cache


def roc(close: pd.Series, window: int = 60, use_cache: bool = True, symbol: Optional[str] = None) -> pd.Series:
    """Compute Rate of Change (ROC) over specified window.

    ROC measures the percentage change in price over a lookback period.
    Formula: (close[t] / close[t-window]) - 1

    Args:
        close: Series of closing prices
        window: Lookback period in days (default 60)

    Returns:
        Series with ROC values. Returns NaN if close[t-window] is missing
        or if there are insufficient data points.

    Example:
        >>> close = pd.Series([100, 101, 102, 103, 104, 105])
        >>> roc60 = roc(close, window=60)
        >>> # First 60 values will be NaN (need close[t-60])
    """
    if len(close) == 0:
        return pd.Series(dtype=float, index=close.index)

    if len(close) < window:
        # Not enough data: return all NaN
        return pd.Series(np.nan, index=close.index)

    # Check cache if enabled
    if use_cache:
        cache = get_cache()
        if cache is not None:
            cache_key = create_cache_key_for_series(close, "roc", window, symbol=symbol)
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

    # Optimized: Use vectorized division
    # Shift by window to get close[t-window]
    close_shifted = close.shift(window)

    # Calculate ROC: (close[t] / close[t-window]) - 1
    # Use np.divide for better performance with NaN handling
    roc_series = pd.Series(
        np.divide(close.values, close_shifted.values, out=np.full(len(close), np.nan), where=close_shifted.values != 0) - 1,
        index=close.index,
    )

    # Explicitly set NaN for first window values (where close[t-window] doesn't exist)
    if len(roc_series) > 0 and len(roc_series) >= window:
        roc_series.iloc[:window] = np.nan

    # Cache result if enabled
    if use_cache:
        cache = get_cache()
        if cache is not None:
            cache_key = create_cache_key_for_series(close, "roc", window, symbol=symbol)
            cache.set(cache_key, roc_series)

    return roc_series

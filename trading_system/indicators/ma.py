"""Moving average indicator calculations."""

import pandas as pd
import numpy as np
from typing import Union, Optional
from .cache import get_cache


def ma(series: pd.Series, window: int, use_cache: bool = True) -> pd.Series:
    """Compute moving average with proper NaN handling.
    
    Optimized version that uses caching and vectorized operations.
    
    Args:
        series: Price series (typically close prices)
        window: Moving average period (e.g., 20, 50, 200)
        use_cache: Whether to use caching (default True)
    
    Returns:
        Series with moving average values. Returns NaN for all dates
        before the window is filled (first window-1 values are NaN).
        Never forward-fills NaN values.
    
    Example:
        >>> close = pd.Series([100, 101, 102, 103, 104])
        >>> ma20 = ma(close, window=20)
        >>> # First 19 values will be NaN, 20th value is mean of first 20
    """
    if len(series) == 0:
        return pd.Series(dtype=float, index=series.index)
    
    # Check cache if enabled
    if use_cache:
        cache = get_cache()
        if cache is not None:
            # Create cache key (simplified - in production, use proper hash)
            cache_key = (f"ma_{id(series)}_{len(series)}_{series.iloc[-1] if len(series) > 0 else 0}", "ma", window)
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
    
    # Optimized: Use vectorized rolling mean (already fast in pandas)
    # min_periods=window ensures NaN until window values are available
    ma_series = series.rolling(window=window, min_periods=window).mean()
    
    # Explicitly set NaN for insufficient lookback (first window-1 values)
    # This ensures we never use partial windows
    # Note: pandas rolling with min_periods already does this, but we're explicit
    if len(ma_series) > 0 and len(ma_series) >= window:
        # Use vectorized assignment instead of iloc loop
        ma_series.iloc[:window-1] = np.nan
    
    # Cache result if enabled
    if use_cache:
        cache = get_cache()
        if cache is not None:
            cache_key = (f"ma_{id(series)}_{len(series)}_{series.iloc[-1] if len(series) > 0 else 0}", "ma", window)
            cache.set(cache_key, ma_series)
    
    return ma_series


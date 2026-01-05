"""Moving average indicator calculations."""

from typing import Optional

import numpy as np
import pandas as pd

from .cache import create_cache_key_for_series, get_cache


def ma(series: pd.Series, window: int, use_cache: bool = True, symbol: Optional[str] = None) -> pd.Series:
    """Compute moving average with proper NaN handling.

    Optimized version that uses caching and vectorized operations.

    Args:
        series: Price series (typically close prices)
        window: Moving average period (e.g., 20, 50, 200)
        use_cache: Whether to use caching (default True)
        symbol: Optional symbol name for better cache keys (enables cross-strategy caching)

    Returns:
        Series with moving average values. Returns NaN for all dates
        before the window is filled (first window-1 values are NaN).
        Never forward-fills NaN values.

    Example:
        >>> close = pd.Series([100, 101, 102, 103, 104])
        >>> ma20 = ma(close, window=20, symbol='AAPL')
        >>> # First 19 values will be NaN, 20th value is mean of first 20
    """
    if len(series) == 0:
        return pd.Series(dtype=float, index=series.index)

    # Check cache if enabled
    if use_cache:
        cache = get_cache()
        if cache is not None:
            # Create proper cache key using hash-based approach
            cache_key = create_cache_key_for_series(series, "ma", window, symbol=symbol)
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

    # Optimized: Use vectorized rolling mean (already fast in pandas)
    # min_periods=window ensures NaN until window values are available
    # This automatically sets first window-1 values to NaN
    ma_series = series.rolling(window=window, min_periods=window).mean()

    # Ensure first window-1 values are NaN (defensive check)
    # When min_periods=window, pandas sets first window-1 to NaN, but we verify
    # Make a copy to avoid SettingWithCopyWarning
    ma_series = ma_series.copy()

    # Handle NaN assignment based on series length and window
    if len(ma_series) == 0:
        # Empty series - already handled above
        pass
    elif len(ma_series) < window:
        # If series is shorter than window, all values should be NaN
        ma_series.iloc[:] = np.nan
    elif window > 1 and len(ma_series) >= window:
        # Explicitly ensure first window-1 values are NaN
        # This handles edge cases where rolling might not set them correctly
        # Use iloc to ensure we're modifying the copy, not a view
        if window - 1 > 0:
            ma_series.iloc[: window - 1] = np.nan
    # If window == 1, no values should be NaN (window-1 = 0)

    # Cache result if enabled
    if use_cache:
        cache = get_cache()
        if cache is not None:
            cache_key = create_cache_key_for_series(series, "ma", window, symbol=symbol)
            cache.set(cache_key, ma_series)

    return ma_series

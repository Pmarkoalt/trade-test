"""Breakout level indicators (highest close over window)."""

import pandas as pd
import numpy as np
from .cache import get_cache


def highest_close(close: pd.Series, window: int, use_cache: bool = True) -> pd.Series:
    """Compute highest close over window, EXCLUDING today's close.
    
    This is critical for avoiding lookahead bias. We use the highest close
    over the prior N days, excluding today, so that signals can be generated
    at the close of day t using only information available at that time.
    
    Args:
        close: Series of closing prices
        window: Lookback window (e.g., 20 or 55)
    
    Returns:
        Series with highest close values. Returns NaN until window is filled.
        The value at index t represents the highest close from t-window to t-1
        (excluding t).
    
    Example:
        >>> close = pd.Series([100, 101, 102, 103, 104, 105])
        >>> highest_20d = highest_close(close, window=20)
        >>> # First 20 values will be NaN
        >>> # Value at index 20 is max(close[0:20]), excluding close[20]
    """
    if len(close) == 0:
        return pd.Series(dtype=float, index=close.index)
    
    # Check cache if enabled
    if use_cache:
        cache = get_cache()
        if cache is not None:
            cache_key = (f"highest_close_{id(close)}_{len(close)}_{close.iloc[-1] if len(close) > 0 else 0}", "highest_close", window)
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
    
    # Shift by 1 to exclude today, then compute rolling max
    # This ensures we use only prior N days (no lookahead)
    close_shifted = close.shift(1)
    highest = close_shifted.rolling(window=window, min_periods=window).max()
    
    # Explicitly set NaN for first window values (insufficient lookback)
    if len(highest) > 0 and len(highest) >= window:
        highest.iloc[:window] = np.nan
    
    # Cache result if enabled
    if use_cache:
        cache = get_cache()
        if cache is not None:
            cache_key = (f"highest_close_{id(close)}_{len(close)}_{close.iloc[-1] if len(close) > 0 else 0}", "highest_close", window)
            cache.set(cache_key, highest)
    
    return highest


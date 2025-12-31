"""Moving average indicator calculations."""

import pandas as pd
import numpy as np
from typing import Union


def ma(series: pd.Series, window: int) -> pd.Series:
    """Compute moving average with proper NaN handling.
    
    Args:
        series: Price series (typically close prices)
        window: Moving average period (e.g., 20, 50, 200)
    
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
    
    # Compute rolling mean
    # min_periods=window ensures NaN until window values are available
    ma_series = series.rolling(window=window, min_periods=window).mean()
    
    # Explicitly set NaN for insufficient lookback (first window-1 values)
    # This ensures we never use partial windows
    # Note: pandas rolling with min_periods already does this, but we're explicit
    if len(ma_series) > 0 and len(ma_series) >= window:
        ma_series.iloc[:window-1] = np.nan
    
    return ma_series


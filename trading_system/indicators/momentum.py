"""Momentum indicators (Rate of Change)."""

import pandas as pd
import numpy as np


def roc(close: pd.Series, window: int = 60) -> pd.Series:
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
    
    # Shift by window to get close[t-window]
    close_shifted = close.shift(window)
    
    # Calculate ROC: (close[t] / close[t-window]) - 1
    roc_series = (close / close_shifted) - 1
    
    # Explicitly set NaN for first window values (where close[t-window] doesn't exist)
    if len(roc_series) > 0 and len(roc_series) >= window:
        roc_series.iloc[:window] = np.nan
    
    return roc_series


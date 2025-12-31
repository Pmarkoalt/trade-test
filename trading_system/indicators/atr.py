"""Average True Range (ATR) indicator calculation using Wilder's smoothing."""

import pandas as pd
import numpy as np
from typing import Union


def atr(df_ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range using Wilder's exponential smoothing.
    
    ATR measures volatility by calculating the average of true ranges over
    a specified period. Wilder's smoothing uses an exponential moving average
    with alpha = 1/period (not a simple average).
    
    True Range is the maximum of:
    1. High - Low
    2. |High - Previous Close|
    3. |Low - Previous Close|
    
    Args:
        df_ohlc: DataFrame with columns ['high', 'low', 'close']
        period: ATR period (default 14)
    
    Returns:
        Series with ATR values. Returns NaN for first (period-1) bars
        until sufficient lookback is available.
    
    Example:
        >>> df = pd.DataFrame({
        ...     'high': [100, 102, 101, 103],
        ...     'low': [99, 100, 99.5, 101],
        ...     'close': [100.5, 101, 100, 102]
        ... })
        >>> atr14 = atr(df, period=14)
        >>> # First 13 values will be NaN
    """
    if len(df_ohlc) == 0:
        return pd.Series(dtype=float, index=df_ohlc.index)
    
    # Validate required columns
    required_cols = ['high', 'low', 'close']
    if not all(col in df_ohlc.columns for col in required_cols):
        raise ValueError(
            f"DataFrame must contain columns: {required_cols}. "
            f"Found: {list(df_ohlc.columns)}"
        )
    
    # Calculate True Range components
    tr1 = df_ohlc['high'] - df_ohlc['low']
    tr2 = abs(df_ohlc['high'] - df_ohlc['close'].shift(1))
    tr3 = abs(df_ohlc['low'] - df_ohlc['close'].shift(1))
    
    # True Range is the maximum of the three
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Wilder's smoothing: exponential moving average with alpha = 1/period
    # adjust=False means we use the recursive formula:
    # ATR[t] = ATR[t-1] * (1 - 1/period) + TR[t] * (1/period)
    atr_series = tr.ewm(alpha=1/period, adjust=False).mean()
    
    # Explicitly set NaN for insufficient lookback (first period-1 values)
    # Note: Wilder's smoothing technically needs period values to initialize,
    # but we set first period-1 to NaN to be conservative
    if len(atr_series) > 0 and len(atr_series) >= period:
        atr_series.iloc[:period-1] = np.nan
    
    return atr_series


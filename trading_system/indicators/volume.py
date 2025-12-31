"""Volume indicators (Average Dollar Volume)."""

import pandas as pd
import numpy as np


def adv(dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    """Compute Average Dollar Volume (ADV) over specified window.
    
    ADV is the average dollar volume over a lookback period, used for
    capacity constraints. Dollar volume = close * volume.
    
    Args:
        dollar_volume: Series of dollar volumes (close * volume)
        window: Lookback window in days (default 20)
    
    Returns:
        Series with ADV values. Returns NaN until window is filled.
    
    Example:
        >>> dollar_vol = pd.Series([1e6, 1.1e6, 1.2e6, 1.3e6])
        >>> adv20 = adv(dollar_vol, window=20)
        >>> # First 19 values will be NaN
    """
    if len(dollar_volume) == 0:
        return pd.Series(dtype=float, index=dollar_volume.index)
    
    # Compute rolling mean of dollar volume
    adv_series = dollar_volume.rolling(window=window, min_periods=window).mean()
    
    # Explicitly set NaN for insufficient lookback (first window-1 values)
    if len(adv_series) > 0 and len(adv_series) >= window:
        adv_series.iloc[:window-1] = np.nan
    
    return adv_series


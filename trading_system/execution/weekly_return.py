"""Weekly return calculation for stress multiplier detection."""

from typing import Optional
import pandas as pd
from ..data.calendar import get_trading_days


def compute_weekly_return(
    benchmark_bars: pd.DataFrame,
    current_date: pd.Timestamp,
    asset_class: str
) -> float:
    """
    Compute weekly return for stress multiplier calculation.
    
    Rules:
    - Equities: Last 5 trading days (Mon-Fri, skip weekends/holidays)
    - Crypto: Last 7 calendar days (UTC, continuous)
    - Formula: (close[t] / close[t-N]) - 1
    
    Args:
        benchmark_bars: DataFrame with 'close' column and date index
        current_date: Current date (end date for calculation)
        asset_class: "equity" or "crypto"
    
    Returns:
        Weekly return as decimal (e.g., -0.03 for -3%)
        Returns 0.0 if insufficient data
    
    Example:
        >>> spy_bars = pd.DataFrame({'close': [100, 101, 102, ...]}, index=dates)
        >>> weekly_ret = compute_weekly_return(spy_bars, pd.Timestamp("2024-01-15"), "equity")
        >>> # Returns (close[t] / close[t-5] - 1) for last 5 trading days
    """
    if current_date not in benchmark_bars.index:
        return 0.0  # Missing data
    
    if asset_class == "equity":
        # Get last 5 trading days
        trading_days = get_trading_days(benchmark_bars.index, current_date, lookback=5)
        
        if len(trading_days) < 5:
            return 0.0  # Insufficient data
        
        start_date = trading_days[0]
        end_date = trading_days[-1]
        
    else:  # crypto
        # Get last 7 calendar days
        end_date = current_date
        start_date = end_date - pd.Timedelta(days=6)  # 7 days total (including end_date)
        
        # Ensure both dates are in the index
        if start_date not in benchmark_bars.index or end_date not in benchmark_bars.index:
            return 0.0
    
    # Get closes
    try:
        start_close = benchmark_bars.loc[start_date, 'close']
        end_close = benchmark_bars.loc[end_date, 'close']
    except (KeyError, IndexError):
        return 0.0  # Missing data
    
    if start_close <= 0 or end_close <= 0:
        return 0.0
    
    weekly_return = (end_close / start_close) - 1
    return float(weekly_return)


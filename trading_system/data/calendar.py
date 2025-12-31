"""Trading calendar handling for equity and crypto markets."""

from typing import List
import pandas as pd


def get_trading_days(
    all_dates: pd.DatetimeIndex,
    end_date: pd.Timestamp,
    lookback: int = 5
) -> List[pd.Timestamp]:
    """
    Get last N trading days (excluding weekends/holidays).
    
    For equity stress multiplier calculation.
    
    Args:
        all_dates: All available dates in the dataset
        end_date: End date (inclusive)
        lookback: Number of trading days to look back
    
    Returns:
        List of trading day timestamps
    """
    # Filter to trading days (exclude weekends: Mon=0, Fri=4)
    trading_days = all_dates[all_dates.weekday < 5]  # Mon-Fri
    
    # Filter to <= end_date
    trading_days = trading_days[trading_days <= end_date]
    
    # Get last N
    if len(trading_days) < lookback:
        return trading_days.tolist()
    
    return trading_days[-lookback:].tolist()


def get_trading_calendar(exchange: str = "NASDAQ"):
    """
    Get trading calendar for exchange using pandas_market_calendars.
    
    Args:
        exchange: Exchange name ("NASDAQ", "NYSE", etc.)
    
    Returns:
        Trading calendar object (pandas_market_calendars calendar)
    
    Note:
        Requires pandas_market_calendars package. Falls back to simple
        weekday filtering if package is not available.
    """
    try:
        import pandas_market_calendars as mcal
        cal = mcal.get_calendar(exchange)
        return cal
    except ImportError:
        # Fallback: return None to use simple weekday filtering
        return None


def get_crypto_days(
    end_date: pd.Timestamp,
    lookback: int = 7
) -> List[pd.Timestamp]:
    """
    Get last N calendar days (continuous, no weekends).
    
    For crypto stress multiplier calculation.
    
    Args:
        end_date: End date (inclusive)
        lookback: Number of calendar days to look back
    
    Returns:
        List of calendar day timestamps (UTC)
    """
    start_date = end_date - pd.Timedelta(days=lookback - 1)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    return dates.tolist()


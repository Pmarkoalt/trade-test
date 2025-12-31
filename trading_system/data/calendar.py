"""Trading calendar handling for equity and crypto markets."""

from typing import Any, List, Optional

import pandas as pd


def get_trading_days(all_dates: pd.DatetimeIndex, end_date: pd.Timestamp, lookback: int = 5) -> List[pd.Timestamp]:
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


def get_trading_calendar(exchange: str = "NASDAQ") -> Optional[Any]:
    """
    Get trading calendar for exchange using pandas_market_calendars.

    Args:
        exchange: Exchange name ("NASDAQ", "NYSE", etc.)

    Returns:
        Trading calendar object (pandas_market_calendars calendar) or None if package not available

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


def get_next_trading_day(date: pd.Timestamp, asset_class: str = "equity") -> pd.Timestamp:
    """
    Get the next trading day after the given date.

    Args:
        date: Current date
        asset_class: "equity" (skip weekends) or "crypto" (next calendar day)

    Returns:
        Next trading day timestamp
    """
    next_day = date + pd.Timedelta(days=1)

    if asset_class == "crypto":
        # Crypto trades 24/7, next calendar day
        return next_day

    # Equity: skip weekends (Saturday=5, Sunday=6)
    while next_day.weekday() >= 5:
        next_day = next_day + pd.Timedelta(days=1)

    return next_day


def get_crypto_days(end_date: pd.Timestamp, lookback: int = 7) -> List[pd.Timestamp]:
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
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    return dates.tolist()

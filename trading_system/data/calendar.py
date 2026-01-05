"""Trading calendar handling for equity and crypto markets."""

import logging
from typing import Any, Dict, List, Optional, Set, cast

import pandas as pd

logger = logging.getLogger(__name__)

# US Market Holidays (NYSE/NASDAQ) - standard observed holidays
US_MARKET_HOLIDAYS_2024: Set[str] = {
    "2024-01-01",  # New Year's Day
    "2024-01-15",  # MLK Day
    "2024-02-19",  # Presidents Day
    "2024-03-29",  # Good Friday
    "2024-05-27",  # Memorial Day
    "2024-06-19",  # Juneteenth
    "2024-07-04",  # Independence Day
    "2024-09-02",  # Labor Day
    "2024-11-28",  # Thanksgiving
    "2024-12-25",  # Christmas
}

US_MARKET_HOLIDAYS_2025: Set[str] = {
    "2025-01-01",  # New Year's Day
    "2025-01-20",  # MLK Day
    "2025-02-17",  # Presidents Day
    "2025-04-18",  # Good Friday
    "2025-05-26",  # Memorial Day
    "2025-06-19",  # Juneteenth
    "2025-07-04",  # Independence Day
    "2025-09-01",  # Labor Day
    "2025-11-27",  # Thanksgiving
    "2025-12-25",  # Christmas
}

US_MARKET_HOLIDAYS_2026: Set[str] = {
    "2026-01-01",  # New Year's Day
    "2026-01-19",  # MLK Day
    "2026-02-16",  # Presidents Day
    "2026-04-03",  # Good Friday
    "2026-05-25",  # Memorial Day
    "2026-06-19",  # Juneteenth
    "2026-07-03",  # Independence Day (observed)
    "2026-09-07",  # Labor Day
    "2026-11-26",  # Thanksgiving
    "2026-12-25",  # Christmas
}

# Combined holidays
US_MARKET_HOLIDAYS: Set[str] = US_MARKET_HOLIDAYS_2024 | US_MARKET_HOLIDAYS_2025 | US_MARKET_HOLIDAYS_2026


def is_weekend(dt: pd.Timestamp) -> bool:
    """Check if date is a weekend (Saturday=5, Sunday=6)."""
    return dt.dayofweek >= 5


def is_us_market_holiday(dt: pd.Timestamp) -> bool:
    """Check if date is a US market holiday."""
    date_str = dt.strftime("%Y-%m-%d")
    return date_str in US_MARKET_HOLIDAYS


def is_trading_day(dt: pd.Timestamp, asset_class: str = "equity") -> bool:
    """
    Check if a date is a valid trading day.

    Args:
        dt: Date to check
        asset_class: "equity" or "crypto"

    Returns:
        True if the market is open on this day
    """
    if asset_class == "crypto":
        return True  # Crypto trades 24/7/365

    if is_weekend(dt):
        return False

    if is_us_market_holiday(dt):
        return False

    return True


def get_expected_trading_days(
    start_date: pd.Timestamp, end_date: pd.Timestamp, asset_class: str = "equity"
) -> pd.DatetimeIndex:
    """
    Get all expected trading days in a date range.

    Args:
        start_date: Start of range
        end_date: End of range
        asset_class: "equity" or "crypto"

    Returns:
        DatetimeIndex of expected trading days
    """
    if asset_class == "crypto":
        return pd.date_range(start=start_date, end=end_date, freq="D")

    # Equity: business days minus holidays
    all_bdays = pd.date_range(start=start_date, end=end_date, freq="B")
    trading_days = [d for d in all_bdays if not is_us_market_holiday(d)]
    return pd.DatetimeIndex(trading_days)


def detect_missing_trading_days(df: pd.DataFrame, symbol: str, asset_class: str = "equity") -> Dict[str, Any]:
    """
    Detect missing trading days in data (excludes weekends/holidays for equity).

    Args:
        df: DataFrame with date index
        symbol: Symbol name for logging
        asset_class: "equity" or "crypto"

    Returns:
        Dictionary with:
        - missing_dates: List of missing trading days
        - consecutive_gaps: List of (start, end) tuples
        - gap_lengths: List of gap lengths in trading days
        - is_healthy: True if no gaps >= 2 trading days
    """
    if len(df) == 0:
        return {"missing_dates": [], "consecutive_gaps": [], "gap_lengths": [], "is_healthy": True}

    start_date = df.index.min()
    end_date = df.index.max()

    # Get expected trading days
    expected_days = get_expected_trading_days(start_date, end_date, asset_class)

    # Find missing trading days
    actual_days = set(df.index)
    missing_dates = [d for d in expected_days if d not in actual_days]

    # Find consecutive gaps
    consecutive_gaps: List[tuple] = []
    gap_lengths: List[int] = []

    if missing_dates:
        sorted_missing = sorted(missing_dates)
        gap_start = sorted_missing[0]
        current_gap = [gap_start]

        for i in range(1, len(sorted_missing)):
            prev_date = sorted_missing[i - 1]
            curr_date = sorted_missing[i]

            # Get next expected trading day after prev_date
            next_expected = get_next_trading_day(prev_date, asset_class)

            if curr_date == next_expected:
                current_gap.append(curr_date)
            else:
                consecutive_gaps.append((gap_start, current_gap[-1]))
                gap_lengths.append(len(current_gap))
                gap_start = curr_date
                current_gap = [curr_date]

        # Last gap
        consecutive_gaps.append((gap_start, current_gap[-1]))
        gap_lengths.append(len(current_gap))

    # Data is unhealthy if any gap >= 2 consecutive trading days
    is_healthy = all(length < 2 for length in gap_lengths)

    return {
        "missing_dates": missing_dates,
        "consecutive_gaps": consecutive_gaps,
        "gap_lengths": gap_lengths,
        "is_healthy": is_healthy,
    }


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
        return cast(List[pd.Timestamp], trading_days.tolist())

    return cast(List[pd.Timestamp], trading_days[-lookback:].tolist())


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
    return cast(List[pd.Timestamp], dates.tolist())

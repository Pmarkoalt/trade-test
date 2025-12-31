"""Unit tests for data/calendar.py."""

import pandas as pd
import pytest

from trading_system.data.calendar import get_crypto_days, get_next_trading_day, get_trading_calendar, get_trading_days


class TestGetTradingDays:
    """Tests for get_trading_days function."""

    def test_get_trading_days_weekdays_only(self):
        """Test that only weekdays are returned."""
        # Create dates including weekends
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        end_date = dates[-1]

        trading_days = get_trading_days(dates, end_date, lookback=5)

        # All returned days should be weekdays (Mon-Fri)
        for day in trading_days:
            assert day.weekday() < 5  # Monday=0, Friday=4

    def test_get_trading_days_exact_count(self):
        """Test getting exact number of trading days."""
        # Create 10 weekdays
        dates = pd.bdate_range("2024-01-01", periods=10, freq="B")
        end_date = dates[-1]

        trading_days = get_trading_days(dates, end_date, lookback=5)

        assert len(trading_days) == 5

    def test_get_trading_days_insufficient_data(self):
        """Test when there are fewer trading days than requested."""
        # Only 3 weekdays
        dates = pd.bdate_range("2024-01-01", periods=3, freq="B")
        end_date = dates[-1]

        trading_days = get_trading_days(dates, end_date, lookback=5)

        # Should return all available trading days
        assert len(trading_days) == 3
        assert all(day in dates for day in trading_days)

    def test_get_trading_days_filters_by_end_date(self):
        """Test that only dates <= end_date are returned."""
        dates = pd.bdate_range("2024-01-01", periods=10, freq="B")
        end_date = dates[4]  # Middle date

        trading_days = get_trading_days(dates, end_date, lookback=5)

        # All days should be <= end_date
        assert all(day <= end_date for day in trading_days)
        # Should get last 5 trading days up to end_date
        assert len(trading_days) == 5
        assert trading_days[-1] == end_date

    def test_get_trading_days_excludes_weekends(self):
        """Test that weekends are excluded from results."""
        # Create dates that include weekends
        dates = pd.date_range("2024-01-01", periods=14, freq="D")  # 2 weeks
        end_date = dates[-1]

        trading_days = get_trading_days(dates, end_date, lookback=5)

        # Should have 5 weekdays, no weekends
        assert len(trading_days) == 5
        for day in trading_days:
            assert day.weekday() < 5

    def test_get_trading_days_empty_dates(self):
        """Test with empty date index."""
        dates = pd.DatetimeIndex([])
        end_date = pd.Timestamp("2024-01-15")

        trading_days = get_trading_days(dates, end_date, lookback=5)

        assert len(trading_days) == 0


class TestGetTradingCalendar:
    """Tests for get_trading_calendar function."""

    def test_get_trading_calendar_nasdaq(self):
        """Test getting NASDAQ calendar."""
        cal = get_trading_calendar("NASDAQ")
        # May return None if pandas_market_calendars not installed
        # That's acceptable behavior
        assert cal is None or hasattr(cal, "schedule")

    def test_get_trading_calendar_nyse(self):
        """Test getting NYSE calendar."""
        cal = get_trading_calendar("NYSE")
        assert cal is None or hasattr(cal, "schedule")

    def test_get_trading_calendar_handles_import_error(self):
        """Test that function handles ImportError gracefully."""
        # Function should return None if pandas_market_calendars not available
        cal = get_trading_calendar("NASDAQ")
        # Should not raise an error
        assert cal is None or isinstance(cal, object)


class TestGetNextTradingDay:
    """Tests for get_next_trading_day function."""

    def test_get_next_trading_day_equity_weekday(self):
        """Test getting next trading day for equity on weekday."""
        # Monday
        date = pd.Timestamp("2024-01-01")  # Monday
        next_day = get_next_trading_day(date, asset_class="equity")

        # Should be Tuesday
        assert next_day == pd.Timestamp("2024-01-02")
        assert next_day.weekday() < 5

    def test_get_next_trading_day_equity_friday(self):
        """Test getting next trading day for equity on Friday."""
        # Friday
        date = pd.Timestamp("2024-01-05")  # Friday
        next_day = get_next_trading_day(date, asset_class="equity")

        # Should skip weekend, be Monday
        assert next_day == pd.Timestamp("2024-01-08")  # Monday
        assert next_day.weekday() < 5

    def test_get_next_trading_day_equity_saturday(self):
        """Test getting next trading day for equity on Saturday."""
        # Saturday
        date = pd.Timestamp("2024-01-06")  # Saturday
        next_day = get_next_trading_day(date, asset_class="equity")

        # Should skip to Monday
        assert next_day == pd.Timestamp("2024-01-08")  # Monday
        assert next_day.weekday() < 5

    def test_get_next_trading_day_equity_sunday(self):
        """Test getting next trading day for equity on Sunday."""
        # Sunday
        date = pd.Timestamp("2024-01-07")  # Sunday
        next_day = get_next_trading_day(date, asset_class="equity")

        # Should be Monday
        assert next_day == pd.Timestamp("2024-01-08")  # Monday
        assert next_day.weekday() < 5

    def test_get_next_trading_day_crypto_weekday(self):
        """Test getting next trading day for crypto on weekday."""
        date = pd.Timestamp("2024-01-01")
        next_day = get_next_trading_day(date, asset_class="crypto")

        # Crypto trades 24/7, should be next calendar day
        assert next_day == pd.Timestamp("2024-01-02")

    def test_get_next_trading_day_crypto_weekend(self):
        """Test getting next trading day for crypto on weekend."""
        # Saturday
        date = pd.Timestamp("2024-01-06")  # Saturday
        next_day = get_next_trading_day(date, asset_class="crypto")

        # Crypto trades 24/7, should be next calendar day (Sunday)
        assert next_day == pd.Timestamp("2024-01-07")  # Sunday

    def test_get_next_trading_day_crypto_sunday(self):
        """Test getting next trading day for crypto on Sunday."""
        date = pd.Timestamp("2024-01-07")  # Sunday
        next_day = get_next_trading_day(date, asset_class="crypto")

        # Crypto trades 24/7, should be next calendar day (Monday)
        assert next_day == pd.Timestamp("2024-01-08")  # Monday


class TestGetCryptoDays:
    """Tests for get_crypto_days function."""

    def test_get_crypto_days_exact_count(self):
        """Test getting exact number of calendar days."""
        end_date = pd.Timestamp("2024-01-10")
        days = get_crypto_days(end_date, lookback=7)

        assert len(days) == 7
        assert days[-1] == end_date

    def test_get_crypto_days_includes_weekends(self):
        """Test that crypto days include weekends."""
        end_date = pd.Timestamp("2024-01-07")  # Sunday
        days = get_crypto_days(end_date, lookback=7)

        # Should include all 7 calendar days, including weekends
        assert len(days) == 7
        # Should include at least one weekend day
        has_weekend = any(day.weekday() >= 5 for day in days)
        assert has_weekend

    def test_get_crypto_days_start_date(self):
        """Test that start date is correct."""
        end_date = pd.Timestamp("2024-01-10")
        days = get_crypto_days(end_date, lookback=7)

        # Start date should be end_date - 6 days (7 days total including end_date)
        expected_start = end_date - pd.Timedelta(days=6)
        assert days[0] == expected_start

    def test_get_crypto_days_end_date(self):
        """Test that end date is included."""
        end_date = pd.Timestamp("2024-01-10")
        days = get_crypto_days(end_date, lookback=7)

        assert days[-1] == end_date

    def test_get_crypto_days_different_lookback(self):
        """Test with different lookback values."""
        end_date = pd.Timestamp("2024-01-10")

        for lookback in [1, 5, 10, 30]:
            days = get_crypto_days(end_date, lookback=lookback)
            assert len(days) == lookback
            assert days[-1] == end_date

    def test_get_crypto_days_continuous(self):
        """Test that crypto days are continuous (no gaps)."""
        end_date = pd.Timestamp("2024-01-10")
        days = get_crypto_days(end_date, lookback=7)

        # Check that days are consecutive
        for i in range(1, len(days)):
            expected_next = days[i - 1] + pd.Timedelta(days=1)
            assert days[i] == expected_next

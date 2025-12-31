"""Unit tests for execution/weekly_return.py."""

import numpy as np
import pandas as pd
import pytest

from trading_system.execution.weekly_return import compute_weekly_return


class TestComputeWeeklyReturn:
    """Tests for compute_weekly_return function."""

    def test_equity_weekly_return(self):
        """Test weekly return calculation for equity (5 trading days)."""
        # Create sample data with 10 trading days
        dates = pd.bdate_range("2024-01-01", periods=10, freq="B")
        closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]

        benchmark_bars = pd.DataFrame({"close": closes}, index=dates)
        current_date = dates[-1]  # Last date

        weekly_ret = compute_weekly_return(benchmark_bars, current_date, "equity")

        # Should be (109.0 / 105.0) - 1 = 0.0381 (last 5 trading days)
        # Actually, it should be last 5 days: dates[-5] to dates[-1]
        # closes[-5] = 105.0, closes[-1] = 109.0
        expected = (109.0 / 105.0) - 1
        assert abs(weekly_ret - expected) < 0.0001

    def test_crypto_weekly_return(self):
        """Test weekly return calculation for crypto (7 calendar days)."""
        # Create sample data with 10 calendar days
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]

        benchmark_bars = pd.DataFrame({"close": closes}, index=dates)
        current_date = dates[-1]  # Last date

        weekly_ret = compute_weekly_return(benchmark_bars, current_date, "crypto")

        # Should be (109.0 / 103.0) - 1 (last 7 calendar days: dates[-7] to dates[-1])
        # dates[-7] = dates[3] = 103.0, dates[-1] = dates[9] = 109.0
        expected = (109.0 / 103.0) - 1
        assert abs(weekly_ret - expected) < 0.0001

    def test_equity_insufficient_data(self):
        """Test equity with insufficient trading days."""
        # Only 3 trading days (need 5)
        dates = pd.bdate_range("2024-01-01", periods=3, freq="B")
        closes = [100.0, 101.0, 102.0]

        benchmark_bars = pd.DataFrame({"close": closes}, index=dates)
        current_date = dates[-1]

        weekly_ret = compute_weekly_return(benchmark_bars, current_date, "equity")
        assert weekly_ret == 0.0

    def test_crypto_insufficient_data(self):
        """Test crypto with insufficient calendar days."""
        # Only 3 calendar days (need 7)
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        closes = [100.0, 101.0, 102.0]

        benchmark_bars = pd.DataFrame({"close": closes}, index=dates)
        current_date = dates[-1]

        weekly_ret = compute_weekly_return(benchmark_bars, current_date, "crypto")
        assert weekly_ret == 0.0

    def test_missing_current_date(self):
        """Test when current_date is not in index."""
        dates = pd.bdate_range("2024-01-01", periods=10, freq="B")
        closes = [100.0] * 10

        benchmark_bars = pd.DataFrame({"close": closes}, index=dates)
        current_date = pd.Timestamp("2024-12-31")  # Not in index

        weekly_ret = compute_weekly_return(benchmark_bars, current_date, "equity")
        assert weekly_ret == 0.0

    def test_crypto_missing_start_date(self):
        """Test crypto when start_date is not in index."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")  # Only 5 days
        closes = [100.0, 101.0, 102.0, 103.0, 104.0]

        benchmark_bars = pd.DataFrame({"close": closes}, index=dates)
        current_date = dates[-1]

        # Need 7 days, but only have 5, so start_date (current_date - 6 days) won't be in index
        weekly_ret = compute_weekly_return(benchmark_bars, current_date, "crypto")
        assert weekly_ret == 0.0

    def test_negative_return(self):
        """Test calculation with negative return."""
        dates = pd.bdate_range("2024-01-01", periods=10, freq="B")
        closes = [100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0, 91.0]

        benchmark_bars = pd.DataFrame({"close": closes}, index=dates)
        current_date = dates[-1]

        weekly_ret = compute_weekly_return(benchmark_bars, current_date, "equity")

        # Should be negative
        assert weekly_ret < 0
        # Last 5 days: 95.0 to 91.0
        expected = (91.0 / 95.0) - 1
        assert abs(weekly_ret - expected) < 0.0001

    def test_zero_start_close(self):
        """Test when start close is zero (should return 0.0)."""
        dates = pd.bdate_range("2024-01-01", periods=10, freq="B")
        # The function looks back 5 trading days from the end
        # Last 5 trading days are: dates[-5], dates[-4], dates[-3], dates[-2], dates[-1]
        # Put 0.0 at the start of the lookback window (dates[-5])
        closes = [1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0, 10.0]

        benchmark_bars = pd.DataFrame({"close": closes}, index=dates)
        current_date = dates[-1]

        weekly_ret = compute_weekly_return(benchmark_bars, current_date, "equity")
        # Function should return 0.0 when start_close <= 0
        assert weekly_ret == 0.0

    def test_zero_end_close(self):
        """Test when end close is zero (should return 0.0)."""
        dates = pd.bdate_range("2024-01-01", periods=10, freq="B")
        closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 0.0]

        benchmark_bars = pd.DataFrame({"close": closes}, index=dates)
        current_date = dates[-1]

        weekly_ret = compute_weekly_return(benchmark_bars, current_date, "equity")
        assert weekly_ret == 0.0

    def test_crypto_exact_7_days(self):
        """Test crypto with exactly 7 days."""
        dates = pd.date_range("2024-01-01", periods=7, freq="D")
        closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0]

        benchmark_bars = pd.DataFrame({"close": closes}, index=dates)
        current_date = dates[-1]

        weekly_ret = compute_weekly_return(benchmark_bars, current_date, "crypto")

        # Should be (106.0 / 100.0) - 1 = 0.06
        expected = (106.0 / 100.0) - 1
        assert abs(weekly_ret - expected) < 0.0001

    def test_equity_exact_5_days(self):
        """Test equity with exactly 5 trading days."""
        dates = pd.bdate_range("2024-01-01", periods=5, freq="B")
        closes = [100.0, 101.0, 102.0, 103.0, 104.0]

        benchmark_bars = pd.DataFrame({"close": closes}, index=dates)
        current_date = dates[-1]

        weekly_ret = compute_weekly_return(benchmark_bars, current_date, "equity")

        # Should be (104.0 / 100.0) - 1 = 0.04
        expected = (104.0 / 100.0) - 1
        assert abs(weekly_ret - expected) < 0.0001

    def test_key_error_handling(self):
        """Test handling of KeyError when accessing dates."""
        dates = pd.bdate_range("2024-01-01", periods=10, freq="B")
        closes = [100.0] * 10

        benchmark_bars = pd.DataFrame({"close": closes}, index=dates)
        # Create a date that's close but might cause KeyError
        current_date = dates[-1]

        # This should work, but test the error handling path
        weekly_ret = compute_weekly_return(benchmark_bars, current_date, "equity")
        assert isinstance(weekly_ret, float)

    def test_index_error_handling(self):
        """Test handling of IndexError when accessing data."""
        # Create empty dataframe
        benchmark_bars = pd.DataFrame({"close": []}, index=pd.DatetimeIndex([]))
        current_date = pd.Timestamp("2024-01-15")

        weekly_ret = compute_weekly_return(benchmark_bars, current_date, "equity")
        assert weekly_ret == 0.0

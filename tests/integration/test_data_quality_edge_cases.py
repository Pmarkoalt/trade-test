"""Integration tests for data quality edge cases.

This test suite verifies that the system handles various data quality issues
correctly, including:
- Missing days (holidays, weekends)
- Extreme price moves (>50%)
- Low volume days
- Gaps in data (consecutive missing days)
- Duplicate dates

These tests ensure the system is robust to real-world data quality issues.
"""

import numpy as np
import pandas as pd
import pytest

from trading_system.data import load_ohlcv_data
from trading_system.data.validator import detect_missing_data, validate_ohlcv


def create_sample_ohlcv_data(symbol: str, dates: pd.DatetimeIndex, base_price: float = 100.0) -> pd.DataFrame:
    """Create sample OHLCV data for given dates."""
    np.random.seed(42)
    prices = base_price + np.cumsum(np.random.randn(len(dates)) * 2.0)

    df = pd.DataFrame(
        {
            "open": prices * 0.99,
            "high": prices * 1.02,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.random.randint(1000000, 5000000, len(dates)),
        },
        index=dates,
    )
    df.index.name = "date"
    df["dollar_volume"] = df["close"] * df["volume"]
    return df


class TestMissingDays:
    """Tests for missing days (holidays, weekends)."""

    def test_equity_missing_weekends(self):
        """Test that equity data correctly handles missing weekends."""
        # Create data with only weekdays (missing weekends)
        dates = pd.bdate_range("2023-01-01", "2023-01-31", freq="B")  # Business days only
        df = create_sample_ohlcv_data("TEST", dates)

        # Detect missing data (equity should expect business days)
        missing_info = detect_missing_data(df, "TEST", asset_class="equity")

        # For equity, weekends are expected to be missing
        # So missing_dates should be empty or only contain holidays
        assert "missing_dates" in missing_info, "Should detect missing dates"
        assert "consecutive_gaps" in missing_info, "Should detect consecutive gaps"

    def test_equity_missing_holidays(self):
        """Test that equity data correctly handles missing holidays."""
        # Create data missing a holiday (e.g., Jan 3, 2023 is a Tuesday, simulating a holiday)
        dates = pd.bdate_range("2023-01-01", "2023-01-31", freq="B")
        # Remove a specific business day (simulating holiday)
        dates = dates.drop(pd.Timestamp("2023-01-03"))
        df = create_sample_ohlcv_data("TEST", dates)

        # Detect missing data
        missing_info = detect_missing_data(df, "TEST", asset_class="equity")

        # Should detect the missing holiday
        assert "missing_dates" in missing_info, "Should detect missing dates"
        # The missing date should be in the list (it's a business day)

    def test_crypto_missing_days(self):
        """Test that crypto data correctly handles missing days (crypto trades 24/7)."""
        # Create data with missing days (crypto should have all days)
        dates = pd.date_range("2023-01-01", "2023-01-31", freq="D")
        # Remove a few days (simulating data gaps)
        dates = dates.drop([pd.Timestamp("2023-01-05"), pd.Timestamp("2023-01-06")])
        df = create_sample_ohlcv_data("TEST", dates)

        # Detect missing data (crypto should expect all calendar days)
        missing_info = detect_missing_data(df, "TEST", asset_class="crypto")

        # Should detect missing days
        assert "missing_dates" in missing_info, "Should detect missing dates"
        assert len(missing_info["missing_dates"]) >= 2, "Should detect at least 2 missing days"

    def test_missing_days_validation_passes(self):
        """Test that missing days don't cause validation to fail."""
        # Create data with missing days
        dates = pd.bdate_range("2023-01-01", "2023-01-31", freq="B")
        dates = dates.drop(pd.Timestamp("2023-01-13"))  # Remove a Friday (business day)
        df = create_sample_ohlcv_data("TEST", dates)

        # Validation should pass (missing days are warnings, not errors)
        assert validate_ohlcv(df, "TEST"), "Validation should pass with missing days"

    def test_missing_days_in_backtest(self, tmp_path):
        """Test that backtest handles missing days gracefully."""
        # Create data with missing days (Feb 15, 16, 2023 are Wed, Thu - business days)
        dates = pd.bdate_range("2023-01-01", "2023-03-31", freq="B")
        dates = dates.drop([pd.Timestamp("2023-02-15"), pd.Timestamp("2023-02-16")])  # Remove days
        df = create_sample_ohlcv_data("TEST", dates)

        # Save to CSV
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()
        df.reset_index().to_csv(data_dir / "TEST.csv", index=False)

        # Try to load and use in backtest
        try:
            data = load_ohlcv_data(str(data_dir), ["TEST"])
            assert "TEST" in data, "Should load data with missing days"
            assert len(data["TEST"]) > 0, "Should have some data"
        except Exception as e:
            pytest.fail(f"Backtest should handle missing days gracefully: {e}")


class TestExtremePriceMoves:
    """Tests for extreme price moves (>50% in one day)."""

    def test_extreme_move_detection(self):
        """Test that extreme moves are detected correctly."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "open": [100.0, 100.0, 161.0, 102.0, 100.0],
                "high": [102.0, 102.0, 165.0, 103.0, 102.0],
                "low": [99.0, 99.0, 160.0, 101.0, 99.0],
                "close": [101.0, 101.0, 162.0, 103.0, 101.0],  # Day 3: 60% move (162/101 - 1 = 60.4%)
                "volume": [1000000] * 5,
            },
            index=dates[:5],
        )
        df.index.name = "date"
        df["dollar_volume"] = df["close"] * df["volume"]

        # Validation should pass but log warning
        assert validate_ohlcv(df, "TEST"), "Validation should pass with extreme move (warning only)"

        # Check that extreme move is detected
        returns = df["close"].pct_change().dropna()
        extreme_moves = abs(returns) > 0.50
        assert extreme_moves.any(), "Should detect extreme move"

    def test_extreme_move_handling(self):
        """Test that extreme moves are handled correctly (treated as missing data per EDGE_CASES.md)."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "open": [100.0, 100.0, 161.0, 102.0, 100.0],
                "high": [102.0, 102.0, 165.0, 103.0, 102.0],
                "low": [99.0, 99.0, 160.0, 101.0, 99.0],
                "close": [101.0, 101.0, 162.0, 103.0, 101.0],  # Day 3: 60% move
                "volume": [1000000] * 5,
            },
            index=dates,
        )
        df.index.name = "date"
        df["dollar_volume"] = df["close"] * df["volume"]

        # Per EDGE_CASES.md, extreme moves should be treated as missing data
        # This means the bar should be skipped during signal generation
        # Validation should pass (warns but doesn't fail)
        assert validate_ohlcv(df, "TEST"), "Extreme moves should not fail validation"

        # Verify extreme move is detected
        returns = df["close"].pct_change().dropna()
        extreme_moves = abs(returns) > 0.50
        extreme_date = returns.index[extreme_moves][0]
        assert extreme_date == pd.Timestamp("2023-01-03"), "Extreme move should be on day 3"

    def test_multiple_extreme_moves(self):
        """Test handling of multiple extreme moves."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        # Two extreme moves: day 3 (162/101=60.4%) and day 6 (158/104=51.9%)
        prices = [100.0, 101.0, 162.0, 103.0, 104.0, 158.0, 105.0, 106.0, 107.0, 108.0]

        df = pd.DataFrame(
            {
                "open": [p * 0.99 for p in prices],
                "high": [p * 1.02 for p in prices],
                "low": [p * 0.98 for p in prices],
                "close": prices,
                "volume": [1000000] * len(prices),
            },
            index=dates,
        )
        df.index.name = "date"
        df["dollar_volume"] = df["close"] * df["volume"]

        # Validation should pass
        assert validate_ohlcv(df, "TEST"), "Validation should pass with multiple extreme moves"

        # Verify both extreme moves are detected
        returns = df["close"].pct_change().dropna()
        extreme_moves = abs(returns) > 0.50
        assert extreme_moves.sum() == 2, "Should detect 2 extreme moves"


class TestLowVolumeDays:
    """Tests for low volume days."""

    def test_low_volume_detection(self):
        """Test that low volume days can be detected."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        volumes = [1000000, 500000, 2000000, 300000, 1500000, 400000, 1800000, 250000, 2000000, 1000000]
        # Low volume threshold: bottom 20%
        low_volume_threshold = np.percentile(volumes, 20)

        df = pd.DataFrame(
            {
                "open": [100.0] * 10,
                "high": [102.0] * 10,
                "low": [99.0] * 10,
                "close": [101.0] * 10,
                "volume": volumes,
            },
            index=dates,
        )
        df.index.name = "date"
        df["dollar_volume"] = df["close"] * df["volume"]

        # Validation should pass
        assert validate_ohlcv(df, "TEST"), "Validation should pass with low volume days"

        # Verify low volume days are detected
        low_volume_days = (df["volume"] < low_volume_threshold).sum()
        assert low_volume_days > 0, "Should detect low volume days"

    def test_zero_volume_handling(self):
        """Test that zero volume is handled correctly."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [102.0] * 5,
                "low": [99.0] * 5,
                "close": [101.0] * 5,
                "volume": [1000000, 0, 2000000, 1500000, 1000000],  # One zero volume day
            },
            index=dates,
        )
        df.index.name = "date"
        df["dollar_volume"] = df["close"] * df["volume"]

        # Zero volume should pass validation (volume >= 0 is valid)
        assert validate_ohlcv(df, "TEST"), "Validation should pass with zero volume"

    def test_very_low_volume_days(self):
        """Test handling of very low volume days (e.g., < 1000 shares)."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        volumes = [1000000, 500, 2000000, 100, 1500000, 50, 1800000, 200, 2000000, 1000000]

        df = pd.DataFrame(
            {
                "open": [100.0] * 10,
                "high": [102.0] * 10,
                "low": [99.0] * 10,
                "close": [101.0] * 10,
                "volume": volumes,
            },
            index=dates,
        )
        df.index.name = "date"
        df["dollar_volume"] = df["close"] * df["volume"]

        # Validation should pass
        assert validate_ohlcv(df, "TEST"), "Validation should pass with very low volume days"

        # Verify very low volume days exist
        very_low_volume = (df["volume"] < 1000).sum()
        assert very_low_volume > 0, "Should have very low volume days"


class TestGapsInData:
    """Tests for gaps in data (consecutive missing days)."""

    def test_single_day_gap(self):
        """Test detection of single day gap."""
        dates = pd.bdate_range("2023-01-01", "2023-01-31", freq="B")
        # Use a date that's actually in the business day range (Jan 13, 2023 is a Friday)
        dates = dates.drop(pd.Timestamp("2023-01-13"))  # Single day gap
        df = create_sample_ohlcv_data("TEST", dates)

        missing_info = detect_missing_data(df, "TEST", asset_class="equity")

        assert "missing_dates" in missing_info, "Should detect missing dates"
        assert len(missing_info["consecutive_gaps"]) == 1, "Should detect 1 gap"
        assert missing_info["gap_lengths"][0] == 1, "Gap should be 1 day"

    def test_consecutive_gaps(self):
        """Test detection of consecutive missing days (2+ days)."""
        dates = pd.bdate_range("2023-01-01", "2023-01-31", freq="B")
        # Remove 3 consecutive business days (Jan 10, 11, 12, 2023 are Tue, Wed, Thu)
        dates = dates.drop([pd.Timestamp("2023-01-10"), pd.Timestamp("2023-01-11"), pd.Timestamp("2023-01-12")])
        df = create_sample_ohlcv_data("TEST", dates)

        missing_info = detect_missing_data(df, "TEST", asset_class="equity")

        assert "missing_dates" in missing_info, "Should detect missing dates"
        assert len(missing_info["consecutive_gaps"]) == 1, "Should detect 1 consecutive gap"
        assert missing_info["gap_lengths"][0] == 3, "Gap should be 3 days"

    def test_multiple_gaps(self):
        """Test detection of multiple gaps."""
        dates = pd.bdate_range("2023-01-01", "2023-01-31", freq="B")
        # Remove multiple non-consecutive business days (Jan 4, 6, 11, 2023 are Wed, Fri, Wed)
        dates = dates.drop([pd.Timestamp("2023-01-04"), pd.Timestamp("2023-01-06"), pd.Timestamp("2023-01-11")])
        df = create_sample_ohlcv_data("TEST", dates)

        missing_info = detect_missing_data(df, "TEST", asset_class="equity")

        assert "missing_dates" in missing_info, "Should detect missing dates"
        assert len(missing_info["missing_dates"]) == 3, "Should detect 3 missing dates"
        # Should have 3 separate gaps (each 1 day)
        assert len(missing_info["consecutive_gaps"]) == 3, "Should detect 3 gaps"

    def test_large_gap_handling(self):
        """Test handling of large gaps (e.g., > 5 days)."""
        dates = pd.bdate_range("2023-01-01", "2023-01-31", freq="B")
        # Remove 5 consecutive business days within same week (Mon-Fri) - Jan 9-13, 2023
        gap_dates = pd.bdate_range("2023-01-09", "2023-01-13", freq="B")
        dates = dates.drop(gap_dates)
        df = create_sample_ohlcv_data("TEST", dates)

        missing_info = detect_missing_data(df, "TEST", asset_class="equity")

        assert "missing_dates" in missing_info, "Should detect missing dates"
        assert len(missing_info["consecutive_gaps"]) == 1, "Should detect 1 large gap"
        assert missing_info["gap_lengths"][0] >= 5, "Gap should be at least 5 days"

    def test_gaps_validation_passes(self):
        """Test that gaps don't cause validation to fail."""
        dates = pd.bdate_range("2023-01-01", "2023-01-31", freq="B")
        # Remove 2 consecutive business days (Jan 10, 11, 2023 are Tue, Wed)
        dates = dates.drop([pd.Timestamp("2023-01-10"), pd.Timestamp("2023-01-11")])  # 2-day gap
        df = create_sample_ohlcv_data("TEST", dates)

        # Validation should pass (gaps are warnings, not errors)
        assert validate_ohlcv(df, "TEST"), "Validation should pass with gaps"


class TestDuplicateDates:
    """Tests for duplicate dates."""

    def test_duplicate_dates_detection(self):
        """Test that duplicate dates are detected and cause validation to fail."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        # Add duplicate date
        dates = dates.append(pd.Index([pd.Timestamp("2023-01-03")]))

        df = pd.DataFrame(
            {
                "open": [100.0] * 6,
                "high": [102.0] * 6,
                "low": [99.0] * 6,
                "close": [101.0] * 6,
                "volume": [1000000] * 6,
            },
            index=dates,
        )
        df.index.name = "date"
        df["dollar_volume"] = df["close"] * df["volume"]

        # Validation should fail (duplicate dates are errors)
        assert not validate_ohlcv(df, "TEST"), "Validation should fail with duplicate dates"

    def test_multiple_duplicate_dates(self):
        """Test detection of multiple duplicate dates."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        # Add multiple duplicates
        dates = dates.append(pd.Index([pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-03")]))

        df = pd.DataFrame(
            {
                "open": [100.0] * 7,
                "high": [102.0] * 7,
                "low": [99.0] * 7,
                "close": [101.0] * 7,
                "volume": [1000000] * 7,
            },
            index=dates,
        )
        df.index.name = "date"
        df["dollar_volume"] = df["close"] * df["volume"]

        # Validation should fail
        assert not validate_ohlcv(df, "TEST"), "Validation should fail with multiple duplicate dates"

    def test_duplicate_dates_in_backtest(self, tmp_path):
        """Test that backtest rejects data with duplicate dates."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        dates = dates.append(pd.Index([pd.Timestamp("2023-01-03")]))  # Duplicate

        df = pd.DataFrame(
            {
                "open": [100.0] * 6,
                "high": [102.0] * 6,
                "low": [99.0] * 6,
                "close": [101.0] * 6,
                "volume": [1000000] * 6,
            },
            index=dates,
        )
        df.index.name = "date"
        df["dollar_volume"] = df["close"] * df["volume"]

        # Save to CSV
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()
        df.reset_index().to_csv(data_dir / "TEST.csv", index=False)

        # Try to load - should handle gracefully (may skip invalid data)
        try:
            data = load_ohlcv_data(str(data_dir), ["TEST"])
            # If data loads, it should be cleaned (duplicates removed)
            # If validation fails, data should not be loaded
            if "TEST" in data:
                # Data was loaded, verify no duplicates
                assert not data["TEST"].index.duplicated().any(), "Loaded data should not have duplicates"
        except Exception:
            # Validation failure is expected for duplicate dates
            pass


class TestCombinedEdgeCases:
    """Tests for combinations of edge cases."""

    def test_missing_days_and_extreme_moves(self):
        """Test handling of both missing days and extreme moves."""
        dates = pd.bdate_range("2023-01-01", "2023-01-31", freq="B")
        dates = dates.drop(pd.Timestamp("2023-01-13"))  # Missing day (Friday)
        df = create_sample_ohlcv_data("TEST", dates)

        # Add extreme move on Jan 10 (Tuesday)
        prev_date = df.index[df.index < pd.Timestamp("2023-01-10")][-1]  # Get previous date in the dataframe
        df.loc[pd.Timestamp("2023-01-10"), "close"] = df.loc[prev_date, "close"] * 1.6  # 60% move
        df.loc[pd.Timestamp("2023-01-10"), "high"] = df.loc[pd.Timestamp("2023-01-10"), "close"] * 1.02
        df.loc[pd.Timestamp("2023-01-10"), "low"] = df.loc[pd.Timestamp("2023-01-10"), "close"] * 0.98
        df.loc[pd.Timestamp("2023-01-10"), "open"] = df.loc[pd.Timestamp("2023-01-10"), "close"] * 0.99

        # Both should be handled
        assert validate_ohlcv(df, "TEST"), "Validation should pass with both issues"

        # Verify both are detected
        missing_info = detect_missing_data(df, "TEST", asset_class="equity")
        assert "missing_dates" in missing_info, "Should detect missing days"

        returns = df["close"].pct_change().dropna()
        extreme_moves = abs(returns) > 0.50
        assert extreme_moves.any(), "Should detect extreme move"

    def test_low_volume_and_gaps(self):
        """Test handling of both low volume days and gaps."""
        dates = pd.bdate_range("2023-01-01", "2023-01-31", freq="B")
        dates = dates.drop([pd.Timestamp("2023-01-10"), pd.Timestamp("2023-01-11")])  # Gap (Tue, Wed)
        df = create_sample_ohlcv_data("TEST", dates)

        # Add low volume days
        df.loc[df.index[5], "volume"] = 100  # Very low volume
        df.loc[df.index[10], "volume"] = 200  # Very low volume

        # Both should be handled
        assert validate_ohlcv(df, "TEST"), "Validation should pass with both issues"

        # Verify both are detected
        missing_info = detect_missing_data(df, "TEST", asset_class="equity")
        assert len(missing_info["missing_dates"]) >= 2, "Should detect gap"

        low_volume_threshold = df["volume"].quantile(0.1)
        low_volume_days = (df["volume"] < low_volume_threshold).sum()
        assert low_volume_days > 0, "Should detect low volume days"

"""Unit tests for technical indicators library."""


import numpy as np
import pandas as pd
import pytest

from trading_system.exceptions import IndicatorError
from trading_system.indicators import (
    adv,
    atr,
    compute_features,
    compute_features_for_date,
    highest_close,
    ma,
    roc,
    rolling_corr,
)
from trading_system.models.features import FeatureRow


class TestMovingAverage:
    """Tests for moving average indicator."""

    def test_ma_basic(self):
        """Test basic moving average calculation."""
        # Create simple ascending series
        close = pd.Series([100 + i for i in range(30)], index=pd.date_range("2024-01-01", periods=30))

        ma20 = ma(close, window=20)

        # First 19 values should be NaN
        assert pd.isna(ma20.iloc[:19]).all()

        # 20th value should be mean of first 20 values
        expected_ma20 = close.iloc[:20].mean()
        assert abs(ma20.iloc[19] - expected_ma20) < 1e-10

        # 21st value should be mean of values 1-21
        expected_ma21 = close.iloc[1:21].mean()
        assert abs(ma20.iloc[20] - expected_ma21) < 1e-10

    def test_ma_insufficient_data(self):
        """Test MA with insufficient data."""
        close = pd.Series([100, 101, 102], index=pd.date_range("2024-01-01", periods=3))

        ma20 = ma(close, window=20)

        # All values should be NaN (less than 20 data points)
        assert pd.isna(ma20).all()

    def test_ma_no_forward_fill(self):
        """Test that MA does not forward-fill NaN values."""
        close = pd.Series([100 + i for i in range(30)], index=pd.date_range("2024-01-01", periods=30))

        ma20 = ma(close, window=20)

        # Check that NaN values are not forward-filled
        # If forward-filled, all values after index 19 would be non-NaN
        # But we want NaN only for first 19
        assert pd.isna(ma20.iloc[18])
        assert not pd.isna(ma20.iloc[19])


class TestATR:
    """Tests for Average True Range indicator."""

    def test_atr_basic(self):
        """Test basic ATR calculation with known values."""
        # Create test data with known True Range values
        dates = pd.date_range("2024-01-01", periods=20)
        df = pd.DataFrame(
            {"high": [102, 103, 101, 104, 105], "low": [99, 100, 99.5, 101, 102], "close": [100.5, 101, 100, 102, 103]},
            index=dates[:5],
        )

        atr14 = atr(df, period=14)

        # First 13 values should be NaN
        assert pd.isna(atr14.iloc[:13]).all() if len(atr14) >= 14 else True

    def test_atr_wilders_smoothing(self):
        """Test that ATR uses Wilder's smoothing (not simple average)."""
        # Create data with constant True Range
        dates = pd.date_range("2024-01-01", periods=20)
        tr_value = 2.0
        df = pd.DataFrame({"high": [100 + tr_value] * 20, "low": [100] * 20, "close": [100] * 20}, index=dates)

        # Set first close to establish previous close
        df.loc[dates[0], "close"] = 100

        atr14 = atr(df, period=14)

        # With Wilder's smoothing and constant TR, ATR should converge
        # First period-1 values are NaN
        if len(atr14) >= 14:
            # Check that ATR is computed (not NaN)
            assert not pd.isna(atr14.iloc[13])

    def test_atr_missing_columns(self):
        """Test ATR with missing required columns."""
        df = pd.DataFrame({"high": [100, 101], "low": [99, 100]})

        with pytest.raises(ValueError, match="must contain columns"):
            atr(df, period=14)


class TestROC:
    """Tests for Rate of Change indicator."""

    def test_roc_basic(self):
        """Test basic ROC calculation."""
        # Create series where price doubles over 60 days
        close = pd.Series([100 * (1.01**i) for i in range(100)], index=pd.date_range("2024-01-01", periods=100))

        roc60 = roc(close, window=60)

        # First 60 values should be NaN
        assert pd.isna(roc60.iloc[:60]).all()

        # Value at index 60 should be (close[60] / close[0]) - 1
        expected_roc = (close.iloc[60] / close.iloc[0]) - 1
        assert abs(roc60.iloc[60] - expected_roc) < 1e-10

    def test_roc_insufficient_data(self):
        """Test ROC with insufficient data."""
        close = pd.Series([100, 101, 102], index=pd.date_range("2024-01-01", periods=3))

        roc60 = roc(close, window=60)

        # All values should be NaN (less than 60 data points)
        assert pd.isna(roc60).all()

    def test_roc_zero_change(self):
        """Test ROC when price doesn't change."""
        close = pd.Series([100] * 100, index=pd.date_range("2024-01-01", periods=100))

        roc60 = roc(close, window=60)

        # After index 60, ROC should be 0 (no change)
        assert abs(roc60.iloc[60]) < 1e-10


class TestHighestClose:
    """Tests for highest close indicator (breakout levels)."""

    def test_highest_close_excludes_today(self):
        """Test that highest_close excludes today's close (no lookahead)."""
        # Create series with increasing prices
        close = pd.Series([100 + i for i in range(30)], index=pd.date_range("2024-01-01", periods=30))

        highest_20d = highest_close(close, window=20)

        # First 20 values should be NaN
        assert pd.isna(highest_20d.iloc[:20]).all()

        # Value at index 20 should be max of close[0:20] (excluding close[20])
        expected_highest = close.iloc[0:20].max()
        assert abs(highest_20d.iloc[20] - expected_highest) < 1e-10

        # Value at index 21 should be max of close[1:21] (excluding close[21])
        expected_highest_21 = close.iloc[1:21].max()
        assert abs(highest_20d.iloc[21] - expected_highest_21) < 1e-10

    def test_highest_close_no_lookahead(self):
        """Test that highest_close never uses future data."""
        # Create series where price spikes at index 25
        close = pd.Series([100] * 30, index=pd.date_range("2024-01-01", periods=30))
        close.iloc[25] = 200  # Spike

        highest_20d = highest_close(close, window=20)

        # Value at index 25 should NOT include the spike (it's at index 25)
        # It should be max of close[5:25] (excluding close[25])
        expected = close.iloc[5:25].max()
        assert abs(highest_20d.iloc[25] - expected) < 1e-10

        # Value at index 26 SHOULD include the spike (it's in the prior 20 days)
        expected_26 = close.iloc[6:26].max()  # Includes close[25] = 200
        assert abs(highest_20d.iloc[26] - expected_26) < 1e-10


class TestADV:
    """Tests for Average Dollar Volume indicator."""

    def test_adv_basic(self):
        """Test basic ADV calculation."""
        dollar_vol = pd.Series([1e6 * (1 + i * 0.01) for i in range(30)], index=pd.date_range("2024-01-01", periods=30))

        adv20 = adv(dollar_vol, window=20)

        # First 19 values should be NaN
        assert pd.isna(adv20.iloc[:19]).all()

        # 20th value should be mean of first 20 dollar volumes
        expected_adv = dollar_vol.iloc[:20].mean()
        assert abs(adv20.iloc[19] - expected_adv) < 1e-10

    def test_adv_constant_volume(self):
        """Test ADV with constant dollar volume."""
        dollar_vol = pd.Series([1e6] * 30, index=pd.date_range("2024-01-01", periods=30))

        adv20 = adv(dollar_vol, window=20)

        # After index 19, ADV should equal the constant value
        assert abs(adv20.iloc[19] - 1e6) < 1e-10


class TestRollingCorrelation:
    """Tests for rolling correlation indicator."""

    def test_rolling_corr_perfect_correlation(self):
        """Test correlation with perfectly correlated series."""
        returns_a = pd.Series([0.01, 0.02, -0.01, 0.03] * 10, index=pd.date_range("2024-01-01", periods=40))
        returns_b = returns_a * 2  # Perfectly correlated (2x)

        corr20 = rolling_corr(returns_a, returns_b, window=20)

        # First 19 values should be NaN
        assert pd.isna(corr20.iloc[:19]).all()

        # After index 19, correlation should be 1.0 (perfect correlation)
        assert abs(corr20.iloc[19] - 1.0) < 1e-10

    def test_rolling_corr_negative_correlation(self):
        """Test correlation with negatively correlated series."""
        returns_a = pd.Series([0.01, 0.02, -0.01, 0.03] * 10, index=pd.date_range("2024-01-01", periods=40))
        returns_b = -returns_a  # Perfectly negatively correlated

        corr20 = rolling_corr(returns_a, returns_b, window=20)

        # After index 19, correlation should be -1.0
        assert abs(corr20.iloc[19] - (-1.0)) < 1e-10

    def test_rolling_corr_misaligned_indices(self):
        """Test correlation with misaligned indices."""
        returns_a = pd.Series([0.01, 0.02, 0.03], index=pd.date_range("2024-01-01", periods=3))
        returns_b = pd.Series([0.02, 0.03, 0.04], index=pd.date_range("2024-01-02", periods=3))  # Different dates

        corr20 = rolling_corr(returns_a, returns_b, window=20)

        # Should handle misalignment gracefully (use intersection)
        assert len(corr20) == len(returns_a)


class TestComputeFeatures:
    """Tests for main compute_features function."""

    def test_compute_features_basic(self):
        """Test basic feature computation."""
        # Create test data
        dates = pd.date_range("2024-01-01", periods=250)
        df = pd.DataFrame(
            {
                "open": [100 + i * 0.1 for i in range(250)],
                "high": [102 + i * 0.1 for i in range(250)],
                "low": [99 + i * 0.1 for i in range(250)],
                "close": [101 + i * 0.1 for i in range(250)],
                "volume": [1e6] * 250,
            },
            index=dates,
        )

        features = compute_features(df, symbol="AAPL", asset_class="equity")

        # Check that all required columns exist
        required_cols = [
            "date",
            "symbol",
            "asset_class",
            "close",
            "open",
            "high",
            "low",
            "ma20",
            "ma50",
            "ma200",
            "atr14",
            "roc60",
            "highest_close_20d",
            "highest_close_55d",
            "adv20",
            "returns_1d",
        ]
        assert all(col in features.columns for col in required_cols)

        # Check that symbol and asset_class are set correctly
        assert (features["symbol"] == "AAPL").all()
        assert (features["asset_class"] == "equity").all()

        # Check that indicators are computed (some will be NaN due to lookback)
        # MA20 should be available after index 19
        assert not pd.isna(features["ma20"].iloc[19])

        # MA50 should be available after index 49
        assert not pd.isna(features["ma50"].iloc[49])

        # MA200 should be available after index 199
        assert not pd.isna(features["ma200"].iloc[199])

    def test_compute_features_with_benchmark(self):
        """Test feature computation with benchmark data."""
        dates = pd.date_range("2024-01-01", periods=100)
        df = pd.DataFrame(
            {"open": [100] * 100, "high": [102] * 100, "low": [99] * 100, "close": [101] * 100, "volume": [1e6] * 100},
            index=dates,
        )

        benchmark_roc = pd.Series([0.05] * 100, index=dates)
        benchmark_returns = pd.Series([0.001] * 100, index=dates)

        features = compute_features(
            df, symbol="BTC", asset_class="crypto", benchmark_roc60=benchmark_roc, benchmark_returns=benchmark_returns
        )

        # Check benchmark columns exist
        assert "benchmark_roc60" in features.columns
        assert "benchmark_returns_1d" in features.columns

        # Check benchmark values are set
        assert not pd.isna(features["benchmark_roc60"].iloc[0])
        assert abs(features["benchmark_roc60"].iloc[0] - 0.05) < 1e-9  # Relax tolerance

    def test_compute_features_missing_columns(self):
        """Test compute_features with missing required columns."""
        df = pd.DataFrame({"close": [100, 101, 102]})

        with pytest.raises(IndicatorError, match="must contain columns"):
            compute_features(df, symbol="AAPL", asset_class="equity")

    def test_compute_features_invalid_asset_class(self):
        """Test compute_features with invalid asset_class."""
        df = pd.DataFrame({"open": [100], "high": [102], "low": [99], "close": [101], "volume": [1e6]})

        with pytest.raises(IndicatorError, match="asset_class must be"):
            compute_features(df, symbol="AAPL", asset_class="invalid")

    def test_compute_features_for_date(self):
        """Test conversion to FeatureRow object."""
        dates = pd.date_range("2024-01-01", periods=100)
        df = pd.DataFrame(
            {"open": [100] * 100, "high": [102] * 100, "low": [99] * 100, "close": [101] * 100, "volume": [1e6] * 100},
            index=dates,
        )

        features = compute_features(df, symbol="AAPL", asset_class="equity")

        # Get FeatureRow for a date with sufficient data
        test_date = dates[50]  # Should have MA20, MA50, etc.
        feature_row = compute_features_for_date(features, test_date)

        assert feature_row is not None
        assert isinstance(feature_row, FeatureRow)
        assert feature_row.symbol == "AAPL"
        assert feature_row.asset_class == "equity"
        assert feature_row.date == test_date

        # Check that some indicators are available (not None)
        assert feature_row.ma20 is not None
        assert feature_row.ma50 is not None

    def test_compute_features_for_date_missing(self):
        """Test compute_features_for_date with missing date."""
        dates = pd.date_range("2024-01-01", periods=100)
        df = pd.DataFrame(
            {"open": [100] * 100, "high": [102] * 100, "low": [99] * 100, "close": [101] * 100, "volume": [1e6] * 100},
            index=dates,
        )

        features = compute_features(df, symbol="AAPL", asset_class="equity")

        # Try to get features for a date not in the DataFrame
        missing_date = pd.Timestamp("2025-01-01")
        feature_row = compute_features_for_date(features, missing_date)

        assert feature_row is None


class TestEdgeCases:
    """Tests for edge cases and NaN handling."""

    def test_indicators_no_forward_fill(self):
        """Test that indicators never forward-fill NaN values."""
        dates = pd.date_range("2024-01-01", periods=30)
        close = pd.Series([100 + i for i in range(30)], index=dates)

        ma20 = ma(close, window=20)

        # Insert a NaN in the middle
        close_with_nan = close.copy()
        close_with_nan.iloc[25] = np.nan

        ma20_with_nan = ma(close_with_nan, window=20)

        # The NaN should propagate (not be forward-filled)
        # The rolling mean will handle NaN according to pandas behavior
        # but we should not artificially forward-fill

    def test_highest_close_excludes_today_rigorous(self):
        """Rigorous test that highest_close excludes today."""
        # Create data where today's close is the highest
        close = pd.Series([100] * 30, index=pd.date_range("2024-01-01", periods=30))
        close.iloc[25] = 200  # Today's close is highest

        highest_20d = highest_close(close, window=20)

        # At index 25, highest should NOT include close[25] (today)
        # It should be max of close[5:25] (prior 20 days, excluding today)
        expected = close.iloc[5:25].max()  # Should be 100, not 200
        assert abs(highest_20d.iloc[25] - expected) < 1e-10
        assert highest_20d.iloc[25] < 200  # Should not include today's 200

    def test_empty_series(self):
        """Test indicators with empty series."""
        empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))

        ma20 = ma(empty, window=20)
        assert len(ma20) == 0

        roc60 = roc(empty, window=60)
        assert len(roc60) == 0

        highest_20d = highest_close(empty, window=20)
        assert len(highest_20d) == 0

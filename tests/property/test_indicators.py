"""Property-based tests for indicators using hypothesis."""

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from trading_system.indicators.atr import atr
from trading_system.indicators.breakouts import highest_close
from trading_system.indicators.ma import ma
from trading_system.indicators.momentum import roc
from trading_system.indicators.volume import adv


# Strategies for generating test data
def price_series(min_length=1, max_length=500):
    """Generate a price series with realistic values."""
    return st.lists(
        st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
        min_size=min_length,
        max_size=max_length,
    ).map(lambda x: pd.Series(x, index=pd.date_range("2020-01-01", periods=len(x), freq="D")))


def ohlc_dataframe(min_length=1, max_length=500):
    """Generate OHLC DataFrame with realistic relationships."""
    return st.lists(
        st.tuples(
            st.floats(min_value=0.01, max_value=10000.0),  # open
            st.floats(min_value=0.01, max_value=10000.0),  # high
            st.floats(min_value=0.01, max_value=10000.0),  # low
            st.floats(min_value=0.01, max_value=10000.0),  # close
            st.floats(min_value=0.0, max_value=1e12),  # volume
        ),
        min_size=min_length,
        max_size=max_length,
    ).map(
        lambda rows: pd.DataFrame(
            {
                "open": [r[0] for r in rows],
                "high": [max(r[0], r[1], r[2], r[3]) for r in rows],  # high >= open, low, close
                "low": [min(r[0], r[1], r[2], r[3]) for r in rows],  # low <= open, high, close
                "close": [r[3] for r in rows],
                "volume": [r[4] for r in rows],
            },
            index=pd.date_range("2020-01-01", periods=len(rows), freq="D"),
        )
    )


class TestMovingAverage:
    """Property-based tests for moving average indicator."""

    @given(price_series(min_length=1, max_length=200), st.integers(min_value=2, max_value=100))
    @settings(max_examples=50, deadline=5000)
    def test_ma_output_length_matches_input(self, series, window):
        """Property: MA output has same length as input."""
        result = ma(series, window=window)
        assert len(result) == len(series)
        assert len(result.index) == len(series.index)

    @given(price_series(min_length=50, max_length=200), st.integers(min_value=2, max_value=50))
    @settings(max_examples=50, deadline=5000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_ma_nan_before_window(self, series, window):
        """Property: First window-1 values are NaN."""
        assume(len(series) >= window)
        result = ma(series, window=window)
        # First window-1 values should be NaN
        assert result.iloc[: window - 1].isna().all()

    @given(price_series(min_length=50, max_length=200), st.integers(min_value=2, max_value=50))
    @settings(max_examples=50, deadline=5000)
    def test_ma_values_within_price_range(self, series, window):
        """Property: MA values are within min/max of input series."""
        assume(len(series) >= window)
        result = ma(series, window=window)
        # Non-NaN values should be within price range
        valid_result = result.dropna()
        if len(valid_result) > 0:
            assert valid_result.min() >= series.min()
            assert valid_result.max() <= series.max()

    @given(price_series(min_length=50, max_length=200), st.integers(min_value=2, max_value=50))
    @settings(max_examples=50, deadline=5000)
    def test_ma_constant_series(self, series, window):
        """Property: MA of constant series equals the constant."""
        assume(len(series) >= window)
        constant_value = series.iloc[0]
        constant_series = pd.Series([constant_value] * len(series), index=series.index)
        result = ma(constant_series, window=window)
        valid_result = result.dropna()
        if len(valid_result) > 0:
            assert np.allclose(valid_result, constant_value, rtol=1e-10)

    @given(price_series(min_length=1, max_length=200), st.integers(min_value=2, max_value=100))
    @settings(max_examples=50, deadline=5000)
    def test_ma_empty_series(self, series, window):
        """Property: Empty series returns empty result."""
        empty_series = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        result = ma(empty_series, window=window)
        assert len(result) == 0

    @given(
        price_series(min_length=50, max_length=200),
        st.integers(min_value=2, max_value=50),
        st.integers(min_value=2, max_value=50),
    )
    @settings(max_examples=30, deadline=5000)
    def test_ma_idempotent(self, series, window1, window2):
        """Property: MA is idempotent (caching doesn't change result)."""
        assume(len(series) >= max(window1, window2))
        result1 = ma(series, window=window1, use_cache=True)
        result2 = ma(series, window=window1, use_cache=False)
        pd.testing.assert_series_equal(result1, result2, check_names=False)


class TestATR:
    """Property-based tests for ATR indicator."""

    @given(ohlc_dataframe(min_length=1, max_length=200))
    @settings(max_examples=50, deadline=5000)
    def test_atr_output_length_matches_input(self, df):
        """Property: ATR output has same length as input."""
        result = atr(df, period=14)
        assert len(result) == len(df)

    @given(ohlc_dataframe(min_length=20, max_length=200))
    @settings(max_examples=50, deadline=5000)
    def test_atr_non_negative(self, df):
        """Property: ATR values are non-negative."""
        result = atr(df, period=14)
        valid_result = result.dropna()
        if len(valid_result) > 0:
            assert (valid_result >= 0).all()

    @given(ohlc_dataframe(min_length=20, max_length=200))
    @settings(max_examples=50, deadline=5000)
    def test_atr_nan_before_period(self, df):
        """Property: First period-1 values are NaN."""
        period = 14
        result = atr(df, period=period)
        assert result.iloc[: period - 1].isna().all()


class TestROC:
    """Property-based tests for rate of change indicator."""

    @given(price_series(min_length=1, max_length=200), st.integers(min_value=1, max_value=100))
    @settings(max_examples=50, deadline=5000)
    def test_roc_output_length_matches_input(self, series, window):
        """Property: ROC output has same length as input."""
        result = roc(series, window=window)
        assert len(result) == len(series)

    @given(price_series(min_length=70, max_length=200), st.integers(min_value=1, max_value=60))
    @settings(max_examples=50, deadline=5000)
    def test_roc_nan_before_window(self, series, window):
        """Property: First window values are NaN."""
        result = roc(series, window=window)
        assert result.iloc[:window].isna().all()

    @given(price_series(min_length=70, max_length=200), st.integers(min_value=1, max_value=60))
    @settings(max_examples=50, deadline=5000)
    def test_roc_finite_values(self, series, window):
        """Property: ROC values are finite (not inf or nan)."""
        assume(len(series) >= window)
        result = roc(series, window=window)
        valid_result = result.dropna()
        if len(valid_result) > 0:
            assert np.isfinite(valid_result).all()


class TestHighestClose:
    """Property-based tests for highest close indicator."""

    @given(price_series(min_length=1, max_length=200), st.integers(min_value=2, max_value=100))
    @settings(max_examples=50, deadline=5000)
    def test_highest_close_output_length_matches_input(self, series, window):
        """Property: Highest close output has same length as input."""
        result = highest_close(series, window=window)
        assert len(result) == len(series)

    @given(price_series(min_length=50, max_length=200), st.integers(min_value=2, max_value=50))
    @settings(max_examples=50, deadline=5000)
    def test_highest_close_greater_than_or_equal_to_prices(self, series, window):
        """Property: Highest close >= all prices in window (excluding today)."""
        assume(len(series) >= window)
        result = highest_close(series, window=window)
        valid_result = result.dropna()
        if len(valid_result) > 0:
            # For each valid result, check it's >= all prices in the window
            for i in range(window, len(series)):
                if not pd.isna(result.iloc[i]):
                    window_prices = series.iloc[i - window : i]  # Excluding today (i)
                    assert result.iloc[i] >= window_prices.max()

    @given(price_series(min_length=50, max_length=200), st.integers(min_value=2, max_value=50))
    @settings(max_examples=50, deadline=5000)
    def test_highest_close_nan_before_window(self, series, window):
        """Property: First window values are NaN."""
        result = highest_close(series, window=window)
        assert result.iloc[:window].isna().all()


class TestADV:
    """Property-based tests for average dollar volume indicator."""

    @given(price_series(min_length=1, max_length=200), st.integers(min_value=2, max_value=100))
    @settings(max_examples=50, deadline=5000)
    def test_adv_output_length_matches_input(self, series, window):
        """Property: ADV output has same length as input."""
        result = adv(series, window=window)
        assert len(result) == len(series)

    @given(price_series(min_length=50, max_length=200), st.integers(min_value=2, max_value=50))
    @settings(max_examples=50, deadline=5000)
    def test_adv_non_negative(self, series, window):
        """Property: ADV values are non-negative."""
        assume(len(series) >= window)
        result = adv(series, window=window)
        valid_result = result.dropna()
        if len(valid_result) > 0:
            assert (valid_result >= 0).all()

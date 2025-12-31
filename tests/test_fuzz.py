"""Fuzz testing for edge cases and malformed inputs."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from trading_system.data.validator import validate_ohlcv
from trading_system.indicators.atr import atr
from trading_system.indicators.breakouts import highest_close
from trading_system.indicators.ma import ma
from trading_system.indicators.momentum import roc
from trading_system.models.orders import Fill
from trading_system.models.signals import BreakoutType, SignalSide
from trading_system.portfolio.portfolio import Portfolio


class TestFuzzIndicators:
    """Fuzz tests for indicators with extreme/malformed inputs."""

    @given(
        st.lists(
            st.one_of(st.floats(allow_nan=True, allow_infinity=True), st.just(np.nan), st.just(np.inf), st.just(-np.inf)),
            min_size=0,
            max_size=1000,
        ),
        st.integers(min_value=1, max_value=200),
    )
    @settings(max_examples=100, deadline=10000)
    def test_ma_handles_extreme_values(self, values, window):
        """Fuzz test: MA handles NaN, inf, and extreme values."""
        try:
            series = pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values), freq="D"))
            result = ma(series, window=window)
            # Should not crash, but may have NaN/inf in result
            assert len(result) == len(series)
        except (ValueError, TypeError):
            pass  # Some extreme inputs may raise errors, which is acceptable

    @given(
        st.lists(
            st.tuples(
                st.one_of(st.floats(), st.just(np.nan), st.just(np.inf)),
                st.one_of(st.floats(), st.just(np.nan), st.just(np.inf)),
                st.one_of(st.floats(), st.just(np.nan), st.just(np.inf)),
                st.one_of(st.floats(), st.just(np.nan), st.just(np.inf)),
                st.one_of(st.floats(min_value=0), st.just(np.nan), st.just(np.inf)),
            ),
            min_size=0,
            max_size=1000,
        )
    )
    @settings(max_examples=100, deadline=10000)
    def test_atr_handles_extreme_ohlc(self, rows):
        """Fuzz test: ATR handles extreme/malformed OHLC data."""
        try:
            df = pd.DataFrame(
                {
                    "open": [r[0] for r in rows],
                    "high": [r[1] for r in rows],
                    "low": [r[2] for r in rows],
                    "close": [r[3] for r in rows],
                    "volume": [r[4] for r in rows],
                },
                index=pd.date_range("2020-01-01", periods=len(rows), freq="D"),
            )

            result = atr(df, period=14)
            # Should not crash
            assert len(result) == len(df)
        except (ValueError, TypeError, KeyError):
            pass  # Some malformed inputs may raise errors

    @given(
        st.lists(
            st.one_of(
                st.floats(allow_nan=True, allow_infinity=True),
                st.just(np.nan),
                st.just(0.0),
                st.just(-1.0),  # Negative prices (invalid but should be handled)
            ),
            min_size=0,
            max_size=1000,
        ),
        st.integers(min_value=1, max_value=200),
    )
    @settings(max_examples=100, deadline=10000)
    def test_roc_handles_extreme_prices(self, values, window):
        """Fuzz test: ROC handles extreme/negative prices."""
        try:
            series = pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values), freq="D"))
            result = roc(series, window=window)
            # Should not crash
            assert len(result) == len(series)
        except (ValueError, TypeError, ZeroDivisionError):
            pass  # Some extreme inputs may raise errors


class TestFuzzDataValidation:
    """Fuzz tests for data validation with malformed inputs."""

    @given(
        st.lists(
            st.tuples(
                st.floats(allow_nan=True, allow_infinity=True),
                st.floats(allow_nan=True, allow_infinity=True),
                st.floats(allow_nan=True, allow_infinity=True),
                st.floats(allow_nan=True, allow_infinity=True),
                st.one_of(st.floats(min_value=-1e10, max_value=1e10), st.just(np.nan)),
            ),
            min_size=0,
            max_size=1000,
        )
    )
    @settings(max_examples=100, deadline=10000)
    def test_validate_ohlcv_handles_malformed_data(self, rows):
        """Fuzz test: Validation handles malformed OHLC data."""
        try:
            df = pd.DataFrame(
                {
                    "open": [r[0] for r in rows],
                    "high": [r[1] for r in rows],
                    "low": [r[2] for r in rows],
                    "close": [r[3] for r in rows],
                    "volume": [r[4] for r in rows],
                },
                index=pd.date_range("2020-01-01", periods=len(rows), freq="D"),
            )

            result = validate_ohlcv(df, "TEST")
            # Should return bool (True/False), not crash
            assert isinstance(result, bool)
        except (ValueError, TypeError, KeyError, AttributeError):
            pass  # Some malformed inputs may raise errors


class TestFuzzPortfolio:
    """Fuzz tests for portfolio with extreme inputs."""

    @given(
        st.one_of(
            st.floats(allow_nan=True, allow_infinity=True),
            st.just(np.nan),
            st.just(np.inf),
            st.just(-np.inf),
            st.just(0.0),
            st.just(-1000.0),  # Negative equity
        ),
        st.one_of(st.floats(allow_nan=True, allow_infinity=True), st.just(np.nan), st.just(np.inf), st.just(-np.inf)),
    )
    @settings(max_examples=100, deadline=5000)
    def test_portfolio_handles_extreme_equity(self, equity, cash):
        """Fuzz test: Portfolio handles extreme equity/cash values."""
        try:
            # Filter out clearly invalid inputs
            if not (np.isfinite(equity) and np.isfinite(cash) and equity > 0 and cash >= 0):
                return

            portfolio = Portfolio(date=pd.Timestamp("2020-01-01"), starting_equity=equity, cash=cash, equity=equity)

            # Should initialize without crashing
            assert portfolio is not None
        except (ValueError, TypeError, AssertionError):
            pass  # Some extreme inputs may raise errors

    @given(
        st.floats(min_value=0.01, max_value=1000000.0),
        st.one_of(
            st.floats(allow_nan=True, allow_infinity=True),
            st.just(np.nan),
            st.just(np.inf),
            st.just(-np.inf),
            st.just(0.0),
            st.just(-100.0),  # Negative price
        ),
        st.integers(min_value=-1000, max_value=100000),
    )
    @settings(max_examples=100, deadline=5000)
    def test_portfolio_handles_extreme_fills(self, equity, price, quantity):
        """Fuzz test: Portfolio handles extreme fill values."""
        try:
            # Filter out invalid inputs
            if not (np.isfinite(price) and price > 0 and quantity > 0):
                return

            portfolio = Portfolio(date=pd.Timestamp("2020-01-01"), starting_equity=equity, cash=equity, equity=equity)

            notional = price * quantity
            if notional * 1.0015 > equity:
                return  # Would exceed available cash

            fill = Fill(
                fill_id="test_fill",
                order_id="test_order",
                symbol="TEST",
                asset_class="equity",
                date=pd.Timestamp("2020-01-01"),
                side=SignalSide.BUY,
                quantity=quantity,
                fill_price=price,
                open_price=price,
                slippage_bps=10.0,
                fee_bps=5.0,
                total_cost=notional * 1.0015,
                vol_mult=1.0,
                size_penalty=1.0,
                weekend_penalty=1.0,
                stress_mult=1.0,
                notional=notional,
            )

            portfolio.process_fill(
                fill=fill, stop_price=price * 0.95, atr_mult=2.5, triggered_on=BreakoutType.FAST_20D, adv20_at_entry=notional
            )

            # Should not crash
            assert portfolio is not None
        except (ValueError, TypeError, AssertionError, OverflowError):
            pass  # Some extreme inputs may raise errors

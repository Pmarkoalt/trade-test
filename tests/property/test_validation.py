"""Property-based tests for validation suite using hypothesis."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from trading_system.validation.bootstrap import compute_max_drawdown_from_r_multiples, compute_sharpe_from_r_multiples
from trading_system.validation.permutation import PermutationTest


# Strategies for generating test data
def r_multiples_list(min_size=1, max_size=1000):
    """Generate list of R-multiples."""
    return st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False), min_size=min_size, max_size=max_size
    )


def valid_date_range():
    """Generate valid date range."""
    start = st.datetimes(min_value=pd.Timestamp("2020-01-01"), max_value=pd.Timestamp("2024-12-31")).map(
        lambda x: pd.Timestamp(x)
    )

    end = st.datetimes(min_value=pd.Timestamp("2020-01-01"), max_value=pd.Timestamp("2025-12-31")).map(
        lambda x: pd.Timestamp(x)
    )

    return st.tuples(start, end).filter(lambda x: x[0] < x[1])


class TestBootstrapProperties:
    """Property-based tests for bootstrap resampling."""

    @given(r_multiples_list(min_size=1, max_size=100))
    @settings(max_examples=50, deadline=5000)
    def test_sharpe_finite(self, r_multiples):
        """Property: Sharpe ratio is always finite."""
        sharpe = compute_sharpe_from_r_multiples(r_multiples)
        assert np.isfinite(sharpe)

    @given(r_multiples_list(min_size=1, max_size=100))
    @settings(max_examples=50, deadline=5000)
    def test_sharpe_empty_returns_zero(self, r_multiples):
        """Property: Empty R-multiples returns zero Sharpe."""
        sharpe = compute_sharpe_from_r_multiples([])
        assert sharpe == 0.0

    @given(r_multiples_list(min_size=2, max_size=100))
    @settings(max_examples=50, deadline=5000)
    def test_sharpe_constant_returns_zero(self, r_multiples):
        """Property: Constant R-multiples (zero std) returns zero Sharpe."""
        constant = r_multiples[0] if r_multiples else 0.0
        constant_list = [constant] * len(r_multiples)
        sharpe = compute_sharpe_from_r_multiples(constant_list)
        # Use approximate comparison for floating point values
        assert abs(sharpe) < 1e-10, f"Expected Sharpe to be approximately 0.0, got {sharpe}"

    @given(r_multiples_list(min_size=1, max_size=100))
    @settings(max_examples=50, deadline=5000)
    def test_max_drawdown_finite(self, r_multiples):
        """Property: Max drawdown is always finite."""
        max_dd = compute_max_drawdown_from_r_multiples(r_multiples)
        assert np.isfinite(max_dd)

    @given(r_multiples_list(min_size=1, max_size=100))
    @settings(max_examples=50, deadline=5000)
    def test_max_drawdown_non_positive(self, r_multiples):
        """Property: Max drawdown is non-positive (<= 0)."""
        max_dd = compute_max_drawdown_from_r_multiples(r_multiples)
        assert max_dd <= 0.0

    @given(r_multiples_list(min_size=1, max_size=100))
    @settings(max_examples=50, deadline=5000)
    def test_max_drawdown_empty_returns_zero(self, r_multiples):
        """Property: Empty R-multiples returns zero drawdown."""
        max_dd = compute_max_drawdown_from_r_multiples([])
        assert max_dd == 0.0


class TestPermutationProperties:
    """Property-based tests for permutation test."""

    @given(
        st.lists(
            st.tuples(valid_date_range(), st.text(min_size=1, max_size=10), st.floats(min_value=-10.0, max_value=10.0)),
            min_size=1,
            max_size=100,
        ),
        valid_date_range(),
        st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=30, deadline=10000)
    def test_permutation_test_structure(self, trades_data, period, n_iterations):
        """Property: Permutation test returns valid structure."""
        # Convert trades data to format expected by PermutationTest
        start_date, end_date = period

        actual_trades = []
        for entry_range, symbol, r_mult in trades_data:
            entry_start, entry_end = entry_range
            assume(entry_start >= start_date and entry_end <= end_date)

            actual_trades.append({"entry_date": entry_start, "exit_date": entry_end, "symbol": symbol, "r_multiple": r_mult})

        if not actual_trades:
            return  # Skip if no valid trades

        def compute_sharpe_func(trades):
            r_multiples = [t.get("r_multiple", 0.0) for t in trades]
            return compute_sharpe_from_r_multiples(r_multiples)

        test = PermutationTest(actual_trades, period, compute_sharpe_func, n_iterations=n_iterations, random_seed=42)

        results = test.run()

        # Check structure
        assert "actual_sharpe" in results
        assert "random_sharpe_5th" in results
        assert "random_sharpe_95th" in results
        assert "percentile_rank" in results
        assert "passed" in results

        # Check values are finite
        assert np.isfinite(results["actual_sharpe"])
        assert np.isfinite(results["random_sharpe_5th"])
        assert np.isfinite(results["random_sharpe_95th"])
        assert np.isfinite(results["percentile_rank"])

        # Check percentile rank is in valid range
        assert 0.0 <= results["percentile_rank"] <= 100.0

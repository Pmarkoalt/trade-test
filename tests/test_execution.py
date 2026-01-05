"""Unit tests for execution simulation (slippage, fees, fills)."""


import numpy as np
import pandas as pd
import pytest

from trading_system.execution import (
    check_capacity_constraint,
    check_capacity_constraint_with_quantity,
    compute_fee_bps,
    compute_fee_cost,
    compute_size_penalty,
    compute_slippage_bps,
    compute_slippage_components,
    compute_stress_multiplier,
    compute_volatility_multiplier,
    compute_weekend_penalty,
    compute_weekly_return,
)
from trading_system.execution.fill_simulator import reject_order_missing_data, simulate_fill
from trading_system.models.bar import Bar
from trading_system.models.orders import Order
from trading_system.models.signals import SignalSide


class TestVolatilityMultiplier:
    """Tests for volatility multiplier calculation."""

    def test_vol_mult_basic(self):
        """Test basic volatility multiplier calculation."""
        atr14 = 2.0
        # Create 60 days of ATR history with mean of 1.0
        atr_history = pd.Series([1.0] * 60)

        vol_mult = compute_volatility_multiplier(atr14, atr_history)

        # Should be 2.0 / 1.0 = 2.0 (clipped to [0.5, 3.0])
        assert abs(vol_mult - 2.0) < 1e-6

    def test_vol_mult_clipping_low(self):
        """Test that vol_mult is clipped to minimum 0.5."""
        atr14 = 0.3
        atr_history = pd.Series([1.0] * 60)

        vol_mult = compute_volatility_multiplier(atr14, atr_history)

        # Should be clipped to 0.5 (0.3/1.0 = 0.3 < 0.5)
        assert vol_mult == 0.5

    def test_vol_mult_clipping_high(self):
        """Test that vol_mult is clipped to maximum 3.0."""
        atr14 = 5.0
        atr_history = pd.Series([1.0] * 60)

        vol_mult = compute_volatility_multiplier(atr14, atr_history)

        # Should be clipped to 3.0 (5.0/1.0 = 5.0 > 3.0)
        assert vol_mult == 3.0

    def test_vol_mult_insufficient_history(self):
        """Test vol_mult with insufficient history (< 60 days)."""
        atr14 = 2.0
        atr_history = pd.Series([1.0] * 30)  # Only 30 days

        vol_mult = compute_volatility_multiplier(atr14, atr_history)

        # Should use current value as baseline (mean = atr14)
        # So ratio = atr14 / atr14 = 1.0
        assert abs(vol_mult - 1.0) < 1e-6


class TestSizePenalty:
    """Tests for size penalty calculation."""

    def test_size_penalty_basic(self):
        """Test basic size penalty calculation."""
        order_notional = 100_000
        adv20 = 10_000_000

        penalty = compute_size_penalty(order_notional, adv20)

        # Should be 100k / (0.01 * 10M) = 100k / 100k = 1.0
        assert abs(penalty - 1.0) < 1e-6

    def test_size_penalty_clipping_low(self):
        """Test that size_penalty is clipped to minimum 0.5."""
        order_notional = 10_000
        adv20 = 10_000_000

        penalty = compute_size_penalty(order_notional, adv20)

        # Should be clipped to 0.5 (10k / 100k = 0.1 < 0.5)
        assert penalty == 0.5

    def test_size_penalty_clipping_high(self):
        """Test that size_penalty is clipped to maximum 2.0."""
        order_notional = 300_000
        adv20 = 10_000_000

        penalty = compute_size_penalty(order_notional, adv20)

        # Should be clipped to 2.0 (300k / 100k = 3.0 > 2.0)
        assert penalty == 2.0

    def test_size_penalty_invalid_adv20(self):
        """Test size_penalty with invalid ADV20."""
        order_notional = 100_000
        adv20 = 0

        penalty = compute_size_penalty(order_notional, adv20)

        # Should return default 1.0
        assert penalty == 1.0


class TestWeekendPenalty:
    """Tests for weekend penalty calculation."""

    def test_weekend_penalty_equity_weekday(self):
        """Test weekend penalty for equity on weekday (should be 1.0)."""
        date = pd.Timestamp("2024-01-15", tz="UTC")  # Monday
        asset_class = "equity"

        penalty = compute_weekend_penalty(date, asset_class)

        assert penalty == 1.0

    def test_weekend_penalty_equity_weekend(self):
        """Test weekend penalty for equity on weekend (should still be 1.0)."""
        date = pd.Timestamp("2024-01-06", tz="UTC")  # Saturday
        asset_class = "equity"

        penalty = compute_weekend_penalty(date, asset_class)

        assert penalty == 1.0

    def test_weekend_penalty_crypto_weekday(self):
        """Test weekend penalty for crypto on weekday (should be 1.0)."""
        date = pd.Timestamp("2024-01-15", tz="UTC")  # Monday
        asset_class = "crypto"

        penalty = compute_weekend_penalty(date, asset_class)

        assert penalty == 1.0

    def test_weekend_penalty_crypto_saturday(self):
        """Test weekend penalty for crypto on Saturday (should be 1.5)."""
        date = pd.Timestamp("2024-01-06", tz="UTC")  # Saturday
        asset_class = "crypto"

        penalty = compute_weekend_penalty(date, asset_class)

        assert penalty == 1.5

    def test_weekend_penalty_crypto_sunday(self):
        """Test weekend penalty for crypto on Sunday (should be 1.5)."""
        date = pd.Timestamp("2024-01-07", tz="UTC")  # Sunday
        asset_class = "crypto"

        penalty = compute_weekend_penalty(date, asset_class)

        assert penalty == 1.5


class TestStressMultiplier:
    """Tests for stress multiplier calculation."""

    def test_stress_mult_equity_below_threshold(self):
        """Test stress multiplier for equity below threshold (-3%)."""
        weekly_return = -0.04  # -4% (below -3% threshold)
        asset_class = "equity"

        mult = compute_stress_multiplier(weekly_return, asset_class)

        assert mult == 2.0

    def test_stress_mult_equity_above_threshold(self):
        """Test stress multiplier for equity above threshold."""
        weekly_return = -0.02  # -2% (above -3% threshold)
        asset_class = "equity"

        mult = compute_stress_multiplier(weekly_return, asset_class)

        assert mult == 1.0

    def test_stress_mult_crypto_below_threshold(self):
        """Test stress multiplier for crypto below threshold (-5%)."""
        weekly_return = -0.06  # -6% (below -5% threshold)
        asset_class = "crypto"

        mult = compute_stress_multiplier(weekly_return, asset_class)

        assert mult == 2.0

    def test_stress_mult_crypto_above_threshold(self):
        """Test stress multiplier for crypto above threshold."""
        weekly_return = -0.04  # -4% (above -5% threshold)
        asset_class = "crypto"

        mult = compute_stress_multiplier(weekly_return, asset_class)

        assert mult == 1.0


class TestSlippageBps:
    """Tests for slippage calculation with variance."""

    def test_slippage_bps_basic(self):
        """Test basic slippage calculation."""
        rng = np.random.default_rng(seed=42)

        slippage_bps, mean, std = compute_slippage_bps(
            base_bps=8.0, vol_mult=1.0, size_penalty=1.0, weekend_penalty=1.0, stress_mult=1.0, rng=rng
        )

        # Mean should be 8.0
        assert abs(mean - 8.0) < 1e-6
        # Std should be 8.0 * 0.75 = 6.0
        assert abs(std - 6.0) < 1e-6
        # Slippage should be in [0, 500]
        assert 0 <= slippage_bps <= 500

    def test_slippage_bps_with_multipliers(self):
        """Test slippage calculation with all multipliers."""
        rng = np.random.default_rng(seed=42)

        slippage_bps, mean, std = compute_slippage_bps(
            base_bps=8.0, vol_mult=2.0, size_penalty=1.5, weekend_penalty=1.0, stress_mult=1.0, rng=rng
        )

        # Mean should be 8 * 2 * 1.5 = 24.0
        assert abs(mean - 24.0) < 1e-6
        # Std should be 24.0 * 0.75 = 18.0
        assert abs(std - 18.0) < 1e-6

    def test_slippage_bps_stress_multiplier(self):
        """Test slippage calculation with stress multiplier (increases std)."""
        rng = np.random.default_rng(seed=42)

        slippage_bps, mean, std = compute_slippage_bps(
            base_bps=8.0, vol_mult=1.0, size_penalty=1.0, weekend_penalty=1.0, stress_mult=2.0, rng=rng
        )

        # Mean should be 8 * 2 = 16.0
        assert abs(mean - 16.0) < 1e-6
        # Std should be 16.0 * 0.75 * 1.5 = 18.0 (fatter tails during stress)
        assert abs(std - 18.0) < 1e-6

    def test_slippage_bps_clipping_min(self):
        """Test that slippage is clipped to minimum 0."""
        rng = np.random.default_rng(seed=42)
        # Use negative mean to test clipping
        # We'll need to use a very small base or negative multipliers
        # But multipliers are always >= 0.5, so we can't get negative mean
        # Instead, test that result is >= 0
        slippage_bps, _, _ = compute_slippage_bps(
            base_bps=0.01, vol_mult=0.5, size_penalty=0.5, weekend_penalty=1.0, stress_mult=1.0, rng=rng
        )

        assert slippage_bps >= 0

    def test_slippage_bps_clipping_max(self):
        """Test that slippage is clipped to maximum 500 bps."""
        rng = np.random.default_rng(seed=42)

        # Use very large multipliers to exceed 500 bps
        slippage_bps, mean, _ = compute_slippage_bps(
            base_bps=100.0, vol_mult=3.0, size_penalty=2.0, weekend_penalty=1.0, stress_mult=2.0, rng=rng
        )

        # Mean would be 100 * 3 * 2 * 2 = 1200, but slippage should be clipped
        assert slippage_bps <= 500


class TestFees:
    """Tests for fee calculation."""

    def test_fee_bps_equity(self):
        """Test fee calculation for equity."""
        fee_bps = compute_fee_bps("equity")
        assert fee_bps == 1

    def test_fee_bps_crypto(self):
        """Test fee calculation for crypto."""
        fee_bps = compute_fee_bps("crypto")
        assert fee_bps == 8

    def test_fee_bps_invalid(self):
        """Test fee calculation with invalid asset_class."""
        with pytest.raises(ValueError):
            compute_fee_bps("invalid")

    def test_fee_cost_equity(self):
        """Test fee cost calculation for equity."""
        notional = 100_000
        cost = compute_fee_cost(notional, "equity")

        # Should be 100k * (1 / 10000) = 10.0
        assert abs(cost - 10.0) < 1e-6

    def test_fee_cost_crypto(self):
        """Test fee cost calculation for crypto."""
        notional = 100_000
        cost = compute_fee_cost(notional, "crypto")

        # Should be 100k * (8 / 10000) = 80.0
        assert abs(cost - 80.0) < 1e-6


class TestWeeklyReturn:
    """Tests for weekly return calculation."""

    def test_weekly_return_equity(self):
        """Test weekly return calculation for equity (5 trading days)."""
        # Create 10 trading days of data
        dates = pd.date_range("2024-01-02", periods=10, freq="B")  # Business days
        closes = [100 + i for i in range(10)]  # Increasing prices
        benchmark_bars = pd.DataFrame({"close": closes}, index=dates)

        current_date = dates[-1]
        weekly_return = compute_weekly_return(benchmark_bars, current_date, "equity")

        # Should be (close[t] / close[t-5]) - 1
        # = (109 / 104) - 1 = 0.0480769...
        expected = (closes[-1] / closes[-5]) - 1
        assert abs(weekly_return - expected) < 1e-6

    def test_weekly_return_crypto(self):
        """Test weekly return calculation for crypto (7 calendar days)."""
        # Create 10 calendar days of data
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        closes = [100 + i for i in range(10)]
        benchmark_bars = pd.DataFrame({"close": closes}, index=dates)

        current_date = dates[-1]
        weekly_return = compute_weekly_return(benchmark_bars, current_date, "crypto")

        # Function uses end_date - 6 days for start_date (7 days total including end_date)
        # If end_date is dates[-1] (2024-01-10), start_date is 2024-01-04
        # So we need to find the close for 2024-01-04, which is dates[3] (closes[3] = 103)
        start_date = current_date - pd.Timedelta(days=6)
        start_close = benchmark_bars.loc[start_date, "close"]
        expected = (closes[-1] / start_close) - 1
        assert abs(weekly_return - expected) < 1e-6

    def test_weekly_return_insufficient_data(self):
        """Test weekly return with insufficient data."""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        closes = [100, 101, 102]
        benchmark_bars = pd.DataFrame({"close": closes}, index=dates)

        current_date = dates[-1]
        weekly_return = compute_weekly_return(benchmark_bars, current_date, "equity")

        # Should return 0.0 (insufficient trading days)
        assert weekly_return == 0.0

    def test_weekly_return_missing_date(self):
        """Test weekly return with missing current_date in index."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        closes = [100 + i for i in range(10)]
        benchmark_bars = pd.DataFrame({"close": closes}, index=dates)

        current_date = pd.Timestamp("2024-01-15")  # Not in index
        weekly_return = compute_weekly_return(benchmark_bars, current_date, "crypto")

        # Should return 0.0 (missing date)
        assert weekly_return == 0.0


class TestCapacity:
    """Tests for capacity constraint checks."""

    def test_capacity_constraint_equity_pass(self):
        """Test capacity constraint for equity that passes."""
        order_notional = 50_000
        adv20 = 10_000_000

        passed, reason = check_capacity_constraint(order_notional, adv20, "equity")

        assert passed is True
        assert reason == ""

    def test_capacity_constraint_equity_fail(self):
        """Test capacity constraint for equity that fails."""
        order_notional = 60_000
        adv20 = 10_000_000

        passed, reason = check_capacity_constraint(order_notional, adv20, "equity")

        assert passed is False
        assert "exceeds capacity limit" in reason

    def test_capacity_constraint_crypto_pass(self):
        """Test capacity constraint for crypto that passes."""
        order_notional = 25_000
        adv20 = 10_000_000

        passed, reason = check_capacity_constraint(order_notional, adv20, "crypto")

        assert passed is True
        assert reason == ""

    def test_capacity_constraint_crypto_fail(self):
        """Test capacity constraint for crypto that fails."""
        order_notional = 30_000
        adv20 = 10_000_000

        passed, reason = check_capacity_constraint(order_notional, adv20, "crypto")

        assert passed is False
        assert "exceeds capacity limit" in reason

    def test_capacity_constraint_invalid_adv20(self):
        """Test capacity constraint with invalid ADV20."""
        order_notional = 50_000
        adv20 = 0

        passed, reason = check_capacity_constraint(order_notional, adv20, "equity")

        assert passed is False
        assert "Invalid" in reason

    def test_capacity_constraint_with_quantity(self):
        """Test capacity constraint using quantity and price."""
        quantity = 100
        price = 500.0
        adv20 = 10_000_000

        passed, reason = check_capacity_constraint_with_quantity(quantity, price, adv20, "equity")

        # Order notional = 100 * 500 = 50k, which is exactly 0.5% of 10M
        assert passed is True
        assert reason == ""


class TestFillSimulator:
    """Tests for fill simulation."""

    def test_simulate_fill_buy(self):
        """Test fill simulation for BUY order."""
        rng = np.random.default_rng(seed=42)

        # Create order
        order = Order(
            order_id="test_order_1",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp("2024-01-15"),
            execution_date=pd.Timestamp("2024-01-16"),
            side=SignalSide.BUY,
            quantity=100,
            signal_date=pd.Timestamp("2024-01-15"),
            expected_fill_price=150.0,
            stop_price=145.0,
        )

        # Create open bar
        open_bar = Bar(
            date=pd.Timestamp("2024-01-16"), symbol="AAPL", open=150.0, high=152.0, low=149.0, close=151.0, volume=1000000
        )

        # Create ATR history
        atr14_history = pd.Series([2.0] * 60)

        # Create benchmark bars
        benchmark_dates = pd.date_range("2024-01-01", periods=30, freq="B")
        benchmark_bars = pd.DataFrame({"close": [100 + i for i in range(30)]}, index=benchmark_dates)

        fill = simulate_fill(
            order=order,
            open_bar=open_bar,
            atr14=2.0,
            atr14_history=atr14_history,
            adv20=10_000_000,
            benchmark_bars=benchmark_bars,
            base_slippage_bps=8.0,
            rng=rng,
        )

        # Validate fill
        assert fill.order_id == order.order_id
        assert fill.symbol == order.symbol
        assert fill.quantity == order.quantity
        assert fill.open_price == 150.0
        # BUY: fill_price should be >= open_price (paying more)
        assert fill.fill_price >= fill.open_price
        assert fill.slippage_bps >= 0
        assert fill.fee_bps == 1  # Equity fee
        assert fill.total_cost > 0
        assert fill.notional > 0

    def test_simulate_fill_sell(self):
        """Test fill simulation for SELL order."""
        rng = np.random.default_rng(seed=42)

        order = Order(
            order_id="test_order_2",
            symbol="BTC",
            asset_class="crypto",
            date=pd.Timestamp("2024-01-15"),
            execution_date=pd.Timestamp("2024-01-16"),
            side=SignalSide.SELL,
            quantity=10,
            signal_date=pd.Timestamp("2024-01-15"),
            expected_fill_price=50000.0,
            stop_price=49000.0,  # Use valid stop price (Order validation requires positive)
        )

        open_bar = Bar(
            date=pd.Timestamp("2024-01-16"), symbol="BTC", open=50000.0, high=51000.0, low=49000.0, close=50500.0, volume=1000
        )

        atr14_history = pd.Series([1000.0] * 60)
        benchmark_dates = pd.date_range("2024-01-01", periods=30, freq="D")
        benchmark_bars = pd.DataFrame({"close": [50000 + i * 100 for i in range(30)]}, index=benchmark_dates)

        fill = simulate_fill(
            order=order,
            open_bar=open_bar,
            atr14=1000.0,
            atr14_history=atr14_history,
            adv20=50_000_000,
            benchmark_bars=benchmark_bars,
            base_slippage_bps=10.0,
            rng=rng,
        )

        # SELL: fill_price should be <= open_price (receiving less)
        assert fill.fill_price <= fill.open_price
        assert fill.fee_bps == 8  # Crypto fee

    def test_simulate_fill_invalid_atr14(self):
        """Test fill simulation with invalid ATR14."""
        order = Order(
            order_id="test_order_3",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp("2024-01-15"),
            execution_date=pd.Timestamp("2024-01-16"),
            side=SignalSide.BUY,
            quantity=100,
            signal_date=pd.Timestamp("2024-01-15"),
            expected_fill_price=150.0,
            stop_price=145.0,
        )

        open_bar = Bar(
            date=pd.Timestamp("2024-01-16"), symbol="AAPL", open=150.0, high=152.0, low=149.0, close=151.0, volume=1000000
        )

        with pytest.raises(ValueError, match="Invalid ATR14"):
            simulate_fill(
                order=order,
                open_bar=open_bar,
                atr14=0.0,  # Invalid
                atr14_history=pd.Series([2.0] * 60),
                adv20=10_000_000,
                benchmark_bars=pd.DataFrame({"close": [100]}, index=[pd.Timestamp("2024-01-16")]),
                base_slippage_bps=8.0,
            )

    def test_reject_order_missing_data(self):
        """Test rejection fill for missing data."""
        order = Order(
            order_id="test_order_4",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp("2024-01-15"),
            execution_date=pd.Timestamp("2024-01-16"),
            side=SignalSide.BUY,
            quantity=100,
            signal_date=pd.Timestamp("2024-01-15"),
            expected_fill_price=150.0,
            stop_price=145.0,
        )

        fill = reject_order_missing_data(order, "Missing ATR14 data")

        assert fill.order_id == order.order_id
        assert fill.quantity == 0
        assert fill.fill_price == 0.0
        assert fill.total_cost == 0.0
        assert fill.notional == 0.0


class TestSlippageComponents:
    """Tests for complete slippage component calculation."""

    def test_slippage_components_integration(self):
        """Test complete slippage component calculation."""
        rng = np.random.default_rng(seed=42)

        order_notional = 100_000
        atr14 = 2.0
        atr14_history = pd.Series([1.0] * 60)
        adv20 = 10_000_000
        date = pd.Timestamp("2024-01-15", tz="UTC")
        asset_class = "equity"
        weekly_return = -0.02
        base_bps = 8.0

        slippage_bps, vol_mult, size_penalty, weekend_penalty, stress_mult = compute_slippage_components(
            order_notional=order_notional,
            atr14=atr14,
            atr14_history=atr14_history,
            adv20=adv20,
            date=date,
            asset_class=asset_class,
            weekly_return=weekly_return,
            base_bps=base_bps,
            rng=rng,
        )

        # Validate components
        assert vol_mult == 2.0  # 2.0 / 1.0 = 2.0
        assert size_penalty == 1.0  # 100k / (0.01 * 10M) = 1.0
        assert weekend_penalty == 1.0  # Weekday
        assert stress_mult == 1.0  # -2% > -3% threshold
        assert 0 <= slippage_bps <= 500

        # Mean should be approximately 8 * 2 * 1 * 1 * 1 = 16
        # (allowing for variance in the draw)
        assert slippage_bps > 0

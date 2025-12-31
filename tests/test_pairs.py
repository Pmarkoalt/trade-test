"""Unit tests for pairs trading strategy."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from trading_system.configs.strategy_config import StrategyConfig
from trading_system.models.features import FeatureRow
from trading_system.models.positions import ExitReason, Position
from trading_system.models.signals import BreakoutType, Signal, SignalSide, SignalType
from trading_system.strategies.pairs.pairs_strategy import PairsTradingStrategy


class TestPairsStrategyInit:
    """Tests for PairsTradingStrategy initialization."""

    def test_init_valid(self):
        """Test initialization with valid pairs config."""
        config = StrategyConfig(
            name="equity_pairs",
            asset_class="equity",
            universe=[],  # Not used for pairs
            benchmark="SPY",
            parameters={
                "pairs": [["XLE", "XLK"], ["GLD", "TLT"]],
                "lookback": 60,
                "entry_zscore": 2.0,
                "exit_zscore": 0.5,
                "max_hold_days": 10,
                "atr_period": 14,
                "stop_atr_mult": 2.0,
            },
        )
        strategy = PairsTradingStrategy(config)
        assert strategy.config == config
        assert len(strategy.pairs) == 2
        assert strategy.pairs[0] == ["XLE", "XLK"]
        assert strategy.lookback == 60
        assert strategy.entry_zscore == 2.0
        assert strategy.exit_zscore == 0.5

    def test_init_no_pairs(self):
        """Test initialization fails with no pairs."""
        config = StrategyConfig(
            name="equity_pairs",
            asset_class="equity",
            universe=[],
            benchmark="SPY",
            parameters={
                "pairs": [],  # Empty
                "lookback": 60,
            },
        )
        with pytest.raises(ValueError, match="requires at least one pair"):
            PairsTradingStrategy(config)

    def test_init_invalid_pair_format(self):
        """Test initialization fails with invalid pair format."""
        config = StrategyConfig(
            name="equity_pairs",
            asset_class="equity",
            universe=[],
            benchmark="SPY",
            parameters={
                "pairs": [["XLE"]],  # Only one symbol
                "lookback": 60,
            },
        )
        with pytest.raises(ValueError, match="must be \\[symbol1, symbol2\\]"):
            PairsTradingStrategy(config)


class TestEligibilityFilters:
    """Tests for eligibility filter logic."""

    @pytest.fixture
    def strategy(self):
        """Create pairs strategy with default config."""
        config = StrategyConfig(
            name="equity_pairs",
            asset_class="equity",
            universe=[],
            benchmark="SPY",
            parameters={
                "pairs": [["XLE", "XLK"]],
                "lookback": 60,
                "entry_zscore": 2.0,
                "exit_zscore": 0.5,
                "max_hold_days": 10,
                "atr_period": 14,
                "stop_atr_mult": 2.0,
            },
        )
        return PairsTradingStrategy(config)

    def test_eligibility_passes_in_pair(self, strategy):
        """Test eligibility passes when symbol is in a pair."""
        features = FeatureRow(
            date=pd.Timestamp("2024-02-01"),
            symbol="XLE",
            asset_class="equity",
            close=100.0,
            open=101.0,
            high=102.0,
            low=99.0,
            atr14=2.0,
            adv20=50_000_000.0,
        )

        is_eligible, failures = strategy.check_eligibility(features)

        assert is_eligible
        assert len(failures) == 0

    def test_eligibility_fails_not_in_pair(self, strategy):
        """Test eligibility fails when symbol is not in any pair."""
        features = FeatureRow(
            date=pd.Timestamp("2024-02-01"),
            symbol="AAPL",  # Not in pairs
            asset_class="equity",
            close=100.0,
            open=101.0,
            high=102.0,
            low=99.0,
            atr14=2.0,
            adv20=50_000_000.0,
        )

        is_eligible, failures = strategy.check_eligibility(features)

        assert not is_eligible
        assert "not_in_pair" in failures

    def test_eligibility_fails_missing_atr(self, strategy):
        """Test eligibility fails when ATR is missing."""
        features = FeatureRow(
            date=pd.Timestamp("2024-02-01"),
            symbol="XLE",
            asset_class="equity",
            close=100.0,
            open=101.0,
            high=102.0,
            low=99.0,
            atr14=None,  # Missing
            adv20=50_000_000.0,
        )

        is_eligible, failures = strategy.check_eligibility(features)

        assert not is_eligible
        assert "atr14_missing" in failures


class TestSpreadZScore:
    """Tests for spread z-score calculation."""

    @pytest.fixture
    def strategy(self):
        """Create pairs strategy."""
        config = StrategyConfig(
            name="equity_pairs",
            asset_class="equity",
            universe=[],
            benchmark="SPY",
            parameters={
                "pairs": [["XLE", "XLK"]],
                "lookback": 20,
                "entry_zscore": 2.0,
                "exit_zscore": 0.5,
                "max_hold_days": 10,
                "atr_period": 14,
                "stop_atr_mult": 2.0,
            },
        )
        return PairsTradingStrategy(config)

    def test_compute_spread_zscore_valid(self, strategy):
        """Test spread z-score calculation with valid history."""
        # Create spread history (log ratio)
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        # Simulate mean-reverting spread around 0.0
        np.random.seed(42)
        spread_values = np.random.randn(30) * 0.1  # Mean 0, std 0.1
        spread_history = pd.Series(spread_values, index=dates)

        # Current prices that give z-score ~ 2.0
        price1 = 100.0
        price2 = 100.0 * np.exp(-0.2)  # Spread = log(100/81.87) ≈ 0.2

        zscore = strategy.compute_spread_zscore(price1, price2, spread_history)

        assert zscore is not None
        assert isinstance(zscore, float)
        # Should be positive (spread above mean)
        assert zscore > 0

    def test_compute_spread_zscore_insufficient_history(self, strategy):
        """Test spread z-score returns None with insufficient history."""
        # Only 10 days (need 20)
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        spread_history = pd.Series(np.random.randn(10), index=dates)

        zscore = strategy.compute_spread_zscore(100.0, 100.0, spread_history)

        assert zscore is None

    def test_compute_spread_zscore_invalid_prices(self, strategy):
        """Test spread z-score returns None with invalid prices."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        spread_history = pd.Series(np.random.randn(30), index=dates)

        zscore = strategy.compute_spread_zscore(0.0, 100.0, spread_history)

        assert zscore is None


class TestPairSignalGeneration:
    """Tests for pair signal generation."""

    @pytest.fixture
    def strategy(self):
        """Create pairs strategy."""
        config = StrategyConfig(
            name="equity_pairs",
            asset_class="equity",
            universe=[],
            benchmark="SPY",
            parameters={
                "pairs": [["XLE", "XLK"]],
                "lookback": 60,
                "entry_zscore": 2.0,
                "exit_zscore": 0.5,
                "max_hold_days": 10,
                "atr_period": 14,
                "stop_atr_mult": 2.0,
            },
        )
        return PairsTradingStrategy(config)

    @pytest.fixture
    def features_xle(self):
        """Create features for XLE."""
        return FeatureRow(
            date=pd.Timestamp("2024-02-01"),
            symbol="XLE",
            asset_class="equity",
            close=100.0,
            open=101.0,
            high=102.0,
            low=99.0,
            ma20=105.0,
            ma50=110.0,
            ma200=115.0,
            atr14=2.0,
            roc60=0.05,
            highest_close_20d=110.0,
            highest_close_55d=115.0,
            adv20=50_000_000.0,
            returns_1d=0.01,
            ma50_slope=0.01,
            benchmark_roc60=0.03,
            benchmark_returns_1d=0.005,
        )

    @pytest.fixture
    def features_xlk(self):
        """Create features for XLK."""
        return FeatureRow(
            date=pd.Timestamp("2024-02-01"),
            symbol="XLK",
            asset_class="equity",
            close=80.0,
            open=81.0,
            high=82.0,
            low=79.0,
            ma20=85.0,
            ma50=90.0,
            ma200=95.0,
            atr14=1.5,
            roc60=0.03,
            highest_close_20d=85.0,
            highest_close_55d=90.0,
            adv20=100_000_000.0,
            returns_1d=0.005,
            ma50_slope=0.01,
            benchmark_roc60=0.03,
            benchmark_returns_1d=0.005,
        )

    def test_generate_pair_signals_spread_too_wide(self, strategy, features_xle, features_xlk):
        """Test signal generation when spread is too wide (zscore > entry_zscore)."""
        spread_zscore = 2.5  # Above 2.0 threshold
        order_notional = 10_000.0
        date = pd.Timestamp("2024-02-01")

        signals = strategy.generate_pair_signals("XLE", "XLK", features_xle, features_xlk, spread_zscore, order_notional, date)

        assert len(signals) == 2

        # First signal: Short XLE (outperformer)
        signal1 = signals[0]
        assert signal1.symbol == "XLE"
        assert signal1.signal_type == SignalType.ENTRY_SHORT
        assert signal1.side == SignalSide.SELL
        assert signal1.trigger_reason.startswith("pairs_divergence_short")
        assert signal1.metadata["leg"] == "short"
        assert signal1.metadata["paired_with"] == "XLK"
        assert signal1.stop_price == 104.0  # 100 + 2.0 * 2.0 (stop above for short)

        # Second signal: Long XLK (underperformer)
        signal2 = signals[1]
        assert signal2.symbol == "XLK"
        assert signal2.signal_type == SignalType.ENTRY_LONG
        assert signal2.side == SignalSide.BUY
        assert signal2.trigger_reason.startswith("pairs_divergence_long")
        assert signal2.metadata["leg"] == "long"
        assert signal2.metadata["paired_with"] == "XLE"
        assert signal2.stop_price == 77.0  # 80 - 2.0 * 1.5 (stop below for long)

    def test_generate_pair_signals_spread_too_narrow(self, strategy, features_xle, features_xlk):
        """Test signal generation when spread is too narrow (zscore < -entry_zscore)."""
        spread_zscore = -2.5  # Below -2.0 threshold
        order_notional = 10_000.0
        date = pd.Timestamp("2024-02-01")

        signals = strategy.generate_pair_signals("XLE", "XLK", features_xle, features_xlk, spread_zscore, order_notional, date)

        assert len(signals) == 2

        # First signal: Long XLE (underperformer)
        signal1 = signals[0]
        assert signal1.symbol == "XLE"
        assert signal1.signal_type == SignalType.ENTRY_LONG
        assert signal1.side == SignalSide.BUY
        assert signal1.trigger_reason.startswith("pairs_divergence_long")
        assert signal1.metadata["leg"] == "long"

        # Second signal: Short XLK (outperformer)
        signal2 = signals[1]
        assert signal2.symbol == "XLK"
        assert signal2.signal_type == SignalType.ENTRY_SHORT
        assert signal2.side == SignalSide.SELL
        assert signal2.trigger_reason.startswith("pairs_divergence_short")
        assert signal2.metadata["leg"] == "short"

    def test_generate_pair_signals_no_divergence(self, strategy, features_xle, features_xlk):
        """Test no signals when spread is not diverged enough."""
        spread_zscore = 1.5  # Between -2.0 and 2.0
        order_notional = 10_000.0
        date = pd.Timestamp("2024-02-01")

        signals = strategy.generate_pair_signals("XLE", "XLK", features_xle, features_xlk, spread_zscore, order_notional, date)

        assert len(signals) == 0

    def test_generate_signal_returns_none(self, strategy):
        """Test that generate_signal returns None (pairs uses generate_pair_signals)."""
        features = FeatureRow(
            date=pd.Timestamp("2024-02-01"),
            symbol="XLE",
            asset_class="equity",
            close=100.0,
            open=101.0,
            high=102.0,
            low=99.0,
            atr14=2.0,
            adv20=50_000_000.0,
        )

        signal = strategy.generate_signal("XLE", features, 10_000.0)

        assert signal is None


class TestPairExitSignals:
    """Tests for pair exit logic."""

    @pytest.fixture
    def strategy(self):
        """Create pairs strategy."""
        config = StrategyConfig(
            name="equity_pairs",
            asset_class="equity",
            universe=[],
            benchmark="SPY",
            parameters={
                "pairs": [["XLE", "XLK"]],
                "lookback": 60,
                "entry_zscore": 2.0,
                "exit_zscore": 0.5,
                "max_hold_days": 10,
                "atr_period": 14,
                "stop_atr_mult": 2.0,
            },
        )
        return PairsTradingStrategy(config)

    @pytest.fixture
    def position_xle(self):
        """Create a test position for XLE."""
        return Position(
            symbol="XLE",
            asset_class="equity",
            entry_date=pd.Timestamp("2024-01-25"),
            entry_price=100.0,
            entry_fill_id="fill_xle",
            quantity=100,
            stop_price=96.0,
            initial_stop_price=96.0,
            hard_stop_atr_mult=2.0,
            entry_slippage_bps=8.0,
            entry_fee_bps=1.0,
            entry_total_cost=90.0,
            triggered_on=BreakoutType.FAST_20D,
            adv20_at_entry=50_000_000.0,
        )

    @pytest.fixture
    def position_xlk(self):
        """Create a test position for XLK."""
        return Position(
            symbol="XLK",
            asset_class="equity",
            entry_date=pd.Timestamp("2024-01-25"),
            entry_price=80.0,
            entry_fill_id="fill_xlk",
            quantity=100,
            stop_price=77.0,
            initial_stop_price=77.0,
            hard_stop_atr_mult=2.0,
            entry_slippage_bps=8.0,
            entry_fee_bps=1.0,
            entry_total_cost=72.0,
            triggered_on=BreakoutType.FAST_20D,
            adv20_at_entry=100_000_000.0,
        )

    def test_check_pair_exit_convergence(self, strategy, position_xle, position_xlk):
        """Test pair exit when spread converged."""
        spread_zscore = 0.3  # Below 0.5 threshold
        date = pd.Timestamp("2024-02-01")

        signals = strategy.check_pair_exit("XLE", "XLK", spread_zscore, position_xle, position_xlk, date)

        assert len(signals) == 2

        # Both should be exit signals
        assert all(s.signal_type in [SignalType.EXIT, SignalType.EXIT_SHORT] for s in signals)
        assert all("pairs_convergence" in s.trigger_reason for s in signals)
        assert all(s.urgency == 0.9 for s in signals)

    def test_check_pair_exit_no_convergence(self, strategy, position_xle, position_xlk):
        """Test no exit when spread not converged."""
        spread_zscore = 1.5  # Above 0.5 threshold
        date = pd.Timestamp("2024-02-01")

        signals = strategy.check_pair_exit("XLE", "XLK", spread_zscore, position_xle, position_xlk, date)

        assert len(signals) == 0

    def test_check_exit_signals_hard_stop(self, strategy, position_xle):
        """Test individual leg exit on hard stop."""
        features = FeatureRow(
            date=pd.Timestamp("2024-02-01"),
            symbol="XLE",
            asset_class="equity",
            close=95.0,  # Below stop
            open=96.0,
            high=97.0,
            low=94.0,
            atr14=2.0,
        )

        exit_reason = strategy.check_exit_signals(position_xle, features)

        assert exit_reason == ExitReason.HARD_STOP

    def test_check_exit_signals_time_stop(self, strategy, position_xle):
        """Test individual leg exit on time stop."""
        # Position entered 11 days ago (exceeds max_hold_days=10)
        position_xle.entry_date = pd.Timestamp("2024-01-21")

        features = FeatureRow(
            date=pd.Timestamp("2024-02-01"),
            symbol="XLE",
            asset_class="equity",
            close=102.0,  # Above stop
            open=101.0,
            high=103.0,
            low=100.0,
            atr14=2.0,
        )

        exit_reason = strategy.check_exit_signals(position_xle, features)

        assert exit_reason == ExitReason.MANUAL  # Time stop

    def test_check_exit_signals_no_exit(self, strategy, position_xle):
        """Test no exit when conditions not met."""
        features = FeatureRow(
            date=pd.Timestamp("2024-02-01"),
            symbol="XLE",
            asset_class="equity",
            close=102.0,  # Above stop
            open=101.0,
            high=103.0,
            low=100.0,
            atr14=2.0,
        )

        exit_reason = strategy.check_exit_signals(position_xle, features)

        assert exit_reason is None


class TestStopUpdates:
    """Tests for stop price updates."""

    @pytest.fixture
    def strategy(self):
        """Create pairs strategy."""
        config = StrategyConfig(
            name="equity_pairs",
            asset_class="equity",
            universe=[],
            benchmark="SPY",
            parameters={
                "pairs": [["XLE", "XLK"]],
                "lookback": 60,
                "entry_zscore": 2.0,
                "exit_zscore": 0.5,
                "max_hold_days": 10,
                "atr_period": 14,
                "stop_atr_mult": 2.0,
            },
        )
        return PairsTradingStrategy(config)

    def test_no_stop_update(self, strategy):
        """Test that pairs trading doesn't update stops."""
        position = Position(
            symbol="XLE",
            asset_class="equity",
            entry_date=pd.Timestamp("2024-01-25"),
            entry_price=100.0,
            entry_fill_id="fill_1",
            quantity=100,
            stop_price=96.0,
            initial_stop_price=96.0,
            hard_stop_atr_mult=2.0,
            entry_slippage_bps=8.0,
            entry_fee_bps=1.0,
            entry_total_cost=90.0,
            triggered_on=BreakoutType.FAST_20D,
            adv20_at_entry=50_000_000.0,
        )

        features = FeatureRow(
            date=pd.Timestamp("2024-02-01"),
            symbol="XLE",
            asset_class="equity",
            close=102.0,
            open=101.0,
            high=103.0,
            low=100.0,
            atr14=2.0,
        )

        new_stop = strategy.update_stop_price(position, features)

        assert new_stop is None  # Pairs trading doesn't update stops


class TestRequiredHistory:
    """Tests for required history days."""

    def test_required_history_days(self):
        """Test required history calculation."""
        config = StrategyConfig(
            name="equity_pairs",
            asset_class="equity",
            universe=[],
            benchmark="SPY",
            parameters={
                "pairs": [["XLE", "XLK"]],
                "lookback": 60,
                "entry_zscore": 2.0,
                "exit_zscore": 0.5,
                "max_hold_days": 10,
                "atr_period": 14,
                "stop_atr_mult": 2.0,
            },
        )
        strategy = PairsTradingStrategy(config)

        required = strategy.get_required_history_days()

        assert required == 80  # lookback (60) + buffer (20)

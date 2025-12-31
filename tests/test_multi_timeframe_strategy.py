"""Unit tests for equity multi-timeframe strategy."""

import pandas as pd
import pytest

from tests.utils.test_helpers import create_sample_feature_row, create_sample_position
from trading_system.configs.strategy_config import StrategyConfig
from trading_system.models.positions import ExitReason
from trading_system.models.signals import BreakoutType, SignalSide
from trading_system.strategies.multi_timeframe.equity_mtf_strategy import EquityMultiTimeframeStrategy


class TestEquityMultiTimeframeStrategyInit:
    """Tests for EquityMultiTimeframeStrategy initialization."""

    def test_init_valid(self):
        """Test initialization with valid equity config."""
        config = StrategyConfig(
            name="equity_mtf",
            asset_class="equity",
            universe="NASDAQ-100",
            benchmark="SPY",
            parameters={
                "min_adv20": 10_000_000,
                "stop_atr_mult": 2.5,
                "max_hold_days": 60,
            },
        )
        strategy = EquityMultiTimeframeStrategy(config)
        assert strategy.config == config
        assert strategy.asset_class == "equity"
        assert strategy.min_adv20 == 10_000_000
        assert strategy.stop_atr_mult == 2.5
        assert strategy.max_hold_days == 60

    def test_init_invalid_asset_class(self):
        """Test initialization fails with non-equity config."""
        config = StrategyConfig(
            name="crypto_mtf",
            asset_class="crypto",
            universe="fixed",
            benchmark="BTC",
        )
        with pytest.raises(ValueError, match="asset_class='equity'"):
            EquityMultiTimeframeStrategy(config)

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        config = StrategyConfig(
            name="equity_mtf",
            asset_class="equity",
            universe="NASDAQ-100",
            benchmark="SPY",
            parameters={},  # Empty dict will use defaults
        )
        strategy = EquityMultiTimeframeStrategy(config)
        assert strategy.min_adv20 == 10_000_000  # Default


class TestEligibilityChecks:
    """Tests for eligibility checking."""

    @pytest.fixture
    def strategy(self):
        """Create equity multi-timeframe strategy."""
        config = StrategyConfig(
            name="equity_mtf",
            asset_class="equity",
            universe="NASDAQ-100",
            benchmark="SPY",
            parameters={
                "min_adv20": 10_000_000,
            },
        )
        return EquityMultiTimeframeStrategy(config)

    def test_eligibility_valid(self, strategy):
        """Test eligibility with valid features."""
        features = create_sample_feature_row(
            date=pd.Timestamp("2024-01-15"),
            symbol="AAPL",
            close=150.0,
            ma50=145.0,  # Required for MTF
            highest_close_55d=148.0,  # Weekly breakout level
            atr14=3.0,
            adv20=20_000_000.0,
        )

        is_eligible, failures = strategy.check_eligibility(features)
        assert is_eligible
        assert len(failures) == 0

    def test_eligibility_insufficient_data(self, strategy):
        """Test eligibility fails with insufficient data."""
        # Can't use close=0.0 due to FeatureRow validation, so test with missing ATR
        features = create_sample_feature_row(
            date=pd.Timestamp("2024-01-15"),
            symbol="AAPL",
            close=150.0,  # Valid close
            atr14=None,  # Missing ATR - will cause insufficient_data
            allow_none_atr14=True,  # Allow None for testing
            adv20=20_000_000.0,
        )

        is_eligible, failures = strategy.check_eligibility(features)
        assert not is_eligible
        assert "insufficient_data" in failures or "atr14_missing" in failures

    def test_eligibility_missing_ma50(self, strategy):
        """Test eligibility fails when MA50 is missing."""
        features = create_sample_feature_row(
            date=pd.Timestamp("2024-01-15"),
            symbol="AAPL",
            close=150.0,
            ma50=None,  # Missing
            allow_none_ma50=True,  # Allow None for testing
            atr14=3.0,
            adv20=20_000_000.0,
        )

        is_eligible, failures = strategy.check_eligibility(features)
        assert not is_eligible
        assert "ma50_missing" in failures

    def test_eligibility_missing_weekly_breakout(self, strategy):
        """Test eligibility fails when weekly breakout level is missing."""
        features = create_sample_feature_row(
            date=pd.Timestamp("2024-01-15"),
            symbol="AAPL",
            close=150.0,
            ma50=145.0,
            highest_close_55d=None,  # Missing
            atr14=3.0,
            adv20=20_000_000.0,
        )

        is_eligible, failures = strategy.check_eligibility(features)
        assert not is_eligible
        assert "weekly_breakout_missing" in failures

    def test_eligibility_insufficient_liquidity(self, strategy):
        """Test eligibility fails with insufficient liquidity."""
        features = create_sample_feature_row(
            date=pd.Timestamp("2024-01-15"),
            symbol="AAPL",
            close=150.0,
            ma50=145.0,
            highest_close_55d=148.0,
            atr14=3.0,
            adv20=5_000_000.0,  # Below minimum
        )

        is_eligible, failures = strategy.check_eligibility(features)
        assert not is_eligible
        assert any("insufficient_liquidity" in f for f in failures)


class TestEntryTriggers:
    """Tests for entry trigger checking."""

    @pytest.fixture
    def strategy(self):
        """Create equity multi-timeframe strategy."""
        config = StrategyConfig(
            name="equity_mtf",
            asset_class="equity",
            universe="NASDAQ-100",
            benchmark="SPY",
        )
        return EquityMultiTimeframeStrategy(config)

    def test_entry_trigger_valid(self, strategy):
        """Test entry trigger when conditions are met."""
        features = create_sample_feature_row(
            date=pd.Timestamp("2024-01-15"),
            symbol="AAPL",
            close=150.0,
            ma50=145.0,  # Close > MA50 (trend filter)
            highest_close_55d=148.0,  # Close >= breakout level
            atr14=3.0,
        )

        breakout_type, clearance = strategy.check_entry_triggers(features)
        assert breakout_type == BreakoutType.SLOW_55D
        assert clearance > 0  # Positive clearance

    def test_entry_trigger_below_ma50(self, strategy):
        """Test entry trigger fails when price below MA50."""
        features = create_sample_feature_row(
            date=pd.Timestamp("2024-01-15"),
            symbol="AAPL",
            close=140.0,  # Below MA50
            ma50=145.0,
            highest_close_55d=148.0,
            atr14=3.0,
        )

        breakout_type, clearance = strategy.check_entry_triggers(features)
        assert breakout_type is None
        assert clearance == 0.0

    def test_entry_trigger_below_breakout(self, strategy):
        """Test entry trigger fails when price below breakout level."""
        features = create_sample_feature_row(
            date=pd.Timestamp("2024-01-15"),
            symbol="AAPL",
            close=147.0,  # Below breakout
            ma50=145.0,
            highest_close_55d=148.0,
            atr14=3.0,
        )

        breakout_type, clearance = strategy.check_entry_triggers(features)
        assert breakout_type is None
        assert clearance == 0.0

    def test_entry_trigger_missing_ma50(self, strategy):
        """Test entry trigger fails when MA50 is missing."""
        features = create_sample_feature_row(
            date=pd.Timestamp("2024-01-15"),
            symbol="AAPL",
            close=150.0,
            ma50=None,  # Missing
            allow_none_ma50=True,  # Allow None for testing
            highest_close_55d=148.0,
            atr14=3.0,
        )

        breakout_type, clearance = strategy.check_entry_triggers(features)
        assert breakout_type is None
        assert clearance == 0.0


class TestSignalGeneration:
    """Tests for signal generation."""

    @pytest.fixture
    def strategy(self):
        """Create equity multi-timeframe strategy."""
        config = StrategyConfig(
            name="equity_mtf",
            asset_class="equity",
            universe="NASDAQ-100",
            benchmark="SPY",
            parameters={
                "stop_atr_mult": 2.5,
            },
        )
        return EquityMultiTimeframeStrategy(config)

    def test_generate_signal_valid(self, strategy):
        """Test signal generation with valid conditions."""
        features = create_sample_feature_row(
            date=pd.Timestamp("2024-01-15"),
            symbol="AAPL",
            close=150.0,
            ma50=145.0,
            highest_close_55d=148.0,
            atr14=3.0,
            adv20=20_000_000.0,
        )

        signal = strategy.generate_signal(symbol="AAPL", features=features, order_notional=100000.0, diversification_bonus=0.5)

        assert signal is not None
        assert signal.symbol == "AAPL"
        assert signal.side == SignalSide.BUY
        assert signal.entry_price == 150.0
        assert signal.stop_price < 150.0  # Stop below entry
        assert "mtf_weekly_breakout" in signal.trigger_reason
        assert "clearance" in signal.metadata

    def test_generate_signal_not_eligible(self, strategy):
        """Test signal generation fails when not eligible."""
        features = create_sample_feature_row(
            date=pd.Timestamp("2024-01-15"),
            symbol="AAPL",
            close=150.0,
            ma50=None,  # Missing - will fail eligibility
            allow_none_ma50=True,  # Allow None for testing
            atr14=3.0,
            adv20=20_000_000.0,
        )

        signal = strategy.generate_signal(
            symbol="AAPL",
            features=features,
            order_notional=100000.0,
        )

        assert signal is None

    def test_generate_signal_no_trigger(self, strategy):
        """Test signal generation fails when trigger not met."""
        features = create_sample_feature_row(
            date=pd.Timestamp("2024-01-15"),
            symbol="AAPL",
            close=140.0,  # Below MA50
            ma50=145.0,
            highest_close_55d=148.0,
            atr14=3.0,
            adv20=20_000_000.0,
        )

        signal = strategy.generate_signal(
            symbol="AAPL",
            features=features,
            order_notional=100000.0,
        )

        assert signal is None


class TestExitSignals:
    """Tests for exit signal checking."""

    @pytest.fixture
    def strategy(self):
        """Create equity multi-timeframe strategy."""
        config = StrategyConfig(
            name="equity_mtf",
            asset_class="equity",
            universe="NASDAQ-100",
            benchmark="SPY",
            parameters={
                "max_hold_days": 60,
                "stop_atr_mult": 2.5,
            },
        )
        return EquityMultiTimeframeStrategy(config)

    def test_exit_hard_stop(self, strategy):
        """Test exit on hard stop."""
        position = create_sample_position(
            symbol="AAPL",
            asset_class="equity",
            entry_date=pd.Timestamp("2024-01-01"),
            entry_price=150.0,
            quantity=100,
            stop_price=145.0,
        )

        features = create_sample_feature_row(
            date=pd.Timestamp("2024-01-10"),
            symbol="AAPL",
            close=144.0,  # Below stop
            ma50=145.0,
            atr14=3.0,
        )

        exit_reason = strategy.check_exit_signals(position, features)
        assert exit_reason == ExitReason.HARD_STOP

    def test_exit_trend_break(self, strategy):
        """Test exit when higher timeframe trend breaks (price < MA50)."""
        position = create_sample_position(
            symbol="AAPL",
            asset_class="equity",
            entry_date=pd.Timestamp("2024-01-01"),
            entry_price=150.0,
            quantity=100,
            stop_price=140.0,  # Lower stop so hard stop doesn't trigger first
        )

        features = create_sample_feature_row(
            date=pd.Timestamp("2024-01-15"),
            symbol="AAPL",
            close=144.0,  # Below MA50 (trend break), but above stop_price
            ma50=145.0,
            atr14=3.0,
        )

        exit_reason = strategy.check_exit_signals(position, features)
        assert exit_reason == ExitReason.TRAILING_MA_CROSS

    def test_exit_time_stop(self, strategy):
        """Test exit on time stop (max hold days)."""
        position = create_sample_position(
            symbol="AAPL",
            asset_class="equity",
            entry_date=pd.Timestamp("2024-01-01"),
            entry_price=150.0,
            quantity=100,
            stop_price=145.0,
        )

        # 61 days later (exceeds max_hold_days=60)
        features = create_sample_feature_row(
            date=pd.Timestamp("2024-03-01"),
            symbol="AAPL",
            close=155.0,  # Above stop and MA50
            ma50=150.0,
            atr14=3.0,
        )

        exit_reason = strategy.check_exit_signals(position, features)
        assert exit_reason == ExitReason.MANUAL  # Time-based exit

    def test_exit_no_exit(self, strategy):
        """Test no exit when conditions not met."""
        position = create_sample_position(
            symbol="AAPL",
            asset_class="equity",
            entry_date=pd.Timestamp("2024-01-01"),
            entry_price=150.0,
            quantity=100,
            stop_price=145.0,
        )

        features = create_sample_feature_row(
            date=pd.Timestamp("2024-01-15"),
            symbol="AAPL",
            close=155.0,  # Above stop and MA50
            ma50=150.0,
            atr14=3.0,
        )

        exit_reason = strategy.check_exit_signals(position, features)
        assert exit_reason is None

    def test_exit_trend_break_missing_ma50(self, strategy):
        """Test exit logic when MA50 is missing (should not exit)."""
        position = create_sample_position(
            symbol="AAPL",
            asset_class="equity",
            entry_date=pd.Timestamp("2024-01-01"),
            entry_price=150.0,
            quantity=100,
            stop_price=145.0,
        )

        features = create_sample_feature_row(
            date=pd.Timestamp("2024-01-15"),
            symbol="AAPL",
            close=155.0,
            ma50=None,  # Missing MA50
            atr14=3.0,
        )

        exit_reason = strategy.check_exit_signals(position, features)
        # Should not exit on trend break if MA50 is missing
        assert exit_reason is None


class TestStopPriceUpdates:
    """Tests for stop price updates."""

    @pytest.fixture
    def strategy(self):
        """Create equity multi-timeframe strategy."""
        config = StrategyConfig(
            name="equity_mtf",
            asset_class="equity",
            universe="NASDAQ-100",
            benchmark="SPY",
        )
        return EquityMultiTimeframeStrategy(config)

    def test_update_stop_price_fixed_stop(self, strategy):
        """Test that MTF strategy uses fixed stops (no updates for now)."""
        position = create_sample_position(
            symbol="AAPL",
            asset_class="equity",
            entry_date=pd.Timestamp("2024-01-01"),
            entry_price=150.0,
            quantity=100,
            stop_price=145.0,
        )

        features = create_sample_feature_row(
            date=pd.Timestamp("2024-01-15"),
            symbol="AAPL",
            close=160.0,  # Price moved up
            ma50=155.0,
            atr14=3.0,
        )

        new_stop = strategy.update_stop_price(position, features)
        # MTF strategy currently uses fixed stops
        assert new_stop is None
        assert position.stop_price == 145.0  # Unchanged

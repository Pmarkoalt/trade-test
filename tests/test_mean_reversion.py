"""Unit tests for mean reversion strategy."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from trading_system.strategies.mean_reversion.equity_mean_reversion import EquityMeanReversionStrategy
from trading_system.configs.strategy_config import (
    StrategyConfig,
    EligibilityConfig,
    EntryConfig,
    ExitConfig,
    RiskConfig,
    CapacityConfig,
    CostsConfig,
    IndicatorsConfig,
)
from trading_system.models.features import FeatureRow
from trading_system.models.signals import Signal, SignalSide, SignalType, BreakoutType
from trading_system.models.positions import Position, ExitReason


class TestMeanReversionStrategyInit:
    """Tests for EquityMeanReversionStrategy initialization."""
    
    def test_init_valid(self):
        """Test initialization with valid equity config."""
        config = StrategyConfig(
            name="equity_mean_reversion",
            asset_class="equity",
            universe=["SPY", "QQQ"],
            benchmark="SPY",
            parameters={
                'lookback': 20,
                'entry_std': 2.0,
                'exit_std': 0.0,
                'max_hold_days': 5,
                'atr_period': 14,
                'stop_atr_mult': 2.0,
                'min_adv20': 10_000_000,
            }
        )
        strategy = EquityMeanReversionStrategy(config)
        assert strategy.config == config
        assert strategy.asset_class == "equity"
        assert strategy.lookback == 20
        assert strategy.entry_std == 2.0
        assert strategy.exit_std == 0.0
        assert strategy.max_hold_days == 5
    
    def test_init_invalid_asset_class(self):
        """Test initialization fails with non-equity config."""
        config = StrategyConfig(
            name="crypto_mean_reversion",
            asset_class="crypto",
            universe=["BTC", "ETH"],
            benchmark="BTC",
        )
        with pytest.raises(ValueError, match="asset_class='equity'"):
            EquityMeanReversionStrategy(config)


class TestEligibilityFilters:
    """Tests for eligibility filter logic."""
    
    @pytest.fixture
    def strategy(self):
        """Create mean reversion strategy with default config."""
        config = StrategyConfig(
            name="equity_mean_reversion",
            asset_class="equity",
            universe=["SPY", "QQQ"],
            benchmark="SPY",
            parameters={
                'lookback': 20,
                'entry_std': 2.0,
                'exit_std': 0.0,
                'max_hold_days': 5,
                'atr_period': 14,
                'stop_atr_mult': 2.0,
                'min_adv20': 10_000_000,
            }
        )
        return EquityMeanReversionStrategy(config)
    
    @pytest.fixture
    def valid_features(self):
        """Create valid feature row with oversold condition."""
        return FeatureRow(
            date=pd.Timestamp('2024-02-01'),
            symbol='SPY',
            asset_class='equity',
            close=100.0,
            open=101.0,
            high=102.0,
            low=99.0,
            ma20=105.0,
            ma50=110.0,
            ma200=115.0,
            atr14=2.0,
            roc60=-0.05,
            highest_close_20d=110.0,
            highest_close_55d=115.0,
            adv20=50_000_000.0,  # Above minimum
            returns_1d=-0.01,
            ma50_slope=-0.01,
            benchmark_roc60=-0.03,
            benchmark_returns_1d=-0.005,
            # Mean reversion indicators
            zscore=-2.5,  # Oversold (below -2.0 threshold)
            ma_lookback=105.0,  # 20-day mean
            std_lookback=2.0,  # 20-day std
        )
    
    def test_eligibility_passes_oversold(self, strategy, valid_features):
        """Test eligibility passes when zscore < -entry_std."""
        is_eligible, failures = strategy.check_eligibility(valid_features)
        
        assert is_eligible
        assert len(failures) == 0
    
    def test_eligibility_fails_missing_zscore(self, strategy):
        """Test eligibility fails when zscore is missing."""
        features = FeatureRow(
            date=pd.Timestamp('2024-02-01'),
            symbol='SPY',
            asset_class='equity',
            close=100.0,
            open=101.0,
            high=102.0,
            low=99.0,
            atr14=2.0,
            adv20=50_000_000.0,
            zscore=None,  # Missing
        )
        
        is_eligible, failures = strategy.check_eligibility(features)
        
        assert not is_eligible
        assert 'zscore_missing' in failures
    
    def test_eligibility_fails_insufficient_liquidity(self, strategy):
        """Test eligibility fails when ADV20 below minimum."""
        features = FeatureRow(
            date=pd.Timestamp('2024-02-01'),
            symbol='SPY',
            asset_class='equity',
            close=100.0,
            open=101.0,
            high=102.0,
            low=99.0,
            atr14=2.0,
            adv20=5_000_000.0,  # Below minimum
            zscore=-2.5,
        )
        
        is_eligible, failures = strategy.check_eligibility(features)
        
        assert not is_eligible
        assert any('insufficient_liquidity' in f for f in failures)
    
    def test_eligibility_fails_missing_atr(self, strategy):
        """Test eligibility fails when ATR14 is missing."""
        features = FeatureRow(
            date=pd.Timestamp('2024-02-01'),
            symbol='SPY',
            asset_class='equity',
            close=100.0,
            open=101.0,
            high=102.0,
            low=99.0,
            atr14=None,  # Missing
            adv20=50_000_000.0,
            zscore=-2.5,
        )
        
        is_eligible, failures = strategy.check_eligibility(features)
        
        assert not is_eligible
        assert 'atr14_missing' in failures


class TestEntryTriggers:
    """Tests for entry trigger logic."""
    
    @pytest.fixture
    def strategy(self):
        """Create mean reversion strategy."""
        config = StrategyConfig(
            name="equity_mean_reversion",
            asset_class="equity",
            universe=["SPY", "QQQ"],
            benchmark="SPY",
            parameters={
                'lookback': 20,
                'entry_std': 2.0,
                'exit_std': 0.0,
                'max_hold_days': 5,
                'atr_period': 14,
                'stop_atr_mult': 2.0,
                'min_adv20': 10_000_000,
            }
        )
        return EquityMeanReversionStrategy(config)
    
    def test_entry_trigger_oversold(self, strategy):
        """Test entry trigger when zscore < -entry_std."""
        features = FeatureRow(
            date=pd.Timestamp('2024-02-01'),
            symbol='SPY',
            asset_class='equity',
            close=100.0,
            open=101.0,
            high=102.0,
            low=99.0,
            atr14=2.0,
            adv20=50_000_000.0,
            zscore=-2.5,  # Below -2.0 threshold
        )
        
        breakout_type, zscore = strategy.check_entry_triggers(features)
        
        assert breakout_type is not None
        assert zscore == -2.5
    
    def test_entry_trigger_not_oversold(self, strategy):
        """Test no entry trigger when zscore >= -entry_std."""
        features = FeatureRow(
            date=pd.Timestamp('2024-02-01'),
            symbol='SPY',
            asset_class='equity',
            close=100.0,
            open=101.0,
            high=102.0,
            low=99.0,
            atr14=2.0,
            adv20=50_000_000.0,
            zscore=-1.5,  # Above -2.0 threshold
        )
        
        breakout_type, zscore = strategy.check_entry_triggers(features)
        
        assert breakout_type is None
        assert zscore == -1.5
    
    def test_entry_trigger_missing_zscore(self, strategy):
        """Test no entry trigger when zscore is missing."""
        features = FeatureRow(
            date=pd.Timestamp('2024-02-01'),
            symbol='SPY',
            asset_class='equity',
            close=100.0,
            open=101.0,
            high=102.0,
            low=99.0,
            atr14=2.0,
            adv20=50_000_000.0,
            zscore=None,
        )
        
        breakout_type, zscore = strategy.check_entry_triggers(features)
        
        assert breakout_type is None
        assert zscore == 0.0


class TestSignalGeneration:
    """Tests for signal generation."""
    
    @pytest.fixture
    def strategy(self):
        """Create mean reversion strategy."""
        config = StrategyConfig(
            name="equity_mean_reversion",
            asset_class="equity",
            universe=["SPY", "QQQ"],
            benchmark="SPY",
            parameters={
                'lookback': 20,
                'entry_std': 2.0,
                'exit_std': 0.0,
                'max_hold_days': 5,
                'atr_period': 14,
                'stop_atr_mult': 2.0,
                'min_adv20': 10_000_000,
            }
        )
        return EquityMeanReversionStrategy(config)
    
    def test_generate_signal_oversold(self, strategy):
        """Test signal generation for oversold condition."""
        features = FeatureRow(
            date=pd.Timestamp('2024-02-01'),
            symbol='SPY',
            asset_class='equity',
            close=100.0,
            open=101.0,
            high=102.0,
            low=99.0,
            ma20=105.0,
            ma50=110.0,
            ma200=115.0,
            atr14=2.0,
            roc60=-0.05,
            highest_close_20d=110.0,
            highest_close_55d=115.0,
            adv20=50_000_000.0,
            returns_1d=-0.01,
            ma50_slope=-0.01,
            benchmark_roc60=-0.03,
            benchmark_returns_1d=-0.005,
            zscore=-2.5,
            ma_lookback=105.0,
            std_lookback=2.0,
        )
        
        order_notional = 10_000.0
        signal = strategy.generate_signal('SPY', features, order_notional)
        
        assert signal is not None
        assert signal.symbol == 'SPY'
        assert signal.signal_type == SignalType.ENTRY_LONG
        assert signal.trigger_reason.startswith('mean_reversion_oversold')
        assert signal.urgency == 0.7
        assert signal.entry_price == 100.0
        assert signal.stop_price == 96.0  # 100 - 2.0 * 2.0
        assert signal.metadata['zscore'] == -2.5
        assert signal.metadata['entry_std'] == 2.0
        assert signal.metadata['exit_std'] == 0.0
        assert signal.passed_eligibility
        assert signal.capacity_passed
    
    def test_generate_signal_not_oversold(self, strategy):
        """Test no signal when not oversold."""
        features = FeatureRow(
            date=pd.Timestamp('2024-02-01'),
            symbol='SPY',
            asset_class='equity',
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
            zscore=-1.5,  # Not oversold enough
            ma_lookback=105.0,
            std_lookback=2.0,
        )
        
        order_notional = 10_000.0
        signal = strategy.generate_signal('SPY', features, order_notional)
        
        assert signal is None


class TestExitSignals:
    """Tests for exit signal logic."""
    
    @pytest.fixture
    def strategy(self):
        """Create mean reversion strategy."""
        config = StrategyConfig(
            name="equity_mean_reversion",
            asset_class="equity",
            universe=["SPY", "QQQ"],
            benchmark="SPY",
            parameters={
                'lookback': 20,
                'entry_std': 2.0,
                'exit_std': 0.0,
                'max_hold_days': 5,
                'atr_period': 14,
                'stop_atr_mult': 2.0,
                'min_adv20': 10_000_000,
            }
        )
        return EquityMeanReversionStrategy(config)
    
    @pytest.fixture
    def position(self):
        """Create a test position."""
        return Position(
            symbol="SPY",
            asset_class="equity",
            entry_date=pd.Timestamp('2024-01-25'),
            entry_price=100.0,
            entry_fill_id="fill_1",
            quantity=100,
            stop_price=96.0,  # entry - 2.0 * ATR
            initial_stop_price=96.0,
            hard_stop_atr_mult=2.0,
            entry_slippage_bps=5.0,
            entry_fee_bps=1.0,
            entry_total_cost=6.0,
            triggered_on=BreakoutType.FAST_20D,
            adv20_at_entry=1000000.0,
        )
    
    def test_exit_hard_stop(self, strategy, position):
        """Test exit on hard stop."""
        features = FeatureRow(
            date=pd.Timestamp('2024-02-01'),
            symbol='SPY',
            asset_class='equity',
            close=95.0,  # Below stop
            open=96.0,
            high=97.0,
            low=94.0,
            atr14=2.0,
            zscore=-1.0,
        )
        
        exit_reason = strategy.check_exit_signals(position, features)
        
        assert exit_reason == ExitReason.HARD_STOP
    
    def test_exit_mean_reversion_target(self, strategy, position):
        """Test exit when price reverted to mean (zscore >= -exit_std)."""
        features = FeatureRow(
            date=pd.Timestamp('2024-02-01'),
            symbol='SPY',
            asset_class='equity',
            close=105.0,  # Above stop
            open=104.0,
            high=106.0,
            low=103.0,
            atr14=2.0,
            zscore=0.5,  # Reverted to mean (above 0.0)
        )
        
        exit_reason = strategy.check_exit_signals(position, features)
        
        assert exit_reason == ExitReason.TRAILING_MA_CROSS
    
    def test_exit_time_stop(self, strategy, position):
        """Test exit on time stop (max_hold_days)."""
        # Position entered 6 days ago (exceeds max_hold_days=5)
        position.entry_date = pd.Timestamp('2024-01-26')
        
        features = FeatureRow(
            date=pd.Timestamp('2024-02-01'),
            symbol='SPY',
            asset_class='equity',
            close=102.0,  # Above stop
            open=101.0,
            high=103.0,
            low=100.0,
            atr14=2.0,
            zscore=-1.5,  # Still oversold
        )
        
        exit_reason = strategy.check_exit_signals(position, features)
        
        assert exit_reason == ExitReason.MANUAL  # Time stop uses MANUAL
    
    def test_no_exit(self, strategy, position):
        """Test no exit when conditions not met."""
        features = FeatureRow(
            date=pd.Timestamp('2024-02-01'),
            symbol='SPY',
            asset_class='equity',
            close=102.0,  # Above stop
            open=101.0,
            high=103.0,
            low=100.0,
            atr14=2.0,
            zscore=-1.5,  # Still oversold (below exit threshold)
        )
        
        exit_reason = strategy.check_exit_signals(position, features)
        
        assert exit_reason is None


class TestStopUpdates:
    """Tests for stop price updates."""
    
    @pytest.fixture
    def strategy(self):
        """Create mean reversion strategy."""
        config = StrategyConfig(
            name="equity_mean_reversion",
            asset_class="equity",
            universe=["SPY", "QQQ"],
            benchmark="SPY",
            parameters={
                'lookback': 20,
                'entry_std': 2.0,
                'exit_std': 0.0,
                'max_hold_days': 5,
                'atr_period': 14,
                'stop_atr_mult': 2.0,
                'min_adv20': 10_000_000,
            }
        )
        return EquityMeanReversionStrategy(config)
    
    def test_no_stop_update(self, strategy):
        """Test that mean reversion doesn't update stops."""
        position = Position(
            symbol="SPY",
            asset_class="equity",
            entry_date=pd.Timestamp('2024-01-25'),
            entry_price=100.0,
            entry_fill_id="fill_1",
            quantity=100,
            stop_price=96.0,
            initial_stop_price=96.0,
            hard_stop_atr_mult=2.0,
            entry_slippage_bps=5.0,
            entry_fee_bps=1.0,
            entry_total_cost=6.0,
            triggered_on=BreakoutType.FAST_20D,
            adv20_at_entry=1000000.0,
        )
        
        features = FeatureRow(
            date=pd.Timestamp('2024-02-01'),
            symbol='SPY',
            asset_class='equity',
            close=102.0,
            open=101.0,
            high=103.0,
            low=100.0,
            atr14=2.0,
            zscore=-1.0,
        )
        
        new_stop = strategy.update_stop_price(position, features)
        
        assert new_stop is None  # Mean reversion doesn't update stops


class TestRequiredHistory:
    """Tests for required history days."""
    
    def test_required_history_days(self):
        """Test required history calculation."""
        config = StrategyConfig(
            name="equity_mean_reversion",
            asset_class="equity",
            universe=["SPY", "QQQ"],
            benchmark="SPY",
            parameters={
                'lookback': 20,
                'entry_std': 2.0,
                'exit_std': 0.0,
                'max_hold_days': 5,
                'atr_period': 14,
                'stop_atr_mult': 2.0,
                'min_adv20': 10_000_000,
            }
        )
        strategy = EquityMeanReversionStrategy(config)
        
        required = strategy.get_required_history_days()
        
        assert required == 40  # lookback (20) + buffer (20)


"""Unit tests for equity momentum strategy."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from trading_system.strategies import EquityStrategy
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
from trading_system.models import (
    FeatureRow, Signal, SignalSide, BreakoutType,
    Position, ExitReason
)


class TestEquityStrategyInit:
    """Tests for EquityStrategy initialization."""
    
    def test_init_valid(self):
        """Test initialization with valid equity config."""
        config = StrategyConfig(
            name="equity_momentum",
            asset_class="equity",
            universe="NASDAQ-100",
            benchmark="SPY",
        )
        strategy = EquityStrategy(config)
        assert strategy.config == config
        assert strategy.asset_class == "equity"
    
    def test_init_invalid_asset_class(self):
        """Test initialization fails with non-equity config."""
        config = StrategyConfig(
            name="crypto_momentum",
            asset_class="crypto",
            universe="fixed",
            benchmark="BTC",
        )
        with pytest.raises(ValueError, match="asset_class='equity'"):
            EquityStrategy(config)


class TestEligibilityFilters:
    """Tests for eligibility filter logic."""
    
    @pytest.fixture
    def strategy(self):
        """Create equity strategy with default config."""
        config = StrategyConfig(
            name="equity_momentum",
            asset_class="equity",
            universe="NASDAQ-100",
            benchmark="SPY",
        )
        return EquityStrategy(config)
    
    @pytest.fixture
    def valid_features(self):
        """Create valid feature row."""
        return FeatureRow(
            date=pd.Timestamp('2024-02-01'),
            symbol='AAPL',
            asset_class='equity',
            close=105.0,
            open=104.5,
            high=106.0,
            low=104.0,
            ma20=103.0,
            ma50=100.0,  # Close > MA50
            ma200=95.0,
            atr14=2.0,
            roc60=0.05,
            highest_close_20d=104.0,
            highest_close_55d=102.0,
            adv20=100000000.0,
            returns_1d=0.001,
            ma50_slope=0.01,  # 1% slope (above 0.5% threshold)
            benchmark_roc60=0.03,
            benchmark_returns_1d=0.0005,
        )
    
    def test_eligibility_close_above_ma50(self, strategy, valid_features):
        """Test eligibility passes when close > MA50."""
        is_eligible, failures = strategy.check_eligibility(valid_features)
        
        assert is_eligible
        assert len(failures) == 0
        assert 'below_MA50' not in failures
    
    def test_eligibility_close_below_ma50(self, strategy):
        """Test eligibility fails when close <= MA50."""
        features = FeatureRow(
            date=pd.Timestamp('2024-02-01'),
            symbol='AAPL',
            asset_class='equity',
            close=98.0,  # Below MA50
            open=97.5,
            high=99.0,
            low=97.0,
            ma20=100.0,
            ma50=100.0,
            ma200=95.0,
            atr14=2.0,
            roc60=0.05,
            highest_close_20d=99.0,
            highest_close_55d=98.0,
            adv20=100000000.0,
            returns_1d=0.001,
            ma50_slope=0.01,  # 1% slope (above 0.5% threshold)
            benchmark_roc60=0.03,
            benchmark_returns_1d=0.0005,
        )
        
        is_eligible, failures = strategy.check_eligibility(features)
        
        assert not is_eligible
        assert 'below_MA50' in failures
    
    def test_eligibility_insufficient_data(self, strategy):
        """Test eligibility fails when features are incomplete."""
        features = FeatureRow(
            date=pd.Timestamp('2024-02-01'),
            symbol='AAPL',
            asset_class='equity',
            close=105.0,
            open=104.5,
            high=106.0,
            low=104.0,
            ma20=None,  # Missing MA20
            ma50=100.0,
            ma200=None,
            atr14=None,  # Missing ATR
            roc60=None,
            highest_close_20d=None,  # Missing breakout levels
            highest_close_55d=None,
            adv20=None,  # Missing ADV
            returns_1d=None,
            ma50_slope=None,
            benchmark_roc60=None,
            benchmark_returns_1d=None,
        )
        
        is_eligible, failures = strategy.check_eligibility(features)
        
        assert not is_eligible
        assert 'insufficient_data' in failures
    
    def test_eligibility_ma50_slope_sufficient(self, strategy, valid_features):
        """Test eligibility passes when MA50 slope > 0.5%."""
        # valid_features has ma50_slope=0.01 (1%), which is > 0.005 (0.5%)
        is_eligible, failures = strategy.check_eligibility(valid_features)
        
        assert is_eligible
        assert len(failures) == 0
        assert 'insufficient_ma50_slope' not in ' '.join(failures)
    
    def test_eligibility_ma50_slope_insufficient(self, strategy):
        """Test eligibility fails when MA50 slope <= 0.5%."""
        features = FeatureRow(
            date=pd.Timestamp('2024-02-01'),
            symbol='AAPL',
            asset_class='equity',
            close=105.0,
            open=104.5,
            high=106.0,
            low=104.0,
            ma20=103.0,
            ma50=100.0,
            ma200=95.0,
            atr14=2.0,
            roc60=0.05,
            highest_close_20d=104.0,
            highest_close_55d=102.0,
            adv20=100000000.0,
            returns_1d=0.001,
            ma50_slope=0.003,  # 0.3% slope (below 0.5% threshold)
            benchmark_roc60=0.03,
            benchmark_returns_1d=0.0005,
        )
        
        is_eligible, failures = strategy.check_eligibility(features)
        
        assert not is_eligible
        assert any('insufficient_ma50_slope' in f for f in failures)
    
    def test_eligibility_ma50_slope_missing(self, strategy):
        """Test eligibility fails when MA50 slope is missing."""
        features = FeatureRow(
            date=pd.Timestamp('2024-02-01'),
            symbol='AAPL',
            asset_class='equity',
            close=105.0,
            open=104.5,
            high=106.0,
            low=104.0,
            ma20=103.0,
            ma50=100.0,
            ma200=95.0,
            atr14=2.0,
            roc60=0.05,
            highest_close_20d=104.0,
            highest_close_55d=102.0,
            adv20=100000000.0,
            returns_1d=0.001,
            ma50_slope=None,  # Missing slope
            benchmark_roc60=0.03,
            benchmark_returns_1d=0.0005,
        )
        
        is_eligible, failures = strategy.check_eligibility(features)
        
        assert not is_eligible
        assert 'ma50_slope_missing' in failures
    
    def test_eligibility_relative_strength_enabled(self, strategy):
        """Test eligibility with relative strength check enabled."""
        config = StrategyConfig(
            name="equity_momentum",
            asset_class="equity",
            universe="NASDAQ-100",
            benchmark="SPY",
            eligibility=EligibilityConfig(
                trend_ma=50,
                ma_slope_lookback=20,
                ma_slope_min=0.005,
                require_close_above_trend_ma=True,
                relative_strength_enabled=True,
                relative_strength_min=0.0,
            ),
        )
        strategy_with_rs = EquityStrategy(config)
        
        # Features with sufficient relative strength
        features = FeatureRow(
            date=pd.Timestamp('2024-02-01'),
            symbol='AAPL',
            asset_class='equity',
            close=105.0,
            open=104.5,
            high=106.0,
            low=104.0,
            ma20=103.0,
            ma50=100.0,
            ma200=95.0,
            atr14=2.0,
            roc60=0.05,  # 5% return
            highest_close_20d=104.0,
            highest_close_55d=102.0,
            adv20=100000000.0,
            returns_1d=0.001,
            benchmark_roc60=0.03,  # 3% return (relative strength = 2%)
            benchmark_returns_1d=0.0005,
        )
        
        is_eligible, failures = strategy_with_rs.check_eligibility(features)
        assert is_eligible
        
        # Features with insufficient relative strength
        features.benchmark_roc60 = 0.06  # 6% return (relative strength = -1%)
        is_eligible, failures = strategy_with_rs.check_eligibility(features)
        assert not is_eligible
        assert any('insufficient_relative_strength' in f for f in failures)


class TestEntryTriggers:
    """Tests for entry trigger logic."""
    
    @pytest.fixture
    def strategy(self):
        """Create equity strategy with default config."""
        config = StrategyConfig(
            name="equity_momentum",
            asset_class="equity",
            universe="NASDAQ-100",
            benchmark="SPY",
            entry=EntryConfig(fast_clearance=0.005, slow_clearance=0.010),
        )
        return EquityStrategy(config)
    
    @pytest.fixture
    def valid_features(self):
        """Create valid feature row."""
        return FeatureRow(
            date=pd.Timestamp('2024-02-01'),
            symbol='AAPL',
            asset_class='equity',
            close=105.0,
            open=104.5,
            high=106.0,
            low=104.0,
            ma20=103.0,
            ma50=100.0,
            ma200=95.0,
            atr14=2.0,
            roc60=0.05,
            highest_close_20d=104.0,  # Close of 105 >= 104.0 * 1.005 = 104.52 (passes fast)
            highest_close_55d=102.0,
            adv20=100000000.0,
            returns_1d=0.001,
            ma50_slope=0.01,  # 1% slope (above 0.5% threshold)
            benchmark_roc60=0.03,
            benchmark_returns_1d=0.0005,
        )
    
    def test_fast_breakout_trigger(self, strategy, valid_features):
        """Test fast breakout (20D) trigger."""
        # Close = 105, highest_close_20d = 104
        # Threshold = 104 * 1.005 = 104.52
        # 105 >= 104.52, so should trigger
        breakout_type, clearance = strategy.check_entry_triggers(valid_features)
        
        assert breakout_type == BreakoutType.FAST_20D
        assert clearance > 0
        assert abs(clearance - ((105.0 / 104.0) - 1.0)) < 1e-6
    
    def test_slow_breakout_trigger(self, strategy):
        """Test slow breakout (55D) trigger."""
        features = FeatureRow(
            date=pd.Timestamp('2024-02-01'),
            symbol='AAPL',
            asset_class='equity',
            close=104.0,
            open=103.5,
            high=105.0,
            low=103.0,
            ma20=102.0,
            ma50=100.0,
            ma200=95.0,
            atr14=2.0,
            roc60=0.05,
            highest_close_20d=103.5,  # Close = 104, threshold = 103.5 * 1.005 = 104.02 (fails)
            highest_close_55d=102.0,  # Close = 104, threshold = 102 * 1.010 = 103.02 (passes)
            adv20=100000000.0,
            returns_1d=0.001,
            ma50_slope=0.01,  # 1% slope (above 0.5% threshold)
            benchmark_roc60=0.03,
            benchmark_returns_1d=0.0005,
        )
        
        breakout_type, clearance = strategy.check_entry_triggers(features)
        
        assert breakout_type == BreakoutType.SLOW_55D
        assert clearance > 0
        assert abs(clearance - ((104.0 / 102.0) - 1.0)) < 1e-6
    
    def test_no_trigger(self, strategy):
        """Test no trigger when neither breakout is met."""
        features = FeatureRow(
            date=pd.Timestamp('2024-02-01'),
            symbol='AAPL',
            asset_class='equity',
            close=103.0,
            open=102.5,
            high=104.0,
            low=102.0,
            ma20=102.0,
            ma50=100.0,
            ma200=95.0,
            atr14=2.0,
            roc60=0.05,
            highest_close_20d=104.0,  # Close = 103, threshold = 104 * 1.005 = 104.52 (fails)
            highest_close_55d=102.0,  # Close = 103, threshold = 102 * 1.010 = 103.02 (fails)
            adv20=100000000.0,
            returns_1d=0.001,
            ma50_slope=0.01,  # 1% slope (above 0.5% threshold)
            benchmark_roc60=0.03,
            benchmark_returns_1d=0.0005,
        )
        
        breakout_type, clearance = strategy.check_entry_triggers(features)
        
        assert breakout_type is None
        assert clearance == 0.0


class TestCapacityCheck:
    """Tests for capacity constraint checking."""
    
    @pytest.fixture
    def strategy(self):
        """Create equity strategy with default config."""
        config = StrategyConfig(
            name="equity_momentum",
            asset_class="equity",
            universe="NASDAQ-100",
            benchmark="SPY",
            capacity=CapacityConfig(max_order_pct_adv=0.005),  # 0.5%
        )
        return EquityStrategy(config)
    
    def test_capacity_passes(self, strategy):
        """Test capacity check passes when order size is within limit."""
        order_notional = 400000.0  # $400k
        adv20 = 100000000.0  # $100M
        
        # Max allowed = 0.005 * 100M = 500k
        # 400k < 500k, so should pass
        result = strategy.check_capacity(order_notional, adv20)
        
        assert result is True
    
    def test_capacity_fails(self, strategy):
        """Test capacity check fails when order size exceeds limit."""
        order_notional = 600000.0  # $600k
        adv20 = 100000000.0  # $100M
        
        # Max allowed = 0.005 * 100M = 500k
        # 600k > 500k, so should fail
        result = strategy.check_capacity(order_notional, adv20)
        
        assert result is False
    
    def test_capacity_exact_limit(self, strategy):
        """Test capacity check at exact limit."""
        adv20 = 100000000.0  # $100M
        max_allowed = 0.005 * adv20  # 500k
        
        # Exactly at limit
        result = strategy.check_capacity(max_allowed, adv20)
        assert result is True
        
        # Just over limit
        result = strategy.check_capacity(max_allowed + 0.01, adv20)
        assert result is False


class TestExitSignals:
    """Tests for exit signal logic."""
    
    @pytest.fixture
    def strategy(self):
        """Create equity strategy with default config."""
        config = StrategyConfig(
            name="equity_momentum",
            asset_class="equity",
            universe="NASDAQ-100",
            benchmark="SPY",
            exit=ExitConfig(mode="ma_cross", exit_ma=20, hard_stop_atr_mult=2.5),
        )
        return EquityStrategy(config)
    
    @pytest.fixture
    def position(self):
        """Create test position."""
        return Position(
            symbol='AAPL',
            asset_class='equity',
            entry_date=pd.Timestamp('2024-01-15'),
            entry_price=100.0,
            entry_fill_id='fill_1',
            quantity=100,
            stop_price=95.0,  # Hard stop: 100 - 2.5 * 2.0 = 95.0
            initial_stop_price=95.0,
            hard_stop_atr_mult=2.5,
            entry_slippage_bps=8.0,
            entry_fee_bps=1.0,
            entry_total_cost=9.0,
            triggered_on=BreakoutType.FAST_20D,
            adv20_at_entry=100000000.0,
        )
    
    def test_hard_stop_exit(self, strategy, position):
        """Test exit triggered by hard stop."""
        features = FeatureRow(
            date=pd.Timestamp('2024-02-01'),
            symbol='AAPL',
            asset_class='equity',
            close=94.0,  # Below stop_price of 95.0
            open=94.5,
            high=95.5,
            low=93.5,
            ma20=96.0,
            ma50=97.0,
            ma200=95.0,
            atr14=2.0,
            roc60=0.05,
            highest_close_20d=98.0,
            highest_close_55d=97.0,
            adv20=100000000.0,
            returns_1d=-0.01,
            benchmark_roc60=0.03,
            benchmark_returns_1d=0.0005,
        )
        
        exit_reason = strategy.check_exit_signals(position, features)
        
        assert exit_reason == ExitReason.HARD_STOP
    
    def test_trailing_ma_cross_exit(self, strategy, position):
        """Test exit triggered by trailing MA cross."""
        features = FeatureRow(
            date=pd.Timestamp('2024-02-01'),
            symbol='AAPL',
            asset_class='equity',
            close=95.5,  # Above stop_price of 95.0, but below MA20 of 96.0
            open=96.0,
            high=96.5,
            low=95.0,
            ma20=96.0,  # Close < MA20
            ma50=97.0,
            ma200=95.0,
            atr14=2.0,
            roc60=0.05,
            highest_close_20d=98.0,
            highest_close_55d=97.0,
            adv20=100000000.0,
            returns_1d=-0.01,
            benchmark_roc60=0.03,
            benchmark_returns_1d=0.0005,
        )
        
        exit_reason = strategy.check_exit_signals(position, features)
        
        assert exit_reason == ExitReason.TRAILING_MA_CROSS
    
    def test_no_exit(self, strategy, position):
        """Test no exit when neither condition is met."""
        features = FeatureRow(
            date=pd.Timestamp('2024-02-01'),
            symbol='AAPL',
            asset_class='equity',
            close=97.0,  # Above stop_price and MA20
            open=96.5,
            high=97.5,
            low=96.0,
            ma20=96.0,
            ma50=97.0,
            ma200=95.0,
            atr14=2.0,
            roc60=0.05,
            highest_close_20d=98.0,
            highest_close_55d=97.0,
            adv20=100000000.0,
            returns_1d=0.01,
            benchmark_roc60=0.03,
            benchmark_returns_1d=0.0005,
        )
        
        exit_reason = strategy.check_exit_signals(position, features)
        
        assert exit_reason is None
    
    def test_exit_priority_hard_stop_first(self, strategy, position):
        """Test that hard stop takes priority over MA cross."""
        features = FeatureRow(
            date=pd.Timestamp('2024-02-01'),
            symbol='AAPL',
            asset_class='equity',
            close=94.5,  # Below both stop_price (95.0) and MA20 (96.0)
            open=95.0,
            high=95.5,
            low=94.0,
            ma20=96.0,
            ma50=97.0,
            ma200=95.0,
            atr14=2.0,
            roc60=0.05,
            highest_close_20d=98.0,
            highest_close_55d=97.0,
            adv20=100000000.0,
            returns_1d=-0.01,
            benchmark_roc60=0.03,
            benchmark_returns_1d=0.0005,
        )
        
        exit_reason = strategy.check_exit_signals(position, features)
        
        # Hard stop should take priority
        assert exit_reason == ExitReason.HARD_STOP


class TestSignalGeneration:
    """Tests for complete signal generation using base class method."""
    
    @pytest.fixture
    def strategy(self):
        """Create equity strategy with default config."""
        config = StrategyConfig(
            name="equity_momentum",
            asset_class="equity",
            universe="NASDAQ-100",
            benchmark="SPY",
            entry=EntryConfig(fast_clearance=0.005, slow_clearance=0.010),
            exit=ExitConfig(mode="ma_cross", exit_ma=20, hard_stop_atr_mult=2.5),
            capacity=CapacityConfig(max_order_pct_adv=0.005),
        )
        return EquityStrategy(config)
    
    def test_generate_signal_valid(self, strategy):
        """Test signal generation with valid conditions."""
        features = FeatureRow(
            date=pd.Timestamp('2024-02-01'),
            symbol='AAPL',
            asset_class='equity',
            close=105.0,  # Above MA50
            open=104.5,
            high=106.0,
            low=104.0,
            ma20=103.0,
            ma50=100.0,
            ma200=95.0,
            atr14=2.0,
            roc60=0.05,
            highest_close_20d=104.0,  # Close will trigger breakout
            highest_close_55d=102.0,
            adv20=100000000.0,
            returns_1d=0.001,
            ma50_slope=0.01,  # 1% slope (above 0.5% threshold)
            benchmark_roc60=0.03,
            benchmark_returns_1d=0.0005,
        )
        
        order_notional = 400000.0  # Within capacity
        
        signal = strategy.generate_signal('AAPL', features, order_notional)
        
        assert signal is not None
        assert signal.symbol == 'AAPL'
        assert signal.asset_class == 'equity'
        assert signal.side == SignalSide.BUY
        assert signal.entry_price == 105.0
        assert signal.triggered_on == BreakoutType.FAST_20D
        assert signal.passed_eligibility is True
        assert signal.capacity_passed is True
    
    def test_generate_signal_not_eligible(self, strategy):
        """Test signal generation fails when not eligible."""
        features = FeatureRow(
            date=pd.Timestamp('2024-02-01'),
            symbol='AAPL',
            asset_class='equity',
            close=98.0,  # Below MA50
            open=97.5,
            high=99.0,
            low=97.0,
            ma20=100.0,
            ma50=100.0,
            ma200=95.0,
            atr14=2.0,
            roc60=0.05,
            highest_close_20d=99.0,
            highest_close_55d=98.0,
            adv20=100000000.0,
            returns_1d=0.001,
            ma50_slope=0.01,  # 1% slope (above 0.5% threshold)
            benchmark_roc60=0.03,
            benchmark_returns_1d=0.0005,
        )
        
        order_notional = 400000.0
        
        signal = strategy.generate_signal('AAPL', features, order_notional)
        
        assert signal is None
    
    def test_generate_signal_no_trigger(self, strategy):
        """Test signal generation fails when no entry trigger."""
        features = FeatureRow(
            date=pd.Timestamp('2024-02-01'),
            symbol='AAPL',
            asset_class='equity',
            close=103.0,  # Above MA50 but no breakout
            open=102.5,
            high=104.0,
            low=102.0,
            ma20=102.0,
            ma50=100.0,
            ma200=95.0,
            atr14=2.0,
            roc60=0.05,
            highest_close_20d=104.0,  # No breakout
            highest_close_55d=102.0,
            adv20=100000000.0,
            returns_1d=0.001,
            ma50_slope=0.01,  # 1% slope (above 0.5% threshold)
            benchmark_roc60=0.03,
            benchmark_returns_1d=0.0005,
        )
        
        order_notional = 400000.0
        
        signal = strategy.generate_signal('AAPL', features, order_notional)
        
        assert signal is None

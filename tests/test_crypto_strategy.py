"""Unit tests for crypto momentum strategy."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from trading_system.strategies.crypto_strategy import CryptoStrategy
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
from trading_system.models.positions import Position, ExitReason
from trading_system.models.signals import BreakoutType


@pytest.fixture
def crypto_config():
    """Create a crypto strategy configuration."""
    return StrategyConfig(
        name="crypto_momentum",
        asset_class="crypto",
        universe=["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOT", "MATIC", "LTC", "LINK"],
        benchmark="BTC",
        indicators=IndicatorsConfig(
            ma_periods=[20, 50, 200],
            atr_period=14,
            roc_period=60,
            breakout_fast=20,
            breakout_slow=55,
            adv_lookback=20,
            corr_lookback=20,
        ),
        eligibility=EligibilityConfig(
            require_close_above_ma200=True,
            relative_strength_enabled=False,  # MVP: off
            relative_strength_min=0.0,
        ),
        entry=EntryConfig(
            fast_clearance=0.005,  # 0.5%
            slow_clearance=0.010,  # 1.0%
        ),
        exit=ExitConfig(
            mode="staged",
            exit_ma=50,
            hard_stop_atr_mult=3.0,
            tightened_stop_atr_mult=2.0,
        ),
        risk=RiskConfig(
            risk_per_trade=0.0075,
            max_positions=8,
            max_exposure=0.80,
            max_position_notional=0.15,
        ),
        capacity=CapacityConfig(
            max_order_pct_adv=0.0025,  # 0.25% for crypto
        ),
        costs=CostsConfig(
            fee_bps=8,
            slippage_base_bps=10,
            slippage_std_mult=0.75,
            weekend_penalty=1.5,
            stress_threshold=-0.05,
            stress_slippage_mult=2.0,
        ),
    )


@pytest.fixture
def crypto_strategy(crypto_config):
    """Create a crypto strategy instance."""
    return CryptoStrategy(crypto_config)


@pytest.fixture
def sample_features():
    """Create sample FeatureRow for testing."""
    date = pd.Timestamp("2024-01-15")
    return FeatureRow(
        date=date,
        symbol="BTC",
        asset_class="crypto",
        close=50000.0,
        open=49500.0,
        high=50500.0,
        low=49400.0,
        ma20=49000.0,
        ma50=48000.0,
        ma200=45000.0,
        atr14=2000.0,
        roc60=0.10,
        highest_close_20d=49000.0,
        highest_close_55d=48000.0,
        adv20=1_000_000_000.0,  # 1B
        returns_1d=0.01,
        benchmark_roc60=0.08,
        benchmark_returns_1d=0.005,
    )


class TestCryptoStrategyInitialization:
    """Test crypto strategy initialization."""
    
    def test_init_with_crypto_config(self, crypto_config):
        """Test initialization with valid crypto config."""
        strategy = CryptoStrategy(crypto_config)
        assert strategy.asset_class == "crypto"
        assert strategy.benchmark == "BTC"
    
    def test_init_with_non_crypto_config(self):
        """Test initialization fails with non-crypto config."""
        config = StrategyConfig(
            name="equity_momentum",
            asset_class="equity",
            universe="NASDAQ-100",
            benchmark="SPY",
        )
        with pytest.raises(ValueError, match="asset_class='crypto'"):
            CryptoStrategy(config)
    
    def test_init_with_non_staged_exit(self):
        """Test initialization fails with non-staged exit mode."""
        config = StrategyConfig(
            name="crypto_momentum",
            asset_class="crypto",
            universe=["BTC"],
            benchmark="BTC",
            exit=ExitConfig(mode="ma_cross"),
        )
        with pytest.raises(ValueError, match="exit.mode='staged'"):
            CryptoStrategy(config)


class TestEligibilityFilter:
    """Test eligibility filter logic."""
    
    def test_eligible_above_ma200(self, crypto_strategy, sample_features):
        """Test eligibility when close > MA200."""
        is_eligible, failures = crypto_strategy.check_eligibility(sample_features)
        assert is_eligible
        assert len(failures) == 0
    
    def test_not_eligible_below_ma200(self, crypto_strategy, sample_features):
        """Test ineligibility when close <= MA200."""
        sample_features.close = 44000.0  # Below MA200 (45000)
        is_eligible, failures = crypto_strategy.check_eligibility(sample_features)
        assert not is_eligible
        assert "below_MA200" in failures
    
    def test_not_eligible_equal_ma200(self, crypto_strategy, sample_features):
        """Test ineligibility when close == MA200 (strict requirement)."""
        sample_features.close = 45000.0  # Equal to MA200
        is_eligible, failures = crypto_strategy.check_eligibility(sample_features)
        assert not is_eligible
        assert "below_MA200" in failures
    
    def test_not_eligible_missing_ma200(self, crypto_strategy, sample_features):
        """Test ineligibility when MA200 is missing."""
        sample_features.ma200 = None
        is_eligible, failures = crypto_strategy.check_eligibility(sample_features)
        assert not is_eligible
        assert "ma200_missing" in failures
    
    def test_relative_strength_enabled_pass(self, crypto_config, sample_features):
        """Test eligibility with relative strength enabled and passing."""
        crypto_config.eligibility.relative_strength_enabled = True
        crypto_config.eligibility.relative_strength_min = 0.0
        strategy = CryptoStrategy(crypto_config)
        
        # roc60 (0.10) > benchmark_roc60 (0.08), so relative strength = 0.02 > 0.0
        is_eligible, failures = strategy.check_eligibility(sample_features)
        assert is_eligible
        assert len(failures) == 0
    
    def test_relative_strength_enabled_fail(self, crypto_config, sample_features):
        """Test eligibility with relative strength enabled and failing."""
        crypto_config.eligibility.relative_strength_enabled = True
        crypto_config.eligibility.relative_strength_min = 0.05
        strategy = CryptoStrategy(crypto_config)
        
        # roc60 (0.10) - benchmark_roc60 (0.08) = 0.02 < 0.05
        is_eligible, failures = strategy.check_eligibility(sample_features)
        assert not is_eligible
        assert "insufficient_relative_strength" in failures
    
    def test_relative_strength_missing_roc60(self, crypto_config, sample_features):
        """Test eligibility fails when roc60 is missing and relative strength enabled."""
        crypto_config.eligibility.relative_strength_enabled = True
        strategy = CryptoStrategy(crypto_config)
        
        sample_features.roc60 = None
        is_eligible, failures = strategy.check_eligibility(sample_features)
        assert not is_eligible
        assert "roc60_missing" in failures
    
    def test_relative_strength_missing_benchmark(self, crypto_config, sample_features):
        """Test eligibility fails when benchmark_roc60 is missing and relative strength enabled."""
        crypto_config.eligibility.relative_strength_enabled = True
        strategy = CryptoStrategy(crypto_config)
        
        sample_features.benchmark_roc60 = None
        is_eligible, failures = strategy.check_eligibility(sample_features)
        assert not is_eligible
        assert "benchmark_roc60_missing" in failures


class TestEntryTriggers:
    """Test entry trigger logic."""
    
    def test_fast_trigger(self, crypto_strategy, sample_features):
        """Test fast 20D breakout trigger."""
        # close (50000) >= highest_close_20d (49000) * 1.005 = 49245
        breakout_type, clearance = crypto_strategy.check_entry_triggers(sample_features)
        assert breakout_type == BreakoutType.FAST_20D
        assert clearance > 0.005  # Should be above threshold
    
    def test_slow_trigger(self, crypto_strategy, sample_features):
        """Test slow 55D breakout trigger."""
        # Set close to trigger slow but not fast
        sample_features.close = 48500.0  # Below fast threshold but above slow
        sample_features.highest_close_20d = 49000.0  # Fast threshold = 49245
        sample_features.highest_close_55d = 48000.0  # Slow threshold = 48480
        
        breakout_type, clearance = crypto_strategy.check_entry_triggers(sample_features)
        assert breakout_type == BreakoutType.SLOW_55D
        assert clearance > 0.010
    
    def test_no_trigger(self, crypto_strategy, sample_features):
        """Test no trigger when both conditions fail."""
        sample_features.close = 48000.0  # Below both thresholds
        sample_features.highest_close_20d = 49000.0  # Fast threshold = 49245
        sample_features.highest_close_55d = 48000.0  # Slow threshold = 48480
        
        breakout_type, clearance = crypto_strategy.check_entry_triggers(sample_features)
        assert breakout_type is None
        assert clearance == 0.0
    
    def test_fast_trigger_exact_threshold(self, crypto_strategy, sample_features):
        """Test fast trigger at exact threshold."""
        # Set close to exactly match threshold
        threshold = sample_features.highest_close_20d * 1.005
        sample_features.close = threshold
        
        breakout_type, clearance = crypto_strategy.check_entry_triggers(sample_features)
        assert breakout_type == BreakoutType.FAST_20D
        assert abs(clearance - 0.005) < 1e-6


class TestStagedExitLogic:
    """Test staged exit logic."""
    
    @pytest.fixture
    def sample_position(self, sample_features):
        """Create a sample position."""
        return Position(
            symbol="BTC",
            asset_class="crypto",
            entry_date=sample_features.date,
            entry_price=50000.0,
            entry_fill_id="fill_001",
            quantity=1,
            stop_price=44000.0,  # entry - 3.0 * ATR14 = 50000 - 6000
            initial_stop_price=44000.0,
            hard_stop_atr_mult=3.0,
            tightened_stop=False,
            entry_slippage_bps=10.0,
            entry_fee_bps=8.0,
            entry_total_cost=90.0,
            triggered_on=BreakoutType.FAST_20D,
            adv20_at_entry=1_000_000_000.0,
        )
    
    def test_exit_hard_stop(self, crypto_strategy, sample_position, sample_features):
        """Test exit on hard stop hit."""
        sample_features.close = 43900.0  # Below stop_price (44000)
        exit_reason = crypto_strategy.check_exit_signals(sample_position, sample_features)
        assert exit_reason == ExitReason.HARD_STOP
    
    def test_exit_ma50_cross(self, crypto_strategy, sample_position, sample_features):
        """Test exit on MA50 cross (stage 2)."""
        sample_features.close = 47900.0  # Below MA50 (48000) but above stop
        exit_reason = crypto_strategy.check_exit_signals(sample_position, sample_features)
        assert exit_reason == ExitReason.TRAILING_MA_CROSS
    
    def test_no_exit_above_ma50(self, crypto_strategy, sample_position, sample_features):
        """Test no exit when above MA50 and above stop."""
        sample_features.close = 51000.0  # Above MA50 and stop
        exit_reason = crypto_strategy.check_exit_signals(sample_position, sample_features)
        assert exit_reason is None
    
    def test_stop_tightening_on_ma20_break(self, crypto_strategy, sample_position, sample_features):
        """Test stop tightening when MA20 breaks (stage 1)."""
        # Close below MA20 but above stop
        sample_features.close = 48900.0  # Below MA20 (49000) but above stop (44000)
        sample_features.atr14 = 2000.0
        
        new_stop = crypto_strategy.update_stop_price(sample_position, sample_features)
        
        # Should tighten to entry - 2.0 * ATR14 = 50000 - 4000 = 46000
        assert new_stop == 46000.0
        assert sample_position.tightened_stop is True
        assert sample_position.tightened_stop_atr_mult == 2.0
    
    def test_stop_tightening_only_once(self, crypto_strategy, sample_position, sample_features):
        """Test that stop tightening happens only once."""
        # First MA20 break
        sample_features.close = 48900.0
        new_stop1 = crypto_strategy.update_stop_price(sample_position, sample_features)
        assert new_stop1 == 46000.0
        assert sample_position.tightened_stop is True
        
        # Second MA20 break (should not tighten again)
        sample_features.close = 48800.0
        new_stop2 = crypto_strategy.update_stop_price(sample_position, sample_features)
        assert new_stop2 is None  # No further tightening
        assert sample_position.stop_price == 46000.0  # Unchanged
    
    def test_stop_tightening_requires_higher_stop(self, crypto_strategy, sample_position, sample_features):
        """Test that stop tightening only happens if new stop is higher."""
        # Set current stop very high
        sample_position.stop_price = 48000.0  # Higher than tightened stop would be
        
        sample_features.close = 48900.0  # Below MA20
        new_stop = crypto_strategy.update_stop_price(sample_position, sample_features)
        
        # Tightened stop would be 46000, but current is 48000, so no change
        assert new_stop is None or new_stop == 48000.0
    
    def test_exit_on_tightened_stop(self, crypto_strategy, sample_position, sample_features):
        """Test exit when tightened stop is hit."""
        # First tighten the stop
        sample_features.close = 48900.0  # Below MA20
        crypto_strategy.update_stop_price(sample_position, sample_features)
        assert sample_position.stop_price == 46000.0
        
        # Now hit the tightened stop
        sample_features.close = 45900.0  # Below tightened stop
        exit_reason = crypto_strategy.check_exit_signals(sample_position, sample_features)
        assert exit_reason == ExitReason.HARD_STOP
    
    def test_should_tighten_stop(self, crypto_strategy, sample_position, sample_features):
        """Test should_tighten_stop helper method."""
        # Not yet tightened and below MA20
        sample_features.close = 48900.0  # Below MA20 (49000)
        assert crypto_strategy.should_tighten_stop(sample_position, sample_features) is True
        
        # Already tightened
        sample_position.tightened_stop = True
        assert crypto_strategy.should_tighten_stop(sample_position, sample_features) is False
        
        # Above MA20
        sample_position.tightened_stop = False
        sample_features.close = 49100.0  # Above MA20
        assert crypto_strategy.should_tighten_stop(sample_position, sample_features) is False


class TestCapacityCheck:
    """Test capacity constraint checking."""
    
    def test_capacity_pass(self, crypto_strategy):
        """Test capacity check passes when order is within limit."""
        order_notional = 2_000_000.0  # 2M
        adv20 = 1_000_000_000.0  # 1B
        # 2M / 1B = 0.002 = 0.2% < 0.25% limit
        
        assert crypto_strategy.check_capacity(order_notional, adv20) is True
    
    def test_capacity_fail(self, crypto_strategy):
        """Test capacity check fails when order exceeds limit."""
        order_notional = 3_000_000.0  # 3M
        adv20 = 1_000_000_000.0  # 1B
        # 3M / 1B = 0.003 = 0.3% > 0.25% limit
        
        assert crypto_strategy.check_capacity(order_notional, adv20) is False
    
    def test_capacity_exact_limit(self, crypto_strategy):
        """Test capacity check at exact limit."""
        order_notional = 2_500_000.0  # 2.5M
        adv20 = 1_000_000_000.0  # 1B
        # 2.5M / 1B = 0.0025 = 0.25% (exact limit)
        
        assert crypto_strategy.check_capacity(order_notional, adv20) is True


class TestSignalGeneration:
    """Test full signal generation."""
    
    def test_generate_valid_signal(self, crypto_strategy, sample_features):
        """Test generating a valid signal."""
        order_notional = 2_000_000.0
        
        signal = crypto_strategy.generate_signal(
            symbol="BTC",
            features=sample_features,
            order_notional=order_notional,
            diversification_bonus=0.5,
        )
        
        assert signal is not None
        assert signal.symbol == "BTC"
        assert signal.asset_class == "crypto"
        assert signal.passed_eligibility is True
        assert signal.capacity_passed is True
        assert signal.triggered_on == BreakoutType.FAST_20D
        assert signal.atr_mult == 3.0  # Crypto uses 3.0
        assert signal.stop_price == 44000.0  # 50000 - 3.0 * 2000
    
    def test_generate_signal_fails_eligibility(self, crypto_strategy, sample_features):
        """Test signal generation fails when eligibility fails."""
        sample_features.close = 44000.0  # Below MA200
        order_notional = 2_000_000.0
        
        signal = crypto_strategy.generate_signal(
            symbol="BTC",
            features=sample_features,
            order_notional=order_notional,
        )
        
        assert signal is None
    
    def test_generate_signal_fails_capacity(self, crypto_strategy, sample_features):
        """Test signal generation fails when capacity check fails."""
        order_notional = 3_000_000.0  # Exceeds capacity
        
        signal = crypto_strategy.generate_signal(
            symbol="BTC",
            features=sample_features,
            order_notional=order_notional,
        )
        
        # Signal is created but capacity_passed is False
        assert signal is not None
        assert signal.capacity_passed is False


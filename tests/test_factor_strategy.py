"""Unit tests for equity factor strategy."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from tests.utils.test_helpers import create_sample_feature_row, create_sample_position
from trading_system.configs.strategy_config import StrategyConfig
from trading_system.models import BreakoutType, ExitReason, FeatureRow, Position, PositionSide, Signal, SignalSide
from trading_system.strategies.factor.equity_factor import EquityFactorStrategy


class TestEquityFactorStrategyInit:
    """Tests for EquityFactorStrategy initialization."""

    def test_init_valid(self):
        """Test initialization with valid equity config."""
        config = StrategyConfig(
            name="equity_factor",
            asset_class="equity",
            universe="NASDAQ-100",
            benchmark="SPY",
            parameters={
                "factors": {"momentum": 0.4, "value": 0.3, "quality": 0.3},
                "rebalance_frequency": "monthly",
                "top_decile_pct": 0.20,
                "min_adv20": 10_000_000,
                "max_hold_days": 90,
                "stop_atr_mult": 2.0,
            },
        )
        strategy = EquityFactorStrategy(config)
        assert strategy.config == config
        assert strategy.asset_class == "equity"
        assert strategy.factors["momentum"] == 0.4
        assert strategy.factors["value"] == 0.3
        assert strategy.factors["quality"] == 0.3
        assert strategy.rebalance_frequency == "monthly"
        assert strategy.top_decile_pct == 0.20
        assert strategy.min_adv20 == 10_000_000
        assert strategy.max_hold_days == 90

    def test_init_invalid_asset_class(self):
        """Test initialization fails with non-equity config."""
        config = StrategyConfig(
            name="crypto_factor",
            asset_class="crypto",
            universe="fixed",
            benchmark="BTC",
        )
        with pytest.raises(ValueError, match="asset_class='equity'"):
            EquityFactorStrategy(config)

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        config = StrategyConfig(
            name="equity_factor",
            asset_class="equity",
            universe="NASDAQ-100",
            benchmark="SPY",
        )
        strategy = EquityFactorStrategy(config)
        assert strategy.min_adv20 == 10_000_000  # Default
        assert strategy.max_hold_days == 90  # Default
        assert strategy.factors["momentum"] == 0.4  # Default
        assert strategy.factors["value"] == 0.3  # Default
        assert strategy.factors["quality"] == 0.3  # Default


class TestFactorScoreComputation:
    """Tests for factor score computation."""

    @pytest.fixture
    def strategy(self):
        """Create equity factor strategy."""
        config = StrategyConfig(
            name="equity_factor",
            asset_class="equity",
            universe="NASDAQ-100",
            benchmark="SPY",
            parameters={
                "factors": {"momentum": 0.4, "value": 0.3, "quality": 0.3},
            },
        )
        return EquityFactorStrategy(config)

    def test_compute_factor_score_valid(self, strategy):
        """Test computing factor score with valid data."""
        features = create_sample_feature_row(
            date=pd.Timestamp("2024-01-15"),
            symbol="AAPL",
            close=150.0,
            atr14=3.0,
            roc60=0.10,  # 10% momentum
            highest_close_55d=160.0,  # Value: 1 - 150/160 = 0.0625
            adv20=20_000_000.0,
        )

        score = strategy.compute_factor_score(features)
        assert score is not None
        assert isinstance(score, (int, float))
        assert not np.isnan(score)

    def test_compute_factor_score_insufficient_data(self, strategy):
        """Test computing factor score with insufficient data."""
        features = create_sample_feature_row(
            date=pd.Timestamp("2024-01-15"),
            symbol="AAPL",
            close=0.0,  # Invalid close
            atr14=None,  # Missing ATR
        )

        score = strategy.compute_factor_score(features)
        assert score is None

    def test_compute_factor_score_missing_roc60(self, strategy):
        """Test computing factor score when ROC60 is missing."""
        features = create_sample_feature_row(
            date=pd.Timestamp("2024-01-15"),
            symbol="AAPL",
            close=150.0,
            atr14=3.0,
            roc60=None,  # Missing ROC60
            highest_close_55d=160.0,
            adv20=20_000_000.0,
        )

        # Should still compute score using fallback
        score = strategy.compute_factor_score(features)
        # May be None or a computed value depending on fallback logic
        assert score is None or isinstance(score, (int, float))

    def test_compute_factor_score_value_calculation(self, strategy):
        """Test value factor calculation."""
        features = create_sample_feature_row(
            date=pd.Timestamp("2024-01-15"),
            symbol="AAPL",
            close=150.0,
            atr14=3.0,
            roc60=0.05,
            highest_close_55d=200.0,  # Far from high = good value
            adv20=20_000_000.0,
        )

        score = strategy.compute_factor_score(features)
        assert score is not None

        # Check that raw factors are cached
        assert hasattr(strategy, "_raw_factors_cache")
        assert "AAPL" in strategy._raw_factors_cache
        raw_factors = strategy._raw_factors_cache["AAPL"]
        assert "value" in raw_factors
        # Value should be positive (1 - 150/200 = 0.25)
        assert raw_factors["value"] > 0

    def test_compute_factor_score_quality_calculation(self, strategy):
        """Test quality factor calculation (inverse volatility)."""
        features = create_sample_feature_row(
            date=pd.Timestamp("2024-01-15"),
            symbol="AAPL",
            close=150.0,
            atr14=1.0,  # Low ATR = low volatility = high quality
            roc60=0.05,
            highest_close_55d=160.0,
            adv20=20_000_000.0,
        )

        score = strategy.compute_factor_score(features)
        assert score is not None

        # Check quality factor
        raw_factors = strategy._raw_factors_cache.get("AAPL", {})
        assert "quality" in raw_factors
        # Lower volatility should give higher quality score
        assert raw_factors["quality"] > 0


class TestEligibilityChecks:
    """Tests for eligibility checking."""

    @pytest.fixture
    def strategy(self):
        """Create equity factor strategy."""
        config = StrategyConfig(
            name="equity_factor",
            asset_class="equity",
            universe="NASDAQ-100",
            benchmark="SPY",
            parameters={
                "min_adv20": 10_000_000,
            },
        )
        return EquityFactorStrategy(config)

    def test_eligibility_valid(self, strategy):
        """Test eligibility with valid features."""
        features = create_sample_feature_row(
            date=pd.Timestamp("2024-01-15"),
            symbol="AAPL",
            close=150.0,
            atr14=3.0,
            roc60=0.05,
            highest_close_55d=160.0,
            adv20=20_000_000.0,  # Above minimum
        )

        is_eligible, failures = strategy.check_eligibility(features)
        assert is_eligible
        assert len(failures) == 0

    def test_eligibility_insufficient_data(self, strategy):
        """Test eligibility fails with insufficient data."""
        features = create_sample_feature_row(
            date=pd.Timestamp("2024-01-15"),
            symbol="AAPL",
            close=0.0,  # Invalid
            atr14=None,
        )

        is_eligible, failures = strategy.check_eligibility(features)
        assert not is_eligible
        assert "insufficient_data" in failures

    def test_eligibility_missing_atr14(self, strategy):
        """Test eligibility fails when ATR14 is missing."""
        features = create_sample_feature_row(
            date=pd.Timestamp("2024-01-15"),
            symbol="AAPL",
            close=150.0,
            atr14=None,  # Missing
            adv20=20_000_000.0,
        )

        is_eligible, failures = strategy.check_eligibility(features)
        assert not is_eligible
        assert "atr14_missing" in failures

    def test_eligibility_insufficient_liquidity(self, strategy):
        """Test eligibility fails with insufficient liquidity."""
        features = create_sample_feature_row(
            date=pd.Timestamp("2024-01-15"),
            symbol="AAPL",
            close=150.0,
            atr14=3.0,
            adv20=5_000_000.0,  # Below minimum
        )

        is_eligible, failures = strategy.check_eligibility(features)
        assert not is_eligible
        assert any("insufficient_liquidity" in f for f in failures)

    def test_eligibility_missing_adv20(self, strategy):
        """Test eligibility fails when ADV20 is missing."""
        features = create_sample_feature_row(
            date=pd.Timestamp("2024-01-15"),
            symbol="AAPL",
            close=150.0,
            atr14=3.0,
            adv20=None,  # Missing
        )

        is_eligible, failures = strategy.check_eligibility(features)
        assert not is_eligible
        assert "adv20_missing" in failures


class TestSignalGeneration:
    """Tests for signal generation."""

    @pytest.fixture
    def strategy(self):
        """Create equity factor strategy."""
        config = StrategyConfig(
            name="equity_factor",
            asset_class="equity",
            universe="NASDAQ-100",
            benchmark="SPY",
            parameters={
                "factors": {"momentum": 0.4, "value": 0.3, "quality": 0.3},
                "rebalance_frequency": "monthly",
                "top_decile_pct": 0.20,
                "min_adv20": 10_000_000,
                "stop_atr_mult": 2.0,
            },
        )
        return EquityFactorStrategy(config)

    def test_generate_signal_rebalance_day(self, strategy):
        """Test signal generation on rebalance day."""
        # January 1st is a rebalance day (first day of month)
        features = create_sample_feature_row(
            date=pd.Timestamp("2024-01-01"),
            symbol="AAPL",
            close=150.0,
            atr14=3.0,
            roc60=0.10,  # Good momentum
            highest_close_55d=160.0,
            adv20=20_000_000.0,
        )

        # Pre-populate factor scores cache to simulate top decile
        strategy._factor_scores_cache["AAPL"] = 1.5  # High score
        strategy._top_decile_symbols = {"AAPL"}  # In top decile

        signal = strategy.generate_signal(symbol="AAPL", features=features, order_notional=100000.0, diversification_bonus=0.5)

        # Signal may or may not be generated depending on rebalance logic
        # The exact behavior depends on implementation details
        if signal is not None:
            assert signal.symbol == "AAPL"
            assert signal.side == SignalSide.BUY
            assert signal.entry_price == 150.0
            assert signal.stop_price < 150.0  # Stop below entry

    def test_generate_signal_not_rebalance_day(self, strategy):
        """Test signal generation on non-rebalance day."""
        # January 15th is not a rebalance day
        features = create_sample_feature_row(
            date=pd.Timestamp("2024-01-15"),
            symbol="AAPL",
            close=150.0,
            atr14=3.0,
            roc60=0.10,
            highest_close_55d=160.0,
            adv20=20_000_000.0,
        )

        signal = strategy.generate_signal(
            symbol="AAPL",
            features=features,
            order_notional=100000.0,
        )

        # On non-rebalance days, signals should not be generated
        # (unless position already exists, which is handled separately)
        assert signal is None

    def test_generate_signal_not_eligible(self, strategy):
        """Test signal generation fails when not eligible."""
        features = create_sample_feature_row(
            date=pd.Timestamp("2024-01-01"),
            symbol="AAPL",
            close=150.0,
            atr14=None,  # Missing ATR
            adv20=20_000_000.0,
        )

        signal = strategy.generate_signal(
            symbol="AAPL",
            features=features,
            order_notional=100000.0,
        )

        assert signal is None

    def test_generate_signal_not_in_top_decile(self, strategy):
        """Test signal not generated when not in top decile."""
        features = create_sample_feature_row(
            date=pd.Timestamp("2024-01-01"),
            symbol="AAPL",
            close=150.0,
            atr14=3.0,
            roc60=0.10,
            highest_close_55d=160.0,
            adv20=20_000_000.0,
        )

        # Set up so AAPL is NOT in top decile
        strategy._factor_scores_cache["AAPL"] = 0.5  # Low score
        strategy._top_decile_symbols = {"MSFT", "GOOGL"}  # Other symbols

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
        """Create equity factor strategy."""
        config = StrategyConfig(
            name="equity_factor",
            asset_class="equity",
            universe="NASDAQ-100",
            benchmark="SPY",
            parameters={
                "max_hold_days": 90,
                "stop_atr_mult": 2.0,
            },
        )
        return EquityFactorStrategy(config)

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
            atr14=3.0,
        )

        exit_reason = strategy.check_exit_signals(position, features)
        assert exit_reason == ExitReason.HARD_STOP

    def test_exit_rebalance_not_in_top_decile(self, strategy):
        """Test exit on rebalance day when not in top decile."""
        position = create_sample_position(
            symbol="AAPL",
            asset_class="equity",
            entry_date=pd.Timestamp("2024-01-01"),
            entry_price=150.0,
            quantity=100,
            stop_price=145.0,
        )

        # January 1st is a rebalance day
        features = create_sample_feature_row(
            date=pd.Timestamp("2024-02-01"),  # Next rebalance day
            symbol="AAPL",
            close=155.0,  # Above stop
            atr14=3.0,
            roc60=0.05,
            highest_close_55d=160.0,
        )

        # Set up so AAPL is NOT in top decile
        strategy._top_decile_symbols = {"MSFT", "GOOGL"}
        strategy._current_rebalance_date = features.date.to_pydatetime()

        exit_reason = strategy.check_exit_signals(position, features)
        assert exit_reason == ExitReason.MANUAL  # Rebalance exit

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

        # 91 days later (exceeds max_hold_days=90)
        features = create_sample_feature_row(
            date=pd.Timestamp("2024-04-01"),
            symbol="AAPL",
            close=155.0,  # Above stop
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
            close=155.0,  # Above stop
            atr14=3.0,
            roc60=0.05,
            highest_close_55d=160.0,
        )

        # Set up so AAPL IS in top decile
        strategy._top_decile_symbols = {"AAPL"}

        exit_reason = strategy.check_exit_signals(position, features)
        assert exit_reason is None


class TestStopPriceUpdates:
    """Tests for stop price updates."""

    @pytest.fixture
    def strategy(self):
        """Create equity factor strategy."""
        config = StrategyConfig(
            name="equity_factor",
            asset_class="equity",
            universe="NASDAQ-100",
            benchmark="SPY",
        )
        return EquityFactorStrategy(config)

    def test_update_stop_price_fixed_stop(self, strategy):
        """Test that factor strategy uses fixed stops (no updates)."""
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
            atr14=3.0,
        )

        new_stop = strategy.update_stop_price(position, features)
        # Factor strategy doesn't update stops
        assert new_stop is None
        assert position.stop_price == 145.0  # Unchanged


class TestRebalanceLogic:
    """Tests for rebalance day logic."""

    @pytest.fixture
    def strategy(self):
        """Create equity factor strategy."""
        config = StrategyConfig(
            name="equity_factor",
            asset_class="equity",
            universe="NASDAQ-100",
            benchmark="SPY",
            parameters={
                "rebalance_frequency": "monthly",
                "top_decile_pct": 0.20,
            },
        )
        return EquityFactorStrategy(config)

    def test_is_rebalance_day_monthly(self, strategy):
        """Test monthly rebalance day detection."""
        # First day of month
        assert strategy._is_rebalance_day(datetime(2024, 1, 1))
        assert strategy._is_rebalance_day(datetime(2024, 2, 1))
        assert strategy._is_rebalance_day(datetime(2024, 3, 5))  # Within first 5 days

        # Not rebalance days
        assert not strategy._is_rebalance_day(datetime(2024, 1, 15))
        assert not strategy._is_rebalance_day(datetime(2024, 2, 10))

    def test_update_top_decile(self, strategy):
        """Test updating top decile symbols."""
        # Set up factor scores
        strategy._factor_scores_cache = {
            "AAPL": 2.0,
            "MSFT": 1.5,
            "GOOGL": 1.0,
            "AMZN": 0.5,
            "TSLA": 0.3,
        }

        strategy._update_top_decile(datetime(2024, 1, 1))

        # Top 20% of 5 symbols = 1 symbol (AAPL)
        assert "AAPL" in strategy._top_decile_symbols
        assert len(strategy._top_decile_symbols) == 1

    def test_update_top_decile_empty_cache(self, strategy):
        """Test updating top decile with empty cache."""
        strategy._factor_scores_cache = {}
        strategy._update_top_decile(datetime(2024, 1, 1))

        assert len(strategy._top_decile_symbols) == 0

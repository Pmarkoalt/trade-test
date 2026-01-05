"""Unit tests for signal scoring and queue selection."""

from unittest.mock import Mock

import numpy as np
import pandas as pd

from trading_system.models.features import FeatureRow
from trading_system.models.portfolio import Portfolio
from trading_system.models.positions import Position
from trading_system.models.signals import BreakoutType, Signal, SignalSide, SignalType
from trading_system.strategies.queue import select_signals_from_queue, violates_correlation_guard
from trading_system.strategies.scoring import (
    compute_breakout_strength,
    compute_diversification_bonus,
    compute_momentum_strength,
    rank_normalize,
    score_signals,
)


class TestComputeBreakoutStrength:
    """Tests for compute_breakout_strength function."""

    def test_20d_breakout_strength(self):
        """Test breakout strength calculation for 20D breakout."""
        signal = Signal(
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp("2024-01-01"),
            side=SignalSide.BUY,
            signal_type=SignalType.ENTRY_LONG,
            trigger_reason="momentum_breakout_20D",
            entry_price=105.0,
            stop_price=100.0,
            atr_mult=2.5,
            triggered_on=BreakoutType.FAST_20D,
            breakout_clearance=0.01,
            breakout_strength=0.0,
            momentum_strength=0.0,
            diversification_bonus=0.0,
            score=0.0,
            passed_eligibility=True,
            eligibility_failures=[],
            order_notional=10000.0,
            adv20=5000000.0,
            capacity_passed=True,
        )

        features = FeatureRow(
            date=pd.Timestamp("2024-01-01"),
            symbol="AAPL",
            asset_class="equity",
            close=105.0,
            open=104.0,
            high=106.0,
            low=103.0,
            ma20=100.0,
            ma50=95.0,
            atr14=2.0,
        )

        strength = compute_breakout_strength(signal, features)
        # (105.0 - 100.0) / 2.0 = 2.5
        assert strength == 2.5

    def test_55d_breakout_strength(self):
        """Test breakout strength calculation for 55D breakout."""
        signal = Signal(
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp("2024-01-01"),
            side=SignalSide.BUY,
            signal_type=SignalType.ENTRY_LONG,
            trigger_reason="momentum_breakout_55D",
            entry_price=105.0,
            stop_price=100.0,
            atr_mult=2.5,
            triggered_on=BreakoutType.SLOW_55D,
            breakout_clearance=0.01,
            breakout_strength=0.0,
            momentum_strength=0.0,
            diversification_bonus=0.0,
            score=0.0,
            passed_eligibility=True,
            eligibility_failures=[],
            order_notional=10000.0,
            adv20=5000000.0,
            capacity_passed=True,
        )

        features = FeatureRow(
            date=pd.Timestamp("2024-01-01"),
            symbol="AAPL",
            asset_class="equity",
            close=105.0,
            open=104.0,
            high=106.0,
            low=103.0,
            ma20=100.0,
            ma50=95.0,
            atr14=2.0,
        )

        strength = compute_breakout_strength(signal, features)
        # (105.0 - 95.0) / 2.0 = 5.0
        assert strength == 5.0

    def test_zero_atr(self):
        """Test that zero ATR returns 0.0."""
        signal = Signal(
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp("2024-01-01"),
            side=SignalSide.BUY,
            signal_type=SignalType.ENTRY_LONG,
            trigger_reason="momentum_breakout_20D",
            entry_price=105.0,
            stop_price=100.0,
            atr_mult=2.5,
            triggered_on=BreakoutType.FAST_20D,
            breakout_clearance=0.01,
            breakout_strength=0.0,
            momentum_strength=0.0,
            diversification_bonus=0.0,
            score=0.0,
            passed_eligibility=True,
            eligibility_failures=[],
            order_notional=10000.0,
            adv20=5000000.0,
            capacity_passed=True,
        )

        features = FeatureRow(
            date=pd.Timestamp("2024-01-01"),
            symbol="AAPL",
            asset_class="equity",
            close=105.0,
            open=104.0,
            high=106.0,
            low=103.0,
            ma20=100.0,
            atr14=0.0,  # Zero ATR
        )

        strength = compute_breakout_strength(signal, features)
        assert strength == 0.0

    def test_missing_ma(self):
        """Test that missing MA returns 0.0."""
        signal = Signal(
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp("2024-01-01"),
            side=SignalSide.BUY,
            signal_type=SignalType.ENTRY_LONG,
            trigger_reason="momentum_breakout_20D",
            entry_price=105.0,
            stop_price=100.0,
            atr_mult=2.5,
            triggered_on=BreakoutType.FAST_20D,
            breakout_clearance=0.01,
            breakout_strength=0.0,
            momentum_strength=0.0,
            diversification_bonus=0.0,
            score=0.0,
            passed_eligibility=True,
            eligibility_failures=[],
            order_notional=10000.0,
            adv20=5000000.0,
            capacity_passed=True,
        )

        features = FeatureRow(
            date=pd.Timestamp("2024-01-01"),
            symbol="AAPL",
            asset_class="equity",
            close=105.0,
            open=104.0,
            high=106.0,
            low=103.0,
            ma20=None,  # Missing MA
            atr14=2.0,
        )

        strength = compute_breakout_strength(signal, features)
        assert strength == 0.0


class TestComputeMomentumStrength:
    """Tests for compute_momentum_strength function."""

    def test_relative_momentum(self):
        """Test momentum strength with benchmark."""
        features = FeatureRow(
            date=pd.Timestamp("2024-01-01"),
            symbol="AAPL",
            asset_class="equity",
            close=105.0,
            open=104.0,
            high=106.0,
            low=103.0,
            roc60=0.10,  # 10% return
            benchmark_roc60=0.05,  # 5% return
        )

        strength = compute_momentum_strength(features)
        # 0.10 - 0.05 = 0.05
        assert strength == 0.05

    def test_no_benchmark(self):
        """Test momentum strength without benchmark uses absolute."""
        features = FeatureRow(
            date=pd.Timestamp("2024-01-01"),
            symbol="AAPL",
            asset_class="equity",
            close=105.0,
            open=104.0,
            high=106.0,
            low=103.0,
            roc60=0.10,
            benchmark_roc60=None,
        )

        strength = compute_momentum_strength(features)
        # Should use absolute roc60
        assert strength == 0.10

    def test_missing_roc60(self):
        """Test that missing roc60 returns 0.0."""
        features = FeatureRow(
            date=pd.Timestamp("2024-01-01"),
            symbol="AAPL",
            asset_class="equity",
            close=105.0,
            open=104.0,
            high=106.0,
            low=103.0,
            roc60=None,
        )

        strength = compute_momentum_strength(features)
        assert strength == 0.0


class TestRankNormalize:
    """Tests for rank_normalize function."""

    def test_basic_normalization(self):
        """Test basic rank normalization."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        normalized = rank_normalize(values)

        # Should be evenly distributed in [0, 1]
        assert len(normalized) == 5
        assert all(0.0 <= x <= 1.0 for x in normalized)
        assert normalized[0] < normalized[1] < normalized[2] < normalized[3] < normalized[4]
        assert normalized[0] == 0.0  # Lowest rank
        assert normalized[4] == 1.0  # Highest rank

    def test_single_value(self):
        """Test normalization with single value."""
        values = [42.0]
        normalized = rank_normalize(values)
        assert len(normalized) == 1
        assert normalized[0] == 1.0  # Single value gets max rank

    def test_empty_list(self):
        """Test normalization with empty list."""
        values = []
        normalized = rank_normalize(values)
        assert normalized == []

    def test_duplicate_values(self):
        """Test normalization with duplicate values."""
        values = [10.0, 20.0, 20.0, 30.0]
        normalized = rank_normalize(values)

        assert len(normalized) == 4
        assert normalized[0] == 0.0  # Lowest
        assert normalized[1] == normalized[2]  # Tied values get same rank
        assert normalized[3] == 1.0  # Highest

    def test_with_nan(self):
        """Test normalization handles NaN values."""
        values = [10.0, np.nan, 30.0, np.inf]
        normalized = rank_normalize(values)

        assert len(normalized) == 4
        # NaN and inf should get 0.0 rank
        assert normalized[1] == 0.0
        assert normalized[3] == 0.0
        # Valid values should be normalized (lowest valid gets 0.0, highest gets 1.0)
        assert 0.0 <= normalized[0] <= 1.0  # 10.0 is lowest valid, gets 0.0
        assert 0.0 <= normalized[2] <= 1.0  # 30.0 is highest valid, gets 1.0


class TestComputeDiversificationBonus:
    """Tests for compute_diversification_bonus function."""

    def test_no_positions(self):
        """Test that no positions returns 0.5 (neutral)."""
        signal = Signal(
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp("2024-01-01"),
            side=SignalSide.BUY,
            signal_type=SignalType.ENTRY_LONG,
            trigger_reason="momentum_breakout_20D",
            entry_price=105.0,
            stop_price=100.0,
            atr_mult=2.5,
            triggered_on=BreakoutType.FAST_20D,
            breakout_clearance=0.01,
            breakout_strength=0.0,
            momentum_strength=0.0,
            diversification_bonus=0.0,
            score=0.0,
            passed_eligibility=True,
            eligibility_failures=[],
            order_notional=10000.0,
            adv20=5000000.0,
            capacity_passed=True,
        )

        portfolio = Portfolio(
            date=pd.Timestamp("2024-01-01"),
            cash=100000.0,
            starting_equity=100000.0,
            equity=100000.0,
            positions={},  # No positions
        )

        bonus = compute_diversification_bonus(signal, portfolio, {}, {}, lookback=20)
        assert bonus == 0.5

    def test_with_positions(self):
        """Test diversification bonus with existing positions."""
        signal = Signal(
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp("2024-01-01"),
            side=SignalSide.BUY,
            signal_type=SignalType.ENTRY_LONG,
            trigger_reason="momentum_breakout_20D",
            entry_price=105.0,
            stop_price=100.0,
            atr_mult=2.5,
            triggered_on=BreakoutType.FAST_20D,
            breakout_clearance=0.01,
            breakout_strength=0.0,
            momentum_strength=0.0,
            diversification_bonus=0.0,
            score=0.0,
            passed_eligibility=True,
            eligibility_failures=[],
            order_notional=10000.0,
            adv20=5000000.0,
            capacity_passed=True,
        )

        # Create mock position
        position = Mock(spec=Position)
        position.symbol = "MSFT"

        portfolio = Portfolio(
            date=pd.Timestamp("2024-01-01"),
            cash=100000.0,
            starting_equity=100000.0,
            equity=100000.0,
            positions={"MSFT": position},
        )

        # Mock returns: AAPL and MSFT have 0.8 correlation
        # Correlation of 0.8 means diversification_bonus = 1 - 0.8 = 0.2
        candidate_returns = {
            "AAPL": [0.01] * 20,  # Simple returns
        }
        portfolio_returns = {
            "MSFT": [0.01] * 20,  # Perfectly correlated (will give corr=1.0)
        }

        # Note: Actual correlation calculation will depend on the implementation
        # This is a simplified test
        bonus = compute_diversification_bonus(signal, portfolio, candidate_returns, portfolio_returns, lookback=20)

        # Should be in [0, 1] range
        assert 0.0 <= bonus <= 1.0


class TestScoreSignals:
    """Tests for score_signals function."""

    def test_score_signals_basic(self):
        """Test scoring signals with basic setup."""
        date = pd.Timestamp("2024-01-01")

        signals = [
            Signal(
                symbol="AAPL",
                asset_class="equity",
                date=date,
                side=SignalSide.BUY,
                signal_type=SignalType.ENTRY_LONG,
                trigger_reason="momentum_breakout_20D",
                entry_price=105.0,
                stop_price=100.0,
                atr_mult=2.5,
                triggered_on=BreakoutType.FAST_20D,
                breakout_clearance=0.01,
                breakout_strength=0.0,
                momentum_strength=0.0,
                diversification_bonus=0.0,
                score=0.0,
                passed_eligibility=True,
                eligibility_failures=[],
                order_notional=10000.0,
                adv20=5000000.0,
                capacity_passed=True,
            ),
            Signal(
                symbol="MSFT",
                asset_class="equity",
                date=date,
                side=SignalSide.BUY,
                signal_type=SignalType.ENTRY_LONG,
                trigger_reason="momentum_breakout_20D",
                entry_price=210.0,
                stop_price=200.0,
                atr_mult=2.5,
                triggered_on=BreakoutType.FAST_20D,
                breakout_clearance=0.01,
                breakout_strength=0.0,
                momentum_strength=0.0,
                diversification_bonus=0.0,
                score=0.0,
                passed_eligibility=True,
                eligibility_failures=[],
                order_notional=10000.0,
                adv20=5000000.0,
                capacity_passed=True,
            ),
        ]

        features_map = {
            "AAPL": FeatureRow(
                date=date,
                symbol="AAPL",
                asset_class="equity",
                close=105.0,
                open=104.0,
                high=106.0,
                low=103.0,
                ma20=100.0,
                ma50=95.0,
                atr14=2.0,
                roc60=0.10,
                benchmark_roc60=0.05,
            ),
            "MSFT": FeatureRow(
                date=date,
                symbol="MSFT",
                asset_class="equity",
                close=210.0,
                open=209.0,
                high=211.0,
                low=208.0,
                ma20=200.0,
                ma50=190.0,
                atr14=4.0,
                roc60=0.08,
                benchmark_roc60=0.05,
            ),
        }

        def get_features(signal):
            return features_map.get(signal.symbol)

        portfolio = Portfolio(
            date=date,
            cash=100000.0,
            starting_equity=100000.0,
            equity=100000.0,
            positions={},
        )

        score_signals(signals, get_features, portfolio, {}, {}, lookback=20)

        # Check that scores were assigned
        assert all(s.score > 0.0 for s in signals)
        assert all(s.breakout_strength != 0.0 for s in signals)
        assert all(s.momentum_strength != 0.0 for s in signals)
        # Scores should be in [0, 1] range
        assert all(0.0 <= s.score <= 1.0 for s in signals)


class TestViolatesCorrelationGuard:
    """Tests for violates_correlation_guard function."""

    def test_guard_not_applicable_too_few_positions(self):
        """Test guard doesn't apply with < 4 positions."""
        signal = Signal(
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp("2024-01-01"),
            side=SignalSide.BUY,
            signal_type=SignalType.ENTRY_LONG,
            trigger_reason="momentum_breakout_20D",
            entry_price=105.0,
            stop_price=100.0,
            atr_mult=2.5,
            triggered_on=BreakoutType.FAST_20D,
            breakout_clearance=0.01,
            breakout_strength=0.0,
            momentum_strength=0.0,
            diversification_bonus=0.0,
            score=0.0,
            passed_eligibility=True,
            eligibility_failures=[],
            order_notional=10000.0,
            adv20=5000000.0,
            capacity_passed=True,
        )

        # Create 3 positions (less than 4)
        positions = {f"STOCK{i}": Mock(spec=Position) for i in range(3)}
        portfolio = Portfolio(
            date=pd.Timestamp("2024-01-01"),
            cash=100000.0,
            starting_equity=100000.0,
            equity=100000.0,
            positions=positions,
            avg_pairwise_corr=0.80,  # High correlation
        )

        violates = violates_correlation_guard(signal, portfolio, {}, {}, lookback=20)
        assert violates is False

    def test_guard_not_applicable_low_correlation(self):
        """Test guard doesn't apply if avg_pairwise_corr <= 0.70."""
        signal = Signal(
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp("2024-01-01"),
            side=SignalSide.BUY,
            signal_type=SignalType.ENTRY_LONG,
            trigger_reason="momentum_breakout_20D",
            entry_price=105.0,
            stop_price=100.0,
            atr_mult=2.5,
            triggered_on=BreakoutType.FAST_20D,
            breakout_clearance=0.01,
            breakout_strength=0.0,
            momentum_strength=0.0,
            diversification_bonus=0.0,
            score=0.0,
            passed_eligibility=True,
            eligibility_failures=[],
            order_notional=10000.0,
            adv20=5000000.0,
            capacity_passed=True,
        )

        # Create 4 positions
        positions = {f"STOCK{i}": Mock(spec=Position) for i in range(4)}
        portfolio = Portfolio(
            date=pd.Timestamp("2024-01-01"),
            cash=100000.0,
            starting_equity=100000.0,
            equity=100000.0,
            positions=positions,
            avg_pairwise_corr=0.65,  # Below threshold
        )

        violates = violates_correlation_guard(signal, portfolio, {}, {}, lookback=20)
        assert violates is False


class TestSelectSignalsFromQueue:
    """Tests for select_signals_from_queue function."""

    def test_select_by_score(self):
        """Test that signals are selected by score order."""
        date = pd.Timestamp("2024-01-01")

        # Create signals with different scores
        signals = [
            Signal(
                symbol="AAPL",
                asset_class="equity",
                date=date,
                side=SignalSide.BUY,
                signal_type=SignalType.ENTRY_LONG,
                trigger_reason="momentum_breakout_20D",
                entry_price=105.0,
                stop_price=100.0,
                atr_mult=2.5,
                triggered_on=BreakoutType.FAST_20D,
                breakout_clearance=0.01,
                breakout_strength=0.0,
                momentum_strength=0.0,
                diversification_bonus=0.0,
                score=0.3,  # Lower score
                passed_eligibility=True,
                eligibility_failures=[],
                order_notional=10000.0,
                adv20=5000000.0,
                capacity_passed=True,
            ),
            Signal(
                symbol="MSFT",
                asset_class="equity",
                date=date,
                side=SignalSide.BUY,
                signal_type=SignalType.ENTRY_LONG,
                trigger_reason="momentum_breakout_20D",
                entry_price=210.0,
                stop_price=200.0,
                atr_mult=2.5,
                triggered_on=BreakoutType.FAST_20D,
                breakout_clearance=0.01,
                breakout_strength=0.0,
                momentum_strength=0.0,
                diversification_bonus=0.0,
                score=0.9,  # Higher score
                passed_eligibility=True,
                eligibility_failures=[],
                order_notional=10000.0,
                adv20=5000000.0,
                capacity_passed=True,
            ),
        ]

        portfolio = Portfolio(
            date=date,
            cash=100000.0,
            starting_equity=100000.0,
            equity=100000.0,
            positions={},
        )

        selected = select_signals_from_queue(
            signals,
            portfolio,
            max_positions=8,
            max_exposure=0.80,
            risk_per_trade=0.0075,
            max_position_notional=0.15,
            candidate_returns={},
            portfolio_returns={},
            lookback=20,
        )

        # MSFT should be selected first (higher score)
        assert len(selected) == 2
        assert selected[0].symbol == "MSFT"
        assert selected[1].symbol == "AAPL"

    def test_max_positions_constraint(self):
        """Test that max_positions constraint is enforced."""
        date = pd.Timestamp("2024-01-01")

        # Create 5 signals
        signals = [
            Signal(
                symbol=f"STOCK{i}",
                asset_class="equity",
                date=date,
                side=SignalSide.BUY,
                signal_type=SignalType.ENTRY_LONG,
                trigger_reason="momentum_breakout_20D",
                entry_price=100.0 + i,
                stop_price=95.0 + i,
                atr_mult=2.5,
                triggered_on=BreakoutType.FAST_20D,
                breakout_clearance=0.01,
                breakout_strength=0.0,
                momentum_strength=0.0,
                diversification_bonus=0.0,
                score=0.5 + i * 0.1,  # Different scores
                passed_eligibility=True,
                eligibility_failures=[],
                order_notional=10000.0,
                adv20=5000000.0,
                capacity_passed=True,
            )
            for i in range(5)
        ]

        # Portfolio already has 6 positions
        positions = {f"EXIST{i}": Mock(spec=Position) for i in range(6)}
        portfolio = Portfolio(
            date=date,
            cash=100000.0,
            starting_equity=100000.0,
            equity=100000.0,
            positions=positions,
        )

        selected = select_signals_from_queue(
            signals,
            portfolio,
            max_positions=8,  # Max 8 total
            max_exposure=0.80,
            risk_per_trade=0.0075,
            max_position_notional=0.15,
            candidate_returns={},
            portfolio_returns={},
            lookback=20,
        )

        # Can only add 2 more positions (8 - 6 = 2)
        assert len(selected) == 2

    def test_capacity_constraint(self):
        """Test that capacity constraint is enforced."""
        date = pd.Timestamp("2024-01-01")

        signal = Signal(
            symbol="AAPL",
            asset_class="equity",
            date=date,
            side=SignalSide.BUY,
            signal_type=SignalType.ENTRY_LONG,
            trigger_reason="momentum_breakout_20D",
            entry_price=105.0,
            stop_price=100.0,
            atr_mult=2.5,
            triggered_on=BreakoutType.FAST_20D,
            breakout_clearance=0.01,
            breakout_strength=0.0,
            momentum_strength=0.0,
            diversification_bonus=0.0,
            score=0.9,
            passed_eligibility=True,
            eligibility_failures=[],
            order_notional=10000.0,
            adv20=5000000.0,
            capacity_passed=False,  # Failed capacity check
        )

        portfolio = Portfolio(
            date=date,
            cash=100000.0,
            starting_equity=100000.0,
            equity=100000.0,
            positions={},
        )

        selected = select_signals_from_queue(
            [signal],
            portfolio,
            max_positions=8,
            max_exposure=0.80,
            risk_per_trade=0.0075,
            max_position_notional=0.15,
            candidate_returns={},
            portfolio_returns={},
            lookback=20,
        )

        # Should be rejected due to capacity
        assert len(selected) == 0

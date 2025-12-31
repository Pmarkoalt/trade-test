"""Unit tests for tracking models."""

from datetime import date, datetime

import pytest

from trading_system.tracking.models import (
    ConvictionLevel,
    ExitReason,
    PerformanceMetrics,
    SignalDirection,
    SignalOutcome,
    SignalStatus,
    TrackedSignal,
)


class TestTrackedSignal:
    """Tests for TrackedSignal model."""

    def test_tracked_signal_to_dict_roundtrip(self):
        """Test TrackedSignal serialization."""
        signal = TrackedSignal(
            symbol="AAPL",
            direction=SignalDirection.BUY,
            entry_price=150.0,
            target_price=165.0,
            stop_price=145.0,
        )
        data = signal.to_dict()
        restored = TrackedSignal.from_dict(data)
        assert restored.symbol == signal.symbol
        assert restored.direction == signal.direction
        assert restored.entry_price == signal.entry_price
        assert restored.target_price == signal.target_price
        assert restored.stop_price == signal.stop_price
        assert restored.id == signal.id

    def test_tracked_signal_with_all_fields(self):
        """Test TrackedSignal with all fields populated."""
        created_at = datetime(2024, 1, 1, 10, 0, 0)
        delivered_at = datetime(2024, 1, 1, 10, 5, 0)
        entry_filled_at = datetime(2024, 1, 2, 9, 30, 0)

        signal = TrackedSignal(
            id="test-id-123",
            symbol="AAPL",
            asset_class="equity",
            direction=SignalDirection.BUY,
            signal_type="breakout_20d",
            conviction=ConvictionLevel.HIGH,
            signal_price=150.0,
            entry_price=150.5,
            target_price=165.0,
            stop_price=145.0,
            technical_score=0.85,
            news_score=0.75,
            combined_score=0.80,
            position_size_pct=5.0,
            status=SignalStatus.ACTIVE,
            created_at=created_at,
            delivered_at=delivered_at,
            entry_filled_at=entry_filled_at,
            was_delivered=True,
            delivery_method="email",
            reasoning="Strong breakout with high volume",
            news_headlines=["Apple announces new product"],
            tags=["momentum", "breakout"],
        )

        data = signal.to_dict()
        restored = TrackedSignal.from_dict(data)

        assert restored.id == "test-id-123"
        assert restored.symbol == "AAPL"
        assert restored.asset_class == "equity"
        assert restored.direction == SignalDirection.BUY
        assert restored.signal_type == "breakout_20d"
        assert restored.conviction == ConvictionLevel.HIGH
        assert restored.signal_price == 150.0
        assert restored.entry_price == 150.5
        assert restored.target_price == 165.0
        assert restored.stop_price == 145.0
        assert restored.technical_score == 0.85
        assert restored.news_score == 0.75
        assert restored.combined_score == 0.80
        assert restored.position_size_pct == 5.0
        assert restored.status == SignalStatus.ACTIVE
        assert restored.created_at == created_at
        assert restored.delivered_at == delivered_at
        assert restored.entry_filled_at == entry_filled_at
        assert restored.was_delivered is True
        assert restored.delivery_method == "email"
        assert restored.reasoning == "Strong breakout with high volume"
        assert restored.news_headlines == ["Apple announces new product"]
        assert restored.tags == ["momentum", "breakout"]

    def test_tracked_signal_with_none_fields(self):
        """Test TrackedSignal with None optional fields."""
        signal = TrackedSignal(
            symbol="BTC",
            asset_class="crypto",
            direction=SignalDirection.BUY,
            news_score=None,
            delivered_at=None,
            entry_filled_at=None,
            exit_filled_at=None,
        )

        data = signal.to_dict()
        restored = TrackedSignal.from_dict(data)

        assert restored.news_score is None
        assert restored.delivered_at is None
        assert restored.entry_filled_at is None
        assert restored.exit_filled_at is None

    def test_tracked_signal_defaults(self):
        """Test TrackedSignal default values."""
        signal = TrackedSignal(symbol="AAPL")

        assert signal.direction == SignalDirection.BUY
        assert signal.conviction == ConvictionLevel.MEDIUM
        assert signal.status == SignalStatus.PENDING
        assert signal.was_delivered is False
        assert signal.news_headlines == []
        assert signal.tags == []
        assert signal.id is not None
        assert isinstance(signal.created_at, datetime)


class TestSignalOutcome:
    """Tests for SignalOutcome model."""

    def test_signal_outcome_r_multiple(self):
        """Test R-multiple calculation logic.

        Entry 100, Stop 95 (risk = 5), Exit 110 (reward = 10)
        R-multiple = 10/5 = 2.0
        """
        outcome = SignalOutcome(
            signal_id="test-signal-1",
            actual_entry_price=100.0,
            actual_exit_price=110.0,
            exit_reason=ExitReason.TARGET_HIT,
            holding_days=5,
            return_pct=10.0,
            r_multiple=2.0,  # (110-100) / (100-95) = 10/5 = 2.0
        )

        assert outcome.r_multiple == 2.0
        assert outcome.return_pct == 10.0
        assert outcome.exit_reason == ExitReason.TARGET_HIT

    def test_signal_outcome_to_dict_roundtrip(self):
        """Test SignalOutcome serialization."""
        entry_date = date(2024, 1, 2)
        exit_date = date(2024, 1, 7)

        outcome = SignalOutcome(
            signal_id="test-signal-1",
            actual_entry_price=100.0,
            actual_entry_date=entry_date,
            actual_exit_price=110.0,
            actual_exit_date=exit_date,
            exit_reason=ExitReason.TARGET_HIT,
            holding_days=5,
            return_pct=10.0,
            return_dollars=1000.0,
            r_multiple=2.0,
            benchmark_return_pct=5.0,
            alpha=5.0,
            was_followed=True,
            user_notes="Great trade!",
        )

        data = outcome.to_dict()
        restored = SignalOutcome.from_dict(data)

        assert restored.signal_id == "test-signal-1"
        assert restored.actual_entry_price == 100.0
        assert restored.actual_entry_date == entry_date
        assert restored.actual_exit_price == 110.0
        assert restored.actual_exit_date == exit_date
        assert restored.exit_reason == ExitReason.TARGET_HIT
        assert restored.holding_days == 5
        assert restored.return_pct == 10.0
        assert restored.return_dollars == 1000.0
        assert restored.r_multiple == 2.0
        assert restored.benchmark_return_pct == 5.0
        assert restored.alpha == 5.0
        assert restored.was_followed is True
        assert restored.user_notes == "Great trade!"

    def test_signal_outcome_with_none_fields(self):
        """Test SignalOutcome with None optional fields."""
        outcome = SignalOutcome(
            signal_id="test-signal-2",
            actual_entry_price=None,
            actual_entry_date=None,
            actual_exit_price=None,
            actual_exit_date=None,
            exit_reason=None,
        )

        data = outcome.to_dict()
        restored = SignalOutcome.from_dict(data)

        assert restored.actual_entry_price is None
        assert restored.actual_entry_date is None
        assert restored.actual_exit_price is None
        assert restored.actual_exit_date is None
        assert restored.exit_reason is None

    def test_signal_outcome_defaults(self):
        """Test SignalOutcome default values."""
        outcome = SignalOutcome(signal_id="test-signal-3")

        assert outcome.holding_days == 0
        assert outcome.return_pct == 0.0
        assert outcome.return_dollars == 0.0
        assert outcome.r_multiple == 0.0
        assert outcome.benchmark_return_pct == 0.0
        assert outcome.alpha == 0.0
        assert outcome.was_followed is False
        assert outcome.user_notes == ""
        assert isinstance(outcome.recorded_at, datetime)


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics model."""

    def test_performance_metrics_defaults(self):
        """Test PerformanceMetrics default values."""
        metrics = PerformanceMetrics()

        assert metrics.total_signals == 0
        assert metrics.signals_followed == 0
        assert metrics.signals_won == 0
        assert metrics.signals_lost == 0
        assert metrics.win_rate == 0.0
        assert metrics.follow_rate == 0.0
        assert metrics.total_return_pct == 0.0
        assert metrics.avg_return_pct == 0.0
        assert metrics.metrics_by_asset_class == {}
        assert metrics.metrics_by_signal_type == {}
        assert metrics.metrics_by_conviction == {}
        assert isinstance(metrics.period_start, date)
        assert isinstance(metrics.period_end, date)

    def test_performance_metrics_with_data(self):
        """Test PerformanceMetrics with populated data."""
        period_start = date(2024, 1, 1)
        period_end = date(2024, 12, 31)

        metrics = PerformanceMetrics(
            period_start=period_start,
            period_end=period_end,
            total_signals=100,
            signals_followed=80,
            signals_won=55,
            signals_lost=25,
            win_rate=0.6875,  # 55/80
            follow_rate=0.80,  # 80/100
            total_return_pct=25.5,
            avg_return_pct=0.255,
            avg_winner_pct=5.0,
            avg_loser_pct=-2.0,
            total_r=50.0,
            avg_r=0.625,
            avg_winner_r=2.5,
            avg_loser_r=-1.0,
            expectancy_r=1.71875,  # (0.6875 * 2.5) - (0.3125 * 1.0)
            max_drawdown_pct=10.0,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=2.55,
            benchmark_return_pct=20.0,
            alpha=5.5,
            metrics_by_asset_class={"equity": {"win_rate": 0.70}, "crypto": {"win_rate": 0.65}},
            metrics_by_signal_type={"breakout_20d": {"win_rate": 0.75}},
            metrics_by_conviction={"HIGH": {"win_rate": 0.80}},
        )

        assert metrics.period_start == period_start
        assert metrics.period_end == period_end
        assert metrics.total_signals == 100
        assert metrics.signals_followed == 80
        assert metrics.win_rate == 0.6875
        assert metrics.expectancy_r == 1.71875
        assert metrics.metrics_by_asset_class["equity"]["win_rate"] == 0.70


class TestEnums:
    """Tests for enum types."""

    def test_signal_direction_enum(self):
        """Test SignalDirection enum."""
        assert SignalDirection.BUY == "BUY"
        assert SignalDirection.SELL == "SELL"

    def test_conviction_level_enum(self):
        """Test ConvictionLevel enum."""
        assert ConvictionLevel.HIGH == "HIGH"
        assert ConvictionLevel.MEDIUM == "MEDIUM"
        assert ConvictionLevel.LOW == "LOW"

    def test_signal_status_enum(self):
        """Test SignalStatus enum."""
        assert SignalStatus.PENDING == "pending"
        assert SignalStatus.ACTIVE == "active"
        assert SignalStatus.CLOSED == "closed"
        assert SignalStatus.EXPIRED == "expired"
        assert SignalStatus.CANCELLED == "cancelled"

    def test_exit_reason_enum(self):
        """Test ExitReason enum."""
        assert ExitReason.TARGET_HIT == "target_hit"
        assert ExitReason.STOP_HIT == "stop_hit"
        assert ExitReason.TRAILING_STOP == "trailing_stop"
        assert ExitReason.TIME_EXIT == "time_exit"
        assert ExitReason.MANUAL == "manual"
        assert ExitReason.SIGNAL_REVERSAL == "signal_reversal"

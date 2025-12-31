"""Integration tests for performance tracking."""

from datetime import date, timedelta

import pytest

from trading_system.tracking.analytics.signal_analytics import SignalAnalyzer
from trading_system.tracking.models import ConvictionLevel, ExitReason, SignalDirection, SignalStatus
from trading_system.tracking.outcome_recorder import OutcomeRecorder
from trading_system.tracking.performance_calculator import PerformanceCalculator
from trading_system.tracking.reports.leaderboard import LeaderboardGenerator
from trading_system.tracking.signal_tracker import SignalTracker
from trading_system.tracking.storage.sqlite_store import SQLiteTrackingStore


@pytest.fixture
def tracking_store(tmp_path):
    """Create a temporary tracking store."""
    db_path = tmp_path / "test_tracking.db"
    store = SQLiteTrackingStore(str(db_path))
    store.initialize()
    yield store
    store.close()


@pytest.fixture
def populated_store(tracking_store):
    """Create store with sample data."""
    tracker = SignalTracker(tracking_store)
    recorder = OutcomeRecorder(tracking_store)

    # Create sample signals and outcomes
    test_data = [
        # (symbol, direction, conviction, entry, exit, r_multiple, exit_reason)
        ("AAPL", "BUY", "HIGH", 150.0, 165.0, 2.0, "target_hit"),
        ("MSFT", "BUY", "MEDIUM", 300.0, 310.0, 1.5, "target_hit"),
        ("GOOGL", "BUY", "HIGH", 140.0, 135.0, -1.0, "stop_hit"),
        ("NVDA", "BUY", "LOW", 450.0, 440.0, -1.0, "stop_hit"),
        ("AMZN", "BUY", "HIGH", 180.0, 195.0, 2.5, "target_hit"),
        ("META", "BUY", "MEDIUM", 350.0, 365.0, 1.8, "target_hit"),
        ("TSLA", "BUY", "LOW", 250.0, 240.0, -1.2, "stop_hit"),
        ("BTC", "BUY", "HIGH", 45000.0, 48000.0, 1.5, "target_hit"),
        ("ETH", "BUY", "MEDIUM", 2500.0, 2400.0, -0.8, "stop_hit"),
        ("SOL", "BUY", "LOW", 100.0, 115.0, 2.0, "target_hit"),
    ]

    base_date = date.today() - timedelta(days=30)

    for i, (symbol, direction, conviction, entry, exit_price, r_mult, reason) in enumerate(test_data):
        # Create signal
        signal_id = tracker.record_signal(
            symbol=symbol,
            asset_class="equity" if symbol not in ["BTC", "ETH", "SOL"] else "crypto",
            direction=SignalDirection(direction),
            signal_type="breakout_20d",
            conviction=ConvictionLevel(conviction),
            signal_price=entry,
            entry_price=entry,
            target_price=entry * 1.10,
            stop_price=entry * 0.95,
            technical_score=7.0,
            combined_score=7.0,
        )

        # Mark delivered
        tracker.mark_delivered(signal_id, method="email")

        # Record outcome with dates spread out
        entry_date = base_date + timedelta(days=i * 2)
        exit_date = entry_date + timedelta(days=5)
        recorder.record_outcome(
            signal_id=signal_id,
            entry_price=entry,
            exit_price=exit_price,
            exit_reason=ExitReason(reason),
            entry_date=entry_date,
            exit_date=exit_date,
            was_followed=True,
        )

    return tracking_store


class TestTrackingIntegration:
    """Integration tests for tracking pipeline."""

    def test_full_signal_lifecycle(self, tracking_store):
        """Test complete signal lifecycle: create -> deliver -> close."""
        tracker = SignalTracker(tracking_store)
        recorder = OutcomeRecorder(tracking_store)

        # 1. Record signal
        signal_id = tracker.record_signal(
            symbol="TEST",
            asset_class="equity",
            direction=SignalDirection.BUY,
            signal_type="breakout_20d",
            conviction=ConvictionLevel.HIGH,
            signal_price=100.0,
            entry_price=100.0,
            target_price=110.0,
            stop_price=95.0,
        )

        # Verify pending
        signal = tracker.get_signal(signal_id)
        assert signal is not None
        assert signal.status == SignalStatus.PENDING

        # 2. Mark delivered
        tracker.mark_delivered(signal_id, method="email")
        signal = tracker.get_signal(signal_id)
        assert signal.was_delivered
        assert signal.delivery_method == "email"

        # 3. Mark entry filled
        tracker.mark_entry_filled(signal_id)
        signal = tracker.get_signal(signal_id)
        assert signal.status == SignalStatus.ACTIVE

        # 4. Record outcome
        recorder.record_outcome(
            signal_id=signal_id,
            entry_price=100.0,
            exit_price=108.0,
            exit_reason=ExitReason.TARGET_HIT,
        )

        # Verify closed
        signal = tracker.get_signal(signal_id)
        assert signal.status == SignalStatus.CLOSED

        outcome = recorder.get_outcome(signal_id)
        assert outcome is not None
        assert outcome.return_pct == pytest.approx(0.08, rel=0.01)
        # R-multiple: (108-100) / (100-95) = 8/5 = 1.6
        assert outcome.r_multiple == pytest.approx(1.6, rel=0.01)

    def test_performance_metrics(self, populated_store):
        """Test performance calculation from populated data."""
        calculator = PerformanceCalculator(populated_store)

        metrics = calculator.calculate_metrics()

        # Should have 10 signals
        assert metrics.total_signals == 10
        assert metrics.signals_followed == 10

        # 6 winners, 4 losers = 60% win rate
        assert metrics.signals_won == 6
        assert metrics.signals_lost == 4
        assert metrics.win_rate == pytest.approx(0.6, rel=0.01)

    def test_signal_analytics(self, populated_store):
        """Test signal analytics generation."""
        analyzer = SignalAnalyzer(populated_store)

        analytics = analyzer.analyze()

        # Should have conviction breakdown
        assert "HIGH" in analytics.performance_by_conviction
        assert "MEDIUM" in analytics.performance_by_conviction
        assert "LOW" in analytics.performance_by_conviction

        # HIGH conviction should have best win rate (4 HIGH: 3 wins, 1 loss = 75%)
        # LOW conviction: 3 LOW: 1 win, 2 losses = 33%
        high_wr = analytics.performance_by_conviction["HIGH"]["win_rate"]
        low_wr = analytics.performance_by_conviction["LOW"]["win_rate"]
        assert high_wr >= low_wr

        # Should generate insights
        assert len(analytics.insights) > 0

    def test_leaderboard_generation(self, populated_store):
        """Test strategy leaderboard."""
        generator = LeaderboardGenerator(populated_store)

        leaderboard = generator.generate_all_time()

        # Should have entries
        assert leaderboard.total_strategies > 0

        # First entry should be top performer
        if leaderboard.entries:
            top = leaderboard.entries[0]
            assert top.rank == 1

    def test_rolling_metrics(self, populated_store):
        """Test rolling metrics calculation."""
        calculator = PerformanceCalculator(populated_store)

        rolling = calculator.calculate_rolling_metrics(window_days=30)

        assert "win_rate" in rolling
        assert "avg_r" in rolling
        assert "expectancy_r" in rolling

    def test_equity_curve(self, populated_store):
        """Test equity curve generation."""
        calculator = PerformanceCalculator(populated_store)

        curve = calculator.get_equity_curve(starting_equity=100000.0)

        # Should have entries
        assert len(curve) > 0

        # Each entry should have required fields
        for point in curve:
            assert "equity" in point
            assert "drawdown_pct" in point


class TestTrackingEdgeCases:
    """Edge case tests for tracking."""

    def test_empty_database(self, tracking_store):
        """Test with no data."""
        calculator = PerformanceCalculator(tracking_store)

        metrics = calculator.calculate_metrics()
        assert metrics.total_signals == 0
        assert metrics.win_rate == 0.0

    def test_signal_not_followed(self, tracking_store):
        """Test signal that wasn't followed."""
        tracker = SignalTracker(tracking_store)
        recorder = OutcomeRecorder(tracking_store)

        signal_id = tracker.record_signal(
            symbol="TEST",
            asset_class="equity",
            direction=SignalDirection.BUY,
            signal_type="test",
            conviction=ConvictionLevel.LOW,
            signal_price=100.0,
            entry_price=100.0,
            target_price=110.0,
            stop_price=95.0,
        )

        # Mark as not followed
        recorder.record_missed_signal(signal_id, "Too risky")

        signal = tracker.get_signal(signal_id)
        assert signal.status == SignalStatus.EXPIRED

    def test_duplicate_outcome(self, tracking_store):
        """Test recording duplicate outcome."""
        tracker = SignalTracker(tracking_store)
        recorder = OutcomeRecorder(tracking_store)

        signal_id = tracker.record_signal(
            symbol="TEST",
            asset_class="equity",
            direction=SignalDirection.BUY,
            signal_type="test",
            conviction=ConvictionLevel.HIGH,
            signal_price=100.0,
            entry_price=100.0,
            target_price=110.0,
            stop_price=95.0,
        )

        # First outcome
        success1 = recorder.record_outcome(
            signal_id=signal_id,
            entry_price=100.0,
            exit_price=105.0,
            exit_reason=ExitReason.MANUAL,
        )
        assert success1

        # Second outcome should fail or update
        # (depending on implementation choice - SQLite store should allow update)
        _ = recorder.record_outcome(
            signal_id=signal_id,
            entry_price=100.0,
            exit_price=110.0,
            exit_reason=ExitReason.TARGET_HIT,
        )
        # The implementation may allow updates, so we just check that it doesn't crash
        # If it raises an exception, that's also acceptable behavior
        outcome = recorder.get_outcome(signal_id)
        assert outcome is not None

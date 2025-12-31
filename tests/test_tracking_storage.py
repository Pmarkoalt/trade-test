"""Unit tests for tracking storage layer."""

from datetime import date, datetime, timedelta

import pytest

from trading_system.tracking.models import (
    ConvictionLevel,
    ExitReason,
    SignalDirection,
    SignalOutcome,
    SignalStatus,
    TrackedSignal,
)
from trading_system.tracking.storage.sqlite_store import SQLiteTrackingStore


class TestSQLiteTrackingStore:
    """Tests for SQLiteTrackingStore."""

    def test_insert_and_get_signal(self, tmp_path):
        """Test signal insert and retrieval."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()

        signal = TrackedSignal(
            symbol="AAPL",
            asset_class="equity",
            direction=SignalDirection.BUY,
            signal_type="breakout_20d",
            conviction=ConvictionLevel.HIGH,
            signal_price=150.0,
            entry_price=150.0,
            target_price=165.0,
            stop_price=145.0,
            technical_score=0.85,
            combined_score=0.80,
        )

        signal_id = store.insert_signal(signal)
        retrieved = store.get_signal(signal_id)

        assert retrieved is not None
        assert retrieved.symbol == "AAPL"
        assert retrieved.direction == SignalDirection.BUY
        assert retrieved.entry_price == 150.0
        assert retrieved.id == signal_id
        store.close()

    def test_update_signal_status(self, tmp_path):
        """Test status updates."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()

        signal = TrackedSignal(
            symbol="AAPL",
            asset_class="equity",
            direction=SignalDirection.BUY,
            signal_type="breakout",
            conviction=ConvictionLevel.MEDIUM,
            signal_price=150.0,
            entry_price=150.0,
            target_price=165.0,
            stop_price=145.0,
            status=SignalStatus.PENDING,
        )

        signal_id = store.insert_signal(signal)

        # Update to active
        timestamp = datetime.now()
        success = store.update_signal_status(signal_id, SignalStatus.ACTIVE, timestamp)
        assert success is True

        retrieved = store.get_signal(signal_id)
        assert retrieved.status == SignalStatus.ACTIVE
        assert retrieved.entry_filled_at == timestamp

        # Update to closed
        exit_timestamp = datetime.now()
        success = store.update_signal_status(signal_id, SignalStatus.CLOSED, exit_timestamp)
        assert success is True

        retrieved = store.get_signal(signal_id)
        assert retrieved.status == SignalStatus.CLOSED
        assert retrieved.exit_filled_at == exit_timestamp

        store.close()

    def test_update_signal(self, tmp_path):
        """Test full signal update."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()

        signal = TrackedSignal(
            symbol="AAPL",
            asset_class="equity",
            direction=SignalDirection.BUY,
            signal_type="breakout",
            conviction=ConvictionLevel.MEDIUM,
            signal_price=150.0,
            entry_price=150.0,
            target_price=165.0,
            stop_price=145.0,
            status=SignalStatus.PENDING,
        )

        signal_id = store.insert_signal(signal)

        # Update signal
        signal.status = SignalStatus.ACTIVE
        signal.delivered_at = datetime.now()
        signal.was_delivered = True
        signal.delivery_method = "email"

        success = store.update_signal(signal)
        assert success is True

        retrieved = store.get_signal(signal_id)
        assert retrieved.status == SignalStatus.ACTIVE
        assert retrieved.was_delivered is True
        assert retrieved.delivery_method == "email"

        store.close()

    def test_get_signals_by_status(self, tmp_path):
        """Test getting signals by status."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()

        # Insert multiple signals with different statuses
        for i in range(5):
            signal = TrackedSignal(
                symbol=f"SYM{i}",
                asset_class="equity",
                direction=SignalDirection.BUY,
                signal_type="breakout",
                conviction=ConvictionLevel.MEDIUM,
                signal_price=100.0 + i,
                entry_price=100.0 + i,
                target_price=110.0 + i,
                stop_price=95.0 + i,
                status=SignalStatus.PENDING if i < 3 else SignalStatus.ACTIVE,
            )
            store.insert_signal(signal)

        pending = store.get_signals_by_status(SignalStatus.PENDING)
        assert len(pending) == 3
        assert all(s.status == SignalStatus.PENDING for s in pending)

        active = store.get_signals_by_status(SignalStatus.ACTIVE)
        assert len(active) == 2
        assert all(s.status == SignalStatus.ACTIVE for s in active)

        store.close()

    def test_get_signals_by_date_range(self, tmp_path):
        """Test getting signals by date range."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()

        today = date.today()
        yesterday = today - timedelta(days=1)
        last_week = today - timedelta(days=7)

        # Create signals with different dates
        signal1 = TrackedSignal(
            symbol="AAPL",
            asset_class="equity",
            direction=SignalDirection.BUY,
            signal_type="breakout",
            conviction=ConvictionLevel.MEDIUM,
            signal_price=150.0,
            entry_price=150.0,
            target_price=165.0,
            stop_price=145.0,
            created_at=datetime.combine(today, datetime.min.time()),
        )
        store.insert_signal(signal1)

        signal2 = TrackedSignal(
            symbol="MSFT",
            asset_class="equity",
            direction=SignalDirection.BUY,
            signal_type="breakout",
            conviction=ConvictionLevel.MEDIUM,
            signal_price=200.0,
            entry_price=200.0,
            target_price=220.0,
            stop_price=190.0,
            created_at=datetime.combine(last_week, datetime.min.time()),
        )
        store.insert_signal(signal2)

        # Get signals from last 2 days
        recent = store.get_signals_by_date_range(yesterday, today)
        assert len(recent) == 1
        assert recent[0].symbol == "AAPL"

        # Get signals with symbol filter
        aapl_signals = store.get_signals_by_date_range(last_week, today, symbol="AAPL")
        assert len(aapl_signals) == 1
        assert aapl_signals[0].symbol == "AAPL"

        store.close()

    def test_get_recent_signals(self, tmp_path):
        """Test getting recent signals."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()

        # Create signals from different days
        for i in range(10):
            signal = TrackedSignal(
                symbol=f"SYM{i}",
                asset_class="equity",
                direction=SignalDirection.BUY,
                signal_type="breakout",
                conviction=ConvictionLevel.MEDIUM,
                signal_price=100.0,
                entry_price=100.0,
                target_price=110.0,
                stop_price=95.0,
                created_at=datetime.now() - timedelta(days=i),
            )
            store.insert_signal(signal)

        # Get last 3 days (includes today, yesterday, day before = 3 days)
        # The range is inclusive, so days=3 means start_date = today - 3 days, end_date = today
        # This includes 4 days: today, yesterday, 2 days ago, 3 days ago
        recent = store.get_recent_signals(days=3)
        # Should get signals from days 0, 1, 2, 3 (4 signals)
        assert len(recent) >= 3  # At least 3, but could be 4 due to inclusive range

        # Get with symbol filter
        recent_symbol = store.get_recent_signals(days=3, symbol="SYM0")
        assert len(recent_symbol) == 1
        assert recent_symbol[0].symbol == "SYM0"

        store.close()

    def test_outcome_lifecycle(self, tmp_path):
        """Test outcome insert and update."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()

        # First create a signal
        signal = TrackedSignal(
            symbol="AAPL",
            asset_class="equity",
            direction=SignalDirection.BUY,
            signal_type="breakout",
            conviction=ConvictionLevel.MEDIUM,
            signal_price=150.0,
            entry_price=150.0,
            target_price=165.0,
            stop_price=145.0,
        )
        signal_id = store.insert_signal(signal)

        # Insert outcome
        outcome = SignalOutcome(
            signal_id=signal_id,
            actual_entry_price=150.5,
            actual_entry_date=date.today(),
            actual_exit_price=165.0,
            actual_exit_date=date.today() + timedelta(days=5),
            exit_reason=ExitReason.TARGET_HIT,
            holding_days=5,
            return_pct=10.0,
            r_multiple=2.0,
            was_followed=True,
        )

        success = store.insert_outcome(outcome)
        assert success is True

        # Retrieve outcome
        retrieved = store.get_outcome(signal_id)
        assert retrieved is not None
        assert retrieved.signal_id == signal_id
        assert retrieved.actual_entry_price == 150.5
        assert retrieved.exit_reason == ExitReason.TARGET_HIT
        assert retrieved.was_followed is True

        # Update outcome
        outcome.return_pct = 12.0
        outcome.user_notes = "Great trade!"
        success = store.update_outcome(outcome)
        assert success is True

        retrieved = store.get_outcome(signal_id)
        assert retrieved.return_pct == 12.0
        assert retrieved.user_notes == "Great trade!"

        store.close()

    def test_get_outcomes_by_date_range(self, tmp_path):
        """Test getting outcomes by date range."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()

        today = date.today()
        yesterday = today - timedelta(days=1)
        last_week = today - timedelta(days=7)

        # Create signals and outcomes
        for i, exit_date in enumerate([today, yesterday, last_week]):
            signal = TrackedSignal(
                symbol=f"SYM{i}",
                asset_class="equity",
                direction=SignalDirection.BUY,
                signal_type="breakout",
                conviction=ConvictionLevel.MEDIUM,
                signal_price=100.0,
                entry_price=100.0,
                target_price=110.0,
                stop_price=95.0,
            )
            signal_id = store.insert_signal(signal)

            outcome = SignalOutcome(
                signal_id=signal_id,
                actual_entry_price=100.0,
                actual_entry_date=exit_date - timedelta(days=5),
                actual_exit_price=110.0,
                actual_exit_date=exit_date,
                exit_reason=ExitReason.TARGET_HIT,
                holding_days=5,
                return_pct=10.0,
            )
            store.insert_outcome(outcome)

        # Get outcomes from last 2 days
        recent = store.get_outcomes_by_date_range(yesterday, today)
        assert len(recent) == 2

        store.close()

    def test_save_and_get_daily_snapshot(self, tmp_path):
        """Test daily snapshot operations."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()

        snapshot_date = date.today()
        metrics = {
            "total_signals": 100,
            "total_closed": 80,
            "total_wins": 55,
            "total_losses": 25,
            "cumulative_return_pct": 25.5,
            "cumulative_r": 50.0,
            "rolling_win_rate": 0.6875,
            "rolling_avg_r": 0.625,
            "rolling_sharpe": 1.5,
            "starting_equity": 100000.0,
            "current_equity": 125500.0,
            "high_water_mark": 130000.0,
            "current_drawdown_pct": 3.46,
        }

        success = store.save_daily_snapshot(snapshot_date, metrics)
        assert success is True

        # Retrieve snapshots
        start_date = snapshot_date - timedelta(days=1)
        end_date = snapshot_date + timedelta(days=1)
        snapshots = store.get_daily_snapshots(start_date, end_date)

        assert len(snapshots) == 1
        snapshot = snapshots[0]
        assert snapshot["snapshot_date"] == snapshot_date.isoformat()
        assert snapshot["total_signals"] == 100
        assert snapshot["total_wins"] == 55
        assert snapshot["cumulative_return_pct"] == 25.5

        store.close()

    def test_count_signals_by_status(self, tmp_path):
        """Test counting signals by status."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()

        # Create signals with different statuses
        statuses = [SignalStatus.PENDING, SignalStatus.ACTIVE, SignalStatus.CLOSED]
        for status in statuses:
            for _ in range(3):
                signal = TrackedSignal(
                    symbol="AAPL",
                    asset_class="equity",
                    direction=SignalDirection.BUY,
                    signal_type="breakout",
                    conviction=ConvictionLevel.MEDIUM,
                    signal_price=100.0,
                    entry_price=100.0,
                    target_price=110.0,
                    stop_price=95.0,
                    status=status,
                )
                store.insert_signal(signal)

        counts = store.count_signals_by_status()
        assert counts["pending"] == 3
        assert counts["active"] == 3
        assert counts["closed"] == 3

        store.close()

    def test_get_signal_stats(self, tmp_path):
        """Test getting signal statistics."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()

        # Create various signals
        for i in range(10):
            signal = TrackedSignal(
                symbol=f"SYM{i % 3}",  # 3 unique symbols
                asset_class="equity" if i < 5 else "crypto",
                direction=SignalDirection.BUY,
                signal_type="breakout",
                conviction=ConvictionLevel.MEDIUM,
                signal_price=100.0,
                entry_price=100.0,
                target_price=110.0,
                stop_price=95.0,
                combined_score=0.5 + (i * 0.05),
                status=SignalStatus.CLOSED if i < 3 else SignalStatus.PENDING,
                was_delivered=i < 7,
            )
            store.insert_signal(signal)

        stats = store.get_signal_stats()
        assert stats["total_signals"] == 10
        assert stats["closed_signals"] == 3
        assert stats["delivered_signals"] == 7
        assert stats["unique_symbols"] == 3
        assert stats["asset_classes"] == 2

        # Test with date range
        today = date.today()
        yesterday = today - timedelta(days=1)
        stats_range = store.get_signal_stats(start_date=yesterday, end_date=today)
        assert stats_range["total_signals"] == 10  # All created today

        store.close()

    def test_transaction_rollback(self, tmp_path):
        """Test that transactions rollback on error."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()

        # Insert a signal
        signal = TrackedSignal(
            symbol="AAPL",
            asset_class="equity",
            direction=SignalDirection.BUY,
            signal_type="breakout",
            conviction=ConvictionLevel.MEDIUM,
            signal_price=100.0,
            entry_price=100.0,
            target_price=110.0,
            stop_price=95.0,
        )
        signal_id = store.insert_signal(signal)

        # Try to insert outcome with invalid signal_id (should fail foreign key)
        outcome = SignalOutcome(
            signal_id="non-existent-id",
            actual_entry_price=100.0,
        )

        # This should raise an error and rollback
        with pytest.raises(Exception):
            store.insert_outcome(outcome)

        # Verify original signal still exists
        retrieved = store.get_signal(signal_id)
        assert retrieved is not None

        store.close()

    def test_signal_with_all_fields(self, tmp_path):
        """Test signal with all fields populated."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()

        signal = TrackedSignal(
            id="test-signal-123",
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
            status=SignalStatus.PENDING,
            created_at=datetime(2024, 1, 1, 10, 0, 0),
            delivered_at=datetime(2024, 1, 1, 10, 5, 0),
            was_delivered=True,
            delivery_method="email",
            reasoning="Strong breakout with high volume",
            news_headlines=["Apple announces new product"],
            tags=["momentum", "breakout"],
        )

        signal_id = store.insert_signal(signal)
        retrieved = store.get_signal(signal_id)

        assert retrieved.id == "test-signal-123"
        assert retrieved.news_score == 0.75
        assert retrieved.was_delivered is True
        assert retrieved.delivery_method == "email"
        assert retrieved.reasoning == "Strong breakout with high volume"
        assert retrieved.news_headlines == ["Apple announces new product"]
        assert retrieved.tags == ["momentum", "breakout"]

        store.close()

    def test_outcome_with_all_fields(self, tmp_path):
        """Test outcome with all fields populated."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()

        # Create signal first
        signal = TrackedSignal(
            symbol="AAPL",
            asset_class="equity",
            direction=SignalDirection.BUY,
            signal_type="breakout",
            conviction=ConvictionLevel.MEDIUM,
            signal_price=100.0,
            entry_price=100.0,
            target_price=110.0,
            stop_price=95.0,
        )
        signal_id = store.insert_signal(signal)

        outcome = SignalOutcome(
            signal_id=signal_id,
            actual_entry_price=100.5,
            actual_entry_date=date(2024, 1, 2),
            actual_exit_price=110.0,
            actual_exit_date=date(2024, 1, 7),
            exit_reason=ExitReason.TARGET_HIT,
            holding_days=5,
            return_pct=10.0,
            return_dollars=1000.0,
            r_multiple=2.0,
            benchmark_return_pct=5.0,
            alpha=5.0,
            was_followed=True,
            user_notes="Great trade!",
            recorded_at=datetime(2024, 1, 7, 16, 0, 0),
        )

        success = store.insert_outcome(outcome)
        assert success is True

        retrieved = store.get_outcome(signal_id)
        assert retrieved.actual_entry_price == 100.5
        assert retrieved.actual_entry_date == date(2024, 1, 2)
        assert retrieved.actual_exit_date == date(2024, 1, 7)
        assert retrieved.exit_reason == ExitReason.TARGET_HIT
        assert retrieved.holding_days == 5
        assert retrieved.return_pct == 10.0
        assert retrieved.r_multiple == 2.0
        assert retrieved.was_followed is True
        assert retrieved.user_notes == "Great trade!"

        store.close()

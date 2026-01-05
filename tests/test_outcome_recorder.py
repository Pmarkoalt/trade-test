"""Unit tests for OutcomeRecorder."""

from datetime import date, timedelta

from trading_system.tracking.models import ConvictionLevel, ExitReason, SignalDirection, SignalStatus
from trading_system.tracking.outcome_recorder import AutoOutcomeRecorder, OutcomeRecorder
from trading_system.tracking.storage.sqlite_store import SQLiteTrackingStore


class TestOutcomeRecorder:
    """Tests for OutcomeRecorder."""

    def test_calculate_returns_buy_winner(self, tmp_path):
        """Test return calculation for winning long trade."""
        # Entry 100, Stop 95, Target 110, Exit 108
        # Return = (108-100)/100 = 8%
        # Risk = 100-95 = 5
        # Reward = 108-100 = 8
        # R = 8/5 = 1.6
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        recorder = OutcomeRecorder(store)

        # Create signal
        from trading_system.tracking.signal_tracker import SignalTracker

        tracker = SignalTracker(store)
        signal_id = tracker.record_signal(
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

        # Record outcome
        success = recorder.record_outcome(
            signal_id=signal_id,
            entry_price=100.0,
            exit_price=108.0,
            exit_reason=ExitReason.TARGET_HIT,
        )

        assert success is True

        outcome = recorder.get_outcome(signal_id)
        assert outcome is not None
        assert abs(outcome.return_pct - 0.08) < 0.0001  # 8%
        assert abs(outcome.r_multiple - 1.6) < 0.0001  # 8/5 = 1.6

        # Check signal status updated
        signal = tracker.get_signal(signal_id)
        assert signal.status == SignalStatus.CLOSED

        store.close()

    def test_calculate_returns_buy_loser(self, tmp_path):
        """Test return calculation for losing long trade."""
        # Entry 100, Stop 95, Exit 94
        # Return = (94-100)/100 = -6%
        # Risk = 5, Reward = -6
        # R = -6/5 = -1.2
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        recorder = OutcomeRecorder(store)

        from trading_system.tracking.signal_tracker import SignalTracker

        tracker = SignalTracker(store)
        signal_id = tracker.record_signal(
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

        success = recorder.record_outcome(
            signal_id=signal_id,
            entry_price=100.0,
            exit_price=94.0,
            exit_reason=ExitReason.STOP_HIT,
        )

        assert success is True

        outcome = recorder.get_outcome(signal_id)
        assert abs(outcome.return_pct - (-0.06)) < 0.0001  # -6%
        assert abs(outcome.r_multiple - (-1.2)) < 0.0001  # -6/5 = -1.2

        store.close()

    def test_calculate_returns_sell_winner(self, tmp_path):
        """Test return calculation for winning short trade."""
        # Entry 100, Stop 105, Target 90, Exit 92
        # Return = (100-92)/100 = 8%
        # Risk = 105-100 = 5
        # Reward = 100-92 = 8
        # R = 8/5 = 1.6
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        recorder = OutcomeRecorder(store)

        from trading_system.tracking.signal_tracker import SignalTracker

        tracker = SignalTracker(store)
        signal_id = tracker.record_signal(
            symbol="AAPL",
            asset_class="equity",
            direction=SignalDirection.SELL,
            signal_type="breakout",
            conviction=ConvictionLevel.MEDIUM,
            signal_price=100.0,
            entry_price=100.0,
            target_price=90.0,
            stop_price=105.0,  # Stop above for short
        )

        success = recorder.record_outcome(
            signal_id=signal_id,
            entry_price=100.0,
            exit_price=92.0,
            exit_reason=ExitReason.TARGET_HIT,
        )

        assert success is True

        outcome = recorder.get_outcome(signal_id)
        assert abs(outcome.return_pct - 0.08) < 0.0001  # 8%
        assert abs(outcome.r_multiple - 1.6) < 0.0001  # 8/5 = 1.6

        store.close()

    def test_calculate_returns_sell_loser(self, tmp_path):
        """Test return calculation for losing short trade."""
        # Entry 100, Stop 105, Exit 106
        # Return = (100-106)/100 = -6%
        # Risk = 5, Reward = -6
        # R = -6/5 = -1.2
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        recorder = OutcomeRecorder(store)

        from trading_system.tracking.signal_tracker import SignalTracker

        tracker = SignalTracker(store)
        signal_id = tracker.record_signal(
            symbol="AAPL",
            asset_class="equity",
            direction=SignalDirection.SELL,
            signal_type="breakout",
            conviction=ConvictionLevel.MEDIUM,
            signal_price=100.0,
            entry_price=100.0,
            target_price=90.0,
            stop_price=105.0,
        )

        success = recorder.record_outcome(
            signal_id=signal_id,
            entry_price=100.0,
            exit_price=106.0,
            exit_reason=ExitReason.STOP_HIT,
        )

        assert success is True

        outcome = recorder.get_outcome(signal_id)
        assert abs(outcome.return_pct - (-0.06)) < 0.0001  # -6%
        assert abs(outcome.r_multiple - (-1.2)) < 0.0001  # -6/5 = -1.2

        store.close()

    def test_record_outcome_with_dates(self, tmp_path):
        """Test recording outcome with entry and exit dates."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        recorder = OutcomeRecorder(store)

        from trading_system.tracking.signal_tracker import SignalTracker

        tracker = SignalTracker(store)
        signal_id = tracker.record_signal(
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

        entry_date = date.today() - timedelta(days=5)
        exit_date = date.today()

        success = recorder.record_outcome(
            signal_id=signal_id,
            entry_price=100.0,
            exit_price=108.0,
            entry_date=entry_date,
            exit_date=exit_date,
            exit_reason=ExitReason.TARGET_HIT,
        )

        assert success is True

        outcome = recorder.get_outcome(signal_id)
        assert outcome.actual_entry_date == entry_date
        assert outcome.actual_exit_date == exit_date
        assert outcome.holding_days == 5

        store.close()

    def test_record_outcome_with_benchmark(self, tmp_path):
        """Test recording outcome with benchmark return."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        recorder = OutcomeRecorder(store)

        from trading_system.tracking.signal_tracker import SignalTracker

        tracker = SignalTracker(store)
        signal_id = tracker.record_signal(
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

        success = recorder.record_outcome(
            signal_id=signal_id,
            entry_price=100.0,
            exit_price=108.0,
            exit_reason=ExitReason.TARGET_HIT,
            benchmark_return_pct=0.05,  # 5% benchmark return
        )

        assert success is True

        outcome = recorder.get_outcome(signal_id)
        assert abs(outcome.return_pct - 0.08) < 0.0001  # 8%
        assert abs(outcome.benchmark_return_pct - 0.05) < 0.0001  # 5%
        assert abs(outcome.alpha - 0.03) < 0.0001  # 8% - 5% = 3%

        store.close()

    def test_record_quick_outcome(self, tmp_path):
        """Test quick outcome recording using signal entry price."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        recorder = OutcomeRecorder(store)

        from trading_system.tracking.signal_tracker import SignalTracker

        tracker = SignalTracker(store)
        signal_id = tracker.record_signal(
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

        # Mark as active first
        tracker.mark_entry_filled(signal_id)

        success = recorder.record_quick_outcome(
            signal_id=signal_id,
            exit_price=108.0,
            exit_reason=ExitReason.TARGET_HIT,
        )

        assert success is True

        outcome = recorder.get_outcome(signal_id)
        assert outcome.actual_entry_price == 100.0  # Uses signal entry price
        assert outcome.actual_exit_price == 108.0

        store.close()

    def test_record_missed_signal(self, tmp_path):
        """Test recording missed signal."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        recorder = OutcomeRecorder(store)

        from trading_system.tracking.signal_tracker import SignalTracker

        tracker = SignalTracker(store)
        signal_id = tracker.record_signal(
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

        success = recorder.record_missed_signal(
            signal_id=signal_id,
            user_notes="Didn't have enough capital",
        )

        assert success is True

        outcome = recorder.get_outcome(signal_id)
        assert outcome.was_followed is False
        assert outcome.user_notes == "Didn't have enough capital"

        # Signal should be marked as expired
        signal = tracker.get_signal(signal_id)
        assert signal.status == SignalStatus.EXPIRED

        store.close()

    def test_update_benchmark_return(self, tmp_path):
        """Test updating benchmark return after recording."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        recorder = OutcomeRecorder(store)

        from trading_system.tracking.signal_tracker import SignalTracker

        tracker = SignalTracker(store)
        signal_id = tracker.record_signal(
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

        # Record outcome without benchmark
        recorder.record_outcome(
            signal_id=signal_id,
            entry_price=100.0,
            exit_price=108.0,
            exit_reason=ExitReason.TARGET_HIT,
        )

        # Update benchmark later
        success = recorder.update_benchmark_return(signal_id, benchmark_return_pct=0.05)

        assert success is True

        outcome = recorder.get_outcome(signal_id)
        assert abs(outcome.benchmark_return_pct - 0.05) < 0.0001
        assert abs(outcome.alpha - 0.03) < 0.0001  # 8% - 5% = 3%

        store.close()

    def test_record_outcome_nonexistent_signal(self, tmp_path):
        """Test recording outcome for non-existent signal."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        recorder = OutcomeRecorder(store)

        success = recorder.record_outcome(
            signal_id="non-existent",
            entry_price=100.0,
            exit_price=108.0,
        )

        assert success is False

        store.close()


class TestAutoOutcomeRecorder:
    """Tests for AutoOutcomeRecorder."""

    def test_auto_outcome_target_hit(self, tmp_path):
        """Test auto-recording when target is hit."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()

        from trading_system.tracking.signal_tracker import SignalTracker

        tracker = SignalTracker(store)
        recorder = OutcomeRecorder(store)
        auto_recorder = AutoOutcomeRecorder(store, recorder)

        # Create and activate signal
        signal_id = tracker.record_signal(
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
        tracker.mark_entry_filled(signal_id)

        # Check with price at target
        price_data = {"AAPL": 110.0}
        closed = auto_recorder.check_and_record_outcomes(price_data)

        assert len(closed) == 1
        assert closed[0] == signal_id

        outcome = recorder.get_outcome(signal_id)
        assert outcome.exit_reason == ExitReason.TARGET_HIT
        assert outcome.actual_exit_price == 110.0

        store.close()

    def test_auto_outcome_stop_hit(self, tmp_path):
        """Test auto-recording when stop is hit."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()

        from trading_system.tracking.signal_tracker import SignalTracker

        tracker = SignalTracker(store)
        recorder = OutcomeRecorder(store)
        auto_recorder = AutoOutcomeRecorder(store, recorder)

        signal_id = tracker.record_signal(
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
        tracker.mark_entry_filled(signal_id)

        # Check with price at stop
        price_data = {"AAPL": 95.0}
        closed = auto_recorder.check_and_record_outcomes(price_data)

        assert len(closed) == 1
        assert closed[0] == signal_id

        outcome = recorder.get_outcome(signal_id)
        assert outcome.exit_reason == ExitReason.STOP_HIT
        assert outcome.actual_exit_price == 95.0

        store.close()

    def test_auto_outcome_no_exit(self, tmp_path):
        """Test auto-recorder when no exit condition is met."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()

        from trading_system.tracking.signal_tracker import SignalTracker

        tracker = SignalTracker(store)
        recorder = OutcomeRecorder(store)
        auto_recorder = AutoOutcomeRecorder(store, recorder)

        signal_id = tracker.record_signal(
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
        tracker.mark_entry_filled(signal_id)

        # Check with price in middle
        price_data = {"AAPL": 102.0}
        closed = auto_recorder.check_and_record_outcomes(price_data)

        assert len(closed) == 0

        outcome = recorder.get_outcome(signal_id)
        assert outcome is None

        store.close()

    def test_auto_outcome_short_target_hit(self, tmp_path):
        """Test auto-recording for short position target hit."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()

        from trading_system.tracking.signal_tracker import SignalTracker

        tracker = SignalTracker(store)
        recorder = OutcomeRecorder(store)
        auto_recorder = AutoOutcomeRecorder(store, recorder)

        signal_id = tracker.record_signal(
            symbol="AAPL",
            asset_class="equity",
            direction=SignalDirection.SELL,
            signal_type="breakout",
            conviction=ConvictionLevel.MEDIUM,
            signal_price=100.0,
            entry_price=100.0,
            target_price=90.0,  # Target below for short
            stop_price=105.0,  # Stop above for short
        )
        tracker.mark_entry_filled(signal_id)

        # Check with price at target (below entry)
        price_data = {"AAPL": 90.0}
        closed = auto_recorder.check_and_record_outcomes(price_data)

        assert len(closed) == 1
        assert closed[0] == signal_id

        outcome = recorder.get_outcome(signal_id)
        assert outcome.exit_reason == ExitReason.TARGET_HIT

        store.close()

    def test_auto_outcome_multiple_signals(self, tmp_path):
        """Test auto-recorder with multiple active signals."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()

        from trading_system.tracking.signal_tracker import SignalTracker

        tracker = SignalTracker(store)
        recorder = OutcomeRecorder(store)
        auto_recorder = AutoOutcomeRecorder(store, recorder)

        # Create multiple signals
        signal_ids = []
        for symbol in ["AAPL", "MSFT", "GOOGL"]:
            signal_id = tracker.record_signal(
                symbol=symbol,
                asset_class="equity",
                direction=SignalDirection.BUY,
                signal_type="breakout",
                conviction=ConvictionLevel.MEDIUM,
                signal_price=100.0,
                entry_price=100.0,
                target_price=110.0,
                stop_price=95.0,
            )
            tracker.mark_entry_filled(signal_id)
            signal_ids.append(signal_id)

        # Check with some hitting target, some not
        price_data = {
            "AAPL": 110.0,  # Target hit
            "MSFT": 95.0,  # Stop hit
            "GOOGL": 102.0,  # No exit
        }
        closed = auto_recorder.check_and_record_outcomes(price_data)

        assert len(closed) == 2
        assert signal_ids[0] in closed  # AAPL target hit
        assert signal_ids[1] in closed  # MSFT stop hit
        assert signal_ids[2] not in closed  # GOOGL no exit

        store.close()

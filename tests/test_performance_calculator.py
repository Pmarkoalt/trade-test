"""Unit tests for PerformanceCalculator."""

from datetime import date, datetime, timedelta

import pytest

from trading_system.tracking.models import ConvictionLevel, ExitReason, SignalDirection, SignalStatus
from trading_system.tracking.outcome_recorder import OutcomeRecorder
from trading_system.tracking.performance_calculator import PerformanceCalculator
from trading_system.tracking.signal_tracker import SignalTracker
from trading_system.tracking.storage.sqlite_store import SQLiteTrackingStore


class TestPerformanceCalculator:
    """Tests for PerformanceCalculator."""

    def test_win_rate_calculation(self, tmp_path):
        """Test win rate is calculated correctly."""
        # 7 wins, 3 losses = 70% win rate
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        tracker = SignalTracker(store)
        recorder = OutcomeRecorder(store)
        calculator = PerformanceCalculator(store)

        # Create 10 signals with outcomes
        for i in range(10):
            signal_id = tracker.record_signal(
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
            tracker.mark_delivered(signal_id)
            tracker.mark_entry_filled(signal_id)

            # 7 winners, 3 losers
            exit_price = 108.0 if i < 7 else 94.0
            exit_reason = ExitReason.TARGET_HIT if i < 7 else ExitReason.STOP_HIT

            recorder.record_outcome(
                signal_id=signal_id,
                entry_price=100.0,
                exit_price=exit_price,
                exit_reason=exit_reason,
                was_followed=True,
            )

        metrics = calculator.calculate_metrics()
        assert abs(metrics.win_rate - 0.70) < 0.01  # 70%
        assert metrics.signals_won == 7
        assert metrics.signals_lost == 3

        store.close()

    def test_expectancy_calculation(self, tmp_path):
        """Test expectancy formula."""
        # Win rate 60%, avg winner 2R, avg loser -1R
        # Expectancy = 0.6 * 2 - 0.4 * 1 = 0.8R
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        tracker = SignalTracker(store)
        recorder = OutcomeRecorder(store)
        calculator = PerformanceCalculator(store)

        # Create signals with specific R-multiples
        # 6 winners with 2R each, 4 losers with -1R each
        for i in range(10):
            signal_id = tracker.record_signal(
                symbol=f"SYM{i}",
                asset_class="equity",
                direction=SignalDirection.BUY,
                signal_type="breakout",
                conviction=ConvictionLevel.MEDIUM,
                signal_price=100.0,
                entry_price=100.0,
                target_price=110.0,
                stop_price=95.0,  # Risk = 5
            )
            tracker.mark_delivered(signal_id)
            tracker.mark_entry_filled(signal_id)

            if i < 6:
                # Winner: 2R = 10 profit (entry 100, exit 110)
                exit_price = 110.0
                exit_reason = ExitReason.TARGET_HIT
            else:
                # Loser: -1R = -5 loss (entry 100, exit 95)
                exit_price = 95.0
                exit_reason = ExitReason.STOP_HIT

            recorder.record_outcome(
                signal_id=signal_id,
                entry_price=100.0,
                exit_price=exit_price,
                exit_reason=exit_reason,
                was_followed=True,
            )

        metrics = calculator.calculate_metrics()
        # Win rate should be 60%
        assert abs(metrics.win_rate - 0.60) < 0.01
        # Avg winner R should be ~2.0
        assert abs(metrics.avg_winner_r - 2.0) < 0.1
        # Avg loser R should be ~-1.0
        assert abs(metrics.avg_loser_r - (-1.0)) < 0.1
        # Expectancy = 0.6 * 2 - 0.4 * 1 = 0.8R
        assert abs(metrics.expectancy_r - 0.8) < 0.1

        store.close()

    def test_sharpe_ratio(self, tmp_path):
        """Test Sharpe ratio calculation."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        tracker = SignalTracker(store)
        recorder = OutcomeRecorder(store)
        calculator = PerformanceCalculator(store)

        # Create signals with varying returns
        returns = [0.10, 0.05, -0.03, 0.08, -0.02, 0.12, 0.04]
        for i, ret in enumerate(returns):
            signal_id = tracker.record_signal(
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
            tracker.mark_delivered(signal_id)
            tracker.mark_entry_filled(signal_id)

            exit_price = 100.0 * (1 + ret)
            recorder.record_outcome(
                signal_id=signal_id,
                entry_price=100.0,
                exit_price=exit_price,
                exit_reason=ExitReason.MANUAL,
                was_followed=True,
            )

        metrics = calculator.calculate_metrics()
        # Sharpe should be positive for positive average returns
        assert metrics.sharpe_ratio > 0

        store.close()

    def test_max_drawdown(self, tmp_path):
        """Test max drawdown from returns."""
        # returns = [0.10, -0.05, -0.08, 0.12, -0.03]
        # Peak at 1.10, drops to 1.10 * 0.95 * 0.92 = 0.961
        # DD = (0.961 - 1.10) / 1.10 = -12.6%
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        tracker = SignalTracker(store)
        recorder = OutcomeRecorder(store)
        calculator = PerformanceCalculator(store)

        returns = [0.10, -0.05, -0.08, 0.12, -0.03]
        for i, ret in enumerate(returns):
            signal_id = tracker.record_signal(
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
            tracker.mark_delivered(signal_id)
            tracker.mark_entry_filled(signal_id)

            exit_price = 100.0 * (1 + ret)
            recorder.record_outcome(
                signal_id=signal_id,
                entry_price=100.0,
                exit_price=exit_price,
                exit_reason=ExitReason.MANUAL,
                was_followed=True,
            )

        metrics = calculator.calculate_metrics()
        # Max drawdown should be negative (around -12.6%)
        assert metrics.max_drawdown_pct < 0
        assert abs(metrics.max_drawdown_pct - (-0.126)) < 0.02

        store.close()

    def test_sortino_ratio(self, tmp_path):
        """Test Sortino ratio calculation."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        tracker = SignalTracker(store)
        recorder = OutcomeRecorder(store)
        calculator = PerformanceCalculator(store)

        returns = [0.10, 0.05, -0.03, 0.08, -0.02]
        for i, ret in enumerate(returns):
            signal_id = tracker.record_signal(
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
            tracker.mark_delivered(signal_id)
            tracker.mark_entry_filled(signal_id)

            exit_price = 100.0 * (1 + ret)
            recorder.record_outcome(
                signal_id=signal_id,
                entry_price=100.0,
                exit_price=exit_price,
                exit_reason=ExitReason.MANUAL,
                was_followed=True,
            )

        metrics = calculator.calculate_metrics()
        # Sortino should be positive for positive average returns
        assert metrics.sortino_ratio > 0

        store.close()

    def test_metrics_by_category(self, tmp_path):
        """Test metrics grouped by category."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        tracker = SignalTracker(store)
        recorder = OutcomeRecorder(store)
        calculator = PerformanceCalculator(store)

        # Create signals for different asset classes
        asset_classes = ["equity", "equity", "crypto", "crypto"]
        for i, asset_class in enumerate(asset_classes):
            signal_id = tracker.record_signal(
                symbol=f"SYM{i}",
                asset_class=asset_class,
                direction=SignalDirection.BUY,
                signal_type="breakout",
                conviction=ConvictionLevel.MEDIUM,
                signal_price=100.0,
                entry_price=100.0,
                target_price=110.0,
                stop_price=95.0,
            )
            tracker.mark_delivered(signal_id)
            tracker.mark_entry_filled(signal_id)

            # Winners for equity, losers for crypto
            exit_price = 108.0 if asset_class == "equity" else 94.0
            exit_reason = ExitReason.TARGET_HIT if asset_class == "equity" else ExitReason.STOP_HIT

            recorder.record_outcome(
                signal_id=signal_id,
                entry_price=100.0,
                exit_price=exit_price,
                exit_reason=exit_reason,
                was_followed=True,
            )

        metrics = calculator.calculate_metrics()
        assert "equity" in metrics.metrics_by_asset_class
        assert "crypto" in metrics.metrics_by_asset_class

        equity_metrics = metrics.metrics_by_asset_class["equity"]
        assert equity_metrics["win_rate"] == 1.0  # Both equity trades won

        crypto_metrics = metrics.metrics_by_asset_class["crypto"]
        assert crypto_metrics["win_rate"] == 0.0  # Both crypto trades lost

        store.close()

    def test_equity_curve(self, tmp_path):
        """Test equity curve generation."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        tracker = SignalTracker(store)
        recorder = OutcomeRecorder(store)
        calculator = PerformanceCalculator(store)

        # Create outcomes with different dates
        base_date = date.today() - timedelta(days=10)
        returns = [0.10, -0.05, 0.08, -0.02, 0.12]

        for i, ret in enumerate(returns):
            signal_id = tracker.record_signal(
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
            tracker.mark_delivered(signal_id)
            tracker.mark_entry_filled(signal_id)

            entry_date = base_date + timedelta(days=i * 2)
            exit_date = entry_date + timedelta(days=1)
            exit_price = 100.0 * (1 + ret)

            recorder.record_outcome(
                signal_id=signal_id,
                entry_price=100.0,
                exit_price=exit_price,
                entry_date=entry_date,
                exit_date=exit_date,
                exit_reason=ExitReason.MANUAL,
                was_followed=True,
            )

        curve = calculator.get_equity_curve(starting_equity=100000.0)
        assert len(curve) == len(returns)
        assert all("date" in point for point in curve)
        assert all("equity" in point for point in curve)
        assert all("high_water_mark" in point for point in curve)
        assert all("drawdown_pct" in point for point in curve)

        # Equity should change based on returns
        assert curve[-1]["equity"] != 100000.0

        store.close()

    def test_calculate_rolling_metrics(self, tmp_path):
        """Test rolling metrics calculation."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        tracker = SignalTracker(store)
        recorder = OutcomeRecorder(store)
        calculator = PerformanceCalculator(store)

        # Create signals from different dates
        for i in range(5):
            signal_id = tracker.record_signal(
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
            tracker.mark_delivered(signal_id)
            tracker.mark_entry_filled(signal_id)

            exit_price = 108.0 if i % 2 == 0 else 94.0
            exit_reason = ExitReason.TARGET_HIT if i % 2 == 0 else ExitReason.STOP_HIT

            recorder.record_outcome(
                signal_id=signal_id,
                entry_price=100.0,
                exit_price=exit_price,
                exit_reason=exit_reason,
                was_followed=True,
            )

        rolling = calculator.calculate_rolling_metrics(window_days=30)
        assert "window_days" in rolling
        assert "win_rate" in rolling
        assert "avg_r" in rolling
        assert "expectancy_r" in rolling
        assert rolling["window_days"] == 30

        store.close()

    def test_filter_by_symbol(self, tmp_path):
        """Test filtering metrics by symbol."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        tracker = SignalTracker(store)
        recorder = OutcomeRecorder(store)
        calculator = PerformanceCalculator(store)

        # Create signals for different symbols
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
            tracker.mark_delivered(signal_id)
            tracker.mark_entry_filled(signal_id)

            recorder.record_outcome(
                signal_id=signal_id,
                entry_price=100.0,
                exit_price=108.0,
                exit_reason=ExitReason.TARGET_HIT,
                was_followed=True,
            )

        # Get metrics for AAPL only
        metrics = calculator.calculate_metrics(symbol="AAPL")
        assert metrics.total_signals == 1

        # Get all metrics
        all_metrics = calculator.calculate_metrics()
        assert all_metrics.total_signals == 3

        store.close()

    def test_filter_by_asset_class(self, tmp_path):
        """Test filtering metrics by asset class."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        tracker = SignalTracker(store)
        recorder = OutcomeRecorder(store)
        calculator = PerformanceCalculator(store)

        # Create signals for different asset classes
        for asset_class in ["equity", "equity", "crypto"]:
            signal_id = tracker.record_signal(
                symbol="SYM",
                asset_class=asset_class,
                direction=SignalDirection.BUY,
                signal_type="breakout",
                conviction=ConvictionLevel.MEDIUM,
                signal_price=100.0,
                entry_price=100.0,
                target_price=110.0,
                stop_price=95.0,
            )
            tracker.mark_delivered(signal_id)
            tracker.mark_entry_filled(signal_id)

            recorder.record_outcome(
                signal_id=signal_id,
                entry_price=100.0,
                exit_price=108.0,
                exit_reason=ExitReason.TARGET_HIT,
                was_followed=True,
            )

        # Get metrics for equity only
        metrics = calculator.calculate_metrics(asset_class="equity")
        assert metrics.total_signals == 2

        # Get metrics for crypto only
        crypto_metrics = calculator.calculate_metrics(asset_class="crypto")
        assert crypto_metrics.total_signals == 1

        store.close()

    def test_empty_metrics(self, tmp_path):
        """Test metrics calculation with no signals."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        calculator = PerformanceCalculator(store)

        metrics = calculator.calculate_metrics()
        assert metrics.total_signals == 0
        assert metrics.win_rate == 0.0
        assert metrics.expectancy_r == 0.0

        store.close()

    def test_follow_rate_calculation(self, tmp_path):
        """Test follow rate calculation."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        tracker = SignalTracker(store)
        recorder = OutcomeRecorder(store)
        calculator = PerformanceCalculator(store)

        # Create 10 signals, deliver 8, follow 6
        for i in range(10):
            signal_id = tracker.record_signal(
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

            # Deliver 8 signals
            if i < 8:
                tracker.mark_delivered(signal_id)

            # Follow 6 signals
            if i < 6:
                tracker.mark_entry_filled(signal_id)
                recorder.record_outcome(
                    signal_id=signal_id,
                    entry_price=100.0,
                    exit_price=108.0,
                    exit_reason=ExitReason.TARGET_HIT,
                    was_followed=True,
                )

        metrics = calculator.calculate_metrics()
        # Follow rate = 6 followed / 8 delivered = 0.75
        assert abs(metrics.follow_rate - 0.75) < 0.01

        store.close()

    def test_benchmark_comparison(self, tmp_path):
        """Test benchmark return and alpha calculation."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        tracker = SignalTracker(store)
        recorder = OutcomeRecorder(store)
        calculator = PerformanceCalculator(store)

        # Create signal with benchmark
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
        tracker.mark_delivered(signal_id)
        tracker.mark_entry_filled(signal_id)

        recorder.record_outcome(
            signal_id=signal_id,
            entry_price=100.0,
            exit_price=108.0,  # 8% return
            exit_reason=ExitReason.TARGET_HIT,
            was_followed=True,
            benchmark_return_pct=0.05,  # 5% benchmark
        )

        metrics = calculator.calculate_metrics()
        # Alpha = 8% - 5% = 3%
        assert abs(metrics.alpha - 0.03) < 0.001
        assert abs(metrics.benchmark_return_pct - 0.05) < 0.001

        store.close()

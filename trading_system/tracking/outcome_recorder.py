"""Outcome recorder for tracking trade results."""

import logging
from datetime import date, datetime
from typing import List, Optional

from trading_system.tracking.models import (
    ExitReason,
    SignalOutcome,
    SignalStatus,
    TrackedSignal,
)
from trading_system.tracking.storage.base_store import BaseTrackingStore

logger = logging.getLogger(__name__)


class OutcomeRecorder:
    """
    Record outcomes for tracked signals.

    This class handles recording actual trade results including
    entry/exit prices, returns, and R-multiples.

    Example:
        recorder = OutcomeRecorder(store)

        # When position is closed
        recorder.record_outcome(
            signal_id=signal_id,
            entry_price=150.0,
            exit_price=165.0,
            exit_reason=ExitReason.TARGET_HIT,
        )
    """

    def __init__(self, store: BaseTrackingStore):
        """
        Initialize outcome recorder.

        Args:
            store: Storage backend for persisting outcomes.
        """
        self.store = store

    def record_outcome(
        self,
        signal_id: str,
        entry_price: float,
        exit_price: float,
        entry_date: Optional[date] = None,
        exit_date: Optional[date] = None,
        exit_reason: ExitReason = ExitReason.MANUAL,
        was_followed: bool = True,
        benchmark_return_pct: float = 0.0,
        user_notes: str = "",
    ) -> bool:
        """
        Record the outcome of a signal.

        Args:
            signal_id: ID of the signal.
            entry_price: Actual entry price.
            exit_price: Actual exit price.
            entry_date: Date position was entered.
            exit_date: Date position was exited.
            exit_reason: Why position was closed.
            was_followed: Whether user followed the recommendation.
            benchmark_return_pct: Benchmark return over same period.
            user_notes: Optional user notes.

        Returns:
            True if successful.
        """
        # Get the original signal for calculations
        signal = self.store.get_signal(signal_id)
        if signal is None:
            logger.error(f"Signal {signal_id} not found")
            return False

        # Calculate returns
        return_pct, r_multiple = self._calculate_returns(
            signal=signal,
            entry_price=entry_price,
            exit_price=exit_price,
        )

        # Calculate holding period
        entry_date = entry_date or date.today()
        exit_date = exit_date or date.today()
        holding_days = (exit_date - entry_date).days

        # Calculate alpha
        alpha = return_pct - benchmark_return_pct

        outcome = SignalOutcome(
            signal_id=signal_id,
            actual_entry_price=entry_price,
            actual_entry_date=entry_date,
            actual_exit_price=exit_price,
            actual_exit_date=exit_date,
            exit_reason=exit_reason,
            holding_days=holding_days,
            return_pct=return_pct,
            r_multiple=r_multiple,
            benchmark_return_pct=benchmark_return_pct,
            alpha=alpha,
            was_followed=was_followed,
            user_notes=user_notes,
            recorded_at=datetime.now(),
        )

        # Save outcome
        success = self.store.insert_outcome(outcome)

        if success:
            # Update signal status to closed
            self.store.update_signal_status(
                signal_id=signal_id,
                status=SignalStatus.CLOSED,
                timestamp=datetime.now(),
            )

            logger.info(
                f"Recorded outcome for {signal_id}: "
                f"return={return_pct:.2%}, R={r_multiple:.2f}, "
                f"reason={exit_reason.value}"
            )

        return success

    def record_quick_outcome(
        self,
        signal_id: str,
        exit_price: float,
        exit_reason: ExitReason,
    ) -> bool:
        """
        Record outcome using signal's entry price as actual entry.

        Convenience method when entry price matches recommendation.

        Args:
            signal_id: ID of the signal.
            exit_price: Actual exit price.
            exit_reason: Why position was closed.

        Returns:
            True if successful.
        """
        signal = self.store.get_signal(signal_id)
        if signal is None:
            return False

        return self.record_outcome(
            signal_id=signal_id,
            entry_price=signal.entry_price,
            exit_price=exit_price,
            exit_reason=exit_reason,
        )

    def record_missed_signal(
        self,
        signal_id: str,
        user_notes: str = "",
    ) -> bool:
        """
        Record that a signal was not followed.

        Args:
            signal_id: ID of the signal.
            user_notes: Why signal wasn't followed.

        Returns:
            True if successful.
        """
        signal = self.store.get_signal(signal_id)
        if signal is None:
            return False

        outcome = SignalOutcome(
            signal_id=signal_id,
            was_followed=False,
            user_notes=user_notes,
            recorded_at=datetime.now(),
        )

        success = self.store.insert_outcome(outcome)

        if success:
            self.store.update_signal_status(
                signal_id=signal_id,
                status=SignalStatus.EXPIRED,
            )

        return success

    def update_benchmark_return(
        self,
        signal_id: str,
        benchmark_return_pct: float,
    ) -> bool:
        """
        Update benchmark return for an existing outcome.

        Useful when benchmark data becomes available after recording.

        Args:
            signal_id: ID of the signal.
            benchmark_return_pct: Benchmark return over holding period.

        Returns:
            True if successful.
        """
        outcome = self.store.get_outcome(signal_id)
        if outcome is None:
            return False

        outcome.benchmark_return_pct = benchmark_return_pct
        outcome.alpha = outcome.return_pct - benchmark_return_pct

        return self.store.update_outcome(outcome)

    def get_outcome(self, signal_id: str) -> Optional[SignalOutcome]:
        """Get outcome for a signal."""
        return self.store.get_outcome(signal_id)

    def _calculate_returns(
        self,
        signal: TrackedSignal,
        entry_price: float,
        exit_price: float,
    ) -> tuple:
        """
        Calculate percentage return and R-multiple.

        Args:
            signal: Original signal with target/stop.
            entry_price: Actual entry price.
            exit_price: Actual exit price.

        Returns:
            Tuple of (return_pct, r_multiple).
        """
        # Percentage return
        if signal.direction.value == "BUY":
            return_pct = (exit_price - entry_price) / entry_price
        else:  # SELL (short)
            return_pct = (entry_price - exit_price) / entry_price

        # R-multiple calculation
        # Risk = distance from entry to stop
        risk = abs(entry_price - signal.stop_price)

        if risk > 0:
            # Reward = actual profit/loss
            if signal.direction.value == "BUY":
                reward = exit_price - entry_price
            else:
                reward = entry_price - exit_price

            r_multiple = reward / risk
        else:
            r_multiple = 0.0

        return return_pct, r_multiple


class AutoOutcomeRecorder:
    """
    Automatically record outcomes based on price data.

    This class monitors active signals and automatically records
    outcomes when target or stop prices are hit.
    """

    def __init__(
        self,
        store: BaseTrackingStore,
        outcome_recorder: OutcomeRecorder,
    ):
        self.store = store
        self.outcome_recorder = outcome_recorder

    def check_and_record_outcomes(
        self,
        price_data: dict,
    ) -> List[str]:
        """
        Check active signals against current prices and record outcomes.

        Args:
            price_data: Dict mapping symbol -> current price.

        Returns:
            List of signal IDs that were closed.
        """
        closed_signals = []
        active_signals = self.store.get_signals_by_status(SignalStatus.ACTIVE)

        for signal in active_signals:
            if signal.symbol not in price_data:
                continue

            current_price = price_data[signal.symbol]
            exit_reason = self._check_exit_condition(signal, current_price)

            if exit_reason:
                success = self.outcome_recorder.record_quick_outcome(
                    signal_id=signal.id,
                    exit_price=current_price,
                    exit_reason=exit_reason,
                )

                if success:
                    closed_signals.append(signal.id)
                    logger.info(
                        f"Auto-closed {signal.symbol}: {exit_reason.value} "
                        f"@ {current_price}"
                    )

        return closed_signals

    def _check_exit_condition(
        self,
        signal: TrackedSignal,
        current_price: float,
    ) -> Optional[ExitReason]:
        """Check if exit condition is met."""
        if signal.direction.value == "BUY":
            if current_price >= signal.target_price:
                return ExitReason.TARGET_HIT
            if current_price <= signal.stop_price:
                return ExitReason.STOP_HIT
        else:  # SELL (short)
            if current_price <= signal.target_price:
                return ExitReason.TARGET_HIT
            if current_price >= signal.stop_price:
                return ExitReason.STOP_HIT

        return None

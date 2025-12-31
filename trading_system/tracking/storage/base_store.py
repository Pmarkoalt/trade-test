"""Abstract base class for tracking storage."""

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Dict, List, Optional

from trading_system.tracking.models import (
    PerformanceMetrics,
    SignalOutcome,
    SignalStatus,
    TrackedSignal,
)


class BaseTrackingStore(ABC):
    """Abstract interface for tracking data storage."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the database (run migrations)."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close database connection."""
        pass

    # Signal operations
    @abstractmethod
    def insert_signal(self, signal: TrackedSignal) -> str:
        """Insert a new tracked signal. Returns signal ID."""
        pass

    @abstractmethod
    def update_signal(self, signal: TrackedSignal) -> bool:
        """Update an existing signal. Returns success."""
        pass

    @abstractmethod
    def update_signal_status(
        self,
        signal_id: str,
        status: SignalStatus,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """Update signal status. Returns success."""
        pass

    @abstractmethod
    def get_signal(self, signal_id: str) -> Optional[TrackedSignal]:
        """Get a signal by ID."""
        pass

    @abstractmethod
    def get_signals_by_status(
        self,
        status: SignalStatus,
        limit: int = 100,
    ) -> List[TrackedSignal]:
        """Get signals by status."""
        pass

    @abstractmethod
    def get_signals_by_date_range(
        self,
        start_date: date,
        end_date: date,
        symbol: Optional[str] = None,
        asset_class: Optional[str] = None,
    ) -> List[TrackedSignal]:
        """Get signals within a date range."""
        pass

    @abstractmethod
    def get_recent_signals(
        self,
        days: int = 7,
        symbol: Optional[str] = None,
    ) -> List[TrackedSignal]:
        """Get signals from the last N days."""
        pass

    # Outcome operations
    @abstractmethod
    def insert_outcome(self, outcome: SignalOutcome) -> bool:
        """Insert a signal outcome. Returns success."""
        pass

    @abstractmethod
    def update_outcome(self, outcome: SignalOutcome) -> bool:
        """Update an existing outcome. Returns success."""
        pass

    @abstractmethod
    def get_outcome(self, signal_id: str) -> Optional[SignalOutcome]:
        """Get outcome for a signal."""
        pass

    @abstractmethod
    def get_outcomes_by_date_range(
        self,
        start_date: date,
        end_date: date,
    ) -> List[SignalOutcome]:
        """Get outcomes within a date range."""
        pass

    # Performance operations
    @abstractmethod
    def save_daily_snapshot(
        self,
        snapshot_date: date,
        metrics: Dict,
    ) -> bool:
        """Save daily performance snapshot."""
        pass

    @abstractmethod
    def get_daily_snapshots(
        self,
        start_date: date,
        end_date: date,
    ) -> List[Dict]:
        """Get daily snapshots for date range."""
        pass

    # Aggregation queries
    @abstractmethod
    def count_signals_by_status(self) -> Dict[str, int]:
        """Count signals grouped by status."""
        pass

    @abstractmethod
    def get_signal_stats(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Dict:
        """Get aggregate signal statistics."""
        pass

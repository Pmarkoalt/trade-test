# Agent Tasks: Phase 3 - Performance Tracking (Part 1: Core Infrastructure)

**Phase Goal**: Track signal outcomes to measure performance and enable ML feedback loop
**Duration**: 1 week (Part 1)
**Prerequisites**: Phase 1 MVP complete, Phase 2 News Integration complete

---

## Phase 3 Part 1 Overview

### What We're Building
1. **Signal Tracker** - Record all generated signals with metadata
2. **Outcome Recorder** - Track actual price movements and returns
3. **Performance Calculator** - Compute metrics (win rate, R-multiples, Sharpe)
4. **Database Layer** - SQLite/PostgreSQL storage for tracking data

### Architecture Addition

```
trading_system/
├── tracking/                        # NEW: Performance tracking
│   ├── __init__.py
│   ├── models.py                    # Data models for tracking
│   ├── signal_tracker.py            # Record generated signals
│   ├── outcome_recorder.py          # Record trade outcomes
│   ├── performance_calculator.py    # Calculate metrics
│   └── storage/
│       ├── __init__.py
│       ├── base_store.py            # Abstract storage interface
│       ├── sqlite_store.py          # SQLite implementation
│       └── migrations/
│           └── 001_initial_schema.sql
```

---

## Task 3.1.1: Create Tracking Module Structure

**Context**:
We need to track every signal generated to measure system performance over time and provide feedback for ML refinement.

**Objective**:
Create the directory structure and core data models for the tracking module.

**Files to Create**:
```
trading_system/tracking/
├── __init__.py
├── models.py
├── signal_tracker.py      # Stub
├── outcome_recorder.py    # Stub
├── performance_calculator.py  # Stub
└── storage/
    ├── __init__.py
    ├── base_store.py
    ├── sqlite_store.py    # Stub
    └── migrations/
        └── 001_initial_schema.sql
```

**Requirements**:

1. Create `models.py` with tracking data models:
```python
"""Data models for performance tracking."""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional
from enum import Enum
import uuid


class SignalDirection(str, Enum):
    """Signal direction."""
    BUY = "BUY"
    SELL = "SELL"


class ConvictionLevel(str, Enum):
    """Signal conviction level."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class SignalStatus(str, Enum):
    """Signal lifecycle status."""
    PENDING = "pending"          # Generated, waiting for entry
    ACTIVE = "active"            # Position entered
    CLOSED = "closed"            # Position exited
    EXPIRED = "expired"          # Never entered, signal expired
    CANCELLED = "cancelled"      # Manually cancelled


class ExitReason(str, Enum):
    """Reason for exiting a position."""
    TARGET_HIT = "target_hit"
    STOP_HIT = "stop_hit"
    TRAILING_STOP = "trailing_stop"
    TIME_EXIT = "time_exit"
    MANUAL = "manual"
    SIGNAL_REVERSAL = "signal_reversal"


@dataclass
class TrackedSignal:
    """A signal tracked for performance measurement."""

    # Identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Signal details
    symbol: str = ""
    asset_class: str = ""              # "equity" or "crypto"
    direction: SignalDirection = SignalDirection.BUY
    signal_type: str = ""              # e.g., "breakout_20d", "news_sentiment"
    conviction: ConvictionLevel = ConvictionLevel.MEDIUM

    # Prices at signal generation
    signal_price: float = 0.0          # Price when signal generated
    entry_price: float = 0.0           # Recommended entry price
    target_price: float = 0.0          # Target price
    stop_price: float = 0.0            # Stop loss price

    # Scores
    technical_score: float = 0.0
    news_score: Optional[float] = None
    combined_score: float = 0.0

    # Sizing
    position_size_pct: float = 0.0     # % of portfolio

    # Status
    status: SignalStatus = SignalStatus.PENDING

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    delivered_at: Optional[datetime] = None
    entry_filled_at: Optional[datetime] = None
    exit_filled_at: Optional[datetime] = None

    # Delivery
    was_delivered: bool = False
    delivery_method: str = ""          # "email", "sms", "push"

    # Metadata
    reasoning: str = ""
    news_headlines: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "asset_class": self.asset_class,
            "direction": self.direction.value,
            "signal_type": self.signal_type,
            "conviction": self.conviction.value,
            "signal_price": self.signal_price,
            "entry_price": self.entry_price,
            "target_price": self.target_price,
            "stop_price": self.stop_price,
            "technical_score": self.technical_score,
            "news_score": self.news_score,
            "combined_score": self.combined_score,
            "position_size_pct": self.position_size_pct,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "entry_filled_at": self.entry_filled_at.isoformat() if self.entry_filled_at else None,
            "exit_filled_at": self.exit_filled_at.isoformat() if self.exit_filled_at else None,
            "was_delivered": self.was_delivered,
            "delivery_method": self.delivery_method,
            "reasoning": self.reasoning,
            "news_headlines": self.news_headlines,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TrackedSignal":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            symbol=data["symbol"],
            asset_class=data["asset_class"],
            direction=SignalDirection(data["direction"]),
            signal_type=data["signal_type"],
            conviction=ConvictionLevel(data["conviction"]),
            signal_price=data["signal_price"],
            entry_price=data["entry_price"],
            target_price=data["target_price"],
            stop_price=data["stop_price"],
            technical_score=data["technical_score"],
            news_score=data.get("news_score"),
            combined_score=data["combined_score"],
            position_size_pct=data["position_size_pct"],
            status=SignalStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            delivered_at=datetime.fromisoformat(data["delivered_at"]) if data.get("delivered_at") else None,
            entry_filled_at=datetime.fromisoformat(data["entry_filled_at"]) if data.get("entry_filled_at") else None,
            exit_filled_at=datetime.fromisoformat(data["exit_filled_at"]) if data.get("exit_filled_at") else None,
            was_delivered=data["was_delivered"],
            delivery_method=data["delivery_method"],
            reasoning=data["reasoning"],
            news_headlines=data.get("news_headlines", []),
            tags=data.get("tags", []),
        )


@dataclass
class SignalOutcome:
    """Outcome of a tracked signal."""

    # Link to signal
    signal_id: str = ""

    # Actual execution
    actual_entry_price: Optional[float] = None
    actual_entry_date: Optional[date] = None
    actual_exit_price: Optional[float] = None
    actual_exit_date: Optional[date] = None

    # Trade result
    exit_reason: Optional[ExitReason] = None
    holding_days: int = 0

    # Returns
    return_pct: float = 0.0            # Percentage return
    return_dollars: float = 0.0        # Dollar return (if position size known)
    r_multiple: float = 0.0            # Return in R-multiples

    # Benchmark comparison
    benchmark_return_pct: float = 0.0  # SPY/BTC return over same period
    alpha: float = 0.0                 # Signal return - benchmark return

    # User feedback
    was_followed: bool = False         # Did user take the trade?
    user_notes: str = ""

    # Timestamps
    recorded_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "signal_id": self.signal_id,
            "actual_entry_price": self.actual_entry_price,
            "actual_entry_date": self.actual_entry_date.isoformat() if self.actual_entry_date else None,
            "actual_exit_price": self.actual_exit_price,
            "actual_exit_date": self.actual_exit_date.isoformat() if self.actual_exit_date else None,
            "exit_reason": self.exit_reason.value if self.exit_reason else None,
            "holding_days": self.holding_days,
            "return_pct": self.return_pct,
            "return_dollars": self.return_dollars,
            "r_multiple": self.r_multiple,
            "benchmark_return_pct": self.benchmark_return_pct,
            "alpha": self.alpha,
            "was_followed": self.was_followed,
            "user_notes": self.user_notes,
            "recorded_at": self.recorded_at.isoformat(),
        }


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics."""

    # Period
    period_start: date = field(default_factory=date.today)
    period_end: date = field(default_factory=date.today)

    # Counts
    total_signals: int = 0
    signals_followed: int = 0
    signals_won: int = 0
    signals_lost: int = 0

    # Rates
    win_rate: float = 0.0              # signals_won / total closed
    follow_rate: float = 0.0           # signals_followed / total delivered

    # Returns
    total_return_pct: float = 0.0
    avg_return_pct: float = 0.0
    avg_winner_pct: float = 0.0
    avg_loser_pct: float = 0.0

    # R-multiples
    total_r: float = 0.0
    avg_r: float = 0.0
    avg_winner_r: float = 0.0
    avg_loser_r: float = 0.0
    expectancy_r: float = 0.0          # (win_rate * avg_winner_r) - (loss_rate * abs(avg_loser_r))

    # Risk metrics
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Benchmark comparison
    benchmark_return_pct: float = 0.0
    alpha: float = 0.0

    # By category
    metrics_by_asset_class: Dict[str, Dict] = field(default_factory=dict)
    metrics_by_signal_type: Dict[str, Dict] = field(default_factory=dict)
    metrics_by_conviction: Dict[str, Dict] = field(default_factory=dict)
```

2. Create `__init__.py` with exports:
```python
"""Performance tracking module."""

from trading_system.tracking.models import (
    ConvictionLevel,
    ExitReason,
    PerformanceMetrics,
    SignalDirection,
    SignalOutcome,
    SignalStatus,
    TrackedSignal,
)

__all__ = [
    "ConvictionLevel",
    "ExitReason",
    "PerformanceMetrics",
    "SignalDirection",
    "SignalOutcome",
    "SignalStatus",
    "TrackedSignal",
]
```

3. Create stub files for other modules with docstrings and placeholder classes.

**Acceptance Criteria**:
- [ ] All directories and files created
- [ ] Models properly defined with all fields
- [ ] to_dict/from_dict methods work correctly
- [ ] Imports work: `from trading_system.tracking import TrackedSignal`
- [ ] Type hints complete

**Tests to Write**:
```python
def test_tracked_signal_to_dict_roundtrip():
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

def test_signal_outcome_r_multiple():
    """Test R-multiple calculation."""
    # Entry 100, Stop 95 (risk = 5), Exit 110 (reward = 10)
    # R-multiple = 10/5 = 2.0
    pass
```

---

## Task 3.1.2: Design and Create Database Schema

**Context**:
We need persistent storage for signals and outcomes. SQLite is used for local development, with PostgreSQL support for production.

**Objective**:
Create the database schema and migration system.

**Files to Create/Modify**:
```
trading_system/tracking/storage/
├── migrations/
│   └── 001_initial_schema.sql
└── schema.py  # Schema definitions
```

**Requirements**:

1. Create `migrations/001_initial_schema.sql`:
```sql
-- Performance Tracking Schema
-- Migration: 001_initial_schema
-- Created: 2024-12-30

-- Tracked Signals Table
CREATE TABLE IF NOT EXISTS tracked_signals (
    id TEXT PRIMARY KEY,

    -- Signal details
    symbol TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    direction TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    conviction TEXT NOT NULL,

    -- Prices
    signal_price REAL NOT NULL,
    entry_price REAL NOT NULL,
    target_price REAL NOT NULL,
    stop_price REAL NOT NULL,

    -- Scores
    technical_score REAL,
    news_score REAL,
    combined_score REAL,

    -- Sizing
    position_size_pct REAL,

    -- Status
    status TEXT NOT NULL DEFAULT 'pending',

    -- Timestamps
    created_at TEXT NOT NULL,
    delivered_at TEXT,
    entry_filled_at TEXT,
    exit_filled_at TEXT,

    -- Delivery
    was_delivered INTEGER NOT NULL DEFAULT 0,
    delivery_method TEXT,

    -- Metadata
    reasoning TEXT,
    news_headlines TEXT,  -- JSON array
    tags TEXT,            -- JSON array

    -- Indexes
    created_date TEXT GENERATED ALWAYS AS (date(created_at)) STORED
);

CREATE INDEX IF NOT EXISTS idx_signals_symbol ON tracked_signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_created_date ON tracked_signals(created_date);
CREATE INDEX IF NOT EXISTS idx_signals_status ON tracked_signals(status);
CREATE INDEX IF NOT EXISTS idx_signals_asset_class ON tracked_signals(asset_class);


-- Signal Outcomes Table
CREATE TABLE IF NOT EXISTS signal_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id TEXT NOT NULL UNIQUE,

    -- Actual execution
    actual_entry_price REAL,
    actual_entry_date TEXT,
    actual_exit_price REAL,
    actual_exit_date TEXT,

    -- Trade result
    exit_reason TEXT,
    holding_days INTEGER,

    -- Returns
    return_pct REAL,
    return_dollars REAL,
    r_multiple REAL,

    -- Benchmark
    benchmark_return_pct REAL,
    alpha REAL,

    -- User feedback
    was_followed INTEGER NOT NULL DEFAULT 0,
    user_notes TEXT,

    -- Timestamps
    recorded_at TEXT NOT NULL,

    FOREIGN KEY (signal_id) REFERENCES tracked_signals(id)
);

CREATE INDEX IF NOT EXISTS idx_outcomes_signal_id ON signal_outcomes(signal_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_exit_date ON signal_outcomes(actual_exit_date);


-- Daily Performance Snapshots
CREATE TABLE IF NOT EXISTS daily_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date TEXT NOT NULL UNIQUE,

    -- Cumulative metrics
    total_signals INTEGER NOT NULL DEFAULT 0,
    total_closed INTEGER NOT NULL DEFAULT 0,
    total_wins INTEGER NOT NULL DEFAULT 0,
    total_losses INTEGER NOT NULL DEFAULT 0,

    -- Returns
    cumulative_return_pct REAL,
    cumulative_r REAL,

    -- Rolling metrics (last 30 days)
    rolling_win_rate REAL,
    rolling_avg_r REAL,
    rolling_sharpe REAL,

    -- Equity curve
    starting_equity REAL,
    current_equity REAL,
    high_water_mark REAL,
    current_drawdown_pct REAL,

    -- Timestamp
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_daily_perf_date ON daily_performance(snapshot_date);


-- Strategy Performance (aggregated by signal type)
CREATE TABLE IF NOT EXISTS strategy_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_type TEXT NOT NULL,
    period_type TEXT NOT NULL,  -- 'daily', 'weekly', 'monthly', 'all_time'
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,

    -- Counts
    total_signals INTEGER NOT NULL DEFAULT 0,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,

    -- Metrics
    win_rate REAL,
    avg_return_pct REAL,
    avg_r REAL,
    expectancy_r REAL,
    sharpe_ratio REAL,

    -- Timestamp
    calculated_at TEXT NOT NULL,

    UNIQUE(signal_type, period_type, period_start)
);


-- Metadata table for migrations
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

INSERT OR IGNORE INTO schema_migrations (version, applied_at)
VALUES (1, datetime('now'));
```

2. Create `schema.py` for programmatic schema access:
```python
"""Database schema definitions."""

from typing import List, Tuple

# Table definitions for validation
TABLES = {
    "tracked_signals": [
        "id", "symbol", "asset_class", "direction", "signal_type", "conviction",
        "signal_price", "entry_price", "target_price", "stop_price",
        "technical_score", "news_score", "combined_score", "position_size_pct",
        "status", "created_at", "delivered_at", "entry_filled_at", "exit_filled_at",
        "was_delivered", "delivery_method", "reasoning", "news_headlines", "tags",
    ],
    "signal_outcomes": [
        "id", "signal_id", "actual_entry_price", "actual_entry_date",
        "actual_exit_price", "actual_exit_date", "exit_reason", "holding_days",
        "return_pct", "return_dollars", "r_multiple", "benchmark_return_pct",
        "alpha", "was_followed", "user_notes", "recorded_at",
    ],
    "daily_performance": [
        "id", "snapshot_date", "total_signals", "total_closed", "total_wins",
        "total_losses", "cumulative_return_pct", "cumulative_r", "rolling_win_rate",
        "rolling_avg_r", "rolling_sharpe", "starting_equity", "current_equity",
        "high_water_mark", "current_drawdown_pct", "created_at",
    ],
    "strategy_performance": [
        "id", "signal_type", "period_type", "period_start", "period_end",
        "total_signals", "wins", "losses", "win_rate", "avg_return_pct",
        "avg_r", "expectancy_r", "sharpe_ratio", "calculated_at",
    ],
}

def get_migration_files() -> List[Tuple[int, str]]:
    """Get ordered list of migration files."""
    import os
    from pathlib import Path

    migrations_dir = Path(__file__).parent / "migrations"
    migrations = []

    for f in migrations_dir.glob("*.sql"):
        # Extract version number from filename (e.g., "001_initial_schema.sql" -> 1)
        version = int(f.stem.split("_")[0])
        migrations.append((version, str(f)))

    return sorted(migrations, key=lambda x: x[0])
```

**Acceptance Criteria**:
- [ ] SQL schema creates all tables without errors
- [ ] Indexes defined for common queries
- [ ] Foreign key relationships correct
- [ ] Migration versioning system works
- [ ] Schema validated in tests

**Tests to Write**:
```python
def test_schema_creates_tables(tmp_path):
    """Test schema creation."""
    import sqlite3
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)

    # Run migration
    with open("trading_system/tracking/storage/migrations/001_initial_schema.sql") as f:
        conn.executescript(f.read())

    # Verify tables exist
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
    tables = {row[0] for row in cursor.fetchall()}

    assert "tracked_signals" in tables
    assert "signal_outcomes" in tables
    assert "daily_performance" in tables
```

---

## Task 3.1.3: Implement Storage Layer

**Context**:
We need an abstraction layer for database operations that supports both SQLite and PostgreSQL.

**Objective**:
Implement the base storage interface and SQLite implementation.

**Files to Create/Modify**:
```
trading_system/tracking/storage/
├── base_store.py
├── sqlite_store.py
└── __init__.py
```

**Requirements**:

1. Create `base_store.py` with abstract interface:
```python
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
        timestamp: Optional[datetime] = None
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
        limit: int = 100
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
```

2. Create `sqlite_store.py` with full implementation:
```python
"""SQLite implementation of tracking storage."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from trading_system.tracking.models import (
    ConvictionLevel,
    ExitReason,
    PerformanceMetrics,
    SignalDirection,
    SignalOutcome,
    SignalStatus,
    TrackedSignal,
)
from trading_system.tracking.storage.base_store import BaseTrackingStore


class SQLiteTrackingStore(BaseTrackingStore):
    """SQLite implementation for tracking storage."""

    def __init__(self, db_path: str = "tracking.db"):
        """
        Initialize SQLite store.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self._connection: Optional[sqlite3.Connection] = None

    @property
    def connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                self.db_path,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            self._connection.row_factory = sqlite3.Row
            # Enable foreign keys
            self._connection.execute("PRAGMA foreign_keys = ON")
        return self._connection

    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        try:
            yield self.connection
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def initialize(self) -> None:
        """Initialize database with schema."""
        migrations_dir = Path(__file__).parent / "migrations"
        migration_file = migrations_dir / "001_initial_schema.sql"

        if not migration_file.exists():
            raise FileNotFoundError(f"Migration file not found: {migration_file}")

        with open(migration_file) as f:
            schema_sql = f.read()

        with self.transaction():
            self.connection.executescript(schema_sql)

        logger.info(f"Initialized tracking database at {self.db_path}")

    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None

    # Signal operations
    def insert_signal(self, signal: TrackedSignal) -> str:
        """Insert a new tracked signal."""
        sql = """
            INSERT INTO tracked_signals (
                id, symbol, asset_class, direction, signal_type, conviction,
                signal_price, entry_price, target_price, stop_price,
                technical_score, news_score, combined_score, position_size_pct,
                status, created_at, delivered_at, entry_filled_at, exit_filled_at,
                was_delivered, delivery_method, reasoning, news_headlines, tags
            ) VALUES (
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?
            )
        """

        with self.transaction():
            self.connection.execute(sql, (
                signal.id,
                signal.symbol,
                signal.asset_class,
                signal.direction.value,
                signal.signal_type,
                signal.conviction.value,
                signal.signal_price,
                signal.entry_price,
                signal.target_price,
                signal.stop_price,
                signal.technical_score,
                signal.news_score,
                signal.combined_score,
                signal.position_size_pct,
                signal.status.value,
                signal.created_at.isoformat(),
                signal.delivered_at.isoformat() if signal.delivered_at else None,
                signal.entry_filled_at.isoformat() if signal.entry_filled_at else None,
                signal.exit_filled_at.isoformat() if signal.exit_filled_at else None,
                1 if signal.was_delivered else 0,
                signal.delivery_method,
                signal.reasoning,
                json.dumps(signal.news_headlines),
                json.dumps(signal.tags),
            ))

        logger.debug(f"Inserted signal {signal.id} for {signal.symbol}")
        return signal.id

    def update_signal(self, signal: TrackedSignal) -> bool:
        """Update an existing signal."""
        sql = """
            UPDATE tracked_signals SET
                status = ?,
                delivered_at = ?,
                entry_filled_at = ?,
                exit_filled_at = ?,
                was_delivered = ?,
                delivery_method = ?
            WHERE id = ?
        """

        with self.transaction():
            cursor = self.connection.execute(sql, (
                signal.status.value,
                signal.delivered_at.isoformat() if signal.delivered_at else None,
                signal.entry_filled_at.isoformat() if signal.entry_filled_at else None,
                signal.exit_filled_at.isoformat() if signal.exit_filled_at else None,
                1 if signal.was_delivered else 0,
                signal.delivery_method,
                signal.id,
            ))

        return cursor.rowcount > 0

    def update_signal_status(
        self,
        signal_id: str,
        status: SignalStatus,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """Update signal status."""
        timestamp = timestamp or datetime.now()

        # Determine which timestamp field to update based on status
        timestamp_field = None
        if status == SignalStatus.ACTIVE:
            timestamp_field = "entry_filled_at"
        elif status in (SignalStatus.CLOSED, SignalStatus.EXPIRED):
            timestamp_field = "exit_filled_at"

        if timestamp_field:
            sql = f"""
                UPDATE tracked_signals
                SET status = ?, {timestamp_field} = ?
                WHERE id = ?
            """
            params = (status.value, timestamp.isoformat(), signal_id)
        else:
            sql = "UPDATE tracked_signals SET status = ? WHERE id = ?"
            params = (status.value, signal_id)

        with self.transaction():
            cursor = self.connection.execute(sql, params)

        return cursor.rowcount > 0

    def get_signal(self, signal_id: str) -> Optional[TrackedSignal]:
        """Get a signal by ID."""
        sql = "SELECT * FROM tracked_signals WHERE id = ?"

        cursor = self.connection.execute(sql, (signal_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_signal(row)

    def get_signals_by_status(
        self,
        status: SignalStatus,
        limit: int = 100,
    ) -> List[TrackedSignal]:
        """Get signals by status."""
        sql = """
            SELECT * FROM tracked_signals
            WHERE status = ?
            ORDER BY created_at DESC
            LIMIT ?
        """

        cursor = self.connection.execute(sql, (status.value, limit))
        return [self._row_to_signal(row) for row in cursor.fetchall()]

    def get_signals_by_date_range(
        self,
        start_date: date,
        end_date: date,
        symbol: Optional[str] = None,
        asset_class: Optional[str] = None,
    ) -> List[TrackedSignal]:
        """Get signals within a date range."""
        conditions = ["created_date >= ? AND created_date <= ?"]
        params: List = [start_date.isoformat(), end_date.isoformat()]

        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)

        if asset_class:
            conditions.append("asset_class = ?")
            params.append(asset_class)

        sql = f"""
            SELECT * FROM tracked_signals
            WHERE {' AND '.join(conditions)}
            ORDER BY created_at DESC
        """

        cursor = self.connection.execute(sql, params)
        return [self._row_to_signal(row) for row in cursor.fetchall()]

    def get_recent_signals(
        self,
        days: int = 7,
        symbol: Optional[str] = None,
    ) -> List[TrackedSignal]:
        """Get signals from the last N days."""
        start_date = date.today() - timedelta(days=days)
        return self.get_signals_by_date_range(
            start_date=start_date,
            end_date=date.today(),
            symbol=symbol,
        )

    # Outcome operations
    def insert_outcome(self, outcome: SignalOutcome) -> bool:
        """Insert a signal outcome."""
        sql = """
            INSERT INTO signal_outcomes (
                signal_id, actual_entry_price, actual_entry_date,
                actual_exit_price, actual_exit_date, exit_reason,
                holding_days, return_pct, return_dollars, r_multiple,
                benchmark_return_pct, alpha, was_followed, user_notes,
                recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        with self.transaction():
            self.connection.execute(sql, (
                outcome.signal_id,
                outcome.actual_entry_price,
                outcome.actual_entry_date.isoformat() if outcome.actual_entry_date else None,
                outcome.actual_exit_price,
                outcome.actual_exit_date.isoformat() if outcome.actual_exit_date else None,
                outcome.exit_reason.value if outcome.exit_reason else None,
                outcome.holding_days,
                outcome.return_pct,
                outcome.return_dollars,
                outcome.r_multiple,
                outcome.benchmark_return_pct,
                outcome.alpha,
                1 if outcome.was_followed else 0,
                outcome.user_notes,
                outcome.recorded_at.isoformat(),
            ))

        return True

    def update_outcome(self, outcome: SignalOutcome) -> bool:
        """Update an existing outcome."""
        sql = """
            UPDATE signal_outcomes SET
                actual_entry_price = ?,
                actual_entry_date = ?,
                actual_exit_price = ?,
                actual_exit_date = ?,
                exit_reason = ?,
                holding_days = ?,
                return_pct = ?,
                return_dollars = ?,
                r_multiple = ?,
                benchmark_return_pct = ?,
                alpha = ?,
                was_followed = ?,
                user_notes = ?
            WHERE signal_id = ?
        """

        with self.transaction():
            cursor = self.connection.execute(sql, (
                outcome.actual_entry_price,
                outcome.actual_entry_date.isoformat() if outcome.actual_entry_date else None,
                outcome.actual_exit_price,
                outcome.actual_exit_date.isoformat() if outcome.actual_exit_date else None,
                outcome.exit_reason.value if outcome.exit_reason else None,
                outcome.holding_days,
                outcome.return_pct,
                outcome.return_dollars,
                outcome.r_multiple,
                outcome.benchmark_return_pct,
                outcome.alpha,
                1 if outcome.was_followed else 0,
                outcome.user_notes,
                outcome.signal_id,
            ))

        return cursor.rowcount > 0

    def get_outcome(self, signal_id: str) -> Optional[SignalOutcome]:
        """Get outcome for a signal."""
        sql = "SELECT * FROM signal_outcomes WHERE signal_id = ?"

        cursor = self.connection.execute(sql, (signal_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_outcome(row)

    def get_outcomes_by_date_range(
        self,
        start_date: date,
        end_date: date,
    ) -> List[SignalOutcome]:
        """Get outcomes within a date range."""
        sql = """
            SELECT * FROM signal_outcomes
            WHERE actual_exit_date >= ? AND actual_exit_date <= ?
            ORDER BY actual_exit_date DESC
        """

        cursor = self.connection.execute(sql, (
            start_date.isoformat(),
            end_date.isoformat(),
        ))
        return [self._row_to_outcome(row) for row in cursor.fetchall()]

    # Performance operations
    def save_daily_snapshot(
        self,
        snapshot_date: date,
        metrics: Dict,
    ) -> bool:
        """Save daily performance snapshot."""
        sql = """
            INSERT OR REPLACE INTO daily_performance (
                snapshot_date, total_signals, total_closed, total_wins,
                total_losses, cumulative_return_pct, cumulative_r,
                rolling_win_rate, rolling_avg_r, rolling_sharpe,
                starting_equity, current_equity, high_water_mark,
                current_drawdown_pct, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        with self.transaction():
            self.connection.execute(sql, (
                snapshot_date.isoformat(),
                metrics.get("total_signals", 0),
                metrics.get("total_closed", 0),
                metrics.get("total_wins", 0),
                metrics.get("total_losses", 0),
                metrics.get("cumulative_return_pct"),
                metrics.get("cumulative_r"),
                metrics.get("rolling_win_rate"),
                metrics.get("rolling_avg_r"),
                metrics.get("rolling_sharpe"),
                metrics.get("starting_equity"),
                metrics.get("current_equity"),
                metrics.get("high_water_mark"),
                metrics.get("current_drawdown_pct"),
                datetime.now().isoformat(),
            ))

        return True

    def get_daily_snapshots(
        self,
        start_date: date,
        end_date: date,
    ) -> List[Dict]:
        """Get daily snapshots for date range."""
        sql = """
            SELECT * FROM daily_performance
            WHERE snapshot_date >= ? AND snapshot_date <= ?
            ORDER BY snapshot_date ASC
        """

        cursor = self.connection.execute(sql, (
            start_date.isoformat(),
            end_date.isoformat(),
        ))

        return [dict(row) for row in cursor.fetchall()]

    # Aggregation queries
    def count_signals_by_status(self) -> Dict[str, int]:
        """Count signals grouped by status."""
        sql = """
            SELECT status, COUNT(*) as count
            FROM tracked_signals
            GROUP BY status
        """

        cursor = self.connection.execute(sql)
        return {row["status"]: row["count"] for row in cursor.fetchall()}

    def get_signal_stats(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Dict:
        """Get aggregate signal statistics."""
        conditions = []
        params: List = []

        if start_date:
            conditions.append("created_date >= ?")
            params.append(start_date.isoformat())

        if end_date:
            conditions.append("created_date <= ?")
            params.append(end_date.isoformat())

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        sql = f"""
            SELECT
                COUNT(*) as total_signals,
                SUM(CASE WHEN status = 'closed' THEN 1 ELSE 0 END) as closed_signals,
                SUM(CASE WHEN was_delivered = 1 THEN 1 ELSE 0 END) as delivered_signals,
                AVG(combined_score) as avg_combined_score,
                COUNT(DISTINCT symbol) as unique_symbols,
                COUNT(DISTINCT asset_class) as asset_classes
            FROM tracked_signals
            {where_clause}
        """

        cursor = self.connection.execute(sql, params)
        row = cursor.fetchone()

        return dict(row) if row else {}

    # Helper methods
    def _row_to_signal(self, row: sqlite3.Row) -> TrackedSignal:
        """Convert database row to TrackedSignal."""
        return TrackedSignal(
            id=row["id"],
            symbol=row["symbol"],
            asset_class=row["asset_class"],
            direction=SignalDirection(row["direction"]),
            signal_type=row["signal_type"],
            conviction=ConvictionLevel(row["conviction"]),
            signal_price=row["signal_price"],
            entry_price=row["entry_price"],
            target_price=row["target_price"],
            stop_price=row["stop_price"],
            technical_score=row["technical_score"],
            news_score=row["news_score"],
            combined_score=row["combined_score"],
            position_size_pct=row["position_size_pct"],
            status=SignalStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            delivered_at=datetime.fromisoformat(row["delivered_at"]) if row["delivered_at"] else None,
            entry_filled_at=datetime.fromisoformat(row["entry_filled_at"]) if row["entry_filled_at"] else None,
            exit_filled_at=datetime.fromisoformat(row["exit_filled_at"]) if row["exit_filled_at"] else None,
            was_delivered=bool(row["was_delivered"]),
            delivery_method=row["delivery_method"] or "",
            reasoning=row["reasoning"] or "",
            news_headlines=json.loads(row["news_headlines"]) if row["news_headlines"] else [],
            tags=json.loads(row["tags"]) if row["tags"] else [],
        )

    def _row_to_outcome(self, row: sqlite3.Row) -> SignalOutcome:
        """Convert database row to SignalOutcome."""
        return SignalOutcome(
            signal_id=row["signal_id"],
            actual_entry_price=row["actual_entry_price"],
            actual_entry_date=date.fromisoformat(row["actual_entry_date"]) if row["actual_entry_date"] else None,
            actual_exit_price=row["actual_exit_price"],
            actual_exit_date=date.fromisoformat(row["actual_exit_date"]) if row["actual_exit_date"] else None,
            exit_reason=ExitReason(row["exit_reason"]) if row["exit_reason"] else None,
            holding_days=row["holding_days"] or 0,
            return_pct=row["return_pct"] or 0.0,
            return_dollars=row["return_dollars"] or 0.0,
            r_multiple=row["r_multiple"] or 0.0,
            benchmark_return_pct=row["benchmark_return_pct"] or 0.0,
            alpha=row["alpha"] or 0.0,
            was_followed=bool(row["was_followed"]),
            user_notes=row["user_notes"] or "",
            recorded_at=datetime.fromisoformat(row["recorded_at"]),
        )
```

3. Update `storage/__init__.py`:
```python
"""Storage implementations for tracking."""

from trading_system.tracking.storage.base_store import BaseTrackingStore
from trading_system.tracking.storage.sqlite_store import SQLiteTrackingStore

__all__ = [
    "BaseTrackingStore",
    "SQLiteTrackingStore",
]
```

**Acceptance Criteria**:
- [ ] SQLiteTrackingStore implements all BaseTrackingStore methods
- [ ] All CRUD operations work correctly
- [ ] Transactions handle errors with rollback
- [ ] Indexes improve query performance
- [ ] Tests cover all methods

**Tests to Write**:
```python
class TestSQLiteTrackingStore:
    def test_insert_and_get_signal(self, tmp_path):
        """Test signal insert and retrieval."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()

        signal = TrackedSignal(
            symbol="AAPL",
            direction=SignalDirection.BUY,
            entry_price=150.0,
        )

        signal_id = store.insert_signal(signal)
        retrieved = store.get_signal(signal_id)

        assert retrieved is not None
        assert retrieved.symbol == "AAPL"
        store.close()

    def test_update_signal_status(self, tmp_path):
        """Test status updates."""
        pass

    def test_outcome_lifecycle(self, tmp_path):
        """Test outcome insert and update."""
        pass
```

---

## Task 3.1.4: Implement Signal Tracker

**Context**:
The SignalTracker is the main interface for recording signals generated by the system.

**Objective**:
Implement the SignalTracker class that integrates with the signal generation pipeline.

**Files to Create/Modify**:
```
trading_system/tracking/signal_tracker.py
```

**Requirements**:

```python
"""Signal tracker for recording generated signals."""

from datetime import datetime
from typing import List, Optional

from loguru import logger

from trading_system.tracking.models import (
    ConvictionLevel,
    SignalDirection,
    SignalStatus,
    TrackedSignal,
)
from trading_system.tracking.storage.base_store import BaseTrackingStore


class SignalTracker:
    """
    Track all signals generated by the system.

    This class is the main interface for recording signals
    and should be called from the signal generation pipeline.

    Example:
        tracker = SignalTracker(store)

        # When generating signals
        for recommendation in recommendations:
            signal_id = tracker.record_signal(recommendation)

        # When delivering signals
        tracker.mark_delivered(signal_id, method="email")
    """

    def __init__(self, store: BaseTrackingStore):
        """
        Initialize signal tracker.

        Args:
            store: Storage backend for persisting signals.
        """
        self.store = store

    def record_signal(
        self,
        symbol: str,
        asset_class: str,
        direction: SignalDirection,
        signal_type: str,
        conviction: ConvictionLevel,
        signal_price: float,
        entry_price: float,
        target_price: float,
        stop_price: float,
        technical_score: float = 0.0,
        news_score: Optional[float] = None,
        combined_score: float = 0.0,
        position_size_pct: float = 0.0,
        reasoning: str = "",
        news_headlines: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Record a newly generated signal.

        Args:
            symbol: Ticker symbol (e.g., "AAPL", "BTC").
            asset_class: "equity" or "crypto".
            direction: BUY or SELL.
            signal_type: Type of signal (e.g., "breakout_20d").
            conviction: HIGH, MEDIUM, or LOW.
            signal_price: Price when signal was generated.
            entry_price: Recommended entry price.
            target_price: Target price for exit.
            stop_price: Stop loss price.
            technical_score: Technical analysis score (0-10).
            news_score: News sentiment score (0-10), if available.
            combined_score: Combined score (0-10).
            position_size_pct: Recommended position size as % of portfolio.
            reasoning: Text explanation of signal.
            news_headlines: List of relevant news headlines.
            tags: Optional tags for categorization.

        Returns:
            Signal ID for tracking.
        """
        signal = TrackedSignal(
            symbol=symbol,
            asset_class=asset_class,
            direction=direction,
            signal_type=signal_type,
            conviction=conviction,
            signal_price=signal_price,
            entry_price=entry_price,
            target_price=target_price,
            stop_price=stop_price,
            technical_score=technical_score,
            news_score=news_score,
            combined_score=combined_score,
            position_size_pct=position_size_pct,
            reasoning=reasoning,
            news_headlines=news_headlines or [],
            tags=tags or [],
            status=SignalStatus.PENDING,
            created_at=datetime.now(),
        )

        signal_id = self.store.insert_signal(signal)

        logger.info(
            f"Recorded signal {signal_id}: {direction.value} {symbol} "
            f"@ {entry_price} (conviction: {conviction.value})"
        )

        return signal_id

    def record_from_recommendation(self, recommendation) -> str:
        """
        Record signal from a Recommendation object.

        Args:
            recommendation: Recommendation dataclass from signals module.

        Returns:
            Signal ID for tracking.
        """
        return self.record_signal(
            symbol=recommendation.symbol,
            asset_class=recommendation.asset_class,
            direction=SignalDirection(recommendation.direction),
            signal_type=recommendation.signal_type,
            conviction=ConvictionLevel(recommendation.conviction),
            signal_price=recommendation.signal_price,
            entry_price=recommendation.entry_price,
            target_price=recommendation.target_price,
            stop_price=recommendation.stop_price,
            technical_score=recommendation.technical_score,
            news_score=recommendation.news_score,
            combined_score=recommendation.combined_score,
            position_size_pct=recommendation.position_size_pct,
            reasoning=recommendation.reasoning,
            news_headlines=recommendation.news_headlines,
        )

    def mark_delivered(
        self,
        signal_id: str,
        method: str = "email",
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Mark a signal as delivered to user.

        Args:
            signal_id: ID of signal to mark.
            method: Delivery method ("email", "sms", "push").
            timestamp: Delivery timestamp (defaults to now).

        Returns:
            True if successful.
        """
        signal = self.store.get_signal(signal_id)
        if signal is None:
            logger.warning(f"Signal {signal_id} not found for delivery marking")
            return False

        signal.was_delivered = True
        signal.delivery_method = method
        signal.delivered_at = timestamp or datetime.now()

        success = self.store.update_signal(signal)

        if success:
            logger.debug(f"Marked signal {signal_id} as delivered via {method}")

        return success

    def mark_entry_filled(
        self,
        signal_id: str,
        fill_price: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Mark a signal as having entry filled (position opened).

        Args:
            signal_id: ID of signal.
            fill_price: Actual fill price (for slippage tracking).
            timestamp: Fill timestamp.

        Returns:
            True if successful.
        """
        success = self.store.update_signal_status(
            signal_id=signal_id,
            status=SignalStatus.ACTIVE,
            timestamp=timestamp or datetime.now(),
        )

        if success:
            logger.info(f"Signal {signal_id} entry filled")

        return success

    def mark_expired(
        self,
        signal_id: str,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Mark a signal as expired (never entered).

        Args:
            signal_id: ID of signal.
            timestamp: Expiration timestamp.

        Returns:
            True if successful.
        """
        return self.store.update_signal_status(
            signal_id=signal_id,
            status=SignalStatus.EXPIRED,
            timestamp=timestamp or datetime.now(),
        )

    def get_pending_signals(self, limit: int = 100) -> List[TrackedSignal]:
        """Get signals waiting for entry."""
        return self.store.get_signals_by_status(SignalStatus.PENDING, limit)

    def get_active_signals(self, limit: int = 100) -> List[TrackedSignal]:
        """Get signals with open positions."""
        return self.store.get_signals_by_status(SignalStatus.ACTIVE, limit)

    def get_recent_signals(
        self,
        days: int = 7,
        symbol: Optional[str] = None,
    ) -> List[TrackedSignal]:
        """Get signals from recent days."""
        return self.store.get_recent_signals(days, symbol)

    def get_signal(self, signal_id: str) -> Optional[TrackedSignal]:
        """Get a specific signal by ID."""
        return self.store.get_signal(signal_id)

    def get_signal_counts(self) -> dict:
        """Get counts of signals by status."""
        return self.store.count_signals_by_status()
```

**Acceptance Criteria**:
- [ ] SignalTracker can record signals with all fields
- [ ] Delivery marking updates timestamp and method
- [ ] Status transitions work correctly
- [ ] Integration with Recommendation dataclass works
- [ ] Logging provides visibility into operations

**Tests to Write**:
```python
class TestSignalTracker:
    def test_record_signal(self, tmp_path):
        """Test recording a new signal."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        tracker = SignalTracker(store)

        signal_id = tracker.record_signal(
            symbol="AAPL",
            asset_class="equity",
            direction=SignalDirection.BUY,
            signal_type="breakout_20d",
            conviction=ConvictionLevel.HIGH,
            signal_price=150.0,
            entry_price=150.0,
            target_price=165.0,
            stop_price=145.0,
        )

        assert signal_id is not None
        signal = tracker.get_signal(signal_id)
        assert signal.symbol == "AAPL"
        assert signal.status == SignalStatus.PENDING

    def test_mark_delivered(self, tmp_path):
        """Test delivery marking."""
        pass

    def test_signal_lifecycle(self, tmp_path):
        """Test full signal lifecycle: pending -> active -> closed."""
        pass
```

---

## Task 3.1.5: Implement Outcome Recorder

**Context**:
The OutcomeRecorder tracks actual trade results after signals are acted upon.

**Objective**:
Implement the OutcomeRecorder class for recording trade outcomes and calculating returns.

**Files to Create/Modify**:
```
trading_system/tracking/outcome_recorder.py
```

**Requirements**:

```python
"""Outcome recorder for tracking trade results."""

from datetime import date, datetime
from typing import Optional

from loguru import logger

from trading_system.tracking.models import (
    ExitReason,
    SignalOutcome,
    SignalStatus,
    TrackedSignal,
)
from trading_system.tracking.storage.base_store import BaseTrackingStore


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
    ) -> list:
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
```

**Acceptance Criteria**:
- [ ] OutcomeRecorder calculates returns correctly for BUY and SELL
- [ ] R-multiple calculation is accurate
- [ ] Signal status updates to CLOSED on outcome recording
- [ ] AutoOutcomeRecorder detects target/stop hits
- [ ] Alpha calculation (signal return - benchmark) works

**Tests to Write**:
```python
class TestOutcomeRecorder:
    def test_calculate_returns_buy_winner(self, tmp_path):
        """Test return calculation for winning long trade."""
        # Entry 100, Stop 95, Target 110, Exit 108
        # Return = (108-100)/100 = 8%
        # Risk = 100-95 = 5
        # Reward = 108-100 = 8
        # R = 8/5 = 1.6
        pass

    def test_calculate_returns_buy_loser(self, tmp_path):
        """Test return calculation for losing long trade."""
        # Entry 100, Stop 95, Exit 94
        # Return = (94-100)/100 = -6%
        # Risk = 5, Reward = -6
        # R = -6/5 = -1.2
        pass

    def test_calculate_returns_sell_winner(self, tmp_path):
        """Test return calculation for winning short trade."""
        pass

    def test_auto_outcome_target_hit(self, tmp_path):
        """Test auto-recording when target is hit."""
        pass
```

---

## Task 3.1.6: Implement Performance Calculator

**Context**:
The PerformanceCalculator computes aggregate metrics from tracked signals and outcomes.

**Objective**:
Implement performance calculation including win rate, expectancy, Sharpe ratio, and more.

**Files to Create/Modify**:
```
trading_system/tracking/performance_calculator.py
```

**Requirements**:

```python
"""Performance calculator for computing trading metrics."""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

from trading_system.tracking.models import (
    PerformanceMetrics,
    SignalOutcome,
    SignalStatus,
    TrackedSignal,
)
from trading_system.tracking.storage.base_store import BaseTrackingStore


class PerformanceCalculator:
    """
    Calculate performance metrics from tracked signals.

    Example:
        calculator = PerformanceCalculator(store)

        # Get overall metrics
        metrics = calculator.calculate_metrics()
        print(f"Win Rate: {metrics.win_rate:.1%}")
        print(f"Expectancy: {metrics.expectancy_r:.2f}R")

        # Get metrics for specific period
        mtd_metrics = calculator.calculate_metrics(
            start_date=date.today().replace(day=1),
            end_date=date.today(),
        )
    """

    # Risk-free rate for Sharpe calculation (annualized)
    RISK_FREE_RATE = 0.05  # 5% (T-bills)

    # Trading days per year for annualization
    TRADING_DAYS_PER_YEAR = 252

    def __init__(self, store: BaseTrackingStore):
        """
        Initialize calculator.

        Args:
            store: Storage backend with signal/outcome data.
        """
        self.store = store

    def calculate_metrics(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        symbol: Optional[str] = None,
        asset_class: Optional[str] = None,
        signal_type: Optional[str] = None,
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.

        Args:
            start_date: Start of period (default: all time).
            end_date: End of period (default: today).
            symbol: Filter by symbol.
            asset_class: Filter by asset class.
            signal_type: Filter by signal type.

        Returns:
            PerformanceMetrics with all calculated values.
        """
        # Get signals and outcomes
        signals = self._get_filtered_signals(
            start_date, end_date, symbol, asset_class, signal_type
        )
        outcomes = self._get_outcomes_for_signals(signals)

        if not signals:
            return PerformanceMetrics(
                period_start=start_date or date.today(),
                period_end=end_date or date.today(),
            )

        # Basic counts
        total_signals = len(signals)
        closed_outcomes = [o for o in outcomes if o.return_pct != 0 or o.was_followed]
        followed_outcomes = [o for o in outcomes if o.was_followed]

        winners = [o for o in followed_outcomes if o.return_pct > 0]
        losers = [o for o in followed_outcomes if o.return_pct < 0]

        # Win rate
        total_closed = len(followed_outcomes)
        win_rate = len(winners) / total_closed if total_closed > 0 else 0.0

        # Follow rate
        delivered_signals = [s for s in signals if s.was_delivered]
        follow_rate = len(followed_outcomes) / len(delivered_signals) if delivered_signals else 0.0

        # Returns
        returns_pct = [o.return_pct for o in followed_outcomes]
        total_return = sum(returns_pct) if returns_pct else 0.0
        avg_return = np.mean(returns_pct) if returns_pct else 0.0

        winner_returns = [o.return_pct for o in winners]
        loser_returns = [o.return_pct for o in losers]
        avg_winner = np.mean(winner_returns) if winner_returns else 0.0
        avg_loser = np.mean(loser_returns) if loser_returns else 0.0

        # R-multiples
        r_values = [o.r_multiple for o in followed_outcomes]
        total_r = sum(r_values) if r_values else 0.0
        avg_r = np.mean(r_values) if r_values else 0.0

        winner_r = [o.r_multiple for o in winners]
        loser_r = [o.r_multiple for o in losers]
        avg_winner_r = np.mean(winner_r) if winner_r else 0.0
        avg_loser_r = np.mean(loser_r) if loser_r else 0.0

        # Expectancy = (Win% * AvgWin) - (Loss% * AvgLoss)
        loss_rate = 1 - win_rate
        expectancy_r = (win_rate * avg_winner_r) - (loss_rate * abs(avg_loser_r))

        # Risk metrics
        sharpe = self._calculate_sharpe(returns_pct)
        sortino = self._calculate_sortino(returns_pct)
        max_dd = self._calculate_max_drawdown(returns_pct)
        calmar = abs(total_return / max_dd) if max_dd != 0 else 0.0

        # Benchmark comparison
        alphas = [o.alpha for o in followed_outcomes if o.alpha is not None]
        benchmark_returns = [o.benchmark_return_pct for o in followed_outcomes]
        total_benchmark = sum(benchmark_returns) if benchmark_returns else 0.0
        total_alpha = sum(alphas) if alphas else 0.0

        # Build metrics by category
        metrics_by_asset = self._metrics_by_category(
            outcomes, lambda o: self._get_signal(o.signal_id).asset_class
        )
        metrics_by_type = self._metrics_by_category(
            outcomes, lambda o: self._get_signal(o.signal_id).signal_type
        )
        metrics_by_conviction = self._metrics_by_category(
            outcomes, lambda o: self._get_signal(o.signal_id).conviction.value
        )

        return PerformanceMetrics(
            period_start=start_date or min(s.created_at.date() for s in signals),
            period_end=end_date or max(s.created_at.date() for s in signals),
            total_signals=total_signals,
            signals_followed=len(followed_outcomes),
            signals_won=len(winners),
            signals_lost=len(losers),
            win_rate=win_rate,
            follow_rate=follow_rate,
            total_return_pct=total_return,
            avg_return_pct=avg_return,
            avg_winner_pct=avg_winner,
            avg_loser_pct=avg_loser,
            total_r=total_r,
            avg_r=avg_r,
            avg_winner_r=avg_winner_r,
            avg_loser_r=avg_loser_r,
            expectancy_r=expectancy_r,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            benchmark_return_pct=total_benchmark,
            alpha=total_alpha,
            metrics_by_asset_class=metrics_by_asset,
            metrics_by_signal_type=metrics_by_type,
            metrics_by_conviction=metrics_by_conviction,
        )

    def calculate_rolling_metrics(
        self,
        window_days: int = 30,
    ) -> Dict:
        """
        Calculate rolling performance metrics.

        Args:
            window_days: Rolling window size in days.

        Returns:
            Dict with rolling metrics.
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=window_days)

        metrics = self.calculate_metrics(
            start_date=start_date,
            end_date=end_date,
        )

        return {
            "window_days": window_days,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "win_rate": metrics.win_rate,
            "avg_r": metrics.avg_r,
            "expectancy_r": metrics.expectancy_r,
            "sharpe": metrics.sharpe_ratio,
            "total_signals": metrics.total_signals,
        }

    def get_equity_curve(
        self,
        starting_equity: float = 100000.0,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Dict]:
        """
        Generate equity curve from outcomes.

        Args:
            starting_equity: Starting portfolio value.
            start_date: Start date.
            end_date: End date.

        Returns:
            List of dicts with date and equity value.
        """
        outcomes = self.store.get_outcomes_by_date_range(
            start_date=start_date or date(2020, 1, 1),
            end_date=end_date or date.today(),
        )

        if not outcomes:
            return []

        # Sort by exit date
        outcomes.sort(key=lambda o: o.actual_exit_date or date.today())

        equity = starting_equity
        curve = []
        high_water_mark = equity

        for outcome in outcomes:
            if not outcome.was_followed:
                continue

            # Calculate position return
            position_size = starting_equity * 0.0075  # Default 0.75% risk
            position_return = position_size * outcome.r_multiple
            equity += position_return

            high_water_mark = max(high_water_mark, equity)
            drawdown = (equity - high_water_mark) / high_water_mark if high_water_mark > 0 else 0

            curve.append({
                "date": outcome.actual_exit_date.isoformat() if outcome.actual_exit_date else None,
                "equity": equity,
                "high_water_mark": high_water_mark,
                "drawdown_pct": drawdown,
            })

        return curve

    def _get_filtered_signals(
        self,
        start_date: Optional[date],
        end_date: Optional[date],
        symbol: Optional[str],
        asset_class: Optional[str],
        signal_type: Optional[str],
    ) -> List[TrackedSignal]:
        """Get signals with filters applied."""
        signals = self.store.get_signals_by_date_range(
            start_date=start_date or date(2020, 1, 1),
            end_date=end_date or date.today(),
            symbol=symbol,
            asset_class=asset_class,
        )

        if signal_type:
            signals = [s for s in signals if s.signal_type == signal_type]

        return signals

    def _get_outcomes_for_signals(
        self,
        signals: List[TrackedSignal],
    ) -> List[SignalOutcome]:
        """Get outcomes for given signals."""
        outcomes = []
        for signal in signals:
            outcome = self.store.get_outcome(signal.id)
            if outcome:
                outcomes.append(outcome)
        return outcomes

    def _get_signal(self, signal_id: str) -> Optional[TrackedSignal]:
        """Get signal by ID (with caching potential)."""
        return self.store.get_signal(signal_id)

    def _calculate_sharpe(
        self,
        returns: List[float],
        annualize: bool = True,
    ) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0

        avg_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        if std_return == 0:
            return 0.0

        # Daily risk-free rate
        daily_rf = self.RISK_FREE_RATE / self.TRADING_DAYS_PER_YEAR

        sharpe = (avg_return - daily_rf) / std_return

        if annualize:
            # Annualize assuming daily returns
            sharpe *= np.sqrt(self.TRADING_DAYS_PER_YEAR)

        return sharpe

    def _calculate_sortino(
        self,
        returns: List[float],
        annualize: bool = True,
    ) -> float:
        """Calculate Sortino ratio (downside deviation only)."""
        if len(returns) < 2:
            return 0.0

        avg_return = np.mean(returns)

        # Downside returns only
        downside = [r for r in returns if r < 0]
        if not downside:
            return float('inf') if avg_return > 0 else 0.0

        downside_std = np.std(downside, ddof=1)

        if downside_std == 0:
            return 0.0

        daily_rf = self.RISK_FREE_RATE / self.TRADING_DAYS_PER_YEAR
        sortino = (avg_return - daily_rf) / downside_std

        if annualize:
            sortino *= np.sqrt(self.TRADING_DAYS_PER_YEAR)

        return sortino

    def _calculate_max_drawdown(
        self,
        returns: List[float],
    ) -> float:
        """Calculate maximum drawdown from returns."""
        if not returns:
            return 0.0

        # Build equity curve
        equity = [1.0]  # Start at 1.0
        for r in returns:
            equity.append(equity[-1] * (1 + r))

        # Calculate running max and drawdown
        running_max = equity[0]
        max_dd = 0.0

        for e in equity:
            running_max = max(running_max, e)
            dd = (e - running_max) / running_max
            max_dd = min(max_dd, dd)

        return max_dd

    def _metrics_by_category(
        self,
        outcomes: List[SignalOutcome],
        category_func,
    ) -> Dict[str, Dict]:
        """Calculate metrics grouped by category."""
        categories = {}

        for outcome in outcomes:
            if not outcome.was_followed:
                continue

            try:
                category = category_func(outcome)
            except Exception:
                continue

            if category not in categories:
                categories[category] = {
                    "returns": [],
                    "r_values": [],
                }

            categories[category]["returns"].append(outcome.return_pct)
            categories[category]["r_values"].append(outcome.r_multiple)

        # Calculate metrics per category
        result = {}
        for cat, data in categories.items():
            wins = sum(1 for r in data["returns"] if r > 0)
            total = len(data["returns"])

            result[cat] = {
                "total": total,
                "win_rate": wins / total if total > 0 else 0,
                "avg_return": np.mean(data["returns"]) if data["returns"] else 0,
                "avg_r": np.mean(data["r_values"]) if data["r_values"] else 0,
            }

        return result
```

**Acceptance Criteria**:
- [ ] All metrics calculate correctly
- [ ] Sharpe/Sortino/Calmar ratios are accurate
- [ ] Max drawdown calculation is correct
- [ ] Metrics by category (asset, type, conviction) work
- [ ] Equity curve generation works

**Tests to Write**:
```python
class TestPerformanceCalculator:
    def test_win_rate_calculation(self):
        """Test win rate is calculated correctly."""
        # 7 wins, 3 losses = 70% win rate
        pass

    def test_expectancy_calculation(self):
        """Test expectancy formula."""
        # Win rate 60%, avg winner 2R, avg loser -1R
        # Expectancy = 0.6 * 2 - 0.4 * 1 = 0.8R
        pass

    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        pass

    def test_max_drawdown(self):
        """Test max drawdown from returns."""
        returns = [0.10, -0.05, -0.08, 0.12, -0.03]
        # Peak at 1.10, drops to 1.10 * 0.95 * 0.92 = 0.961
        # DD = (0.961 - 1.10) / 1.10 = -12.6%
        pass
```

---

## Summary: Part 1 Tasks

| Task | Description | Key Deliverable |
|------|-------------|-----------------|
| 3.1.1 | Module Structure | Data models, directory layout |
| 3.1.2 | Database Schema | SQLite schema, migrations |
| 3.1.3 | Storage Layer | SQLiteTrackingStore implementation |
| 3.1.4 | Signal Tracker | Record and manage signals |
| 3.1.5 | Outcome Recorder | Record trade results, calculate returns |
| 3.1.6 | Performance Calculator | Win rate, expectancy, Sharpe, drawdown |

---

**Part 2 will cover**:
- 3.2.1: Performance Analytics (detailed breakdowns)
- 3.2.2: Weekly Performance Email Template
- 3.2.3: Daily Email Performance Section
- 3.2.4: Performance CLI Commands
- 3.2.5: Strategy Leaderboard
- 3.2.6: Integration with Signal Generation
- 3.2.7: Integration Tests

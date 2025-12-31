"""SQLite implementation of tracking storage."""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

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

logger = logging.getLogger(__name__)


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
                str(self.db_path),
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
            self.connection.execute(
                sql,
                (
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
                ),
            )

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
            cursor = self.connection.execute(
                sql,
                (
                    signal.status.value,
                    signal.delivered_at.isoformat() if signal.delivered_at else None,
                    signal.entry_filled_at.isoformat() if signal.entry_filled_at else None,
                    signal.exit_filled_at.isoformat() if signal.exit_filled_at else None,
                    1 if signal.was_delivered else 0,
                    signal.delivery_method,
                    signal.id,
                ),
            )

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
            sql = """
                UPDATE tracked_signals
                SET status = ?, {timestamp_field} = ?
                WHERE id = ?
            """
            update_params: tuple[str, str, str] = (status.value, timestamp.isoformat(), signal_id)
        else:
            sql = "UPDATE tracked_signals SET status = ? WHERE id = ?"
            update_params_else: tuple[str, str] = (status.value, signal_id)
            update_params = update_params_else

        with self.transaction():
            cursor = self.connection.execute(sql, update_params)

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

        sql = """
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
            self.connection.execute(
                sql,
                (
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
                ),
            )

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
            cursor = self.connection.execute(
                sql,
                (
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
                ),
            )

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

        cursor = self.connection.execute(
            sql,
            (
                start_date.isoformat(),
                end_date.isoformat(),
            ),
        )
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
            self.connection.execute(
                sql,
                (
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
                ),
            )

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

        cursor = self.connection.execute(
            sql,
            (
                start_date.isoformat(),
                end_date.isoformat(),
            ),
        )

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

        sql = """
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

"""Manual trade tracking and storage."""

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..models.positions import ExitReason, Position, PositionSide
from ..models.signals import BreakoutType

logger = logging.getLogger(__name__)


@dataclass
class ManualTrade:
    """Manual trade record (user-managed position)."""

    # Unique identifier
    trade_id: str

    # Position details
    symbol: str
    asset_class: str  # equity or crypto
    side: PositionSide  # LONG or SHORT

    # Entry
    entry_date: datetime
    entry_price: float
    quantity: int

    # Stop management
    stop_price: float
    initial_stop_price: float

    # Exit (optional - None if still open)
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None

    # P&L tracking
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    # User notes
    notes: Optional[str] = None
    tags: Optional[str] = None  # Comma-separated tags

    # Metadata
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        """Set timestamps if not provided."""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

    def is_open(self) -> bool:
        """Check if trade is still open."""
        return self.exit_date is None

    def to_position(self) -> Position:
        """Convert to Position model for unified reporting.

        Returns:
            Position object compatible with backtest/paper positions
        """
        return Position(
            symbol=self.symbol,
            asset_class=self.asset_class,
            entry_date=pd.Timestamp(self.entry_date),
            entry_price=self.entry_price,
            entry_fill_id=f"manual_{self.trade_id}",
            quantity=self.quantity,
            side=self.side,
            stop_price=self.stop_price,
            initial_stop_price=self.initial_stop_price,
            hard_stop_atr_mult=2.5,
            entry_slippage_bps=0.0,
            entry_fee_bps=0.0,
            entry_total_cost=0.0,
            triggered_on=BreakoutType.FAST_20D,
            adv20_at_entry=1000000.0,
            strategy_name="manual",
            exit_date=pd.Timestamp(self.exit_date) if self.exit_date else None,
            exit_price=self.exit_price,
            exit_fill_id=f"manual_exit_{self.trade_id}" if self.exit_date else None,
            exit_reason=ExitReason.MANUAL if self.exit_date else None,
            exit_slippage_bps=0.0 if self.exit_date else None,
            exit_fee_bps=0.0 if self.exit_date else None,
            exit_total_cost=0.0 if self.exit_date else None,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self.unrealized_pnl,
        )


class ManualTradeDatabase:
    """Database for storing and querying manual trades."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize manual trade database.

        Args:
            db_path: Path to SQLite database file (default: results/manual_trades.db)
        """
        if db_path is None:
            db_path = Path("results/manual_trades.db")

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            self._create_schema(conn)
        finally:
            conn.close()

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        """Create manual trades table schema.

        Args:
            conn: SQLite database connection
        """
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS manual_trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                asset_class TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_date TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                stop_price REAL NOT NULL,
                initial_stop_price REAL NOT NULL,
                exit_date TEXT,
                exit_price REAL,
                exit_reason TEXT,
                realized_pnl REAL DEFAULT 0.0,
                unrealized_pnl REAL DEFAULT 0.0,
                notes TEXT,
                tags TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """
        )

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_manual_symbol ON manual_trades(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_manual_entry_date ON manual_trades(entry_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_manual_exit_date ON manual_trades(exit_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_manual_asset_class ON manual_trades(asset_class)")

        conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection.

        Returns:
            SQLite connection
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def create_trade(self, trade: ManualTrade) -> str:
        """Create a new manual trade.

        Args:
            trade: ManualTrade to create

        Returns:
            trade_id of created trade
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO manual_trades
                (trade_id, symbol, asset_class, side, entry_date, entry_price, quantity,
                 stop_price, initial_stop_price, exit_date, exit_price, exit_reason,
                 realized_pnl, unrealized_pnl, notes, tags, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    trade.trade_id,
                    trade.symbol,
                    trade.asset_class,
                    trade.side.value,
                    trade.entry_date.isoformat(),
                    trade.entry_price,
                    trade.quantity,
                    trade.stop_price,
                    trade.initial_stop_price,
                    trade.exit_date.isoformat() if trade.exit_date else None,
                    trade.exit_price,
                    trade.exit_reason,
                    trade.realized_pnl,
                    trade.unrealized_pnl,
                    trade.notes,
                    trade.tags,
                    trade.created_at.isoformat(),
                    trade.updated_at.isoformat(),
                ),
            )

            conn.commit()
            logger.info(f"Created manual trade {trade.trade_id}: {trade.side} {trade.quantity} {trade.symbol}")
            return trade.trade_id

        except Exception as e:
            conn.rollback()
            logger.error(f"Error creating manual trade: {e}", exc_info=True)
            raise
        finally:
            conn.close()

    def update_trade(self, trade: ManualTrade) -> None:
        """Update an existing manual trade.

        Args:
            trade: ManualTrade with updated values
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            trade.updated_at = datetime.now()

            cursor.execute(
                """
                UPDATE manual_trades
                SET symbol = ?, asset_class = ?, side = ?, entry_date = ?, entry_price = ?,
                    quantity = ?, stop_price = ?, initial_stop_price = ?, exit_date = ?,
                    exit_price = ?, exit_reason = ?, realized_pnl = ?, unrealized_pnl = ?,
                    notes = ?, tags = ?, updated_at = ?
                WHERE trade_id = ?
            """,
                (
                    trade.symbol,
                    trade.asset_class,
                    trade.side.value,
                    trade.entry_date.isoformat(),
                    trade.entry_price,
                    trade.quantity,
                    trade.stop_price,
                    trade.initial_stop_price,
                    trade.exit_date.isoformat() if trade.exit_date else None,
                    trade.exit_price,
                    trade.exit_reason,
                    trade.realized_pnl,
                    trade.unrealized_pnl,
                    trade.notes,
                    trade.tags,
                    trade.updated_at.isoformat(),
                    trade.trade_id,
                ),
            )

            conn.commit()
            logger.info(f"Updated manual trade {trade.trade_id}")

        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating manual trade: {e}", exc_info=True)
            raise
        finally:
            conn.close()

    def close_trade(self, trade_id: str, exit_date: datetime, exit_price: float, exit_reason: str = "manual") -> None:
        """Close a manual trade.

        Args:
            trade_id: Trade ID to close
            exit_date: Exit date
            exit_price: Exit price
            exit_reason: Reason for exit
        """
        trade = self.get_trade(trade_id)
        if not trade:
            raise ValueError(f"Trade {trade_id} not found")

        if not trade.is_open():
            raise ValueError(f"Trade {trade_id} is already closed")

        # Calculate realized P&L
        if trade.side == PositionSide.LONG:
            price_pnl = (exit_price - trade.entry_price) * trade.quantity
        else:  # SHORT
            price_pnl = (trade.entry_price - exit_price) * trade.quantity

        trade.exit_date = exit_date
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.realized_pnl = price_pnl
        trade.unrealized_pnl = 0.0

        self.update_trade(trade)
        logger.info(f"Closed manual trade {trade_id}: realized P&L = {price_pnl:.2f}")

    def delete_trade(self, trade_id: str) -> None:
        """Delete a manual trade.

        Args:
            trade_id: Trade ID to delete
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM manual_trades WHERE trade_id = ?", (trade_id,))
            conn.commit()
            logger.info(f"Deleted manual trade {trade_id}")

        except Exception as e:
            conn.rollback()
            logger.error(f"Error deleting manual trade: {e}", exc_info=True)
            raise
        finally:
            conn.close()

    def get_trade(self, trade_id: str) -> Optional[ManualTrade]:
        """Get a manual trade by ID.

        Args:
            trade_id: Trade ID to retrieve

        Returns:
            ManualTrade if found, None otherwise
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT * FROM manual_trades WHERE trade_id = ?", (trade_id,))
            row = cursor.fetchone()

            if row:
                return self._row_to_trade(row)
            return None

        finally:
            conn.close()

    def get_open_trades(self) -> List[ManualTrade]:
        """Get all open manual trades.

        Returns:
            List of open ManualTrade objects
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT * FROM manual_trades WHERE exit_date IS NULL ORDER BY entry_date DESC")
            rows = cursor.fetchall()
            return [self._row_to_trade(row) for row in rows]

        finally:
            conn.close()

    def get_closed_trades(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> List[ManualTrade]:
        """Get closed manual trades within date range.

        Args:
            start_date: Start date filter (inclusive)
            end_date: End date filter (inclusive)

        Returns:
            List of closed ManualTrade objects
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            query = "SELECT * FROM manual_trades WHERE exit_date IS NOT NULL"
            params = []

            if start_date:
                query += " AND exit_date >= ?"
                params.append(start_date.isoformat())

            if end_date:
                query += " AND exit_date <= ?"
                params.append(end_date.isoformat())

            query += " ORDER BY exit_date DESC"

            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [self._row_to_trade(row) for row in rows]

        finally:
            conn.close()

    def get_all_trades(self) -> List[ManualTrade]:
        """Get all manual trades (open and closed).

        Returns:
            List of all ManualTrade objects
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT * FROM manual_trades ORDER BY entry_date DESC")
            rows = cursor.fetchall()
            return [self._row_to_trade(row) for row in rows]

        finally:
            conn.close()

    def update_unrealized_pnl(self, trade_id: str, current_price: float) -> None:
        """Update unrealized P&L for an open trade.

        Args:
            trade_id: Trade ID to update
            current_price: Current market price
        """
        trade = self.get_trade(trade_id)
        if not trade:
            raise ValueError(f"Trade {trade_id} not found")

        if not trade.is_open():
            logger.warning(f"Trade {trade_id} is closed, skipping unrealized P&L update")
            return

        # Calculate unrealized P&L
        if trade.side == PositionSide.LONG:
            price_pnl = (current_price - trade.entry_price) * trade.quantity
        else:  # SHORT
            price_pnl = (trade.entry_price - current_price) * trade.quantity

        trade.unrealized_pnl = price_pnl
        self.update_trade(trade)

    def _row_to_trade(self, row: sqlite3.Row) -> ManualTrade:
        """Convert database row to ManualTrade object.

        Args:
            row: Database row

        Returns:
            ManualTrade object
        """
        return ManualTrade(
            trade_id=row["trade_id"],
            symbol=row["symbol"],
            asset_class=row["asset_class"],
            side=PositionSide(row["side"]),
            entry_date=datetime.fromisoformat(row["entry_date"]),
            entry_price=row["entry_price"],
            quantity=row["quantity"],
            stop_price=row["stop_price"],
            initial_stop_price=row["initial_stop_price"],
            exit_date=datetime.fromisoformat(row["exit_date"]) if row["exit_date"] else None,
            exit_price=row["exit_price"],
            exit_reason=row["exit_reason"],
            realized_pnl=row["realized_pnl"],
            unrealized_pnl=row["unrealized_pnl"],
            notes=row["notes"],
            tags=row["tags"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

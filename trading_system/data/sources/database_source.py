"""Database data source implementations (PostgreSQL, SQLite)."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..validator import validate_ohlcv
from .base_source import BaseDataSource

logger = logging.getLogger(__name__)


class DatabaseDataSource(BaseDataSource):
    """Base class for database data sources.

    Expected table structure:
    - Table with columns: symbol, date, open, high, low, close, volume
    - Primary key: (symbol, date)
    """

    def __init__(self, connection_string: str, table_name: str = "ohlcv_data"):
        """Initialize database data source.

        Args:
            connection_string: Database connection string
            table_name: Name of the OHLCV table (must be a valid SQL identifier)
        """
        self.connection_string = connection_string
        self._validate_table_name(table_name)
        self.table_name = table_name
        self._connection: Optional[Any] = None

    @staticmethod
    def _validate_table_name(table_name: str) -> None:
        """Validate that table name is a safe SQL identifier.

        Args:
            table_name: Table name to validate

        Raises:
            ValueError: If table name contains unsafe characters
        """
        # Only allow alphanumeric characters and underscores
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table_name):
            raise ValueError(
                f"Invalid table name: {table_name}. Must be a valid SQL identifier (alphanumeric and underscores only)"
            )

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        """Quote a SQL identifier safely.

        Args:
            identifier: SQL identifier to quote (table/column name)

        Returns:
            Properly quoted identifier (works for both SQLite and PostgreSQL)
        """
        # Double quotes work for both SQLite and PostgreSQL
        # Identifier is already validated to be safe, but quoting adds extra safety
        return f'"{identifier}"'

    def _get_connection(self):
        """Get database connection (implemented by subclasses)."""
        raise NotImplementedError("Subclass must implement _get_connection")

    def _get_param_style(self) -> str:
        """Get SQL parameter style for this database type.

        Returns:
            Parameter placeholder string ('?' for SQLite, '%s' for PostgreSQL)
        """
        # Default to '?' (SQLite style), subclasses can override
        return "?"

    def load_ohlcv(
        self, symbols: List[str], start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None
    ) -> Dict[str, pd.DataFrame]:
        """Load OHLCV data from database.

        Args:
            symbols: List of symbols to load
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Dictionary mapping symbol -> DataFrame
        """
        if not symbols:
            return {}

        conn = self._get_connection()

        # Get parameter style for this database
        param_style = self._get_param_style()

        # Build query with proper parameter placeholders
        placeholders = ",".join([param_style for _ in symbols])
        # table_name is validated and quoted for safety
        quoted_table = self._quote_identifier(self.table_name)
        query = f"""
            SELECT symbol, date, open, high, low, close, volume
            FROM {quoted_table}
            WHERE symbol IN ({placeholders})
        """  # nosec B608 - quoted_table is validated/quoted identifier, symbols are parameterized
        params = list(symbols)

        if start_date is not None:
            query += f" AND date >= {param_style}"
            # Convert to string for SQLite compatibility
            params.append(start_date.strftime("%Y-%m-%d") if hasattr(start_date, "strftime") else str(start_date))

        if end_date is not None:
            query += f" AND date <= {param_style}"
            # Convert to string for SQLite compatibility
            params.append(end_date.strftime("%Y-%m-%d") if hasattr(end_date, "strftime") else str(end_date))

        query += " ORDER BY symbol, date"

        try:
            # Load data
            df = pd.read_sql_query(query, conn, params=params, parse_dates=["date"])

            if df.empty:
                return {}

            # Compute dollar_volume
            df["dollar_volume"] = df["close"] * df["volume"]

            # Group by symbol and convert to dict
            data = {}
            for symbol in symbols:
                symbol_df = df[df["symbol"] == symbol].copy()
                if symbol_df.empty:
                    continue

                # Set date as index
                symbol_df.set_index("date", inplace=True)
                symbol_df.sort_index(inplace=True)

                # Remove symbol column
                symbol_df.drop("symbol", axis=1, inplace=True)

                # Validate
                if validate_ohlcv(symbol_df, symbol):
                    data[symbol] = symbol_df
                else:
                    logger.error(f"Validation failed for {symbol}, skipping")

        except Exception as e:
            logger.error(f"Error loading data from database: {e}")
            return {}

        return data

    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols from database.

        Returns:
            List of symbols
        """
        conn = self._get_connection()

        try:
            # table_name is validated and quoted for safety
            # The table name is validated in __init__ via _validate_table_name()
            # to only contain alphanumeric characters and underscores, then properly
            # quoted via _quote_identifier(). This is safe for SQL identifier usage.
            quoted_table = self._quote_identifier(self.table_name)
            query = f"SELECT DISTINCT symbol FROM {quoted_table} ORDER BY symbol"  # nosec B608
            df = pd.read_sql_query(query, conn)
            return [str(s) for s in df["symbol"].tolist()]
        except Exception as e:
            logger.error(f"Error getting available symbols: {e}")
            return []

    def get_date_range(self, symbol: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Get available date range for a symbol.

        Args:
            symbol: Symbol to check

        Returns:
            Tuple of (start_date, end_date) or None if symbol not available
        """
        conn = self._get_connection()

        param_style = self._get_param_style()
        try:
            # table_name is validated and quoted for safety
            quoted_table = self._quote_identifier(self.table_name)
            query = f"""
                SELECT MIN(date) as min_date, MAX(date) as max_date
                FROM {quoted_table}
                WHERE symbol = {param_style}
            """  # nosec B608 - quoted_table is validated/quoted identifier, symbol is parameterized
            df = pd.read_sql_query(query, conn, params=[symbol], parse_dates=["min_date", "max_date"])

            if df.empty or df.iloc[0]["min_date"] is None:
                return None

            return (pd.Timestamp(df.iloc[0]["min_date"]), pd.Timestamp(df.iloc[0]["max_date"]))
        except Exception as e:
            logger.error(f"Error getting date range for {symbol}: {e}")
            return None

    def supports_incremental(self) -> bool:
        """Database sources support incremental loading."""
        return True

    def load_incremental(self, symbol: str, last_update_date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Load data incrementally since last update.

        Args:
            symbol: Symbol to load
            last_update_date: Last known date (exclusive)

        Returns:
            DataFrame with new data or None if no new data
        """
        data = self.load_ohlcv([symbol], start_date=last_update_date + pd.Timedelta(days=1))
        if symbol in data:
            return data[symbol]
        return None


class SQLiteSource(DatabaseDataSource):
    """SQLite database data source."""

    def __init__(self, db_path: str, table_name: str = "ohlcv_data"):
        """Initialize SQLite source.

        Args:
            db_path: Path to SQLite database file
            table_name: Name of the OHLCV table
        """
        self.db_path = db_path
        self._connection: Optional[Any] = None
        super().__init__(f"sqlite:///{db_path}", table_name)

    def _get_connection(self):
        """Get SQLite connection."""
        try:
            import sqlite3

            if self._connection is None:
                self._connection = sqlite3.connect(self.db_path)
            return self._connection
        except ImportError:
            raise ImportError("sqlite3 module not available")


class PostgreSQLSource(DatabaseDataSource):
    """PostgreSQL database data source."""

    def __init__(self, host: str, port: int, database: str, user: str, password: str, table_name: str = "ohlcv_data"):
        """Initialize PostgreSQL source.

        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Username
            password: Password
            table_name: Name of the OHLCV table
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self._connection: Optional[Any] = None
        super().__init__(f"postgresql://{user}:{password}@{host}:{port}/{database}", table_name)

    def _get_connection(self):
        """Get PostgreSQL connection."""
        try:
            import psycopg2

            if self._connection is None:
                self._connection = psycopg2.connect(
                    host=self.host, port=self.port, database=self.database, user=self.user, password=self.password
                )
            return self._connection
        except ImportError:
            raise ImportError("psycopg2 module not available. Install with: pip install psycopg2-binary")

    def _get_param_style(self) -> str:
        """PostgreSQL uses %s for parameters."""
        return "%s"

"""Unit tests for storage/schema.py - database schema creation."""

import shutil
import sqlite3
import tempfile
from pathlib import Path

import pytest

from trading_system.storage.schema import create_schema, get_schema_version, migrate_schema


class TestCreateSchema:
    """Tests for create_schema function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db_path = self.temp_dir / "test_schema.db"

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, "temp_dir") and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_create_schema_creates_tables(self):
        """Test that create_schema creates all required tables."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            create_schema(conn)

            cursor = conn.cursor()

            # Check that all tables exist
            cursor.execute(
                """
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """
            )
            tables = [row[0] for row in cursor.fetchall()]

            expected_tables = ["backtest_runs", "daily_returns", "equity_curve", "monthly_summary", "run_metrics", "trades"]

            for table in expected_tables:
                assert table in tables, f"Table {table} not found"
        finally:
            conn.close()

    def test_create_schema_backtest_runs_structure(self):
        """Test backtest_runs table structure."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            create_schema(conn)

            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(backtest_runs)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}

            # Check required columns
            assert "run_id" in columns
            assert columns["run_id"] == "INTEGER"
            assert "config_path" in columns
            assert "strategy_name" in columns
            assert "split_name" in columns
            assert "period" in columns
            assert columns["period"] == "TEXT"
            assert "start_date" in columns
            assert "end_date" in columns
            assert "starting_equity" in columns
            assert columns["starting_equity"] == "REAL"
            assert "created_at" in columns
            assert "notes" in columns
        finally:
            conn.close()

    def test_create_schema_run_metrics_structure(self):
        """Test run_metrics table structure."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            create_schema(conn)

            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(run_metrics)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}

            # Check key metric columns
            assert "metric_id" in columns
            assert "run_id" in columns
            assert "sharpe_ratio" in columns
            assert columns["sharpe_ratio"] == "REAL"
            assert "max_drawdown" in columns
            assert "total_return" in columns
            assert "total_trades" in columns
            assert columns["total_trades"] == "INTEGER"
            assert "win_rate" in columns
            assert "calmar_ratio" in columns
            assert "profit_factor" in columns
        finally:
            conn.close()

    def test_create_schema_trades_structure(self):
        """Test trades table structure."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            create_schema(conn)

            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(trades)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}

            # Check key trade columns
            assert "trade_id" in columns
            assert "run_id" in columns
            assert "symbol" in columns
            assert columns["symbol"] == "TEXT"
            assert "asset_class" in columns
            assert "entry_date" in columns
            assert "exit_date" in columns
            assert "entry_price" in columns
            assert columns["entry_price"] == "REAL"
            assert "exit_price" in columns
            assert "quantity" in columns
            assert columns["quantity"] == "INTEGER"
            assert "realized_pnl" in columns
            assert "r_multiple" in columns
        finally:
            conn.close()

    def test_create_schema_equity_curve_structure(self):
        """Test equity_curve table structure."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            create_schema(conn)

            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(equity_curve)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}

            # Check key columns
            assert "equity_id" in columns
            assert "run_id" in columns
            assert "date" in columns
            assert "equity" in columns
            assert columns["equity"] == "REAL"
            assert "cash" in columns
            assert "open_positions" in columns
            assert columns["open_positions"] == "INTEGER"
            assert "gross_exposure" in columns
        finally:
            conn.close()

    def test_create_schema_daily_returns_structure(self):
        """Test daily_returns table structure."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            create_schema(conn)

            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(daily_returns)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}

            # Check key columns
            assert "return_id" in columns
            assert "run_id" in columns
            assert "date" in columns
            assert "daily_return" in columns
            assert columns["daily_return"] == "REAL"
        finally:
            conn.close()

    def test_create_schema_monthly_summary_structure(self):
        """Test monthly_summary table structure."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            create_schema(conn)

            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(monthly_summary)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}

            # Check key columns
            assert "monthly_id" in columns
            assert "run_id" in columns
            assert "month" in columns
            assert columns["month"] == "TEXT"
            assert "month_start" in columns
            assert "month_end" in columns
            assert "start_equity" in columns
            assert columns["start_equity"] == "REAL"
            assert "end_equity" in columns
            assert "monthly_return" in columns
            assert columns["monthly_return"] == "REAL"
            assert "trades_count" in columns
            assert "win_rate" in columns
            assert "volatility_annualized" in columns
            assert "sharpe_ratio" in columns
        finally:
            conn.close()

    def test_create_schema_creates_indexes(self):
        """Test that create_schema creates indexes."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            create_schema(conn)

            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT name FROM sqlite_master 
                WHERE type='index' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """
            )
            indexes = [row[0] for row in cursor.fetchall()]

            # Check for key indexes
            expected_indexes = [
                "idx_equity_date",
                "idx_equity_run",
                "idx_monthly_month",
                "idx_monthly_run",
                "idx_returns_run",
                "idx_runs_config",
                "idx_runs_dates",
                "idx_runs_period",
                "idx_runs_strategy",
                "idx_trades_run",
                "idx_trades_symbol",
            ]

            for idx in expected_indexes:
                assert idx in indexes, f"Index {idx} not found"
        finally:
            conn.close()

    def test_create_schema_idempotent(self):
        """Test that create_schema can be called multiple times safely."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            # Call create_schema multiple times
            create_schema(conn)
            create_schema(conn)
            create_schema(conn)

            # Should still have all tables
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """
            )
            tables = [row[0] for row in cursor.fetchall()]

            assert len(tables) == 6  # All 6 tables should exist
        finally:
            conn.close()

    def test_create_schema_foreign_keys(self):
        """Test that foreign key constraints are set up correctly."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            create_schema(conn)

            # Enable foreign key checks
            conn.execute("PRAGMA foreign_keys = ON")

            # Insert a backtest run
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO backtest_runs 
                (config_path, strategy_name, period, start_date, end_date, starting_equity)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                ("test.yaml", "test_strategy", "train", "2023-01-01", "2023-12-31", 100000.0),
            )
            run_id = cursor.lastrowid
            conn.commit()

            # Insert metrics with valid run_id (should succeed)
            cursor.execute(
                """
                INSERT INTO run_metrics (run_id, sharpe_ratio, max_drawdown, total_return)
                VALUES (?, ?, ?, ?)
            """,
                (run_id, 1.5, 0.15, 0.25),
            )
            conn.commit()

            # Try to insert metrics with invalid run_id (should fail if foreign keys are enforced)
            # SQLite foreign keys need to be enabled with PRAGMA foreign_keys = ON
            with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
                cursor.execute(
                    """
                    INSERT INTO run_metrics (run_id, sharpe_ratio, max_drawdown, total_return)
                    VALUES (?, ?, ?, ?)
                """,
                    (99999, 1.5, 0.15, 0.25),
                )
                conn.commit()
        finally:
            conn.close()

    def test_create_schema_unique_constraints(self):
        """Test that unique constraints are set up correctly."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            create_schema(conn)

            cursor = conn.cursor()

            # Insert a backtest run
            cursor.execute(
                """
                INSERT INTO backtest_runs 
                (config_path, strategy_name, split_name, period, start_date, end_date, starting_equity)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                ("test.yaml", "test_strategy", "split_1", "train", "2023-01-01", "2023-12-31", 100000.0),
            )
            conn.commit()

            # Try to insert duplicate (should fail due to UNIQUE constraint)
            with pytest.raises(sqlite3.IntegrityError):
                cursor.execute(
                    """
                    INSERT INTO backtest_runs 
                    (config_path, strategy_name, split_name, period, start_date, end_date, starting_equity)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    ("test.yaml", "test_strategy", "split_1", "train", "2023-01-01", "2023-12-31", 100000.0),
                )
                conn.commit()
        finally:
            conn.close()


class TestGetSchemaVersion:
    """Tests for get_schema_version function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db_path = self.temp_dir / "test_schema_version.db"

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, "temp_dir") and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_get_schema_version_no_table(self):
        """Test get_schema_version when schema_version table doesn't exist."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            # Don't create schema_version table
            version = get_schema_version(conn)
            # Should return default version 1
            assert version == 1
        finally:
            conn.close()

    def test_get_schema_version_with_table(self):
        """Test get_schema_version when schema_version table exists."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            # Create schema_version table manually
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
            """
            )
            cursor.execute("INSERT INTO schema_version (version) VALUES (2)")
            conn.commit()

            version = get_schema_version(conn)
            assert version == 2
        finally:
            conn.close()


class TestMigrateSchema:
    """Tests for migrate_schema function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db_path = self.temp_dir / "test_migrate.db"

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, "temp_dir") and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_migrate_schema_current_version(self):
        """Test migrate_schema with current version (no migration needed)."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            # Migrate from version 1 to 1 (no-op)
            migrate_schema(conn, from_version=1, to_version=1)
            # Should not raise any errors
        finally:
            conn.close()

    def test_migrate_schema_future_version(self):
        """Test migrate_schema placeholder for future migrations."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            # Currently only version 1 exists, so migration to version 2 is a placeholder
            migrate_schema(conn, from_version=1, to_version=2)
            # Should not raise any errors (placeholder implementation)
        finally:
            conn.close()

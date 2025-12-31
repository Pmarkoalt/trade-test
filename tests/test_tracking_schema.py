"""Unit tests for tracking database schema."""

import sqlite3
from pathlib import Path

import pytest

from trading_system.tracking.storage.schema import TABLES, get_migration_files


class TestSchemaCreation:
    """Tests for schema creation."""

    def test_schema_creates_tables(self, tmp_path):
        """Test schema creation."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        # Run migration
        migration_file = (
            Path(__file__).parent.parent / "trading_system" / "tracking" / "storage" / "migrations" / "001_initial_schema.sql"
        )
        with open(migration_file) as f:
            conn.executescript(f.read())

        # Verify tables exist
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        assert "tracked_signals" in tables
        assert "signal_outcomes" in tables
        assert "daily_performance" in tables
        assert "strategy_performance" in tables
        assert "schema_migrations" in tables

        conn.close()

    def test_tracked_signals_table_structure(self, tmp_path):
        """Test tracked_signals table has correct columns."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        migration_file = (
            Path(__file__).parent.parent / "trading_system" / "tracking" / "storage" / "migrations" / "001_initial_schema.sql"
        )
        with open(migration_file) as f:
            conn.executescript(f.read())

        cursor = conn.execute("PRAGMA table_info(tracked_signals)")
        columns = {row[1] for row in cursor.fetchall()}

        expected_columns = set(TABLES["tracked_signals"])

        # Check that all expected columns exist
        missing = expected_columns - columns
        assert not missing, f"Missing columns: {missing}"

        # Check that generated column created_date exists (by trying to query it)
        # PRAGMA table_info may not show generated columns in some SQLite versions
        try:
            conn.execute("SELECT created_date FROM tracked_signals LIMIT 1")
            generated_column_exists = True
        except sqlite3.OperationalError:
            generated_column_exists = False

        assert generated_column_exists, "Generated column 'created_date' does not exist or is not queryable"

        conn.close()

    def test_signal_outcomes_table_structure(self, tmp_path):
        """Test signal_outcomes table has correct columns."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        migration_file = (
            Path(__file__).parent.parent / "trading_system" / "tracking" / "storage" / "migrations" / "001_initial_schema.sql"
        )
        with open(migration_file) as f:
            conn.executescript(f.read())

        cursor = conn.execute("PRAGMA table_info(signal_outcomes)")
        columns = {row[1] for row in cursor.fetchall()}

        expected_columns = set(TABLES["signal_outcomes"])

        assert expected_columns.issubset(columns), f"Missing columns: {expected_columns - columns}"

        conn.close()

    def test_daily_performance_table_structure(self, tmp_path):
        """Test daily_performance table has correct columns."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        migration_file = (
            Path(__file__).parent.parent / "trading_system" / "tracking" / "storage" / "migrations" / "001_initial_schema.sql"
        )
        with open(migration_file) as f:
            conn.executescript(f.read())

        cursor = conn.execute("PRAGMA table_info(daily_performance)")
        columns = {row[1] for row in cursor.fetchall()}

        expected_columns = set(TABLES["daily_performance"])

        assert expected_columns.issubset(columns), f"Missing columns: {expected_columns - columns}"

        conn.close()

    def test_strategy_performance_table_structure(self, tmp_path):
        """Test strategy_performance table has correct columns."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        migration_file = (
            Path(__file__).parent.parent / "trading_system" / "tracking" / "storage" / "migrations" / "001_initial_schema.sql"
        )
        with open(migration_file) as f:
            conn.executescript(f.read())

        cursor = conn.execute("PRAGMA table_info(strategy_performance)")
        columns = {row[1] for row in cursor.fetchall()}

        expected_columns = set(TABLES["strategy_performance"])

        assert expected_columns.issubset(columns), f"Missing columns: {expected_columns - columns}"

        conn.close()

    def test_indexes_created(self, tmp_path):
        """Test that indexes are created."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        migration_file = (
            Path(__file__).parent.parent / "trading_system" / "tracking" / "storage" / "migrations" / "001_initial_schema.sql"
        )
        with open(migration_file) as f:
            conn.executescript(f.read())

        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'")
        indexes = {row[0] for row in cursor.fetchall()}

        expected_indexes = {
            "idx_signals_symbol",
            "idx_signals_created_date",
            "idx_signals_status",
            "idx_signals_asset_class",
            "idx_outcomes_signal_id",
            "idx_outcomes_exit_date",
            "idx_daily_perf_date",
        }

        assert expected_indexes.issubset(indexes), f"Missing indexes: {expected_indexes - indexes}"

        conn.close()

    def test_foreign_key_constraint(self, tmp_path):
        """Test foreign key constraint between signal_outcomes and tracked_signals."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        # Enable foreign key constraints
        conn.execute("PRAGMA foreign_keys = ON")

        migration_file = (
            Path(__file__).parent.parent / "trading_system" / "tracking" / "storage" / "migrations" / "001_initial_schema.sql"
        )
        with open(migration_file) as f:
            conn.executescript(f.read())

        # Try to insert outcome without signal - should fail
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO signal_outcomes (signal_id, recorded_at)
                VALUES ('non-existent-id', datetime('now'))
                """
            )
            conn.commit()

        conn.close()

    def test_migration_version_recorded(self, tmp_path):
        """Test that migration version is recorded."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        migration_file = (
            Path(__file__).parent.parent / "trading_system" / "tracking" / "storage" / "migrations" / "001_initial_schema.sql"
        )
        with open(migration_file) as f:
            conn.executescript(f.read())

        cursor = conn.execute("SELECT version, applied_at FROM schema_migrations")
        rows = cursor.fetchall()

        assert len(rows) == 1
        assert rows[0][0] == 1
        assert rows[0][1] is not None

        conn.close()

    def test_unique_constraints(self, tmp_path):
        """Test unique constraints."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        migration_file = (
            Path(__file__).parent.parent / "trading_system" / "tracking" / "storage" / "migrations" / "001_initial_schema.sql"
        )
        with open(migration_file) as f:
            conn.executescript(f.read())

        # Test signal_outcomes.signal_id is unique
        conn.execute(
            """
            INSERT INTO tracked_signals (id, symbol, asset_class, direction, signal_type, conviction,
                                        signal_price, entry_price, target_price, stop_price, status, created_at)
            VALUES ('test-1', 'AAPL', 'equity', 'BUY', 'breakout', 'HIGH', 150.0, 150.0, 165.0, 145.0, 'pending', datetime('now'))
            """
        )

        conn.execute(
            """
            INSERT INTO signal_outcomes (signal_id, recorded_at)
            VALUES ('test-1', datetime('now'))
            """
        )
        conn.commit()

        # Try to insert duplicate signal_id - should fail
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO signal_outcomes (signal_id, recorded_at)
                VALUES ('test-1', datetime('now'))
                """
            )
            conn.commit()

        conn.close()

    def test_default_values(self, tmp_path):
        """Test default values are applied."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))

        migration_file = (
            Path(__file__).parent.parent / "trading_system" / "tracking" / "storage" / "migrations" / "001_initial_schema.sql"
        )
        with open(migration_file) as f:
            conn.executescript(f.read())

        # Insert signal with minimal fields
        conn.execute(
            """
            INSERT INTO tracked_signals (id, symbol, asset_class, direction, signal_type, conviction,
                                        signal_price, entry_price, target_price, stop_price, created_at)
            VALUES ('test-2', 'AAPL', 'equity', 'BUY', 'breakout', 'HIGH', 150.0, 150.0, 165.0, 145.0, datetime('now'))
            """
        )
        conn.commit()

        cursor = conn.execute("SELECT status, was_delivered FROM tracked_signals WHERE id = 'test-2'")
        row = cursor.fetchone()

        assert row[0] == "pending"  # Default status
        assert row[1] == 0  # Default was_delivered

        conn.close()


class TestSchemaModule:
    """Tests for schema.py module."""

    def test_get_migration_files(self):
        """Test get_migration_files returns ordered list."""
        migrations = get_migration_files()

        assert len(migrations) > 0
        assert all(isinstance(m, tuple) and len(m) == 2 for m in migrations)
        assert all(isinstance(version, int) for version, _ in migrations)
        assert all(isinstance(path, str) for _, path in migrations)

        # Check ordering
        versions = [version for version, _ in migrations]
        assert versions == sorted(versions)

        # Check that 001_initial_schema.sql is included
        paths = [path for _, path in migrations]
        assert any("001_initial_schema.sql" in path for path in paths)

    def test_tables_definition(self):
        """Test TABLES dictionary is properly defined."""
        assert "tracked_signals" in TABLES
        assert "signal_outcomes" in TABLES
        assert "daily_performance" in TABLES
        assert "strategy_performance" in TABLES

        # Check that all tables have columns
        for table_name, columns in TABLES.items():
            assert len(columns) > 0, f"Table {table_name} has no columns"
            assert all(isinstance(col, str) for col in columns), f"Table {table_name} has non-string column names"

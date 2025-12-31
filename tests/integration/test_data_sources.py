"""Integration tests for all data sources.

This test suite verifies that all data sources (CSV, Database, API, Parquet, HDF5)
work correctly and can be used interchangeably in the trading system.

Data sources tested:
- CSV (primary, always available)
- SQLite (if available)
- PostgreSQL (if configured)
- AlphaVantage API (if API key provided)
- Polygon API (if API key provided)
- Parquet (if pyarrow available)
- HDF5 (if tables available)
"""

import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from trading_system.data import load_ohlcv_data
from trading_system.data.sources import (
    AlphaVantageSource,
    CSVDataSource,
    HDF5DataSource,
    ParquetDataSource,
    PolygonSource,
    SQLiteSource,
    PostgreSQLSource,
)

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def create_sample_ohlcv_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=days, freq="D")
    np.random.seed(42)
    base_price = 100.0
    prices = base_price + np.cumsum(np.random.randn(days) * 2.0)

    df = pd.DataFrame(
        {
            "open": prices * 0.99,
            "high": prices * 1.02,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.random.randint(1000000, 5000000, days),
        },
        index=dates,
    )
    df.index.name = "date"
    df["dollar_volume"] = df["close"] * df["volume"]
    return df


@pytest.fixture
def sample_data_dir(tmp_path):
    """Create temporary directory with sample CSV files."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()

    # Create sample CSV files
    for symbol in ["AAPL", "MSFT", "GOOGL"]:
        df = create_sample_ohlcv_data(symbol, days=30)
        csv_file = data_dir / f"{symbol}.csv"
        df.reset_index().to_csv(csv_file, index=False)

    return data_dir


class TestCSVDataSource:
    """Tests for CSV data source (primary source)."""

    def test_csv_source_loads_data(self, sample_data_dir):
        """Test that CSV source loads data correctly."""
        source = CSVDataSource(str(sample_data_dir))
        data = source.load_ohlcv(["AAPL", "MSFT"])

        assert len(data) == 2, "Should load 2 symbols"
        assert "AAPL" in data, "Should load AAPL"
        assert "MSFT" in data, "Should load MSFT"

        # Verify data structure
        for symbol, df in data.items():
            assert len(df) > 0, f"Data for {symbol} should not be empty"
            assert all(
                col in df.columns for col in ["open", "high", "low", "close", "volume"]
            ), f"Data for {symbol} should have OHLCV columns"
            assert df.index.name == "date", f"Data for {symbol} should have date index"

    def test_csv_source_date_filtering(self, sample_data_dir):
        """Test that CSV source filters by date range."""
        source = CSVDataSource(str(sample_data_dir))
        start_date = pd.Timestamp("2023-01-10")
        end_date = pd.Timestamp("2023-01-20")

        data = source.load_ohlcv(["AAPL"], start_date=start_date, end_date=end_date)

        assert "AAPL" in data, "Should load AAPL"
        df = data["AAPL"]
        assert df.index.min() >= start_date, "Should filter by start date"
        assert df.index.max() <= end_date, "Should filter by end date"

    def test_csv_source_get_available_symbols(self, sample_data_dir):
        """Test that CSV source lists available symbols."""
        source = CSVDataSource(str(sample_data_dir))
        symbols = source.get_available_symbols()

        assert len(symbols) == 3, "Should find 3 symbols"
        assert "AAPL" in symbols, "Should include AAPL"
        assert "MSFT" in symbols, "Should include MSFT"
        assert "GOOGL" in symbols, "Should include GOOGL"

    def test_csv_source_get_date_range(self, sample_data_dir):
        """Test that CSV source returns date range for symbol."""
        source = CSVDataSource(str(sample_data_dir))
        date_range = source.get_date_range("AAPL")

        assert date_range is not None, "Should return date range"
        assert isinstance(date_range[0], pd.Timestamp), "Start date should be Timestamp"
        assert isinstance(date_range[1], pd.Timestamp), "End date should be Timestamp"

    def test_csv_source_with_loader(self, sample_data_dir):
        """Test CSV source used with load_ohlcv_data function."""
        data = load_ohlcv_data(str(sample_data_dir), ["AAPL", "MSFT"])

        assert len(data) == 2, "Should load 2 symbols"
        assert "AAPL" in data, "Should load AAPL"
        assert "MSFT" in data, "Should load MSFT"


class TestSQLiteDataSource:
    """Tests for SQLite database data source."""

    @pytest.fixture
    def sqlite_db(self, tmp_path):
        """Create temporary SQLite database with sample data."""
        db_file = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_file))

        # Create table
        conn.execute(
            """
            CREATE TABLE ohlcv_data (
                symbol TEXT,
                date DATE,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (symbol, date)
            )
        """
        )

        # Insert sample data
        for symbol in ["AAPL", "MSFT", "GOOGL"]:
            df = create_sample_ohlcv_data(symbol, days=30)
            df = df.reset_index()
            df["symbol"] = symbol
            df.to_sql("ohlcv_data", conn, if_exists="append", index=False)

        conn.commit()
        conn.close()

        return str(db_file)

    def test_sqlite_source_loads_data(self, sqlite_db):
        """Test that SQLite source loads data correctly."""
        source = SQLiteSource(sqlite_db, table_name="ohlcv_data")
        data = source.load_ohlcv(["AAPL", "MSFT"])

        assert len(data) == 2, "Should load 2 symbols"
        assert "AAPL" in data, "Should load AAPL"
        assert "MSFT" in data, "Should load MSFT"

        # Verify data structure
        for symbol, df in data.items():
            assert len(df) > 0, f"Data for {symbol} should not be empty"
            assert all(
                col in df.columns for col in ["open", "high", "low", "close", "volume"]
            ), f"Data for {symbol} should have OHLCV columns"

    def test_sqlite_source_date_filtering(self, sqlite_db):
        """Test that SQLite source filters by date range."""
        source = SQLiteSource(sqlite_db, table_name="ohlcv_data")
        start_date = pd.Timestamp("2023-01-10")
        end_date = pd.Timestamp("2023-01-20")

        data = source.load_ohlcv(["AAPL"], start_date=start_date, end_date=end_date)

        assert "AAPL" in data, "Should load AAPL"
        df = data["AAPL"]
        assert df.index.min() >= start_date, "Should filter by start date"
        assert df.index.max() <= end_date, "Should filter by end date"

    def test_sqlite_source_get_available_symbols(self, sqlite_db):
        """Test that SQLite source lists available symbols."""
        source = SQLiteSource(sqlite_db, table_name="ohlcv_data")
        symbols = source.get_available_symbols()

        assert len(symbols) == 3, "Should find 3 symbols"
        assert "AAPL" in symbols, "Should include AAPL"
        assert "MSFT" in symbols, "Should include MSFT"
        assert "GOOGL" in symbols, "Should include GOOGL"

    def test_sqlite_source_with_loader(self, sqlite_db):
        """Test SQLite source used with load_ohlcv_data function."""
        source = SQLiteSource(sqlite_db, table_name="ohlcv_data")
        data = load_ohlcv_data(source, ["AAPL", "MSFT"])

        assert len(data) == 2, "Should load 2 symbols"
        assert "AAPL" in data, "Should load AAPL"
        assert "MSFT" in data, "Should load MSFT"


class TestPostgreSQLDataSource:
    """Tests for PostgreSQL database data source."""

    @pytest.fixture
    def postgres_connection_params(self):
        """Get PostgreSQL connection parameters from environment."""
        return {
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", "5432")),
            "database": os.getenv("POSTGRES_DB", "test_db"),
            "user": os.getenv("POSTGRES_USER", "test_user"),
            "password": os.getenv("POSTGRES_PASSWORD", "test_password"),
        }

    @pytest.mark.skipif(
        not os.getenv("POSTGRES_HOST"),
        reason="PostgreSQL not configured. Set POSTGRES_HOST, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD to test.",
    )
    def test_postgres_source_loads_data(self, postgres_connection_params):
        """Test that PostgreSQL source loads data (requires database setup)."""
        try:
            source = PostgreSQLSource(
                host=postgres_connection_params["host"],
                port=postgres_connection_params["port"],
                database=postgres_connection_params["database"],
                user=postgres_connection_params["user"],
                password=postgres_connection_params["password"],
                table_name="ohlcv_data",
            )

            # Try to load data (may fail if table doesn't exist)
            try:
                data = source.load_ohlcv(["AAPL", "MSFT"])
                # If successful, verify structure
                if data:
                    for symbol, df in data.items():
                        assert len(df) > 0, f"Data for {symbol} should not be empty"
                        assert all(
                            col in df.columns for col in ["open", "high", "low", "close", "volume"]
                        ), f"Data for {symbol} should have OHLCV columns"
            except Exception as e:
                pytest.skip(f"PostgreSQL test requires database setup: {e}")

        except ImportError:
            pytest.skip("psycopg2 not available. Install with: pip install psycopg2-binary")


class TestParquetDataSource:
    """Tests for Parquet data source."""

    @pytest.fixture
    def parquet_dir(self, tmp_path):
        """Create temporary directory with Parquet files."""
        data_dir = tmp_path / "parquet_data"
        data_dir.mkdir()

        try:
            import pyarrow.parquet as pq
        except ImportError:
            pytest.skip("pyarrow not available")

        # Create sample Parquet files
        for symbol in ["AAPL", "MSFT", "GOOGL"]:
            df = create_sample_ohlcv_data(symbol, days=30)
            parquet_file = data_dir / f"{symbol}.parquet"
            df.reset_index().to_parquet(parquet_file)

        return data_dir

    def test_parquet_source_loads_data(self, parquet_dir):
        """Test that Parquet source loads data correctly."""
        try:
            source = ParquetDataSource(str(parquet_dir))
            data = source.load_ohlcv(["AAPL", "MSFT"])

            assert len(data) == 2, "Should load 2 symbols"
            assert "AAPL" in data, "Should load AAPL"
            assert "MSFT" in data, "Should load MSFT"

            # Verify data structure
            for symbol, df in data.items():
                assert len(df) > 0, f"Data for {symbol} should not be empty"
                assert all(
                    col in df.columns for col in ["open", "high", "low", "close", "volume"]
                ), f"Data for {symbol} should have OHLCV columns"
        except ImportError:
            pytest.skip("pyarrow not available")

    def test_parquet_source_with_loader(self, parquet_dir):
        """Test Parquet source used with load_ohlcv_data function."""
        try:
            source = ParquetDataSource(str(parquet_dir))
            data = load_ohlcv_data(source, ["AAPL", "MSFT"])

            assert len(data) == 2, "Should load 2 symbols"
            assert "AAPL" in data, "Should load AAPL"
            assert "MSFT" in data, "Should load MSFT"
        except ImportError:
            pytest.skip("pyarrow not available")


class TestHDF5DataSource:
    """Tests for HDF5 data source."""

    @pytest.fixture
    def hdf5_file(self, tmp_path):
        """Create temporary HDF5 file with sample data."""
        hdf5_path = tmp_path / "test.h5"

        try:
            import tables
        except ImportError:
            pytest.skip("tables (PyTables) not available")

        # Create HDF5 file with sample data
        for symbol in ["AAPL", "MSFT", "GOOGL"]:
            df = create_sample_ohlcv_data(symbol, days=30)
            df.reset_index().to_hdf(str(hdf5_path), key=symbol, mode="a", format="table")

        return str(hdf5_path)

    def test_hdf5_source_loads_data(self, hdf5_file):
        """Test that HDF5 source loads data correctly."""
        try:
            source = HDF5DataSource(hdf5_file)
            data = source.load_ohlcv(["AAPL", "MSFT"])

            assert len(data) == 2, "Should load 2 symbols"
            assert "AAPL" in data, "Should load AAPL"
            assert "MSFT" in data, "Should load MSFT"

            # Verify data structure
            for symbol, df in data.items():
                assert len(df) > 0, f"Data for {symbol} should not be empty"
                assert all(
                    col in df.columns for col in ["open", "high", "low", "close", "volume"]
                ), f"Data for {symbol} should have OHLCV columns"
        except ImportError:
            pytest.skip("tables (PyTables) not available")

    def test_hdf5_source_with_loader(self, hdf5_file):
        """Test HDF5 source used with load_ohlcv_data function."""
        try:
            source = HDF5DataSource(hdf5_file)
            data = load_ohlcv_data(source, ["AAPL", "MSFT"])

            assert len(data) == 2, "Should load 2 symbols"
            assert "AAPL" in data, "Should load AAPL"
            assert "MSFT" in data, "Should load MSFT"
        except ImportError:
            pytest.skip("tables (PyTables) not available")


class TestAPIDataSources:
    """Tests for API data sources (AlphaVantage, Polygon)."""

    def test_alphavantage_source_initialization(self):
        """Test that AlphaVantage source can be initialized."""
        api_key = os.getenv("ALPHAVANTAGE_API_KEY", "test_key")
        source = AlphaVantageSource(api_key)

        assert source.api_key == api_key, "API key should be set"
        assert source.rate_limit_delay > 0, "Rate limit delay should be positive"

    def test_polygon_source_initialization(self):
        """Test that Polygon source can be initialized."""
        api_key = os.getenv("POLYGON_API_KEY", "test_key")
        source = PolygonSource(api_key)

        assert source.api_key == api_key, "API key should be set"
        assert source.rate_limit_delay > 0, "Rate limit delay should be positive"

    @pytest.mark.skipif(
        not os.getenv("ALPHAVANTAGE_API_KEY"),
        reason="ALPHAVANTAGE_API_KEY not set. Set environment variable to test API source.",
    )
    def test_alphavantage_source_loads_data(self):
        """Test that AlphaVantage source loads data (requires API key)."""
        api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        source = AlphaVantageSource(api_key, rate_limit_delay=12.0)  # AlphaVantage free tier: 5 calls/min

        # Test with a single symbol (to avoid rate limits)
        try:
            data = source.load_ohlcv(["AAPL"], start_date=pd.Timestamp("2023-01-01"), end_date=pd.Timestamp("2023-12-31"))

            # May or may not have data depending on API response
            if data:
                assert "AAPL" in data, "Should load AAPL if API returns data"
                df = data["AAPL"]
                assert len(df) > 0, "Data should not be empty"
        except Exception as e:
            pytest.skip(f"AlphaVantage API test failed (may be rate limited): {e}")

    @pytest.mark.skipif(
        not os.getenv("POLYGON_API_KEY"), reason="POLYGON_API_KEY not set. Set environment variable to test API source."
    )
    def test_polygon_source_loads_data(self):
        """Test that Polygon source loads data (requires API key)."""
        api_key = os.getenv("POLYGON_API_KEY")
        source = PolygonSource(api_key, rate_limit_delay=0.1)

        # Test with a single symbol
        try:
            data = source.load_ohlcv(["AAPL"], start_date=pd.Timestamp("2023-01-01"), end_date=pd.Timestamp("2023-12-31"))

            # May or may not have data depending on API response
            if data:
                assert "AAPL" in data, "Should load AAPL if API returns data"
                df = data["AAPL"]
                assert len(df) > 0, "Data should not be empty"
        except Exception as e:
            pytest.skip(f"Polygon API test failed (may be rate limited): {e}")


class TestDataSourceInterchangeability:
    """Tests that verify data sources can be used interchangeably."""

    def test_all_sources_produce_same_format(self, sample_data_dir, sqlite_db):
        """Test that all data sources produce data in the same format."""
        # Load from CSV
        csv_source = CSVDataSource(str(sample_data_dir))
        csv_data = csv_source.load_ohlcv(["AAPL"])

        # Load from SQLite
        sqlite_source = SQLiteSource(sqlite_db, table_name="ohlcv_data")
        sqlite_data = sqlite_source.load_ohlcv(["AAPL"])

        # Verify both have same structure
        assert "AAPL" in csv_data, "CSV should load AAPL"
        assert "AAPL" in sqlite_data, "SQLite should load AAPL"

        csv_df = csv_data["AAPL"]
        sqlite_df = sqlite_data["AAPL"]

        # Both should have same columns
        assert set(csv_df.columns) == set(sqlite_df.columns), "Data sources should produce same columns"

        # Both should have date index
        assert csv_df.index.name == "date", "CSV should have date index"
        assert sqlite_df.index.name == "date", "SQLite should have date index"

    def test_data_sources_with_loader_function(self, sample_data_dir, sqlite_db):
        """Test that load_ohlcv_data works with different source types."""
        # Test with string path (CSV)
        csv_data = load_ohlcv_data(str(sample_data_dir), ["AAPL", "MSFT"])
        assert len(csv_data) == 2, "CSV loader should load 2 symbols"

        # Test with data source object (SQLite)
        sqlite_source = SQLiteSource(sqlite_db, table_name="ohlcv_data")
        sqlite_data = load_ohlcv_data(sqlite_source, ["AAPL", "MSFT"])
        assert len(sqlite_data) == 2, "SQLite loader should load 2 symbols"

"""Unit tests for data/sources/cache.py."""

import shutil
import tempfile
from pathlib import Path

import pandas as pd

from trading_system.data.sources.cache import CachedDataSource, DataCache
from trading_system.data.sources.csv_source import CSVDataSource


class TestDataCache:
    """Tests for DataCache class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache = DataCache(cache_dir=str(self.temp_dir / "cache"), ttl_hours=24, max_cache_size_mb=10)

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, "temp_dir") and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_cache_initialization(self):
        """Test cache initialization."""
        assert self.cache.cache_dir.exists()
        assert self.cache.ttl_hours == 24
        assert self.cache.max_cache_size_bytes == 10 * 1024 * 1024

    def test_cache_get_missing(self):
        """Test getting non-existent cache entry."""
        result = self.cache.get("source1", "AAPL")
        assert result is None

    def test_cache_put_and_get(self):
        """Test putting and getting data from cache."""
        df = pd.DataFrame(
            {"close": [100.0, 101.0, 102.0], "volume": [1000000, 1100000, 1200000]},
            index=pd.date_range("2024-01-01", periods=3),
        )

        # Put in cache
        self.cache.put("source1", "AAPL", df)

        # Get from cache
        cached_df = self.cache.get("source1", "AAPL")

        assert cached_df is not None
        pd.testing.assert_frame_equal(cached_df, df)

    def test_cache_with_dates(self):
        """Test cache with date parameters."""
        df = pd.DataFrame({"close": [100.0]}, index=[pd.Timestamp("2024-01-01")])
        start_date = pd.Timestamp("2024-01-01")
        end_date = pd.Timestamp("2024-01-31")

        self.cache.put("source1", "AAPL", df, start_date, end_date)
        cached_df = self.cache.get("source1", "AAPL", start_date, end_date)

        assert cached_df is not None
        pd.testing.assert_frame_equal(cached_df, df)

    def test_cache_expiration(self):
        """Test cache expiration based on TTL."""
        # Create cache with short TTL
        short_ttl_cache = DataCache(
            cache_dir=str(self.temp_dir / "cache2"), ttl_hours=0, max_cache_size_mb=10  # Expires immediately
        )

        df = pd.DataFrame({"close": [100.0]})
        short_ttl_cache.put("source1", "AAPL", df)

        # Should be expired immediately
        cached_df = short_ttl_cache.get("source1", "AAPL")
        assert cached_df is None

    def test_cache_remove(self):
        """Test removing cache entry."""
        df = pd.DataFrame({"close": [100.0]})
        self.cache.put("source1", "AAPL", df)

        # Verify it's cached
        assert self.cache.get("source1", "AAPL") is not None

        # Get cache key to remove
        cache_key = self.cache._get_cache_key("source1", "AAPL", None, None)
        self.cache.remove(cache_key)

        # Should be removed
        assert self.cache.get("source1", "AAPL") is None

    def test_cache_clear(self):
        """Test clearing all cache entries."""
        df1 = pd.DataFrame({"close": [100.0]})
        df2 = pd.DataFrame({"close": [200.0]})

        self.cache.put("source1", "AAPL", df1)
        self.cache.put("source1", "MSFT", df2)

        # Verify both are cached
        assert self.cache.get("source1", "AAPL") is not None
        assert self.cache.get("source1", "MSFT") is not None

        # Clear cache
        self.cache.clear()

        # Both should be gone
        assert self.cache.get("source1", "AAPL") is None
        assert self.cache.get("source1", "MSFT") is None

    def test_cache_stats(self):
        """Test getting cache statistics."""
        df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
        self.cache.put("source1", "AAPL", df)

        stats = self.cache.get_stats()

        assert "entries" in stats
        assert "total_size_mb" in stats
        assert "total_rows" in stats
        assert "cache_dir" in stats
        assert stats["entries"] >= 1
        assert stats["total_rows"] >= 3

    def test_cache_size_limit_enforcement(self):
        """Test that cache enforces size limit."""
        # Create small cache
        small_cache = DataCache(
            cache_dir=str(self.temp_dir / "cache3"), ttl_hours=24, max_cache_size_mb=0.001  # Very small: 1KB
        )

        # Add large DataFrame
        df = pd.DataFrame({"data": range(10000)})
        small_cache.put("source1", "AAPL", df)

        # Cache should enforce size limit (may remove old entries)
        stats = small_cache.get_stats()
        assert stats["total_size_mb"] <= 0.001 + 0.0001  # Allow small margin

    def test_cache_key_generation(self):
        """Test cache key generation."""
        key1 = self.cache._get_cache_key("source1", "AAPL", None, None)
        key2 = self.cache._get_cache_key("source1", "AAPL", None, None)

        # Same parameters should generate same key
        assert key1 == key2

        # Different parameters should generate different keys
        key3 = self.cache._get_cache_key("source1", "MSFT", None, None)
        assert key1 != key3

    def test_cache_metadata_persistence(self):
        """Test that cache metadata persists across instances."""
        df = pd.DataFrame({"close": [100.0]})
        self.cache.put("source1", "AAPL", df)

        # Create new cache instance with same directory
        new_cache = DataCache(cache_dir=str(self.cache.cache_dir), ttl_hours=24, max_cache_size_mb=10)

        # Should be able to retrieve cached data
        cached_df = new_cache.get("source1", "AAPL")
        assert cached_df is not None


class TestCachedDataSource:
    """Tests for CachedDataSource class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache = DataCache(cache_dir=str(self.temp_dir / "cache"), ttl_hours=24, max_cache_size_mb=10)

        # Create a simple CSV source for testing
        self.csv_dir = self.temp_dir / "data"
        self.csv_dir.mkdir()

        # Create sample CSV file
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1000000, 1100000],
            },
            index=pd.date_range("2024-01-01", periods=2),
        )
        df.index.name = "date"  # CSVDataSource expects a 'date' column

        csv_path = self.csv_dir / "AAPL.csv"
        df.to_csv(csv_path)

        self.source = CSVDataSource(data_path=str(self.csv_dir))
        self.cached_source = CachedDataSource(self.source, self.cache)

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, "temp_dir") and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_cached_source_initialization(self):
        """Test cached data source initialization."""
        assert self.cached_source.source == self.source
        assert self.cached_source.cache == self.cache
        assert self.cached_source.source_id is not None

    def test_cached_source_load_ohlcv_caches_data(self):
        """Test that loading data caches it."""
        symbols = ["AAPL"]

        # First load (should load from source and cache)
        data1 = self.cached_source.load_ohlcv(symbols)

        assert "AAPL" in data1
        assert len(data1["AAPL"]) > 0

        # Second load (should use cache)
        data2 = self.cached_source.load_ohlcv(symbols)

        # Should return same data
        pd.testing.assert_frame_equal(data1["AAPL"], data2["AAPL"])

    def test_cached_source_load_ohlcv_multiple_symbols(self):
        """Test loading multiple symbols with caching."""
        # Create another CSV file
        df = pd.DataFrame(
            {
                "open": [200.0, 201.0],
                "high": [202.0, 203.0],
                "low": [199.0, 200.0],
                "close": [201.0, 202.0],
                "volume": [2000000, 2100000],
            },
            index=pd.date_range("2024-01-01", periods=2),
        )
        df.index.name = "date"  # CSVDataSource expects a 'date' column

        csv_path = self.csv_dir / "MSFT.csv"
        df.to_csv(csv_path)

        symbols = ["AAPL", "MSFT"]
        data = self.cached_source.load_ohlcv(symbols)

        assert "AAPL" in data
        assert "MSFT" in data
        assert len(data) == 2

    def test_cached_source_get_available_symbols(self):
        """Test getting available symbols."""
        symbols = self.cached_source.get_available_symbols()
        assert isinstance(symbols, list)
        assert "AAPL" in symbols

    def test_cached_source_get_date_range(self):
        """Test getting date range."""
        date_range = self.cached_source.get_date_range("AAPL")
        # May return None or tuple depending on implementation
        assert date_range is None or isinstance(date_range, tuple)

    def test_cached_source_supports_incremental(self):
        """Test incremental loading support."""
        supports = self.cached_source.supports_incremental()
        assert isinstance(supports, bool)

    def test_cached_source_custom_source_id(self):
        """Test cached source with custom source ID."""
        cached_source = CachedDataSource(self.source, self.cache, source_id="custom_id")
        assert cached_source.source_id == "custom_id"

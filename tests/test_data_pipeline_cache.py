"""Unit tests for data pipeline cache."""

import json
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from trading_system.data_pipeline.cache.data_cache import DataCache


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def data_cache(temp_cache_dir):
    """Create a DataCache instance for testing."""
    return DataCache(cache_dir=temp_cache_dir, ttl_hours=24)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0],
            "close": [104.0, 105.0, 106.0],
            "volume": [1000000, 1100000, 1200000],
        }
    )


class TestDataCache:
    """Test DataCache class."""

    def test_get_cache_key(self, data_cache):
        """Test cache key generation."""
        key = data_cache.get_cache_key("AAPL", "equity", date(2023, 1, 1), date(2023, 12, 31))
        assert key == "equity_AAPL_2023-01-01_2023-12-31"

        key = data_cache.get_cache_key("BTC", "crypto", date(2024, 1, 1), date(2024, 1, 31))
        assert key == "crypto_BTC_2024-01-01_2024-01-31"

    def test_get_cache_key_deterministic(self, data_cache):
        """Test that cache key generation is deterministic."""
        key1 = data_cache.get_cache_key("AAPL", "equity", date(2023, 1, 1), date(2023, 12, 31))
        key2 = data_cache.get_cache_key("AAPL", "equity", date(2023, 1, 1), date(2023, 12, 31))
        assert key1 == key2

    def test_set_and_get(self, data_cache, sample_dataframe):
        """Test storing and retrieving data from cache."""
        key = data_cache.get_cache_key("AAPL", "equity", date(2023, 1, 1), date(2023, 1, 3))

        # Initially not in cache
        assert data_cache.get(key) is None

        # Store data
        data_cache.set(key, sample_dataframe)

        # Retrieve data
        cached_df = data_cache.get(key)
        assert cached_df is not None
        assert len(cached_df) == len(sample_dataframe)
        pd.testing.assert_frame_equal(cached_df, sample_dataframe)

    def test_is_valid(self, data_cache, sample_dataframe):
        """Test is_valid method."""
        key = data_cache.get_cache_key("AAPL", "equity", date(2023, 1, 1), date(2023, 1, 3))

        # Initially invalid
        assert not data_cache.is_valid(key)

        # Store data
        data_cache.set(key, sample_dataframe)

        # Should be valid
        assert data_cache.is_valid(key)

    def test_ttl_expiration(self, data_cache, sample_dataframe):
        """Test TTL expiration."""
        # Create cache with very short TTL
        short_ttl_cache = DataCache(cache_dir=data_cache.cache_dir, ttl_hours=0.001)  # ~3.6 seconds

        key = short_ttl_cache.get_cache_key("AAPL", "equity", date(2023, 1, 1), date(2023, 1, 3))

        # Store data
        short_ttl_cache.set(key, sample_dataframe)

        # Should be valid immediately
        assert short_ttl_cache.is_valid(key)
        assert short_ttl_cache.get(key) is not None

        # Wait for expiration (we'll manually expire by modifying metadata)
        metadata_path = short_ttl_cache._get_metadata_path(key)
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Set cached_at to past
        past_time = (datetime.now() - timedelta(hours=1)).isoformat()
        metadata["cached_at"] = past_time

        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Should be expired now
        assert not short_ttl_cache.is_valid(key)
        assert short_ttl_cache.get(key) is None

    def test_clear_specific_key(self, data_cache, sample_dataframe):
        """Test clearing a specific cache key."""
        key1 = data_cache.get_cache_key("AAPL", "equity", date(2023, 1, 1), date(2023, 1, 3))
        key2 = data_cache.get_cache_key("MSFT", "equity", date(2023, 1, 1), date(2023, 1, 3))

        # Store both
        data_cache.set(key1, sample_dataframe)
        data_cache.set(key2, sample_dataframe)

        # Both should be valid
        assert data_cache.is_valid(key1)
        assert data_cache.is_valid(key2)

        # Clear one
        data_cache.clear(key1)

        # One should be gone, other should remain
        assert not data_cache.is_valid(key1)
        assert data_cache.is_valid(key2)

    def test_clear_all(self, data_cache, sample_dataframe):
        """Test clearing all cache."""
        key1 = data_cache.get_cache_key("AAPL", "equity", date(2023, 1, 1), date(2023, 1, 3))
        key2 = data_cache.get_cache_key("MSFT", "equity", date(2023, 1, 1), date(2023, 1, 3))

        # Store both
        data_cache.set(key1, sample_dataframe)
        data_cache.set(key2, sample_dataframe)

        # Both should be valid
        assert data_cache.is_valid(key1)
        assert data_cache.is_valid(key2)

        # Clear all
        data_cache.clear()

        # Both should be gone
        assert not data_cache.is_valid(key1)
        assert not data_cache.is_valid(key2)

    def test_missing_files(self, data_cache):
        """Test handling of missing cache files."""
        key = data_cache.get_cache_key("AAPL", "equity", date(2023, 1, 1), date(2023, 1, 3))

        # Should return None for missing files
        assert data_cache.get(key) is None
        assert not data_cache.is_valid(key)

    def test_corrupted_metadata(self, data_cache, sample_dataframe):
        """Test handling of corrupted metadata."""
        key = data_cache.get_cache_key("AAPL", "equity", date(2023, 1, 1), date(2023, 1, 3))

        # Store data
        data_cache.set(key, sample_dataframe)

        # Corrupt metadata file
        metadata_path = data_cache._get_metadata_path(key)
        with open(metadata_path, "w") as f:
            f.write("invalid json{")

        # Should handle gracefully
        assert not data_cache.is_valid(key)
        assert data_cache.get(key) is None

    def test_corrupted_parquet(self, data_cache):
        """Test handling of corrupted parquet file."""
        key = data_cache.get_cache_key("AAPL", "equity", date(2023, 1, 1), date(2023, 1, 3))

        # Create invalid parquet file
        cache_path = data_cache._get_cache_path(key)
        cache_path.write_text("invalid parquet data")

        # Create valid metadata
        metadata = {
            "cached_at": datetime.now().isoformat(),
            "ttl_hours": 24,
            "rows": 3,
            "columns": ["date", "symbol", "open", "high", "low", "close", "volume"],
        }
        data_cache._save_metadata(key, metadata)

        # Should handle gracefully
        assert not data_cache.is_valid(key)
        assert data_cache.get(key) is None

    def test_metadata_contains_timestamp(self, data_cache, sample_dataframe):
        """Test that metadata contains cached_at timestamp."""
        key = data_cache.get_cache_key("AAPL", "equity", date(2023, 1, 1), date(2023, 1, 3))

        # Store data
        data_cache.set(key, sample_dataframe)

        # Check metadata
        metadata = data_cache._load_metadata(key)
        assert metadata is not None
        assert "cached_at" in metadata
        assert "ttl_hours" in metadata
        assert "rows" in metadata
        assert "columns" in metadata

        # Verify timestamp is valid ISO format
        cached_at = metadata["cached_at"]
        datetime.fromisoformat(cached_at.replace("Z", "+00:00"))

    def test_multiple_asset_classes(self, data_cache, sample_dataframe):
        """Test caching data for different asset classes."""
        equity_key = data_cache.get_cache_key("AAPL", "equity", date(2023, 1, 1), date(2023, 1, 3))
        crypto_key = data_cache.get_cache_key("BTC", "crypto", date(2023, 1, 1), date(2023, 1, 3))

        # Store both
        data_cache.set(equity_key, sample_dataframe)
        data_cache.set(crypto_key, sample_dataframe)

        # Both should be valid
        assert data_cache.is_valid(equity_key)
        assert data_cache.is_valid(crypto_key)

        # Both should be retrievable
        equity_df = data_cache.get(equity_key)
        crypto_df = data_cache.get(crypto_key)

        assert equity_df is not None
        assert crypto_df is not None
        pd.testing.assert_frame_equal(equity_df, sample_dataframe)
        pd.testing.assert_frame_equal(crypto_df, sample_dataframe)

    def test_empty_dataframe(self, data_cache):
        """Test caching empty DataFrame."""
        key = data_cache.get_cache_key("AAPL", "equity", date(2023, 1, 1), date(2023, 1, 3))
        empty_df = pd.DataFrame()

        # Store empty DataFrame
        data_cache.set(key, empty_df)

        # Should be valid
        assert data_cache.is_valid(key)

        # Should retrieve empty DataFrame
        cached_df = data_cache.get(key)
        assert cached_df is not None
        assert len(cached_df) == 0

    def test_cache_directory_creation(self, temp_cache_dir):
        """Test that cache directory is created if it doesn't exist."""
        new_cache_dir = temp_cache_dir / "new_cache"
        assert not new_cache_dir.exists()

        # Creating cache should create directory
        cache = DataCache(cache_dir=new_cache_dir)
        assert new_cache_dir.exists()
        assert new_cache_dir.is_dir()

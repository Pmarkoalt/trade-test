"""Data cache for storing and retrieving fetched OHLCV data."""

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataCache:
    """Cache for storing and retrieving fetched OHLCV data.

    Uses Parquet files for data storage and JSON files for metadata.
    Supports TTL (time-to-live) for cache expiration.
    """

    def __init__(self, cache_dir: Path, ttl_hours: int = 24):
        """Initialize the data cache.

        Args:
            cache_dir: Path to the cache directory
            ttl_hours: Time-to-live for cached data in hours (default: 24)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_hours = ttl_hours

    def get_cache_key(self, symbol: str, asset_class: str, start_date: date, end_date: date) -> str:
        """Generate cache key.

        Args:
            symbol: Symbol name (e.g., "AAPL", "BTC")
            asset_class: Asset class (e.g., "equity", "crypto")
            start_date: Start date
            end_date: End date

        Returns:
            Cache key string in format: {asset_class}_{symbol}_{start_date}_{end_date}
        """
        return f"{asset_class}_{symbol}_{start_date}_{end_date}"

    def _get_cache_path(self, key: str) -> Path:
        """Get path to cache file for a key.

        Args:
            key: Cache key

        Returns:
            Path to parquet file
        """
        return self.cache_dir / f"{key}.parquet"

    def _get_metadata_path(self, key: str) -> Path:
        """Get path to metadata file for a key.

        Args:
            key: Cache key

        Returns:
            Path to metadata JSON file
        """
        return self.cache_dir / f"{key}.meta.json"

    def _load_metadata(self, key: str) -> Optional[dict]:
        """Load metadata for a cache key.

        Args:
            key: Cache key

        Returns:
            Metadata dictionary or None if not found
        """
        metadata_path = self._get_metadata_path(key)
        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, "r") as f:
                result = json.load(f)
                return dict(result) if result else None
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error loading metadata for {key}: {e}")
            return None

    def _save_metadata(self, key: str, metadata: dict) -> None:
        """Save metadata for a cache key.

        Args:
            key: Cache key
            metadata: Metadata dictionary
        """
        metadata_path = self._get_metadata_path(key)
        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Error saving metadata for {key}: {e}")

    def _is_expired(self, metadata: dict) -> bool:
        """Check if cache entry is expired based on metadata.

        Args:
            metadata: Metadata dictionary

        Returns:
            True if expired, False otherwise
        """
        cached_at_str = metadata.get("cached_at")
        if not cached_at_str:
            return True

        try:
            # Parse ISO format timestamp
            cached_at = datetime.fromisoformat(cached_at_str.replace("Z", "+00:00"))
            if cached_at.tzinfo is None:
                # Assume UTC if no timezone
                cached_at = cached_at.replace(tzinfo=None)

            age = datetime.now() - cached_at.replace(tzinfo=None)
            ttl = timedelta(hours=metadata.get("ttl_hours", self.ttl_hours))

            return age > ttl
        except (ValueError, TypeError) as e:
            logger.warning(f"Error parsing cached_at timestamp: {e}")
            return True

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get cached data if not expired.

        Args:
            key: Cache key

        Returns:
            DataFrame with cached data, or None if not found or expired
        """
        cache_path = self._get_cache_path(key)
        metadata_path = self._get_metadata_path(key)

        # Check if files exist
        if not cache_path.exists() or not metadata_path.exists():
            return None

        # Load and check metadata
        metadata = self._load_metadata(key)
        if metadata is None:
            # Invalid metadata, clean up
            self._delete_cache_entry(key)
            return None

        # Check if expired
        if self._is_expired(metadata):
            logger.debug(f"Cache entry {key} expired, deleting")
            self._delete_cache_entry(key)
            return None

        # Load data from parquet
        try:
            df = pd.read_parquet(cache_path)
            logger.debug(f"Cache hit for {key}")
            return df
        except Exception as e:
            logger.error(f"Error loading cache file {cache_path}: {e}")
            self._delete_cache_entry(key)
            return None

    def set(self, key: str, data: pd.DataFrame) -> None:
        """Cache data with timestamp.

        Args:
            key: Cache key
            data: DataFrame to cache
        """
        cache_path = self._get_cache_path(key)

        try:
            # Save DataFrame to parquet
            data.to_parquet(cache_path, index=False)

            # Save metadata
            metadata = {
                "cached_at": datetime.now().isoformat(),
                "ttl_hours": self.ttl_hours,
                "rows": len(data),
                "columns": list(data.columns),
            }
            self._save_metadata(key, metadata)

            logger.debug(f"Cached {key} ({len(data)} rows)")

        except Exception as e:
            logger.error(f"Error caching {key}: {e}")
            # Clean up partial files
            if cache_path.exists():
                try:
                    cache_path.unlink()
                except Exception:
                    pass

    def is_valid(self, key: str) -> bool:
        """Check if cache entry exists and is not expired.

        Args:
            key: Cache key

        Returns:
            True if cache entry is valid, False otherwise
        """
        cache_path = self._get_cache_path(key)
        metadata_path = self._get_metadata_path(key)

        # Check if files exist
        if not cache_path.exists() or not metadata_path.exists():
            return False

        # Load and check metadata
        metadata = self._load_metadata(key)
        if metadata is None:
            return False

        # Check if expired
        if self._is_expired(metadata):
            # Delete expired entry
            self._delete_cache_entry(key)
            return False

        return True

    def clear(self, key: Optional[str] = None) -> None:
        """Clear specific key or all cache.

        Args:
            key: Cache key to clear, or None to clear all cache
        """
        if key is None:
            # Clear all cache
            for cache_file in self.cache_dir.glob("*.parquet"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Error deleting cache file {cache_file}: {e}")

            for metadata_file in self.cache_dir.glob("*.meta.json"):
                try:
                    metadata_file.unlink()
                except Exception as e:
                    logger.warning(f"Error deleting metadata file {metadata_file}: {e}")

            logger.info("All cache cleared")
        else:
            # Clear specific key
            self._delete_cache_entry(key)
            logger.debug(f"Cache entry {key} cleared")

    def _delete_cache_entry(self, key: str) -> None:
        """Delete cache entry (both data and metadata files).

        Args:
            key: Cache key
        """
        cache_path = self._get_cache_path(key)
        metadata_path = self._get_metadata_path(key)

        try:
            if cache_path.exists():
                cache_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
        except Exception as e:
            logger.warning(f"Error deleting cache entry {key}: {e}")

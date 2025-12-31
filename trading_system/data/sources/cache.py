"""Data caching layer for data sources."""

import hashlib
import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .base_source import BaseDataSource

logger = logging.getLogger(__name__)


class DataCache:
    """Cache layer for data sources.

    Caches data to disk to avoid repeated API calls or slow database queries.
    Supports TTL (time-to-live) for cache invalidation.
    """

    def __init__(self, cache_dir: str = ".cache", ttl_hours: int = 24, max_cache_size_mb: int = 1000):
        """Initialize data cache.

        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time-to-live for cache entries (hours)
            max_cache_size_mb: Maximum cache size in MB (older entries evicted first)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_hours = ttl_hours
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        self._metadata_file = self.cache_dir / "cache_metadata.pkl"
        self._metadata: Dict[str, Dict] = {}
        self._load_metadata()

    def _load_metadata(self):
        """Load cache metadata."""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, "rb") as f:
                    self._metadata = pickle.load(f)
            except Exception as e:
                logger.warning(f"Error loading cache metadata: {e}")
                self._metadata = {}
        else:
            self._metadata = {}

    def _save_metadata(self):
        """Save cache metadata."""
        try:
            with open(self._metadata_file, "wb") as f:
                pickle.dump(self._metadata, f)
        except Exception as e:
            logger.error(f"Error saving cache metadata: {e}")

    def _get_cache_key(
        self, source_id: str, symbol: str, start_date: Optional[pd.Timestamp], end_date: Optional[pd.Timestamp]
    ) -> str:
        """Generate cache key for a request.

        Args:
            source_id: Identifier for the data source
            symbol: Symbol
            start_date: Start date
            end_date: End date

        Returns:
            Cache key string
        """
        # Create hash of request parameters
        key_str = f"{source_id}:{symbol}:{start_date}:{end_date}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cache entry.

        Args:
            cache_key: Cache key

        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{cache_key}.pkl"

    def get(
        self, source_id: str, symbol: str, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None
    ) -> Optional[pd.DataFrame]:
        """Get data from cache.

        Args:
            source_id: Identifier for the data source
            symbol: Symbol
            start_date: Start date
            end_date: End date

        Returns:
            Cached DataFrame or None if not found/expired
        """
        cache_key = self._get_cache_key(source_id, symbol, start_date, end_date)

        # Check metadata
        if cache_key not in self._metadata:
            return None

        entry = self._metadata[cache_key]

        # Check if expired
        cache_time = entry.get("timestamp")
        if cache_time is None:
            return None

        age = datetime.now() - cache_time
        if age > timedelta(hours=self.ttl_hours):
            # Expired, remove from cache
            self.remove(cache_key)
            return None

        # Load from disk
        cache_path = self._get_cache_path(cache_key)
        if not cache_path.exists():
            # File missing, remove metadata
            self.remove(cache_key)
            return None

        try:
            df = pd.read_pickle(cache_path)
            logger.debug(f"Cache hit for {source_id}:{symbol}")
            return df
        except Exception as e:
            logger.error(f"Error loading cache entry {cache_key}: {e}")
            self.remove(cache_key)
            return None

    def put(
        self,
        source_id: str,
        symbol: str,
        df: pd.DataFrame,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ):
        """Store data in cache.

        Args:
            source_id: Identifier for the data source
            symbol: Symbol
            df: DataFrame to cache
            start_date: Start date
            end_date: End date
        """
        cache_key = self._get_cache_key(source_id, symbol, start_date, end_date)
        cache_path = self._get_cache_path(cache_key)

        try:
            # Save to disk
            df.to_pickle(cache_path)

            # Update metadata
            file_size = cache_path.stat().st_size
            self._metadata[cache_key] = {
                "source_id": source_id,
                "symbol": symbol,
                "timestamp": datetime.now(),
                "size": file_size,
                "rows": len(df),
            }

            # Enforce cache size limit
            self._enforce_size_limit()

            # Save metadata
            self._save_metadata()

            logger.debug(f"Cached {source_id}:{symbol} ({len(df)} rows)")

        except Exception as e:
            logger.error(f"Error caching {source_id}:{symbol}: {e}")

    def remove(self, cache_key: str):
        """Remove cache entry.

        Args:
            cache_key: Cache key to remove
        """
        if cache_key in self._metadata:
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                try:
                    cache_path.unlink()
                except Exception as e:
                    logger.warning(f"Error removing cache file {cache_path}: {e}")
            del self._metadata[cache_key]
            self._save_metadata()

    def _enforce_size_limit(self):
        """Enforce maximum cache size by removing oldest entries."""
        # Calculate total size
        total_size = sum(entry.get("size", 0) for entry in self._metadata.values())

        if total_size <= self.max_cache_size_bytes:
            return

        # Sort by timestamp (oldest first)
        sorted_entries = sorted(self._metadata.items(), key=lambda x: x[1].get("timestamp", datetime.min))

        # Remove oldest entries until under limit
        for cache_key, entry in sorted_entries:
            if total_size <= self.max_cache_size_bytes:
                break

            total_size -= entry.get("size", 0)
            self.remove(cache_key)

    def clear(self):
        """Clear all cache entries."""
        for cache_key in list(self._metadata.keys()):
            self.remove(cache_key)
        logger.info("Cache cleared")

    def get_stats(self) -> Dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_size = sum(entry.get("size", 0) for entry in self._metadata.values())
        total_rows = sum(entry.get("rows", 0) for entry in self._metadata.values())

        return {
            "entries": len(self._metadata),
            "total_size_mb": total_size / (1024 * 1024),
            "total_rows": total_rows,
            "cache_dir": str(self.cache_dir),
        }


class CachedDataSource(BaseDataSource):
    """Wrapper that adds caching to any data source."""

    def __init__(self, source: BaseDataSource, cache: DataCache, source_id: Optional[str] = None):
        """Initialize cached data source.

        Args:
            source: Underlying data source to wrap
            cache: DataCache instance
            source_id: Optional identifier for cache keys (defaults to source type)
        """
        self.source = source
        self.cache = cache
        self.source_id = source_id or type(source).__name__

    def load_ohlcv(
        self, symbols: List[str], start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None
    ) -> Dict[str, pd.DataFrame]:
        """Load OHLCV data with caching.

        Args:
            symbols: List of symbols to load
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Dictionary mapping symbol -> DataFrame
        """
        data = {}

        # Try cache first
        for symbol in symbols:
            cached_df = self.cache.get(self.source_id, symbol, start_date, end_date)
            if cached_df is not None:
                data[symbol] = cached_df

        # Load missing symbols from source
        missing_symbols = [s for s in symbols if s not in data]
        if missing_symbols:
            source_data = self.source.load_ohlcv(missing_symbols, start_date, end_date)

            # Cache new data
            for symbol, df in source_data.items():
                self.cache.put(self.source_id, symbol, df, start_date, end_date)
                data[symbol] = df

        return data

    def get_available_symbols(self) -> List[str]:
        """Get available symbols from underlying source."""
        return self.source.get_available_symbols()

    def get_date_range(self, symbol: str) -> Optional[tuple[pd.Timestamp, pd.Timestamp]]:
        """Get date range from underlying source."""
        return self.source.get_date_range(symbol)

    def supports_incremental(self) -> bool:
        """Incremental loading support from underlying source."""
        return self.source.supports_incremental()

    def load_incremental(self, symbol: str, last_update_date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Load incremental data with caching."""
        # Check cache first
        end_date = pd.Timestamp.now()
        cached_df = self.cache.get(self.source_id, symbol, last_update_date + pd.Timedelta(days=1), end_date)
        if cached_df is not None:
            return cached_df

        # Load from source
        df = self.source.load_incremental(symbol, last_update_date)
        if df is not None:
            self.cache.put(self.source_id, symbol, df, last_update_date + pd.Timedelta(days=1), end_date)

        return df

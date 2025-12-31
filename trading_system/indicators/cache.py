"""Caching utilities for indicator calculations to avoid recomputing same data."""

import hashlib
import logging
from collections import OrderedDict
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class IndicatorCache:
    """Cache for indicator calculations based on data hash.

    This cache stores computed indicators to avoid recomputing them
    when the same data is processed multiple times. The cache key is
    based on the data hash and indicator parameters.

    Enhanced features:
    - Proper LRU eviction using OrderedDict
    - Better cache key generation using data hash
    - Cache invalidation by symbol or pattern
    - Cross-strategy caching (shared global cache)
    """

    def __init__(self, max_size: int = 256):
        """Initialize cache.

        Args:
            max_size: Maximum number of cached results (default 256, increased from 128)
        """
        self.max_size = max_size
        # Use OrderedDict for proper LRU eviction
        self._cache: OrderedDict[Tuple[str, str, int], Any] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._invalidations = 0

    def _hash_series(self, series: pd.Series, symbol: Optional[str] = None) -> str:
        """Create hash of Series for cache key.

        Uses a stable hash based on:
        - Series length
        - Last N values (enough to cover indicator windows)
        - Index (dates) if available

        Args:
            series: Series to hash
            symbol: Optional symbol name for better cache keys

        Returns:
            Hash string
        """
        if len(series) == 0:
            return f"{symbol or 'unknown'}_empty"

        # Hash last 250 rows (enough for all indicator windows including MA200)
        # This ensures cache works across strategies using same data
        hash_window = min(250, len(series))
        hash_data = series.tail(hash_window)

        # Create stable hash from:
        # - Symbol (if provided)
        # - Series length
        # - Last date in index
        # - Last few values (for data content)
        # - First value (for data integrity check)
        data_parts = [
            symbol or "unknown",
            str(len(series)),
            str(hash_data.index[-1]) if hasattr(hash_data.index[-1], "isoformat") else str(hash_data.index[-1]),
            str(hash_data.iloc[-1]),
            str(hash_data.iloc[0]) if len(hash_data) > 0 else "",
            str(hash_data.iloc[-10:].sum()) if len(hash_data) >= 10 else "",
        ]
        data_str = "_".join(data_parts)
        return hashlib.md5(data_str.encode()).hexdigest()

    def _hash_dataframe(self, df: pd.DataFrame, symbol: str) -> str:
        """Create hash of DataFrame for cache key.

        Args:
            df: DataFrame to hash
            symbol: Symbol name

        Returns:
            Hash string
        """
        if len(df) == 0:
            return f"{symbol}_empty"

        # Hash the last 250 rows (enough to cover all indicator windows)
        hash_data = df.tail(250)

        # Create hash from key columns and index
        data_parts = [
            symbol,
            str(len(df)),
            str(hash_data.index[-1]) if len(hash_data) > 0 else "",
            str(hash_data["close"].iloc[-1]) if "close" in hash_data.columns and len(hash_data) > 0 else "",
            str(hash_data["close"].iloc[0]) if "close" in hash_data.columns and len(hash_data) > 0 else "",
            str(hash_data["close"].iloc[-10:].sum()) if "close" in hash_data.columns and len(hash_data) >= 10 else "",
        ]
        data_str = "_".join(data_parts)
        return hashlib.md5(data_str.encode()).hexdigest()

    def _create_cache_key(self, data_hash: str, indicator_name: str, window: int, **kwargs) -> Tuple[str, str, int]:
        """Create cache key from components.

        Args:
            data_hash: Hash of the input data
            indicator_name: Name of indicator (e.g., 'ma', 'atr', 'roc')
            window: Window/period parameter
            **kwargs: Additional parameters (e.g., period for ATR)

        Returns:
            Cache key tuple
        """
        # Include additional params in hash if provided
        if kwargs:
            param_str = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
            data_hash = hashlib.md5(f"{data_hash}_{param_str}".encode()).hexdigest()

        return (data_hash, indicator_name, window)

    def get(self, cache_key: Tuple[str, str, int]) -> Optional[Any]:
        """Get cached result.

        Implements LRU: moves accessed item to end.

        Args:
            cache_key: Tuple of (hash, indicator_name, window)

        Returns:
            Cached result or None
        """
        if cache_key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(cache_key)
            self._hits += 1
            return self._cache[cache_key]
        self._misses += 1
        return None

    def set(self, cache_key: Tuple[str, str, int], value: Any) -> None:
        """Store result in cache.

        Implements LRU: removes oldest items when cache is full.

        Args:
            cache_key: Tuple of (hash, indicator_name, window)
            value: Result to cache
        """
        # If key exists, move to end
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
        else:
            # Add new item
            self._cache[cache_key] = value

            # Remove oldest if cache is full
            if len(self._cache) > self.max_size:
                # Remove first item (oldest)
                self._cache.popitem(last=False)

    def invalidate(self, symbol: Optional[str] = None, indicator_name: Optional[str] = None) -> int:
        """Invalidate cache entries matching criteria.

        Args:
            symbol: If provided, invalidate entries for this symbol (checks hash prefix)
            indicator_name: If provided, invalidate entries for this indicator

        Returns:
            Number of entries invalidated
        """
        if not symbol and not indicator_name:
            # Clear all
            count = len(self._cache)
            self._cache.clear()
            self._invalidations += count
            return count

        # Remove matching entries
        keys_to_remove = []
        for key in self._cache.keys():
            data_hash, ind_name, _ = key

            # Check symbol match (if symbol provided, check if hash starts with symbol)
            symbol_match = True
            if symbol:
                symbol_match = symbol.lower() in data_hash.lower() or data_hash.startswith(symbol.lower())

            # Check indicator match
            indicator_match = True
            if indicator_name:
                indicator_match = ind_name == indicator_name

            if symbol_match and indicator_match:
                keys_to_remove.append(key)

        # Remove matched keys
        for key in keys_to_remove:
            del self._cache[key]

        self._invalidations += len(keys_to_remove)
        return len(keys_to_remove)

    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        self._invalidations = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "size": len(self._cache),
            "max_size": self.max_size,
            "invalidations": self._invalidations,
            "utilization": len(self._cache) / self.max_size if self.max_size > 0 else 0.0,
        }


# Global cache instance (can be disabled by setting to None)
_global_cache: Optional[IndicatorCache] = None


def get_cache() -> Optional[IndicatorCache]:
    """Get global cache instance.

    Returns:
        IndicatorCache instance or None if caching is disabled
    """
    return _global_cache


def set_cache(cache: Optional[IndicatorCache]) -> None:
    """Set global cache instance.

    Args:
        cache: IndicatorCache instance or None to disable caching
    """
    global _global_cache
    _global_cache = cache


def enable_caching(max_size: int = 128) -> IndicatorCache:
    """Enable caching with specified max size.

    Args:
        max_size: Maximum cache size

    Returns:
        IndicatorCache instance
    """
    cache = IndicatorCache(max_size=max_size)
    set_cache(cache)
    return cache


def disable_caching() -> None:
    """Disable caching."""
    set_cache(None)


def create_cache_key_for_series(
    series: pd.Series, indicator_name: str, window: int, symbol: Optional[str] = None, **kwargs
) -> Tuple[str, str, int]:
    """Create cache key for a Series-based indicator.

    This is a convenience function that indicators can use to create
    proper cache keys based on the input data.

    Args:
        series: Input Series
        indicator_name: Name of indicator
        window: Window/period parameter
        symbol: Optional symbol name for better cache keys
        **kwargs: Additional parameters

    Returns:
        Cache key tuple
    """
    cache = get_cache()
    if cache is None:
        # Return dummy key if caching disabled
        return ("", indicator_name, window)

    data_hash = cache._hash_series(series, symbol)
    return cache._create_cache_key(data_hash, indicator_name, window, **kwargs)


def create_cache_key_for_dataframe(
    df: pd.DataFrame, indicator_name: str, window: int, symbol: Optional[str] = None, **kwargs
) -> Tuple[str, str, int]:
    """Create cache key for a DataFrame-based indicator.

    Args:
        df: Input DataFrame
        indicator_name: Name of indicator
        window: Window/period parameter
        symbol: Optional symbol name for better cache keys
        **kwargs: Additional parameters

    Returns:
        Cache key tuple
    """
    cache = get_cache()
    if cache is None:
        # Return dummy key if caching disabled
        return ("", indicator_name, window)

    data_hash = cache._hash_dataframe(df, symbol or "unknown")
    return cache._create_cache_key(data_hash, indicator_name, window, **kwargs)

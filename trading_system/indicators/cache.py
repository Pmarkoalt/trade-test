"""Caching utilities for indicator calculations to avoid recomputing same data."""

import hashlib
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from functools import lru_cache
import numpy as np


class IndicatorCache:
    """Cache for indicator calculations based on data hash.
    
    This cache stores computed indicators to avoid recomputing them
    when the same data is processed multiple times. The cache key is
    based on the data hash and indicator parameters.
    """
    
    def __init__(self, max_size: int = 128):
        """Initialize cache.
        
        Args:
            max_size: Maximum number of cached results (default 128)
        """
        self.max_size = max_size
        self._cache: Dict[Tuple[str, str, int], Any] = {}
        self._hits = 0
        self._misses = 0
    
    def _hash_dataframe(self, df: pd.DataFrame, symbol: str) -> str:
        """Create hash of DataFrame for cache key.
        
        Args:
            df: DataFrame to hash
            symbol: Symbol name
            
        Returns:
            Hash string
        """
        # Use index (dates) and relevant columns for hash
        # Only hash the last few rows since we're computing incrementally
        if len(df) == 0:
            return f"{symbol}_empty"
        
        # Hash the last 200 rows (enough to cover all indicator windows)
        hash_data = df.tail(200)
        data_str = f"{symbol}_{hash_data.index[-1]}_{len(hash_data)}_{hash_data['close'].iloc[-1] if 'close' in hash_data.columns else ''}"
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get(self, cache_key: Tuple[str, str, int]) -> Optional[Any]:
        """Get cached result.
        
        Args:
            cache_key: Tuple of (hash, indicator_name, window)
            
        Returns:
            Cached result or None
        """
        if cache_key in self._cache:
            self._hits += 1
            return self._cache[cache_key]
        self._misses += 1
        return None
    
    def set(self, cache_key: Tuple[str, str, int], value: Any) -> None:
        """Store result in cache.
        
        Args:
            cache_key: Tuple of (hash, indicator_name, window)
            value: Result to cache
        """
        # Simple LRU: remove oldest if cache is full
        if len(self._cache) >= self.max_size:
            # Remove first item (oldest)
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        
        self._cache[cache_key] = value
    
    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'size': len(self._cache),
            'max_size': self.max_size
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


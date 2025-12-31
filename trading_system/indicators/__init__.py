"""Technical indicators library for momentum trading system."""

from .atr import atr
from .breakouts import highest_close
from .cache import IndicatorCache, disable_caching, enable_caching, get_cache, set_cache
from .correlation import rolling_corr
from .feature_computer import compute_features, compute_features_for_date
from .ma import ma
from .momentum import roc
from .parallel import batch_compute_features, compute_features_parallel
from .profiling import IndicatorProfiler, disable_profiling, enable_profiling, get_profiler, set_profiler
from .volume import adv

__all__ = [
    "ma",
    "atr",
    "roc",
    "highest_close",
    "adv",
    "rolling_corr",
    "compute_features",
    "compute_features_for_date",
    "IndicatorCache",
    "get_cache",
    "set_cache",
    "enable_caching",
    "disable_caching",
    "IndicatorProfiler",
    "get_profiler",
    "set_profiler",
    "enable_profiling",
    "disable_profiling",
    "compute_features_parallel",
    "batch_compute_features",
]

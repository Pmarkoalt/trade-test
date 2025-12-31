"""Technical indicators library for momentum trading system."""

from .ma import ma
from .atr import atr
from .momentum import roc
from .breakouts import highest_close
from .volume import adv
from .correlation import rolling_corr
from .feature_computer import compute_features, compute_features_for_date
from .cache import (
    IndicatorCache,
    get_cache,
    set_cache,
    enable_caching,
    disable_caching
)
from .profiling import (
    IndicatorProfiler,
    get_profiler,
    set_profiler,
    enable_profiling,
    disable_profiling
)
from .parallel import compute_features_parallel, batch_compute_features

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


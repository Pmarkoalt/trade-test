"""Data loading and validation module."""

from .calendar import get_crypto_days, get_trading_calendar, get_trading_days
from .lazy_loader import LazyMarketData
from .loader import CRYPTO_UNIVERSE, load_all_data, load_benchmark, load_ohlcv_data, load_universe
from .memory_profiler import MemoryProfiler, estimate_dataframe_memory, optimize_dataframe_dtypes
from .universe import (
    FIXED_CRYPTO_UNIVERSE,
    CryptoUniverseManager,
    UniverseConfig,
    create_universe_config_from_dict,
    select_crypto_universe,
)
from .validator import detect_missing_data, validate_ohlcv

__all__ = [
    # Loading functions
    "load_ohlcv_data",
    "load_universe",
    "load_benchmark",
    "load_all_data",
    "CRYPTO_UNIVERSE",
    # Validation functions
    "validate_ohlcv",
    "detect_missing_data",
    # Calendar functions
    "get_trading_days",
    "get_trading_calendar",
    "get_crypto_days",
    # Universe functions
    "UniverseConfig",
    "CryptoUniverseManager",
    "create_universe_config_from_dict",
    "select_crypto_universe",
    "FIXED_CRYPTO_UNIVERSE",
    # Memory optimization
    "MemoryProfiler",
    "optimize_dataframe_dtypes",
    "estimate_dataframe_memory",
    "LazyMarketData",
]

"""Data loading and validation module."""

from .loader import (
    load_ohlcv_data,
    load_universe,
    load_benchmark,
    load_all_data,
    CRYPTO_UNIVERSE
)
from .universe import (
    UniverseConfig,
    CryptoUniverseManager,
    create_universe_config_from_dict,
    select_crypto_universe,
    FIXED_CRYPTO_UNIVERSE
)
from .validator import (
    validate_ohlcv,
    detect_missing_data
)
from .calendar import (
    get_trading_days,
    get_trading_calendar,
    get_crypto_days
)
from .memory_profiler import (
    MemoryProfiler,
    optimize_dataframe_dtypes,
    estimate_dataframe_memory
)
from .lazy_loader import LazyMarketData

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


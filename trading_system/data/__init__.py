"""Data loading and validation module."""

from .loader import (
    load_ohlcv_data,
    load_universe,
    load_benchmark,
    load_all_data,
    CRYPTO_UNIVERSE
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
]


"""Signals module for generating live trading recommendations."""

from .config import SignalConfig
from .factory import create_backtest_signal_generator, create_live_signal_generator
from .live_signal_generator import LiveSignalGenerator
from .recommendation import Recommendation

__all__ = [
    "Recommendation",
    "SignalConfig",
    "LiveSignalGenerator",
    "create_live_signal_generator",
    "create_backtest_signal_generator",
]

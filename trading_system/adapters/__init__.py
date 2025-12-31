"""Broker adapter interfaces for paper trading and live trading.

This module provides adapters for connecting to broker APIs for paper trading
and live trading. Currently supports:
- Alpaca (paper and live trading)
- Interactive Brokers (paper and live trading)

Each adapter implements the BaseAdapter interface and can work in paper trading
mode (simulated execution) or with real broker accounts.
"""

from .alpaca_adapter import AlpacaAdapter
from .base_adapter import AccountInfo, AdapterConfig, BaseAdapter
from .ib_adapter import IBAdapter

__all__ = [
    "BaseAdapter",
    "AdapterConfig",
    "AccountInfo",
    "AlpacaAdapter",
    "IBAdapter",
]

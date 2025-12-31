"""Factor-based trading strategies."""

from .equity_factor import EquityFactorStrategy
from .factor_base import FactorBaseStrategy

__all__ = [
    "FactorBaseStrategy",
    "EquityFactorStrategy",
]

"""Factor-based trading strategies."""

from .factor_base import FactorBaseStrategy
from .equity_factor import EquityFactorStrategy

__all__ = [
    "FactorBaseStrategy",
    "EquityFactorStrategy",
]


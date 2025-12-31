"""Trading strategies module.

This module provides:
- Strategy base classes and interfaces
- Strategy implementations (momentum, mean reversion, etc.)
- Strategy factory (registry) and loader
- Signal scoring and queue selection utilities
"""

# Strategy base and interface
from .base.strategy_interface import StrategyInterface
from .factor.equity_factor import EquityFactorStrategy
from .mean_reversion.equity_mean_reversion import EquityMeanReversionStrategy
from .momentum.crypto_momentum import CryptoMomentumStrategy

# Strategy implementations
from .momentum.equity_momentum import EquityMomentumStrategy
from .multi_timeframe.equity_mtf_strategy import EquityMultiTimeframeStrategy
from .pairs.pairs_strategy import PairsTradingStrategy
from .queue import select_signals_from_queue, violates_correlation_guard

# Signal scoring and queue selection
from .scoring import (
    compute_breakout_strength,
    compute_diversification_bonus,
    compute_momentum_strength,
    rank_normalize,
    score_signals,
)
from .strategy_loader import load_strategies_from_configs, load_strategies_from_run_config, load_strategy_from_config

# Strategy factory and loader
from .strategy_registry import create_strategy, get_strategy_class, list_available_strategies, register_strategy

# Backward compatibility aliases
EquityStrategy = EquityMomentumStrategy
CryptoStrategy = CryptoMomentumStrategy

__all__ = [
    # Base and interface
    "StrategyInterface",
    # Strategy implementations
    "EquityMomentumStrategy",
    "CryptoMomentumStrategy",
    "EquityMeanReversionStrategy",
    "EquityMultiTimeframeStrategy",
    "EquityFactorStrategy",
    "PairsTradingStrategy",
    # Backward compatibility
    "EquityStrategy",
    "CryptoStrategy",
    # Factory and loader
    "register_strategy",
    "get_strategy_class",
    "create_strategy",
    "list_available_strategies",
    "load_strategy_from_config",
    "load_strategies_from_configs",
    "load_strategies_from_run_config",
    # Signal scoring and queue
    "compute_breakout_strength",
    "compute_momentum_strength",
    "compute_diversification_bonus",
    "rank_normalize",
    "score_signals",
    "violates_correlation_guard",
    "select_signals_from_queue",
]

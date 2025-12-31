"""Trading strategies module.

This module provides:
- Strategy base classes and interfaces
- Strategy implementations (momentum, mean reversion, etc.)
- Strategy factory (registry) and loader
- Signal scoring and queue selection utilities
"""

# Strategy base and interface
from .base.strategy_interface import StrategyInterface

# Strategy implementations
from .momentum.equity_momentum import EquityMomentumStrategy
from .momentum.crypto_momentum import CryptoMomentumStrategy
from .mean_reversion.equity_mean_reversion import EquityMeanReversionStrategy
from .multi_timeframe.equity_mtf_strategy import EquityMultiTimeframeStrategy
from .factor.equity_factor import EquityFactorStrategy
from .pairs.pairs_strategy import PairsTradingStrategy

# Strategy factory and loader
from .strategy_registry import (
    register_strategy,
    get_strategy_class,
    create_strategy,
    list_available_strategies,
)
from .strategy_loader import (
    load_strategy_from_config,
    load_strategies_from_configs,
    load_strategies_from_run_config,
)

# Signal scoring and queue selection
from .scoring import (
    compute_breakout_strength,
    compute_momentum_strength,
    compute_diversification_bonus,
    rank_normalize,
    score_signals,
)
from .queue import (
    violates_correlation_guard,
    select_signals_from_queue,
)

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

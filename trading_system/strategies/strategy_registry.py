"""Strategy registry (factory) for instantiating strategies.

This module provides a factory pattern for creating strategy instances
based on strategy type and asset class. Strategies are registered by
their type name (e.g., "momentum") and asset class (e.g., "equity", "crypto").
"""

from typing import Dict, Optional, Type

from ..configs.strategy_config import StrategyConfig
from ..exceptions import StrategyError, StrategyNotFoundError
from .base.strategy_interface import StrategyInterface
from .factor.equity_factor import EquityFactorStrategy
from .mean_reversion.equity_mean_reversion import EquityMeanReversionStrategy
from .momentum.crypto_momentum import CryptoMomentumStrategy

# Import all available strategies
from .momentum.equity_momentum import EquityMomentumStrategy
from .multi_timeframe.equity_mtf_strategy import EquityMultiTimeframeStrategy

try:
    from .pairs.pairs_strategy import PairsTradingStrategy
except ImportError:
    PairsTradingStrategy = None  # type: ignore[misc, assignment]  # Pairs strategy may not exist yet


# Registry mapping: (strategy_type, asset_class) -> StrategyClass
_STRATEGY_REGISTRY: Dict[tuple[str, str], Type[StrategyInterface]] = {
    ("momentum", "equity"): EquityMomentumStrategy,
    ("momentum", "crypto"): CryptoMomentumStrategy,
    ("mean_reversion", "equity"): EquityMeanReversionStrategy,
    ("multi_timeframe", "equity"): EquityMultiTimeframeStrategy,
    ("factor", "equity"): EquityFactorStrategy,
}
if PairsTradingStrategy is not None:
    _STRATEGY_REGISTRY[("pairs", "equity")] = PairsTradingStrategy
    _STRATEGY_REGISTRY[("pairs", "crypto")] = PairsTradingStrategy  # Pairs can work for both


def register_strategy(strategy_type: str, asset_class: str, strategy_class: Type[StrategyInterface]) -> None:
    """Register a strategy class in the registry.

    Args:
        strategy_type: Strategy type (e.g., "momentum", "mean_reversion", "pairs")
        asset_class: Asset class (e.g., "equity", "crypto")
        strategy_class: Strategy class that implements StrategyInterface

    Raises:
        ValueError: If strategy_type or asset_class is invalid
    """
    if not strategy_type:
        raise ValueError("strategy_type cannot be empty")

    if asset_class not in ["equity", "crypto"]:
        raise ValueError(f"Invalid asset_class: {asset_class}, must be 'equity' or 'crypto'")

    if not issubclass(strategy_class, StrategyInterface):
        raise ValueError(f"Strategy class must implement StrategyInterface, got {strategy_class}")

    key = (strategy_type, asset_class)
    if key in _STRATEGY_REGISTRY:
        raise ValueError(f"Strategy already registered for type='{strategy_type}', asset_class='{asset_class}'")

    _STRATEGY_REGISTRY[key] = strategy_class


def get_strategy_class(strategy_type: str, asset_class: str) -> Optional[Type[StrategyInterface]]:
    """Get strategy class from registry.

    Args:
        strategy_type: Strategy type (e.g., "momentum", "mean_reversion")
        asset_class: Asset class (e.g., "equity", "crypto")

    Returns:
        Strategy class if found, None otherwise
    """
    key = (strategy_type, asset_class)
    return _STRATEGY_REGISTRY.get(key)


def create_strategy(config: StrategyConfig) -> StrategyInterface:
    """Create a strategy instance from configuration.

    This is the main factory method. It determines the strategy type
    from the config and instantiates the appropriate strategy class.

    Currently, strategy type is inferred from the config name or
    asset class. In the future, config may have an explicit "strategy_type" field.

    Args:
        config: Strategy configuration

    Returns:
        Strategy instance

    Raises:
        ValueError: If no strategy class is found for the given config
    """
    # For now, we infer strategy type from config name or default to "momentum"
    # In the future, config may have an explicit strategy_type field
    strategy_type = _infer_strategy_type(config)

    strategy_class = get_strategy_class(strategy_type, config.asset_class)

    if strategy_class is None:
        available = list(_STRATEGY_REGISTRY.keys())
        raise StrategyNotFoundError(
            f"No strategy found for type='{strategy_type}', asset_class='{config.asset_class}'. "
            f"Available strategies: {available}",
            strategy_name=strategy_type,
            symbol=None,
        )

    return strategy_class(config)


def _infer_strategy_type(config: StrategyConfig) -> str:
    """Infer strategy type from configuration.

    For now, we default to "momentum" since that's the only type implemented.
    In the future, this could check config.name or a new config.strategy_type field.

    Args:
        config: Strategy configuration

    Returns:
        Strategy type string
    """
    # Check if config name contains strategy type hints
    name_lower = config.name.lower()

    if "pairs" in name_lower or "pair" in name_lower:
        return "pairs"
    elif "mean_reversion" in name_lower or "mean-reversion" in name_lower or "meanreversion" in name_lower:
        return "mean_reversion"
    elif "momentum" in name_lower:
        return "momentum"
    elif "multi_timeframe" in name_lower or "mtf" in name_lower:
        return "multi_timeframe"
    elif "factor" in name_lower:
        return "factor"

    # Default to momentum for backward compatibility
    return "momentum"


def list_available_strategies() -> list[tuple[str, str]]:
    """List all available strategy types and asset classes.

    Returns:
        List of (strategy_type, asset_class) tuples
    """
    return list(_STRATEGY_REGISTRY.keys())

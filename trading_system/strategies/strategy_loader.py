"""Config-driven strategy loader.

This module provides utilities for loading strategies from configuration files.
It integrates with the strategy registry to instantiate strategies based on
their configuration.
"""

from typing import List, Optional
from pathlib import Path

from ..configs.strategy_config import StrategyConfig
from ..logging.logger import get_logger
from .strategy_registry import create_strategy
from .base.strategy_interface import StrategyInterface

logger = get_logger(__name__)


def load_strategy_from_config(config_path: str) -> StrategyInterface:
    """Load a single strategy from a configuration file.
    
    Args:
        config_path: Path to strategy configuration YAML file
        
    Returns:
        Strategy instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If strategy cannot be created from config
    """
    logger.info(f"Loading strategy from {config_path}")
    
    # Validate file exists
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Strategy config file not found: {config_path}")
    
    # Load configuration
    config = StrategyConfig.from_yaml(config_path)
    
    # Create strategy using registry
    strategy = create_strategy(config)
    
    logger.info(f"Loaded strategy: {config.name} (type={_infer_strategy_type(config)}, asset_class={config.asset_class})")
    
    return strategy


def load_strategies_from_configs(config_paths: List[str]) -> List[StrategyInterface]:
    """Load multiple strategies from configuration files.
    
    Args:
        config_paths: List of paths to strategy configuration YAML files
        
    Returns:
        List of strategy instances
        
    Raises:
        FileNotFoundError: If any config file doesn't exist
        ValueError: If any strategy cannot be created
    """
    strategies = []
    
    for config_path in config_paths:
        try:
            strategy = load_strategy_from_config(config_path)
            strategies.append(strategy)
        except Exception as e:
            logger.error(f"Failed to load strategy from {config_path}: {e}")
            raise
    
    return strategies


def load_strategies_from_run_config(
    equity_config_path: Optional[str] = None,
    crypto_config_path: Optional[str] = None
) -> List[StrategyInterface]:
    """Load strategies from run configuration pattern.
    
    This is a convenience function that matches the current pattern used
    in BacktestRunner and create_backtest_engine_from_config.
    
    Args:
        equity_config_path: Optional path to equity strategy config
        crypto_config_path: Optional path to crypto strategy config
        
    Returns:
        List of strategy instances (empty if no configs provided)
        
    Raises:
        ValueError: If no strategies are loaded
    """
    strategies = []
    
    if equity_config_path:
        strategy = load_strategy_from_config(equity_config_path)
        strategies.append(strategy)
    
    if crypto_config_path:
        strategy = load_strategy_from_config(crypto_config_path)
        strategies.append(strategy)
    
    if not strategies:
        raise ValueError("No strategy configs provided")
    
    return strategies


def _infer_strategy_type(config: StrategyConfig) -> str:
    """Infer strategy type from configuration (helper function).
    
    Args:
        config: Strategy configuration
        
    Returns:
        Strategy type string
    """
    name_lower = config.name.lower()
    
    if "momentum" in name_lower:
        return "momentum"
    elif "mean_reversion" in name_lower or "mean-reversion" in name_lower:
        return "mean_reversion"
    elif "pairs" in name_lower:
        return "pairs"
    elif "multi_timeframe" in name_lower or "mtf" in name_lower:
        return "multi_timeframe"
    elif "factor" in name_lower:
        return "factor"
    
    return "momentum"  # Default


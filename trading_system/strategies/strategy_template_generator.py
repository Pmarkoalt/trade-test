"""Strategy template generator for creating new strategy classes."""

from pathlib import Path
from typing import Dict, Optional

# Template for strategy class file
STRATEGY_TEMPLATE = '''"""{{strategy_name}} strategy implementation."""

from typing import List, Optional
import numpy as np

from {{base_import}}
from ...configs.strategy_config import StrategyConfig
from ...models.signals import BreakoutType
from ...models.features import FeatureRow
from ...models.positions import Position, ExitReason


class {{class_name}}({{base_class}}):
    """{{strategy_name}} strategy for {{asset_class}}.

    TODO: Add strategy description here.

    Eligibility:
    - TODO: Describe eligibility requirements

    Entry triggers:
    - TODO: Describe entry trigger conditions

    Exit logic:
    - TODO: Describe exit conditions

    Capacity check: order_notional <= max_pct * ADV20
    """

    def __init__(self, config: StrategyConfig):
        """Initialize {{strategy_name}} strategy.

        Args:
            config: Strategy configuration (must have asset_class="{{asset_class}}")
        """
        if config.asset_class != "{{asset_class}}":
            raise ValueError(
                f"{{class_name}} requires asset_class='{{asset_class}}', got '{{asset_class_actual}}'"
            )

        super().__init__(config)

    def check_eligibility(self, features: FeatureRow) -> tuple[bool, List[str]]:
        """Check if symbol is eligible for entry.

        Args:
            features: FeatureRow with indicators for the symbol

        Returns:
            Tuple of (is_eligible, failure_reasons)
        """
        failures = []

        # Check if features are valid
        if not features.is_valid_for_entry():
            failures.append("insufficient_data")
            return False, failures

        # TODO: Implement eligibility checks
        # Example:
        # if features.close <= features.ma50:
        #     failures.append("below_MA50")
        #     return False, failures

        return True, []

    def check_entry_triggers(
        self, features: FeatureRow
    ) -> tuple[Optional[BreakoutType], float]:
        """Check if entry triggers are met.

        Args:
            features: FeatureRow with indicators for the symbol

        Returns:
            Tuple of (breakout_type, clearance) or (None, 0.0) if no trigger.
            For non-breakout strategies, breakout_type may be a custom enum value.
        """
        # TODO: Implement entry trigger logic
        # Example for momentum:
        # fast_clearance = self.config.entry.fast_clearance
        # if features.highest_close_20d is not None:
        #     fast_threshold = features.highest_close_20d * (1 + fast_clearance)
        #     if features.close >= fast_threshold:
        #         clearance = (features.close / features.highest_close_20d) - 1
        #         return BreakoutType.FAST_20D, clearance

        return None, 0.0

    def check_exit_signals(
        self, position: Position, features: FeatureRow
    ) -> Optional[ExitReason]:
        """Check if position should be exited.

        Args:
            position: Open position to check
            features: FeatureRow with current indicators

        Returns:
            ExitReason if exit triggered, None otherwise
        """
        # TODO: Implement exit logic
        # Example:
        # if features.close < features.ma20:
        #     return ExitReason.TRAILING_STOP
        # if features.close < position.stop_price:
        #     return ExitReason.HARD_STOP

        return None

    def update_stop_price(
        self, position: Position, features: FeatureRow
    ) -> Optional[float]:
        """Update stop price for position (trailing stops, tightening).

        Args:
            position: Open position
            features: FeatureRow with current indicators

        Returns:
            New stop price if updated, None if unchanged
        """
        # TODO: Implement stop price update logic
        # Example for trailing stop:
        # atr_mult = self.config.exit.hard_stop_atr_mult
        # new_stop = features.close - (atr_mult * features.atr14)
        # if new_stop > position.stop_price:
        #     return new_stop

        return None
'''

# Base class mappings
BASE_CLASS_MAP: Dict[str, Dict[str, tuple[str, str]]] = {
    "momentum": {
        "equity": ("..momentum.momentum_base", "MomentumBaseStrategy"),
        "crypto": ("..momentum.momentum_base", "MomentumBaseStrategy"),
    },
    "mean_reversion": {
        "equity": ("..mean_reversion.mean_reversion_base", "MeanReversionBaseStrategy"),
        "crypto": ("..mean_reversion.mean_reversion_base", "MeanReversionBaseStrategy"),
    },
    "factor": {
        "equity": ("..factor.factor_base", "FactorBaseStrategy"),
        "crypto": ("..factor.factor_base", "FactorBaseStrategy"),
    },
    "multi_timeframe": {
        "equity": ("..multi_timeframe.mtf_strategy_base", "MultiTimeframeBaseStrategy"),
        "crypto": ("..multi_timeframe.mtf_strategy_base", "MultiTimeframeBaseStrategy"),
    },
    "pairs": {
        "equity": ("..base.strategy_interface", "StrategyInterface"),
        "crypto": ("..base.strategy_interface", "StrategyInterface"),
    },
    "custom": {
        "equity": ("..base.strategy_interface", "StrategyInterface"),
        "crypto": ("..base.strategy_interface", "StrategyInterface"),
    },
}


def generate_strategy_template(
    strategy_name: str,
    strategy_type: str = "custom",
    asset_class: str = "equity",
    output_path: Optional[str] = None,
    directory: Optional[str] = None,
) -> str:
    """Generate a strategy class template file.

    Args:
        strategy_name: Name of the strategy (e.g., "my_custom_strategy")
        strategy_type: Type of strategy (momentum, mean_reversion, factor,
                      multi_timeframe, pairs, or custom)
        asset_class: Asset class (equity or crypto)
        output_path: Optional path to save the template. If None, returns as string.
        directory: Optional directory to create the strategy in. If None, uses
                   strategy_type directory (or "custom" for custom strategies)

    Returns:
        Template Python code as string

    Raises:
        ValueError: If strategy_type or asset_class is invalid
    """
    if strategy_type not in BASE_CLASS_MAP:
        raise ValueError(f"Invalid strategy_type: {strategy_type}. " f"Must be one of: {', '.join(BASE_CLASS_MAP.keys())}")

    if asset_class not in ["equity", "crypto"]:
        raise ValueError(f"asset_class must be 'equity' or 'crypto', got '{asset_class}'")

    # Get base class info
    base_import_path, base_class_name = BASE_CLASS_MAP[strategy_type][asset_class]

    # Generate class name from strategy name
    # Pattern: {AssetClass}{StrategyType}Strategy (e.g., EquityMomentumStrategy)
    # Convert "my_custom_strategy" -> "MyCustomStrategy"
    class_name_parts = strategy_name.split("_")
    strategy_name_camel = "".join(word.capitalize() for word in class_name_parts)

    # Build class name following the pattern: {AssetClass}{StrategyType}Strategy
    if strategy_type == "custom":
        # For custom strategies, use: {AssetClass}{StrategyName}Strategy
        class_name = f"{asset_class.capitalize()}{strategy_name_camel}Strategy"
    else:
        # For typed strategies, use: {AssetClass}{StrategyType}Strategy
        # But if strategy_name is not the default, incorporate it
        strategy_type_camel = "".join(word.capitalize() for word in strategy_type.split("_"))
        if strategy_name_camel.lower() != strategy_type.lower():
            # Custom name provided, use: {AssetClass}{StrategyName}{StrategyType}Strategy
            class_name = f"{asset_class.capitalize()}{strategy_name_camel}{strategy_type_camel}Strategy"
        else:
            # Default name, use: {AssetClass}{StrategyType}Strategy
            class_name = f"{asset_class.capitalize()}{strategy_type_camel}Strategy"

    # Format template
    template_vars = {
        "strategy_name": strategy_name.replace("_", " ").title(),
        "class_name": class_name,
        "base_import": f"from {base_import_path} import {base_class_name}",
        "base_class": base_class_name,
        "asset_class": asset_class,
        "asset_class_actual": "{config.asset_class}",  # For error message formatting
    }

    # Replace template variables
    content = STRATEGY_TEMPLATE
    for key, value in template_vars.items():
        content = content.replace(f"{{{{{key}}}}}", str(value))

    # Determine output path
    if output_path:
        file_path = Path(output_path)
    else:
        # Determine directory
        if directory:
            strategy_dir = Path(directory)
        else:
            if strategy_type == "custom":
                strategy_dir = Path("trading_system/strategies/custom")
            else:
                strategy_dir = Path(f"trading_system/strategies/{strategy_type}")

        # Create directory if it doesn't exist
        strategy_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = f"{strategy_name}_{asset_class}.py"
        file_path = strategy_dir / filename

    # Save file
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        f.write(content)

    return content

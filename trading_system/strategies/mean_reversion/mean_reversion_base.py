"""Base class for mean reversion strategies with shared logic."""

from typing import List, Optional

import numpy as np

from ...configs.strategy_config import StrategyConfig
from ...models.features import FeatureRow
from ...models.positions import ExitReason, Position
from ...models.signals import BreakoutType
from ..base.strategy_interface import StrategyInterface


class MeanReversionBaseStrategy(StrategyInterface):
    """Base class for mean reversion strategies.

    Mean reversion strategies look for oversold conditions (price below mean)
    and enter long positions expecting price to revert to the mean.

    Shared logic:
    - Compute z-score (distance from mean in standard deviations)
    - Entry when z-score < -entry_std (oversold)
    - Exit when z-score >= -exit_std (reverted to mean) or time stop
    """

    def __init__(self, config: StrategyConfig):
        """Initialize mean reversion strategy with configuration.

        Args:
            config: Strategy configuration loaded from YAML
        """
        super().__init__(config)

        # Extract mean reversion parameters
        self.lookback = config.parameters.get("lookback", 20)
        self.entry_std = config.parameters.get("entry_std", 2.0)  # Enter at -2 std
        self.exit_std = config.parameters.get("exit_std", 0.0)  # Exit at mean (0 std)
        self.max_hold_days = config.parameters.get("max_hold_days", 5)
        self.atr_period = config.parameters.get("atr_period", 14)
        self.stop_atr_mult = config.parameters.get("stop_atr_mult", 2.0)

    def check_entry_triggers(self, features: FeatureRow) -> tuple[Optional[BreakoutType], float]:
        """Check if mean reversion entry trigger is met.

        Entry trigger: z-score < -entry_std (oversold condition)

        Args:
            features: FeatureRow with indicators for the symbol

        Returns:
            Tuple of (trigger_type, zscore) or (None, 0.0) if no trigger.
            For mean reversion, trigger_type is "mean_reversion_oversold"
        """
        # Check if zscore is available
        if not hasattr(features, "zscore") or features.zscore is None:
            return None, 0.0

        if np.isnan(features.zscore):
            return None, 0.0

        # Entry trigger: z-score below entry threshold (oversold)
        # For compatibility with StrategyInterface, we return BreakoutType-like value
        # but mean reversion doesn't use BreakoutType, so we'll handle this in generate_signal
        if features.zscore < -self.entry_std:
            # Return a placeholder - actual trigger type handled in generate_signal
            return BreakoutType.FAST_20D, features.zscore  # Reuse enum for compatibility

        # Return zscore even when no trigger (for informational purposes)
        return None, features.zscore

    def get_required_history_days(self) -> int:
        """Get minimum lookback period needed for indicators.

        Returns:
            Number of days (lookback + buffer)
        """
        return self.lookback + 20  # Buffer for stability

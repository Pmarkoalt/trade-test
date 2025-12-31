"""Base class for multi-timeframe strategies with shared logic."""

from typing import List, Optional

import numpy as np

from ...configs.strategy_config import StrategyConfig
from ...models.features import FeatureRow
from ...models.positions import ExitReason, Position
from ...models.signals import BreakoutType
from ..base.strategy_interface import StrategyInterface


class MultiTimeframeBaseStrategy(StrategyInterface):
    """Base class for multi-timeframe strategies.

    Multi-timeframe strategies use higher timeframe for trend filtering
    and lower timeframe for entry signals.

    For daily-only systems:
    - Higher timeframe: Daily MA50 (trend filter)
    - Lower timeframe: Weekly breakout (highest close over N weeks)

    Shared logic:
    - Entry: Price above MA50 (HTF trend) AND price >= weekly breakout level
    - Exit: Price breaks below MA50 (HTF trend break)
    """

    def __init__(self, config: StrategyConfig):
        """Initialize multi-timeframe strategy with configuration.

        Args:
            config: Strategy configuration loaded from YAML
        """
        super().__init__(config)

        # Extract multi-timeframe parameters
        self.higher_tf_ma = config.parameters.get("higher_tf_ma", 50)  # Daily MA50
        self.weekly_lookback = config.parameters.get("weekly_lookback", 4)  # 4 weeks
        self.weekly_days = config.parameters.get("weekly_days", 28)  # ~4 weeks in trading days
        self.atr_period = config.parameters.get("atr_period", 14)
        self.stop_atr_mult = config.parameters.get("stop_atr_mult", 2.0)
        self.max_hold_days = config.parameters.get("max_hold_days", 60)  # Longer hold for trend following

    def check_entry_triggers(self, features: FeatureRow) -> tuple[Optional[BreakoutType], float]:
        """Check if multi-timeframe entry trigger is met.

        Entry trigger:
        1. Price above higher timeframe MA (trend filter)
        2. Price >= weekly breakout level (lower timeframe entry)

        Args:
            features: FeatureRow with indicators for the symbol

        Returns:
            Tuple of (breakout_type, clearance) or (None, 0.0) if no trigger.
            Uses BreakoutType.SLOW_55D as placeholder for MTF breakout.
        """
        # Check if required indicators are available
        if features.ma50 is None or np.isnan(features.ma50):
            return None, 0.0

        # For daily-only system, use highest_close_55d as proxy for weekly breakout
        # (55 days ≈ 11 weeks, but we'll use it as approximation)
        # In a true MTF system with intraday data, this would be computed differently
        weekly_breakout = features.highest_close_55d
        if weekly_breakout is None or np.isnan(weekly_breakout):
            return None, 0.0

        # Higher timeframe trend filter: price must be above MA50
        if features.close <= features.ma50:
            return None, 0.0

        # Lower timeframe breakout: price must be >= weekly breakout level
        if features.close >= weekly_breakout:
            # Calculate clearance (how much above breakout level)
            clearance = (features.close - weekly_breakout) / weekly_breakout if weekly_breakout > 0 else 0.0
            return BreakoutType.SLOW_55D, clearance  # Reuse enum for compatibility

        return None, 0.0

    def get_required_history_days(self) -> int:
        """Get minimum lookback period needed for indicators.

        Returns:
            Number of days (higher_tf_ma + weekly_days + buffer)
        """
        return self.higher_tf_ma + self.weekly_days + 20  # Buffer for stability

"""Base class for momentum strategies with shared logic."""

from typing import Optional

import numpy as np

from ...configs.strategy_config import StrategyConfig
from ...models.features import FeatureRow
from ...models.signals import BreakoutType
from ..base.strategy_interface import StrategyInterface


class MomentumBaseStrategy(StrategyInterface):
    """Base class for momentum strategies (equity and crypto).

    This class contains shared logic for momentum strategies, including:
    - Common entry trigger logic (breakout detection)
    - Shared eligibility checks
    - Common scoring calculations

    Subclasses should implement asset-class-specific logic:
    - Eligibility filters (MA50 vs MA200, slope requirements, etc.)
    - Exit logic (trailing vs staged)
    - Stop price updates
    """

    def __init__(self, config: StrategyConfig):
        """Initialize momentum strategy with configuration.

        Args:
            config: Strategy configuration loaded from YAML
        """
        super().__init__(config)

    def check_entry_triggers(self, features: FeatureRow) -> tuple[Optional[BreakoutType], float]:
        """Check if entry triggers are met (shared momentum logic).

        Entry triggers (OR logic):
        - Fast: close >= highest_close_20d * (1 + fast_clearance)
        - Slow: close >= highest_close_55d * (1 + slow_clearance)

        Args:
            features: FeatureRow with indicators for the symbol

        Returns:
            Tuple of (breakout_type, clearance) or (None, 0.0) if no trigger
        """
        fast_clearance = self.config.entry.fast_clearance
        slow_clearance = self.config.entry.slow_clearance

        # Check fast trigger (20D) - prioritize fast over slow
        if features.highest_close_20d is not None and not np.isnan(features.highest_close_20d):
            fast_threshold = features.highest_close_20d * (1 + fast_clearance)
            if features.close >= fast_threshold:
                clearance = (features.close / features.highest_close_20d) - 1
                return BreakoutType.FAST_20D, clearance

        # Check slow trigger (55D)
        if features.highest_close_55d is not None and not np.isnan(features.highest_close_55d):
            slow_threshold = features.highest_close_55d * (1 + slow_clearance)
            if features.close >= slow_threshold:
                clearance = (features.close / features.highest_close_55d) - 1
                return BreakoutType.SLOW_55D, clearance

        return None, 0.0

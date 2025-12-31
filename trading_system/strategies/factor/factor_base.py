"""Base class for factor-based strategies with shared logic."""

from datetime import datetime
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from ...configs.strategy_config import StrategyConfig
from ...models.features import FeatureRow
from ...models.positions import ExitReason, Position
from ...models.signals import BreakoutType
from ..base.strategy_interface import StrategyInterface


class FactorBaseStrategy(StrategyInterface):
    """Base class for factor-based strategies.

    Factor strategies rank assets by multiple factors (value, momentum, quality)
    and go long the top decile. Rebalances on a schedule (monthly/quarterly).

    Shared logic:
    - Compute factor scores (momentum, value, quality)
    - Rank assets by composite factor score
    - Enter top decile, exit positions not in top decile
    """

    def __init__(self, config: StrategyConfig):
        """Initialize factor strategy with configuration.

        Args:
            config: Strategy configuration loaded from YAML
        """
        super().__init__(config)

        # Factor weights
        parameters = config.parameters or {}
        factors_config = parameters.get("factors", {})
        self.factors = {
            "momentum": factors_config.get("momentum", 0.4),
            "value": factors_config.get("value", 0.3),
            "quality": factors_config.get("quality", 0.3),
        }

        # Rebalancing parameters
        self.rebalance_frequency = parameters.get("rebalance_frequency", "monthly")
        self.top_decile_pct = parameters.get("top_decile_pct", 0.20)  # top 20%

        # State tracking for rebalancing
        self._current_rebalance_date: Optional[datetime] = None
        self._top_decile_symbols: Set[str] = set()
        self._factor_scores_cache: Dict[str, float] = {}  # symbol -> factor_score

        # Risk management
        self.atr_period = parameters.get("atr_period", 14)
        self.stop_atr_mult = parameters.get("stop_atr_mult", 2.0)

    def compute_factor_score(self, features: FeatureRow) -> Optional[float]:
        """Compute composite factor score for a symbol.

        Factors:
        - Momentum: 12-month return (ROC252)
        - Value: Distance from 52W high (proxy for value)
        - Quality: Inverse volatility (lower vol = higher quality)

        Args:
            features: FeatureRow with indicators for the symbol

        Returns:
            Composite factor score, or None if insufficient data
        """
        # This will be computed per-symbol, but ranking happens across universe
        # For now, return None - actual computation in subclass
        return None

    def _is_rebalance_day(self, date: datetime) -> bool:
        """Check if this is a rebalance date.

        Args:
            date: Date to check

        Returns:
            True if this is a rebalance day
        """
        if self.rebalance_frequency == "monthly":
            # First trading day of month (within first 5 days)
            return date.day <= 5
        elif self.rebalance_frequency == "quarterly":
            # First trading day of quarter (Jan, Apr, Jul, Oct, within first 5 days)
            return date.month in [1, 4, 7, 10] and date.day <= 5
        else:
            return False

    def check_entry_triggers(self, features: FeatureRow) -> tuple[Optional[BreakoutType], float]:
        """Check if factor-based entry trigger is met.

        Entry trigger: Symbol is in top decile on rebalance day.

        Args:
            features: FeatureRow with indicators for the symbol

        Returns:
            Tuple of (trigger_type, factor_score) or (None, 0.0) if no trigger.
        """
        # Check if it's a rebalance day
        if not self._is_rebalance_day(features.date.to_pydatetime()):
            return None, 0.0

        # Compute factor score
        factor_score = self.compute_factor_score(features)
        if factor_score is None or np.isnan(factor_score):
            return None, 0.0

        # Store in cache for ranking
        self._factor_scores_cache[features.symbol] = factor_score

        # Check if symbol is in top decile (will be determined after all symbols processed)
        # For now, return trigger if score is positive (will be refined)
        # Actual ranking happens in generate_signal after all scores collected
        return BreakoutType.FAST_20D, factor_score  # Reuse enum for compatibility

    def get_required_history_days(self) -> int:
        """Get minimum lookback period needed for indicators.

        Returns:
            Number of days (252 for 12-month momentum + buffer)
        """
        return 252 + 20  # 1 year + buffer

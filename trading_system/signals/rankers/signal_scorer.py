"""Signal scorer for ranking signals."""

from typing import Dict, List, Tuple

import numpy as np

from ...models.features import FeatureRow
from ...models.signals import BreakoutType, Signal
from ..config import SignalConfig


class SignalScorer:
    """Score and rank signals."""

    def __init__(self, config: SignalConfig):
        """Initialize signal scorer.

        Args:
            config: Signal configuration
        """
        self.config = config

    def score_signals(
        self,
        signals: List[Signal],
        features: Dict[str, FeatureRow],
    ) -> List[Tuple[Signal, float]]:
        """Score each signal and return sorted list.

        Args:
            signals: List of signals to score
            features: Dictionary mapping symbol to FeatureRow

        Returns:
            List of (signal, score) tuples sorted by score descending (0-10 scale)
        """
        scored = []

        for signal in signals:
            feature = features.get(signal.symbol)
            if not feature:
                continue

            # Calculate component scores (0-10 scale)
            breakout_score = self._score_breakout(signal, feature)
            momentum_score = self._score_momentum(feature)

            # Combined score (Phase 1: technical only)
            combined = breakout_score * 0.6 + momentum_score * 0.4

            scored.append((signal, combined))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _score_breakout(self, signal: Signal, feature: FeatureRow) -> float:
        """Score breakout strength (0-10).

        Args:
            signal: Signal object
            feature: FeatureRow with indicators

        Returns:
            Breakout score (0-10 scale)
        """
        if feature.atr14 is None or feature.atr14 <= 0:
            return 0.0

        # Breakout clearance above high
        if signal.triggered_on == BreakoutType.FAST_20D:
            if feature.highest_close_20d is None:
                return 0.0
            clearance = (feature.close - feature.highest_close_20d) / feature.atr14
        elif signal.triggered_on == BreakoutType.SLOW_55D:
            if feature.highest_close_55d is None:
                return 0.0
            clearance = (feature.close - feature.highest_close_55d) / feature.atr14
        else:
            # For non-breakout strategies, use breakout_strength if available
            if signal.breakout_strength > 0:
                clearance = signal.breakout_strength
            else:
                return 0.0

        # Normalize to 0-10 (1 ATR clearance = 10)
        return min(clearance * 10, 10.0)

    def _score_momentum(self, feature: FeatureRow) -> float:
        """Score momentum strength (0-10).

        Args:
            feature: FeatureRow with indicators

        Returns:
            Momentum score (0-10 scale)
        """
        if feature.roc60 is None or np.isnan(feature.roc60) or feature.roc60 <= 0:
            return 0.0

        # ROC60 normalized (20% = 10)
        # Formula: min(roc60 * 50, 10)
        # This means: 0.20 (20%) * 50 = 10.0
        return min(feature.roc60 * 50, 10.0)

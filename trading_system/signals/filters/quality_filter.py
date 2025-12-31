"""Quality filter for signals."""

from dataclasses import dataclass
from typing import List, Optional
import logging

from ...models.signals import Signal

logger = logging.getLogger(__name__)


@dataclass
class QualityFilterConfig:
    """Configuration for quality filtering."""

    # Minimum combined score (0-10)
    min_combined_score: float = 5.0

    # Minimum reward-to-risk ratio
    min_reward_risk_ratio: float = 1.5

    # Maximum ATR multiple for stop distance
    max_stop_atr_multiple: float = 3.0

    # Minimum ATR multiple for stop distance (too tight = noise)
    min_stop_atr_multiple: float = 0.5

    # Minimum conviction level (LOW=1, MEDIUM=2, HIGH=3)
    min_conviction_level: int = 1

    # Maximum spread percentage (for liquidity check)
    max_spread_pct: float = 0.02

    # Minimum volume (average daily volume)
    min_avg_volume: int = 100000

    # Whether to filter signals near major support/resistance
    filter_near_sr: bool = False

    # Distance from S/R to filter (in ATR)
    sr_distance_atr: float = 0.5


class QualityFilter:
    """Filter signals based on quality metrics."""

    def __init__(self, config: Optional[QualityFilterConfig] = None):
        """Initialize quality filter.

        Args:
            config: Filter configuration
        """
        self.config = config or QualityFilterConfig()

    def filter(self, signals: List[Signal]) -> List[Signal]:
        """Filter signals based on quality criteria.

        Args:
            signals: List of signals to filter

        Returns:
            Filtered list of high-quality signals
        """
        filtered = []

        for signal in signals:
            if self._passes_quality_check(signal):
                filtered.append(signal)
            else:
                logger.debug(f"Signal {signal.symbol} filtered out by quality check")

        logger.info(f"Quality filter: {len(filtered)}/{len(signals)} signals passed")
        return filtered

    def _passes_quality_check(self, signal: Signal) -> bool:
        """Check if a signal passes all quality criteria.

        Args:
            signal: Signal to check

        Returns:
            True if signal passes, False otherwise
        """
        # Check combined score
        if signal.combined_score < self.config.min_combined_score:
            logger.debug(f"{signal.symbol}: Score {signal.combined_score} < {self.config.min_combined_score}")
            return False

        # Check reward-to-risk ratio
        rr_ratio = self._calculate_reward_risk(signal)
        if rr_ratio < self.config.min_reward_risk_ratio:
            logger.debug(f"{signal.symbol}: R:R {rr_ratio:.2f} < {self.config.min_reward_risk_ratio}")
            return False

        # Check stop distance (ATR multiples)
        if hasattr(signal, "atr") and signal.atr > 0:
            stop_distance_atr = self._calculate_stop_atr_multiple(signal)
            if stop_distance_atr > self.config.max_stop_atr_multiple:
                logger.debug(f"{signal.symbol}: Stop distance {stop_distance_atr:.2f} ATR too wide")
                return False
            if stop_distance_atr < self.config.min_stop_atr_multiple:
                logger.debug(f"{signal.symbol}: Stop distance {stop_distance_atr:.2f} ATR too tight")
                return False

        # Check conviction level
        conviction_value = self._conviction_to_value(signal.conviction)
        if conviction_value < self.config.min_conviction_level:
            logger.debug(f"{signal.symbol}: Conviction {signal.conviction} below minimum")
            return False

        return True

    def _calculate_reward_risk(self, signal: Signal) -> float:
        """Calculate reward-to-risk ratio.

        Args:
            signal: Signal

        Returns:
            Reward-to-risk ratio
        """
        if signal.entry_price <= 0:
            return 0.0

        if signal.direction == "BUY":
            risk = abs(signal.entry_price - signal.stop_price)
            reward = abs(signal.target_price - signal.entry_price)
        else:
            risk = abs(signal.stop_price - signal.entry_price)
            reward = abs(signal.entry_price - signal.target_price)

        if risk <= 0:
            return float("inf")

        return reward / risk

    def _calculate_stop_atr_multiple(self, signal: Signal) -> float:
        """Calculate stop distance in ATR multiples.

        Args:
            signal: Signal

        Returns:
            Stop distance in ATR
        """
        if not hasattr(signal, "atr") or signal.atr <= 0:
            return 0.0

        stop_distance = abs(signal.entry_price - signal.stop_price)
        return stop_distance / signal.atr

    def _conviction_to_value(self, conviction: str) -> int:
        """Convert conviction string to numeric value.

        Args:
            conviction: Conviction level string

        Returns:
            Numeric value (1=LOW, 2=MEDIUM, 3=HIGH)
        """
        mapping = {
            "LOW": 1,
            "MEDIUM": 2,
            "HIGH": 3,
        }
        return mapping.get(conviction.upper(), 1)

    def get_quality_score(self, signal: Signal) -> float:
        """Get a quality score for a signal.

        Args:
            signal: Signal to score

        Returns:
            Quality score (0 to 1)
        """
        scores = []

        # Score from combined score (normalized to 0-1)
        score_normalized = min(signal.combined_score / 10.0, 1.0)
        scores.append(score_normalized)

        # Score from R:R ratio
        rr = self._calculate_reward_risk(signal)
        rr_score = min(rr / 3.0, 1.0)  # 3:1 = perfect
        scores.append(rr_score)

        # Score from conviction
        conviction_score = self._conviction_to_value(signal.conviction) / 3.0
        scores.append(conviction_score)

        # Average all scores
        return sum(scores) / len(scores)

    def rank_by_quality(
        self,
        signals: List[Signal],
        top_k: Optional[int] = None,
    ) -> List[Signal]:
        """Rank signals by quality score.

        Args:
            signals: List of signals
            top_k: Return only top K signals

        Returns:
            Sorted list of signals (highest quality first)
        """
        scored = [(signal, self.get_quality_score(signal)) for signal in signals]
        sorted_signals = sorted(scored, key=lambda x: x[1], reverse=True)

        result = [signal for signal, _ in sorted_signals]

        if top_k:
            return result[:top_k]
        return result

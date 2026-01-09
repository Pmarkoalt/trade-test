"""
Dynamic Position Sizing Module

ML-based confidence weighting for position sizes:
- Adjust position size based on ML prediction confidence
- Scale by regime detection
- Risk-adjusted sizing with drawdown protection
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SizingMode(Enum):
    """Position sizing modes."""

    FIXED = "fixed"  # Fixed percentage
    CONFIDENCE = "confidence"  # Scale by ML confidence
    KELLY = "kelly"  # Kelly criterion
    VOLATILITY = "volatility"  # Inverse volatility
    REGIME = "regime"  # Regime-adjusted
    DYNAMIC = "dynamic"  # Combine all factors


@dataclass
class SizingResult:
    """Position sizing calculation result."""

    base_size: float  # Base position size (% of equity)
    adjusted_size: float  # Final adjusted size
    confidence_multiplier: float
    volatility_multiplier: float
    regime_multiplier: float
    drawdown_multiplier: float
    factors: Dict[str, float]


class DynamicPositionSizer:
    """Calculate position sizes using ML confidence and market conditions."""

    def __init__(
        self,
        mode: SizingMode = SizingMode.DYNAMIC,
        base_risk_per_trade: float = 0.01,
        max_position_size: float = 0.10,
        min_position_size: float = 0.005,
        confidence_scale: float = 1.5,
        drawdown_threshold: float = 0.10,
        drawdown_scale: float = 0.5,
    ):
        """
        Initialize dynamic position sizer.

        Args:
            mode: Sizing mode to use
            base_risk_per_trade: Base risk per trade (% of equity)
            max_position_size: Maximum position size (% of equity)
            min_position_size: Minimum position size (% of equity)
            confidence_scale: How much to scale by confidence (1.0 = no scaling)
            drawdown_threshold: Drawdown % at which to start reducing size
            drawdown_scale: Size multiplier when at drawdown threshold
        """
        self.mode = mode
        self.base_risk_per_trade = base_risk_per_trade
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.confidence_scale = confidence_scale
        self.drawdown_threshold = drawdown_threshold
        self.drawdown_scale = drawdown_scale

        # Historical tracking
        self.sizing_history: List[SizingResult] = []
        self.win_rate_history: List[float] = []

    def calculate_size(
        self,
        equity: float,
        ml_confidence: float = 0.5,
        current_volatility: float = 0.02,
        historical_volatility: float = 0.02,
        current_drawdown: float = 0.0,
        regime: str = "unknown",
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
    ) -> SizingResult:
        """
        Calculate position size based on multiple factors.

        Args:
            equity: Current portfolio equity
            ml_confidence: ML model confidence (0-1)
            current_volatility: Current market volatility
            historical_volatility: Historical avg volatility
            current_drawdown: Current drawdown from peak (0-1)
            regime: Current market regime
            win_rate: Historical win rate
            avg_win: Average winning trade
            avg_loss: Average losing trade

        Returns:
            SizingResult with position size and factors
        """
        base_size = self.base_risk_per_trade
        factors = {}

        # Confidence multiplier
        confidence_mult = self._confidence_multiplier(ml_confidence)
        factors["confidence"] = confidence_mult

        # Volatility multiplier
        volatility_mult = self._volatility_multiplier(current_volatility, historical_volatility)
        factors["volatility"] = volatility_mult

        # Regime multiplier
        regime_mult = self._regime_multiplier(regime)
        factors["regime"] = regime_mult

        # Drawdown multiplier
        drawdown_mult = self._drawdown_multiplier(current_drawdown)
        factors["drawdown"] = drawdown_mult

        # Kelly criterion (if we have win rate data)
        if win_rate is not None and avg_win is not None and avg_loss is not None:
            kelly_mult = self._kelly_multiplier(win_rate, avg_win, avg_loss)
            factors["kelly"] = kelly_mult
        else:
            kelly_mult = 1.0

        # Calculate final size based on mode
        if self.mode == SizingMode.FIXED:
            adjusted_size = base_size
        elif self.mode == SizingMode.CONFIDENCE:
            adjusted_size = base_size * confidence_mult
        elif self.mode == SizingMode.VOLATILITY:
            adjusted_size = base_size * volatility_mult
        elif self.mode == SizingMode.REGIME:
            adjusted_size = base_size * regime_mult
        elif self.mode == SizingMode.KELLY:
            adjusted_size = base_size * kelly_mult
        elif self.mode == SizingMode.DYNAMIC:
            # Combine all factors
            adjusted_size = (
                base_size * (confidence_mult * volatility_mult * regime_mult * drawdown_mult * kelly_mult) ** 0.5
            )  # Geometric mean to avoid extreme values
        else:
            adjusted_size = base_size

        # Apply bounds
        adjusted_size = np.clip(
            adjusted_size,
            self.min_position_size,
            self.max_position_size,
        )

        result = SizingResult(
            base_size=base_size,
            adjusted_size=adjusted_size,
            confidence_multiplier=confidence_mult,
            volatility_multiplier=volatility_mult,
            regime_multiplier=regime_mult,
            drawdown_multiplier=drawdown_mult,
            factors=factors,
        )

        self.sizing_history.append(result)

        return result

    def _confidence_multiplier(self, confidence: float) -> float:
        """Calculate confidence-based multiplier."""
        # Scale from 0.5 to confidence_scale based on confidence
        # confidence=0.5 -> mult=1.0
        # confidence=1.0 -> mult=confidence_scale
        # confidence=0.0 -> mult=0.5
        if confidence >= 0.5:
            return 1.0 + (confidence - 0.5) * 2 * (self.confidence_scale - 1)
        else:
            return 0.5 + confidence

    def _volatility_multiplier(self, current_vol: float, historical_vol: float) -> float:
        """Calculate volatility-based multiplier (inverse)."""
        if historical_vol == 0:
            return 1.0

        vol_ratio = current_vol / historical_vol

        # Higher volatility = smaller position
        # vol_ratio=1.0 -> mult=1.0
        # vol_ratio=2.0 -> mult=0.5
        # vol_ratio=0.5 -> mult=1.5 (capped)
        mult = 1.0 / vol_ratio
        return np.clip(mult, 0.3, 1.5)

    def _regime_multiplier(self, regime: str) -> float:
        """Calculate regime-based multiplier."""
        regime_multipliers = {
            "bull": 1.2,
            "bear": 0.5,
            "sideways": 0.8,
            "high_volatility": 0.3,
            "unknown": 0.7,
        }
        return regime_multipliers.get(regime.lower(), 1.0)

    def _drawdown_multiplier(self, current_drawdown: float) -> float:
        """Calculate drawdown-based multiplier."""
        if current_drawdown < self.drawdown_threshold:
            return 1.0

        # Linear reduction from 1.0 to drawdown_scale
        # as drawdown goes from threshold to 2x threshold
        excess_dd = current_drawdown - self.drawdown_threshold
        reduction = excess_dd / self.drawdown_threshold
        mult = 1.0 - (1.0 - self.drawdown_scale) * min(reduction, 1.0)

        return max(mult, self.drawdown_scale)

    def _kelly_multiplier(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly criterion multiplier."""
        if avg_loss == 0:
            return 1.0

        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
        b = avg_win / abs(avg_loss)
        p = win_rate
        q = 1 - win_rate

        kelly = (b * p - q) / b if b > 0 else 0

        # Use fractional Kelly (half Kelly) for safety
        kelly = kelly * 0.5

        # Clip to reasonable range
        return np.clip(kelly + 0.5, 0.3, 2.0)

    def get_sizing_stats(self) -> Dict[str, float]:
        """Get statistics on position sizing."""
        if not self.sizing_history:
            return {}

        sizes = [r.adjusted_size for r in self.sizing_history]
        confidences = [r.confidence_multiplier for r in self.sizing_history]

        return {
            "avg_size": np.mean(sizes),
            "std_size": np.std(sizes),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "avg_confidence_mult": np.mean(confidences),
            "times_at_max": sum(1 for s in sizes if s >= self.max_position_size * 0.99),
            "times_at_min": sum(1 for s in sizes if s <= self.min_position_size * 1.01),
        }

    def generate_report(self) -> str:
        """Generate sizing report."""
        stats = self.get_sizing_stats()

        lines = []
        lines.append("=" * 60)
        lines.append("DYNAMIC POSITION SIZING REPORT")
        lines.append("=" * 60)
        lines.append(f"Mode: {self.mode.value}")
        lines.append(f"Base Risk: {self.base_risk_per_trade:.2%}")
        lines.append(f"Size Range: {self.min_position_size:.2%} - {self.max_position_size:.2%}")
        lines.append("")

        if stats:
            lines.append("SIZING STATISTICS")
            lines.append("-" * 40)
            lines.append(f"Avg Size: {stats['avg_size']:.2%}")
            lines.append(f"Std Dev: {stats['std_size']:.2%}")
            lines.append(f"Range: {stats['min_size']:.2%} - {stats['max_size']:.2%}")
            lines.append(f"Avg Confidence Mult: {stats['avg_confidence_mult']:.2f}")
            lines.append(f"Times at Max: {stats['times_at_max']}")
            lines.append(f"Times at Min: {stats['times_at_min']}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

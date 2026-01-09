"""
Market Regime Detection Module

Classifies market conditions into regimes:
- Bull: Strong uptrend with low volatility
- Bear: Downtrend with increasing volatility
- Sideways: Range-bound with normal volatility
- High Volatility: Any trend with extreme volatility

Used for:
- Adjusting position sizes based on regime
- Filtering signals in unfavorable regimes
- Adapting strategy parameters dynamically
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    UNKNOWN = "unknown"


@dataclass
class RegimeState:
    """Current regime state with metadata."""

    regime: MarketRegime
    confidence: float  # 0-1 confidence in classification
    trend_strength: float  # -1 to 1 (bearish to bullish)
    volatility_percentile: float  # 0-100 percentile of volatility
    days_in_regime: int
    start_date: pd.Timestamp


class RegimeDetector:
    """Detect market regimes from price data."""

    def __init__(
        self,
        trend_lookback: int = 50,
        volatility_lookback: int = 20,
        volatility_threshold_pct: float = 80,
        trend_threshold: float = 0.02,
    ):
        """
        Initialize regime detector.

        Args:
            trend_lookback: Days for trend calculation
            volatility_lookback: Days for volatility calculation
            volatility_threshold_pct: Percentile above which is "high volatility"
            trend_threshold: Minimum slope for trend classification
        """
        self.trend_lookback = trend_lookback
        self.volatility_lookback = volatility_lookback
        self.volatility_threshold_pct = volatility_threshold_pct
        self.trend_threshold = trend_threshold

        self._regime_history: List[RegimeState] = []
        self._current_regime: Optional[RegimeState] = None

    def detect_regime(
        self,
        prices: pd.Series,
        benchmark_prices: Optional[pd.Series] = None,
    ) -> RegimeState:
        """
        Detect current market regime.

        Args:
            prices: Price series (typically benchmark or portfolio)
            benchmark_prices: Optional separate benchmark for comparison

        Returns:
            RegimeState with current classification
        """
        if len(prices) < self.trend_lookback:
            return RegimeState(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                trend_strength=0.0,
                volatility_percentile=50.0,
                days_in_regime=0,
                start_date=prices.index[-1] if len(prices) > 0 else pd.Timestamp.now(),
            )

        # Calculate trend
        trend_strength = self._calculate_trend_strength(prices)

        # Calculate volatility
        volatility_pct = self._calculate_volatility_percentile(prices)

        # Classify regime
        regime, confidence = self._classify_regime(trend_strength, volatility_pct)

        # Track regime persistence
        days_in_regime = self._calculate_regime_persistence(regime)
        start_date = self._get_regime_start_date(regime, prices.index[-1])

        state = RegimeState(
            regime=regime,
            confidence=confidence,
            trend_strength=trend_strength,
            volatility_percentile=volatility_pct,
            days_in_regime=days_in_regime,
            start_date=start_date,
        )

        self._current_regime = state
        self._regime_history.append(state)

        return state

    def _calculate_trend_strength(self, prices: pd.Series) -> float:
        """Calculate trend strength from -1 (bearish) to 1 (bullish)."""
        recent = prices.iloc[-self.trend_lookback :]

        # Calculate slope using linear regression
        x = np.arange(len(recent))
        y = recent.values

        # Normalize by mean price
        mean_price = y.mean()
        if mean_price == 0:
            return 0.0

        # Linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        normalized_slope = slope / mean_price * self.trend_lookback

        # Also consider MA relationships
        ma_short = recent.iloc[-10:].mean()
        ma_long = recent.mean()
        ma_signal = (ma_short - ma_long) / ma_long if ma_long != 0 else 0

        # Combine signals
        combined = (normalized_slope + ma_signal) / 2

        # Clip to [-1, 1]
        return np.clip(combined * 10, -1, 1)

    def _calculate_volatility_percentile(self, prices: pd.Series) -> float:
        """Calculate current volatility as percentile of historical."""
        returns = prices.pct_change().dropna()

        if len(returns) < self.volatility_lookback * 2:
            return 50.0

        # Current volatility (recent window)
        current_vol = returns.iloc[-self.volatility_lookback :].std()

        # Rolling volatility for historical comparison
        rolling_vol = returns.rolling(self.volatility_lookback).std().dropna()

        if len(rolling_vol) == 0 or current_vol == 0:
            return 50.0

        # Calculate percentile
        percentile = (rolling_vol < current_vol).sum() / len(rolling_vol) * 100

        return percentile

    def _classify_regime(self, trend_strength: float, volatility_pct: float) -> Tuple[MarketRegime, float]:
        """Classify regime based on trend and volatility."""

        # High volatility overrides other classifications
        if volatility_pct > self.volatility_threshold_pct:
            confidence = min(1.0, (volatility_pct - self.volatility_threshold_pct) / 20)
            return MarketRegime.HIGH_VOLATILITY, confidence

        # Trend-based classification
        if trend_strength > self.trend_threshold:
            confidence = min(1.0, abs(trend_strength) / 0.5)
            return MarketRegime.BULL, confidence
        elif trend_strength < -self.trend_threshold:
            confidence = min(1.0, abs(trend_strength) / 0.5)
            return MarketRegime.BEAR, confidence
        else:
            # Sideways - confidence inversely related to trend strength
            confidence = 1.0 - abs(trend_strength) / self.trend_threshold
            return MarketRegime.SIDEWAYS, confidence

    def _calculate_regime_persistence(self, regime: MarketRegime) -> int:
        """Calculate how many days we've been in current regime."""
        if not self._regime_history:
            return 1

        count = 0
        for state in reversed(self._regime_history):
            if state.regime == regime:
                count += 1
            else:
                break

        return count + 1

    def _get_regime_start_date(self, regime: MarketRegime, current_date: pd.Timestamp) -> pd.Timestamp:
        """Get start date of current regime."""
        if not self._regime_history:
            return current_date

        for state in reversed(self._regime_history):
            if state.regime != regime:
                return current_date

        return self._regime_history[0].start_date

    def get_position_size_multiplier(self, regime: MarketRegime) -> float:
        """Get recommended position size multiplier for regime."""
        multipliers = {
            MarketRegime.BULL: 1.2,  # Increase size in uptrends
            MarketRegime.BEAR: 0.5,  # Reduce size in downtrends
            MarketRegime.SIDEWAYS: 0.8,  # Slightly reduce in choppy markets
            MarketRegime.HIGH_VOLATILITY: 0.3,  # Significantly reduce in high vol
            MarketRegime.UNKNOWN: 0.5,  # Conservative when uncertain
        }
        return multipliers.get(regime, 1.0)

    def should_trade(self, regime: MarketRegime, strategy_type: str = "momentum") -> bool:
        """Determine if trading is advisable in current regime."""
        if strategy_type == "momentum":
            # Momentum works best in trending markets
            return regime in [MarketRegime.BULL, MarketRegime.BEAR]
        elif strategy_type == "mean_reversion":
            # Mean reversion works in sideways markets
            return regime == MarketRegime.SIDEWAYS
        else:
            # Default: trade in all except high volatility
            return regime != MarketRegime.HIGH_VOLATILITY

    def get_regime_features(self, prices: pd.Series) -> Dict[str, float]:
        """Get regime-based features for ML model."""
        state = self.detect_regime(prices)

        return {
            "regime_bull": 1.0 if state.regime == MarketRegime.BULL else 0.0,
            "regime_bear": 1.0 if state.regime == MarketRegime.BEAR else 0.0,
            "regime_sideways": 1.0 if state.regime == MarketRegime.SIDEWAYS else 0.0,
            "regime_high_vol": 1.0 if state.regime == MarketRegime.HIGH_VOLATILITY else 0.0,
            "regime_confidence": state.confidence,
            "trend_strength": state.trend_strength,
            "volatility_percentile": state.volatility_percentile / 100.0,
            "days_in_regime": min(state.days_in_regime / 30, 1.0),  # Normalize to ~1 month
        }

    def get_regime_summary(self, prices: pd.Series) -> str:
        """Get human-readable regime summary."""
        state = self.detect_regime(prices)

        regime_emoji = {
            MarketRegime.BULL: "🐂",
            MarketRegime.BEAR: "🐻",
            MarketRegime.SIDEWAYS: "➡️",
            MarketRegime.HIGH_VOLATILITY: "⚡",
            MarketRegime.UNKNOWN: "❓",
        }

        emoji = regime_emoji.get(state.regime, "❓")

        return (
            f"{emoji} {state.regime.value.upper()} "
            f"(confidence: {state.confidence:.0%}, "
            f"trend: {state.trend_strength:+.2f}, "
            f"vol: {state.volatility_percentile:.0f}th pctl, "
            f"duration: {state.days_in_regime}d)"
        )


class MultiAssetRegimeDetector:
    """Detect regimes across multiple assets."""

    def __init__(self, **kwargs):
        self.detectors: Dict[str, RegimeDetector] = {}
        self.kwargs = kwargs

    def get_detector(self, symbol: str) -> RegimeDetector:
        """Get or create detector for symbol."""
        if symbol not in self.detectors:
            self.detectors[symbol] = RegimeDetector(**self.kwargs)
        return self.detectors[symbol]

    def detect_all(self, price_data: Dict[str, pd.Series]) -> Dict[str, RegimeState]:
        """Detect regimes for all assets."""
        results = {}
        for symbol, prices in price_data.items():
            detector = self.get_detector(symbol)
            results[symbol] = detector.detect_regime(prices)
        return results

    def get_market_regime(self, price_data: Dict[str, pd.Series], weights: Optional[Dict[str, float]] = None) -> RegimeState:
        """Get overall market regime from multiple assets."""
        if not price_data:
            return RegimeState(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                trend_strength=0.0,
                volatility_percentile=50.0,
                days_in_regime=0,
                start_date=pd.Timestamp.now(),
            )

        states = self.detect_all(price_data)

        if weights is None:
            weights = {s: 1.0 / len(states) for s in states}

        # Weighted average of metrics
        total_weight = sum(weights.get(s, 0) for s in states)
        if total_weight == 0:
            total_weight = 1

        avg_trend = sum(states[s].trend_strength * weights.get(s, 0) for s in states) / total_weight

        avg_vol = sum(states[s].volatility_percentile * weights.get(s, 0) for s in states) / total_weight

        # Determine overall regime
        detector = RegimeDetector(**self.kwargs)
        regime, confidence = detector._classify_regime(avg_trend, avg_vol)

        return RegimeState(
            regime=regime,
            confidence=confidence,
            trend_strength=avg_trend,
            volatility_percentile=avg_vol,
            days_in_regime=1,
            start_date=pd.Timestamp.now(),
        )

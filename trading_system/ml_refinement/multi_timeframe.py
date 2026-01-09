"""
Multi-Timeframe Features Module

Generate features from multiple timeframes for ML models:
- Daily + Weekly signals
- Monthly trend context
- Cross-timeframe momentum
- Timeframe alignment scoring
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TimeframeFeatures:
    """Features for a single timeframe."""

    timeframe: str
    trend: float  # -1 to 1
    momentum: float
    volatility: float
    rsi: float
    ma_position: float  # Price relative to MA
    features: Dict[str, float]


class MultiTimeframeAnalyzer:
    """Analyze price data across multiple timeframes."""

    def __init__(
        self,
        timeframes: List[str] = None,
        ma_periods: Dict[str, int] = None,
    ):
        """
        Initialize multi-timeframe analyzer.

        Args:
            timeframes: List of timeframes to analyze
            ma_periods: MA periods for each timeframe
        """
        self.timeframes = timeframes or ["daily", "weekly", "monthly"]
        self.ma_periods = ma_periods or {
            "daily": 20,
            "weekly": 10,
            "monthly": 6,
        }

    def resample_to_timeframe(
        self,
        daily_data: pd.DataFrame,
        timeframe: str,
    ) -> pd.DataFrame:
        """Resample daily data to target timeframe."""
        if timeframe == "daily":
            return daily_data.copy()

        # Determine resampling rule
        if timeframe == "weekly":
            rule = "W"
        elif timeframe == "monthly":
            rule = "ME"
        else:
            logger.warning(f"Unknown timeframe: {timeframe}, using daily")
            return daily_data.copy()

        # Resample OHLCV
        resampled = (
            daily_data.resample(rule)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

        if "dollar_volume" in daily_data.columns:
            resampled["dollar_volume"] = daily_data["dollar_volume"].resample(rule).sum()

        return resampled

    def calculate_timeframe_features(
        self,
        data: pd.DataFrame,
        timeframe: str,
    ) -> TimeframeFeatures:
        """Calculate features for a single timeframe."""
        if len(data) < 5:
            return self._empty_features(timeframe)

        close = data["close"]
        ma_period = self.ma_periods.get(timeframe, 20)

        # Trend (linear regression slope)
        x = np.arange(min(len(close), ma_period))
        y = close.iloc[-len(x) :].values
        if len(x) > 1 and y.std() > 0:
            slope = np.polyfit(x, y, 1)[0]
            trend = np.clip(slope / y.mean() * len(x), -1, 1)
        else:
            trend = 0.0

        # Momentum (rate of change)
        if len(close) >= ma_period:
            roc = close.iloc[-1] / close.iloc[-ma_period] - 1
            momentum = np.clip(roc * 10, -1, 1)
        else:
            momentum = 0.0

        # Volatility
        returns = close.pct_change().dropna()
        if len(returns) > 1:
            volatility = returns.std() * np.sqrt(252 if timeframe == "daily" else 52 if timeframe == "weekly" else 12)
        else:
            volatility = 0.0

        # RSI
        rsi = self._calculate_rsi(close, min(14, len(close) - 1))

        # MA position
        if len(close) >= ma_period:
            ma = close.rolling(ma_period).mean().iloc[-1]
            ma_position = (close.iloc[-1] / ma - 1) * 10
            ma_position = np.clip(ma_position, -1, 1)
        else:
            ma_position = 0.0

        features = {
            f"{timeframe}_trend": trend,
            f"{timeframe}_momentum": momentum,
            f"{timeframe}_volatility": volatility,
            f"{timeframe}_rsi": rsi / 100,  # Normalize to 0-1
            f"{timeframe}_ma_position": ma_position,
            f"{timeframe}_return_5": self._period_return(close, 5),
            f"{timeframe}_return_10": self._period_return(close, 10),
            f"{timeframe}_high_low_range": self._high_low_range(data),
        }

        return TimeframeFeatures(
            timeframe=timeframe,
            trend=trend,
            momentum=momentum,
            volatility=volatility,
            rsi=rsi,
            ma_position=ma_position,
            features=features,
        )

    def _empty_features(self, timeframe: str) -> TimeframeFeatures:
        """Return empty features for a timeframe."""
        return TimeframeFeatures(
            timeframe=timeframe,
            trend=0.0,
            momentum=0.0,
            volatility=0.0,
            rsi=50.0,
            ma_position=0.0,
            features={
                f"{timeframe}_trend": 0.0,
                f"{timeframe}_momentum": 0.0,
                f"{timeframe}_volatility": 0.0,
                f"{timeframe}_rsi": 0.5,
                f"{timeframe}_ma_position": 0.0,
                f"{timeframe}_return_5": 0.0,
                f"{timeframe}_return_10": 0.0,
                f"{timeframe}_high_low_range": 0.0,
            },
        )

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50.0

        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

        if loss.iloc[-1] == 0:
            return 100.0

        rs = gain.iloc[-1] / loss.iloc[-1]
        rsi = 100 - (100 / (1 + rs))

        return float(rsi)

    def _period_return(self, prices: pd.Series, periods: int) -> float:
        """Calculate return over N periods."""
        if len(prices) < periods + 1:
            return 0.0
        return float(prices.iloc[-1] / prices.iloc[-periods - 1] - 1)

    def _high_low_range(self, data: pd.DataFrame) -> float:
        """Calculate high-low range as % of close."""
        if "high" not in data.columns or "low" not in data.columns:
            return 0.0

        recent = data.iloc[-5:]
        if len(recent) == 0:
            return 0.0

        high = recent["high"].max()
        low = recent["low"].min()
        close = recent["close"].iloc[-1]

        if close == 0:
            return 0.0

        return (high - low) / close

    def generate_all_features(
        self,
        daily_data: pd.DataFrame,
    ) -> Dict[str, float]:
        """Generate features for all timeframes."""
        all_features = {}
        tf_results = {}

        for timeframe in self.timeframes:
            resampled = self.resample_to_timeframe(daily_data, timeframe)
            tf_features = self.calculate_timeframe_features(resampled, timeframe)
            tf_results[timeframe] = tf_features
            all_features.update(tf_features.features)

        # Add cross-timeframe features
        cross_features = self._calculate_cross_timeframe_features(tf_results)
        all_features.update(cross_features)

        return all_features

    def _calculate_cross_timeframe_features(
        self,
        tf_results: Dict[str, TimeframeFeatures],
    ) -> Dict[str, float]:
        """Calculate features that compare timeframes."""
        features = {}

        # Trend alignment
        trends = [tf.trend for tf in tf_results.values()]
        if trends:
            # All positive or all negative = aligned
            all_positive = all(t > 0 for t in trends)
            all_negative = all(t < 0 for t in trends)
            features["trend_alignment"] = 1.0 if (all_positive or all_negative) else 0.0
            features["avg_trend"] = np.mean(trends)

        # Momentum alignment
        momentums = [tf.momentum for tf in tf_results.values()]
        if momentums:
            features["momentum_alignment"] = 1.0 if all(m > 0 for m in momentums) or all(m < 0 for m in momentums) else 0.0
            features["avg_momentum"] = np.mean(momentums)

        # Volatility comparison
        vols = {tf: result.volatility for tf, result in tf_results.items()}
        if "daily" in vols and "weekly" in vols and vols["weekly"] > 0:
            features["vol_ratio_daily_weekly"] = vols["daily"] / vols["weekly"]

        # RSI divergence
        rsis = [tf.rsi for tf in tf_results.values()]
        if len(rsis) >= 2:
            features["rsi_divergence"] = max(rsis) - min(rsis)

        # Higher timeframe context
        if "weekly" in tf_results:
            features["weekly_bullish"] = 1.0 if tf_results["weekly"].trend > 0 else 0.0
        if "monthly" in tf_results:
            features["monthly_bullish"] = 1.0 if tf_results["monthly"].trend > 0 else 0.0

        return features

    def get_timeframe_summary(
        self,
        daily_data: pd.DataFrame,
    ) -> str:
        """Get human-readable summary of all timeframes."""
        lines = []
        lines.append("=" * 50)
        lines.append("MULTI-TIMEFRAME ANALYSIS")
        lines.append("=" * 50)

        for timeframe in self.timeframes:
            resampled = self.resample_to_timeframe(daily_data, timeframe)
            tf = self.calculate_timeframe_features(resampled, timeframe)

            trend_emoji = "🔼" if tf.trend > 0.1 else "🔽" if tf.trend < -0.1 else "➡️"

            lines.append(f"\n{timeframe.upper()}")
            lines.append("-" * 30)
            lines.append(f"  Trend: {trend_emoji} {tf.trend:+.2f}")
            lines.append(f"  Momentum: {tf.momentum:+.2f}")
            lines.append(f"  RSI: {tf.rsi:.1f}")
            lines.append(f"  Volatility: {tf.volatility:.1%}")
            lines.append(f"  MA Position: {tf.ma_position:+.2f}")

        # Cross-timeframe
        tf_results = {}
        for timeframe in self.timeframes:
            resampled = self.resample_to_timeframe(daily_data, timeframe)
            tf_results[timeframe] = self.calculate_timeframe_features(resampled, timeframe)

        cross = self._calculate_cross_timeframe_features(tf_results)

        lines.append(f"\nCROSS-TIMEFRAME")
        lines.append("-" * 30)
        lines.append(f"  Trend Aligned: {'✅' if cross.get('trend_alignment', 0) > 0.5 else '❌'}")
        lines.append(f"  Momentum Aligned: {'✅' if cross.get('momentum_alignment', 0) > 0.5 else '❌'}")

        lines.append("")
        lines.append("=" * 50)

        return "\n".join(lines)


class MultiTimeframeFeatureEngineer:
    """Feature engineer that incorporates multi-timeframe data."""

    def __init__(self, analyzer: Optional[MultiTimeframeAnalyzer] = None):
        self.analyzer = analyzer or MultiTimeframeAnalyzer()

    def engineer_features(
        self,
        symbol: str,
        daily_data: pd.DataFrame,
        existing_features: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Engineer features including multi-timeframe data.

        Args:
            symbol: Symbol being analyzed
            daily_data: Daily OHLCV data
            existing_features: Existing features to extend

        Returns:
            Combined feature dictionary
        """
        features = existing_features.copy() if existing_features else {}

        # Add multi-timeframe features
        mtf_features = self.analyzer.generate_all_features(daily_data)
        features.update(mtf_features)

        return features

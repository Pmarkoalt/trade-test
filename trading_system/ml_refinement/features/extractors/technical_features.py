"""Technical indicator feature extractors."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from trading_system.ml_refinement.features.extractors.base_extractor import (
    OHLCVExtractor,
)


class TrendFeatures(OHLCVExtractor):
    """Extract trend-related features."""

    def __init__(self, lookbacks: Optional[List[int]] = None):
        """
        Initialize with lookback periods.

        Args:
            lookbacks: List of lookback periods for moving averages.
        """
        self.lookbacks = lookbacks or [5, 10, 20, 50, 200]

    @property
    def name(self) -> str:
        return "trend_features"

    @property
    def category(self) -> str:
        return "technical"

    @property
    def feature_names(self) -> List[str]:
        names = []

        for lb in self.lookbacks:
            names.extend([
                f"price_vs_ma{lb}",        # Price relative to MA
                f"ma{lb}_slope",           # MA slope (normalized)
            ])

        names.extend([
            "price_vs_ma_fast_slow",       # Fast MA vs Slow MA
            "trend_strength",               # ADX-like measure
            "higher_highs",                 # Count of higher highs
            "lower_lows",                   # Count of lower lows
        ])

        return names

    def extract(self, ohlcv: pd.DataFrame, current_idx: int = -1) -> Dict[str, float]:
        """Extract trend features."""
        features = {}
        close = ohlcv["close"]
        high = ohlcv["high"]
        low = ohlcv["low"]

        current_price = self._safe_get(close, current_idx)

        # Price vs MAs and MA slopes
        for lb in self.lookbacks:
            if len(close) >= lb:
                ma = close.rolling(lb).mean()
                ma_val = self._safe_get(ma, current_idx)

                # Price relative to MA (normalized)
                if ma_val > 0:
                    features[f"price_vs_ma{lb}"] = (current_price - ma_val) / ma_val
                else:
                    features[f"price_vs_ma{lb}"] = 0.0

                # MA slope (5-day change, normalized by ATR)
                if len(ma) >= 5:
                    ma_prev = self._safe_get(ma, current_idx - 5)
                    atr = self._calculate_atr(ohlcv, 14)
                    if atr > 0 and ma_prev > 0:
                        features[f"ma{lb}_slope"] = (ma_val - ma_prev) / (atr * 5)
                    else:
                        features[f"ma{lb}_slope"] = 0.0
                else:
                    features[f"ma{lb}_slope"] = 0.0
            else:
                features[f"price_vs_ma{lb}"] = 0.0
                features[f"ma{lb}_slope"] = 0.0

        # Fast vs Slow MA
        if len(close) >= 50:
            ma_fast = close.rolling(10).mean().iloc[current_idx]
            ma_slow = close.rolling(50).mean().iloc[current_idx]
            if ma_slow > 0:
                features["price_vs_ma_fast_slow"] = (ma_fast - ma_slow) / ma_slow
            else:
                features["price_vs_ma_fast_slow"] = 0.0
        else:
            features["price_vs_ma_fast_slow"] = 0.0

        # Trend strength (simplified ADX-like)
        features["trend_strength"] = self._calculate_trend_strength(ohlcv, 14)

        # Higher highs / lower lows count (last 10 bars)
        features["higher_highs"] = self._count_higher_highs(high, 10)
        features["lower_lows"] = self._count_lower_lows(low, 10)

        return features

    def _calculate_atr(self, ohlcv: pd.DataFrame, period: int) -> float:
        """Calculate ATR."""
        high = ohlcv["high"]
        low = ohlcv["low"]
        close = ohlcv["close"]

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        atr = tr.rolling(period).mean()
        return self._safe_get(atr, -1)

    def _calculate_trend_strength(self, ohlcv: pd.DataFrame, period: int) -> float:
        """Calculate trend strength (0-1)."""
        close = ohlcv["close"]
        if len(close) < period:
            return 0.0

        # Use R-squared of linear regression as trend strength
        y = close.iloc[-period:].values
        x = np.arange(period)

        if np.std(y) == 0:
            return 0.0

        correlation = np.corrcoef(x, y)[0, 1]
        return correlation ** 2 if not np.isnan(correlation) else 0.0

    def _count_higher_highs(self, high: pd.Series, lookback: int) -> float:
        """Count consecutive higher highs."""
        if len(high) < lookback:
            return 0.0

        recent = high.iloc[-lookback:]
        count = 0
        for i in range(1, len(recent)):
            if recent.iloc[i] > recent.iloc[i - 1]:
                count += 1

        return count / (lookback - 1)  # Normalize to 0-1

    def _count_lower_lows(self, low: pd.Series, lookback: int) -> float:
        """Count consecutive lower lows."""
        if len(low) < lookback:
            return 0.0

        recent = low.iloc[-lookback:]
        count = 0
        for i in range(1, len(recent)):
            if recent.iloc[i] < recent.iloc[i - 1]:
                count += 1

        return count / (lookback - 1)


class MomentumFeatures(OHLCVExtractor):
    """Extract momentum-related features."""

    def __init__(self, lookbacks: Optional[List[int]] = None):
        self.lookbacks = lookbacks or [5, 10, 20, 50]

    @property
    def name(self) -> str:
        return "momentum_features"

    @property
    def category(self) -> str:
        return "technical"

    @property
    def feature_names(self) -> List[str]:
        names = []

        for lb in self.lookbacks:
            names.append(f"roc_{lb}")       # Rate of change

        names.extend([
            "rsi_14",                       # RSI
            "rsi_deviation",                # RSI distance from 50
            "momentum_divergence",          # Price vs momentum divergence
            "acceleration",                 # Momentum acceleration
        ])

        return names

    def extract(self, ohlcv: pd.DataFrame, current_idx: int = -1) -> Dict[str, float]:
        """Extract momentum features."""
        features = {}
        close = ohlcv["close"]

        # Rate of change for various periods
        for lb in self.lookbacks:
            if len(close) > lb:
                current = self._safe_get(close, current_idx)
                past = self._safe_get(close, current_idx - lb)
                if past > 0:
                    features[f"roc_{lb}"] = (current - past) / past
                else:
                    features[f"roc_{lb}"] = 0.0
            else:
                features[f"roc_{lb}"] = 0.0

        # RSI
        rsi = self._calculate_rsi(close, 14)
        features["rsi_14"] = rsi / 100  # Normalize to 0-1
        features["rsi_deviation"] = (rsi - 50) / 50  # -1 to 1

        # Momentum divergence
        features["momentum_divergence"] = self._calculate_divergence(ohlcv, 14)

        # Acceleration
        features["acceleration"] = self._calculate_acceleration(close, 10)

        return features

    def _calculate_rsi(self, close: pd.Series, period: int) -> float:
        """Calculate RSI."""
        if len(close) < period + 1:
            return 50.0

        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))

        return self._safe_get(rsi, -1, 50.0)

    def _calculate_divergence(self, ohlcv: pd.DataFrame, period: int) -> float:
        """Calculate price-momentum divergence."""
        close = ohlcv["close"]
        if len(close) < period * 2:
            return 0.0

        # Price trend
        price_change = (close.iloc[-1] - close.iloc[-period]) / close.iloc[-period]

        # Momentum trend (ROC of ROC)
        roc = close.pct_change(period)
        if len(roc) >= period:
            mom_change = roc.iloc[-1] - roc.iloc[-period]
        else:
            mom_change = 0.0

        # Divergence is when they disagree
        if price_change > 0 and mom_change < 0:
            return -abs(mom_change)  # Bearish divergence
        elif price_change < 0 and mom_change > 0:
            return abs(mom_change)   # Bullish divergence

        return 0.0

    def _calculate_acceleration(self, close: pd.Series, period: int) -> float:
        """Calculate momentum acceleration."""
        if len(close) < period * 2:
            return 0.0

        roc = close.pct_change(period)
        roc_of_roc = roc.diff(period)

        return self._safe_get(roc_of_roc, -1, 0.0)


class VolatilityFeatures(OHLCVExtractor):
    """Extract volatility-related features."""

    @property
    def name(self) -> str:
        return "volatility_features"

    @property
    def category(self) -> str:
        return "technical"

    @property
    def feature_names(self) -> List[str]:
        return [
            "atr_ratio",                   # ATR vs price
            "volatility_percentile",       # Current vol vs historical
            "volatility_trend",            # Vol expanding or contracting
            "range_ratio",                 # Today's range vs avg
            "gap_size",                    # Gap from previous close
            "intraday_volatility",         # High-low / close
        ]

    def extract(self, ohlcv: pd.DataFrame, current_idx: int = -1) -> Dict[str, float]:
        """Extract volatility features."""
        features = {}
        high = ohlcv["high"]
        low = ohlcv["low"]
        close = ohlcv["close"]
        open_ = ohlcv["open"]

        current_close = self._safe_get(close, current_idx)

        # ATR ratio
        atr = self._calculate_atr(ohlcv, 14)
        if current_close > 0:
            features["atr_ratio"] = atr / current_close
        else:
            features["atr_ratio"] = 0.0

        # Volatility percentile
        features["volatility_percentile"] = self._vol_percentile(close, 20, 252)

        # Volatility trend
        features["volatility_trend"] = self._vol_trend(close, 20)

        # Range ratio
        avg_range = (high - low).rolling(20).mean()
        current_range = self._safe_get(high, current_idx) - self._safe_get(low, current_idx)
        avg_range_val = self._safe_get(avg_range, current_idx)
        if avg_range_val > 0:
            features["range_ratio"] = current_range / avg_range_val
        else:
            features["range_ratio"] = 1.0

        # Gap size
        if len(close) > 1:
            prev_close = self._safe_get(close, current_idx - 1)
            current_open = self._safe_get(open_, current_idx)
            if prev_close > 0:
                features["gap_size"] = (current_open - prev_close) / prev_close
            else:
                features["gap_size"] = 0.0
        else:
            features["gap_size"] = 0.0

        # Intraday volatility
        current_high = self._safe_get(high, current_idx)
        current_low = self._safe_get(low, current_idx)
        if current_close > 0:
            features["intraday_volatility"] = (current_high - current_low) / current_close
        else:
            features["intraday_volatility"] = 0.0

        return features

    def _calculate_atr(self, ohlcv: pd.DataFrame, period: int) -> float:
        """Calculate ATR."""
        high = ohlcv["high"]
        low = ohlcv["low"]
        close = ohlcv["close"]

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        atr = tr.rolling(period).mean()
        return self._safe_get(atr, -1)

    def _vol_percentile(
        self,
        close: pd.Series,
        short_period: int,
        long_period: int,
    ) -> float:
        """Calculate volatility percentile."""
        if len(close) < long_period:
            return 0.5

        returns = close.pct_change()
        current_vol = returns.iloc[-short_period:].std()
        historical_vol = returns.rolling(short_period).std()

        if len(historical_vol) < long_period:
            return 0.5

        percentile = (historical_vol.iloc[-long_period:] < current_vol).mean()
        return percentile

    def _vol_trend(self, close: pd.Series, period: int) -> float:
        """Calculate volatility trend (expanding = positive)."""
        if len(close) < period * 2:
            return 0.0

        returns = close.pct_change()
        vol = returns.rolling(period).std()

        current = self._safe_get(vol, -1)
        past = self._safe_get(vol, -period)

        if past > 0:
            return (current - past) / past
        return 0.0


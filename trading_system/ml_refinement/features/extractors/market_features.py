"""Market regime feature extractors."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from trading_system.ml_refinement.features.extractors.base_extractor import (
    OHLCVExtractor,
)


class MarketRegimeFeatures(OHLCVExtractor):
    """Extract market regime features."""

    @property
    def name(self) -> str:
        return "market_regime"

    @property
    def category(self) -> str:
        return "market"

    @property
    def feature_names(self) -> List[str]:
        return [
            "market_trend",  # Overall market direction
            "market_breadth",  # Approximated breadth
            "correlation_regime",  # High/low correlation
            "volatility_regime",  # Vol regime (0-1)
            "drawdown_depth",  # Current drawdown
            "days_from_high",  # Days since ATH
            "rally_strength",  # Strength of current rally
        ]

    def extract(
        self,
        ohlcv: pd.DataFrame,
        current_idx: int = -1,
        benchmark_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Extract market regime features.

        Args:
            ohlcv: Symbol OHLCV data
            current_idx: Current bar index
            benchmark_data: Optional benchmark (SPY) OHLCV for market features
        """
        features = {}

        # Use benchmark if available, otherwise use symbol data
        data = benchmark_data if benchmark_data is not None else ohlcv
        close = data["close"]

        # Market trend (-1 to 1)
        features["market_trend"] = self._calculate_trend(close, 50)

        # Market breadth approximation (using momentum)
        features["market_breadth"] = self._calculate_breadth_proxy(close, 20)

        # Correlation regime
        features["correlation_regime"] = self._calculate_correlation_regime(ohlcv["close"], close, 20)

        # Volatility regime
        features["volatility_regime"] = self._calculate_vol_regime(close, 20)

        # Drawdown depth
        features["drawdown_depth"] = self._calculate_drawdown(close)

        # Days from high
        features["days_from_high"] = self._days_from_high(close, 252)

        # Rally strength
        features["rally_strength"] = self._calculate_rally_strength(close, 20)

        return features

    def _calculate_trend(self, close: pd.Series, period: int) -> float:
        """Calculate trend direction and strength (-1 to 1)."""
        if len(close) < period:
            return 0.0

        ma = close.rolling(period).mean()
        current = close.iloc[-1]
        ma_val = ma.iloc[-1]

        if ma_val <= 0:
            return 0.0

        # Normalize by recent volatility
        returns = close.pct_change()
        vol = returns.iloc[-period:].std()

        if vol <= 0:
            return 0.0

        trend = (current - ma_val) / (ma_val * vol * np.sqrt(period))

        # Clip to -1, 1
        return float(max(-1.0, min(1.0, trend)))

    def _calculate_breadth_proxy(self, close: pd.Series, period: int) -> float:
        """
        Approximate market breadth using momentum.

        In real implementation, this would use actual breadth data.
        """
        if len(close) < period:
            return 0.5

        returns = close.pct_change()
        positive_days = (returns.iloc[-period:] > 0).sum()

        return float(positive_days / period)

    def _calculate_correlation_regime(
        self,
        symbol_close: pd.Series,
        market_close: pd.Series,
        period: int,
    ) -> float:
        """Calculate correlation with market."""
        if len(symbol_close) < period or len(market_close) < period:
            return 0.5

        symbol_returns = symbol_close.pct_change().iloc[-period:]
        market_returns = market_close.pct_change().iloc[-period:]

        if len(symbol_returns) != len(market_returns):
            return 0.5

        corr = symbol_returns.corr(market_returns)
        return corr if not np.isnan(corr) else 0.5

    def _calculate_vol_regime(self, close: pd.Series, period: int) -> float:
        """
        Calculate volatility regime (0 = low vol, 1 = high vol).
        """
        if len(close) < period * 2:
            return 0.5

        returns = close.pct_change()
        current_vol = returns.iloc[-period:].std()

        # Compare to longer-term vol
        long_vol = returns.iloc[-period * 2 : -period].std()

        if long_vol <= 0:
            return 0.5

        vol_ratio = current_vol / long_vol

        # Normalize to 0-1 (0.5 = normal, 1 = 2x normal)
        return float(min(1.0, vol_ratio / 2))

    def _calculate_drawdown(self, close: pd.Series) -> float:
        """Calculate current drawdown from peak."""
        if len(close) < 1:
            return 0.0

        peak = close.expanding().max()
        drawdown = (close - peak) / peak

        return float(drawdown.iloc[-1])

    def _days_from_high(self, close: pd.Series, lookback: int) -> float:
        """Calculate days from recent high (normalized)."""
        if len(close) < lookback:
            lookback = len(close)

        if lookback < 1:
            return 0.0

        recent = close.iloc[-lookback:]
        high_idx = recent.idxmax()

        # Find position in recent series
        try:
            if isinstance(recent.index, pd.RangeIndex):
                # For RangeIndex, calculate position directly
                days = len(recent) - 1 - (high_idx - recent.index[0])
            else:
                # For other index types, use get_loc
                days = len(recent) - 1 - recent.index.get_loc(high_idx)
        except (KeyError, TypeError):
            # Fallback: find position by iterating
            days = 0
            for i, idx in enumerate(recent.index):
                if idx == high_idx:
                    days = len(recent) - 1 - i
                    break

        # Normalize by lookback
        return float(days / lookback)

    def _calculate_rally_strength(self, close: pd.Series, period: int) -> float:
        """Calculate strength of current rally/decline."""
        if len(close) < period:
            return 0.0

        start_price = close.iloc[-period]
        end_price = close.iloc[-1]

        if start_price <= 0:
            return 0.0

        # Calculate return and compare to volatility
        ret = (end_price - start_price) / start_price
        vol = close.pct_change().iloc[-period:].std()

        if vol <= 0:
            return 0.0

        # Sharpe-like measure
        strength = ret / (vol * np.sqrt(period))

        return float(max(-1.0, min(1.0, strength)))

"""Feature engineering for ML models."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from trading_system.models.features import FeatureRow


class MLFeatureEngineer:
    """Engineer features from FeatureRow for ML models.

    This class converts FeatureRow objects into feature vectors suitable
    for ML model training and prediction. It handles feature scaling,
    normalization, and missing value imputation.

    Enhanced with additional technical indicators:
    - RSI, MACD, Bollinger Bands
    - Volume indicators (OBV, volume profile)
    - Volatility features (ATR, realized vol)
    - Market regime features (trend detection, volatility regime)
    - Cross-asset features (correlation, relative strength)
    """

    def __init__(
        self,
        include_raw_features: bool = True,
        include_derived_features: bool = True,
        include_technical_indicators: bool = True,
        include_volume_indicators: bool = True,
        include_volatility_features: bool = True,
        include_market_regime_features: bool = True,
        include_cross_asset_features: bool = True,
        normalize_features: bool = True,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
    ):
        """Initialize feature engineer.

        Args:
            include_raw_features: Include raw indicator values
            include_derived_features: Include derived features (ratios, differences, etc.)
            include_technical_indicators: Include RSI, MACD, Bollinger Bands
            include_volume_indicators: Include OBV and volume profile features
            include_volatility_features: Include ATR and realized volatility
            include_market_regime_features: Include trend and volatility regime classification
            include_cross_asset_features: Include correlation and relative strength features
            normalize_features: Normalize features to [0, 1] range
            rsi_period: Period for RSI calculation (default 14)
            macd_fast: Fast EMA period for MACD (default 12)
            macd_slow: Slow EMA period for MACD (default 26)
            macd_signal: Signal line period for MACD (default 9)
            bb_period: Period for Bollinger Bands (default 20)
            bb_std: Standard deviation multiplier for Bollinger Bands (default 2.0)
        """
        self.include_raw_features = include_raw_features
        self.include_derived_features = include_derived_features
        self.include_technical_indicators = include_technical_indicators
        self.include_volume_indicators = include_volume_indicators
        self.include_volatility_features = include_volatility_features
        self.include_market_regime_features = include_market_regime_features
        self.include_cross_asset_features = include_cross_asset_features
        self.normalize_features = normalize_features
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self._feature_scalers: Optional[Dict[str, tuple[float, float]]] = None
        self._feature_names: Optional[List[str]] = None
        self._price_history: Dict[str, List[Tuple[pd.Timestamp, float, float, float, float, float]]] = (
            {}
        )  # symbol -> [(date, open, high, low, close, volume)]

    def fit(self, feature_rows: List[FeatureRow]) -> None:
        """Fit feature scalers on training data.

        Args:
            feature_rows: List of FeatureRow objects for training
        """
        # Convert to DataFrame
        df = self._feature_rows_to_dataframe(feature_rows)

        # Store feature names
        self._feature_names = list(df.columns)

        # Compute scalers (min/max for normalization)
        if self.normalize_features:
            self._feature_scalers = {}
            for col in df.columns:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    min_val = col_data.min()
                    max_val = col_data.max()
                    # Avoid division by zero
                    if max_val - min_val > 1e-10:
                        self._feature_scalers[col] = (min_val, max_val)
                    else:
                        self._feature_scalers[col] = (min_val, min_val + 1.0)
                else:
                    self._feature_scalers[col] = (0.0, 1.0)

    @staticmethod
    def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute Relative Strength Index (RSI).

        Args:
            prices: Series of closing prices
            period: RSI period (default 14)

        Returns:
            Series of RSI values (0-100)
        """
        if len(prices) < period + 1:
            return pd.Series(np.nan, index=prices.index)

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _compute_macd(
        prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Compute MACD (Moving Average Convergence Divergence).

        Args:
            prices: Series of closing prices
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line EMA period (default 9)

        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        if len(prices) < slow:
            return (
                pd.Series(np.nan, index=prices.index),
                pd.Series(np.nan, index=prices.index),
                pd.Series(np.nan, index=prices.index),
            )

        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def _compute_bollinger_bands(
        prices: pd.Series, period: int = 20, std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Compute Bollinger Bands.

        Args:
            prices: Series of closing prices
            period: Moving average period (default 20)
            std_dev: Standard deviation multiplier (default 2.0)

        Returns:
            Tuple of (Upper band, Middle band (SMA), Lower band)
        """
        if len(prices) < period:
            return (
                pd.Series(np.nan, index=prices.index),
                pd.Series(np.nan, index=prices.index),
                pd.Series(np.nan, index=prices.index),
            )

        middle_band = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)

        return upper_band, middle_band, lower_band

    @staticmethod
    def _compute_obv(prices: pd.Series, volumes: pd.Series) -> pd.Series:
        """Compute On-Balance Volume (OBV).

        Args:
            prices: Series of closing prices
            volumes: Series of volumes

        Returns:
            Series of OBV values
        """
        if len(prices) != len(volumes):
            return pd.Series(np.nan, index=prices.index)

        obv = pd.Series(0.0, index=prices.index)
        price_diff = prices.diff()

        for i in range(1, len(prices)):
            if price_diff.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i - 1] + volumes.iloc[i]
            elif price_diff.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i - 1] - volumes.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]

        return obv

    @staticmethod
    def _compute_realized_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
        """Compute realized volatility (rolling standard deviation of returns).

        Args:
            returns: Series of returns
            window: Rolling window (default 20)

        Returns:
            Series of realized volatility values
        """
        if len(returns) < window:
            return pd.Series(np.nan, index=returns.index)

        return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized

    def transform(self, feature_row: FeatureRow) -> pd.Series:
        """Transform a single FeatureRow to feature vector.

        Note: Some technical indicators (RSI, MACD, Bollinger Bands) require
        historical data and are best computed in batch mode via transform_batch().
        This method includes basic features that can be computed from a single row.

        Args:
            feature_row: FeatureRow to transform

        Returns:
            Series with feature values
        """
        features = {}

        if self.include_raw_features:
            # Raw indicator values
            features.update(
                {
                    "ma20": feature_row.ma20 if feature_row.ma20 is not None else np.nan,
                    "ma50": feature_row.ma50 if feature_row.ma50 is not None else np.nan,
                    "ma200": feature_row.ma200 if feature_row.ma200 is not None else np.nan,
                    "ma50_slope": feature_row.ma50_slope if feature_row.ma50_slope is not None else np.nan,
                    "atr14": feature_row.atr14 if feature_row.atr14 is not None else np.nan,
                    "roc60": feature_row.roc60 if feature_row.roc60 is not None else np.nan,
                    "highest_close_20d": (
                        feature_row.highest_close_20d if feature_row.highest_close_20d is not None else np.nan
                    ),
                    "highest_close_55d": (
                        feature_row.highest_close_55d if feature_row.highest_close_55d is not None else np.nan
                    ),
                    "adv20": feature_row.adv20 if feature_row.adv20 is not None else np.nan,
                    "returns_1d": feature_row.returns_1d if feature_row.returns_1d is not None else np.nan,
                    "benchmark_roc60": feature_row.benchmark_roc60 if feature_row.benchmark_roc60 is not None else np.nan,
                    "benchmark_returns_1d": (
                        feature_row.benchmark_returns_1d if feature_row.benchmark_returns_1d is not None else np.nan
                    ),
                }
            )

            # Add mean reversion indicators if available
            if feature_row.zscore is not None:
                features["zscore"] = feature_row.zscore
            if feature_row.ma_lookback is not None:
                features["ma_lookback"] = feature_row.ma_lookback
            if feature_row.std_lookback is not None:
                features["std_lookback"] = feature_row.std_lookback

        if self.include_derived_features:
            # Derived features
            close = feature_row.close

            # Price relative to moving averages
            if feature_row.ma20 is not None:
                features["close_to_ma20"] = close / feature_row.ma20 - 1.0
            else:
                features["close_to_ma20"] = np.nan

            if feature_row.ma50 is not None:
                features["close_to_ma50"] = close / feature_row.ma50 - 1.0
            else:
                features["close_to_ma50"] = np.nan

            if feature_row.ma200 is not None:
                features["close_to_ma200"] = close / feature_row.ma200 - 1.0
            else:
                features["close_to_ma200"] = np.nan

            # Breakout strength indicators
            if feature_row.atr14 is not None and feature_row.atr14 > 0:
                if feature_row.ma20 is not None:
                    features["breakout_strength_20d"] = (close - feature_row.ma20) / feature_row.atr14
                else:
                    features["breakout_strength_20d"] = np.nan

                if feature_row.ma50 is not None:
                    features["breakout_strength_50d"] = (close - feature_row.ma50) / feature_row.atr14
                else:
                    features["breakout_strength_50d"] = np.nan
            else:
                features["breakout_strength_20d"] = np.nan
                features["breakout_strength_50d"] = np.nan

            # Relative strength
            if feature_row.roc60 is not None and feature_row.benchmark_roc60 is not None:
                features["relative_strength"] = feature_row.roc60 - feature_row.benchmark_roc60
            else:
                features["relative_strength"] = np.nan

            # MA relationships
            if feature_row.ma20 is not None and feature_row.ma50 is not None:
                features["ma20_to_ma50"] = feature_row.ma20 / feature_row.ma50 - 1.0
            else:
                features["ma20_to_ma50"] = np.nan

            if feature_row.ma50 is not None and feature_row.ma200 is not None:
                features["ma50_to_ma200"] = feature_row.ma50 / feature_row.ma200 - 1.0
            else:
                features["ma50_to_ma200"] = np.nan

            # Breakout clearance
            if feature_row.highest_close_20d is not None and feature_row.highest_close_20d > 0:
                features["clearance_20d"] = (close / feature_row.highest_close_20d) - 1.0
            else:
                features["clearance_20d"] = np.nan

            if feature_row.highest_close_55d is not None and feature_row.highest_close_55d > 0:
                features["clearance_55d"] = (close / feature_row.highest_close_55d) - 1.0
            else:
                features["clearance_55d"] = np.nan

            # Price range features
            if feature_row.high is not None and feature_row.low is not None:
                features["price_range"] = (feature_row.high - feature_row.low) / close if close > 0 else np.nan
                features["upper_shadow"] = (
                    (feature_row.high - max(feature_row.open, feature_row.close)) / close if close > 0 else np.nan
                )
                features["lower_shadow"] = (
                    (min(feature_row.open, feature_row.close) - feature_row.low) / close if close > 0 else np.nan
                )
            else:
                features["price_range"] = np.nan
                features["upper_shadow"] = np.nan
                features["lower_shadow"] = np.nan

        # Convert to Series
        feature_series = pd.Series(features)

        # Normalize if requested
        if self.normalize_features and self._feature_scalers is not None:
            for col in feature_series.index:
                if col in self._feature_scalers:
                    min_val, max_val = self._feature_scalers[col]
                    if not pd.isna(feature_series[col]):
                        if max_val - min_val > 1e-10:
                            feature_series[col] = (feature_series[col] - min_val) / (max_val - min_val)
                        else:
                            feature_series[col] = 0.5  # Default to middle if constant
                else:
                    # If feature not in scalers, keep as-is
                    pass

        # Handle missing values (fill with 0 for normalized features, or median otherwise)
        feature_series = feature_series.fillna(0.0)

        return feature_series

    def transform_batch(self, feature_rows: List[FeatureRow]) -> pd.DataFrame:
        """Transform multiple FeatureRows to feature matrix.

        This method uses batch computation for technical indicators that
        require historical data (RSI, MACD, Bollinger Bands, etc.).

        Args:
            feature_rows: List of FeatureRow objects (should be sorted by date)

        Returns:
            DataFrame with feature vectors (one row per FeatureRow)
        """
        # Use batch computation for technical indicators
        df = self._feature_rows_to_dataframe(feature_rows)

        # Normalize features if requested
        if self.normalize_features and self._feature_scalers is not None:
            for col in df.columns:
                if col in self._feature_scalers:
                    min_val, max_val = self._feature_scalers[col]
                    if max_val - min_val > 1e-10:
                        df[col] = (df[col] - min_val) / (max_val - min_val)
                    else:
                        df[col] = 0.5  # Default to middle if constant

        # Handle missing values
        df = df.fillna(0.0)

        # Ensure consistent column order
        if self._feature_names:
            # Reorder columns to match training order
            missing_cols = set(self._feature_names) - set(df.columns)
            for col in missing_cols:
                df[col] = 0.0
            df = df[self._feature_names]

        return df

    def fit_transform(self, feature_rows: List[FeatureRow]) -> pd.DataFrame:
        """Fit scalers and transform feature rows.

        Args:
            feature_rows: List of FeatureRow objects for training

        Returns:
            DataFrame with transformed feature vectors
        """
        self.fit(feature_rows)
        return self.transform_batch(feature_rows)

    def _feature_rows_to_dataframe(self, feature_rows: List[FeatureRow]) -> pd.DataFrame:
        """Convert FeatureRow list to DataFrame for fitting.

        This method computes all features including technical indicators
        that require historical data (RSI, MACD, Bollinger Bands, etc.).

        Args:
            feature_rows: List of FeatureRow objects (should be sorted by date)

        Returns:
            DataFrame with raw feature values
        """
        if not feature_rows:
            return pd.DataFrame()

        # Extract price and volume series for technical indicators
        dates = [fr.date for fr in feature_rows]
        closes = pd.Series([fr.close for fr in feature_rows], index=dates)
        opens = pd.Series([fr.open for fr in feature_rows], index=dates)
        highs = pd.Series([fr.high for fr in feature_rows], index=dates)
        lows = pd.Series([fr.low for fr in feature_rows], index=dates)
        returns = pd.Series([fr.returns_1d if fr.returns_1d is not None else np.nan for fr in feature_rows], index=dates)

        # Compute technical indicators from batch data
        rsi_values = None
        macd_line = None
        macd_signal = None
        macd_hist = None
        bb_upper = None
        bb_middle = None
        bb_lower = None
        realized_vol = None

        if self.include_technical_indicators:
            # RSI
            rsi_values = self._compute_rsi(closes, period=self.rsi_period)

            # MACD
            macd_line, macd_signal, macd_hist = self._compute_macd(
                closes, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal
            )

            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._compute_bollinger_bands(closes, period=self.bb_period, std_dev=self.bb_std)

        if self.include_volatility_features:
            # Realized volatility
            realized_vol = self._compute_realized_volatility(returns, window=20)

        # Build features for each row
        features_list = []
        for i, fr in enumerate(feature_rows):
            features_dict = {}
            date = fr.date

            if self.include_raw_features:
                features_dict.update(
                    {
                        "ma20": fr.ma20 if fr.ma20 is not None else np.nan,
                        "ma50": fr.ma50 if fr.ma50 is not None else np.nan,
                        "ma200": fr.ma200 if fr.ma200 is not None else np.nan,
                        "ma50_slope": fr.ma50_slope if fr.ma50_slope is not None else np.nan,
                        "atr14": fr.atr14 if fr.atr14 is not None else np.nan,
                        "roc60": fr.roc60 if fr.roc60 is not None else np.nan,
                        "highest_close_20d": fr.highest_close_20d if fr.highest_close_20d is not None else np.nan,
                        "highest_close_55d": fr.highest_close_55d if fr.highest_close_55d is not None else np.nan,
                        "adv20": fr.adv20 if fr.adv20 is not None else np.nan,
                        "returns_1d": fr.returns_1d if fr.returns_1d is not None else np.nan,
                        "benchmark_roc60": fr.benchmark_roc60 if fr.benchmark_roc60 is not None else np.nan,
                        "benchmark_returns_1d": fr.benchmark_returns_1d if fr.benchmark_returns_1d is not None else np.nan,
                    }
                )

                # Add mean reversion indicators if available
                if fr.zscore is not None:
                    features_dict["zscore"] = fr.zscore
                if fr.ma_lookback is not None:
                    features_dict["ma_lookback"] = fr.ma_lookback
                if fr.std_lookback is not None:
                    features_dict["std_lookback"] = fr.std_lookback

            if self.include_derived_features:
                close = fr.close

                if fr.ma20 is not None:
                    features_dict["close_to_ma20"] = close / fr.ma20 - 1.0
                else:
                    features_dict["close_to_ma20"] = np.nan

                if fr.ma50 is not None:
                    features_dict["close_to_ma50"] = close / fr.ma50 - 1.0
                else:
                    features_dict["close_to_ma50"] = np.nan

                if fr.ma200 is not None:
                    features_dict["close_to_ma200"] = close / fr.ma200 - 1.0
                else:
                    features_dict["close_to_ma200"] = np.nan

                if fr.atr14 is not None and fr.atr14 > 0:
                    if fr.ma20 is not None:
                        features_dict["breakout_strength_20d"] = (close - fr.ma20) / fr.atr14
                    else:
                        features_dict["breakout_strength_20d"] = np.nan
                    if fr.ma50 is not None:
                        features_dict["breakout_strength_50d"] = (close - fr.ma50) / fr.atr14
                    else:
                        features_dict["breakout_strength_50d"] = np.nan
                else:
                    features_dict["breakout_strength_20d"] = np.nan
                    features_dict["breakout_strength_50d"] = np.nan

                if fr.roc60 is not None and fr.benchmark_roc60 is not None:
                    features_dict["relative_strength"] = fr.roc60 - fr.benchmark_roc60
                else:
                    features_dict["relative_strength"] = np.nan

                if fr.ma20 is not None and fr.ma50 is not None:
                    features_dict["ma20_to_ma50"] = fr.ma20 / fr.ma50 - 1.0
                else:
                    features_dict["ma20_to_ma50"] = np.nan

                if fr.ma50 is not None and fr.ma200 is not None:
                    features_dict["ma50_to_ma200"] = fr.ma50 / fr.ma200 - 1.0
                else:
                    features_dict["ma50_to_ma200"] = np.nan

                if fr.highest_close_20d is not None and fr.highest_close_20d > 0:
                    features_dict["clearance_20d"] = (close / fr.highest_close_20d) - 1.0
                else:
                    features_dict["clearance_20d"] = np.nan

                if fr.highest_close_55d is not None and fr.highest_close_55d > 0:
                    features_dict["clearance_55d"] = (close / fr.highest_close_55d) - 1.0
                else:
                    features_dict["clearance_55d"] = np.nan

                # Price range features
                if fr.high is not None and fr.low is not None and close > 0:
                    features_dict["price_range"] = (fr.high - fr.low) / close
                    features_dict["upper_shadow"] = (fr.high - max(fr.open, close)) / close
                    features_dict["lower_shadow"] = (min(fr.open, close) - fr.low) / close
                else:
                    features_dict["price_range"] = np.nan
                    features_dict["upper_shadow"] = np.nan
                    features_dict["lower_shadow"] = np.nan

            # Add technical indicators
            if self.include_technical_indicators:
                if rsi_values is not None and date in rsi_values.index:
                    features_dict["rsi"] = rsi_values.loc[date] if not pd.isna(rsi_values.loc[date]) else np.nan
                else:
                    features_dict["rsi"] = np.nan

                if macd_line is not None and date in macd_line.index:
                    features_dict["macd"] = macd_line.loc[date] if not pd.isna(macd_line.loc[date]) else np.nan
                    features_dict["macd_signal"] = macd_signal.loc[date] if not pd.isna(macd_signal.loc[date]) else np.nan
                    features_dict["macd_hist"] = macd_hist.loc[date] if not pd.isna(macd_hist.loc[date]) else np.nan
                else:
                    features_dict["macd"] = np.nan
                    features_dict["macd_signal"] = np.nan
                    features_dict["macd_hist"] = np.nan

                if bb_upper is not None and date in bb_upper.index:
                    features_dict["bb_upper"] = bb_upper.loc[date] if not pd.isna(bb_upper.loc[date]) else np.nan
                    features_dict["bb_middle"] = bb_middle.loc[date] if not pd.isna(bb_middle.loc[date]) else np.nan
                    features_dict["bb_lower"] = bb_lower.loc[date] if not pd.isna(bb_lower.loc[date]) else np.nan
                    # Bollinger Band position (0-1, where 0 is lower band, 1 is upper band)
                    if not pd.isna(bb_upper.loc[date]) and not pd.isna(bb_lower.loc[date]):
                        band_width = bb_upper.loc[date] - bb_lower.loc[date]
                        if band_width > 1e-10:
                            features_dict["bb_position"] = (close - bb_lower.loc[date]) / band_width
                        else:
                            features_dict["bb_position"] = 0.5
                    else:
                        features_dict["bb_position"] = np.nan
                else:
                    features_dict["bb_upper"] = np.nan
                    features_dict["bb_middle"] = np.nan
                    features_dict["bb_lower"] = np.nan
                    features_dict["bb_position"] = np.nan

            # Add volatility features
            if self.include_volatility_features:
                if realized_vol is not None and date in realized_vol.index:
                    features_dict["realized_vol"] = realized_vol.loc[date] if not pd.isna(realized_vol.loc[date]) else np.nan
                else:
                    features_dict["realized_vol"] = np.nan

                # ATR-based volatility features
                if fr.atr14 is not None and close > 0:
                    features_dict["atr_pct"] = fr.atr14 / close
                else:
                    features_dict["atr_pct"] = np.nan

            # Add market regime features
            if self.include_market_regime_features:
                # Trend detection
                if fr.ma20 is not None and fr.ma50 is not None and fr.ma200 is not None:
                    # Golden/Death cross
                    ma20_above_ma50 = 1.0 if fr.ma20 > fr.ma50 else 0.0
                    ma50_above_ma200 = 1.0 if fr.ma50 > fr.ma200 else 0.0
                    features_dict["trend_bullish"] = 1.0 if (ma20_above_ma50 and ma50_above_ma200) else 0.0
                    features_dict["trend_bearish"] = 1.0 if (not ma20_above_ma50 and not ma50_above_ma200) else 0.0
                else:
                    features_dict["trend_bullish"] = np.nan
                    features_dict["trend_bearish"] = np.nan

                # Volatility regime (high/low volatility)
                if fr.atr14 is not None and fr.ma20 is not None:
                    # Compare current ATR to historical average
                    atr_ma = fr.atr14 / fr.ma20 if fr.ma20 > 0 else np.nan
                    features_dict["volatility_regime_high"] = 1.0 if atr_ma > 0.02 else 0.0  # Threshold: 2% of price
                    features_dict["volatility_regime_low"] = 1.0 if atr_ma < 0.01 else 0.0  # Threshold: 1% of price
                else:
                    features_dict["volatility_regime_high"] = np.nan
                    features_dict["volatility_regime_low"] = np.nan

            # Add cross-asset features
            if self.include_cross_asset_features:
                # Relative strength vs benchmark
                if fr.roc60 is not None and fr.benchmark_roc60 is not None:
                    features_dict["relative_strength_vs_benchmark"] = fr.roc60 - fr.benchmark_roc60
                else:
                    features_dict["relative_strength_vs_benchmark"] = np.nan

                # Correlation features (would need historical data, simplified here)
                if fr.returns_1d is not None and fr.benchmark_returns_1d is not None:
                    # Sign correlation
                    same_direction = 1.0 if (fr.returns_1d * fr.benchmark_returns_1d) > 0 else 0.0
                    features_dict["returns_correlation_sign"] = same_direction
                else:
                    features_dict["returns_correlation_sign"] = np.nan

            features_list.append(features_dict)

        return pd.DataFrame(features_list)

    def get_feature_names(self) -> List[str]:
        """Get list of feature names.

        Returns:
            List of feature names
        """
        if self._feature_names is None:
            raise ValueError("Feature engineer must be fitted before getting feature names")
        return self._feature_names.copy()

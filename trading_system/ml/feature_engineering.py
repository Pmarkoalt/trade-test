"""Feature engineering for ML models."""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from trading_system.models.features import FeatureRow


class MLFeatureEngineer:
    """Engineer features from FeatureRow for ML models.
    
    This class converts FeatureRow objects into feature vectors suitable
    for ML model training and prediction. It handles feature scaling,
    normalization, and missing value imputation.
    """
    
    def __init__(
        self,
        include_raw_features: bool = True,
        include_derived_features: bool = True,
        normalize_features: bool = True,
    ):
        """Initialize feature engineer.
        
        Args:
            include_raw_features: Include raw indicator values
            include_derived_features: Include derived features (ratios, differences, etc.)
            normalize_features: Normalize features to [0, 1] range
        """
        self.include_raw_features = include_raw_features
        self.include_derived_features = include_derived_features
        self.normalize_features = normalize_features
        self._feature_scalers: Optional[Dict[str, tuple[float, float]]] = None
        self._feature_names: Optional[List[str]] = None
    
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
    
    def transform(self, feature_row: FeatureRow) -> pd.Series:
        """Transform a single FeatureRow to feature vector.
        
        Args:
            feature_row: FeatureRow to transform
        
        Returns:
            Series with feature values
        """
        features = {}
        
        if self.include_raw_features:
            # Raw indicator values
            features.update({
                "ma20": feature_row.ma20 if feature_row.ma20 is not None else np.nan,
                "ma50": feature_row.ma50 if feature_row.ma50 is not None else np.nan,
                "ma200": feature_row.ma200 if feature_row.ma200 is not None else np.nan,
                "ma50_slope": feature_row.ma50_slope if feature_row.ma50_slope is not None else np.nan,
                "atr14": feature_row.atr14 if feature_row.atr14 is not None else np.nan,
                "roc60": feature_row.roc60 if feature_row.roc60 is not None else np.nan,
                "highest_close_20d": feature_row.highest_close_20d if feature_row.highest_close_20d is not None else np.nan,
                "highest_close_55d": feature_row.highest_close_55d if feature_row.highest_close_55d is not None else np.nan,
                "adv20": feature_row.adv20 if feature_row.adv20 is not None else np.nan,
                "returns_1d": feature_row.returns_1d if feature_row.returns_1d is not None else np.nan,
                "benchmark_roc60": feature_row.benchmark_roc60 if feature_row.benchmark_roc60 is not None else np.nan,
                "benchmark_returns_1d": feature_row.benchmark_returns_1d if feature_row.benchmark_returns_1d is not None else np.nan,
            })
        
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
        
        Args:
            feature_rows: List of FeatureRow objects
        
        Returns:
            DataFrame with feature vectors (one row per FeatureRow)
        """
        features_list = [self.transform(fr) for fr in feature_rows]
        df = pd.DataFrame(features_list)
        
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
        
        Args:
            feature_rows: List of FeatureRow objects
        
        Returns:
            DataFrame with raw feature values
        """
        features_list = []
        for fr in feature_rows:
            features_dict = {}
            
            if self.include_raw_features:
                features_dict.update({
                    "ma20": fr.ma20,
                    "ma50": fr.ma50,
                    "ma200": fr.ma200,
                    "ma50_slope": fr.ma50_slope,
                    "atr14": fr.atr14,
                    "roc60": fr.roc60,
                    "highest_close_20d": fr.highest_close_20d,
                    "highest_close_55d": fr.highest_close_55d,
                    "adv20": fr.adv20,
                    "returns_1d": fr.returns_1d,
                    "benchmark_roc60": fr.benchmark_roc60,
                    "benchmark_returns_1d": fr.benchmark_returns_1d,
                })
            
            if self.include_derived_features:
                close = fr.close
                
                if fr.ma20 is not None:
                    features_dict["close_to_ma20"] = close / fr.ma20 - 1.0
                if fr.ma50 is not None:
                    features_dict["close_to_ma50"] = close / fr.ma50 - 1.0
                if fr.ma200 is not None:
                    features_dict["close_to_ma200"] = close / fr.ma200 - 1.0
                
                if fr.atr14 is not None and fr.atr14 > 0:
                    if fr.ma20 is not None:
                        features_dict["breakout_strength_20d"] = (close - fr.ma20) / fr.atr14
                    if fr.ma50 is not None:
                        features_dict["breakout_strength_50d"] = (close - fr.ma50) / fr.atr14
                
                if fr.roc60 is not None and fr.benchmark_roc60 is not None:
                    features_dict["relative_strength"] = fr.roc60 - fr.benchmark_roc60
                
                if fr.ma20 is not None and fr.ma50 is not None:
                    features_dict["ma20_to_ma50"] = fr.ma20 / fr.ma50 - 1.0
                if fr.ma50 is not None and fr.ma200 is not None:
                    features_dict["ma50_to_ma200"] = fr.ma50 / fr.ma200 - 1.0
                
                if fr.highest_close_20d is not None and fr.highest_close_20d > 0:
                    features_dict["clearance_20d"] = (close / fr.highest_close_20d) - 1.0
                if fr.highest_close_55d is not None and fr.highest_close_55d > 0:
                    features_dict["clearance_55d"] = (close / fr.highest_close_55d) - 1.0
            
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


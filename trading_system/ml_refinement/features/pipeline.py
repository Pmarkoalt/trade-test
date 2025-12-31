"""Feature extraction pipeline."""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from trading_system.ml_refinement.config import FeatureConfig, FeatureSet, FeatureVector
from trading_system.ml_refinement.features.feature_registry import FeatureRegistry
from trading_system.ml_refinement.features.extractors.base_extractor import (
    OHLCVExtractor,
)


class FeaturePipeline:
    """
    Pipeline for extracting features from signals.

    Example:
        pipeline = FeaturePipeline(config)

        # Extract features for a signal
        features = pipeline.extract_features(
            signal_id="sig-123",
            ohlcv_data=ohlcv_df,
            signal_metadata=signal_dict,
            benchmark_data=spy_df,
        )

        # Create feature vector
        fv = pipeline.create_feature_vector(
            signal_id="sig-123",
            features=features,
            target=2.5,  # Optional, fill in later
        )
    """

    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        registry: Optional[FeatureRegistry] = None,
    ):
        """
        Initialize pipeline.

        Args:
            config: Feature configuration.
            registry: Feature registry (uses default if not provided).
        """
        self.config = config or FeatureConfig()
        self.registry = registry or FeatureRegistry()

        # Get extractors based on feature set
        self._extractors = self._get_extractors_for_set()

    def _get_extractors_for_set(self) -> List:
        """Get extractors for configured feature set."""
        if self.config.feature_set == FeatureSet.MINIMAL:
            # Only trend and signal features
            return [
                self.registry.get("trend_features"),
                self.registry.get("signal_metadata"),
            ]
        elif self.config.feature_set == FeatureSet.STANDARD:
            # Standard set
            return [
                self.registry.get("trend_features"),
                self.registry.get("momentum_features"),
                self.registry.get("volatility_features"),
                self.registry.get("signal_metadata"),
            ]
        elif self.config.feature_set == FeatureSet.EXTENDED:
            # All features
            return self.registry.get_all()
        else:
            # Custom - use specified features
            extractors = []
            for name in self.config.custom_features:
                try:
                    extractors.append(self.registry.get(name))
                except KeyError:
                    logger.warning(f"Unknown extractor: {name}")
            return extractors

    def extract_features(
        self,
        signal_id: str,
        ohlcv_data: pd.DataFrame,
        signal_metadata: Dict[str, Any],
        benchmark_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Extract all features for a signal.

        Args:
            signal_id: Unique signal identifier.
            ohlcv_data: OHLCV data for the symbol.
            signal_metadata: Signal metadata dictionary.
            benchmark_data: Optional benchmark OHLCV data.

        Returns:
            Dictionary of feature_name -> value.
        """
        all_features = {}

        for extractor in self._extractors:
            try:
                if isinstance(extractor, OHLCVExtractor):
                    # OHLCV-based extractor
                    if extractor.category == "market" and benchmark_data is not None:
                        features = extractor.extract(
                            ohlcv_data,
                            benchmark_data=benchmark_data,
                        )
                    else:
                        features = extractor.extract(ohlcv_data)
                else:
                    # Metadata-based extractor
                    features = extractor.extract(signal_metadata)

                all_features.update(features)

            except Exception as e:
                logger.warning(
                    f"Error extracting features from {extractor.name}: {e}"
                )
                # Fill with zeros for failed extractor
                for name in extractor.feature_names:
                    all_features[name] = 0.0

        # Apply scaling if configured
        if self.config.scale_features:
            all_features = self._scale_features(all_features)

        logger.debug(
            f"Extracted {len(all_features)} features for signal {signal_id}"
        )

        return all_features

    def create_feature_vector(
        self,
        signal_id: str,
        features: Dict[str, float],
        target: Optional[float] = None,
        timestamp: Optional[str] = None,
    ) -> FeatureVector:
        """
        Create a FeatureVector from extracted features.

        Args:
            signal_id: Signal identifier.
            features: Extracted features dictionary.
            target: Optional target value (R-multiple).
            timestamp: Optional timestamp (defaults to now).

        Returns:
            FeatureVector instance.
        """
        return FeatureVector(
            signal_id=signal_id,
            timestamp=timestamp or datetime.now().isoformat(),
            features=features,
            target=target,
            target_binary=1 if target and target > 0 else (0 if target else None),
        )

    def _scale_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Apply feature scaling.

        Note: In production, scaling parameters should be fitted on training
        data and stored for consistent scaling of new data.
        """
        # For now, just clip extreme values
        scaled = {}
        for name, value in features.items():
            # Clip to reasonable range
            scaled[name] = max(-10.0, min(10.0, value))
        return scaled

    def get_feature_names(self) -> List[str]:
        """Get all feature names produced by this pipeline."""
        names = []
        for extractor in self._extractors:
            names.extend(extractor.feature_names)
        return names

    def get_feature_count(self) -> int:
        """Get total feature count."""
        return len(self.get_feature_names())


class FeatureScaler:
    """
    Feature scaler for normalizing features.

    Fits scaling parameters on training data and applies to new data.
    """

    def __init__(self, method: str = "standard"):
        """
        Initialize scaler.

        Args:
            method: Scaling method ("standard", "minmax", "robust").
        """
        self.method = method
        self.fitted = False
        self._params: Dict[str, Dict] = {}

    def fit(self, feature_data: Dict[str, List[float]]):
        """
        Fit scaling parameters from training data.

        Args:
            feature_data: Dict of feature_name -> list of values.
        """
        import numpy as np

        for name, values in feature_data.items():
            values_arr = np.array(values)

            if self.method == "standard":
                self._params[name] = {
                    "mean": np.mean(values_arr),
                    "std": np.std(values_arr) or 1.0,
                }
            elif self.method == "minmax":
                self._params[name] = {
                    "min": np.min(values_arr),
                    "max": np.max(values_arr) or 1.0,
                }
            elif self.method == "robust":
                self._params[name] = {
                    "median": np.median(values_arr),
                    "iqr": np.percentile(values_arr, 75) - np.percentile(values_arr, 25) or 1.0,
                }

        self.fitted = True

    def transform(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Transform features using fitted parameters.

        Args:
            features: Feature dictionary to transform.

        Returns:
            Transformed features.
        """
        if not self.fitted:
            return features

        transformed = {}
        for name, value in features.items():
            if name not in self._params:
                transformed[name] = value
                continue

            params = self._params[name]

            if self.method == "standard":
                transformed[name] = (value - params["mean"]) / params["std"]
            elif self.method == "minmax":
                range_val = params["max"] - params["min"]
                if range_val > 0:
                    transformed[name] = (value - params["min"]) / range_val
                else:
                    transformed[name] = 0.0
            elif self.method == "robust":
                transformed[name] = (value - params["median"]) / params["iqr"]

        return transformed

    def fit_transform(
        self,
        feature_data: Dict[str, List[float]],
    ) -> List[Dict[str, float]]:
        """Fit and transform in one step."""
        self.fit(feature_data)

        # Reconstruct individual feature dicts
        n_samples = len(list(feature_data.values())[0])
        result = []

        for i in range(n_samples):
            sample = {name: values[i] for name, values in feature_data.items()}
            result.append(self.transform(sample))

        return result


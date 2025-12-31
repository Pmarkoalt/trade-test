"""Prediction service for ML inference."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from trading_system.ml_refinement.config import MLConfig, ModelType
from trading_system.ml_refinement.features.pipeline import FeaturePipeline
from trading_system.ml_refinement.storage.feature_db import FeatureDatabase

# Try to import ModelRegistry, create stub if not available
try:
    from trading_system.ml_refinement.models.model_registry import ModelRegistry
except ImportError:
    # Create minimal ModelRegistry stub
    class ModelRegistry:
        """Minimal model registry stub."""

        def __init__(self, feature_db: FeatureDatabase):
            self.feature_db = feature_db
            self._models: Dict[ModelType, object] = {}

        def get_active(self, model_type: ModelType):
            """Get active model for type."""
            if model_type in self._models:
                return self._models[model_type]

            # Try to get from database
            metadata = self.feature_db.get_active_model(model_type.value)
            if metadata:
                # In real implementation, would load model from disk
                # For now, return None to indicate no model available
                return None
            return None


class PredictionService:
    """
    Service for making ML predictions on signals.

    Example:
        service = PredictionService(config, feature_db, model_registry)

        # Predict quality for a signal
        quality_score = service.predict_signal_quality(
            signal_id="sig-123",
            ohlcv_data=ohlcv_df,
            signal_metadata=signal_dict,
        )

        print(f"Quality: {quality_score:.2f}")  # 0-1
    """

    def __init__(
        self,
        config: MLConfig,
        feature_db: FeatureDatabase,
        model_registry: Optional[ModelRegistry] = None,
    ):
        """
        Initialize prediction service.

        Args:
            config: ML configuration.
            feature_db: Feature database.
            model_registry: Model registry (creates default if not provided).
        """
        self.config = config
        self.feature_db = feature_db
        self.model_registry = model_registry or ModelRegistry(feature_db)
        self.feature_pipeline = FeaturePipeline(config.features)

        # Cached models
        self._models: Dict[ModelType, object] = {}

    def predict_signal_quality(
        self,
        signal_id: str,
        ohlcv_data,
        signal_metadata: Dict,
        benchmark_data=None,
        store_features: bool = True,
    ) -> float:
        """
        Predict signal quality (probability of success).

        Args:
            signal_id: Unique signal identifier.
            ohlcv_data: OHLCV DataFrame for symbol.
            signal_metadata: Signal metadata dictionary.
            benchmark_data: Optional benchmark OHLCV.
            store_features: Whether to store features in database.

        Returns:
            Quality score 0-1 (probability of profitable trade).
        """
        if not self.config.enabled:
            return 0.5  # Default neutral score

        # Extract features
        features = self.feature_pipeline.extract_features(
            signal_id=signal_id,
            ohlcv_data=ohlcv_data,
            signal_metadata=signal_metadata,
            benchmark_data=benchmark_data,
        )

        # Store features if requested
        if store_features:
            fv = self.feature_pipeline.create_feature_vector(
                signal_id=signal_id,
                features=features,
            )
            self.feature_db.store_feature_vector(
                fv,
                symbol=signal_metadata.get("symbol", ""),
                asset_class=signal_metadata.get("asset_class", ""),
                signal_type=signal_metadata.get("signal_type", ""),
                conviction=signal_metadata.get("conviction", ""),
            )

        # Get model
        model = self._get_model(ModelType.SIGNAL_QUALITY)
        if model is None:
            logger.warning("No active signal quality model")
            return 0.5

        # Prepare feature array
        feature_names = self.feature_pipeline.get_feature_names()
        X = np.array([[features.get(name, 0.0) for name in feature_names]])

        # Predict
        try:
            y_proba = model.predict_proba(X)

            # Extract positive class probability
            if y_proba.ndim > 1:
                quality_score = float(y_proba[0, 1] if y_proba.shape[1] > 1 else y_proba[0, 0])
            else:
                quality_score = float(y_proba[0])

            # Log prediction
            self.feature_db.log_prediction(
                signal_id=signal_id,
                model_id=getattr(model, "model_id", "unknown"),
                quality_score=quality_score,
            )

            return quality_score

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.5

    def predict_batch(
        self,
        signals: List[Dict],
        ohlcv_dict: Dict,
        benchmark_data=None,
    ) -> Dict[str, float]:
        """
        Predict quality for multiple signals.

        Args:
            signals: List of signal dictionaries.
            ohlcv_dict: Dict of symbol -> OHLCV DataFrame.
            benchmark_data: Optional benchmark OHLCV.

        Returns:
            Dict of signal_id -> quality_score.
        """
        results = {}

        for signal in signals:
            signal_id = signal.get("signal_id", signal.get("id", ""))
            symbol = signal.get("symbol", "")

            if symbol not in ohlcv_dict:
                logger.warning(f"No OHLCV data for {symbol}")
                results[signal_id] = 0.5
                continue

            quality = self.predict_signal_quality(
                signal_id=signal_id,
                ohlcv_data=ohlcv_dict[symbol],
                signal_metadata=signal,
                benchmark_data=benchmark_data,
            )
            results[signal_id] = quality

        return results

    def update_prediction_outcomes(
        self,
        outcomes: List[Tuple[str, float]],
    ):
        """
        Update predictions with actual outcomes.

        Args:
            outcomes: List of (signal_id, actual_r_multiple).
        """
        for signal_id, actual_r in outcomes:
            # Update feature target
            self.feature_db.update_target(
                signal_id=signal_id,
                r_multiple=actual_r,
            )

            # Update prediction log
            self.feature_db.update_prediction_actual(
                signal_id=signal_id,
                actual_r=actual_r,
            )

    def _get_model(self, model_type: ModelType):
        """Get or load model."""
        if model_type in self._models:
            return self._models[model_type]

        model = self.model_registry.get_active(model_type)
        if model:
            self._models[model_type] = model

        return model

    def reload_models(self):
        """Reload all models from registry."""
        self._models.clear()
        for model_type in ModelType:
            self._get_model(model_type)

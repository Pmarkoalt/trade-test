"""Model registry for managing model versions."""

from pathlib import Path
from typing import Dict, List, Optional, Type

from loguru import logger

from trading_system.ml_refinement.config import ModelMetadata, ModelType
from trading_system.ml_refinement.models.base_model import BaseModel, SignalQualityModel
from trading_system.ml_refinement.storage.feature_db import FeatureDatabase


class ModelRegistry:
    """
    Registry for managing ML models.

    Example:
        registry = ModelRegistry(model_dir="models/", db=feature_db)

        # Train and register
        model = SignalQualityModel()
        model.fit(X_train, y_train)
        registry.register(model)

        # Activate for production
        registry.activate(model.model_id)

        # Get active model
        active = registry.get_active(ModelType.SIGNAL_QUALITY)
    """

    # Model class mapping
    MODEL_CLASSES: Dict[ModelType, Type[BaseModel]] = {
        ModelType.SIGNAL_QUALITY: SignalQualityModel,
    }

    def __init__(
        self,
        model_dir: str = "models/",
        db: Optional[FeatureDatabase] = None,
    ):
        """
        Initialize registry.

        Args:
            model_dir: Directory for storing model files.
            db: Feature database for metadata storage.
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.db = db

        # In-memory cache of active models
        self._active_models: Dict[ModelType, BaseModel] = {}

    def register(self, model: BaseModel) -> bool:
        """
        Register a trained model.

        Args:
            model: Trained model to register.

        Returns:
            True if successful.
        """
        if not model.is_fitted:
            logger.error("Cannot register unfitted model")
            return False

        # Save model file
        model_path = self.model_dir / f"{model.model_id}.pkl"
        if not model.save(str(model_path)):
            return False

        # Save metadata to database
        if self.db:
            metadata = model.get_metadata()
            self.db.register_model(metadata)

        logger.info(f"Registered model {model.model_id}")
        return True

    def activate(self, model_id: str) -> bool:
        """
        Activate a model for production use.

        Args:
            model_id: ID of model to activate.

        Returns:
            True if successful.
        """
        # Load model
        model_path = self.model_dir / f"{model_id}.pkl"
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False

        # Determine model type from ID
        model_type = None
        for mt in ModelType:
            if model_id.startswith(mt.value):
                model_type = mt
                break

        if not model_type:
            logger.error(f"Cannot determine model type from ID: {model_id}")
            return False

        # Load model
        model_class = self.MODEL_CLASSES.get(model_type)
        if not model_class:
            logger.error(f"No model class for type: {model_type}")
            return False

        model = model_class()
        if not model.load(str(model_path)):
            return False

        # Update database
        if self.db:
            self.db.activate_model(model_id)

        # Cache active model
        self._active_models[model_type] = model

        logger.info(f"Activated model {model_id}")
        return True

    def get_active(self, model_type: ModelType) -> Optional[BaseModel]:
        """
        Get the active model for a type.

        Args:
            model_type: Type of model.

        Returns:
            Active model or None.
        """
        # Check cache first
        if model_type in self._active_models:
            return self._active_models[model_type]

        # Load from database
        if self.db:
            metadata = self.db.get_active_model(model_type.value)
            if metadata:
                return self._load_model(metadata.model_id, model_type)

        return None

    def get_model_history(
        self,
        model_type: ModelType,
        limit: int = 10,
    ) -> List[ModelMetadata]:
        """Get model version history."""
        if not self.db:
            return []

        return self.db.get_model_history(model_type.value, limit)

    def _load_model(
        self,
        model_id: str,
        model_type: ModelType,
    ) -> Optional[BaseModel]:
        """Load a model by ID."""
        model_path = self.model_dir / f"{model_id}.pkl"
        if not model_path.exists():
            return None

        model_class = self.MODEL_CLASSES.get(model_type)
        if not model_class:
            return None

        model = model_class()
        if model.load(str(model_path)):
            self._active_models[model_type] = model
            return model

        return None

    def compare_models(
        self,
        model_type: ModelType,
        metric: str = "auc",
    ) -> List[Dict]:
        """
        Compare all models of a type by metric.

        Args:
            model_type: Type of models to compare.
            metric: Metric to compare by.

        Returns:
            List of models sorted by metric.
        """
        history = self.get_model_history(model_type, limit=50)

        results = []
        for metadata in history:
            val_metric = metadata.validation_metrics.get(metric, 0)
            results.append(
                {
                    "model_id": metadata.model_id,
                    "version": metadata.version,
                    "created_at": metadata.created_at,
                    "metric": val_metric,
                    "is_active": metadata.is_active,
                }
            )

        return sorted(results, key=lambda x: x["metric"], reverse=True)

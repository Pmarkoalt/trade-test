"""Base class for ML models."""

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from trading_system.ml_refinement.config import ModelMetadata, ModelType

if TYPE_CHECKING:
    from sklearn.ensemble import GradientBoostingClassifier


class BaseModel(ABC):
    """
    Abstract base class for all ML models.

    All models should implement:
    - fit: Train on data
    - predict: Make predictions
    - save/load: Persistence
    """

    def __init__(self, model_type: ModelType, version: str = "1.0"):
        """
        Initialize base model.

        Args:
            model_type: Type of model.
            version: Model version string.
        """
        self.model_type = model_type
        self.version = version
        self.model_id = f"{model_type.value}_{version}_{uuid.uuid4().hex[:8]}"
        self.is_fitted = False

        # Training metadata
        self._train_start_date: Optional[str] = None
        self._train_end_date: Optional[str] = None
        self._train_samples: int = 0
        self._validation_samples: int = 0
        self._feature_names: List[str] = []
        self._feature_importance: Dict[str, float] = {}
        self._train_metrics: Dict[str, float] = {}
        self._validation_metrics: Dict[str, float] = {}

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Train the model.

        Args:
            X_train: Training features (n_samples, n_features).
            y_train: Training targets (n_samples,).
            X_val: Optional validation features.
            y_val: Optional validation targets.
            feature_names: Optional list of feature names.

        Returns:
            Dictionary of training metrics.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features (n_samples, n_features).

        Returns:
            Predictions (n_samples,).
        """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities (for classification).

        Args:
            X: Features (n_samples, n_features).

        Returns:
            Probabilities (n_samples, n_classes) or (n_samples,) for binary.
        """

    @abstractmethod
    def save(self, path: str) -> bool:
        """
        Save model to disk.

        Args:
            path: Path to save model.

        Returns:
            True if successful.
        """

    @abstractmethod
    def load(self, path: str) -> bool:
        """
        Load model from disk.

        Args:
            path: Path to load model from.

        Returns:
            True if successful.
        """

    def get_metadata(self) -> ModelMetadata:
        """Get model metadata."""
        return ModelMetadata(
            model_id=self.model_id,
            model_type=self.model_type,
            version=self.version,
            created_at=datetime.now().isoformat(),
            train_start_date=self._train_start_date or "",
            train_end_date=self._train_end_date or "",
            train_samples=self._train_samples,
            validation_samples=self._validation_samples,
            train_metrics=self._train_metrics,
            validation_metrics=self._validation_metrics,
            feature_names=self._feature_names,
            feature_importance=self._feature_importance,
        )

    def set_training_period(self, start_date: str, end_date: str):
        """Set training period for metadata."""
        self._train_start_date = start_date
        self._train_end_date = end_date


class SignalQualityModel(BaseModel):
    """
    Model for predicting signal quality (probability of success).

    Uses gradient boosting by default for interpretability and performance.
    """

    def __init__(self, version: str = "1.0", **kwargs):
        """
        Initialize signal quality model.

        Args:
            version: Model version.
            **kwargs: Additional model parameters.
        """
        super().__init__(ModelType.SIGNAL_QUALITY, version)

        self.params = {
            "n_estimators": kwargs.get("n_estimators", 100),
            "max_depth": kwargs.get("max_depth", 5),
            "learning_rate": kwargs.get("learning_rate", 0.1),
            "min_samples_leaf": kwargs.get("min_samples_leaf", 20),
            "subsample": kwargs.get("subsample", 0.8),
            "random_state": kwargs.get("random_state", 42),
        }

        self._model: Optional["GradientBoostingClassifier"] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Train the signal quality model."""
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        except ImportError:
            logger.error("scikit-learn required for ML models")
            raise

        # Convert to binary classification (win/loss)
        y_train_binary = (y_train > 0).astype(int)

        # Initialize model
        self._model = GradientBoostingClassifier(**self.params)

        # Fit
        self._model.fit(X_train, y_train_binary)

        # Store metadata
        self._train_samples = len(X_train)
        self._feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        # Calculate training metrics
        train_pred = self._model.predict(X_train)
        train_proba = self._model.predict_proba(X_train)[:, 1]

        self._train_metrics = {
            "accuracy": accuracy_score(y_train_binary, train_pred),
            "auc": roc_auc_score(y_train_binary, train_proba),
            "f1": f1_score(y_train_binary, train_pred),
        }

        # Validation metrics
        if X_val is not None and y_val is not None:
            y_val_binary = (y_val > 0).astype(int)
            val_pred = self._model.predict(X_val)
            val_proba = self._model.predict_proba(X_val)[:, 1]

            self._validation_samples = len(X_val)
            self._validation_metrics = {
                "accuracy": accuracy_score(y_val_binary, val_pred),
                "auc": roc_auc_score(y_val_binary, val_proba),
                "f1": f1_score(y_val_binary, val_pred),
            }

        # Feature importance
        if self._model is not None and hasattr(self._model, "feature_importances_"):
            importances = self._model.feature_importances_
            self._feature_importance = {name: float(imp) for name, imp in zip(self._feature_names, importances)}

        self.is_fitted = True
        logger.info(f"Trained {self.model_type.value} model: " f"train_auc={self._train_metrics.get('auc', 0):.3f}")

        return self._train_metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict win/loss (1/0)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        if self._model is None:
            raise ValueError("Model not initialized")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of success (win)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        if self._model is None:
            raise ValueError("Model not initialized")
        return self._model.predict_proba(X)[:, 1]

    def save(self, path: str) -> bool:
        """Save model to disk."""
        import pickle

        try:
            model_data = {
                "model": self._model,
                "params": self.params,
                "metadata": self.get_metadata(),
                "feature_names": self._feature_names,
            }

            with open(path, "wb") as f:
                pickle.dump(model_data, f)

            logger.info(f"Saved model to {path}")
            return True

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def load(self, path: str) -> bool:
        """Load model from disk."""
        import pickle

        try:
            with open(path, "rb") as f:
                model_data = pickle.load(f)

            self._model = model_data["model"]
            self.params = model_data["params"]
            self._feature_names = model_data.get("feature_names", [])

            metadata = model_data.get("metadata")
            if metadata:
                self.model_id = metadata.model_id
                self.version = metadata.version
                self._train_metrics = metadata.train_metrics
                self._validation_metrics = metadata.validation_metrics
                self._feature_importance = metadata.feature_importance

            self.is_fitted = True
            logger.info(f"Loaded model from {path}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        sorted_features = sorted(
            self._feature_importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_features[:n]

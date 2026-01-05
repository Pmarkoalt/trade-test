"""ML model wrappers and interfaces."""

import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class ModelType(str, Enum):
    """Supported ML model types."""

    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    LINEAR_REGRESSION = "linear_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    NEURAL_NETWORK = "neural_network"


@dataclass
class ModelMetadata:
    """Metadata for a trained ML model."""

    model_id: str
    model_type: ModelType
    version: str
    training_date: pd.Timestamp
    training_samples: int
    feature_names: list[str]
    target_name: str
    hyperparameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    training_time_seconds: float
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type.value,
            "version": self.version,
            "training_date": self.training_date.isoformat(),
            "training_samples": self.training_samples,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "hyperparameters": self.hyperparameters,
            "performance_metrics": self.performance_metrics,
            "training_time_seconds": self.training_time_seconds,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create metadata from dictionary."""
        data = data.copy()
        data["model_type"] = ModelType(data["model_type"])
        data["training_date"] = pd.Timestamp(data["training_date"])
        return cls(**data)

    def save(self, filepath: Path) -> None:
        """Save metadata to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "ModelMetadata":
        """Load metadata from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


class MLModel(ABC):
    """Abstract base class for ML models.

    This class provides a unified interface for different ML model types
    and handles serialization, prediction, and metadata management.
    """

    def __init__(
        self,
        model_type: ModelType,
        model_id: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ):
        """Initialize ML model.

        Args:
            model_type: Type of model
            model_id: Unique identifier for the model
            hyperparameters: Model hyperparameters
        """
        self.model_type = model_type
        self.model_id = model_id or f"{model_type.value}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        self.hyperparameters = hyperparameters or {}
        self._model: Any = None
        self._metadata: Optional[ModelMetadata] = None
        self._is_trained = False

    @abstractmethod
    def _create_model(self) -> Any:
        """Create the underlying model instance.

        Returns:
            The model instance (e.g., sklearn model, xgboost model, etc.)
        """

    @abstractmethod
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[tuple[pd.DataFrame, pd.Series]] = None,
    ) -> Dict[str, float]:
        """Train the model.

        Args:
            X: Feature matrix (DataFrame with feature names)
            y: Target vector (Series)
            validation_data: Optional (X_val, y_val) tuple for validation

        Returns:
            Dictionary of performance metrics
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.

        Args:
            X: Feature matrix (DataFrame with feature names)

        Returns:
            Array of predictions
        """

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Make probability predictions (if supported).

        Args:
            X: Feature matrix (DataFrame with feature names)

        Returns:
            Array of probability predictions, or None if not supported
        """
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        return None

    def set_metadata(self, metadata: ModelMetadata) -> None:
        """Set model metadata.

        Args:
            metadata: Model metadata
        """
        self._metadata = metadata
        self.model_id = metadata.model_id

    def get_metadata(self) -> Optional[ModelMetadata]:
        """Get model metadata.

        Returns:
            Model metadata if available, None otherwise
        """
        return self._metadata

    def save(self, directory: Path) -> None:
        """Save model and metadata to directory.

        Args:
            directory: Directory to save model files
        """
        directory.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = directory / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self._model, f)

        # Save metadata
        if self._metadata:
            metadata_path = directory / "metadata.json"
            self._metadata.save(metadata_path)

        # Save hyperparameters
        # Detect task type if not already set (for RandomForest/GradientBoosting)
        save_hyperparameters = self.hyperparameters.copy()
        if self.model_type in [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING]:
            if "task" not in save_hyperparameters:
                # Determine task based on whether model has predict_proba
                if hasattr(self._model, "predict_proba"):
                    # Check if it's actually a classifier by examining the model
                    # This is a heuristic - classifiers have predict_proba
                    save_hyperparameters["task"] = "classification"
                else:
                    save_hyperparameters["task"] = "regression"

        config_path = directory / "config.json"
        with open(config_path, "w") as f:
            json.dump(
                {
                    "model_type": self.model_type.value,
                    "model_id": self.model_id,
                    "hyperparameters": save_hyperparameters,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, directory: Path) -> "MLModel":
        """Load model and metadata from directory.

        Args:
            directory: Directory containing model files

        Returns:
            Loaded MLModel instance
        """
        # Load config
        config_path = directory / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        # Create model instance
        model_type = ModelType(config["model_type"])
        model = cls._create_model_instance(model_type, config.get("hyperparameters", {}))
        model.model_id = config["model_id"]

        # Load model
        model_path = directory / "model.pkl"
        with open(model_path, "rb") as f:
            model._model = pickle.load(f)

        model._is_trained = True

        # Load metadata if available
        metadata_path = directory / "metadata.json"
        if metadata_path.exists():
            model._metadata = ModelMetadata.load(metadata_path)

        return model

    @staticmethod
    def _create_model_instance(model_type: ModelType, hyperparameters: Dict[str, Any]) -> "MLModel":
        """Create a model instance based on type.

        Args:
            model_type: Type of model to create
            hyperparameters: Model hyperparameters

        Returns:
            MLModel instance
        """
        # Lazy import to avoid requiring ML libraries if not using ML
        if model_type == ModelType.RANDOM_FOREST:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            # Determine if classification or regression based on hyperparameters
            if hyperparameters.get("task") == "classification":
                return SklearnMLModel(
                    model_type=model_type,
                    hyperparameters=hyperparameters,
                    sklearn_model_class=RandomForestClassifier,
                )
            else:
                return SklearnMLModel(
                    model_type=model_type,
                    hyperparameters=hyperparameters,
                    sklearn_model_class=RandomForestRegressor,
                )
        elif model_type == ModelType.GRADIENT_BOOSTING:
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

            if hyperparameters.get("task") == "classification":
                return SklearnMLModel(
                    model_type=model_type,
                    hyperparameters=hyperparameters,
                    sklearn_model_class=GradientBoostingClassifier,
                )
            else:
                return SklearnMLModel(
                    model_type=model_type,
                    hyperparameters=hyperparameters,
                    sklearn_model_class=GradientBoostingRegressor,
                )
        elif model_type == ModelType.LINEAR_REGRESSION:
            from sklearn.linear_model import LinearRegression

            return SklearnMLModel(
                model_type=model_type,
                hyperparameters=hyperparameters,
                sklearn_model_class=LinearRegression,
            )
        elif model_type == ModelType.LOGISTIC_REGRESSION:
            from sklearn.linear_model import LogisticRegression

            return SklearnMLModel(
                model_type=model_type,
                hyperparameters=hyperparameters,
                sklearn_model_class=LogisticRegression,
            )
        else:
            raise ValueError(f"Model type {model_type} not yet implemented")


class SklearnMLModel(MLModel):
    """Wrapper for scikit-learn models."""

    def __init__(
        self,
        model_type: ModelType,
        sklearn_model_class: type,
        model_id: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ):
        """Initialize sklearn model wrapper.

        Args:
            model_type: Type of model
            sklearn_model_class: sklearn model class to use
            model_id: Unique identifier for the model
            hyperparameters: Model hyperparameters (passed to sklearn model)
        """
        super().__init__(model_type, model_id, hyperparameters)
        self._sklearn_model_class = sklearn_model_class
        self._model = None

    def _create_model(self) -> Any:
        """Create sklearn model instance."""
        # Filter hyperparameters to exclude "task" which is not a sklearn parameter
        # This is a simplified approach - in production, validate against model's __init__ signature
        if self._model is None:
            sklearn_params = {k: v for k, v in self.hyperparameters.items() if k != "task"}
            self._model = self._sklearn_model_class(**sklearn_params)
        return self._model

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[tuple[pd.DataFrame, pd.Series]] = None,
    ) -> Dict[str, float]:
        """Train the sklearn model.

        Args:
            X: Feature matrix
            y: Target vector
            validation_data: Optional validation data (not used for sklearn, but kept for interface)

        Returns:
            Dictionary of performance metrics
        """
        import time

        from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

        start_time = time.time()

        self._create_model()
        self._model.fit(X, y)

        training_time = time.time() - start_time
        self._is_trained = True

        # Compute metrics
        train_pred = self._model.predict(X)

        # Determine if classification or regression
        if hasattr(self._model, "predict_proba") or len(np.unique(y)) <= 10:
            # Classification metrics
            accuracy = accuracy_score(y, train_pred)
            metrics = {
                "accuracy": accuracy,
                "training_time_seconds": training_time,
            }
            if validation_data:
                X_val, y_val = validation_data
                val_pred = self._model.predict(X_val)
                metrics["val_accuracy"] = accuracy_score(y_val, val_pred)
        else:
            # Regression metrics
            mse = mean_squared_error(y, train_pred)
            mae = mean_absolute_error(y, train_pred)
            r2 = r2_score(y, train_pred)
            metrics = {
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "rmse": np.sqrt(mse),
                "training_time_seconds": training_time,
            }
            if validation_data:
                X_val, y_val = validation_data
                val_pred = self._model.predict(X_val)
                metrics["val_mse"] = mean_squared_error(y_val, val_pred)
                metrics["val_mae"] = mean_absolute_error(y_val, val_pred)
                metrics["val_r2"] = r2_score(y_val, val_pred)

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Array of predictions
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before prediction")
        return self._model.predict(X)


def hyperparameter_tuning(
    model_type: ModelType,
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: Dict[str, List[Any]],
    cv: int = 5,
    scoring: Optional[str] = None,
    n_jobs: int = -1,
) -> Tuple[Dict[str, Any], float]:
    """Perform hyperparameter tuning using grid search.

    Args:
        model_type: Type of model to tune
        X: Feature matrix
        y: Target vector
        param_grid: Dictionary mapping parameter names to lists of values to try
        cv: Number of cross-validation folds
        scoring: Scoring metric (default: auto-detect based on task)
        n_jobs: Number of parallel jobs (-1 for all cores)

    Returns:
        Tuple of (best_parameters, best_score)
    """
    from sklearn.model_selection import GridSearchCV

    # Create base model
    base_model = MLModel._create_model_instance(
        model_type=model_type,
        hyperparameters={},
    )

    # Determine scoring if not provided
    if scoring is None:
        # Auto-detect: classification if few unique values, regression otherwise
        if len(np.unique(y)) <= 10:
            scoring = "accuracy"
        else:
            scoring = "r2"

    # Create grid search
    # Note: We need to work with the underlying sklearn model
    if hasattr(base_model, "_sklearn_model_class"):
        sklearn_model_class = base_model._sklearn_model_class
    else:
        raise ValueError(f"Hyperparameter tuning not supported for {model_type}")

    # Filter param_grid to exclude "task" parameter
    filtered_param_grid = {k: v for k, v in param_grid.items() if k != "task"}

    grid_search = GridSearchCV(
        sklearn_model_class(),
        filtered_param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1,
    )

    # Perform grid search
    grid_search.fit(X, y)

    return grid_search.best_params_, grid_search.best_score_


def random_search_tuning(
    model_type: ModelType,
    X: pd.DataFrame,
    y: pd.Series,
    param_distributions: Dict[str, Any],
    n_iter: int = 100,
    cv: int = 5,
    scoring: Optional[str] = None,
    n_jobs: int = -1,
    random_state: Optional[int] = None,
) -> Tuple[Dict[str, Any], float]:
    """Perform hyperparameter tuning using random search.

    Args:
        model_type: Type of model to tune
        X: Feature matrix
        y: Target vector
        param_distributions: Dictionary mapping parameter names to distributions
        n_iter: Number of parameter settings sampled
        cv: Number of cross-validation folds
        scoring: Scoring metric (default: auto-detect based on task)
        n_jobs: Number of parallel jobs (-1 for all cores)
        random_state: Random state for reproducibility

    Returns:
        Tuple of (best_parameters, best_score)
    """
    from sklearn.model_selection import RandomizedSearchCV

    # Create base model
    base_model = MLModel._create_model_instance(
        model_type=model_type,
        hyperparameters={},
    )

    # Determine scoring if not provided
    if scoring is None:
        if len(np.unique(y)) <= 10:
            scoring = "accuracy"
        else:
            scoring = "r2"

    # Create random search
    if hasattr(base_model, "_sklearn_model_class"):
        sklearn_model_class = base_model._sklearn_model_class
    else:
        raise ValueError(f"Hyperparameter tuning not supported for {model_type}")

    # Filter param_distributions to exclude "task" parameter
    filtered_param_dist = {k: v for k, v in param_distributions.items() if k != "task"}

    random_search = RandomizedSearchCV(
        sklearn_model_class(),
        filtered_param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=1,
    )

    # Perform random search
    random_search.fit(X, y)

    return random_search.best_params_, random_search.best_score_

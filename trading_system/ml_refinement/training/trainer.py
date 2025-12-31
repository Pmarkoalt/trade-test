"""Model training orchestrator."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
from loguru import logger

from trading_system.ml_refinement.config import (
    MLConfig,
    ModelMetadata,
    ModelType,
    TrainingConfig,
)
from trading_system.ml_refinement.storage.feature_db import FeatureDatabase
from trading_system.ml_refinement.validation.walk_forward import (
    WalkForwardResults,
    WalkForwardValidator,
)
from trading_system.ml_refinement.validation.metrics import (
    calculate_classification_metrics,
    calculate_trading_metrics,
)

# Try to import base models, create minimal interface if not available
try:
    from trading_system.ml_refinement.models.base_model import BaseModel, SignalQualityModel
except ImportError:
    # Create minimal interface for models
    class BaseModel:
        """Base model interface."""

        def __init__(self, **kwargs):
            self.model_id = f"model_{uuid.uuid4().hex[:8]}"
            self.feature_names: List[str] = []
            self.feature_importance: Dict[str, float] = {}

        def fit(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
            """Train the model."""
            if feature_names:
                self.feature_names = feature_names
            return {"accuracy": 0.0}

        def predict(self, X):
            """Make predictions."""
            return np.zeros(len(X))

        def predict_proba(self, X):
            """Predict probabilities."""
            return np.ones((len(X), 2)) * 0.5

        def get_top_features(self, n=10):
            """Get top N features by importance."""
            return []

        def get_metadata(self):
            """Get model metadata."""
            return ModelMetadata(
                model_id=self.model_id,
                model_type=ModelType.SIGNAL_QUALITY,
                version="1.0",
                created_at=datetime.now().isoformat(),
                train_start_date="",
                train_end_date="",
                train_samples=0,
                validation_samples=0,
            )

        def save(self, path):
            """Save model."""
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            # Stub implementation

    class SignalQualityModel(BaseModel):
        """Signal quality model."""

        pass


@dataclass
class TrainingResult:
    """Result of a training run."""

    run_id: str = ""
    model_id: str = ""
    success: bool = False

    # Training info
    train_samples: int = 0
    val_samples: int = 0
    n_folds: int = 0
    total_time_seconds: float = 0.0

    # Metrics
    cv_metrics: Dict[str, float] = field(default_factory=dict)
    final_metrics: Dict[str, float] = field(default_factory=dict)

    # Feature info
    n_features: int = 0
    top_features: List[Tuple[str, float]] = field(default_factory=list)

    # Error info
    error_message: str = ""


class ModelTrainer:
    """
    End-to-end model training pipeline.

    Example:
        trainer = ModelTrainer(config, feature_db)

        # Train with walk-forward validation
        result = trainer.train(
            model_type=ModelType.SIGNAL_QUALITY,
            start_date="2022-01-01",
            end_date="2024-01-01",
        )

        if result.success:
            print(f"Trained model: {result.model_id}")
            print(f"CV AUC: {result.cv_metrics['auc']:.3f}")
    """

    MODEL_CLASSES: Dict[ModelType, Type[BaseModel]] = {
        ModelType.SIGNAL_QUALITY: SignalQualityModel,
    }

    def __init__(
        self,
        config: MLConfig,
        feature_db: FeatureDatabase,
        model_dir: str = "models/",
    ):
        """
        Initialize trainer.

        Args:
            config: ML configuration.
            feature_db: Feature database.
            model_dir: Directory to save models.
        """
        self.config = config
        self.feature_db = feature_db
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        model_type: ModelType,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        hyperparameters: Optional[Dict] = None,
    ) -> TrainingResult:
        """
        Train a model with walk-forward validation.

        Args:
            model_type: Type of model to train.
            start_date: Training data start date.
            end_date: Training data end date.
            hyperparameters: Optional model hyperparameters.

        Returns:
            TrainingResult with model info and metrics.
        """
        run_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        start_time = datetime.now()

        result = TrainingResult(run_id=run_id)

        try:
            # Get training data
            logger.info(f"Loading training data from {start_date} to {end_date}")
            X, y, feature_names = self.feature_db.get_training_data(
                start_date=start_date or "2020-01-01",
                end_date=end_date or datetime.now().strftime("%Y-%m-%d"),
            )

            if len(X) < self.config.training.min_training_samples:
                result.error_message = f"Insufficient samples: {len(X)}"
                logger.error(result.error_message)
                return result

            result.train_samples = len(X)
            result.n_features = len(feature_names)

            # Run walk-forward validation
            logger.info("Running walk-forward validation")
            cv_results = self._run_walk_forward_cv(X, y, feature_names, model_type, hyperparameters)

            result.n_folds = cv_results.n_folds
            result.val_samples = cv_results.total_val_samples
            result.cv_metrics = cv_results.avg_metrics

            # Train final model on all data
            logger.info("Training final model on all data")
            final_model, final_metrics = self._train_final_model(X, y, feature_names, model_type, hyperparameters)

            result.final_metrics = final_metrics
            result.top_features = final_model.get_top_features(10)

            # Save model
            model_path = self.model_dir / f"{final_model.model_id}.pkl"
            final_model.save(str(model_path))

            # Register in database
            metadata = final_model.get_metadata()
            metadata.train_metrics = final_metrics
            metadata.validation_metrics = cv_results.avg_metrics
            self.feature_db.register_model(metadata)

            result.model_id = final_model.model_id
            result.success = True

            elapsed = (datetime.now() - start_time).total_seconds()
            result.total_time_seconds = elapsed

            logger.info(
                f"Training complete: model={result.model_id}, "
                f"cv_auc={result.cv_metrics.get('auc', 0):.3f}, "
                f"time={elapsed:.1f}s"
            )

        except Exception as e:
            logger.exception(f"Training failed: {e}")
            result.error_message = str(e)

        return result

    def _run_walk_forward_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        model_type: ModelType,
        hyperparameters: Optional[Dict],
    ) -> WalkForwardResults:
        """Run walk-forward cross-validation."""
        validator = WalkForwardValidator.from_config(self.config.training)

        results = WalkForwardResults()
        fold_metrics_list = []
        all_predictions = []

        for fold_idx, split in enumerate(validator.generate_splits(len(X))):
            # Split data
            X_train = X[split.train_start : split.train_end]
            y_train = y[split.train_start : split.train_end]
            X_val = X[split.val_start : split.val_end]
            y_val = y[split.val_start : split.val_end]

            # Create and train model
            model_class = self.MODEL_CLASSES[model_type]
            model = model_class(**(hyperparameters or {}))
            model.fit(X_train, y_train, X_val, y_val, feature_names)

            # Predict on validation
            y_pred = model.predict(X_val)
            y_proba_full = model.predict_proba(X_val)

            # Extract positive class probabilities (assuming binary classification)
            if y_proba_full.ndim > 1:
                y_proba = y_proba_full[:, 1] if y_proba_full.shape[1] > 1 else y_proba_full[:, 0]
            else:
                y_proba = y_proba_full

            # Calculate metrics
            y_val_binary = (y_val > 0).astype(int)
            metrics = calculate_classification_metrics(y_val_binary, y_pred, y_proba)
            metrics.update(calculate_trading_metrics(y_val_binary, y_proba))

            fold_metrics_list.append(metrics)
            results.fold_results.append(
                {
                    "fold": fold_idx,
                    "train_size": split.train_size,
                    "val_size": split.val_size,
                    "metrics": metrics,
                }
            )

            # Store predictions
            for i, (pred, actual) in enumerate(zip(y_proba, y_val)):
                all_predictions.append((split.val_start + i, float(pred), float(actual)))

            logger.debug(
                f"Fold {fold_idx}: train={split.train_size}, val={split.val_size}, " f"auc={metrics.get('auc', 0):.3f}"
            )

        # Aggregate metrics
        results.n_folds = len(fold_metrics_list)
        results.total_val_samples = sum(r["val_size"] for r in results.fold_results)
        results.all_predictions = all_predictions

        if fold_metrics_list:
            # Average metrics
            all_keys = fold_metrics_list[0].keys()
            for key in all_keys:
                values = [m.get(key, 0) for m in fold_metrics_list if key in m]
                if values:
                    results.avg_metrics[key] = float(np.mean(values))
                    results.std_metrics[key] = float(np.std(values))

        return results

    def _train_final_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        model_type: ModelType,
        hyperparameters: Optional[Dict],
    ) -> Tuple[BaseModel, Dict[str, float]]:
        """Train final model on all data."""
        # Use last portion as validation
        val_size = int(len(X) * 0.15)
        X_train, X_val = X[:-val_size], X[-val_size:]
        y_train, y_val = y[:-val_size], y[-val_size:]

        model_class = self.MODEL_CLASSES[model_type]
        model = model_class(**(hyperparameters or {}))
        train_metrics = model.fit(X_train, y_train, X_val, y_val, feature_names)

        # Evaluate on validation
        y_pred = model.predict(X_val)
        y_proba_full = model.predict_proba(X_val)

        # Extract positive class probabilities (assuming binary classification)
        if y_proba_full.ndim > 1:
            y_proba = y_proba_full[:, 1] if y_proba_full.shape[1] > 1 else y_proba_full[:, 0]
        else:
            y_proba = y_proba_full

        y_val_binary = (y_val > 0).astype(int)

        final_metrics = calculate_classification_metrics(y_val_binary, y_pred, y_proba)
        final_metrics["train_accuracy"] = train_metrics.get("accuracy", 0)

        return model, final_metrics

    def retrain_if_needed(
        self,
        model_type: ModelType,
        min_new_samples: int = 20,
    ) -> Optional[TrainingResult]:
        """
        Retrain model if enough new samples available.

        Args:
            model_type: Type of model.
            min_new_samples: Minimum new samples to trigger retrain.

        Returns:
            TrainingResult if retrained, None otherwise.
        """
        # Get active model
        active = self.feature_db.get_active_model(model_type.value)

        if active is None:
            # No active model, train from scratch
            logger.info(f"No active {model_type.value} model, training new")
            return self.train(model_type)

        # Count samples since last training
        new_samples = self.feature_db.count_samples(
            start_date=active.train_end_date,
            require_target=True,
        )

        if new_samples >= min_new_samples:
            logger.info(f"Found {new_samples} new samples since {active.train_end_date}, " f"retraining {model_type.value}")
            return self.train(model_type)
        else:
            logger.debug(f"Only {new_samples} new samples, need {min_new_samples} to retrain")
            return None

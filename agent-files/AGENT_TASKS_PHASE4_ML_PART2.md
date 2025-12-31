# Agent Tasks: Phase 4 - ML Refinement (Part 2: Training & Optimization)

**Phase Goal**: Training pipelines, validation, and integration for ML refinement
**Duration**: 1-2 weeks (Part 2)
**Prerequisites**: Phase 4 Part 1 complete (feature store, extractors, base models)

---

## Phase 4 Part 2 Overview

### What We're Building
1. **Walk-Forward Validation** - Time-series aware cross-validation
2. **Training Pipeline** - End-to-end model training orchestration
3. **Signal Integration** - Integrate ML predictions into signal scoring
4. **Retraining Jobs** - Automated model updates
5. **Parameter Optimization** - Optimize strategy parameters using ML

### Architecture Addition

```
trading_system/
├── ml_refinement/
│   ├── validation/
│   │   ├── walk_forward.py          # Walk-forward CV
│   │   └── metrics.py               # Evaluation metrics
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py               # Training orchestrator
│   │   └── hyperparameter_tuner.py  # Hyperparameter optimization
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── parameter_optimizer.py   # Strategy parameter optimization
│   │   └── weight_optimizer.py      # Signal weight optimization
│   └── integration/
│       ├── __init__.py
│       ├── signal_scorer.py         # ML-enhanced signal scoring
│       └── prediction_service.py    # Prediction service
│
├── scheduler/
│   └── jobs/
│       └── ml_retrain_job.py        # NEW: Retraining job
│
└── cli/
    └── commands/
        └── ml.py                    # NEW: ML CLI commands
```

---

## Task 4.2.1: Implement Walk-Forward Validation

**Context**:
Standard cross-validation doesn't work for time-series data due to look-ahead bias. Walk-forward validation trains on past data and validates on future data.

**Objective**:
Create walk-forward validation framework for time-series ML.

**Files to Create**:
```
trading_system/ml_refinement/validation/
├── __init__.py
├── walk_forward.py
└── metrics.py
```

**Requirements**:

1. Create `walk_forward.py`:
```python
"""Walk-forward validation for time-series ML."""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
from loguru import logger

from trading_system.ml_refinement.config import TrainingConfig


@dataclass
class WalkForwardSplit:
    """A single train/validation split."""

    # Split indices
    train_start: int
    train_end: int
    val_start: int
    val_end: int

    # Date boundaries
    train_start_date: str = ""
    train_end_date: str = ""
    val_start_date: str = ""
    val_end_date: str = ""

    # Sizes
    train_size: int = 0
    val_size: int = 0


@dataclass
class WalkForwardResults:
    """Results from walk-forward validation."""

    # Per-fold results
    fold_results: List[Dict] = field(default_factory=list)

    # Aggregated metrics
    avg_metrics: Dict[str, float] = field(default_factory=dict)
    std_metrics: Dict[str, float] = field(default_factory=dict)

    # Predictions
    all_predictions: List[Tuple[int, float, float]] = field(default_factory=list)
    # List of (sample_idx, predicted, actual)

    # Summary
    n_folds: int = 0
    total_train_samples: int = 0
    total_val_samples: int = 0


class WalkForwardValidator:
    """
    Walk-forward cross-validation for time-series data.

    Walk-forward validation works by:
    1. Training on historical data window
    2. Validating on subsequent data
    3. Rolling forward and repeating

    Example:
        validator = WalkForwardValidator(
            train_window=252,  # ~1 year
            val_window=63,     # ~3 months
            step_size=21,      # ~1 month
        )

        for split in validator.generate_splits(n_samples=500, dates=date_list):
            # Train on split.train_start:split.train_end
            # Validate on split.val_start:split.val_end
            pass
    """

    def __init__(
        self,
        train_window: int = 252,
        val_window: int = 63,
        step_size: int = 21,
        min_train_samples: int = 100,
        min_val_samples: int = 20,
    ):
        """
        Initialize validator.

        Args:
            train_window: Number of samples in training window.
            val_window: Number of samples in validation window.
            step_size: How far to step forward each fold.
            min_train_samples: Minimum training samples required.
            min_val_samples: Minimum validation samples required.
        """
        self.train_window = train_window
        self.val_window = val_window
        self.step_size = step_size
        self.min_train_samples = min_train_samples
        self.min_val_samples = min_val_samples

    def generate_splits(
        self,
        n_samples: int,
        dates: Optional[List[str]] = None,
    ) -> Generator[WalkForwardSplit, None, None]:
        """
        Generate train/validation splits.

        Args:
            n_samples: Total number of samples.
            dates: Optional list of dates for each sample.

        Yields:
            WalkForwardSplit objects.
        """
        if n_samples < self.min_train_samples + self.min_val_samples:
            logger.warning(f"Insufficient samples: {n_samples}")
            return

        # Start after we have enough training data
        train_start = 0
        train_end = self.train_window

        fold_idx = 0
        while train_end + self.val_window <= n_samples:
            val_start = train_end
            val_end = min(val_start + self.val_window, n_samples)

            # Check minimum sizes
            actual_train_size = train_end - train_start
            actual_val_size = val_end - val_start

            if actual_train_size >= self.min_train_samples and \
               actual_val_size >= self.min_val_samples:

                split = WalkForwardSplit(
                    train_start=train_start,
                    train_end=train_end,
                    val_start=val_start,
                    val_end=val_end,
                    train_size=actual_train_size,
                    val_size=actual_val_size,
                )

                # Add dates if available
                if dates:
                    split.train_start_date = dates[train_start]
                    split.train_end_date = dates[train_end - 1]
                    split.val_start_date = dates[val_start]
                    split.val_end_date = dates[val_end - 1]

                yield split
                fold_idx += 1

            # Step forward
            train_start += self.step_size
            train_end = train_start + self.train_window

    def count_folds(self, n_samples: int) -> int:
        """Count the number of folds for given sample size."""
        count = 0
        for _ in self.generate_splits(n_samples):
            count += 1
        return count

    @classmethod
    def from_config(cls, config: TrainingConfig) -> "WalkForwardValidator":
        """Create validator from config."""
        return cls(
            train_window=config.train_window_days,
            val_window=config.validation_window_days,
            step_size=config.step_size_days,
            min_train_samples=config.min_training_samples,
            min_val_samples=config.min_validation_samples,
        )


class ExpandingWindowValidator:
    """
    Expanding window validation.

    Unlike walk-forward which uses a fixed training window,
    expanding window uses all available historical data.
    """

    def __init__(
        self,
        initial_train_size: int = 252,
        val_window: int = 63,
        step_size: int = 21,
        min_val_samples: int = 20,
    ):
        """
        Initialize validator.

        Args:
            initial_train_size: Initial training window size.
            val_window: Validation window size.
            step_size: Step size between folds.
            min_val_samples: Minimum validation samples.
        """
        self.initial_train_size = initial_train_size
        self.val_window = val_window
        self.step_size = step_size
        self.min_val_samples = min_val_samples

    def generate_splits(
        self,
        n_samples: int,
        dates: Optional[List[str]] = None,
    ) -> Generator[WalkForwardSplit, None, None]:
        """Generate expanding window splits."""
        train_end = self.initial_train_size

        while train_end + self.val_window <= n_samples:
            val_start = train_end
            val_end = min(val_start + self.val_window, n_samples)

            if val_end - val_start >= self.min_val_samples:
                split = WalkForwardSplit(
                    train_start=0,  # Always start from beginning
                    train_end=train_end,
                    val_start=val_start,
                    val_end=val_end,
                    train_size=train_end,
                    val_size=val_end - val_start,
                )

                if dates:
                    split.train_start_date = dates[0]
                    split.train_end_date = dates[train_end - 1]
                    split.val_start_date = dates[val_start]
                    split.val_end_date = dates[val_end - 1]

                yield split

            train_end += self.step_size


class PurgedKFold:
    """
    K-Fold with purging and embargo for overlapping labels.

    Used when labels span multiple time periods to prevent leakage.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_window: int = 5,
        embargo_pct: float = 0.01,
    ):
        """
        Initialize purged k-fold.

        Args:
            n_splits: Number of folds.
            purge_window: Samples to purge around test set.
            embargo_pct: Percentage of training data to embargo after test.
        """
        self.n_splits = n_splits
        self.purge_window = purge_window
        self.embargo_pct = embargo_pct

    def split(
        self,
        n_samples: int,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices with purging.

        Yields:
            Tuple of (train_indices, test_indices).
        """
        fold_size = n_samples // self.n_splits
        embargo_size = int(n_samples * self.embargo_pct)

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples

            # Purge before test
            train_end_before = max(0, test_start - self.purge_window)

            # Embargo after test
            train_start_after = min(n_samples, test_end + embargo_size)

            # Build train indices
            train_before = np.arange(0, train_end_before)
            train_after = np.arange(train_start_after, n_samples)
            train_idx = np.concatenate([train_before, train_after])

            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx
```

2. Create `metrics.py`:
```python
"""Evaluation metrics for ML models."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calculate classification metrics.

    Args:
        y_true: True labels (0/1).
        y_pred: Predicted labels (0/1).
        y_proba: Predicted probabilities (optional).

    Returns:
        Dictionary of metrics.
    """
    metrics = {}

    # Basic metrics
    n = len(y_true)
    if n == 0:
        return {"error": "No samples"}

    # Confusion matrix elements
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Accuracy
    metrics["accuracy"] = (tp + tn) / n

    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1"] = f1

    # Class balance
    metrics["positive_rate"] = np.mean(y_true)
    metrics["pred_positive_rate"] = np.mean(y_pred)

    # AUC-ROC if probabilities available
    if y_proba is not None:
        metrics["auc"] = calculate_auc(y_true, y_proba)
        metrics["log_loss"] = calculate_log_loss(y_true, y_proba)

        # Brier score
        metrics["brier"] = np.mean((y_proba - y_true) ** 2)

    return metrics


def calculate_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Calculate AUC-ROC.

    Simple implementation without sklearn dependency.
    """
    # Sort by predicted probability
    desc_score_indices = np.argsort(y_proba)[::-1]
    y_true_sorted = y_true[desc_score_indices]

    # Calculate ROC curve points
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)

    total_pos = np.sum(y_true)
    total_neg = len(y_true) - total_pos

    if total_pos == 0 or total_neg == 0:
        return 0.5

    tpr = tps / total_pos
    fpr = fps / total_neg

    # AUC via trapezoidal rule
    auc = np.trapz(tpr, fpr)
    return auc


def calculate_log_loss(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """Calculate log loss (cross-entropy)."""
    # Clip probabilities
    y_proba = np.clip(y_proba, eps, 1 - eps)

    # Log loss
    loss = -np.mean(
        y_true * np.log(y_proba) + (1 - y_true) * np.log(1 - y_proba)
    )
    return loss


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Dictionary of metrics.
    """
    metrics = {}

    if len(y_true) == 0:
        return {"error": "No samples"}

    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)
    metrics["mse"] = mse
    metrics["rmse"] = np.sqrt(mse)

    # Mean Absolute Error
    metrics["mae"] = np.mean(np.abs(y_true - y_pred))

    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    metrics["r2"] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Correlation
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        metrics["correlation"] = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        metrics["correlation"] = 0

    return metrics


def calculate_trading_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calculate trading-specific metrics.

    These metrics evaluate how useful predictions are for trading.
    """
    metrics = {}

    # Filter to high-conviction predictions
    high_conf_mask = y_proba >= threshold

    if np.sum(high_conf_mask) == 0:
        return {"high_conf_count": 0}

    # High confidence accuracy
    y_pred_high = (y_proba[high_conf_mask] >= 0.5).astype(int)
    y_true_high = y_true[high_conf_mask]

    metrics["high_conf_count"] = int(np.sum(high_conf_mask))
    metrics["high_conf_accuracy"] = np.mean(y_pred_high == y_true_high)

    # Expected value calculation
    # Assuming we trade when p > threshold
    # EV = p * avg_win - (1-p) * avg_loss
    # Simplified: use 2:1 risk/reward
    avg_win = 2.0
    avg_loss = 1.0

    ev_per_trade = y_proba[high_conf_mask] * avg_win - (1 - y_proba[high_conf_mask]) * avg_loss
    metrics["expected_value"] = float(np.mean(ev_per_trade))

    # Calibration
    metrics["calibration_error"] = calculate_calibration_error(y_true, y_proba)

    return metrics


def calculate_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Calculate Expected Calibration Error.

    Lower is better - means probabilities are well calibrated.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_proba[mask])
            bin_size = np.sum(mask) / len(y_true)
            ece += bin_size * abs(bin_acc - bin_conf)

    return ece


def calculate_feature_importance_stability(
    importance_history: List[Dict[str, float]],
) -> Dict[str, float]:
    """
    Calculate stability of feature importances across folds.

    High stability means features are consistently important.
    """
    if not importance_history:
        return {}

    # Get all feature names
    all_features = set()
    for imp_dict in importance_history:
        all_features.update(imp_dict.keys())

    stability = {}
    for feature in all_features:
        values = [d.get(feature, 0) for d in importance_history]
        mean_imp = np.mean(values)
        std_imp = np.std(values)

        # Coefficient of variation (lower = more stable)
        cv = std_imp / mean_imp if mean_imp > 0 else float('inf')

        stability[feature] = {
            "mean": mean_imp,
            "std": std_imp,
            "cv": cv,
        }

    return stability
```

3. Create `validation/__init__.py`:
```python
"""Validation module for ML."""

from trading_system.ml_refinement.validation.walk_forward import (
    ExpandingWindowValidator,
    PurgedKFold,
    WalkForwardResults,
    WalkForwardSplit,
    WalkForwardValidator,
)
from trading_system.ml_refinement.validation.metrics import (
    calculate_auc,
    calculate_calibration_error,
    calculate_classification_metrics,
    calculate_regression_metrics,
    calculate_trading_metrics,
)

__all__ = [
    "WalkForwardValidator",
    "WalkForwardSplit",
    "WalkForwardResults",
    "ExpandingWindowValidator",
    "PurgedKFold",
    "calculate_classification_metrics",
    "calculate_regression_metrics",
    "calculate_trading_metrics",
    "calculate_auc",
    "calculate_calibration_error",
]
```

**Acceptance Criteria**:
- [ ] Walk-forward generates correct splits
- [ ] No look-ahead bias in splits
- [ ] Expanding window works correctly
- [ ] Purged K-Fold handles overlapping labels
- [ ] Metrics calculate correctly
- [ ] Trading metrics provide useful info

**Tests to Write**:
```python
def test_walk_forward_splits():
    """Test walk-forward split generation."""
    validator = WalkForwardValidator(
        train_window=100,
        val_window=20,
        step_size=20,
    )

    splits = list(validator.generate_splits(200))

    # Should have multiple folds
    assert len(splits) >= 3

    # No overlap between train and val
    for split in splits:
        assert split.train_end <= split.val_start

    # Correct sizes
    for split in splits:
        assert split.train_size == 100
        assert split.val_size == 20

def test_classification_metrics():
    """Test classification metric calculation."""
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])
    y_proba = np.array([0.8, 0.4, 0.2, 0.6, 0.9])

    metrics = calculate_classification_metrics(y_true, y_pred, y_proba)

    assert "accuracy" in metrics
    assert "auc" in metrics
    assert 0 <= metrics["auc"] <= 1
```

---

## Task 4.2.2: Implement Training Pipeline

**Context**:
The training pipeline orchestrates end-to-end model training with proper validation.

**Objective**:
Create a complete training pipeline with logging and checkpointing.

**Files to Create**:
```
trading_system/ml_refinement/training/
├── __init__.py
├── trainer.py
└── hyperparameter_tuner.py
```

**Requirements**:

1. Create `trainer.py`:
```python
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
from trading_system.ml_refinement.features.pipeline import FeaturePipeline
from trading_system.ml_refinement.models.base_model import BaseModel, SignalQualityModel
from trading_system.ml_refinement.storage.feature_db import FeatureDatabase
from trading_system.ml_refinement.validation.walk_forward import (
    WalkForwardResults,
    WalkForwardValidator,
)
from trading_system.ml_refinement.validation.metrics import (
    calculate_classification_metrics,
    calculate_trading_metrics,
)


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
            cv_results = self._run_walk_forward_cv(
                X, y, feature_names, model_type, hyperparameters
            )

            result.n_folds = cv_results.n_folds
            result.val_samples = cv_results.total_val_samples
            result.cv_metrics = cv_results.avg_metrics

            # Train final model on all data
            logger.info("Training final model on all data")
            final_model, final_metrics = self._train_final_model(
                X, y, feature_names, model_type, hyperparameters
            )

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
            X_train = X[split.train_start:split.train_end]
            y_train = y[split.train_start:split.train_end]
            X_val = X[split.val_start:split.val_end]
            y_val = y[split.val_start:split.val_end]

            # Create and train model
            model_class = self.MODEL_CLASSES[model_type]
            model = model_class(**(hyperparameters or {}))
            model.fit(X_train, y_train, X_val, y_val, feature_names)

            # Predict on validation
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)

            # Calculate metrics
            y_val_binary = (y_val > 0).astype(int)
            metrics = calculate_classification_metrics(y_val_binary, y_pred, y_proba)
            metrics.update(calculate_trading_metrics(y_val_binary, y_proba))

            fold_metrics_list.append(metrics)
            results.fold_results.append({
                "fold": fold_idx,
                "train_size": split.train_size,
                "val_size": split.val_size,
                "metrics": metrics,
            })

            # Store predictions
            for i, (pred, actual) in enumerate(zip(y_proba, y_val)):
                all_predictions.append((split.val_start + i, pred, actual))

            logger.debug(
                f"Fold {fold_idx}: train={split.train_size}, val={split.val_size}, "
                f"auc={metrics.get('auc', 0):.3f}"
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
        y_proba = model.predict_proba(X_val)
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
            logger.info(
                f"Found {new_samples} new samples since {active.train_end_date}, "
                f"retraining {model_type.value}"
            )
            return self.train(model_type)
        else:
            logger.debug(
                f"Only {new_samples} new samples, need {min_new_samples} to retrain"
            )
            return None
```

2. Create `hyperparameter_tuner.py`:
```python
"""Hyperparameter tuning for ML models."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from trading_system.ml_refinement.models.base_model import BaseModel, SignalQualityModel
from trading_system.ml_refinement.validation.walk_forward import WalkForwardValidator
from trading_system.ml_refinement.validation.metrics import calculate_classification_metrics


@dataclass
class HyperparameterSearchResult:
    """Result of hyperparameter search."""

    best_params: Dict[str, Any] = field(default_factory=dict)
    best_score: float = 0.0
    best_fold_scores: List[float] = field(default_factory=list)

    # All results
    all_results: List[Dict] = field(default_factory=list)

    # Search info
    n_trials: int = 0
    total_time_seconds: float = 0.0


class HyperparameterTuner:
    """
    Hyperparameter tuning via grid or random search.

    Example:
        tuner = HyperparameterTuner()

        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
        }

        result = tuner.grid_search(
            X, y, feature_names,
            param_grid=param_grid,
            scoring="auc",
        )

        print(f"Best params: {result.best_params}")
    """

    def __init__(
        self,
        model_class: type = SignalQualityModel,
        cv: Optional[WalkForwardValidator] = None,
    ):
        """
        Initialize tuner.

        Args:
            model_class: Model class to tune.
            cv: Cross-validator (default: WalkForwardValidator).
        """
        self.model_class = model_class
        self.cv = cv or WalkForwardValidator()

    def grid_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        param_grid: Dict[str, List[Any]],
        scoring: str = "auc",
    ) -> HyperparameterSearchResult:
        """
        Grid search over parameter combinations.

        Args:
            X: Features.
            y: Targets.
            feature_names: Feature names.
            param_grid: Dict of param_name -> list of values.
            scoring: Metric to optimize.

        Returns:
            HyperparameterSearchResult.
        """
        import itertools
        from datetime import datetime

        start_time = datetime.now()
        result = HyperparameterSearchResult()

        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        logger.info(f"Grid search: {len(combinations)} combinations")

        best_score = -float('inf')
        best_params = {}

        for combo in combinations:
            params = dict(zip(param_names, combo))

            # Evaluate this combination
            scores = self._evaluate_params(X, y, feature_names, params, scoring)
            avg_score = np.mean(scores)

            result.all_results.append({
                "params": params,
                "scores": scores,
                "mean_score": avg_score,
                "std_score": np.std(scores),
            })

            if avg_score > best_score:
                best_score = avg_score
                best_params = params
                result.best_fold_scores = scores

            logger.debug(f"Params {params}: {scoring}={avg_score:.4f}")

        result.best_params = best_params
        result.best_score = best_score
        result.n_trials = len(combinations)
        result.total_time_seconds = (datetime.now() - start_time).total_seconds()

        logger.info(f"Best params: {best_params}, {scoring}={best_score:.4f}")

        return result

    def random_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        param_distributions: Dict[str, Any],
        n_trials: int = 20,
        scoring: str = "auc",
        random_state: int = 42,
    ) -> HyperparameterSearchResult:
        """
        Random search over parameter distributions.

        Args:
            X: Features.
            y: Targets.
            feature_names: Feature names.
            param_distributions: Dict of param_name -> distribution or list.
            n_trials: Number of random trials.
            scoring: Metric to optimize.
            random_state: Random seed.

        Returns:
            HyperparameterSearchResult.
        """
        from datetime import datetime

        np.random.seed(random_state)
        start_time = datetime.now()
        result = HyperparameterSearchResult()

        best_score = -float('inf')
        best_params = {}

        for trial in range(n_trials):
            # Sample random parameters
            params = {}
            for name, dist in param_distributions.items():
                if isinstance(dist, list):
                    params[name] = np.random.choice(dist)
                elif isinstance(dist, tuple) and len(dist) == 2:
                    # Uniform distribution
                    low, high = dist
                    if isinstance(low, int) and isinstance(high, int):
                        params[name] = np.random.randint(low, high + 1)
                    else:
                        params[name] = np.random.uniform(low, high)
                else:
                    params[name] = dist

            # Evaluate
            scores = self._evaluate_params(X, y, feature_names, params, scoring)
            avg_score = np.mean(scores)

            result.all_results.append({
                "trial": trial,
                "params": params,
                "scores": scores,
                "mean_score": avg_score,
            })

            if avg_score > best_score:
                best_score = avg_score
                best_params = params
                result.best_fold_scores = scores

            logger.debug(f"Trial {trial}: {scoring}={avg_score:.4f}")

        result.best_params = best_params
        result.best_score = best_score
        result.n_trials = n_trials
        result.total_time_seconds = (datetime.now() - start_time).total_seconds()

        return result

    def _evaluate_params(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        params: Dict[str, Any],
        scoring: str,
    ) -> List[float]:
        """Evaluate parameters with cross-validation."""
        scores = []

        for split in self.cv.generate_splits(len(X)):
            X_train = X[split.train_start:split.train_end]
            y_train = y[split.train_start:split.train_end]
            X_val = X[split.val_start:split.val_end]
            y_val = y[split.val_start:split.val_end]

            # Train model
            model = self.model_class(**params)
            model.fit(X_train, y_train, feature_names=feature_names)

            # Evaluate
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)
            y_val_binary = (y_val > 0).astype(int)

            metrics = calculate_classification_metrics(y_val_binary, y_pred, y_proba)
            scores.append(metrics.get(scoring, 0))

        return scores
```

3. Create `training/__init__.py`:
```python
"""Training module for ML."""

from trading_system.ml_refinement.training.trainer import (
    ModelTrainer,
    TrainingResult,
)
from trading_system.ml_refinement.training.hyperparameter_tuner import (
    HyperparameterSearchResult,
    HyperparameterTuner,
)

__all__ = [
    "ModelTrainer",
    "TrainingResult",
    "HyperparameterTuner",
    "HyperparameterSearchResult",
]
```

**Acceptance Criteria**:
- [ ] Trainer runs complete pipeline
- [ ] Walk-forward CV integrated correctly
- [ ] Final model trained on all data
- [ ] Model saved and registered
- [ ] Hyperparameter tuning works
- [ ] Retrain logic checks for new samples

---

## Task 4.2.3: Implement Signal Scoring Integration

**Context**:
ML predictions need to be integrated into the signal scoring system.

**Objective**:
Create a prediction service that enhances signal scores with ML.

**Files to Create**:
```
trading_system/ml_refinement/integration/
├── __init__.py
├── signal_scorer.py
└── prediction_service.py
```

**Requirements**:

1. Create `prediction_service.py`:
```python
"""Prediction service for ML inference."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from trading_system.ml_refinement.config import MLConfig, ModelType
from trading_system.ml_refinement.features.pipeline import FeaturePipeline
from trading_system.ml_refinement.models.model_registry import ModelRegistry
from trading_system.ml_refinement.storage.feature_db import FeatureDatabase


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
        model_registry: ModelRegistry,
    ):
        """
        Initialize prediction service.

        Args:
            config: ML configuration.
            feature_db: Feature database.
            model_registry: Model registry.
        """
        self.config = config
        self.feature_db = feature_db
        self.model_registry = model_registry
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
            quality_score = float(model.predict_proba(X)[0])

            # Log prediction
            self.feature_db.log_prediction(
                signal_id=signal_id,
                model_id=model.model_id,
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
```

2. Create `signal_scorer.py`:
```python
"""ML-enhanced signal scoring."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from loguru import logger

from trading_system.ml_refinement.config import MLConfig
from trading_system.ml_refinement.integration.prediction_service import PredictionService


@dataclass
class EnhancedSignalScore:
    """Signal score with ML enhancement."""

    signal_id: str
    symbol: str

    # Original scores
    technical_score: float = 0.0
    news_score: float = 0.0

    # ML scores
    ml_quality_score: float = 0.5

    # Combined
    combined_score: float = 0.0

    # Metadata
    ml_enabled: bool = False
    ml_confidence: str = ""  # "high", "medium", "low"


class MLSignalScorer:
    """
    Enhance signal scores with ML predictions.

    Example:
        scorer = MLSignalScorer(config, prediction_service)

        enhanced = scorer.score_signal(
            signal_id="sig-123",
            technical_score=7.5,
            news_score=6.0,
            ohlcv_data=ohlcv_df,
            signal_metadata=signal_dict,
        )

        print(f"Combined score: {enhanced.combined_score:.1f}")
    """

    def __init__(
        self,
        config: MLConfig,
        prediction_service: PredictionService,
    ):
        """
        Initialize scorer.

        Args:
            config: ML configuration.
            prediction_service: Prediction service.
        """
        self.config = config
        self.prediction_service = prediction_service

    def score_signal(
        self,
        signal_id: str,
        technical_score: float,
        news_score: Optional[float],
        ohlcv_data,
        signal_metadata: Dict,
        benchmark_data=None,
    ) -> EnhancedSignalScore:
        """
        Score a signal with ML enhancement.

        Args:
            signal_id: Signal identifier.
            technical_score: Technical analysis score (0-10).
            news_score: News sentiment score (0-10), optional.
            ohlcv_data: OHLCV DataFrame.
            signal_metadata: Signal metadata.
            benchmark_data: Optional benchmark data.

        Returns:
            EnhancedSignalScore with combined score.
        """
        result = EnhancedSignalScore(
            signal_id=signal_id,
            symbol=signal_metadata.get("symbol", ""),
            technical_score=technical_score,
            news_score=news_score or 0.0,
        )

        # Get ML prediction if enabled
        if self.config.enabled and self.config.use_ml_scores:
            ml_quality = self.prediction_service.predict_signal_quality(
                signal_id=signal_id,
                ohlcv_data=ohlcv_data,
                signal_metadata=signal_metadata,
                benchmark_data=benchmark_data,
            )
            result.ml_quality_score = ml_quality
            result.ml_enabled = True

            # Determine confidence level
            if ml_quality >= self.config.quality_threshold_high:
                result.ml_confidence = "high"
            elif ml_quality <= self.config.quality_threshold_low:
                result.ml_confidence = "low"
            else:
                result.ml_confidence = "medium"

        # Calculate combined score
        result.combined_score = self._calculate_combined_score(
            technical_score=technical_score,
            news_score=news_score,
            ml_quality=result.ml_quality_score if result.ml_enabled else None,
        )

        return result

    def score_signals_batch(
        self,
        signals: List[Dict],
        ohlcv_dict: Dict,
        benchmark_data=None,
    ) -> List[EnhancedSignalScore]:
        """Score multiple signals."""
        results = []

        for signal in signals:
            signal_id = signal.get("signal_id", signal.get("id", ""))
            symbol = signal.get("symbol", "")

            if symbol not in ohlcv_dict:
                continue

            enhanced = self.score_signal(
                signal_id=signal_id,
                technical_score=signal.get("technical_score", 5.0),
                news_score=signal.get("news_score"),
                ohlcv_data=ohlcv_dict[symbol],
                signal_metadata=signal,
                benchmark_data=benchmark_data,
            )
            results.append(enhanced)

        return results

    def _calculate_combined_score(
        self,
        technical_score: float,
        news_score: Optional[float],
        ml_quality: Optional[float],
    ) -> float:
        """
        Calculate combined score from components.

        Uses configurable weights to combine scores.
        """
        # Normalize to 0-10 scale
        normalized_technical = technical_score  # Already 0-10
        normalized_news = news_score if news_score else 5.0  # Default neutral

        # ML quality is 0-1, convert to 0-10
        normalized_ml = (ml_quality * 10) if ml_quality else 5.0

        # Get weights based on config and availability
        if ml_quality is not None and self.config.use_ml_scores:
            # Use ML-weighted combination
            tech_weight = 0.4
            news_weight = 0.2 if news_score else 0.0
            ml_weight = self.config.ml_score_weight  # Default 0.3

            # Redistribute news weight if not available
            if news_score is None:
                tech_weight += 0.1
                ml_weight += 0.1

            total_weight = tech_weight + news_weight + ml_weight

            combined = (
                normalized_technical * tech_weight +
                normalized_news * news_weight +
                normalized_ml * ml_weight
            ) / total_weight

        else:
            # Without ML
            if news_score is not None:
                combined = normalized_technical * 0.6 + normalized_news * 0.4
            else:
                combined = normalized_technical

        return combined

    def filter_by_ml_quality(
        self,
        scored_signals: List[EnhancedSignalScore],
        min_quality: Optional[float] = None,
    ) -> List[EnhancedSignalScore]:
        """
        Filter signals by ML quality score.

        Args:
            scored_signals: List of scored signals.
            min_quality: Minimum ML quality (default: use config).

        Returns:
            Filtered list of signals.
        """
        min_quality = min_quality or self.config.quality_threshold_low

        filtered = []
        for signal in scored_signals:
            if not signal.ml_enabled:
                # Include signals without ML (no filtering)
                filtered.append(signal)
            elif signal.ml_quality_score >= min_quality:
                filtered.append(signal)
            else:
                logger.debug(
                    f"Filtered out {signal.symbol}: ML quality "
                    f"{signal.ml_quality_score:.2f} < {min_quality}"
                )

        return filtered
```

3. Create `integration/__init__.py`:
```python
"""ML integration module."""

from trading_system.ml_refinement.integration.prediction_service import (
    PredictionService,
)
from trading_system.ml_refinement.integration.signal_scorer import (
    EnhancedSignalScore,
    MLSignalScorer,
)

__all__ = [
    "PredictionService",
    "EnhancedSignalScore",
    "MLSignalScorer",
]
```

**Acceptance Criteria**:
- [ ] Prediction service makes predictions
- [ ] Features stored during prediction
- [ ] Predictions logged for tracking
- [ ] Signal scorer combines ML with other scores
- [ ] Filtering by ML quality works
- [ ] Batch prediction efficient

---

## Task 4.2.4: Create Retraining Job

**Context**:
Models need periodic retraining as new data accumulates.

**Objective**:
Create a scheduled job for automated model retraining.

**Files to Create**:
```
trading_system/scheduler/jobs/ml_retrain_job.py
```

**Requirements**:

```python
"""ML model retraining job."""

from datetime import datetime, timedelta
from typing import Optional

from loguru import logger

from trading_system.ml_refinement.config import MLConfig, ModelType
from trading_system.ml_refinement.integration.prediction_service import PredictionService
from trading_system.ml_refinement.models.model_registry import ModelRegistry
from trading_system.ml_refinement.storage.feature_db import FeatureDatabase
from trading_system.ml_refinement.training.trainer import ModelTrainer


class MLRetrainJob:
    """
    Job for periodic ML model retraining.

    This job:
    1. Checks if enough new samples are available
    2. Retrains models with new data
    3. Evaluates new model vs current
    4. Activates new model if better

    Example:
        job = MLRetrainJob(config, feature_db, model_registry)

        # Run manually
        result = job.run()

        # Or schedule with APScheduler
        scheduler.add_job(
            job.run,
            'interval',
            days=7,  # Weekly
        )
    """

    def __init__(
        self,
        config: MLConfig,
        feature_db: FeatureDatabase,
        model_registry: ModelRegistry,
        model_dir: str = "models/",
    ):
        """
        Initialize job.

        Args:
            config: ML configuration.
            feature_db: Feature database.
            model_registry: Model registry.
            model_dir: Directory for model storage.
        """
        self.config = config
        self.feature_db = feature_db
        self.model_registry = model_registry
        self.trainer = ModelTrainer(config, feature_db, model_dir)

    def run(
        self,
        force: bool = False,
        model_types: Optional[list] = None,
    ) -> dict:
        """
        Run the retraining job.

        Args:
            force: Force retraining regardless of sample count.
            model_types: Specific model types to retrain (default: all).

        Returns:
            Dict with job results.
        """
        logger.info("Starting ML retrain job")
        start_time = datetime.now()

        results = {
            "started_at": start_time.isoformat(),
            "models_retrained": [],
            "models_skipped": [],
            "errors": [],
        }

        model_types = model_types or [ModelType.SIGNAL_QUALITY]

        for model_type in model_types:
            try:
                result = self._retrain_model(model_type, force)
                if result:
                    results["models_retrained"].append({
                        "model_type": model_type.value,
                        "model_id": result.model_id,
                        "cv_auc": result.cv_metrics.get("auc", 0),
                        "samples": result.train_samples,
                    })
                else:
                    results["models_skipped"].append(model_type.value)

            except Exception as e:
                logger.exception(f"Error retraining {model_type.value}")
                results["errors"].append({
                    "model_type": model_type.value,
                    "error": str(e),
                })

        elapsed = (datetime.now() - start_time).total_seconds()
        results["completed_at"] = datetime.now().isoformat()
        results["elapsed_seconds"] = elapsed

        logger.info(
            f"Retrain job complete: "
            f"{len(results['models_retrained'])} retrained, "
            f"{len(results['models_skipped'])} skipped, "
            f"{len(results['errors'])} errors"
        )

        return results

    def _retrain_model(
        self,
        model_type: ModelType,
        force: bool,
    ) -> Optional[object]:
        """Retrain a single model type."""
        # Check if retraining needed
        if not force:
            current_model = self.model_registry.get_active(model_type)

            if current_model:
                # Count new samples since last training
                new_samples = self.feature_db.count_samples(
                    start_date=current_model._train_end_date,
                    require_target=True,
                )

                if new_samples < self.config.min_new_samples_for_retrain:
                    logger.info(
                        f"Skipping {model_type.value}: only {new_samples} new samples "
                        f"(need {self.config.min_new_samples_for_retrain})"
                    )
                    return None

        # Calculate training date range
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (
            datetime.now() - timedelta(days=self.config.training.train_window_days * 2)
        ).strftime("%Y-%m-%d")

        # Train new model
        result = self.trainer.train(
            model_type=model_type,
            start_date=start_date,
            end_date=end_date,
        )

        if not result.success:
            raise RuntimeError(f"Training failed: {result.error_message}")

        # Compare with current model
        current_model = self.model_registry.get_active(model_type)
        should_activate = True

        if current_model:
            current_auc = current_model._validation_metrics.get("auc", 0)
            new_auc = result.cv_metrics.get("auc", 0)

            # Only activate if better (with small tolerance)
            if new_auc < current_auc - 0.02:
                logger.info(
                    f"New model ({new_auc:.3f}) not better than current ({current_auc:.3f}), "
                    f"keeping current"
                )
                should_activate = False

        if should_activate:
            self.model_registry.activate(result.model_id)
            logger.info(f"Activated new model: {result.model_id}")

        return result

    def check_retrain_needed(self, model_type: ModelType) -> dict:
        """
        Check if retraining is needed for a model type.

        Returns dict with status info.
        """
        current = self.model_registry.get_active(model_type)

        if current is None:
            return {
                "needed": True,
                "reason": "No active model",
                "new_samples": self.feature_db.count_samples(require_target=True),
            }

        new_samples = self.feature_db.count_samples(
            start_date=current._train_end_date,
            require_target=True,
        )

        needed = new_samples >= self.config.min_new_samples_for_retrain

        return {
            "needed": needed,
            "reason": f"{new_samples} new samples" if needed else "Insufficient new samples",
            "new_samples": new_samples,
            "threshold": self.config.min_new_samples_for_retrain,
            "current_model": current.model_id,
            "current_auc": current._validation_metrics.get("auc", 0),
        }
```

**Acceptance Criteria**:
- [ ] Job checks for new samples before retraining
- [ ] Force option bypasses sample check
- [ ] New model compared to current before activation
- [ ] Only activates if performance improves
- [ ] Logs provide visibility into decisions

---

## Task 4.2.5: Create ML CLI Commands

**Context**:
Users need command-line access to ML functionality.

**Objective**:
Add CLI commands for training, prediction, and model management.

**Files to Create**:
```
trading_system/cli/commands/ml.py
```

**Requirements**:

```python
"""CLI commands for ML functionality."""

import argparse
from datetime import datetime, timedelta
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from trading_system.ml_refinement.config import MLConfig, ModelType
from trading_system.ml_refinement.storage.feature_db import FeatureDatabase
from trading_system.ml_refinement.models.model_registry import ModelRegistry
from trading_system.ml_refinement.training.trainer import ModelTrainer
from trading_system.ml_refinement.scheduler.jobs.ml_retrain_job import MLRetrainJob


console = Console()


def setup_parser(subparsers):
    """Set up ML CLI commands."""
    ml_parser = subparsers.add_parser(
        "ml",
        help="ML model management",
    )

    ml_subparsers = ml_parser.add_subparsers(dest="ml_command")

    # Train command
    train_parser = ml_subparsers.add_parser(
        "train",
        help="Train a new model",
    )
    train_parser.add_argument(
        "--model-type",
        choices=["signal_quality"],
        default="signal_quality",
        help="Type of model to train",
    )
    train_parser.add_argument(
        "--start-date",
        help="Training data start date (YYYY-MM-DD)",
    )
    train_parser.add_argument(
        "--end-date",
        help="Training data end date (YYYY-MM-DD)",
    )
    train_parser.add_argument(
        "--feature-db",
        default="features.db",
        help="Path to feature database",
    )
    train_parser.add_argument(
        "--model-dir",
        default="models/",
        help="Directory for model storage",
    )

    # Status command
    status_parser = ml_subparsers.add_parser(
        "status",
        help="Show ML system status",
    )
    status_parser.add_argument(
        "--feature-db",
        default="features.db",
        help="Path to feature database",
    )
    status_parser.add_argument(
        "--model-dir",
        default="models/",
        help="Directory for model storage",
    )

    # Models command
    models_parser = ml_subparsers.add_parser(
        "models",
        help="List trained models",
    )
    models_parser.add_argument(
        "--model-type",
        choices=["signal_quality", "all"],
        default="all",
        help="Filter by model type",
    )
    models_parser.add_argument(
        "--feature-db",
        default="features.db",
        help="Path to feature database",
    )

    # Features command
    features_parser = ml_subparsers.add_parser(
        "features",
        help="Show feature statistics",
    )
    features_parser.add_argument(
        "--feature-db",
        default="features.db",
        help="Path to feature database",
    )

    # Retrain command
    retrain_parser = ml_subparsers.add_parser(
        "retrain",
        help="Run retraining job",
    )
    retrain_parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining regardless of sample count",
    )
    retrain_parser.add_argument(
        "--feature-db",
        default="features.db",
        help="Path to feature database",
    )
    retrain_parser.add_argument(
        "--model-dir",
        default="models/",
        help="Directory for model storage",
    )


def handle_command(args):
    """Handle ML commands."""
    if args.ml_command == "train":
        cmd_train(args)
    elif args.ml_command == "status":
        cmd_status(args)
    elif args.ml_command == "models":
        cmd_models(args)
    elif args.ml_command == "features":
        cmd_features(args)
    elif args.ml_command == "retrain":
        cmd_retrain(args)
    else:
        console.print("[yellow]Use --help to see available commands[/yellow]")


def cmd_train(args):
    """Train a new model."""
    config = MLConfig()
    feature_db = FeatureDatabase(args.feature_db)
    feature_db.initialize()

    trainer = ModelTrainer(config, feature_db, args.model_dir)

    model_type = ModelType(args.model_type)

    with console.status("Training model..."):
        result = trainer.train(
            model_type=model_type,
            start_date=args.start_date,
            end_date=args.end_date,
        )

    if result.success:
        console.print(Panel(
            f"[green]Model trained successfully![/green]\n\n"
            f"Model ID: {result.model_id}\n"
            f"Samples: {result.train_samples}\n"
            f"CV AUC: {result.cv_metrics.get('auc', 0):.3f}\n"
            f"Time: {result.total_time_seconds:.1f}s",
            title="Training Complete",
            box=box.ROUNDED,
        ))

        # Show top features
        if result.top_features:
            table = Table(title="Top Features", box=box.SIMPLE)
            table.add_column("Feature")
            table.add_column("Importance", justify="right")

            for name, importance in result.top_features[:10]:
                table.add_row(name, f"{importance:.4f}")

            console.print(table)

    else:
        console.print(f"[red]Training failed: {result.error_message}[/red]")

    feature_db.close()


def cmd_status(args):
    """Show ML system status."""
    config = MLConfig()
    feature_db = FeatureDatabase(args.feature_db)
    feature_db.initialize()
    model_registry = ModelRegistry(args.model_dir, feature_db)

    # Feature statistics
    total_features = feature_db.count_samples(require_target=False)
    labeled_features = feature_db.count_samples(require_target=True)

    console.print(Panel(
        f"Total feature vectors: {total_features}\n"
        f"With labels: {labeled_features}\n"
        f"Unlabeled: {total_features - labeled_features}",
        title="Feature Database",
        box=box.ROUNDED,
    ))

    # Model status
    table = Table(title="Active Models", box=box.ROUNDED)
    table.add_column("Model Type")
    table.add_column("Model ID")
    table.add_column("AUC")
    table.add_column("Deployed")

    for model_type in ModelType:
        model = model_registry.get_active(model_type)
        if model:
            table.add_row(
                model_type.value,
                model.model_id[:20] + "...",
                f"{model._validation_metrics.get('auc', 0):.3f}",
                model._train_end_date or "Unknown",
            )
        else:
            table.add_row(
                model_type.value,
                "[dim]None[/dim]",
                "-",
                "-",
            )

    console.print(table)

    # Retrain status
    job = MLRetrainJob(config, feature_db, model_registry, args.model_dir)
    for model_type in ModelType:
        status = job.check_retrain_needed(model_type)
        if status["needed"]:
            console.print(
                f"[yellow]Retrain recommended for {model_type.value}: "
                f"{status['reason']}[/yellow]"
            )

    feature_db.close()


def cmd_models(args):
    """List trained models."""
    feature_db = FeatureDatabase(args.feature_db)
    feature_db.initialize()

    model_types = [ModelType.SIGNAL_QUALITY] if args.model_type != "all" else list(ModelType)

    for model_type in model_types:
        history = feature_db.get_model_history(model_type.value, limit=10)

        if not history:
            console.print(f"[dim]No models for {model_type.value}[/dim]")
            continue

        table = Table(title=f"{model_type.value} Models", box=box.ROUNDED)
        table.add_column("Model ID")
        table.add_column("Version")
        table.add_column("AUC")
        table.add_column("Samples")
        table.add_column("Active")

        for model in history:
            active_mark = "[green]Yes[/green]" if model.is_active else "[dim]No[/dim]"
            table.add_row(
                model.model_id[:24],
                model.version,
                f"{model.validation_metrics.get('auc', 0):.3f}",
                str(model.train_samples),
                active_mark,
            )

        console.print(table)

    feature_db.close()


def cmd_features(args):
    """Show feature statistics."""
    feature_db = FeatureDatabase(args.feature_db)
    feature_db.initialize()

    # Get sample of features to analyze
    X, y, feature_names = feature_db.get_training_data(
        start_date="2020-01-01",
        end_date=datetime.now().strftime("%Y-%m-%d"),
    )

    if len(X) == 0:
        console.print("[yellow]No features in database[/yellow]")
        feature_db.close()
        return

    import numpy as np

    console.print(f"\nTotal samples: {len(X)}")
    console.print(f"Features: {len(feature_names)}")
    console.print(f"Target win rate: {(y > 0).mean():.1%}")

    # Feature statistics
    table = Table(title="Feature Statistics", box=box.SIMPLE)
    table.add_column("Feature")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")

    for i, name in enumerate(feature_names[:20]):  # Show first 20
        col = X[:, i]
        table.add_row(
            name[:25],
            f"{np.mean(col):.3f}",
            f"{np.std(col):.3f}",
            f"{np.min(col):.3f}",
            f"{np.max(col):.3f}",
        )

    console.print(table)

    if len(feature_names) > 20:
        console.print(f"[dim]... and {len(feature_names) - 20} more features[/dim]")

    feature_db.close()


def cmd_retrain(args):
    """Run retraining job."""
    config = MLConfig()
    feature_db = FeatureDatabase(args.feature_db)
    feature_db.initialize()
    model_registry = ModelRegistry(args.model_dir, feature_db)

    job = MLRetrainJob(config, feature_db, model_registry, args.model_dir)

    with console.status("Running retrain job..."):
        results = job.run(force=args.force)

    # Show results
    if results["models_retrained"]:
        table = Table(title="Models Retrained", box=box.ROUNDED)
        table.add_column("Model Type")
        table.add_column("Model ID")
        table.add_column("AUC")
        table.add_column("Samples")

        for model in results["models_retrained"]:
            table.add_row(
                model["model_type"],
                model["model_id"][:24],
                f"{model['cv_auc']:.3f}",
                str(model["samples"]),
            )

        console.print(table)

    if results["models_skipped"]:
        console.print(f"[dim]Skipped: {', '.join(results['models_skipped'])}[/dim]")

    if results["errors"]:
        for error in results["errors"]:
            console.print(f"[red]Error ({error['model_type']}): {error['error']}[/red]")

    console.print(f"\nCompleted in {results['elapsed_seconds']:.1f}s")

    feature_db.close()
```

**Acceptance Criteria**:
- [ ] `trading-system ml train` trains a model
- [ ] `trading-system ml status` shows system status
- [ ] `trading-system ml models` lists trained models
- [ ] `trading-system ml features` shows feature stats
- [ ] `trading-system ml retrain` runs retraining job
- [ ] Rich formatting displays correctly

---

## Task 4.2.6: Write Integration Tests

**Context**:
End-to-end tests ensure the ML pipeline works correctly.

**Objective**:
Write comprehensive tests for ML functionality.

**Files to Create**:
```
tests/test_ml_integration.py
```

**Requirements**:

```python
"""Integration tests for ML refinement."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from trading_system.ml_refinement.config import (
    FeatureConfig,
    FeatureSet,
    MLConfig,
    ModelType,
)
from trading_system.ml_refinement.storage.feature_db import FeatureDatabase
from trading_system.ml_refinement.features.pipeline import FeaturePipeline
from trading_system.ml_refinement.features.extractors.technical_features import (
    TrendFeatures,
    MomentumFeatures,
)
from trading_system.ml_refinement.models.base_model import SignalQualityModel
from trading_system.ml_refinement.models.model_registry import ModelRegistry
from trading_system.ml_refinement.training.trainer import ModelTrainer
from trading_system.ml_refinement.validation.walk_forward import WalkForwardValidator
from trading_system.ml_refinement.integration.prediction_service import PredictionService
from trading_system.ml_refinement.integration.signal_scorer import MLSignalScorer


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data."""
    dates = pd.date_range(start="2023-01-01", periods=300, freq="D")
    np.random.seed(42)

    # Generate trending price with noise
    trend = np.cumsum(np.random.randn(300) * 0.5) + 100
    noise = np.random.randn(300) * 2

    close = trend + noise
    high = close + np.abs(np.random.randn(300)) * 2
    low = close - np.abs(np.random.randn(300)) * 2
    open_ = close + np.random.randn(300) * 1

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(1000000, 10000000, 300),
    }, index=dates)


@pytest.fixture
def feature_db(tmp_path):
    """Create temporary feature database."""
    db = FeatureDatabase(str(tmp_path / "features.db"))
    db.initialize()
    yield db
    db.close()


@pytest.fixture
def populated_feature_db(feature_db):
    """Feature database with sample data."""
    from trading_system.ml_refinement.config import FeatureVector

    # Create sample feature vectors with targets
    np.random.seed(42)

    for i in range(200):
        fv = FeatureVector(
            signal_id=f"sig-{i:04d}",
            timestamp=(datetime.now() - timedelta(days=200-i)).isoformat(),
            features={
                "rsi": np.random.uniform(20, 80),
                "price_vs_ma20": np.random.uniform(-0.1, 0.1),
                "volatility": np.random.uniform(0.01, 0.05),
                "trend_strength": np.random.uniform(0, 1),
            },
            target=np.random.uniform(-2, 3),  # R-multiples
        )

        feature_db.store_feature_vector(
            fv,
            symbol="TEST",
            asset_class="equity",
            signal_type="breakout_20d",
        )

    return feature_db


class TestFeatureExtraction:
    """Test feature extraction."""

    def test_trend_features(self, sample_ohlcv):
        """Test trend feature extraction."""
        extractor = TrendFeatures()
        features = extractor.extract(sample_ohlcv)

        assert "price_vs_ma20" in features
        assert "trend_strength" in features
        assert all(isinstance(v, float) for v in features.values())

    def test_momentum_features(self, sample_ohlcv):
        """Test momentum feature extraction."""
        extractor = MomentumFeatures()
        features = extractor.extract(sample_ohlcv)

        assert "rsi_14" in features
        assert 0 <= features["rsi_14"] <= 1  # Normalized

    def test_feature_pipeline(self, sample_ohlcv):
        """Test complete feature pipeline."""
        config = FeatureConfig(feature_set=FeatureSet.STANDARD)
        pipeline = FeaturePipeline(config)

        signal_metadata = {
            "technical_score": 7.5,
            "conviction": "HIGH",
            "entry_price": 100,
            "target_price": 110,
            "stop_price": 95,
        }

        features = pipeline.extract_features(
            signal_id="test-123",
            ohlcv_data=sample_ohlcv,
            signal_metadata=signal_metadata,
        )

        assert len(features) > 10
        assert all(isinstance(v, float) for v in features.values())


class TestFeatureDatabase:
    """Test feature database operations."""

    def test_store_and_retrieve(self, feature_db):
        """Test feature storage roundtrip."""
        from trading_system.ml_refinement.config import FeatureVector

        fv = FeatureVector(
            signal_id="test-001",
            timestamp=datetime.now().isoformat(),
            features={"rsi": 65.0, "atr": 2.5},
            target=1.5,
        )

        feature_db.store_feature_vector(fv, symbol="AAPL")

        retrieved = feature_db.get_feature_vector("test-001")
        assert retrieved is not None
        assert retrieved.features["rsi"] == 65.0
        assert retrieved.target == 1.5

    def test_get_training_data(self, populated_feature_db):
        """Test training data retrieval."""
        X, y, names = populated_feature_db.get_training_data(
            start_date="2020-01-01",
            end_date=datetime.now().strftime("%Y-%m-%d"),
        )

        assert X.shape[0] == 200
        assert X.shape[1] == 4  # 4 features
        assert len(y) == 200
        assert len(names) == 4


class TestWalkForwardValidation:
    """Test walk-forward validation."""

    def test_split_generation(self):
        """Test split generation."""
        validator = WalkForwardValidator(
            train_window=100,
            val_window=20,
            step_size=20,
        )

        splits = list(validator.generate_splits(200))

        assert len(splits) >= 2

        # Verify no overlap
        for split in splits:
            assert split.train_end <= split.val_start

    def test_no_leakage(self):
        """Test that validation never sees training data."""
        validator = WalkForwardValidator(
            train_window=50,
            val_window=10,
            step_size=10,
        )

        for split in validator.generate_splits(100):
            train_indices = set(range(split.train_start, split.train_end))
            val_indices = set(range(split.val_start, split.val_end))

            # No overlap
            assert len(train_indices & val_indices) == 0

            # Validation always after training
            assert min(val_indices) > max(train_indices)


class TestModelTraining:
    """Test model training."""

    def test_signal_quality_model(self):
        """Test SignalQualityModel training."""
        np.random.seed(42)

        X = np.random.randn(200, 10)
        y = np.random.randn(200)  # Will be converted to binary

        model = SignalQualityModel()
        metrics = model.fit(X, y)

        assert model.is_fitted
        assert "accuracy" in metrics
        assert "auc" in metrics
        assert 0.4 <= metrics["auc"] <= 0.7  # Reasonable for random data

    def test_model_predict(self):
        """Test model prediction."""
        np.random.seed(42)

        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100)

        model = SignalQualityModel()
        model.fit(X_train, y_train)

        X_test = np.random.randn(10, 5)
        proba = model.predict_proba(X_test)

        assert len(proba) == 10
        assert all(0 <= p <= 1 for p in proba)

    def test_model_save_load(self, tmp_path):
        """Test model persistence."""
        np.random.seed(42)

        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        model = SignalQualityModel()
        model.fit(X, y)

        # Save
        path = str(tmp_path / "model.pkl")
        model.save(path)

        # Load
        loaded = SignalQualityModel()
        loaded.load(path)

        assert loaded.is_fitted

        # Predictions should match
        X_test = np.random.randn(5, 5)
        np.testing.assert_array_almost_equal(
            model.predict_proba(X_test),
            loaded.predict_proba(X_test),
        )

    def test_trainer_pipeline(self, populated_feature_db, tmp_path):
        """Test complete training pipeline."""
        config = MLConfig()
        trainer = ModelTrainer(
            config,
            populated_feature_db,
            str(tmp_path / "models"),
        )

        result = trainer.train(
            model_type=ModelType.SIGNAL_QUALITY,
        )

        assert result.success
        assert result.model_id
        assert result.cv_metrics.get("auc", 0) > 0


class TestPredictionService:
    """Test prediction service."""

    def test_predict_signal_quality(
        self,
        populated_feature_db,
        sample_ohlcv,
        tmp_path,
    ):
        """Test signal quality prediction."""
        # First train a model
        config = MLConfig()
        model_dir = str(tmp_path / "models")
        trainer = ModelTrainer(config, populated_feature_db, model_dir)

        result = trainer.train(ModelType.SIGNAL_QUALITY)
        assert result.success

        # Activate model
        registry = ModelRegistry(model_dir, populated_feature_db)
        registry.activate(result.model_id)

        # Make prediction
        service = PredictionService(config, populated_feature_db, registry)

        quality = service.predict_signal_quality(
            signal_id="new-signal",
            ohlcv_data=sample_ohlcv,
            signal_metadata={
                "technical_score": 7.5,
                "conviction": "HIGH",
                "entry_price": 100,
                "target_price": 110,
                "stop_price": 95,
            },
        )

        assert 0 <= quality <= 1


class TestMLSignalScorer:
    """Test ML-enhanced signal scoring."""

    def test_score_signal(
        self,
        populated_feature_db,
        sample_ohlcv,
        tmp_path,
    ):
        """Test signal scoring with ML."""
        config = MLConfig()
        model_dir = str(tmp_path / "models")

        # Train and activate model
        trainer = ModelTrainer(config, populated_feature_db, model_dir)
        result = trainer.train(ModelType.SIGNAL_QUALITY)
        registry = ModelRegistry(model_dir, populated_feature_db)
        registry.activate(result.model_id)

        # Create scorer
        service = PredictionService(config, populated_feature_db, registry)
        scorer = MLSignalScorer(config, service)

        # Score a signal
        enhanced = scorer.score_signal(
            signal_id="test-signal",
            technical_score=7.5,
            news_score=6.0,
            ohlcv_data=sample_ohlcv,
            signal_metadata={
                "symbol": "TEST",
                "conviction": "HIGH",
                "entry_price": 100,
                "target_price": 110,
                "stop_price": 95,
            },
        )

        assert enhanced.ml_enabled
        assert 0 <= enhanced.ml_quality_score <= 1
        assert 0 <= enhanced.combined_score <= 10


class TestModelRegistry:
    """Test model registry."""

    def test_register_and_activate(self, feature_db, tmp_path):
        """Test model registration and activation."""
        np.random.seed(42)

        registry = ModelRegistry(str(tmp_path / "models"), feature_db)

        # Train model
        model = SignalQualityModel()
        X, y = np.random.randn(100, 5), np.random.randn(100)
        model.fit(X, y)

        # Register
        registry.register(model)

        # Activate
        registry.activate(model.model_id)

        # Retrieve
        active = registry.get_active(ModelType.SIGNAL_QUALITY)
        assert active is not None
        assert active.model_id == model.model_id
```

**Acceptance Criteria**:
- [ ] Feature extraction tests pass
- [ ] Database operations tested
- [ ] Walk-forward validation correct
- [ ] Model training works end-to-end
- [ ] Prediction service returns valid scores
- [ ] Signal scorer integrates ML correctly
- [ ] All tests pass with `pytest tests/test_ml_integration.py`

---

## Summary: Part 2 Tasks

| Task | Description | Key Deliverable |
|------|-------------|-----------------|
| 4.2.1 | Walk-Forward Validation | Time-series CV, no look-ahead bias |
| 4.2.2 | Training Pipeline | End-to-end trainer, hyperparameter tuning |
| 4.2.3 | Signal Scoring Integration | PredictionService, MLSignalScorer |
| 4.2.4 | Retraining Job | Automated model updates |
| 4.2.5 | CLI Commands | `ml train/status/models/retrain` |
| 4.2.6 | Integration Tests | Complete ML pipeline tests |

---

## Phase 4 Complete Checklist

After completing both Part 1 and Part 2:

- [ ] Feature store with SQLite backend
- [ ] Technical, market, signal feature extractors
- [ ] Feature pipeline combines all extractors
- [ ] SignalQualityModel with gradient boosting
- [ ] Model registry with versioning
- [ ] Walk-forward cross-validation
- [ ] Training pipeline with metrics
- [ ] Hyperparameter tuning support
- [ ] Prediction service for inference
- [ ] ML-enhanced signal scoring
- [ ] Automated retraining job
- [ ] CLI commands for all operations
- [ ] Comprehensive integration tests

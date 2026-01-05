"""Hyperparameter tuning for ML models."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

# Try to import base models, create minimal interface if not available
try:
    from trading_system.ml_refinement.models.base_model import BaseModel, SignalQualityModel

    BaseModelType = BaseModel
    SignalQualityModelType = SignalQualityModel
except ImportError:
    # Create minimal interface for models
    class _BaseModelStub:
        """Base model interface."""

        def __init__(self, **kwargs):
            self.model_id = f"model_{np.random.randint(10000, 99999)}"

        def fit(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
            """Train the model."""
            return {"accuracy": 0.0}

        def predict(self, X):
            """Make predictions."""
            return np.zeros(len(X))

        def predict_proba(self, X):
            """Predict probabilities."""
            return np.ones((len(X), 2)) * 0.5

    class _SignalQualityModelStub(_BaseModelStub):
        """Signal quality model."""

    BaseModelType = _BaseModelStub
    SignalQualityModelType = _SignalQualityModelStub


from trading_system.ml_refinement.validation.metrics import calculate_classification_metrics
from trading_system.ml_refinement.validation.walk_forward import WalkForwardValidator


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
        model_class: type = SignalQualityModelType,
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

        best_score = -float("inf")
        best_params = {}

        for combo in combinations:
            params = dict(zip(param_names, combo))

            # Evaluate this combination
            scores = self._evaluate_params(X, y, feature_names, params, scoring)
            avg_score = np.mean(scores)

            result.all_results.append(
                {
                    "params": params,
                    "scores": scores,
                    "mean_score": avg_score,
                    "std_score": np.std(scores),
                }
            )

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

        best_score = -float("inf")
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

            result.all_results.append(
                {
                    "trial": trial,
                    "params": params,
                    "scores": scores,
                    "mean_score": avg_score,
                }
            )

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
            X_train = X[split.train_start : split.train_end]
            y_train = y[split.train_start : split.train_end]
            X_val = X[split.val_start : split.val_end]
            y_val = y[split.val_start : split.val_end]

            # Train model
            model = self.model_class(**params)
            model.fit(X_train, y_train, None, None, feature_names)

            # Evaluate
            y_pred = model.predict(X_val)
            y_proba_full = model.predict_proba(X_val)

            # Extract positive class probabilities (assuming binary classification)
            if y_proba_full.ndim > 1:
                y_proba = y_proba_full[:, 1] if y_proba_full.shape[1] > 1 else y_proba_full[:, 0]
            else:
                y_proba = y_proba_full

            y_val_binary = (y_val > 0).astype(int)

            metrics = calculate_classification_metrics(y_val_binary, y_pred, y_proba)
            scores.append(metrics.get(scoring, 0))

        return scores

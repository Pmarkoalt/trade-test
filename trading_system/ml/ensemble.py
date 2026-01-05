"""Ensemble models for ML predictions.

This module provides ensemble methods including:
- Voting classifier/regressor
- Stacking
- Boosting (XGBoost, LightGBM)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from trading_system.ml.models import MLModel, ModelType


class EnsembleModel(ABC):
    """Abstract base class for ensemble models."""

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[tuple[pd.DataFrame, pd.Series]] = None,
    ) -> Dict[str, float]:
        """Train the ensemble model.

        Args:
            X: Feature matrix
            y: Target vector
            validation_data: Optional validation data

        Returns:
            Dictionary of performance metrics
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Array of predictions
        """

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Make probability predictions (if supported).

        Args:
            X: Feature matrix

        Returns:
            Array of probability predictions, or None if not supported
        """


class VotingEnsemble(EnsembleModel):
    """Voting ensemble (hard voting for classification, average for regression).

    Combines predictions from multiple base models using voting or averaging.
    """

    def __init__(
        self,
        base_models: List[MLModel],
        voting: str = "hard",  # "hard" for classification, "soft" for probabilities, "average" for regression
        weights: Optional[List[float]] = None,
    ):
        """Initialize voting ensemble.

        Args:
            base_models: List of base MLModel instances
            voting: Voting strategy ("hard", "soft", or "average")
            weights: Optional weights for each model (must sum to 1.0)
        """
        self.base_models = base_models
        self.voting = voting
        self.weights = weights

        if weights is not None:
            if len(weights) != len(base_models):
                raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({len(base_models)})")
            if abs(sum(weights) - 1.0) > 1e-6:
                raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")
        else:
            # Equal weights
            self.weights = [1.0 / len(base_models)] * len(base_models)

        self._is_trained = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[tuple[pd.DataFrame, pd.Series]] = None,
    ) -> Dict[str, float]:
        """Train all base models.

        Args:
            X: Feature matrix
            y: Target vector
            validation_data: Optional validation data

        Returns:
            Dictionary of average performance metrics
        """
        metrics_list = []

        for model in self.base_models:
            metrics = model.train(X, y, validation_data=validation_data)
            metrics_list.append(metrics)

        self._is_trained = True

        # Return average metrics
        avg_metrics = {}
        if metrics_list:
            for key in metrics_list[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in metrics_list if key in m])

        return avg_metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using voting/averaging.

        Args:
            X: Feature matrix

        Returns:
            Array of predictions
        """
        if not self._is_trained:
            raise ValueError("Ensemble must be trained before prediction")

        predictions = []
        for model in self.base_models:
            pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)  # Shape: (n_models, n_samples)

        if self.voting == "hard":
            # For classification: majority vote
            from scipy import stats

            return stats.mode(predictions, axis=0)[0].flatten()
        elif self.voting == "soft":
            # For classification: weighted average of probabilities
            probas = []
            for model in self.base_models:
                proba = model.predict_proba(X)
                if proba is not None:
                    probas.append(proba)

            if probas:
                weighted_proba = np.average(probas, axis=0, weights=self.weights)
                return np.argmax(weighted_proba, axis=1)
            else:
                # Fallback to hard voting
                from scipy import stats

                return stats.mode(predictions, axis=0)[0].flatten()
        else:  # "average" for regression
            # Weighted average
            return np.average(predictions, axis=0, weights=self.weights)

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Make probability predictions (for classification).

        Args:
            X: Feature matrix

        Returns:
            Array of probability predictions, or None if not supported
        """
        if not self._is_trained:
            raise ValueError("Ensemble must be trained before prediction")

        probas = []
        for model in self.base_models:
            proba = model.predict_proba(X)
            if proba is not None:
                probas.append(proba)

        if probas:
            # Weighted average of probabilities
            return np.average(probas, axis=0, weights=self.weights)
        return None


class StackingEnsemble(EnsembleModel):
    """Stacking ensemble with meta-learner.

    Trains base models and uses a meta-learner to combine their predictions.
    """

    def __init__(
        self,
        base_models: List[MLModel],
        meta_model: MLModel,
        use_proba: bool = False,  # Use probabilities instead of predictions for meta-features
    ):
        """Initialize stacking ensemble.

        Args:
            base_models: List of base MLModel instances
            meta_model: Meta-learner model
            use_proba: If True, use probability predictions as meta-features (for classification)
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.use_proba = use_proba
        self._is_trained = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[tuple[pd.DataFrame, pd.Series]] = None,
    ) -> Dict[str, float]:
        """Train base models and meta-learner.

        Args:
            X: Feature matrix
            y: Target vector
            validation_data: Optional validation data (used for meta-learner training)

        Returns:
            Dictionary of performance metrics
        """
        # Train base models
        for model in self.base_models:
            model.train(X, y, validation_data=None)  # Don't use validation for base models

        # Generate meta-features from base models
        meta_features = self._generate_meta_features(X)

        # Train meta-learner
        if validation_data is not None:
            X_val, y_val = validation_data
            meta_features_val = self._generate_meta_features(X_val)
            metrics = self.meta_model.train(
                pd.DataFrame(meta_features), y, validation_data=(pd.DataFrame(meta_features_val), y_val)
            )
        else:
            metrics = self.meta_model.train(pd.DataFrame(meta_features), y, validation_data=None)

        self._is_trained = True
        return metrics

    def _generate_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        """Generate meta-features from base model predictions.

        Args:
            X: Feature matrix

        Returns:
            Array of meta-features (n_samples, n_models or n_classes * n_models)
        """
        meta_features_list = []

        for model in self.base_models:
            if self.use_proba:
                proba = model.predict_proba(X)
                if proba is not None:
                    # Flatten probability matrix if multi-class
                    if proba.ndim == 2:
                        meta_features_list.append(proba)
                    else:
                        meta_features_list.append(proba.reshape(-1, 1))
                else:
                    # Fallback to predictions
                    pred = model.predict(X)
                    meta_features_list.append(pred.reshape(-1, 1))
            else:
                pred = model.predict(X)
                meta_features_list.append(pred.reshape(-1, 1))

        return np.hstack(meta_features_list)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using meta-learner.

        Args:
            X: Feature matrix

        Returns:
            Array of predictions
        """
        if not self._is_trained:
            raise ValueError("Ensemble must be trained before prediction")

        meta_features = self._generate_meta_features(X)
        return self.meta_model.predict(pd.DataFrame(meta_features))

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Make probability predictions (if meta-model supports it).

        Args:
            X: Feature matrix

        Returns:
            Array of probability predictions, or None if not supported
        """
        if not self._is_trained:
            raise ValueError("Ensemble must be trained before prediction")

        meta_features = self._generate_meta_features(X)
        return self.meta_model.predict_proba(pd.DataFrame(meta_features))


class BoostingEnsemble(EnsembleModel):
    """Boosting ensemble using XGBoost or LightGBM.

    This is a wrapper around XGBoost/LightGBM models that are already
    boosting algorithms. This class provides a unified interface.
    """

    def __init__(
        self,
        model_type: ModelType,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ):
        """Initialize boosting ensemble.

        Args:
            model_type: Model type (XGBOOST or LIGHTGBM)
            hyperparameters: Model hyperparameters
        """
        if model_type not in [ModelType.XGBOOST, ModelType.LIGHTGBM]:
            raise ValueError(f"Boosting ensemble only supports XGBOOST or LIGHTGBM, got {model_type}")

        self.model_type = model_type
        self.hyperparameters = hyperparameters or {}
        self._model: Optional[MLModel] = None
        self._is_trained = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[tuple[pd.DataFrame, pd.Series]] = None,
    ) -> Dict[str, float]:
        """Train boosting model.

        Args:
            X: Feature matrix
            y: Target vector
            validation_data: Optional validation data

        Returns:
            Dictionary of performance metrics
        """
        self._model = MLModel._create_model_instance(
            model_type=self.model_type,
            hyperparameters=self.hyperparameters,
        )

        metrics = self._model.train(X, y, validation_data=validation_data)
        self._is_trained = True

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Array of predictions
        """
        if not self._is_trained or self._model is None:
            raise ValueError("Model must be trained before prediction")

        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Make probability predictions (if supported).

        Args:
            X: Feature matrix

        Returns:
            Array of probability predictions, or None if not supported
        """
        if not self._is_trained or self._model is None:
            raise ValueError("Model must be trained before prediction")

        return self._model.predict_proba(X)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance (if supported).

        Returns:
            Dictionary mapping feature names to importance scores, or None
        """
        if not self._is_trained or self._model is None:
            return None

        if hasattr(self._model._model, "feature_importances_"):
            importances = self._model._model.feature_importances_
            # Try to get feature names from the model
            if hasattr(self._model._model, "feature_names_in_"):
                feature_names = self._model._model.feature_names_in_
            else:
                feature_names = [f"feature_{i}" for i in range(len(importances))]

            return dict(zip(feature_names, importances))

        return None

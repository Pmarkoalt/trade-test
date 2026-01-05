"""Online learning and incremental model updates.

This module provides:
- Incremental learning for models
- Concept drift detection
- Model retraining pipeline
- Model versioning
"""

import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from trading_system.ml.models import MLModel


class ConceptDriftDetector:
    """Detect concept drift in data distribution or model performance."""

    def __init__(
        self,
        window_size: int = 100,
        threshold: float = 0.1,
        method: str = "performance",  # "performance" or "distribution"
    ):
        """Initialize concept drift detector.

        Args:
            window_size: Size of sliding window for drift detection
            threshold: Threshold for drift detection (performance degradation or distribution shift)
            method: Detection method ("performance" or "distribution")
        """
        self.window_size = window_size
        self.threshold = threshold
        self.method = method

        # Performance-based drift detection
        self._performance_history: Deque[float] = deque(maxlen=window_size)
        self._baseline_performance: Optional[float] = None

        # Distribution-based drift detection (using KL divergence or similar)
        self._feature_distributions: Optional[Dict[str, Tuple[float, float]]] = None

    def update_performance(self, performance: float) -> bool:
        """Update performance history and detect drift.

        Args:
            performance: Current performance metric (e.g., accuracy, R²)

        Returns:
            True if drift detected, False otherwise
        """
        self._performance_history.append(performance)

        if len(self._performance_history) < self.window_size:
            return False

        if self._baseline_performance is None:
            # Set baseline from first window
            self._baseline_performance = np.mean(list(self._performance_history)[: self.window_size // 2])
            return False

        # Compare recent performance to baseline
        recent_performance = np.mean(list(self._performance_history)[-self.window_size // 2 :])
        performance_drop = self._baseline_performance - recent_performance

        if performance_drop > self.threshold:
            # Drift detected - update baseline
            self._baseline_performance = recent_performance
            return True

        return False

    def update_distribution(self, features: pd.DataFrame) -> bool:
        """Update feature distribution and detect drift.

        Args:
            features: Feature matrix

        Returns:
            True if drift detected, False otherwise
        """
        # Compute feature statistics
        current_stats = {}
        for col in features.columns:
            col_data = features[col].dropna()
            if len(col_data) > 0:
                current_stats[col] = (col_data.mean(), col_data.std())

        if self._feature_distributions is None:
            # Initialize baseline
            self._feature_distributions = current_stats
            return False

        # Compare distributions (simplified: compare means)
        drift_detected = False
        for col, (mean, std) in current_stats.items():
            if col in self._feature_distributions:
                baseline_mean, baseline_std = self._feature_distributions[col]
                # Simple drift detection: significant change in mean
                if baseline_std > 0:
                    z_score = abs(mean - baseline_mean) / baseline_std
                    if z_score > 2.0:  # 2 standard deviations
                        drift_detected = True
                        break

        if drift_detected:
            # Update baseline
            self._feature_distributions = current_stats

        return drift_detected


class IncrementalLearner:
    """Incremental learning wrapper for models that support partial_fit."""

    def __init__(
        self,
        model: MLModel,
        batch_size: int = 100,
        max_samples: Optional[int] = None,  # Maximum samples to keep in memory
    ):
        """Initialize incremental learner.

        Args:
            model: MLModel instance (must support partial_fit if available)
            batch_size: Number of samples per batch
            max_samples: Maximum samples to keep in memory (for memory management)
        """
        self.model = model
        self.batch_size = batch_size
        self.max_samples = max_samples
        self._sample_buffer: List[Tuple[pd.DataFrame, pd.Series]] = []
        self._total_samples = 0

    def partial_fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        """Perform incremental update.

        Args:
            X: Feature matrix
            y: Target vector
        """
        # Check if underlying model supports partial_fit
        if hasattr(self.model._model, "partial_fit"):
            # Use partial_fit if available
            self.model._model.partial_fit(X, y)
        else:
            # Buffer samples and retrain periodically
            self._sample_buffer.append((X, y))
            self._total_samples += len(X)

            # Retrain if buffer is full
            if len(self._sample_buffer) * self.batch_size >= self.batch_size:
                self._retrain_from_buffer()

    def _retrain_from_buffer(self) -> None:
        """Retrain model from buffered samples."""
        if not self._sample_buffer:
            return

        # Combine all buffered samples
        X_list = []
        y_list = []

        for X, y in self._sample_buffer:
            X_list.append(X)
            y_list.append(y)

        X_combined = pd.concat(X_list, ignore_index=True)
        y_combined = pd.concat(y_list, ignore_index=True)

        # Limit samples if max_samples is set
        if self.max_samples is not None and len(X_combined) > self.max_samples:
            # Keep most recent samples
            X_combined = X_combined.iloc[-self.max_samples :]
            y_combined = y_combined.iloc[-self.max_samples :]

        # Retrain model
        self.model.train(X_combined, y_combined, validation_data=None)

        # Clear buffer
        self._sample_buffer = []


class ModelRetrainingPipeline:
    """Pipeline for automated model retraining."""

    def __init__(
        self,
        model: MLModel,
        retrain_trigger: str = "drift",  # "drift", "time", "performance", or "manual"
        retrain_interval_days: int = 30,
        min_samples_for_retrain: int = 100,
        concept_drift_detector: Optional[ConceptDriftDetector] = None,
    ):
        """Initialize retraining pipeline.

        Args:
            model: MLModel instance to retrain
            retrain_trigger: Trigger for retraining ("drift", "time", "performance", or "manual")
            retrain_interval_days: Days between retraining (for "time" trigger)
            min_samples_for_retrain: Minimum samples required for retraining
            concept_drift_detector: Optional concept drift detector
        """
        self.model = model
        self.retrain_trigger = retrain_trigger
        self.retrain_interval_days = retrain_interval_days
        self.min_samples_for_retrain = min_samples_for_retrain
        self.concept_drift_detector = concept_drift_detector or ConceptDriftDetector()

        self._last_retrain_date: Optional[datetime] = None
        self._performance_history: List[float] = []
        self._training_samples: List[Tuple[pd.DataFrame, pd.Series]] = []

    def should_retrain(
        self,
        current_performance: Optional[float] = None,
        new_samples: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    ) -> bool:
        """Check if model should be retrained.

        Args:
            current_performance: Current performance metric
            new_samples: New training samples

        Returns:
            True if retraining is needed
        """
        if self.retrain_trigger == "manual":
            return False

        if self.retrain_trigger == "time":
            if self._last_retrain_date is None:
                return True
            days_since_retrain = (datetime.now() - self._last_retrain_date).days
            return days_since_retrain >= self.retrain_interval_days

        if self.retrain_trigger == "drift":
            if current_performance is not None:
                drift_detected = self.concept_drift_detector.update_performance(current_performance)
                return drift_detected
            return False

        if self.retrain_trigger == "performance":
            if current_performance is not None:
                self._performance_history.append(current_performance)
                if len(self._performance_history) >= 10:
                    # Check if performance is degrading
                    recent_avg = np.mean(self._performance_history[-5:])
                    earlier_avg = np.mean(self._performance_history[-10:-5])
                    if recent_avg < earlier_avg * 0.95:  # 5% degradation
                        return True
            return False

        return False

    def retrain(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    ) -> Dict[str, float]:
        """Retrain the model.

        Args:
            X: Feature matrix
            y: Target vector
            validation_data: Optional validation data

        Returns:
            Dictionary of training metrics
        """
        if len(X) < self.min_samples_for_retrain:
            raise ValueError(f"Insufficient samples for retraining: {len(X)} < {self.min_samples_for_retrain}")

        # Train model
        metrics = self.model.train(X, y, validation_data=validation_data)

        # Update metadata
        if self.model._metadata:
            self.model._metadata.training_date = pd.Timestamp.now()
            self.model._metadata.training_samples = len(X)
            self.model._metadata.performance_metrics.update(metrics)

        self._last_retrain_date = datetime.now()

        return metrics


class ModelVersionManager:
    """Manage model versions and versioning."""

    def __init__(self, model_directory: Path):
        """Initialize version manager.

        Args:
            model_directory: Base directory for storing model versions
        """
        self.model_directory = Path(model_directory)
        self.model_directory.mkdir(parents=True, exist_ok=True)

    def save_version(
        self,
        model: MLModel,
        version: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a model version.

        Args:
            model: MLModel instance
            version: Version string (e.g., "1.0.0", "2024-01-15")
            metadata: Additional metadata to store

        Returns:
            Path to saved model directory
        """
        version_dir = self.model_directory / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model.save(version_dir)

        # Save additional metadata
        if metadata:
            metadata_path = version_dir / "additional_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

        # Update version index
        self._update_version_index(version)

        return version_dir

    def load_version(self, version: str) -> MLModel:
        """Load a model version.

        Args:
            version: Version string

        Returns:
            Loaded MLModel instance
        """
        version_dir = self.model_directory / version
        if not version_dir.exists():
            raise ValueError(f"Model version {version} not found")

        return MLModel.load(version_dir)

    def list_versions(self) -> List[str]:
        """List all available model versions.

        Returns:
            List of version strings
        """
        versions = []
        for item in self.model_directory.iterdir():
            if item.is_dir() and (item / "config.json").exists():
                versions.append(item.name)

        # Sort versions (assuming semantic versioning or date format)
        try:
            versions.sort(key=lambda v: [int(x) for x in v.split(".")])
        except ValueError:
            # Fallback to string sort
            versions.sort()

        return versions

    def get_latest_version(self) -> Optional[str]:
        """Get the latest model version.

        Returns:
            Latest version string, or None if no versions exist
        """
        versions = self.list_versions()
        return versions[-1] if versions else None

    def _update_version_index(self, version: str) -> None:
        """Update version index file."""
        index_path = self.model_directory / "versions.json"

        if index_path.exists():
            with open(index_path, "r") as f:
                index = json.load(f)
        else:
            index = {"versions": []}

        if version not in index["versions"]:
            index["versions"].append(version)
            index["versions"].sort()

        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

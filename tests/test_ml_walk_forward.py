"""Tests for walk-forward validation."""

import numpy as np
import pytest

from trading_system.ml_refinement.config import TrainingConfig
from trading_system.ml_refinement.validation.metrics import (
    calculate_auc,
    calculate_calibration_error,
    calculate_classification_metrics,
    calculate_log_loss,
    calculate_regression_metrics,
    calculate_trading_metrics,
)
from trading_system.ml_refinement.validation.walk_forward import (
    ExpandingWindowValidator,
    PurgedKFold,
    WalkForwardValidator,
)


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


def test_walk_forward_with_dates():
    """Test walk-forward with date information."""
    validator = WalkForwardValidator(
        train_window=10,
        val_window=5,
        step_size=5,
    )

    dates = [f"2024-01-{i:02d}" for i in range(1, 31)]
    splits = list(validator.generate_splits(30, dates=dates))

    assert len(splits) > 0
    for split in splits:
        assert split.train_start_date != ""
        assert split.val_start_date != ""
        assert split.train_end_date != ""
        assert split.val_end_date != ""


def test_walk_forward_insufficient_samples():
    """Test walk-forward with insufficient samples."""
    validator = WalkForwardValidator(
        train_window=100,
        val_window=20,
        step_size=20,
        min_train_samples=100,
        min_val_samples=20,
    )

    splits = list(validator.generate_splits(50))  # Too few samples
    assert len(splits) == 0


def test_walk_forward_count_folds():
    """Test fold counting."""
    validator = WalkForwardValidator(
        train_window=100,
        val_window=20,
        step_size=20,
    )

    count = validator.count_folds(200)
    splits = list(validator.generate_splits(200))
    assert count == len(splits)


def test_walk_forward_from_config():
    """Test creating validator from config."""
    config = TrainingConfig(
        train_window_days=252,
        validation_window_days=63,
        step_size_days=21,
        min_training_samples=100,
        min_validation_samples=20,
    )

    validator = WalkForwardValidator.from_config(config)
    assert validator.train_window == 252
    assert validator.val_window == 63
    assert validator.step_size == 21


def test_expanding_window_validator():
    """Test expanding window validation."""
    validator = ExpandingWindowValidator(
        initial_train_size=50,
        val_window=20,
        step_size=20,
    )

    splits = list(validator.generate_splits(150))

    assert len(splits) > 0

    # First split should start at 0
    assert splits[0].train_start == 0

    # Training window should expand
    assert splits[0].train_size == 50
    if len(splits) > 1:
        assert splits[1].train_size > splits[0].train_size


def test_purged_kfold():
    """Test purged K-Fold."""
    kfold = PurgedKFold(n_splits=5, purge_window=5, embargo_pct=0.01)

    splits = list(kfold.split(100))

    assert len(splits) == 5

    for train_idx, test_idx in splits:
        # No overlap
        assert len(np.intersect1d(train_idx, test_idx)) == 0

        # All indices covered
        all_indices = np.concatenate([train_idx, test_idx])
        assert len(all_indices) <= 100


def test_classification_metrics():
    """Test classification metric calculation."""
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])
    y_proba = np.array([0.8, 0.4, 0.2, 0.6, 0.9])

    metrics = calculate_classification_metrics(y_true, y_pred, y_proba)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "auc" in metrics
    assert 0 <= metrics["auc"] <= 1
    assert 0 <= metrics["accuracy"] <= 1


def test_classification_metrics_no_proba():
    """Test classification metrics without probabilities."""
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])

    metrics = calculate_classification_metrics(y_true, y_pred)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "auc" not in metrics


def test_classification_metrics_empty():
    """Test classification metrics with empty arrays."""
    y_true = np.array([])
    y_pred = np.array([])

    metrics = calculate_classification_metrics(y_true, y_pred)
    assert "error" in metrics


def test_auc_perfect():
    """Test AUC calculation with perfect predictions."""
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])

    auc = calculate_auc(y_true, y_proba)
    assert auc > 0.9  # Should be close to 1.0


def test_auc_random():
    """Test AUC calculation with random predictions."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_proba = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    auc = calculate_auc(y_true, y_proba)
    # Random should be around 0.5
    assert 0.4 <= auc <= 0.6


def test_auc_all_positive():
    """Test AUC with all positive labels."""
    y_true = np.array([1, 1, 1, 1])
    y_proba = np.array([0.7, 0.8, 0.6, 0.9])

    auc = calculate_auc(y_true, y_proba)
    assert auc == 0.5  # Should default to 0.5 when no negatives


def test_log_loss():
    """Test log loss calculation."""
    y_true = np.array([1, 0, 1, 0])
    y_proba = np.array([0.9, 0.1, 0.8, 0.2])

    loss = calculate_log_loss(y_true, y_proba)
    assert loss >= 0
    assert loss < 1.0  # Should be relatively low for good predictions


def test_regression_metrics():
    """Test regression metric calculation."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])

    metrics = calculate_regression_metrics(y_true, y_pred)

    assert "mse" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert "correlation" in metrics

    assert metrics["mse"] >= 0
    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0
    assert metrics["r2"] <= 1.0


def test_regression_metrics_perfect():
    """Test regression metrics with perfect predictions."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    metrics = calculate_regression_metrics(y_true, y_pred)

    assert metrics["mse"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["correlation"] == 1.0


def test_regression_metrics_empty():
    """Test regression metrics with empty arrays."""
    y_true = np.array([])
    y_pred = np.array([])

    metrics = calculate_regression_metrics(y_true, y_pred)
    assert "error" in metrics


def test_trading_metrics():
    """Test trading-specific metrics."""
    y_true = np.array([1, 1, 0, 0, 1, 0, 1])
    y_proba = np.array([0.9, 0.8, 0.3, 0.2, 0.85, 0.4, 0.75])

    metrics = calculate_trading_metrics(y_true, y_proba, threshold=0.5)

    assert "high_conf_count" in metrics
    assert "high_conf_accuracy" in metrics
    assert "expected_value" in metrics
    assert "calibration_error" in metrics

    assert metrics["high_conf_count"] > 0


def test_trading_metrics_no_high_conf():
    """Test trading metrics when no high confidence predictions."""
    y_true = np.array([1, 0, 1])
    y_proba = np.array([0.3, 0.2, 0.4])  # All below threshold

    metrics = calculate_trading_metrics(y_true, y_proba, threshold=0.5)
    assert metrics["high_conf_count"] == 0


def test_calibration_error():
    """Test calibration error calculation."""
    # Well-calibrated probabilities
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_proba = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.3, 0.6, 0.4])

    ece = calculate_calibration_error(y_true, y_proba)
    assert ece >= 0
    assert ece < 0.5  # Should be relatively low for well-calibrated


def test_calibration_error_poor():
    """Test calibration error with poorly calibrated probabilities."""
    # Poorly calibrated: high confidence but wrong
    y_true = np.array([0, 0, 0, 0])
    y_proba = np.array([0.9, 0.95, 0.85, 0.9])  # High confidence but all wrong

    ece = calculate_calibration_error(y_true, y_proba)
    assert ece > 0.5  # Should be high for poorly calibrated


def test_walk_forward_minimum_sizes():
    """Test that minimum sizes are enforced."""
    validator = WalkForwardValidator(
        train_window=100,
        val_window=20,
        step_size=20,
        min_train_samples=150,  # Higher than train_window
        min_val_samples=20,
    )

    splits = list(validator.generate_splits(200))
    # Should have no splits because train_window < min_train_samples
    assert len(splits) == 0


def test_walk_forward_sequential_splits():
    """Test that splits are sequential and non-overlapping."""
    validator = WalkForwardValidator(
        train_window=50,
        val_window=20,
        step_size=20,
    )

    splits = list(validator.generate_splits(200))

    if len(splits) > 1:
        # Each subsequent split should start after previous
        for i in range(len(splits) - 1):
            assert splits[i + 1].train_start >= splits[i].train_start
            assert splits[i + 1].val_start >= splits[i].val_start

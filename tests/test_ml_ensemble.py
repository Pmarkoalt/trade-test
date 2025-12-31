"""Tests for ML ensemble models."""

import numpy as np
import pandas as pd
import pytest

from trading_system.ml.ensemble import (
    BoostingEnsemble,
    StackingEnsemble,
    VotingEnsemble,
)
from trading_system.ml.models import MLModel, ModelType, SklearnMLModel


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 200
    n_features = 10

    X = pd.DataFrame(np.random.randn(n_samples, n_features), columns=[f"feature_{i}" for i in range(n_features)])

    # Create binary classification target
    y = pd.Series((X.sum(axis=1) > 0).astype(int))

    return X, y


@pytest.fixture
def base_models(sample_data):
    """Create base models for ensemble."""
    X, y = sample_data

    models = []
    for model_type in [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING]:
        model = MLModel._create_model_instance(
            model_type=model_type,
            hyperparameters={"n_estimators": 10, "task": "classification"},
        )
        model.train(X, y)
        models.append(model)

    return models


def test_voting_ensemble_hard(sample_data, base_models):
    """Test hard voting ensemble."""
    X, y = sample_data

    ensemble = VotingEnsemble(
        base_models=base_models,
        voting="hard",
    )

    # Train ensemble
    metrics = ensemble.fit(X, y)

    assert "accuracy" in metrics or "training_time_seconds" in metrics

    # Make predictions
    predictions = ensemble.predict(X)

    assert len(predictions) == len(X)
    assert predictions.dtype in [np.int64, np.int32, np.float64]


def test_voting_ensemble_average(sample_data):
    """Test average voting ensemble (for regression)."""
    X, y = sample_data

    # Convert to regression problem
    y_reg = pd.Series(X.sum(axis=1))

    models = []
    for model_type in [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING]:
        model = MLModel._create_model_instance(
            model_type=model_type,
            hyperparameters={"n_estimators": 10, "task": "regression"},
        )
        model.train(X, y_reg)
        models.append(model)

    ensemble = VotingEnsemble(
        base_models=models,
        voting="average",
    )

    predictions = ensemble.predict(X)

    assert len(predictions) == len(X)
    assert np.all(np.isfinite(predictions))


def test_voting_ensemble_weights(sample_data, base_models):
    """Test weighted voting ensemble."""
    X, y = sample_data

    ensemble = VotingEnsemble(
        base_models=base_models,
        voting="hard",
        weights=[0.7, 0.3],
    )

    ensemble.fit(X, y)
    predictions = ensemble.predict(X)

    assert len(predictions) == len(X)


def test_stacking_ensemble(sample_data, base_models):
    """Test stacking ensemble."""
    X, y = sample_data

    # Create meta-learner
    meta_model = MLModel._create_model_instance(
        model_type=ModelType.LOGISTIC_REGRESSION,
        hyperparameters={},
    )

    ensemble = StackingEnsemble(
        base_models=base_models,
        meta_model=meta_model,
        use_proba=False,
    )

    # Train ensemble
    metrics = ensemble.fit(X, y)

    assert "accuracy" in metrics or "training_time_seconds" in metrics

    # Make predictions
    predictions = ensemble.predict(X)

    assert len(predictions) == len(X)


def test_stacking_ensemble_with_proba(sample_data, base_models):
    """Test stacking ensemble with probability features."""
    X, y = sample_data

    meta_model = MLModel._create_model_instance(
        model_type=ModelType.LOGISTIC_REGRESSION,
        hyperparameters={},
    )

    ensemble = StackingEnsemble(
        base_models=base_models,
        meta_model=meta_model,
        use_proba=True,
    )

    ensemble.fit(X, y)
    predictions = ensemble.predict(X)

    assert len(predictions) == len(X)


def test_boosting_ensemble_xgboost(sample_data):
    """Test XGBoost boosting ensemble."""
    pytest.importorskip("xgboost")

    X, y = sample_data

    ensemble = BoostingEnsemble(
        model_type=ModelType.XGBOOST,
        hyperparameters={"n_estimators": 10, "task": "classification"},
    )

    metrics = ensemble.fit(X, y)

    assert "accuracy" in metrics or "training_time_seconds" in metrics

    predictions = ensemble.predict(X)
    assert len(predictions) == len(X)


def test_boosting_ensemble_lightgbm(sample_data):
    """Test LightGBM boosting ensemble."""
    pytest.importorskip("lightgbm")

    X, y = sample_data

    ensemble = BoostingEnsemble(
        model_type=ModelType.LIGHTGBM,
        hyperparameters={"n_estimators": 10, "task": "classification"},
    )

    metrics = ensemble.fit(X, y)

    assert "accuracy" in metrics or "training_time_seconds" in metrics

    predictions = ensemble.predict(X)
    assert len(predictions) == len(X)


def test_boosting_ensemble_feature_importance(sample_data):
    """Test feature importance for boosting ensemble."""
    pytest.importorskip("xgboost")

    X, y = sample_data

    ensemble = BoostingEnsemble(
        model_type=ModelType.XGBOOST,
        hyperparameters={"n_estimators": 10, "task": "classification"},
    )

    ensemble.fit(X, y)

    importance = ensemble.get_feature_importance()

    # Feature importance may or may not be available
    if importance is not None:
        assert len(importance) > 0

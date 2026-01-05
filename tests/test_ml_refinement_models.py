"""Tests for ML refinement models."""

import numpy as np
import pytest

from trading_system.ml_refinement import ModelType  # noqa: E402
from trading_system.ml_refinement.models import BaseModel, ModelRegistry, SignalQualityModel  # noqa: E402
from trading_system.ml_refinement.storage import FeatureDatabase  # noqa: E402

# Skip all tests in this module if sklearn is not installed
pytest.importorskip("sklearn", reason="sklearn required for ML refinement model tests")


def test_signal_quality_model_fit():
    """Test model training."""
    model = SignalQualityModel()

    X = np.random.randn(100, 10)
    y = np.random.randn(100)  # Will be converted to binary

    metrics = model.fit(X, y)

    assert model.is_fitted
    assert "accuracy" in metrics
    assert "auc" in metrics
    assert "f1" in metrics


def test_signal_quality_model_fit_with_validation():
    """Test model training with validation set."""
    model = SignalQualityModel()

    X_train = np.random.randn(100, 10)
    y_train = np.random.randn(100)
    X_val = np.random.randn(20, 10)
    y_val = np.random.randn(20)

    model.fit(X_train, y_train, X_val, y_val)

    assert model.is_fitted
    assert model._validation_samples == 20
    assert "accuracy" in model._validation_metrics
    assert "auc" in model._validation_metrics


def test_signal_quality_model_predict():
    """Test model prediction."""
    model = SignalQualityModel()

    X_train = np.random.randn(100, 10)
    y_train = np.random.randn(100)
    model.fit(X_train, y_train)

    X_test = np.random.randn(10, 10)
    predictions = model.predict(X_test)

    assert len(predictions) == 10
    assert all(pred in [0, 1] for pred in predictions)


def test_signal_quality_model_predict_proba():
    """Test model probability prediction."""
    model = SignalQualityModel()

    X_train = np.random.randn(100, 10)
    y_train = np.random.randn(100)
    model.fit(X_train, y_train)

    X_test = np.random.randn(10, 10)
    probabilities = model.predict_proba(X_test)

    assert len(probabilities) == 10
    assert all(0 <= prob <= 1 for prob in probabilities)


def test_signal_quality_model_unfitted():
    """Test that unfitted model raises error."""
    model = SignalQualityModel()

    X_test = np.random.randn(10, 10)

    with pytest.raises(ValueError, match="not fitted"):
        model.predict(X_test)

    with pytest.raises(ValueError, match="not fitted"):
        model.predict_proba(X_test)


def test_model_save_load(tmp_path):
    """Test model persistence."""
    model = SignalQualityModel()
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    model.fit(X, y)

    # Save
    path = str(tmp_path / "model.pkl")
    assert model.save(path)

    # Load
    loaded = SignalQualityModel()
    assert loaded.load(path)

    assert loaded.is_fitted

    # Predictions should match
    pred1 = model.predict_proba(X[:5])
    pred2 = loaded.predict_proba(X[:5])
    np.testing.assert_array_almost_equal(pred1, pred2)


def test_model_metadata():
    """Test model metadata generation."""
    model = SignalQualityModel()
    X = np.random.randn(100, 10)
    y = np.random.randn(100)

    model.set_training_period("2023-01-01", "2023-12-31")
    model.fit(X, y, feature_names=[f"feature_{i}" for i in range(10)])

    metadata = model.get_metadata()

    assert metadata.model_type == ModelType.SIGNAL_QUALITY
    assert metadata.train_samples == 100
    assert len(metadata.feature_names) == 10
    assert metadata.train_start_date == "2023-01-01"
    assert metadata.train_end_date == "2023-12-31"
    assert "accuracy" in metadata.train_metrics


def test_model_feature_importance():
    """Test feature importance tracking."""
    model = SignalQualityModel()
    X = np.random.randn(100, 10)
    y = np.random.randn(100)

    feature_names = [f"feature_{i}" for i in range(10)]
    model.fit(X, y, feature_names=feature_names)

    # Should have feature importance
    assert len(model._feature_importance) == 10

    # Get top features
    top_features = model.get_top_features(n=5)
    assert len(top_features) == 5
    assert all(isinstance(f, tuple) and len(f) == 2 for f in top_features)


def test_model_registry_register(tmp_path):
    """Test model registration."""
    registry = ModelRegistry(model_dir=str(tmp_path / "models"))

    model = SignalQualityModel()
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    model.fit(X, y)

    assert registry.register(model)

    # Model file should exist
    model_path = tmp_path / "models" / f"{model.model_id}.pkl"
    assert model_path.exists()


def test_model_registry_register_unfitted():
    """Test that unfitted model cannot be registered."""
    registry = ModelRegistry()

    model = SignalQualityModel()  # Not fitted

    assert not registry.register(model)


def test_model_registry_activate(tmp_path):
    """Test model activation."""
    registry = ModelRegistry(model_dir=str(tmp_path / "models"))

    model = SignalQualityModel()
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    model.fit(X, y)

    registry.register(model)

    # Activate
    assert registry.activate(model.model_id)

    # Get active model
    active = registry.get_active(ModelType.SIGNAL_QUALITY)
    assert active is not None
    assert active.model_id == model.model_id
    assert active.is_fitted


def test_model_registry_get_active_none():
    """Test getting active model when none exists."""
    registry = ModelRegistry()

    active = registry.get_active(ModelType.SIGNAL_QUALITY)
    assert active is None


def test_model_registry_with_database(tmp_path):
    """Test registry with database."""
    db = FeatureDatabase(str(tmp_path / "test.db"))
    db.initialize()

    registry = ModelRegistry(model_dir=str(tmp_path / "models"), db=db)

    model = SignalQualityModel()
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    model.fit(X, y)

    # Register should save to database
    assert registry.register(model)

    # Activate should update database
    assert registry.activate(model.model_id)

    # Get active from database
    active = registry.get_active(ModelType.SIGNAL_QUALITY)
    assert active is not None

    db.close()


def test_model_registry_get_history(tmp_path):
    """Test getting model history."""
    db = FeatureDatabase(str(tmp_path / "test.db"))
    db.initialize()

    registry = ModelRegistry(model_dir=str(tmp_path / "models"), db=db)

    # Register multiple models
    for i in range(3):
        model = SignalQualityModel(version=f"1.{i}")
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        model.fit(X, y)
        registry.register(model)

    history = registry.get_model_history(ModelType.SIGNAL_QUALITY, limit=10)
    assert len(history) == 3

    db.close()


def test_model_registry_compare_models(tmp_path):
    """Test model comparison."""
    db = FeatureDatabase(str(tmp_path / "test.db"))
    db.initialize()

    registry = ModelRegistry(model_dir=str(tmp_path / "models"), db=db)

    # Register multiple models
    for i in range(3):
        model = SignalQualityModel(version=f"1.{i}")
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        X_val = np.random.randn(20, 10)
        y_val = np.random.randn(20)
        model.fit(X, y, X_val, y_val)
        registry.register(model)

    # Compare by AUC
    comparisons = registry.compare_models(ModelType.SIGNAL_QUALITY, metric="auc")
    assert len(comparisons) == 3
    # Should be sorted by metric (descending)
    metrics = [c["metric"] for c in comparisons]
    assert metrics == sorted(metrics, reverse=True)

    db.close()


def test_model_custom_parameters():
    """Test model with custom parameters."""
    model = SignalQualityModel(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.05,
    )

    assert model.params["n_estimators"] == 50
    assert model.params["max_depth"] == 3
    assert model.params["learning_rate"] == 0.05

    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    model.fit(X, y)

    assert model.is_fitted


def test_model_load_invalid_path():
    """Test loading from invalid path."""
    model = SignalQualityModel()
    assert not model.load("/nonexistent/path/model.pkl")


def test_model_save_invalid_path():
    """Test saving to invalid path."""
    model = SignalQualityModel()
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    model.fit(X, y)

    # Try to save to invalid path
    assert not model.save("/nonexistent/directory/model.pkl")


def test_model_registry_activate_invalid_id(tmp_path):
    """Test activating invalid model ID."""
    registry = ModelRegistry(model_dir=str(tmp_path / "models"))

    assert not registry.activate("invalid_model_id")


def test_base_model_interface():
    """Test that BaseModel defines complete interface."""
    # Check that BaseModel has all required abstract methods
    assert hasattr(BaseModel, "fit")
    assert hasattr(BaseModel, "predict")
    assert hasattr(BaseModel, "predict_proba")
    assert hasattr(BaseModel, "save")
    assert hasattr(BaseModel, "load")
    assert hasattr(BaseModel, "get_metadata")
    assert hasattr(BaseModel, "set_training_period")

    # Check that SignalQualityModel implements all methods
    model = SignalQualityModel()
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    assert hasattr(model, "save")
    assert hasattr(model, "load")


def test_model_id_generation():
    """Test that model IDs are unique."""
    model1 = SignalQualityModel()
    model2 = SignalQualityModel()

    assert model1.model_id != model2.model_id
    assert model1.model_id.startswith("signal_quality")
    assert model2.model_id.startswith("signal_quality")

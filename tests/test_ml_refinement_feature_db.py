"""Tests for ML refinement feature database."""

import numpy as np
import pytest
from datetime import datetime

from trading_system.ml_refinement import FeatureVector, ModelMetadata, ModelType
from trading_system.ml_refinement.storage import FeatureDatabase


def test_store_and_retrieve_features(tmp_path):
    """Test feature storage roundtrip."""
    db = FeatureDatabase(str(tmp_path / "test.db"))
    db.initialize()

    fv = FeatureVector(
        signal_id="test-123",
        timestamp="2024-01-01T10:00:00",
        features={"rsi": 65.0, "atr": 2.5},
    )

    db.store_feature_vector(fv, symbol="AAPL")
    retrieved = db.get_feature_vector("test-123")

    assert retrieved is not None
    assert retrieved.features["rsi"] == 65.0
    assert retrieved.features["atr"] == 2.5
    assert retrieved.signal_id == "test-123"
    db.close()


def test_get_training_data(tmp_path):
    """Test training data retrieval."""
    db = FeatureDatabase(str(tmp_path / "test.db"))
    db.initialize()

    # Store multiple vectors with targets
    for i in range(5):
        fv = FeatureVector(
            signal_id=f"test-{i}",
            timestamp=f"2024-01-{i+1:02d}T10:00:00",
            features={"rsi": 50.0 + i, "atr": 1.0 + i * 0.1},
            target=float(i - 2),  # Mix of positive and negative
            target_binary=1 if i > 2 else 0,
        )
        db.store_feature_vector(fv, symbol="AAPL")

    X, y, feature_names = db.get_training_data(
        start_date="2024-01-01",
        end_date="2024-01-10",
    )

    # Verify shapes
    assert X.shape[0] == 5  # 5 samples
    assert X.shape[1] == 2  # 2 features (rsi, atr)
    assert y.shape[0] == 5  # 5 targets
    assert len(feature_names) == 2
    assert "atr" in feature_names
    assert "rsi" in feature_names

    # Verify values
    assert X[0, feature_names.index("rsi")] == 50.0
    assert y[0] == -2.0  # First target
    db.close()


def test_get_training_data_binary(tmp_path):
    """Test binary training data retrieval."""
    db = FeatureDatabase(str(tmp_path / "test.db"))
    db.initialize()

    # Store vectors with mixed outcomes
    for i in range(4):
        fv = FeatureVector(
            signal_id=f"test-{i}",
            timestamp=f"2024-01-{i+1:02d}T10:00:00",
            features={"rsi": 50.0},
            target=float(i - 1.5),  # Mix of positive and negative
        )
        db.store_feature_vector(fv)

    X, y_binary, feature_names = db.get_training_data_binary(
        start_date="2024-01-01",
        end_date="2024-01-10",
    )

    assert len(y_binary) == 4
    assert y_binary.dtype == int or y_binary.dtype == np.int64
    # First two should be 0 (negative), last two should be 1 (positive)
    assert y_binary[0] == 0
    assert y_binary[1] == 0
    assert y_binary[2] == 1
    assert y_binary[3] == 1
    db.close()


def test_update_target(tmp_path):
    """Test target updates work correctly."""
    db = FeatureDatabase(str(tmp_path / "test.db"))
    db.initialize()

    # Store feature vector without target
    fv = FeatureVector(
        signal_id="test-123",
        timestamp="2024-01-01T10:00:00",
        features={"rsi": 65.0},
    )
    db.store_feature_vector(fv)

    # Update with target
    success = db.update_target("test-123", r_multiple=2.5, return_pct=5.0)
    assert success

    # Retrieve and verify
    retrieved = db.get_feature_vector("test-123")
    assert retrieved.target == 2.5
    assert retrieved.target_binary == 1  # Positive R-multiple

    # Update with negative target
    db.update_target("test-123", r_multiple=-1.0)
    retrieved = db.get_feature_vector("test-123")
    assert retrieved.target == -1.0
    assert retrieved.target_binary == 0  # Negative R-multiple
    db.close()


def test_count_samples(tmp_path):
    """Test sample counting."""
    db = FeatureDatabase(str(tmp_path / "test.db"))
    db.initialize()

    # Store vectors with and without targets
    for i in range(3):
        fv = FeatureVector(
            signal_id=f"test-{i}",
            timestamp=f"2024-01-{i+1:02d}T10:00:00",
            features={"rsi": 50.0},
            target=1.0 if i < 2 else None,  # First two have targets
        )
        db.store_feature_vector(fv)

    # Count all samples
    total = db.count_samples(require_target=False)
    assert total == 3

    # Count only with targets
    with_targets = db.count_samples(require_target=True)
    assert with_targets == 2

    # Count with date range
    in_range = db.count_samples(start_date="2024-01-01", end_date="2024-01-02", require_target=True)
    assert in_range == 2
    db.close()


def test_model_registry_tracks_versions(tmp_path):
    """Test model registry tracks versions."""
    db = FeatureDatabase(str(tmp_path / "test.db"))
    db.initialize()

    # Register multiple models
    for i in range(3):
        metadata = ModelMetadata(
            model_id=f"model-{i}",
            model_type=ModelType.SIGNAL_QUALITY,
            version=f"1.{i}",
            created_at=datetime.now().isoformat(),
            train_start_date="2023-01-01",
            train_end_date="2023-12-31",
            train_samples=1000,
            validation_samples=200,
            train_metrics={"accuracy": 0.8 + i * 0.01},
            validation_metrics={"accuracy": 0.75 + i * 0.01},
            feature_names=["rsi", "atr"],
            feature_importance={"rsi": 0.6, "atr": 0.4},
        )
        db.register_model(metadata)

    # Get model history
    history = db.get_model_history(ModelType.SIGNAL_QUALITY.value, limit=10)
    assert len(history) == 3

    # Verify versions
    versions = [m.version for m in history]
    assert "1.0" in versions
    assert "1.1" in versions
    assert "1.2" in versions

    # Verify metrics
    assert history[0].train_metrics["accuracy"] == 0.82
    db.close()


def test_activate_model(tmp_path):
    """Test model activation."""
    db = FeatureDatabase(str(tmp_path / "test.db"))
    db.initialize()

    # Register two models of same type
    for i in range(2):
        metadata = ModelMetadata(
            model_id=f"model-{i}",
            model_type=ModelType.SIGNAL_QUALITY,
            version=f"1.{i}",
            created_at=datetime.now().isoformat(),
            train_start_date="2023-01-01",
            train_end_date="2023-12-31",
            train_samples=1000,
            validation_samples=200,
        )
        db.register_model(metadata)

    # Activate first model
    success = db.activate_model("model-0")
    assert success

    # Get active model
    active = db.get_active_model(ModelType.SIGNAL_QUALITY.value)
    assert active is not None
    assert active.model_id == "model-0"
    assert active.is_active

    # Activate second model - should deactivate first
    db.activate_model("model-1")
    active = db.get_active_model(ModelType.SIGNAL_QUALITY.value)
    assert active.model_id == "model-1"
    assert active.is_active

    # Verify first is deactivated
    history = db.get_model_history(ModelType.SIGNAL_QUALITY.value, limit=10)
    model_0 = next(m for m in history if m.model_id == "model-0")
    assert not model_0.is_active
    db.close()


def test_prediction_logging(tmp_path):
    """Test prediction logging works."""
    db = FeatureDatabase(str(tmp_path / "test.db"))
    db.initialize()

    # Register a model first
    metadata = ModelMetadata(
        model_id="model-1",
        model_type=ModelType.SIGNAL_QUALITY,
        version="1.0",
        created_at=datetime.now().isoformat(),
        train_start_date="2023-01-01",
        train_end_date="2023-12-31",
        train_samples=1000,
        validation_samples=200,
    )
    db.register_model(metadata)

    # Log a prediction
    success = db.log_prediction(
        signal_id="signal-123",
        model_id="model-1",
        quality_score=0.75,
        predicted_r=2.0,
        confidence=0.8,
    )
    assert success

    # Update with actual outcome
    success = db.update_prediction_actual("signal-123", actual_r=2.5)
    assert success

    # Verify prediction was logged (check database directly)
    cursor = db.connection.execute("SELECT * FROM prediction_log WHERE signal_id = ?", ("signal-123",))
    row = cursor.fetchone()
    assert row is not None
    assert row["quality_score"] == 0.75
    assert row["predicted_r"] == 2.0
    assert row["actual_r"] == 2.5
    assert row["prediction_error"] == 0.5  # 2.5 - 2.0
    db.close()


def test_training_data_filtering(tmp_path):
    """Test training data filtering by asset class and signal type."""
    db = FeatureDatabase(str(tmp_path / "test.db"))
    db.initialize()

    # Store vectors with different attributes
    for i, (asset_class, signal_type) in enumerate(
        [
            ("equity", "momentum"),
            ("equity", "mean_reversion"),
            ("crypto", "momentum"),
        ]
    ):
        fv = FeatureVector(
            signal_id=f"test-{i}",
            timestamp=f"2024-01-{i+1:02d}T10:00:00",
            features={"rsi": 50.0},
            target=1.0,
        )
        db.store_feature_vector(
            fv,
            asset_class=asset_class,
            signal_type=signal_type,
        )

    # Get all equity
    X, y, _ = db.get_training_data(
        start_date="2024-01-01",
        end_date="2024-01-10",
        asset_class="equity",
    )
    assert X.shape[0] == 2

    # Get only momentum
    X, y, _ = db.get_training_data(
        start_date="2024-01-01",
        end_date="2024-01-10",
        signal_type="momentum",
    )
    assert X.shape[0] == 2

    # Get equity momentum
    X, y, _ = db.get_training_data(
        start_date="2024-01-01",
        end_date="2024-01-10",
        asset_class="equity",
        signal_type="momentum",
    )
    assert X.shape[0] == 1
    db.close()


def test_empty_training_data(tmp_path):
    """Test handling of empty training data."""
    db = FeatureDatabase(str(tmp_path / "test.db"))
    db.initialize()

    # Get training data when no data exists
    X, y, feature_names = db.get_training_data(
        start_date="2024-01-01",
        end_date="2024-01-10",
    )

    assert X.shape == (0,)
    assert y.shape == (0,)
    assert feature_names == []
    db.close()


def test_feature_vector_replace(tmp_path):
    """Test that storing same signal_id replaces existing."""
    db = FeatureDatabase(str(tmp_path / "test.db"))
    db.initialize()

    # Store initial vector
    fv1 = FeatureVector(
        signal_id="test-123",
        timestamp="2024-01-01T10:00:00",
        features={"rsi": 50.0},
    )
    db.store_feature_vector(fv1)

    # Store with same signal_id but different features
    fv2 = FeatureVector(
        signal_id="test-123",
        timestamp="2024-01-01T10:00:00",
        features={"rsi": 75.0, "atr": 2.0},
    )
    db.store_feature_vector(fv2)

    # Should only have one entry
    count = db.count_samples(require_target=False)
    assert count == 1

    # Should have updated features
    retrieved = db.get_feature_vector("test-123")
    assert retrieved.features["rsi"] == 75.0
    assert retrieved.features["atr"] == 2.0
    db.close()


def test_schema_creation(tmp_path):
    """Test that schema creates all tables."""
    db = FeatureDatabase(str(tmp_path / "test.db"))
    db.initialize()

    # Check that all tables exist
    cursor = db.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}

    expected_tables = {
        "feature_vectors",
        "model_registry",
        "feature_definitions",
        "training_runs",
        "prediction_log",
        "ml_schema_migrations",
    }

    assert expected_tables.issubset(tables)

    # Check migrations table has entry
    cursor = db.connection.execute("SELECT version FROM ml_schema_migrations")
    versions = [row[0] for row in cursor.fetchall()]
    assert 1 in versions
    db.close()

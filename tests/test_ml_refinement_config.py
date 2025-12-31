"""Tests for ML refinement configuration module."""

import pytest

from trading_system.ml_refinement import (
    FeatureConfig,
    FeatureSet,
    FeatureVector,
    MLConfig,
    ModelMetadata,
    ModelType,
    TrainingConfig,
)


def test_feature_vector_roundtrip():
    """Test FeatureVector serialization."""
    fv = FeatureVector(
        signal_id="test-123",
        timestamp="2024-01-01T10:00:00",
        features={"rsi": 65.0, "atr_ratio": 1.2},
        target=2.0,
        target_binary=1,
    )
    data = fv.to_dict()
    restored = FeatureVector.from_dict(data)
    assert restored.signal_id == fv.signal_id
    assert restored.features["rsi"] == 65.0
    assert restored.target == 2.0
    assert restored.target_binary == 1


def test_ml_config_defaults():
    """Test MLConfig has sensible defaults."""
    config = MLConfig()
    assert config.enabled
    assert config.training.min_training_samples == 100
    assert config.training.min_validation_samples == 20
    assert config.features.feature_set == FeatureSet.STANDARD
    assert config.use_ml_scores
    assert config.ml_score_weight == 0.3


def test_feature_config_defaults():
    """Test FeatureConfig has sensible defaults."""
    config = FeatureConfig()
    assert config.feature_set == FeatureSet.STANDARD
    assert config.technical_lookbacks == [5, 10, 20, 50, 200]
    assert config.market_regime_window == 20
    assert config.volatility_window == 20
    assert config.news_lookback_hours == 48
    assert config.include_news_features
    assert config.scale_features
    assert config.scaling_method == "standard"


def test_training_config_defaults():
    """Test TrainingConfig has sensible defaults."""
    config = TrainingConfig()
    assert config.train_window_days == 252
    assert config.validation_window_days == 63
    assert config.step_size_days == 21
    assert config.min_training_samples == 100
    assert config.min_validation_samples == 20
    assert config.model_type == "gradient_boosting"
    assert config.early_stopping
    assert config.early_stopping_rounds == 50
    assert config.max_features == 50
    assert config.feature_selection
    assert config.feature_importance_threshold == 0.01


def test_feature_vector_optional_fields():
    """Test FeatureVector with optional fields."""
    fv = FeatureVector(
        signal_id="test-456",
        timestamp="2024-01-01T10:00:00",
        features={"rsi": 50.0},
    )
    assert fv.target is None
    assert fv.target_binary is None
    
    data = fv.to_dict()
    restored = FeatureVector.from_dict(data)
    assert restored.target is None
    assert restored.target_binary is None


def test_model_metadata():
    """Test ModelMetadata creation."""
    metadata = ModelMetadata(
        model_id="model-001",
        model_type=ModelType.SIGNAL_QUALITY,
        version="1.0.0",
        created_at="2024-01-01T10:00:00",
        train_start_date="2023-01-01",
        train_end_date="2023-12-31",
        train_samples=1000,
        validation_samples=200,
    )
    assert metadata.model_id == "model-001"
    assert metadata.model_type == ModelType.SIGNAL_QUALITY
    assert metadata.train_samples == 1000
    assert not metadata.is_active
    assert metadata.deployed_at is None


def test_model_type_enum():
    """Test ModelType enum values."""
    assert ModelType.SIGNAL_QUALITY == "signal_quality"
    assert ModelType.RETURN_PREDICTOR == "return_predictor"
    assert ModelType.REGIME_CLASSIFIER == "regime_classifier"
    assert ModelType.RISK_PREDICTOR == "risk_predictor"


def test_feature_set_enum():
    """Test FeatureSet enum values."""
    assert FeatureSet.MINIMAL == "minimal"
    assert FeatureSet.STANDARD == "standard"
    assert FeatureSet.EXTENDED == "extended"
    assert FeatureSet.CUSTOM == "custom"


def test_ml_config_custom_feature_config():
    """Test MLConfig with custom FeatureConfig."""
    feature_config = FeatureConfig(
        feature_set=FeatureSet.EXTENDED,
        technical_lookbacks=[10, 20, 50],
        include_news_features=False,
    )
    ml_config = MLConfig(features=feature_config)
    assert ml_config.features.feature_set == FeatureSet.EXTENDED
    assert ml_config.features.technical_lookbacks == [10, 20, 50]
    assert not ml_config.features.include_news_features


def test_ml_config_custom_training_config():
    """Test MLConfig with custom TrainingConfig."""
    training_config = TrainingConfig(
        train_window_days=500,
        min_training_samples=200,
        model_type="random_forest",
    )
    ml_config = MLConfig(training=training_config)
    assert ml_config.training.train_window_days == 500
    assert ml_config.training.min_training_samples == 200
    assert ml_config.training.model_type == "random_forest"


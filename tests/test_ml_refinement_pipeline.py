"""Tests for ML refinement feature pipeline."""

import pandas as pd
import pytest

from trading_system.ml_refinement import FeatureConfig, FeatureSet, FeatureVector
from trading_system.ml_refinement.features import FeaturePipeline, FeatureRegistry, FeatureScaler


def create_sample_ohlcv(n_bars: int = 100) -> pd.DataFrame:
    """Create sample OHLCV data."""
    return pd.DataFrame(
        {
            "open": [100.0] * n_bars,
            "high": [101.0] * n_bars,
            "low": [99.0] * n_bars,
            "close": list(range(100, 100 + n_bars)),
            "volume": [1000] * n_bars,
        }
    )


def test_pipeline_standard_features():
    """Test standard feature set extraction."""
    config = FeatureConfig(feature_set=FeatureSet.STANDARD)
    pipeline = FeaturePipeline(config)

    ohlcv = create_sample_ohlcv(100)
    signal_meta = {"technical_score": 7.5, "conviction": "HIGH"}

    features = pipeline.extract_features(
        signal_id="test",
        ohlcv_data=ohlcv,
        signal_metadata=signal_meta,
    )

    assert len(features) > 0
    assert all(isinstance(v, float) for v in features.values())


def test_pipeline_minimal_features():
    """Test minimal feature set."""
    config = FeatureConfig(feature_set=FeatureSet.MINIMAL)
    pipeline = FeaturePipeline(config)

    ohlcv = create_sample_ohlcv(100)
    signal_meta = {"technical_score": 7.5, "conviction": "HIGH"}

    features = pipeline.extract_features(
        signal_id="test",
        ohlcv_data=ohlcv,
        signal_metadata=signal_meta,
    )

    # Minimal should have fewer features than standard
    assert len(features) > 0
    feature_names = pipeline.get_feature_names()
    assert "trend_features" in str(feature_names) or "price_vs_ma" in str(feature_names)


def test_pipeline_extended_features():
    """Test extended feature set."""
    config = FeatureConfig(feature_set=FeatureSet.EXTENDED)
    pipeline = FeaturePipeline(config)

    ohlcv = create_sample_ohlcv(100)
    signal_meta = {"technical_score": 7.5, "conviction": "HIGH"}

    features = pipeline.extract_features(
        signal_id="test",
        ohlcv_data=ohlcv,
        signal_metadata=signal_meta,
    )

    # Extended should have more features
    assert len(features) > 0
    extended_count = pipeline.get_feature_count()
    assert extended_count > 0


def test_pipeline_custom_features():
    """Test custom feature selection."""
    config = FeatureConfig(
        feature_set=FeatureSet.CUSTOM,
        custom_features=["trend_features", "signal_metadata"],
    )
    pipeline = FeaturePipeline(config)

    ohlcv = create_sample_ohlcv(100)
    signal_meta = {"technical_score": 7.5, "conviction": "HIGH"}

    features = pipeline.extract_features(
        signal_id="test",
        ohlcv_data=ohlcv,
        signal_metadata=signal_meta,
    )

    assert len(features) > 0


def test_pipeline_custom_features_invalid():
    """Test custom features with invalid extractor name."""
    config = FeatureConfig(
        feature_set=FeatureSet.CUSTOM,
        custom_features=["trend_features", "invalid_extractor"],
    )
    pipeline = FeaturePipeline(config)

    ohlcv = create_sample_ohlcv(100)
    signal_meta = {"technical_score": 7.5, "conviction": "HIGH"}

    # Should handle invalid extractor gracefully
    features = pipeline.extract_features(
        signal_id="test",
        ohlcv_data=ohlcv,
        signal_metadata=signal_meta,
    )

    # Should still extract from valid extractors
    assert len(features) > 0


def test_pipeline_with_benchmark():
    """Test pipeline with benchmark data."""
    config = FeatureConfig(feature_set=FeatureSet.EXTENDED)
    pipeline = FeaturePipeline(config)

    ohlcv = create_sample_ohlcv(100)
    benchmark = create_sample_ohlcv(100)
    signal_meta = {"technical_score": 7.5, "conviction": "HIGH"}

    features = pipeline.extract_features(
        signal_id="test",
        ohlcv_data=ohlcv,
        signal_metadata=signal_meta,
        benchmark_data=benchmark,
    )

    # Should extract market features using benchmark
    assert len(features) > 0
    assert "market_trend" in features or "correlation_regime" in features


def test_pipeline_create_feature_vector():
    """Test feature vector creation."""
    config = FeatureConfig(feature_set=FeatureSet.STANDARD)
    pipeline = FeaturePipeline(config)

    features = {"feature1": 1.0, "feature2": 2.0}

    fv = pipeline.create_feature_vector(
        signal_id="test-123",
        features=features,
        target=2.5,
    )

    assert isinstance(fv, FeatureVector)
    assert fv.signal_id == "test-123"
    assert fv.features == features
    assert fv.target == 2.5
    assert fv.target_binary == 1  # Positive target


def test_pipeline_create_feature_vector_negative_target():
    """Test feature vector with negative target."""
    config = FeatureConfig(feature_set=FeatureSet.STANDARD)
    pipeline = FeaturePipeline(config)

    features = {"feature1": 1.0}

    fv = pipeline.create_feature_vector(
        signal_id="test-123",
        features=features,
        target=-1.5,
    )

    assert fv.target == -1.5
    assert fv.target_binary == 0  # Negative target


def test_pipeline_create_feature_vector_no_target():
    """Test feature vector without target."""
    config = FeatureConfig(feature_set=FeatureSet.STANDARD)
    pipeline = FeaturePipeline(config)

    features = {"feature1": 1.0}

    fv = pipeline.create_feature_vector(
        signal_id="test-123",
        features=features,
    )

    assert fv.target is None
    assert fv.target_binary is None


def test_feature_scaler_standard():
    """Test feature scaling with standard method."""
    scaler = FeatureScaler(method="standard")

    train_data = {
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
    }

    scaler.fit(train_data)

    test_features = {"feature1": 3.0, "feature2": 30.0}
    transformed = scaler.transform(test_features)

    # Mean of both is middle value, so should be ~0
    assert abs(transformed["feature1"]) < 0.1
    assert abs(transformed["feature2"]) < 0.1


def test_feature_scaler_minmax():
    """Test feature scaling with minmax method."""
    scaler = FeatureScaler(method="minmax")

    train_data = {
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
    }

    scaler.fit(train_data)

    # Min should map to 0, max to 1
    transformed_min = scaler.transform({"feature1": 1.0})
    transformed_max = scaler.transform({"feature1": 5.0})

    assert abs(transformed_min["feature1"]) < 0.01
    assert abs(transformed_max["feature1"] - 1.0) < 0.01


def test_feature_scaler_robust():
    """Test feature scaling with robust method."""
    scaler = FeatureScaler(method="robust")

    train_data = {
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
    }

    scaler.fit(train_data)

    # Median should map to ~0
    transformed_median = scaler.transform({"feature1": 3.0})
    assert abs(transformed_median["feature1"]) < 0.1


def test_feature_scaler_fit_transform():
    """Test fit_transform method."""
    scaler = FeatureScaler(method="standard")

    train_data = {
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
    }

    result = scaler.fit_transform(train_data)

    assert len(result) == 5
    assert all(isinstance(d, dict) for d in result)
    assert all("feature1" in d and "feature2" in d for d in result)


def test_feature_scaler_unfitted():
    """Test scaler transform without fitting."""
    scaler = FeatureScaler(method="standard")

    features = {"feature1": 1.0, "feature2": 2.0}
    transformed = scaler.transform(features)

    # Should return unchanged if not fitted
    assert transformed == features


def test_feature_scaler_unknown_feature():
    """Test scaler with unknown feature."""
    scaler = FeatureScaler(method="standard")

    train_data = {"feature1": [1.0, 2.0, 3.0]}
    scaler.fit(train_data)

    # Unknown feature should pass through unchanged
    test_features = {"feature1": 1.0, "unknown": 5.0}
    transformed = scaler.transform(test_features)

    assert "unknown" in transformed
    assert transformed["unknown"] == 5.0


def test_feature_registry():
    """Test feature registry."""
    registry = FeatureRegistry()

    # Should have default extractors
    assert len(registry.get_all()) > 0

    # Should be able to get extractors by name
    trend = registry.get("trend_features")
    assert trend.name == "trend_features"

    # Should get all feature names
    all_names = registry.get_all_feature_names()
    assert len(all_names) > 0

    # Should get by category
    technical = registry.get_by_category("technical")
    assert len(technical) > 0


def test_feature_registry_register_custom():
    """Test registering custom extractor."""
    from trading_system.ml_refinement.features.extractors.base_extractor import BaseFeatureExtractor

    class CustomExtractor(BaseFeatureExtractor):
        @property
        def name(self) -> str:
            return "custom_extractor"

        @property
        def feature_names(self):
            return ["custom_feature"]

        def extract(self, data):
            return {"custom_feature": 1.0}

    registry = FeatureRegistry()
    custom = CustomExtractor()
    registry.register(custom)

    assert registry.get("custom_extractor") == custom
    assert "custom_feature" in registry.get_all_feature_names()


def test_pipeline_handles_missing_data():
    """Test pipeline handles missing/invalid data gracefully."""
    config = FeatureConfig(feature_set=FeatureSet.STANDARD)
    pipeline = FeaturePipeline(config)

    # Empty OHLCV
    ohlcv = pd.DataFrame(
        {
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        }
    )
    signal_meta = {}

    # Should handle gracefully
    features = pipeline.extract_features(
        signal_id="test",
        ohlcv_data=ohlcv,
        signal_metadata=signal_meta,
    )

    # Should return features (possibly zeros for failed extractors)
    assert isinstance(features, dict)


def test_pipeline_scaling():
    """Test pipeline applies scaling when configured."""
    config = FeatureConfig(feature_set=FeatureSet.STANDARD, scale_features=True)
    pipeline = FeaturePipeline(config)

    ohlcv = create_sample_ohlcv(100)
    signal_meta = {"technical_score": 7.5, "conviction": "HIGH"}

    features = pipeline.extract_features(
        signal_id="test",
        ohlcv_data=ohlcv,
        signal_metadata=signal_meta,
    )

    # All values should be clipped to reasonable range
    assert all(-10.0 <= v <= 10.0 for v in features.values())


def test_pipeline_no_scaling():
    """Test pipeline without scaling."""
    config = FeatureConfig(feature_set=FeatureSet.STANDARD, scale_features=False)
    pipeline = FeaturePipeline(config)

    ohlcv = create_sample_ohlcv(100)
    signal_meta = {"technical_score": 7.5, "conviction": "HIGH"}

    features = pipeline.extract_features(
        signal_id="test",
        ohlcv_data=ohlcv,
        signal_metadata=signal_meta,
    )

    # Values may be outside -10 to 10 range if not scaled
    assert len(features) > 0


def test_pipeline_get_feature_names():
    """Test getting feature names from pipeline."""
    config = FeatureConfig(feature_set=FeatureSet.STANDARD)
    pipeline = FeaturePipeline(config)

    feature_names = pipeline.get_feature_names()
    assert len(feature_names) > 0
    assert all(isinstance(name, str) for name in feature_names)

    feature_count = pipeline.get_feature_count()
    assert feature_count == len(feature_names)

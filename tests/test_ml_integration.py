"""Integration tests for ML refinement."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from trading_system.ml_refinement.config import (
    FeatureConfig,
    FeatureSet,
    MLConfig,
    ModelType,
    FeatureVector,
)
from trading_system.ml_refinement.storage.feature_db import FeatureDatabase
from trading_system.ml_refinement.features.pipeline import FeaturePipeline
from trading_system.ml_refinement.features.extractors.technical_features import (
    TrendFeatures,
    MomentumFeatures,
)
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

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000000, 10000000, 300),
        },
        index=dates,
    )


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
    # Create sample feature vectors with targets
    np.random.seed(42)

    for i in range(200):
        fv = FeatureVector(
            signal_id=f"sig-{i:04d}",
            timestamp=(datetime.now() - timedelta(days=200 - i)).isoformat(),
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

        # Check that features are extracted
        assert len(features) > 0
        assert all(isinstance(v, float) for v in features.values())
        # Check for common trend features
        feature_names = list(features.keys())
        assert any("ma" in name.lower() or "trend" in name.lower() for name in feature_names)

    def test_momentum_features(self, sample_ohlcv):
        """Test momentum feature extraction."""
        extractor = MomentumFeatures()
        features = extractor.extract(sample_ohlcv)

        # Check that features are extracted
        assert len(features) > 0
        assert all(isinstance(v, float) for v in features.values())
        # Check for momentum-related features
        feature_names = list(features.keys())
        assert any("rsi" in name.lower() or "momentum" in name.lower() for name in feature_names)

    def test_feature_pipeline(self, sample_ohlcv):
        """Test complete feature pipeline."""
        config = FeatureConfig(feature_set=FeatureSet.STANDARD)
        pipeline = FeaturePipeline(config)

        signal_metadata = {
            "symbol": "TEST",
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

        assert len(features) > 0
        assert all(isinstance(v, float) for v in features.values())


class TestFeatureDatabase:
    """Test feature database operations."""

    def test_store_and_retrieve(self, feature_db):
        """Test feature storage roundtrip."""
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

    def test_trainer_pipeline(self, populated_feature_db, tmp_path):
        """Test complete training pipeline."""
        config = MLConfig()
        config.training.min_training_samples = 30  # Lower for testing
        config.training.min_validation_samples = 10

        trainer = ModelTrainer(
            config,
            populated_feature_db,
            str(tmp_path / "models"),
        )

        # Mock the model class to use a simple mock
        with patch.object(trainer, "MODEL_CLASSES", {ModelType.SIGNAL_QUALITY: MagicMock}):
            # Create a proper mock model
            mock_model = MagicMock()
            mock_model.model_id = "test-model-123"
            mock_model.get_top_features.return_value = [("feature_1", 0.5), ("feature_2", 0.3)]
            mock_model.get_metadata.return_value = MagicMock(
                model_id="test-model-123",
                model_type=ModelType.SIGNAL_QUALITY,
                version="1.0",
                created_at=datetime.now().isoformat(),
                train_start_date="2023-01-01",
                train_end_date="2023-12-31",
                train_samples=200,
                validation_samples=50,
                train_metrics={},
                validation_metrics={},
            )
            mock_model.predict.return_value = np.array([1, 0, 1])
            mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2], [0.4, 0.6]])

            trainer.MODEL_CLASSES[ModelType.SIGNAL_QUALITY] = lambda **kwargs: mock_model

            result = trainer.train(
                model_type=ModelType.SIGNAL_QUALITY,
            )

            assert result.success
            assert result.model_id
            assert result.train_samples > 0


class TestPredictionService:
    """Test prediction service."""

    def test_predict_signal_quality(
        self,
        populated_feature_db,
        sample_ohlcv,
        tmp_path,
    ):
        """Test signal quality prediction."""
        config = MLConfig()
        model_dir = str(tmp_path / "models")

        # Mock model for prediction
        mock_model = MagicMock()
        mock_model.model_id = "test-model-123"
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])  # 70% probability

        # Create service with mocked model registry
        from trading_system.ml_refinement.integration.prediction_service import ModelRegistry

        class MockModelRegistry:
            def __init__(self, feature_db):
                self.feature_db = feature_db
                self._models = {}

            def get_active(self, model_type):
                return mock_model

        registry = MockModelRegistry(populated_feature_db)
        service = PredictionService(config, populated_feature_db, registry)

        quality = service.predict_signal_quality(
            signal_id="new-signal",
            ohlcv_data=sample_ohlcv,
            signal_metadata={
                "symbol": "TEST",
                "asset_class": "equity",
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
        config.enabled = True
        config.use_ml_scores = True

        # Mock prediction service
        mock_service = MagicMock()
        mock_service.predict_signal_quality.return_value = 0.75  # 75% quality

        scorer = MLSignalScorer(config, mock_service)

        # Score a signal
        enhanced = scorer.score_signal(
            signal_id="test-signal",
            technical_score=7.5,
            news_score=6.0,
            ohlcv_data=sample_ohlcv,
            signal_metadata={
                "symbol": "TEST",
                "asset_class": "equity",
                "conviction": "HIGH",
                "entry_price": 100,
                "target_price": 110,
                "stop_price": 95,
            },
        )

        assert enhanced.ml_enabled
        assert 0 <= enhanced.ml_quality_score <= 1
        assert 0 <= enhanced.combined_score <= 10
        assert enhanced.combined_score > 0  # Should have a score

    def test_score_signal_no_ml(self, sample_ohlcv):
        """Test signal scoring without ML."""
        config = MLConfig()
        config.enabled = False

        mock_service = MagicMock()
        scorer = MLSignalScorer(config, mock_service)

        enhanced = scorer.score_signal(
            signal_id="test-signal",
            technical_score=7.5,
            news_score=6.0,
            ohlcv_data=sample_ohlcv,
            signal_metadata={
                "symbol": "TEST",
                "asset_class": "equity",
            },
        )

        assert not enhanced.ml_enabled
        assert enhanced.combined_score > 0  # Should still have a score from technical/news

    def test_filter_by_ml_quality(self, sample_ohlcv):
        """Test filtering signals by ML quality."""
        config = MLConfig()
        config.enabled = True
        config.use_ml_scores = True
        config.quality_threshold_low = 0.5

        mock_service = MagicMock()
        scorer = MLSignalScorer(config, mock_service)

        # Create multiple scored signals
        signals = []
        for i, quality in enumerate([0.3, 0.6, 0.4, 0.8]):
            mock_service.predict_signal_quality.return_value = quality
            enhanced = scorer.score_signal(
                signal_id=f"sig-{i}",
                technical_score=7.5,
                news_score=None,
                ohlcv_data=sample_ohlcv,
                signal_metadata={"symbol": f"TEST{i}"},
            )
            signals.append(enhanced)

        # Filter
        filtered = scorer.filter_by_ml_quality(signals, min_quality=0.5)

        # Should keep signals with quality >= 0.5
        assert len(filtered) >= 2  # At least sig-1 (0.6) and sig-3 (0.8)
        assert all(s.ml_quality_score >= 0.5 for s in filtered if s.ml_enabled)


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow."""

    def test_full_ml_pipeline(self, populated_feature_db, sample_ohlcv, tmp_path):
        """Test complete ML pipeline from training to prediction."""
        config = MLConfig()
        config.training.min_training_samples = 30
        config.training.min_validation_samples = 10
        config.enabled = True
        config.use_ml_scores = True

        model_dir = str(tmp_path / "models")

        # Step 1: Train model
        trainer = ModelTrainer(config, populated_feature_db, model_dir)

        with patch.object(trainer, "MODEL_CLASSES", {ModelType.SIGNAL_QUALITY: MagicMock}):
            mock_model = MagicMock()
            mock_model.model_id = "e2e-model-123"
            mock_model.get_top_features.return_value = []
            mock_model.get_metadata.return_value = MagicMock(
                model_id="e2e-model-123",
                model_type=ModelType.SIGNAL_QUALITY,
                version="1.0",
                created_at=datetime.now().isoformat(),
                train_start_date="2023-01-01",
                train_end_date="2023-12-31",
                train_samples=200,
                validation_samples=50,
            )
            mock_model.predict.return_value = np.array([1])
            mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

            trainer.MODEL_CLASSES[ModelType.SIGNAL_QUALITY] = lambda **kwargs: mock_model

            result = trainer.train(ModelType.SIGNAL_QUALITY)
            assert result.success

        # Step 2: Activate model (simulated)
        populated_feature_db.activate_model("e2e-model-123")

        # Step 3: Create prediction service
        from trading_system.ml_refinement.integration.prediction_service import ModelRegistry

        class MockRegistry:
            def __init__(self, feature_db):
                self.feature_db = feature_db

            def get_active(self, model_type):
                return mock_model

        registry = MockRegistry(populated_feature_db)
        service = PredictionService(config, populated_feature_db, registry)

        # Step 4: Make prediction
        quality = service.predict_signal_quality(
            signal_id="e2e-signal",
            ohlcv_data=sample_ohlcv,
            signal_metadata={
                "symbol": "TEST",
                "asset_class": "equity",
                "signal_type": "breakout",
            },
        )

        assert 0 <= quality <= 1

        # Step 5: Score signal
        scorer = MLSignalScorer(config, service)
        enhanced = scorer.score_signal(
            signal_id="e2e-signal",
            technical_score=7.5,
            news_score=6.0,
            ohlcv_data=sample_ohlcv,
            signal_metadata={
                "symbol": "TEST",
                "asset_class": "equity",
            },
        )

        assert enhanced.ml_enabled
        assert enhanced.combined_score > 0

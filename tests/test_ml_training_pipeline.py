"""Tests for ML training pipeline."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from trading_system.ml_refinement.config import MLConfig, ModelType, TrainingConfig
from trading_system.ml_refinement.storage.feature_db import FeatureDatabase
from trading_system.ml_refinement.training.hyperparameter_tuner import HyperparameterTuner
from trading_system.ml_refinement.training.trainer import ModelTrainer


class MockModel:
    """Mock model for testing."""

    def __init__(self, **kwargs):
        self.model_id = f"mock_model_{np.random.randint(10000, 99999)}"
        self.feature_names = []
        self.feature_importance = {}
        self._is_trained = False
        self._fit_called = False

    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        """Mock fit method."""
        self._fit_called = True
        self._is_trained = True
        if feature_names:
            self.feature_names = feature_names
        return {"accuracy": 0.75}

    def predict(self, X):
        """Mock predict method."""
        if not self._is_trained:
            raise ValueError("Model not trained")
        # Return binary predictions
        return (np.random.rand(len(X)) > 0.5).astype(int)

    def predict_proba(self, X):
        """Mock predict_proba method."""
        if not self._is_trained:
            raise ValueError("Model not trained")
        # Return 2D array of probabilities
        probs = np.random.rand(len(X), 2)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    def get_top_features(self, n=10):
        """Mock get top features."""
        return [("feature_1", 0.5), ("feature_2", 0.3)]

    def get_metadata(self):
        """Mock get metadata."""
        from datetime import datetime

        from trading_system.ml_refinement.config import ModelMetadata

        return ModelMetadata(
            model_id=self.model_id,
            model_type=ModelType.SIGNAL_QUALITY,
            version="1.0",
            created_at=datetime.now().isoformat(),
            train_start_date="2023-01-01",
            train_end_date="2024-01-01",
            train_samples=100,
            validation_samples=20,
        )

    def save(self, path):
        """Mock save method."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Create empty file to simulate saving
        Path(path).touch()


@pytest.fixture
def temp_db(tmp_path):
    """Create temporary feature database with test data."""
    db_path = tmp_path / "test_features.db"
    db = FeatureDatabase(str(db_path))
    db.initialize()

    # Add training data
    from trading_system.ml_refinement.config import FeatureVector

    for i in range(300):  # Enough for walk-forward validation
        fv = FeatureVector(
            signal_id=f"signal-{i}",
            timestamp=f"2023-01-{i % 30 + 1:02d}T10:00:00",
            features={
                "rsi": 50.0 + np.random.randn() * 10,
                "atr_ratio": 1.0 + np.random.randn() * 0.2,
                "momentum": np.random.randn() * 0.1,
            },
            target=np.random.randn() * 2.0,  # Mix of positive and negative
        )
        db.store_feature_vector(fv, symbol="AAPL")

    yield db
    db.close()


@pytest.fixture
def ml_config():
    """Create ML config for testing."""
    config = MLConfig()
    config.training = TrainingConfig(
        train_window_days=50,  # Small for testing
        validation_window_days=20,
        step_size_days=10,
        min_training_samples=30,
        min_validation_samples=10,
    )
    return config


@pytest.fixture
def trainer(ml_config, temp_db, tmp_path):
    """Create trainer instance."""
    model_dir = tmp_path / "models"
    return ModelTrainer(
        config=ml_config,
        feature_db=temp_db,
        model_dir=str(model_dir),
    )


class TestModelTrainer:
    """Test ModelTrainer class."""

    def test_trainer_initialization(self, trainer):
        """Test trainer initializes correctly."""
        assert trainer.config is not None
        assert trainer.feature_db is not None
        assert trainer.model_dir.exists()

    def test_train_complete_pipeline(self, trainer, temp_db):
        """Test: Trainer runs complete pipeline."""
        # Patch the model class to use MockModel
        with patch.object(trainer, "MODEL_CLASSES", {ModelType.SIGNAL_QUALITY: MockModel}):
            result = trainer.train(
                model_type=ModelType.SIGNAL_QUALITY,
                start_date="2023-01-01",
                end_date="2023-12-31",
            )

            # Verify complete pipeline ran
            assert result.run_id != ""
            assert result.success, f"Training failed: {result.error_message}"
            assert result.train_samples > 0
            assert result.n_features > 0
            assert result.total_time_seconds > 0

    def test_walk_forward_cv_integration(self, trainer, temp_db):
        """Test: Walk-forward CV integrated correctly."""
        with patch.object(trainer, "MODEL_CLASSES", {ModelType.SIGNAL_QUALITY: MockModel}):
            result = trainer.train(
                model_type=ModelType.SIGNAL_QUALITY,
                start_date="2023-01-01",
                end_date="2023-12-31",
            )

            # Verify walk-forward CV ran
            assert result.n_folds > 0, "Should have multiple folds"
            assert result.val_samples > 0, "Should have validation samples"
            assert "auc" in result.cv_metrics or "accuracy" in result.cv_metrics, "Should have CV metrics"
            assert result.cv_metrics, "CV metrics should not be empty"

    def test_final_model_trained(self, trainer, temp_db):
        """Test: Final model trained on all data."""
        with patch.object(trainer, "MODEL_CLASSES", {ModelType.SIGNAL_QUALITY: MockModel}):
            result = trainer.train(
                model_type=ModelType.SIGNAL_QUALITY,
                start_date="2023-01-01",
                end_date="2023-12-31",
            )

            # Verify final model was trained
            assert result.model_id != "", "Should have model ID"
            assert result.final_metrics, "Should have final metrics"
            assert result.top_features, "Should have top features"

    def test_model_saved_and_registered(self, trainer, temp_db):
        """Test: Model saved and registered."""
        with patch.object(trainer, "MODEL_CLASSES", {ModelType.SIGNAL_QUALITY: MockModel}):
            result = trainer.train(
                model_type=ModelType.SIGNAL_QUALITY,
                start_date="2023-01-01",
                end_date="2023-12-31",
            )

            # Verify model was saved
            model_path = trainer.model_dir / f"{result.model_id}.pkl"
            assert model_path.exists(), f"Model file should exist at {model_path}"

            # Verify model was registered in database
            # Check model history to see if it's registered
            history = temp_db.get_model_history(ModelType.SIGNAL_QUALITY.value, limit=10)
            model_ids = [m.model_id for m in history]
            assert result.model_id in model_ids, "Model should be registered in database"

    def test_insufficient_samples(self, trainer, temp_db):
        """Test handling of insufficient samples."""
        # Create config with very high minimum
        trainer.config.training.min_training_samples = 10000

        result = trainer.train(
            model_type=ModelType.SIGNAL_QUALITY,
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        assert not result.success
        assert "insufficient" in result.error_message.lower()

    def test_retrain_logic_checks_new_samples(self, trainer, temp_db):
        """Test: Retrain logic checks for new samples."""
        # First, register an active model
        from datetime import datetime

        from trading_system.ml_refinement.config import ModelMetadata

        # The temp_db fixture creates samples with timestamps in January 2023
        # Set train_end_date to before the samples so they count as "new"
        active_model = ModelMetadata(
            model_id="test-model-1",
            model_type=ModelType.SIGNAL_QUALITY,
            version="1.0",
            created_at=datetime.now().isoformat(),
            train_start_date="2022-01-01",
            train_end_date="2022-12-31",  # Before the temp_db samples (Jan 2023)
            train_samples=100,
            validation_samples=20,
        )
        temp_db.register_model(active_model)
        temp_db.activate_model("test-model-1")

        # Test retrain with enough new samples
        with patch.object(trainer, "MODEL_CLASSES", {ModelType.SIGNAL_QUALITY: MockModel}):
            result = trainer.retrain_if_needed(
                model_type=ModelType.SIGNAL_QUALITY,
                min_new_samples=10,
            )

            # Should retrain because we have 300 samples total, more than needed
            assert result is not None, "Should retrain when enough new samples available"
            assert result.success, "Retraining should succeed"

        # Test retrain with not enough new samples
        trainer.config.training.min_training_samples = 10000  # Set high threshold
        result = trainer.retrain_if_needed(
            model_type=ModelType.SIGNAL_QUALITY,
            min_new_samples=1000,  # Very high threshold
        )

        # Should not retrain
        assert result is None or not result.success

    def test_retrain_no_active_model(self, trainer, temp_db):
        """Test retrain when no active model exists."""
        with patch.object(trainer, "MODEL_CLASSES", {ModelType.SIGNAL_QUALITY: MockModel}):
            result = trainer.retrain_if_needed(
                model_type=ModelType.SIGNAL_QUALITY,
                min_new_samples=10,
            )

            # Should train new model
            assert result is not None
            assert result.success


class TestHyperparameterTuner:
    """Test HyperparameterTuner class."""

    def test_grid_search(self, temp_db):
        """Test: Hyperparameter tuning works - grid search."""
        from trading_system.ml_refinement.validation.walk_forward import WalkForwardValidator

        # Get training data
        X, y, feature_names = temp_db.get_training_data(
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        if len(X) < 50:
            pytest.skip("Not enough data for hyperparameter tuning")

        # Use custom CV with lower minimums for the test data size
        cv = WalkForwardValidator(
            train_window=50,
            val_window=20,
            step_size=10,
            min_train_samples=50,
            min_val_samples=20,
        )

        tuner = HyperparameterTuner(
            model_class=MockModel,
            cv=cv,
        )

        # Small grid for testing
        param_grid = {
            "param1": [1, 2],
            "param2": [0.1, 0.2],
        }

        result = tuner.grid_search(
            X,
            y,
            feature_names,
            param_grid=param_grid,
            scoring="accuracy",
        )

        assert result.n_trials == 4  # 2 * 2 combinations
        assert result.best_params is not None
        assert result.best_score >= 0
        assert len(result.all_results) == 4

    def test_random_search(self, temp_db):
        """Test: Hyperparameter tuning works - random search."""
        from trading_system.ml_refinement.validation.walk_forward import WalkForwardValidator

        # Get training data
        X, y, feature_names = temp_db.get_training_data(
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        if len(X) < 50:
            pytest.skip("Not enough data for hyperparameter tuning")

        # Use custom CV with lower minimums for the test data size
        cv = WalkForwardValidator(
            train_window=50,
            val_window=20,
            step_size=10,
            min_train_samples=50,
            min_val_samples=20,
        )

        tuner = HyperparameterTuner(
            model_class=MockModel,
            cv=cv,
        )

        param_distributions = {
            "param1": [1, 2, 3],
            "param2": (0.1, 0.5),  # Uniform distribution
        }

        result = tuner.random_search(
            X,
            y,
            feature_names,
            param_distributions=param_distributions,
            n_trials=5,
            scoring="accuracy",
        )

        assert result.n_trials == 5
        assert result.best_params is not None
        assert result.best_score >= 0
        assert len(result.all_results) == 5

    def test_hyperparameter_evaluation(self, temp_db):
        """Test parameter evaluation with walk-forward CV."""
        X, y, feature_names = temp_db.get_training_data(
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        if len(X) < 50:
            pytest.skip("Not enough data")

        from trading_system.ml_refinement.validation.walk_forward import WalkForwardValidator

        cv = WalkForwardValidator(
            train_window=30,
            val_window=10,
            step_size=10,
            min_train_samples=30,  # Match train_window
            min_val_samples=10,  # Match val_window
        )

        tuner = HyperparameterTuner(
            model_class=MockModel,
            cv=cv,
        )

        scores = tuner._evaluate_params(
            X,
            y,
            feature_names,
            params={"param1": 1},
            scoring="accuracy",
        )

        assert len(scores) > 0, "Should have scores from multiple folds"
        assert all(0 <= s <= 1 for s in scores), "Scores should be between 0 and 1"


class TestTrainingIntegration:
    """Integration tests for training pipeline."""

    def test_end_to_end_training(self, trainer, temp_db):
        """Test complete end-to-end training workflow."""
        with patch.object(trainer, "MODEL_CLASSES", {ModelType.SIGNAL_QUALITY: MockModel}):
            # Train model
            result = trainer.train(
                model_type=ModelType.SIGNAL_QUALITY,
                start_date="2023-01-01",
                end_date="2023-12-31",
            )

            # Verify all components worked
            assert result.success
            assert result.model_id != ""
            assert result.n_folds > 0
            assert result.cv_metrics
            assert result.final_metrics
            assert result.top_features

            # Verify model file exists
            model_file = trainer.model_dir / f"{result.model_id}.pkl"
            assert model_file.exists()

            # Verify model in database
            history = temp_db.get_model_history(ModelType.SIGNAL_QUALITY.value, limit=10)
            model_ids = [m.model_id for m in history]
            assert result.model_id in model_ids

    def test_training_with_hyperparameters(self, trainer, temp_db):
        """Test training with custom hyperparameters."""
        hyperparameters = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
        }

        with patch.object(trainer, "MODEL_CLASSES", {ModelType.SIGNAL_QUALITY: MockModel}):
            result = trainer.train(
                model_type=ModelType.SIGNAL_QUALITY,
                start_date="2023-01-01",
                end_date="2023-12-31",
                hyperparameters=hyperparameters,
            )

            assert result.success
            assert result.model_id != ""

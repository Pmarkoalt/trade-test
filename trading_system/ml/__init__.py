"""Machine Learning integration for trading system.

This module provides ML model training, prediction, and integration
with the trading system's signal generation and scoring pipeline.

Modules:
    models: ML model wrappers and interfaces
    training: Training pipeline for ML models
    feature_engineering: Feature engineering for ML models
    predictor: Model prediction integration with signal generation
    versioning: Model versioning and management
    ensemble: Ensemble models (voting, stacking, boosting)
    online_learning: Incremental learning and concept drift detection
    feature_store: Feature caching, versioning, and validation
"""

from trading_system.ml.ensemble import (
    BoostingEnsemble,
    EnsembleModel,
    StackingEnsemble,
    VotingEnsemble,
)
from trading_system.ml.evaluation import (
    ModelPerformanceTracker,
    compute_backtest_metrics,
    compute_confidence_intervals,
    compute_feature_importance,
)
from trading_system.ml.feature_engineering import MLFeatureEngineer
from trading_system.ml.feature_store import (
    FeatureStore,
    FeatureVersionManager,
)
from trading_system.ml.models import (
    MLModel,
    ModelType,
    hyperparameter_tuning,
    random_search_tuning,
)
from trading_system.ml.online_learning import (
    ConceptDriftDetector,
    IncrementalLearner,
    ModelRetrainingPipeline,
)
from trading_system.ml.online_learning import ModelVersionManager as OnlineModelVersionManager
from trading_system.ml.predictor import MLPredictor
from trading_system.ml.training import MLTrainer
from trading_system.ml.versioning import ModelVersionManager

__all__ = [
    "MLModel",
    "ModelType",
    "MLPredictor",
    "MLTrainer",
    "MLFeatureEngineer",
    "ModelVersionManager",
    "EnsembleModel",
    "VotingEnsemble",
    "StackingEnsemble",
    "BoostingEnsemble",
    "ConceptDriftDetector",
    "IncrementalLearner",
    "ModelRetrainingPipeline",
    "OnlineModelVersionManager",
    "FeatureStore",
    "FeatureVersionManager",
    "compute_backtest_metrics",
    "compute_feature_importance",
    "compute_confidence_intervals",
    "ModelPerformanceTracker",
    "hyperparameter_tuning",
    "random_search_tuning",
]

"""Machine Learning integration for trading system.

This module provides ML model training, prediction, and integration
with the trading system's signal generation and scoring pipeline.

Modules:
    models: ML model wrappers and interfaces
    training: Training pipeline for ML models
    feature_engineering: Feature engineering for ML models
    predictor: Model prediction integration with signal generation
    versioning: Model versioning and management
"""

from trading_system.ml.models import MLModel, ModelType
from trading_system.ml.predictor import MLPredictor
from trading_system.ml.training import MLTrainer
from trading_system.ml.feature_engineering import MLFeatureEngineer
from trading_system.ml.versioning import ModelVersionManager

__all__ = [
    "MLModel",
    "ModelType",
    "MLPredictor",
    "MLTrainer",
    "MLFeatureEngineer",
    "ModelVersionManager",
]


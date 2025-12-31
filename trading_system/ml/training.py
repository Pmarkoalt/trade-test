"""ML model training pipeline."""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import time
from trading_system.models.features import FeatureRow
from trading_system.ml.models import MLModel, ModelType, ModelMetadata
from trading_system.ml.feature_engineering import MLFeatureEngineer


class MLTrainer:
    """Training pipeline for ML models.
    
    This class handles the full training workflow including:
    - Data preparation from FeatureRow objects
    - Feature engineering
    - Model training with validation
    - Performance evaluation
    - Model serialization
    """
    
    def __init__(
        self,
        model_type: ModelType,
        feature_engineer: Optional[MLFeatureEngineer] = None,
        hyperparameters: Optional[Dict] = None,
    ):
        """Initialize trainer.
        
        Args:
            model_type: Type of model to train
            feature_engineer: Feature engineer instance (will create default if None)
            hyperparameters: Model hyperparameters
        """
        self.model_type = model_type
        self.feature_engineer = feature_engineer or MLFeatureEngineer()
        self.hyperparameters = hyperparameters or {}
        self.model: Optional[MLModel] = None
    
    def prepare_data(
        self,
        feature_rows: List[FeatureRow],
        target_values: List[float],
        train_split: float = 0.7,
        val_split: float = 0.15,
        shuffle: bool = False,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Prepare training, validation, and test data.
        
        Args:
            feature_rows: List of FeatureRow objects
            target_values: List of target values (e.g., future returns, signal quality)
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            shuffle: Whether to shuffle data before splitting
        
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test) DataFrames/Series
        """
        if len(feature_rows) != len(target_values):
            raise ValueError(f"Feature rows ({len(feature_rows)}) and target values ({len(target_values)}) must have same length")
        
        # Fit feature engineer on all data
        self.feature_engineer.fit(feature_rows)
        
        # Transform features
        X = self.feature_engineer.transform_batch(feature_rows)
        y = pd.Series(target_values, index=X.index)
        
        # Shuffle if requested
        if shuffle:
            indices = np.random.permutation(len(X))
            X = X.iloc[indices].reset_index(drop=True)
            y = y.iloc[indices].reset_index(drop=True)
        
        # Split data
        n_total = len(X)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        X_train = X.iloc[:n_train].copy()
        y_train = y.iloc[:n_train].copy()
        X_val = X.iloc[n_train:n_train + n_val].copy()
        y_val = y.iloc[n_train:n_train + n_val].copy()
        X_test = X.iloc[n_train + n_val:].copy()
        y_test = y.iloc[n_train + n_val:].copy()
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def train(
        self,
        feature_rows: List[FeatureRow],
        target_values: List[float],
        validation_data: Optional[Tuple[List[FeatureRow], List[float]]] = None,
        model_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """Train ML model.
        
        Args:
            feature_rows: List of FeatureRow objects for training
            target_values: List of target values
            validation_data: Optional (feature_rows_val, target_values_val) tuple
            model_id: Optional model identifier
        
        Returns:
            Dictionary of training metrics
        """
        start_time = time.time()
        
        # Fit feature engineer
        self.feature_engineer.fit(feature_rows)
        
        # Transform features
        X_train = self.feature_engineer.transform_batch(feature_rows)
        y_train = pd.Series(target_values, index=X_train.index)
        
        # Prepare validation data if provided
        val_tuple = None
        if validation_data:
            feature_rows_val, target_values_val = validation_data
            X_val = self.feature_engineer.transform_batch(feature_rows_val)
            y_val = pd.Series(target_values_val, index=X_val.index)
            val_tuple = (X_val, y_val)
        
        # Create and train model
        self.model = MLModel._create_model_instance(
            model_type=self.model_type,
            hyperparameters=self.hyperparameters,
        )
        if model_id:
            self.model.model_id = model_id
        
        # Train model
        training_metrics = self.model.train(X_train, y_train, validation_data=val_tuple)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=self.model.model_id,
            model_type=self.model_type,
            version="1.0.0",
            training_date=pd.Timestamp.now(),
            training_samples=len(feature_rows),
            feature_names=self.feature_engineer.get_feature_names(),
            target_name="target",
            hyperparameters=self.hyperparameters,
            performance_metrics=training_metrics,
            training_time_seconds=time.time() - start_time,
        )
        self.model.set_metadata(metadata)
        
        return training_metrics
    
    def evaluate(
        self,
        feature_rows: List[FeatureRow],
        target_values: List[float],
    ) -> Dict[str, float]:
        """Evaluate trained model on test data.
        
        Args:
            feature_rows: List of FeatureRow objects for testing
            target_values: List of target values
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error, r2_score,
            accuracy_score, precision_score, recall_score, f1_score,
        )
        
        # Transform features
        X_test = self.feature_engineer.transform_batch(feature_rows)
        y_test = pd.Series(target_values, index=X_test.index)
        
        # Make predictions
        predictions = self.model.predict(X_test)
        
        # Compute metrics based on task type
        if len(np.unique(y_test)) <= 10:
            # Classification metrics
            metrics = {
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
                "recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
                "f1": f1_score(y_test, predictions, average="weighted", zero_division=0),
            }
        else:
            # Regression metrics
            metrics = {
                "mse": mean_squared_error(y_test, predictions),
                "mae": mean_absolute_error(y_test, predictions),
                "r2": r2_score(y_test, predictions),
                "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
            }
        
        return metrics
    
    def save_model(self, directory: Path) -> None:
        """Save trained model to directory.
        
        Args:
            directory: Directory to save model files
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save(directory)
        
        # Also save feature engineer
        import pickle
        feature_engineer_path = directory / "feature_engineer.pkl"
        with open(feature_engineer_path, "wb") as f:
            pickle.dump(self.feature_engineer, f)
    
    def load_model(self, directory: Path) -> None:
        """Load trained model from directory.
        
        Args:
            directory: Directory containing model files
        """
        self.model = MLModel.load(directory)
        
        # Load feature engineer
        import pickle
        feature_engineer_path = directory / "feature_engineer.pkl"
        if feature_engineer_path.exists():
            with open(feature_engineer_path, "rb") as f:
                self.feature_engineer = pickle.load(f)
    
    def get_model(self) -> Optional[MLModel]:
        """Get trained model.
        
        Returns:
            Trained MLModel instance, or None if not trained
        """
        return self.model


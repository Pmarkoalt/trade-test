"""ML model prediction integration with signal generation."""

from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np
from trading_system.models.features import FeatureRow
from trading_system.models.signals import Signal
from trading_system.ml.models import MLModel
from trading_system.ml.feature_engineering import MLFeatureEngineer


class MLPredictor:
    """Integrate ML model predictions into signal generation and scoring.
    
    This class provides a bridge between trained ML models and the trading
    system's signal generation pipeline. It can:
    - Generate predictions from FeatureRow objects
    - Enhance signal scores with ML predictions
    - Filter signals based on ML confidence
    """
    
    def __init__(
        self,
        model: MLModel,
        feature_engineer: MLFeatureEngineer,
        prediction_mode: str = "score_enhancement",
        confidence_threshold: float = 0.5,
    ):
        """Initialize predictor.
        
        Args:
            model: Trained MLModel instance
            feature_engineer: FeatureEngineer instance (must be fitted)
            prediction_mode: How to use predictions
                - "score_enhancement": Add ML prediction as weighted component to signal score
                - "filter": Filter signals below confidence threshold
                - "replace": Replace signal score with ML prediction
            confidence_threshold: Minimum confidence for filtering (0-1)
        """
        self.model = model
        self.feature_engineer = feature_engineer
        self.prediction_mode = prediction_mode
        self.confidence_threshold = confidence_threshold
        
        if prediction_mode not in ["score_enhancement", "filter", "replace"]:
            raise ValueError(f"Invalid prediction_mode: {prediction_mode}")
    
    def predict_single(self, feature_row: FeatureRow) -> float:
        """Make prediction for a single FeatureRow.
        
        Args:
            feature_row: FeatureRow to predict
        
        Returns:
            Prediction value (signal quality score, expected return, etc.)
        """
        # Transform to features
        feature_vector = self.feature_engineer.transform(feature_row)
        feature_df = pd.DataFrame([feature_vector])
        
        # Make prediction
        prediction = self.model.predict(feature_df)[0]
        
        return float(prediction)
    
    def predict_batch(self, feature_rows: List[FeatureRow]) -> np.ndarray:
        """Make predictions for multiple FeatureRows.
        
        Args:
            feature_rows: List of FeatureRow objects
        
        Returns:
            Array of predictions
        """
        if not feature_rows:
            return np.array([])
        
        # Transform to feature matrix
        feature_df = self.feature_engineer.transform_batch(feature_rows)
        
        # Make predictions
        predictions = self.model.predict(feature_df)
        
        return predictions
    
    def predict_proba_batch(self, feature_rows: List[FeatureRow]) -> Optional[np.ndarray]:
        """Make probability predictions for multiple FeatureRows (if supported).
        
        Args:
            feature_rows: List of FeatureRow objects
        
        Returns:
            Array of probability predictions, or None if not supported
        """
        if not feature_rows:
            return None
        
        # Transform to feature matrix
        feature_df = self.feature_engineer.transform_batch(feature_rows)
        
        # Make probability predictions
        proba = self.model.predict_proba(feature_df)
        
        return proba
    
    def enhance_signal_scores(
        self,
        signals: List[Signal],
        get_features: Callable[[Signal], Optional[FeatureRow]],
        ml_weight: float = 0.3,
    ) -> None:
        """Enhance signal scores with ML predictions.
        
        Modifies signal scores in-place based on ML predictions.
        
        Args:
            signals: List of signals to enhance
            get_features: Function to get FeatureRow for a signal
            ml_weight: Weight for ML prediction (0-1), rest goes to original score
        """
        if not signals:
            return
        
        # Collect feature rows
        feature_rows = []
        valid_indices = []
        
        for i, signal in enumerate(signals):
            features = get_features(signal)
            if features is not None:
                feature_rows.append(features)
                valid_indices.append(i)
        
        if not feature_rows:
            return
        
        # Get ML predictions
        ml_predictions = self.predict_batch(feature_rows)
        
        # Normalize ML predictions to [0, 1] range
        if len(ml_predictions) > 1:
            min_pred = ml_predictions.min()
            max_pred = ml_predictions.max()
            if max_pred - min_pred > 1e-10:
                ml_scores = (ml_predictions - min_pred) / (max_pred - min_pred)
            else:
                ml_scores = np.full_like(ml_predictions, 0.5)
        else:
            ml_scores = np.array([0.5])
        
        # Update signal scores
        if self.prediction_mode == "score_enhancement":
            # Weighted combination of original score and ML prediction
            for idx, signal_idx in enumerate(valid_indices):
                original_score = signals[signal_idx].score
                ml_score = ml_scores[idx]
                signals[signal_idx].score = (1 - ml_weight) * original_score + ml_weight * ml_score
        
        elif self.prediction_mode == "replace":
            # Replace score with ML prediction
            for idx, signal_idx in enumerate(valid_indices):
                signals[signal_idx].score = ml_scores[idx]
        
        elif self.prediction_mode == "filter":
            # Filter signals below threshold
            for idx, signal_idx in enumerate(valid_indices):
                if ml_scores[idx] < self.confidence_threshold:
                    # Mark signal as filtered (set score to 0)
                    signals[signal_idx].score = 0.0
    
    def filter_signals(
        self,
        signals: List[Signal],
        get_features: Callable[[Signal], Optional[FeatureRow]],
    ) -> List[Signal]:
        """Filter signals based on ML predictions.
        
        Args:
            signals: List of signals to filter
            get_features: Function to get FeatureRow for a signal
        
        Returns:
            Filtered list of signals
        """
        if not signals:
            return []
        
        # Collect feature rows
        feature_rows = []
        signal_map = {}
        
        for signal in signals:
            features = get_features(signal)
            if features is not None:
                feature_rows.append(features)
                signal_map[len(feature_rows) - 1] = signal
        
        if not feature_rows:
            return []
        
        # Get ML predictions
        ml_predictions = self.predict_batch(feature_rows)
        
        # Normalize to [0, 1]
        if len(ml_predictions) > 1:
            min_pred = ml_predictions.min()
            max_pred = ml_predictions.max()
            if max_pred - min_pred > 1e-10:
                ml_scores = (ml_predictions - min_pred) / (max_pred - min_pred)
            else:
                ml_scores = np.full_like(ml_predictions, 0.5)
        else:
            ml_scores = np.array([0.5])
        
        # Filter signals
        filtered_signals = []
        for idx, signal in signal_map.items():
            if ml_scores[idx] >= self.confidence_threshold:
                filtered_signals.append(signal)
        
        return filtered_signals
    
    def get_prediction_stats(
        self,
        feature_rows: List[FeatureRow],
    ) -> Dict[str, float]:
        """Get statistics about predictions.
        
        Args:
            feature_rows: List of FeatureRow objects
        
        Returns:
            Dictionary of prediction statistics
        """
        if not feature_rows:
            return {}
        
        predictions = self.predict_batch(feature_rows)
        
        stats = {
            "mean": float(np.mean(predictions)),
            "std": float(np.std(predictions)),
            "min": float(np.min(predictions)),
            "max": float(np.max(predictions)),
            "median": float(np.median(predictions)),
            "q25": float(np.percentile(predictions, 25)),
            "q75": float(np.percentile(predictions, 75)),
        }
        
        return stats


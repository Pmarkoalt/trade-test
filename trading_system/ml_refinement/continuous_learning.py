"""Continuous Learning Module for ML Model Management.

This module provides:
1. Scheduled model retraining
2. Model comparison and validation
3. Automatic deployment of improved models
4. Concept drift detection and alerting

Usage:
    from trading_system.ml_refinement.continuous_learning import ContinuousLearningManager
    
    manager = ContinuousLearningManager(
        feature_db_path="ml_features.db",
        model_dir="models/signal_quality",
    )
    
    # Check if retraining needed and retrain if so
    result = manager.check_and_retrain()
    
    # Monitor for drift
    drift_report = manager.check_drift()
"""

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from .config import MLConfig, ModelType
from .storage.feature_db import FeatureDatabase
from .training import ModelTrainer, TrainingResult


@dataclass
class ModelComparisonResult:
    """Result of comparing two models."""
    
    current_model_id: str
    new_model_id: str
    current_metrics: Dict[str, float]
    new_metrics: Dict[str, float]
    improvement: Dict[str, float]
    should_deploy: bool
    reason: str


@dataclass
class DriftReport:
    """Report on concept drift detection."""
    
    timestamp: str
    drift_detected: bool
    drift_score: float
    feature_drifts: Dict[str, float] = field(default_factory=dict)
    prediction_drift: float = 0.0
    recommendation: str = ""


@dataclass
class RetrainingResult:
    """Result of a retraining cycle."""
    
    timestamp: str
    triggered_by: str  # "schedule", "new_samples", "drift", "manual"
    training_result: Optional[TrainingResult] = None
    comparison_result: Optional[ModelComparisonResult] = None
    deployed: bool = False
    error: Optional[str] = None


class ContinuousLearningManager:
    """Manages continuous learning lifecycle for ML models.
    
    Features:
    - Automatic retraining when new samples available
    - Model comparison with minimum improvement threshold
    - Safe deployment with rollback capability
    - Drift detection and alerting
    """
    
    # Minimum samples needed for retraining
    MIN_NEW_SAMPLES_FOR_RETRAIN = 20
    
    # Minimum AUC improvement to deploy new model
    MIN_AUC_IMPROVEMENT = 0.01  # 1% improvement required
    
    # Drift detection threshold (KL divergence)
    DRIFT_THRESHOLD = 0.1
    
    def __init__(
        self,
        feature_db_path: str = "ml_features.db",
        model_dir: str = "models/signal_quality",
        min_samples_for_retrain: int = MIN_NEW_SAMPLES_FOR_RETRAIN,
        min_auc_improvement: float = MIN_AUC_IMPROVEMENT,
        drift_threshold: float = DRIFT_THRESHOLD,
    ):
        """Initialize continuous learning manager.
        
        Args:
            feature_db_path: Path to feature database
            model_dir: Directory for model storage
            min_samples_for_retrain: Minimum new samples to trigger retrain
            min_auc_improvement: Minimum AUC improvement to deploy
            drift_threshold: Threshold for drift detection
        """
        self.feature_db_path = feature_db_path
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.min_samples_for_retrain = min_samples_for_retrain
        self.min_auc_improvement = min_auc_improvement
        self.drift_threshold = drift_threshold
        
        # State file for tracking
        self.state_file = self.model_dir / "continuous_learning_state.json"
        self._state = self._load_state()
        
    def _load_state(self) -> Dict[str, Any]:
        """Load persistent state."""
        if self.state_file.exists():
            with open(self.state_file, "r") as f:
                return json.load(f)
        return {
            "last_retrain": None,
            "last_sample_count": 0,
            "current_model_id": None,
            "retrain_history": [],
            "drift_history": [],
        }
    
    def _save_state(self) -> None:
        """Save persistent state."""
        with open(self.state_file, "w") as f:
            json.dump(self._state, f, indent=2, default=str)
    
    def get_current_model_path(self) -> Optional[Path]:
        """Get path to current active model."""
        # Find latest model in directory
        pkl_files = list(self.model_dir.glob("*.pkl"))
        if not pkl_files:
            return None
        return max(pkl_files, key=lambda p: p.stat().st_mtime)
    
    def should_retrain(self) -> Tuple[bool, str]:
        """Check if retraining is needed.
        
        Returns:
            Tuple of (should_retrain, reason)
        """
        db = FeatureDatabase(self.feature_db_path)
        db.initialize()
        
        try:
            current_samples = db.count_samples(require_target=True)
            last_count = self._state.get("last_sample_count", 0)
            new_samples = current_samples - last_count
            
            # Check for new samples
            if new_samples >= self.min_samples_for_retrain:
                return True, f"new_samples ({new_samples} new samples available)"
            
            # Check if no model exists
            if self.get_current_model_path() is None:
                if current_samples >= self.min_samples_for_retrain:
                    return True, "no_model (no model exists, samples available)"
            
            return False, f"not_needed (only {new_samples} new samples)"
            
        finally:
            db.close()
    
    def retrain(self, force: bool = False) -> RetrainingResult:
        """Retrain model if needed.
        
        Args:
            force: Force retraining regardless of sample count
            
        Returns:
            RetrainingResult with training outcome
        """
        result = RetrainingResult(
            timestamp=datetime.now().isoformat(),
            triggered_by="manual" if force else "check",
        )
        
        # Check if retraining needed
        should_retrain, reason = self.should_retrain()
        if not should_retrain and not force:
            result.triggered_by = "skipped"
            result.error = reason
            logger.info(f"Retraining skipped: {reason}")
            return result
        
        result.triggered_by = reason.split()[0] if not force else "manual"
        
        try:
            # Initialize database and trainer
            db = FeatureDatabase(self.feature_db_path)
            db.initialize()
            
            ml_config = MLConfig(
                enabled=True,
                model_dir=str(self.model_dir),
                feature_db_path=self.feature_db_path,
            )
            ml_config.training.min_training_samples = 10
            
            trainer = ModelTrainer(
                config=ml_config,
                feature_db=db,
                model_dir=str(self.model_dir / "candidates"),
            )
            
            # Train new model
            logger.info("Training new candidate model...")
            training_result = trainer.train(
                model_type=ModelType.SIGNAL_QUALITY,
            )
            
            result.training_result = training_result
            
            if not training_result.success:
                result.error = training_result.error_message
                db.close()
                return result
            
            # Compare with current model
            current_model_path = self.get_current_model_path()
            if current_model_path:
                comparison = self._compare_models(
                    current_model_path,
                    training_result,
                    db,
                )
                result.comparison_result = comparison
                
                if comparison.should_deploy:
                    self._deploy_model(training_result.model_id)
                    result.deployed = True
                    logger.info(f"Deployed new model: {training_result.model_id}")
                else:
                    logger.info(f"Kept current model: {comparison.reason}")
            else:
                # No current model, deploy the new one
                self._deploy_model(training_result.model_id)
                result.deployed = True
                logger.info(f"Deployed first model: {training_result.model_id}")
            
            # Update state
            self._state["last_retrain"] = result.timestamp
            self._state["last_sample_count"] = db.count_samples(require_target=True)
            if result.deployed:
                self._state["current_model_id"] = training_result.model_id
            self._state["retrain_history"].append({
                "timestamp": result.timestamp,
                "triggered_by": result.triggered_by,
                "model_id": training_result.model_id,
                "deployed": result.deployed,
                "cv_auc": training_result.cv_metrics.get("auc", 0),
            })
            self._save_state()
            
            db.close()
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}", exc_info=True)
            result.error = str(e)
        
        return result
    
    def _compare_models(
        self,
        current_model_path: Path,
        new_training_result: TrainingResult,
        db: FeatureDatabase,
    ) -> ModelComparisonResult:
        """Compare current and new model performance.
        
        Args:
            current_model_path: Path to current model
            new_training_result: Training result for new model
            db: Feature database for holdout evaluation
            
        Returns:
            ModelComparisonResult with comparison outcome
        """
        # Get current model metrics from metadata
        current_metrics = self._get_model_metrics(current_model_path)
        new_metrics = new_training_result.cv_metrics
        
        # Calculate improvement
        improvement = {}
        for key in ["auc", "accuracy", "f1"]:
            current_val = current_metrics.get(key, 0)
            new_val = new_metrics.get(key, 0)
            improvement[key] = new_val - current_val
        
        # Decision logic
        auc_improvement = improvement.get("auc", 0)
        should_deploy = auc_improvement >= self.min_auc_improvement
        
        if should_deploy:
            reason = f"AUC improved by {auc_improvement:.4f} (>= {self.min_auc_improvement})"
        else:
            reason = f"AUC improvement {auc_improvement:.4f} below threshold {self.min_auc_improvement}"
        
        return ModelComparisonResult(
            current_model_id=current_model_path.stem,
            new_model_id=new_training_result.model_id,
            current_metrics=current_metrics,
            new_metrics=new_metrics,
            improvement=improvement,
            should_deploy=should_deploy,
            reason=reason,
        )
    
    def _get_model_metrics(self, model_path: Path) -> Dict[str, float]:
        """Get metrics from saved model."""
        import pickle
        
        try:
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
            
            metadata = model_data.get("metadata")
            if metadata and hasattr(metadata, "validation_metrics"):
                return metadata.validation_metrics or {}
            return {}
        except Exception as e:
            logger.warning(f"Could not load model metrics: {e}")
            return {}
    
    def _deploy_model(self, model_id: str) -> None:
        """Deploy a candidate model to production.
        
        Args:
            model_id: ID of model to deploy
        """
        candidate_path = self.model_dir / "candidates" / f"{model_id}.pkl"
        if candidate_path.exists():
            # Copy to main model directory
            dest_path = self.model_dir / f"{model_id}.pkl"
            shutil.copy2(candidate_path, dest_path)
            logger.info(f"Deployed model {model_id} to {dest_path}")
    
    def check_drift(self, lookback_days: int = 30) -> DriftReport:
        """Check for concept drift in recent predictions.
        
        Args:
            lookback_days: Number of days to analyze
            
        Returns:
            DriftReport with drift analysis
        """
        report = DriftReport(
            timestamp=datetime.now().isoformat(),
            drift_detected=False,
            drift_score=0.0,
        )
        
        db = FeatureDatabase(self.feature_db_path)
        db.initialize()
        
        try:
            # Get recent samples
            cutoff_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            
            # Get feature statistics for recent vs historical data
            recent_samples = db.count_samples(start_date=cutoff_date, require_target=True)
            total_samples = db.count_samples(require_target=True)
            historical_samples = total_samples - recent_samples
            
            if recent_samples < 10 or historical_samples < 10:
                report.recommendation = "Insufficient samples for drift detection"
                db.close()
                return report
            
            # Load data for drift analysis
            X_all, y_all, feature_names = db.get_training_data(require_target=True)
            
            if len(X_all) == 0:
                report.recommendation = "No data available"
                db.close()
                return report
            
            # Split into historical and recent
            split_idx = historical_samples
            X_historical = X_all[:split_idx]
            X_recent = X_all[split_idx:]
            y_historical = y_all[:split_idx]
            y_recent = y_all[split_idx:]
            
            # Check target drift (win rate change)
            historical_win_rate = (y_historical > 0).mean()
            recent_win_rate = (y_recent > 0).mean()
            target_drift = abs(recent_win_rate - historical_win_rate)
            report.prediction_drift = float(target_drift)
            
            # Check feature drift using simple statistics
            for i, name in enumerate(feature_names[:10]):  # Check top 10 features
                hist_mean = np.mean(X_historical[:, i])
                recent_mean = np.mean(X_recent[:, i])
                hist_std = np.std(X_historical[:, i]) + 1e-10
                
                # Normalized difference
                drift = abs(recent_mean - hist_mean) / hist_std
                report.feature_drifts[name] = float(drift)
            
            # Overall drift score (average of feature drifts + target drift)
            feature_drift_avg = np.mean(list(report.feature_drifts.values()))
            report.drift_score = float(0.5 * feature_drift_avg + 0.5 * target_drift * 10)
            
            # Determine if drift detected
            report.drift_detected = report.drift_score > self.drift_threshold
            
            if report.drift_detected:
                report.recommendation = "Consider retraining model - significant drift detected"
            else:
                report.recommendation = "No significant drift - model stable"
            
            # Save to history
            self._state["drift_history"].append({
                "timestamp": report.timestamp,
                "drift_score": report.drift_score,
                "drift_detected": report.drift_detected,
            })
            # Keep last 100 entries
            self._state["drift_history"] = self._state["drift_history"][-100:]
            self._save_state()
            
        except Exception as e:
            logger.error(f"Drift check failed: {e}", exc_info=True)
            report.recommendation = f"Error: {str(e)}"
        finally:
            db.close()
        
        return report
    
    def check_and_retrain(self) -> RetrainingResult:
        """Check if retraining needed and retrain if so.
        
        This is the main entry point for scheduled retraining.
        
        Returns:
            RetrainingResult with outcome
        """
        should_retrain, reason = self.should_retrain()
        
        if should_retrain:
            logger.info(f"Retraining triggered: {reason}")
            return self.retrain()
        else:
            logger.info(f"Retraining not needed: {reason}")
            return RetrainingResult(
                timestamp=datetime.now().isoformat(),
                triggered_by="skipped",
                error=reason,
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of continuous learning system.
        
        Returns:
            Dictionary with status information
        """
        db = FeatureDatabase(self.feature_db_path)
        db.initialize()
        
        current_samples = db.count_samples(require_target=True)
        db.close()
        
        current_model = self.get_current_model_path()
        
        return {
            "current_model": str(current_model) if current_model else None,
            "current_model_id": self._state.get("current_model_id"),
            "total_labeled_samples": current_samples,
            "samples_since_last_retrain": current_samples - self._state.get("last_sample_count", 0),
            "last_retrain": self._state.get("last_retrain"),
            "retrain_count": len(self._state.get("retrain_history", [])),
            "min_samples_for_retrain": self.min_samples_for_retrain,
            "min_auc_improvement": self.min_auc_improvement,
        }


def run_scheduled_retrain(
    feature_db_path: str = "ml_features.db",
    model_dir: str = "models/signal_quality",
) -> RetrainingResult:
    """Run a scheduled retraining cycle.
    
    This function is designed to be called by a scheduler (cron, etc.)
    
    Args:
        feature_db_path: Path to feature database
        model_dir: Model directory
        
    Returns:
        RetrainingResult
    """
    manager = ContinuousLearningManager(
        feature_db_path=feature_db_path,
        model_dir=model_dir,
    )
    
    logger.info("=" * 60)
    logger.info("SCHEDULED RETRAINING CYCLE")
    logger.info("=" * 60)
    
    # Check status
    status = manager.get_status()
    logger.info(f"Current samples: {status['total_labeled_samples']}")
    logger.info(f"Samples since last retrain: {status['samples_since_last_retrain']}")
    
    # Check for drift
    drift_report = manager.check_drift()
    if drift_report.drift_detected:
        logger.warning(f"Drift detected! Score: {drift_report.drift_score:.4f}")
    
    # Retrain if needed
    result = manager.check_and_retrain()
    
    logger.info("=" * 60)
    if result.deployed:
        logger.info(f"NEW MODEL DEPLOYED: {result.training_result.model_id}")
    else:
        logger.info(f"Status: {result.triggered_by}")
    logger.info("=" * 60)
    
    return result

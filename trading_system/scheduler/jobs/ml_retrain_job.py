"""ML model retraining job."""

from datetime import datetime, timedelta
from typing import Optional

from loguru import logger

from trading_system.ml_refinement.config import MLConfig, ModelType
from trading_system.ml_refinement.storage.feature_db import FeatureDatabase
from trading_system.ml_refinement.training.trainer import ModelTrainer

# Try to import ModelRegistry, create stub if not available
try:
    from trading_system.ml_refinement.models.model_registry import ModelRegistry
except ImportError:
    # Create minimal ModelRegistry stub
    class ModelRegistry:
        """Minimal model registry stub."""
        
        def __init__(self, feature_db: FeatureDatabase):
            self.feature_db = feature_db
            
        def get_active(self, model_type: ModelType):
            """Get active model metadata for type."""
            return self.feature_db.get_active_model(model_type.value)
            
        def activate(self, model_id: str):
            """Activate a model."""
            return self.feature_db.activate_model(model_id)


class MLRetrainJob:
    """
    Job for periodic ML model retraining.

    This job:
    1. Checks if enough new samples are available
    2. Retrains models with new data
    3. Evaluates new model vs current
    4. Activates new model if better

    Example:
        job = MLRetrainJob(config, feature_db, model_registry)

        # Run manually
        result = job.run()

        # Or schedule with APScheduler
        scheduler.add_job(
            job.run,
            'interval',
            days=7,  # Weekly
        )
    """

    def __init__(
        self,
        config: MLConfig,
        feature_db: FeatureDatabase,
        model_registry: Optional[ModelRegistry] = None,
        model_dir: str = "models/",
    ):
        """
        Initialize job.

        Args:
            config: ML configuration.
            feature_db: Feature database.
            model_registry: Model registry (creates default if not provided).
            model_dir: Directory for model storage.
        """
        self.config = config
        self.feature_db = feature_db
        self.model_registry = model_registry or ModelRegistry(feature_db)
        self.trainer = ModelTrainer(config, feature_db, model_dir)

    def run(
        self,
        force: bool = False,
        model_types: Optional[list] = None,
    ) -> dict:
        """
        Run the retraining job.

        Args:
            force: Force retraining regardless of sample count.
            model_types: Specific model types to retrain (default: all).

        Returns:
            Dict with job results.
        """
        logger.info("Starting ML retrain job")
        start_time = datetime.now()

        results = {
            "started_at": start_time.isoformat(),
            "models_retrained": [],
            "models_skipped": [],
            "errors": [],
        }

        model_types = model_types or [ModelType.SIGNAL_QUALITY]

        for model_type in model_types:
            try:
                result = self._retrain_model(model_type, force)
                if result:
                    results["models_retrained"].append({
                        "model_type": model_type.value,
                        "model_id": result.model_id,
                        "cv_auc": result.cv_metrics.get("auc", 0),
                        "samples": result.train_samples,
                    })
                else:
                    results["models_skipped"].append(model_type.value)

            except Exception as e:
                logger.exception(f"Error retraining {model_type.value}")
                results["errors"].append({
                    "model_type": model_type.value,
                    "error": str(e),
                })

        elapsed = (datetime.now() - start_time).total_seconds()
        results["completed_at"] = datetime.now().isoformat()
        results["elapsed_seconds"] = elapsed

        logger.info(
            f"Retrain job complete: "
            f"{len(results['models_retrained'])} retrained, "
            f"{len(results['models_skipped'])} skipped, "
            f"{len(results['errors'])} errors"
        )

        return results

    def _retrain_model(
        self,
        model_type: ModelType,
        force: bool,
    ) -> Optional[object]:
        """Retrain a single model type."""
        # Check if retraining needed
        if not force:
            current_model = self.model_registry.get_active(model_type)

            if current_model:
                # Count new samples since last training
                new_samples = self.feature_db.count_samples(
                    start_date=current_model.train_end_date,
                    require_target=True,
                )

                if new_samples < self.config.min_new_samples_for_retrain:
                    logger.info(
                        f"Skipping {model_type.value}: only {new_samples} new samples "
                        f"(need {self.config.min_new_samples_for_retrain})"
                    )
                    return None

        # Calculate training date range
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (
            datetime.now() - timedelta(days=self.config.training.train_window_days * 2)
        ).strftime("%Y-%m-%d")

        # Train new model
        result = self.trainer.train(
            model_type=model_type,
            start_date=start_date,
            end_date=end_date,
        )

        if not result.success:
            raise RuntimeError(f"Training failed: {result.error_message}")

        # Compare with current model
        current_model = self.model_registry.get_active(model_type)
        should_activate = True

        if current_model:
            current_auc = current_model.validation_metrics.get("auc", 0)
            new_auc = result.cv_metrics.get("auc", 0)

            # Only activate if better (with small tolerance)
            if new_auc < current_auc - 0.02:
                logger.info(
                    f"New model ({new_auc:.3f}) not better than current ({current_auc:.3f}), "
                    f"keeping current"
                )
                should_activate = False

        if should_activate:
            self.model_registry.activate(result.model_id)
            logger.info(f"Activated new model: {result.model_id}")

        return result

    def check_retrain_needed(self, model_type: ModelType) -> dict:
        """
        Check if retraining is needed for a model type.

        Returns dict with status info.
        """
        current = self.model_registry.get_active(model_type)

        if current is None:
            return {
                "needed": True,
                "reason": "No active model",
                "new_samples": self.feature_db.count_samples(require_target=True),
            }

        new_samples = self.feature_db.count_samples(
            start_date=current.train_end_date,
            require_target=True,
        )

        needed = new_samples >= self.config.min_new_samples_for_retrain

        return {
            "needed": needed,
            "reason": f"{new_samples} new samples" if needed else "Insufficient new samples",
            "new_samples": new_samples,
            "threshold": self.config.min_new_samples_for_retrain,
            "current_model": current.model_id,
            "current_auc": current.validation_metrics.get("auc", 0),
        }


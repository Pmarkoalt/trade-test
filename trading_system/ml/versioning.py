"""Model versioning and management."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from trading_system.ml.models import MLModel, ModelMetadata, ModelType


class ModelVersionManager:
    """Manage versions of trained ML models.

    This class handles:
    - Model versioning and tracking
    - Model registry (metadata database)
    - Model deployment (promoting models to production)
    - Model comparison and selection
    """

    def __init__(self, model_directory: Path):
        """Initialize version manager.

        Args:
            model_directory: Base directory for storing models
        """
        self.model_directory = Path(model_directory)
        self.model_directory.mkdir(parents=True, exist_ok=True)

        self.registry_file = self.model_directory / "registry.json"
        self._registry: Dict[str, Dict] = self._load_registry()

    def _load_registry(self) -> Dict[str, Dict]:
        """Load model registry from file.

        Returns:
            Dictionary mapping model_id to registry entry
        """
        if self.registry_file.exists():
            with open(self.registry_file, "r") as f:
                return json.load(f)
        return {}

    def _save_registry(self) -> None:
        """Save model registry to file."""
        with open(self.registry_file, "w") as f:
            json.dump(self._registry, f, indent=2, default=str)

    def register_model(
        self,
        model_id: str,
        model_path: Path,
        metadata: ModelMetadata,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Register a model in the registry.

        Args:
            model_id: Unique model identifier
            model_path: Path to model directory
            metadata: Model metadata
            tags: Optional list of tags (e.g., ["production", "experimental"])
            notes: Optional notes about the model
        """
        registry_entry = {
            "model_id": model_id,
            "model_path": str(model_path),
            "model_type": metadata.model_type.value,
            "version": metadata.version,
            "training_date": metadata.training_date.isoformat(),
            "training_samples": metadata.training_samples,
            "feature_names": metadata.feature_names,
            "target_name": metadata.target_name,
            "hyperparameters": metadata.hyperparameters,
            "performance_metrics": metadata.performance_metrics,
            "training_time_seconds": metadata.training_time_seconds,
            "tags": tags or [],
            "notes": notes,
            "registered_date": pd.Timestamp.now().isoformat(),
            "is_production": False,
        }

        self._registry[model_id] = registry_entry
        self._save_registry()

    def list_models(
        self,
        model_type: Optional[ModelType] = None,
        tags: Optional[List[str]] = None,
        production_only: bool = False,
    ) -> List[Dict]:
        """List models in registry.

        Args:
            model_type: Filter by model type
            tags: Filter by tags (models must have all tags)
            production_only: Only return production models

        Returns:
            List of registry entries
        """
        models = list(self._registry.values())

        # Filter by type
        if model_type:
            models = [m for m in models if m["model_type"] == model_type.value]

        # Filter by tags
        if tags:
            models = [m for m in models if all(tag in m.get("tags", []) for tag in tags)]

        # Filter by production status
        if production_only:
            models = [m for m in models if m.get("is_production", False)]

        return models

    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """Get information about a specific model.

        Args:
            model_id: Model identifier

        Returns:
            Registry entry, or None if not found
        """
        return self._registry.get(model_id)

    def load_model(self, model_id: str) -> MLModel:
        """Load a model by ID.

        Args:
            model_id: Model identifier

        Returns:
            Loaded MLModel instance
        """
        if model_id not in self._registry:
            raise ValueError(f"Model {model_id} not found in registry")

        model_path = Path(self._registry[model_id]["model_path"])
        return MLModel.load(model_path)

    def set_production(self, model_id: str, is_production: bool = True) -> None:
        """Set production status of a model.

        Args:
            model_id: Model identifier
            is_production: Whether model should be marked as production
        """
        if model_id not in self._registry:
            raise ValueError(f"Model {model_id} not found in registry")

        if is_production:
            # Unset production flag on other models of same type
            model_type = self._registry[model_id]["model_type"]
            for other_id, other_entry in self._registry.items():
                if other_id != model_id and other_entry["model_type"] == model_type:
                    other_entry["is_production"] = False

        self._registry[model_id]["is_production"] = is_production
        self._save_registry()

    def get_production_model(self, model_type: Optional[ModelType] = None) -> Optional[str]:
        """Get production model ID.

        Args:
            model_type: Optional filter by model type

        Returns:
            Model ID of production model, or None
        """
        for model_id, entry in self._registry.items():
            if entry.get("is_production", False):
                if model_type is None or entry["model_type"] == model_type.value:
                    return model_id
        return None

    def compare_models(
        self,
        model_ids: List[str],
        metric: str = "r2",
    ) -> Dict[str, Dict]:
        """Compare multiple models.

        Args:
            model_ids: List of model IDs to compare
            metric: Performance metric to compare

        Returns:
            Dictionary mapping model_id to comparison data
        """
        comparison = {}

        for model_id in model_ids:
            if model_id not in self._registry:
                continue

            entry = self._registry[model_id]
            metrics = entry.get("performance_metrics", {})

            comparison[model_id] = {
                "model_type": entry["model_type"],
                "version": entry["version"],
                "training_date": entry["training_date"],
                "training_samples": entry["training_samples"],
                "metric_value": metrics.get(metric, None),
                "all_metrics": metrics,
                "is_production": entry.get("is_production", False),
            }

        return comparison

    def delete_model(self, model_id: str) -> None:
        """Delete a model from registry.

        Args:
            model_id: Model identifier
        """
        if model_id not in self._registry:
            raise ValueError(f"Model {model_id} not found in registry")

        # Don't delete production models without explicit confirmation
        if self._registry[model_id].get("is_production", False):
            raise ValueError(f"Cannot delete production model {model_id}. Set to non-production first.")

        del self._registry[model_id]
        self._save_registry()

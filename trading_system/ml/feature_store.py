"""Feature store for caching, versioning, and validation of features.

This module provides:
- Feature caching/storage
- Feature versioning
- Feature validation
"""

import hashlib
import json
import pickle
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from trading_system.models.features import FeatureRow


class FeatureStore:
    """Store and manage features with caching, versioning, and validation."""

    def __init__(
        self,
        cache_directory: Path,
        max_cache_size: int = 1000,  # Maximum number of cached feature sets
        enable_validation: bool = True,
    ):
        """Initialize feature store.

        Args:
            cache_directory: Directory for storing cached features
            max_cache_size: Maximum number of cached feature sets
            enable_validation: Enable feature validation
        """
        self.cache_directory = Path(cache_directory)
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size
        self.enable_validation = enable_validation

        # In-memory cache (LRU)
        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._cache_metadata: Dict[str, Dict[str, Any]] = {}

    def _compute_hash(self, feature_rows: List[FeatureRow]) -> str:
        """Compute hash for feature rows.

        Args:
            feature_rows: List of FeatureRow objects

        Returns:
            Hash string
        """
        # Create a string representation of feature rows
        data_str = ""
        for fr in feature_rows[:10]:  # Use first 10 rows for hash
            data_str += f"{fr.symbol}_{fr.date}_{fr.close}"

        # noqa: S324 - MD5 used for cache key, not security
        return hashlib.md5(data_str.encode(), usedforsecurity=False).hexdigest()

    def store(
        self,
        feature_rows: List[FeatureRow],
        feature_df: pd.DataFrame,
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store features in cache.

        Args:
            feature_rows: Original FeatureRow objects
            feature_df: Transformed feature DataFrame
            version: Feature version
            metadata: Additional metadata

        Returns:
            Cache key
        """
        cache_key = self._compute_hash(feature_rows)

        # Validate features if enabled
        if self.enable_validation:
            self._validate_features(feature_df)

        # Store in memory cache
        self._cache[cache_key] = feature_df.copy()
        self._cache_metadata[cache_key] = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(feature_df),
            "n_features": len(feature_df.columns),
            "metadata": metadata or {},
        }

        # Move to end (LRU)
        self._cache.move_to_end(cache_key)

        # Evict if cache is too large
        if len(self._cache) > self.max_cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            del self._cache_metadata[oldest_key]

        # Store to disk
        cache_file = self.cache_directory / f"{cache_key}.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(
                {
                    "features": feature_df,
                    "metadata": self._cache_metadata[cache_key],
                },
                f,
            )

        return cache_key

    def retrieve(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Retrieve features from cache.

        Args:
            cache_key: Cache key

        Returns:
            Feature DataFrame if found, None otherwise
        """
        # Check memory cache first
        if cache_key in self._cache:
            # Move to end (LRU)
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key].copy()

        # Check disk cache
        cache_file = self.cache_directory / f"{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
                feature_df = data["features"]

                # Load into memory cache
                self._cache[cache_key] = feature_df.copy()
                self._cache_metadata[cache_key] = data["metadata"]
                self._cache.move_to_end(cache_key)

                return feature_df.copy()

        return None

    def get_metadata(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for cached features.

        Args:
            cache_key: Cache key

        Returns:
            Metadata dictionary if found, None otherwise
        """
        if cache_key in self._cache_metadata:
            return self._cache_metadata[cache_key].copy()

        # Check disk
        cache_file = self.cache_directory / f"{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
                metadata = data.get("metadata") if isinstance(data, dict) else None
                return dict(metadata) if metadata else None

        return None

    def _validate_features(self, feature_df: pd.DataFrame) -> None:
        """Validate feature DataFrame.

        Args:
            feature_df: Feature DataFrame

        Raises:
            ValueError: If validation fails
        """
        if feature_df.empty:
            raise ValueError("Feature DataFrame is empty")

        # Check for infinite values
        if np.isinf(feature_df.select_dtypes(include=[np.number])).any().any():
            raise ValueError("Feature DataFrame contains infinite values")

        # Check for excessive missing values
        missing_pct = feature_df.isna().sum() / len(feature_df)
        high_missing = missing_pct[missing_pct > 0.5]
        if len(high_missing) > 0:
            raise ValueError(f"Features with >50% missing values: {list(high_missing.index)}")

        # Check for constant features
        constant_features = []
        for col in feature_df.columns:
            if feature_df[col].nunique() <= 1:
                constant_features.append(col)

        if constant_features:
            # Warning, not error
            import warnings

            warnings.warn(f"Constant features detected: {constant_features}")

    def list_cached_features(self) -> List[Dict[str, Any]]:
        """List all cached features.

        Returns:
            List of cache entries with metadata
        """
        entries = []

        # Memory cache
        for cache_key, metadata in self._cache_metadata.items():
            entries.append(
                {
                    "cache_key": cache_key,
                    **metadata,
                }
            )

        # Disk cache
        for cache_file in self.cache_directory.glob("*.pkl"):
            cache_key = cache_file.stem
            if cache_key not in self._cache_metadata:
                try:
                    with open(cache_file, "rb") as f:
                        data = pickle.load(f)
                        entries.append(
                            {
                                "cache_key": cache_key,
                                **data["metadata"],
                            }
                        )
                except Exception:  # nosec B112 - exception handling for cache file loading, skip invalid entries
                    continue

        return entries

    def clear_cache(self) -> None:
        """Clear all cached features."""
        self._cache.clear()
        self._cache_metadata.clear()

        # Clear disk cache
        for cache_file in self.cache_directory.glob("*.pkl"):
            cache_file.unlink()


class FeatureVersionManager:
    """Manage feature versions and schema changes."""

    def __init__(self, version_directory: Path):
        """Initialize version manager.

        Args:
            version_directory: Directory for storing feature versions
        """
        self.version_directory = Path(version_directory)
        self.version_directory.mkdir(parents=True, exist_ok=True)

    def save_version(
        self,
        feature_schema: Dict[str, Any],
        version: str,
        description: Optional[str] = None,
    ) -> Path:
        """Save a feature schema version.

        Args:
            feature_schema: Feature schema (feature names, types, etc.)
            version: Version string
            description: Optional description

        Returns:
            Path to saved version file
        """
        version_file = self.version_directory / f"{version}.json"

        version_data = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "schema": feature_schema,
            "description": description,
        }

        with open(version_file, "w") as f:
            json.dump(version_data, f, indent=2)

        # Update version index
        self._update_version_index(version)

        return version_file

    def load_version(self, version: str) -> Dict[str, Any]:
        """Load a feature schema version.

        Args:
            version: Version string

        Returns:
            Version data dictionary
        """
        version_file = self.version_directory / f"{version}.json"
        if not version_file.exists():
            raise ValueError(f"Feature version {version} not found")

        with open(version_file, "r") as f:
            result = json.load(f)
            return dict(result) if result else {}

    def list_versions(self) -> List[str]:
        """List all available feature versions.

        Returns:
            List of version strings
        """
        versions = []
        for version_file in self.version_directory.glob("*.json"):
            if version_file.name != "versions.json":
                versions.append(version_file.stem)

        # Sort versions
        try:
            versions.sort(key=lambda v: [int(x) for x in v.split(".")])
        except ValueError:
            versions.sort()

        return versions

    def get_latest_version(self) -> Optional[str]:
        """Get the latest feature version.

        Returns:
            Latest version string, or None if no versions exist
        """
        versions = self.list_versions()
        return versions[-1] if versions else None

    def _update_version_index(self, version: str) -> None:
        """Update version index file."""
        index_path = self.version_directory / "versions.json"

        if index_path.exists():
            with open(index_path, "r") as f:
                index = json.load(f)
        else:
            index = {"versions": []}

        if version not in index["versions"]:
            index["versions"].append(version)
            index["versions"].sort()

        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

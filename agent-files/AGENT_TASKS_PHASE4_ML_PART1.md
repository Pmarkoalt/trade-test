# Agent Tasks: Phase 4 - ML Refinement (Part 1: Feature Store & Infrastructure)

**Phase Goal**: Use ML to improve signal quality over time through learning from outcomes
**Duration**: 1-2 weeks (Part 1)
**Prerequisites**: Phase 3 Performance Tracking complete

---

## Phase 4 Part 1 Overview

### What We're Building
1. **Feature Store** - Store and retrieve ML features for signals
2. **Feature Engineering** - Extract predictive features from market data
3. **Model Infrastructure** - Base classes and interfaces for ML models
4. **Walk-Forward Framework** - Time-series aware validation

### Architecture Addition

```
trading_system/
├── ml_refinement/                   # NEW: ML refinement module
│   ├── __init__.py
│   ├── config.py                    # ML configuration
│   ├── features/
│   │   ├── __init__.py
│   │   ├── feature_store.py         # Feature storage and retrieval
│   │   ├── feature_registry.py      # Register available features
│   │   ├── extractors/
│   │   │   ├── __init__.py
│   │   │   ├── base_extractor.py    # Abstract extractor
│   │   │   ├── technical_features.py # Technical indicator features
│   │   │   ├── market_features.py   # Market regime features
│   │   │   ├── signal_features.py   # Signal metadata features
│   │   │   └── news_features.py     # News/sentiment features
│   │   └── pipeline.py              # Feature extraction pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py            # Abstract model interface
│   │   └── model_registry.py        # Model versioning registry
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── walk_forward.py          # Walk-forward validation
│   │   └── metrics.py               # ML evaluation metrics
│   └── storage/
│       ├── __init__.py
│       ├── feature_db.py            # Feature database
│       └── migrations/
│           └── 001_feature_schema.sql
```

---

## Task 4.1.1: Create ML Refinement Module Structure

**Context**:
The ML refinement module enables the system to learn from historical signal outcomes and improve future predictions.

**Objective**:
Create the directory structure, configuration, and core data models.

**Files to Create**:
```
trading_system/ml_refinement/
├── __init__.py
├── config.py
├── features/
│   ├── __init__.py
│   └── ... (stubs)
├── models/
│   ├── __init__.py
│   └── ... (stubs)
├── validation/
│   ├── __init__.py
│   └── ... (stubs)
└── storage/
    ├── __init__.py
    └── ... (stubs)
```

**Requirements**:

1. Create `config.py` with ML configuration:
```python
"""Configuration for ML refinement module."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class ModelType(str, Enum):
    """Types of ML models."""
    SIGNAL_QUALITY = "signal_quality"
    RETURN_PREDICTOR = "return_predictor"
    REGIME_CLASSIFIER = "regime_classifier"
    RISK_PREDICTOR = "risk_predictor"


class FeatureSet(str, Enum):
    """Predefined feature sets."""
    MINIMAL = "minimal"          # Basic features only
    STANDARD = "standard"        # Standard feature set
    EXTENDED = "extended"        # All available features
    CUSTOM = "custom"            # User-defined


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""

    # Feature sets to use
    feature_set: FeatureSet = FeatureSet.STANDARD

    # Custom features (if feature_set is CUSTOM)
    custom_features: List[str] = field(default_factory=list)

    # Technical feature parameters
    technical_lookbacks: List[int] = field(
        default_factory=lambda: [5, 10, 20, 50, 200]
    )

    # Market feature parameters
    market_regime_window: int = 20
    volatility_window: int = 20

    # News feature parameters
    news_lookback_hours: int = 48
    include_news_features: bool = True

    # Feature scaling
    scale_features: bool = True
    scaling_method: str = "standard"  # "standard", "minmax", "robust"


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Walk-forward parameters
    train_window_days: int = 252        # ~1 year
    validation_window_days: int = 63    # ~3 months
    step_size_days: int = 21            # ~1 month

    # Minimum data requirements
    min_training_samples: int = 100
    min_validation_samples: int = 20

    # Model parameters
    model_type: str = "gradient_boosting"
    hyperparameters: Dict = field(default_factory=dict)

    # Training options
    early_stopping: bool = True
    early_stopping_rounds: int = 50

    # Regularization
    max_features: int = 50              # Limit feature count
    feature_selection: bool = True
    feature_importance_threshold: float = 0.01


@dataclass
class MLConfig:
    """Main ML configuration."""

    # Enable/disable ML
    enabled: bool = True

    # Feature configuration
    features: FeatureConfig = field(default_factory=FeatureConfig)

    # Training configuration
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Model paths
    model_dir: str = "models/"
    feature_db_path: str = "features.db"

    # Retraining schedule
    retrain_frequency_days: int = 7     # Weekly retraining
    min_new_samples_for_retrain: int = 20

    # Prediction thresholds
    quality_threshold_high: float = 0.7
    quality_threshold_low: float = 0.3

    # Integration
    use_ml_scores: bool = True
    ml_score_weight: float = 0.3        # Weight in combined score


@dataclass
class FeatureVector:
    """A single feature vector for a signal."""

    signal_id: str
    timestamp: str
    features: Dict[str, float]
    target: Optional[float] = None      # R-multiple outcome
    target_binary: Optional[int] = None # 1 = win, 0 = loss

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "signal_id": self.signal_id,
            "timestamp": self.timestamp,
            "features": self.features,
            "target": self.target,
            "target_binary": self.target_binary,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "FeatureVector":
        """Create from dictionary."""
        return cls(
            signal_id=data["signal_id"],
            timestamp=data["timestamp"],
            features=data["features"],
            target=data.get("target"),
            target_binary=data.get("target_binary"),
        )


@dataclass
class ModelMetadata:
    """Metadata for a trained model."""

    model_id: str
    model_type: ModelType
    version: str
    created_at: str

    # Training info
    train_start_date: str
    train_end_date: str
    train_samples: int
    validation_samples: int

    # Performance metrics
    train_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)

    # Feature info
    feature_names: List[str] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)

    # Status
    is_active: bool = False
    deployed_at: Optional[str] = None
```

2. Create `__init__.py` with exports:
```python
"""ML Refinement module for improving signal quality."""

from trading_system.ml_refinement.config import (
    FeatureConfig,
    FeatureSet,
    FeatureVector,
    MLConfig,
    ModelMetadata,
    ModelType,
    TrainingConfig,
)

__all__ = [
    "FeatureConfig",
    "FeatureSet",
    "FeatureVector",
    "MLConfig",
    "ModelMetadata",
    "ModelType",
    "TrainingConfig",
]
```

3. Create stub files for all submodules with docstrings.

**Acceptance Criteria**:
- [ ] All directories and files created
- [ ] Config dataclasses properly defined
- [ ] FeatureVector serialization works
- [ ] Imports work: `from trading_system.ml_refinement import MLConfig`

**Tests to Write**:
```python
def test_feature_vector_roundtrip():
    """Test FeatureVector serialization."""
    fv = FeatureVector(
        signal_id="test-123",
        timestamp="2024-01-01T10:00:00",
        features={"rsi": 65.0, "atr_ratio": 1.2},
        target=2.0,
        target_binary=1,
    )
    data = fv.to_dict()
    restored = FeatureVector.from_dict(data)
    assert restored.signal_id == fv.signal_id
    assert restored.features["rsi"] == 65.0

def test_ml_config_defaults():
    """Test MLConfig has sensible defaults."""
    config = MLConfig()
    assert config.enabled
    assert config.training.min_training_samples == 100
```

---

## Task 4.1.2: Design and Create Feature Database Schema

**Context**:
Features need to be stored persistently for training and analysis.

**Objective**:
Create SQLite schema for feature storage with efficient retrieval.

**Files to Create**:
```
trading_system/ml_refinement/storage/
├── __init__.py
├── migrations/
│   └── 001_feature_schema.sql
└── feature_db.py
```

**Requirements**:

1. Create `migrations/001_feature_schema.sql`:
```sql
-- Feature Store Schema
-- Migration: 001_feature_schema
-- Created: 2024-12-30

-- Feature vectors table
CREATE TABLE IF NOT EXISTS feature_vectors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id TEXT NOT NULL UNIQUE,
    timestamp TEXT NOT NULL,

    -- Features stored as JSON
    features TEXT NOT NULL,

    -- Target values (filled in after outcome known)
    target_r_multiple REAL,
    target_binary INTEGER,           -- 1 = win, 0 = loss
    target_return_pct REAL,

    -- Metadata
    feature_version TEXT NOT NULL,   -- Version of feature extraction
    created_at TEXT NOT NULL,
    updated_at TEXT,

    -- Signal metadata for filtering
    symbol TEXT,
    asset_class TEXT,
    signal_type TEXT,
    conviction TEXT
);

CREATE INDEX IF NOT EXISTS idx_fv_signal_id ON feature_vectors(signal_id);
CREATE INDEX IF NOT EXISTS idx_fv_timestamp ON feature_vectors(timestamp);
CREATE INDEX IF NOT EXISTS idx_fv_symbol ON feature_vectors(symbol);
CREATE INDEX IF NOT EXISTS idx_fv_asset_class ON feature_vectors(asset_class);
CREATE INDEX IF NOT EXISTS idx_fv_has_target ON feature_vectors(target_r_multiple IS NOT NULL);


-- Model registry table
CREATE TABLE IF NOT EXISTS model_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL UNIQUE,
    model_type TEXT NOT NULL,
    version TEXT NOT NULL,
    created_at TEXT NOT NULL,

    -- Training info
    train_start_date TEXT,
    train_end_date TEXT,
    train_samples INTEGER,
    validation_samples INTEGER,

    -- Metrics stored as JSON
    train_metrics TEXT,
    validation_metrics TEXT,

    -- Feature info
    feature_names TEXT,              -- JSON array
    feature_importance TEXT,         -- JSON dict

    -- Model binary
    model_path TEXT,                 -- Path to serialized model

    -- Status
    is_active INTEGER NOT NULL DEFAULT 0,
    deployed_at TEXT,
    retired_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_model_type ON model_registry(model_type);
CREATE INDEX IF NOT EXISTS idx_model_active ON model_registry(is_active);
CREATE UNIQUE INDEX IF NOT EXISTS idx_model_type_active
    ON model_registry(model_type) WHERE is_active = 1;


-- Feature definitions table
CREATE TABLE IF NOT EXISTS feature_definitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,          -- "technical", "market", "signal", "news"
    description TEXT,
    data_type TEXT NOT NULL,         -- "float", "int", "bool"
    computation TEXT,                -- Description of how it's computed
    dependencies TEXT,               -- JSON array of data dependencies
    created_at TEXT NOT NULL
);


-- Training runs table
CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    model_type TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,            -- "running", "completed", "failed"

    -- Configuration
    config TEXT,                     -- JSON of training config

    -- Results
    train_samples INTEGER,
    validation_samples INTEGER,
    best_iteration INTEGER,
    metrics TEXT,                    -- JSON of final metrics

    -- Resulting model
    model_id TEXT,

    -- Errors
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_training_status ON training_runs(status);
CREATE INDEX IF NOT EXISTS idx_training_model_type ON training_runs(model_type);


-- Predictions log table
CREATE TABLE IF NOT EXISTS prediction_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    predicted_at TEXT NOT NULL,

    -- Predictions
    quality_score REAL,              -- 0-1 probability of success
    predicted_r REAL,                -- Predicted R-multiple
    confidence REAL,                 -- Model confidence

    -- Actual outcome (filled later)
    actual_r REAL,
    prediction_error REAL,

    FOREIGN KEY (model_id) REFERENCES model_registry(model_id)
);

CREATE INDEX IF NOT EXISTS idx_pred_signal ON prediction_log(signal_id);
CREATE INDEX IF NOT EXISTS idx_pred_model ON prediction_log(model_id);


-- Schema migrations tracking
CREATE TABLE IF NOT EXISTS ml_schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

INSERT OR IGNORE INTO ml_schema_migrations (version, applied_at)
VALUES (1, datetime('now'));
```

2. Create `feature_db.py`:
```python
"""Feature database implementation."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from trading_system.ml_refinement.config import FeatureVector, ModelMetadata


class FeatureDatabase:
    """
    SQLite database for feature storage.

    Example:
        db = FeatureDatabase("features.db")
        db.initialize()

        # Store features
        db.store_feature_vector(feature_vector)

        # Retrieve for training
        X, y = db.get_training_data(
            start_date="2023-01-01",
            end_date="2023-12-31",
        )
    """

    FEATURE_VERSION = "1.0"

    def __init__(self, db_path: str = "features.db"):
        """Initialize feature database."""
        self.db_path = Path(db_path)
        self._connection: Optional[sqlite3.Connection] = None

    @property
    def connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                self.db_path,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            self._connection.row_factory = sqlite3.Row
        return self._connection

    @contextmanager
    def transaction(self):
        """Transaction context manager."""
        try:
            yield self.connection
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def initialize(self) -> None:
        """Initialize database with schema."""
        migrations_dir = Path(__file__).parent / "migrations"
        migration_file = migrations_dir / "001_feature_schema.sql"

        with open(migration_file) as f:
            schema_sql = f.read()

        with self.transaction():
            self.connection.executescript(schema_sql)

        logger.info(f"Initialized feature database at {self.db_path}")

    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None

    # Feature vector operations
    def store_feature_vector(
        self,
        fv: FeatureVector,
        symbol: str = "",
        asset_class: str = "",
        signal_type: str = "",
        conviction: str = "",
    ) -> bool:
        """Store a feature vector."""
        sql = """
            INSERT OR REPLACE INTO feature_vectors (
                signal_id, timestamp, features,
                target_r_multiple, target_binary, target_return_pct,
                feature_version, created_at,
                symbol, asset_class, signal_type, conviction
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        with self.transaction():
            self.connection.execute(sql, (
                fv.signal_id,
                fv.timestamp,
                json.dumps(fv.features),
                fv.target,
                fv.target_binary,
                None,  # target_return_pct - can be added later
                self.FEATURE_VERSION,
                datetime.now().isoformat(),
                symbol,
                asset_class,
                signal_type,
                conviction,
            ))

        return True

    def update_target(
        self,
        signal_id: str,
        r_multiple: float,
        return_pct: Optional[float] = None,
    ) -> bool:
        """Update target values after outcome is known."""
        target_binary = 1 if r_multiple > 0 else 0

        sql = """
            UPDATE feature_vectors
            SET target_r_multiple = ?,
                target_binary = ?,
                target_return_pct = ?,
                updated_at = ?
            WHERE signal_id = ?
        """

        with self.transaction():
            cursor = self.connection.execute(sql, (
                r_multiple,
                target_binary,
                return_pct,
                datetime.now().isoformat(),
                signal_id,
            ))

        return cursor.rowcount > 0

    def get_feature_vector(self, signal_id: str) -> Optional[FeatureVector]:
        """Get a feature vector by signal ID."""
        sql = "SELECT * FROM feature_vectors WHERE signal_id = ?"

        cursor = self.connection.execute(sql, (signal_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return FeatureVector(
            signal_id=row["signal_id"],
            timestamp=row["timestamp"],
            features=json.loads(row["features"]),
            target=row["target_r_multiple"],
            target_binary=row["target_binary"],
        )

    def get_training_data(
        self,
        start_date: str,
        end_date: str,
        asset_class: Optional[str] = None,
        signal_type: Optional[str] = None,
        require_target: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Get training data as numpy arrays.

        Returns:
            Tuple of (X, y, feature_names)
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            feature_names: List of feature names
        """
        conditions = ["timestamp >= ? AND timestamp <= ?"]
        params: List = [start_date, end_date]

        if require_target:
            conditions.append("target_r_multiple IS NOT NULL")

        if asset_class:
            conditions.append("asset_class = ?")
            params.append(asset_class)

        if signal_type:
            conditions.append("signal_type = ?")
            params.append(signal_type)

        sql = f"""
            SELECT features, target_r_multiple
            FROM feature_vectors
            WHERE {' AND '.join(conditions)}
            ORDER BY timestamp ASC
        """

        cursor = self.connection.execute(sql, params)
        rows = cursor.fetchall()

        if not rows:
            return np.array([]), np.array([]), []

        # Parse features
        feature_dicts = [json.loads(row["features"]) for row in rows]
        targets = [row["target_r_multiple"] for row in rows]

        # Get consistent feature names
        all_keys = set()
        for fd in feature_dicts:
            all_keys.update(fd.keys())
        feature_names = sorted(all_keys)

        # Build arrays
        X = np.array([
            [fd.get(key, 0.0) for key in feature_names]
            for fd in feature_dicts
        ])
        y = np.array(targets)

        return X, y, feature_names

    def get_training_data_binary(
        self,
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Get training data with binary target (win/loss)."""
        X, y, feature_names = self.get_training_data(
            start_date, end_date, **kwargs
        )

        if len(y) > 0:
            y_binary = (y > 0).astype(int)
        else:
            y_binary = y

        return X, y_binary, feature_names

    def count_samples(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        require_target: bool = True,
    ) -> int:
        """Count available samples."""
        conditions = []
        params = []

        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date)

        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date)

        if require_target:
            conditions.append("target_r_multiple IS NOT NULL")

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        sql = f"SELECT COUNT(*) FROM feature_vectors {where_clause}"

        cursor = self.connection.execute(sql, params)
        return cursor.fetchone()[0]

    # Model registry operations
    def register_model(self, metadata: ModelMetadata) -> bool:
        """Register a trained model."""
        sql = """
            INSERT INTO model_registry (
                model_id, model_type, version, created_at,
                train_start_date, train_end_date,
                train_samples, validation_samples,
                train_metrics, validation_metrics,
                feature_names, feature_importance,
                model_path, is_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        with self.transaction():
            self.connection.execute(sql, (
                metadata.model_id,
                metadata.model_type.value,
                metadata.version,
                metadata.created_at,
                metadata.train_start_date,
                metadata.train_end_date,
                metadata.train_samples,
                metadata.validation_samples,
                json.dumps(metadata.train_metrics),
                json.dumps(metadata.validation_metrics),
                json.dumps(metadata.feature_names),
                json.dumps(metadata.feature_importance),
                f"{metadata.model_id}.pkl",
                0,  # Not active by default
            ))

        return True

    def activate_model(self, model_id: str) -> bool:
        """Activate a model (deactivate others of same type)."""
        # Get model type
        sql = "SELECT model_type FROM model_registry WHERE model_id = ?"
        cursor = self.connection.execute(sql, (model_id,))
        row = cursor.fetchone()

        if not row:
            return False

        model_type = row["model_type"]

        with self.transaction():
            # Deactivate all models of this type
            self.connection.execute(
                "UPDATE model_registry SET is_active = 0 WHERE model_type = ?",
                (model_type,)
            )

            # Activate this model
            self.connection.execute(
                """UPDATE model_registry
                   SET is_active = 1, deployed_at = ?
                   WHERE model_id = ?""",
                (datetime.now().isoformat(), model_id)
            )

        return True

    def get_active_model(self, model_type: str) -> Optional[ModelMetadata]:
        """Get the active model for a type."""
        sql = """
            SELECT * FROM model_registry
            WHERE model_type = ? AND is_active = 1
        """

        cursor = self.connection.execute(sql, (model_type,))
        row = cursor.fetchone()

        if not row:
            return None

        return self._row_to_metadata(row)

    def get_model_history(
        self,
        model_type: str,
        limit: int = 10,
    ) -> List[ModelMetadata]:
        """Get model version history."""
        sql = """
            SELECT * FROM model_registry
            WHERE model_type = ?
            ORDER BY created_at DESC
            LIMIT ?
        """

        cursor = self.connection.execute(sql, (model_type, limit))
        return [self._row_to_metadata(row) for row in cursor.fetchall()]

    def _row_to_metadata(self, row: sqlite3.Row) -> ModelMetadata:
        """Convert row to ModelMetadata."""
        from trading_system.ml_refinement.config import ModelType

        return ModelMetadata(
            model_id=row["model_id"],
            model_type=ModelType(row["model_type"]),
            version=row["version"],
            created_at=row["created_at"],
            train_start_date=row["train_start_date"],
            train_end_date=row["train_end_date"],
            train_samples=row["train_samples"],
            validation_samples=row["validation_samples"],
            train_metrics=json.loads(row["train_metrics"]) if row["train_metrics"] else {},
            validation_metrics=json.loads(row["validation_metrics"]) if row["validation_metrics"] else {},
            feature_names=json.loads(row["feature_names"]) if row["feature_names"] else [],
            feature_importance=json.loads(row["feature_importance"]) if row["feature_importance"] else {},
            is_active=bool(row["is_active"]),
            deployed_at=row["deployed_at"],
        )

    # Prediction logging
    def log_prediction(
        self,
        signal_id: str,
        model_id: str,
        quality_score: float,
        predicted_r: Optional[float] = None,
        confidence: Optional[float] = None,
    ) -> bool:
        """Log a prediction for tracking."""
        sql = """
            INSERT INTO prediction_log (
                signal_id, model_id, predicted_at,
                quality_score, predicted_r, confidence
            ) VALUES (?, ?, ?, ?, ?, ?)
        """

        with self.transaction():
            self.connection.execute(sql, (
                signal_id,
                model_id,
                datetime.now().isoformat(),
                quality_score,
                predicted_r,
                confidence,
            ))

        return True

    def update_prediction_actual(
        self,
        signal_id: str,
        actual_r: float,
    ) -> bool:
        """Update prediction with actual outcome."""
        # Get predicted values
        sql = "SELECT quality_score, predicted_r FROM prediction_log WHERE signal_id = ?"
        cursor = self.connection.execute(sql, (signal_id,))
        row = cursor.fetchone()

        if not row:
            return False

        # Calculate error if we predicted R
        prediction_error = None
        if row["predicted_r"] is not None:
            prediction_error = actual_r - row["predicted_r"]

        sql = """
            UPDATE prediction_log
            SET actual_r = ?, prediction_error = ?
            WHERE signal_id = ?
        """

        with self.transaction():
            cursor = self.connection.execute(sql, (
                actual_r,
                prediction_error,
                signal_id,
            ))

        return cursor.rowcount > 0
```

3. Update `storage/__init__.py`:
```python
"""Storage implementations for ML refinement."""

from trading_system.ml_refinement.storage.feature_db import FeatureDatabase

__all__ = ["FeatureDatabase"]
```

**Acceptance Criteria**:
- [ ] Schema creates all tables
- [ ] Feature vector CRUD operations work
- [ ] Target updates work correctly
- [ ] Training data retrieval returns numpy arrays
- [ ] Model registry tracks versions
- [ ] Prediction logging works

**Tests to Write**:
```python
def test_store_and_retrieve_features(tmp_path):
    """Test feature storage roundtrip."""
    db = FeatureDatabase(str(tmp_path / "test.db"))
    db.initialize()

    fv = FeatureVector(
        signal_id="test-123",
        timestamp="2024-01-01T10:00:00",
        features={"rsi": 65.0, "atr": 2.5},
    )

    db.store_feature_vector(fv, symbol="AAPL")
    retrieved = db.get_feature_vector("test-123")

    assert retrieved is not None
    assert retrieved.features["rsi"] == 65.0
    db.close()

def test_get_training_data(tmp_path):
    """Test training data retrieval."""
    # Store multiple vectors with targets
    # Verify X, y shapes are correct
    pass
```

---

## Task 4.1.3: Implement Feature Extractors

**Context**:
Feature extractors compute predictive features from market data and signal metadata.

**Objective**:
Create base extractor and implement technical, market, and signal feature extractors.

**Files to Create**:
```
trading_system/ml_refinement/features/extractors/
├── __init__.py
├── base_extractor.py
├── technical_features.py
├── market_features.py
└── signal_features.py
```

**Requirements**:

1. Create `base_extractor.py`:
```python
"""Base class for feature extractors."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pandas as pd


class BaseFeatureExtractor(ABC):
    """
    Abstract base class for feature extractors.

    All extractors should:
    1. Have a unique name
    2. Define what features they provide
    3. Extract features from input data
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this extractor."""
        pass

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """List of feature names this extractor provides."""
        pass

    @property
    def category(self) -> str:
        """Category of features (technical, market, signal, news)."""
        return "general"

    @abstractmethod
    def extract(self, data: Any) -> Dict[str, float]:
        """
        Extract features from input data.

        Args:
            data: Input data (format depends on extractor type)

        Returns:
            Dictionary of feature_name -> value
        """
        pass

    def validate_output(self, features: Dict[str, float]) -> bool:
        """Validate extracted features."""
        expected = set(self.feature_names)
        actual = set(features.keys())
        return expected == actual


class OHLCVExtractor(BaseFeatureExtractor):
    """Base class for extractors that work with OHLCV data."""

    @abstractmethod
    def extract(self, ohlcv: pd.DataFrame, current_idx: int = -1) -> Dict[str, float]:
        """
        Extract features from OHLCV dataframe.

        Args:
            ohlcv: DataFrame with columns [open, high, low, close, volume]
            current_idx: Index of current bar (default: last bar)

        Returns:
            Dictionary of feature values
        """
        pass

    def _safe_get(
        self,
        series: pd.Series,
        idx: int,
        default: float = 0.0,
    ) -> float:
        """Safely get value from series."""
        try:
            val = series.iloc[idx]
            return float(val) if pd.notna(val) else default
        except (IndexError, KeyError):
            return default
```

2. Create `technical_features.py`:
```python
"""Technical indicator feature extractors."""

from typing import Dict, List

import numpy as np
import pandas as pd

from trading_system.ml_refinement.features.extractors.base_extractor import (
    OHLCVExtractor,
)


class TrendFeatures(OHLCVExtractor):
    """Extract trend-related features."""

    def __init__(self, lookbacks: List[int] = None):
        """
        Initialize with lookback periods.

        Args:
            lookbacks: List of lookback periods for moving averages.
        """
        self.lookbacks = lookbacks or [5, 10, 20, 50, 200]

    @property
    def name(self) -> str:
        return "trend_features"

    @property
    def category(self) -> str:
        return "technical"

    @property
    def feature_names(self) -> List[str]:
        names = []

        for lb in self.lookbacks:
            names.extend([
                f"price_vs_ma{lb}",        # Price relative to MA
                f"ma{lb}_slope",           # MA slope (normalized)
            ])

        names.extend([
            "price_vs_ma_fast_slow",       # Fast MA vs Slow MA
            "trend_strength",               # ADX-like measure
            "higher_highs",                 # Count of higher highs
            "lower_lows",                   # Count of lower lows
        ])

        return names

    def extract(self, ohlcv: pd.DataFrame, current_idx: int = -1) -> Dict[str, float]:
        """Extract trend features."""
        features = {}
        close = ohlcv["close"]
        high = ohlcv["high"]
        low = ohlcv["low"]

        current_price = self._safe_get(close, current_idx)

        # Price vs MAs and MA slopes
        for lb in self.lookbacks:
            if len(close) >= lb:
                ma = close.rolling(lb).mean()
                ma_val = self._safe_get(ma, current_idx)

                # Price relative to MA (normalized)
                if ma_val > 0:
                    features[f"price_vs_ma{lb}"] = (current_price - ma_val) / ma_val
                else:
                    features[f"price_vs_ma{lb}"] = 0.0

                # MA slope (5-day change, normalized by ATR)
                if len(ma) >= 5:
                    ma_prev = self._safe_get(ma, current_idx - 5)
                    atr = self._calculate_atr(ohlcv, 14)
                    if atr > 0 and ma_prev > 0:
                        features[f"ma{lb}_slope"] = (ma_val - ma_prev) / (atr * 5)
                    else:
                        features[f"ma{lb}_slope"] = 0.0
                else:
                    features[f"ma{lb}_slope"] = 0.0
            else:
                features[f"price_vs_ma{lb}"] = 0.0
                features[f"ma{lb}_slope"] = 0.0

        # Fast vs Slow MA
        if len(close) >= 50:
            ma_fast = close.rolling(10).mean().iloc[current_idx]
            ma_slow = close.rolling(50).mean().iloc[current_idx]
            if ma_slow > 0:
                features["price_vs_ma_fast_slow"] = (ma_fast - ma_slow) / ma_slow
            else:
                features["price_vs_ma_fast_slow"] = 0.0
        else:
            features["price_vs_ma_fast_slow"] = 0.0

        # Trend strength (simplified ADX-like)
        features["trend_strength"] = self._calculate_trend_strength(ohlcv, 14)

        # Higher highs / lower lows count (last 10 bars)
        features["higher_highs"] = self._count_higher_highs(high, 10)
        features["lower_lows"] = self._count_lower_lows(low, 10)

        return features

    def _calculate_atr(self, ohlcv: pd.DataFrame, period: int) -> float:
        """Calculate ATR."""
        high = ohlcv["high"]
        low = ohlcv["low"]
        close = ohlcv["close"]

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        atr = tr.rolling(period).mean()
        return self._safe_get(atr, -1)

    def _calculate_trend_strength(self, ohlcv: pd.DataFrame, period: int) -> float:
        """Calculate trend strength (0-1)."""
        close = ohlcv["close"]
        if len(close) < period:
            return 0.0

        # Use R-squared of linear regression as trend strength
        y = close.iloc[-period:].values
        x = np.arange(period)

        if np.std(y) == 0:
            return 0.0

        correlation = np.corrcoef(x, y)[0, 1]
        return correlation ** 2 if not np.isnan(correlation) else 0.0

    def _count_higher_highs(self, high: pd.Series, lookback: int) -> float:
        """Count consecutive higher highs."""
        if len(high) < lookback:
            return 0.0

        recent = high.iloc[-lookback:]
        count = 0
        for i in range(1, len(recent)):
            if recent.iloc[i] > recent.iloc[i - 1]:
                count += 1

        return count / (lookback - 1)  # Normalize to 0-1

    def _count_lower_lows(self, low: pd.Series, lookback: int) -> float:
        """Count consecutive lower lows."""
        if len(low) < lookback:
            return 0.0

        recent = low.iloc[-lookback:]
        count = 0
        for i in range(1, len(recent)):
            if recent.iloc[i] < recent.iloc[i - 1]:
                count += 1

        return count / (lookback - 1)


class MomentumFeatures(OHLCVExtractor):
    """Extract momentum-related features."""

    def __init__(self, lookbacks: List[int] = None):
        self.lookbacks = lookbacks or [5, 10, 20, 50]

    @property
    def name(self) -> str:
        return "momentum_features"

    @property
    def category(self) -> str:
        return "technical"

    @property
    def feature_names(self) -> List[str]:
        names = []

        for lb in self.lookbacks:
            names.append(f"roc_{lb}")       # Rate of change

        names.extend([
            "rsi_14",                       # RSI
            "rsi_deviation",                # RSI distance from 50
            "momentum_divergence",          # Price vs momentum divergence
            "acceleration",                 # Momentum acceleration
        ])

        return names

    def extract(self, ohlcv: pd.DataFrame, current_idx: int = -1) -> Dict[str, float]:
        """Extract momentum features."""
        features = {}
        close = ohlcv["close"]

        # Rate of change for various periods
        for lb in self.lookbacks:
            if len(close) > lb:
                current = self._safe_get(close, current_idx)
                past = self._safe_get(close, current_idx - lb)
                if past > 0:
                    features[f"roc_{lb}"] = (current - past) / past
                else:
                    features[f"roc_{lb}"] = 0.0
            else:
                features[f"roc_{lb}"] = 0.0

        # RSI
        rsi = self._calculate_rsi(close, 14)
        features["rsi_14"] = rsi / 100  # Normalize to 0-1
        features["rsi_deviation"] = (rsi - 50) / 50  # -1 to 1

        # Momentum divergence
        features["momentum_divergence"] = self._calculate_divergence(ohlcv, 14)

        # Acceleration
        features["acceleration"] = self._calculate_acceleration(close, 10)

        return features

    def _calculate_rsi(self, close: pd.Series, period: int) -> float:
        """Calculate RSI."""
        if len(close) < period + 1:
            return 50.0

        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))

        return self._safe_get(rsi, -1, 50.0)

    def _calculate_divergence(self, ohlcv: pd.DataFrame, period: int) -> float:
        """Calculate price-momentum divergence."""
        close = ohlcv["close"]
        if len(close) < period * 2:
            return 0.0

        # Price trend
        price_change = (close.iloc[-1] - close.iloc[-period]) / close.iloc[-period]

        # Momentum trend (ROC of ROC)
        roc = close.pct_change(period)
        if len(roc) >= period:
            mom_change = roc.iloc[-1] - roc.iloc[-period]
        else:
            mom_change = 0.0

        # Divergence is when they disagree
        if price_change > 0 and mom_change < 0:
            return -abs(mom_change)  # Bearish divergence
        elif price_change < 0 and mom_change > 0:
            return abs(mom_change)   # Bullish divergence

        return 0.0

    def _calculate_acceleration(self, close: pd.Series, period: int) -> float:
        """Calculate momentum acceleration."""
        if len(close) < period * 2:
            return 0.0

        roc = close.pct_change(period)
        roc_of_roc = roc.diff(period)

        return self._safe_get(roc_of_roc, -1, 0.0)


class VolatilityFeatures(OHLCVExtractor):
    """Extract volatility-related features."""

    @property
    def name(self) -> str:
        return "volatility_features"

    @property
    def category(self) -> str:
        return "technical"

    @property
    def feature_names(self) -> List[str]:
        return [
            "atr_ratio",                   # ATR vs price
            "volatility_percentile",       # Current vol vs historical
            "volatility_trend",            # Vol expanding or contracting
            "range_ratio",                 # Today's range vs avg
            "gap_size",                    # Gap from previous close
            "intraday_volatility",         # High-low / close
        ]

    def extract(self, ohlcv: pd.DataFrame, current_idx: int = -1) -> Dict[str, float]:
        """Extract volatility features."""
        features = {}
        high = ohlcv["high"]
        low = ohlcv["low"]
        close = ohlcv["close"]
        open_ = ohlcv["open"]

        current_close = self._safe_get(close, current_idx)

        # ATR ratio
        atr = self._calculate_atr(ohlcv, 14)
        if current_close > 0:
            features["atr_ratio"] = atr / current_close
        else:
            features["atr_ratio"] = 0.0

        # Volatility percentile
        features["volatility_percentile"] = self._vol_percentile(close, 20, 252)

        # Volatility trend
        features["volatility_trend"] = self._vol_trend(close, 20)

        # Range ratio
        avg_range = (high - low).rolling(20).mean()
        current_range = self._safe_get(high, current_idx) - self._safe_get(low, current_idx)
        avg_range_val = self._safe_get(avg_range, current_idx)
        if avg_range_val > 0:
            features["range_ratio"] = current_range / avg_range_val
        else:
            features["range_ratio"] = 1.0

        # Gap size
        if len(close) > 1:
            prev_close = self._safe_get(close, current_idx - 1)
            current_open = self._safe_get(open_, current_idx)
            if prev_close > 0:
                features["gap_size"] = (current_open - prev_close) / prev_close
            else:
                features["gap_size"] = 0.0
        else:
            features["gap_size"] = 0.0

        # Intraday volatility
        current_high = self._safe_get(high, current_idx)
        current_low = self._safe_get(low, current_idx)
        if current_close > 0:
            features["intraday_volatility"] = (current_high - current_low) / current_close
        else:
            features["intraday_volatility"] = 0.0

        return features

    def _calculate_atr(self, ohlcv: pd.DataFrame, period: int) -> float:
        """Calculate ATR."""
        high = ohlcv["high"]
        low = ohlcv["low"]
        close = ohlcv["close"]

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        atr = tr.rolling(period).mean()
        return self._safe_get(atr, -1)

    def _vol_percentile(
        self,
        close: pd.Series,
        short_period: int,
        long_period: int,
    ) -> float:
        """Calculate volatility percentile."""
        if len(close) < long_period:
            return 0.5

        returns = close.pct_change()
        current_vol = returns.iloc[-short_period:].std()
        historical_vol = returns.rolling(short_period).std()

        if len(historical_vol) < long_period:
            return 0.5

        percentile = (historical_vol.iloc[-long_period:] < current_vol).mean()
        return percentile

    def _vol_trend(self, close: pd.Series, period: int) -> float:
        """Calculate volatility trend (expanding = positive)."""
        if len(close) < period * 2:
            return 0.0

        returns = close.pct_change()
        vol = returns.rolling(period).std()

        current = self._safe_get(vol, -1)
        past = self._safe_get(vol, -period)

        if past > 0:
            return (current - past) / past
        return 0.0
```

3. Create `signal_features.py`:
```python
"""Signal metadata feature extractors."""

from typing import Any, Dict, List

from trading_system.ml_refinement.features.extractors.base_extractor import (
    BaseFeatureExtractor,
)


class SignalMetadataFeatures(BaseFeatureExtractor):
    """Extract features from signal metadata."""

    @property
    def name(self) -> str:
        return "signal_metadata"

    @property
    def category(self) -> str:
        return "signal"

    @property
    def feature_names(self) -> List[str]:
        return [
            "technical_score",
            "news_score",
            "combined_score",
            "conviction_high",
            "conviction_medium",
            "conviction_low",
            "risk_reward_ratio",
            "position_size",
            "is_equity",
            "is_crypto",
            "is_breakout",
            "is_momentum",
        ]

    def extract(self, signal_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract features from signal metadata.

        Args:
            signal_data: Dictionary with signal information including:
                - technical_score
                - news_score
                - combined_score
                - conviction
                - entry_price
                - target_price
                - stop_price
                - position_size_pct
                - asset_class
                - signal_type
        """
        features = {}

        # Scores (normalize to 0-1)
        features["technical_score"] = signal_data.get("technical_score", 0) / 10
        features["news_score"] = (signal_data.get("news_score") or 0) / 10
        features["combined_score"] = signal_data.get("combined_score", 0) / 10

        # Conviction one-hot encoding
        conviction = signal_data.get("conviction", "").upper()
        features["conviction_high"] = 1.0 if conviction == "HIGH" else 0.0
        features["conviction_medium"] = 1.0 if conviction == "MEDIUM" else 0.0
        features["conviction_low"] = 1.0 if conviction == "LOW" else 0.0

        # Risk/reward ratio
        entry = signal_data.get("entry_price", 0)
        target = signal_data.get("target_price", 0)
        stop = signal_data.get("stop_price", 0)

        if entry > 0 and stop > 0 and target > 0:
            risk = abs(entry - stop)
            reward = abs(target - entry)
            if risk > 0:
                features["risk_reward_ratio"] = reward / risk
            else:
                features["risk_reward_ratio"] = 0.0
        else:
            features["risk_reward_ratio"] = 0.0

        # Position size
        features["position_size"] = signal_data.get("position_size_pct", 0)

        # Asset class one-hot
        asset_class = signal_data.get("asset_class", "").lower()
        features["is_equity"] = 1.0 if asset_class == "equity" else 0.0
        features["is_crypto"] = 1.0 if asset_class == "crypto" else 0.0

        # Signal type one-hot
        signal_type = signal_data.get("signal_type", "").lower()
        features["is_breakout"] = 1.0 if "breakout" in signal_type else 0.0
        features["is_momentum"] = 1.0 if "momentum" in signal_type else 0.0

        return features
```

4. Create `market_features.py`:
```python
"""Market regime feature extractors."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from trading_system.ml_refinement.features.extractors.base_extractor import (
    OHLCVExtractor,
)


class MarketRegimeFeatures(OHLCVExtractor):
    """Extract market regime features."""

    @property
    def name(self) -> str:
        return "market_regime"

    @property
    def category(self) -> str:
        return "market"

    @property
    def feature_names(self) -> List[str]:
        return [
            "market_trend",               # Overall market direction
            "market_breadth",             # Approximated breadth
            "correlation_regime",         # High/low correlation
            "volatility_regime",          # Vol regime (0-1)
            "drawdown_depth",             # Current drawdown
            "days_from_high",             # Days since ATH
            "rally_strength",             # Strength of current rally
        ]

    def extract(
        self,
        ohlcv: pd.DataFrame,
        current_idx: int = -1,
        benchmark_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Extract market regime features.

        Args:
            ohlcv: Symbol OHLCV data
            current_idx: Current bar index
            benchmark_data: Optional benchmark (SPY) OHLCV for market features
        """
        features = {}

        # Use benchmark if available, otherwise use symbol data
        data = benchmark_data if benchmark_data is not None else ohlcv
        close = data["close"]

        # Market trend (-1 to 1)
        features["market_trend"] = self._calculate_trend(close, 50)

        # Market breadth approximation (using momentum)
        features["market_breadth"] = self._calculate_breadth_proxy(close, 20)

        # Correlation regime
        features["correlation_regime"] = self._calculate_correlation_regime(
            ohlcv["close"], close, 20
        )

        # Volatility regime
        features["volatility_regime"] = self._calculate_vol_regime(close, 20)

        # Drawdown depth
        features["drawdown_depth"] = self._calculate_drawdown(close)

        # Days from high
        features["days_from_high"] = self._days_from_high(close, 252)

        # Rally strength
        features["rally_strength"] = self._calculate_rally_strength(close, 20)

        return features

    def _calculate_trend(self, close: pd.Series, period: int) -> float:
        """Calculate trend direction and strength (-1 to 1)."""
        if len(close) < period:
            return 0.0

        ma = close.rolling(period).mean()
        current = close.iloc[-1]
        ma_val = ma.iloc[-1]

        if ma_val <= 0:
            return 0.0

        # Normalize by recent volatility
        returns = close.pct_change()
        vol = returns.iloc[-period:].std()

        if vol <= 0:
            return 0.0

        trend = (current - ma_val) / (ma_val * vol * np.sqrt(period))

        # Clip to -1, 1
        return max(-1.0, min(1.0, trend))

    def _calculate_breadth_proxy(self, close: pd.Series, period: int) -> float:
        """
        Approximate market breadth using momentum.

        In real implementation, this would use actual breadth data.
        """
        if len(close) < period:
            return 0.5

        returns = close.pct_change()
        positive_days = (returns.iloc[-period:] > 0).sum()

        return positive_days / period

    def _calculate_correlation_regime(
        self,
        symbol_close: pd.Series,
        market_close: pd.Series,
        period: int,
    ) -> float:
        """Calculate correlation with market."""
        if len(symbol_close) < period or len(market_close) < period:
            return 0.5

        symbol_returns = symbol_close.pct_change().iloc[-period:]
        market_returns = market_close.pct_change().iloc[-period:]

        if len(symbol_returns) != len(market_returns):
            return 0.5

        corr = symbol_returns.corr(market_returns)
        return corr if not np.isnan(corr) else 0.5

    def _calculate_vol_regime(self, close: pd.Series, period: int) -> float:
        """
        Calculate volatility regime (0 = low vol, 1 = high vol).
        """
        if len(close) < period * 2:
            return 0.5

        returns = close.pct_change()
        current_vol = returns.iloc[-period:].std()

        # Compare to longer-term vol
        long_vol = returns.iloc[-period * 2:-period].std()

        if long_vol <= 0:
            return 0.5

        vol_ratio = current_vol / long_vol

        # Normalize to 0-1 (0.5 = normal, 1 = 2x normal)
        return min(1.0, vol_ratio / 2)

    def _calculate_drawdown(self, close: pd.Series) -> float:
        """Calculate current drawdown from peak."""
        if len(close) < 1:
            return 0.0

        peak = close.expanding().max()
        drawdown = (close - peak) / peak

        return drawdown.iloc[-1]

    def _days_from_high(self, close: pd.Series, lookback: int) -> float:
        """Calculate days from recent high (normalized)."""
        if len(close) < lookback:
            lookback = len(close)

        if lookback < 1:
            return 0.0

        recent = close.iloc[-lookback:]
        high_idx = recent.idxmax()

        if isinstance(high_idx, int):
            days = len(close) - 1 - high_idx
        else:
            days = len(recent) - 1 - recent.index.get_loc(high_idx)

        # Normalize by lookback
        return days / lookback

    def _calculate_rally_strength(self, close: pd.Series, period: int) -> float:
        """Calculate strength of current rally/decline."""
        if len(close) < period:
            return 0.0

        start_price = close.iloc[-period]
        end_price = close.iloc[-1]

        if start_price <= 0:
            return 0.0

        # Calculate return and compare to volatility
        ret = (end_price - start_price) / start_price
        vol = close.pct_change().iloc[-period:].std()

        if vol <= 0:
            return 0.0

        # Sharpe-like measure
        strength = ret / (vol * np.sqrt(period))

        return max(-1.0, min(1.0, strength))
```

5. Create `extractors/__init__.py`:
```python
"""Feature extractors package."""

from trading_system.ml_refinement.features.extractors.base_extractor import (
    BaseFeatureExtractor,
    OHLCVExtractor,
)
from trading_system.ml_refinement.features.extractors.market_features import (
    MarketRegimeFeatures,
)
from trading_system.ml_refinement.features.extractors.signal_features import (
    SignalMetadataFeatures,
)
from trading_system.ml_refinement.features.extractors.technical_features import (
    MomentumFeatures,
    TrendFeatures,
    VolatilityFeatures,
)

__all__ = [
    "BaseFeatureExtractor",
    "OHLCVExtractor",
    "TrendFeatures",
    "MomentumFeatures",
    "VolatilityFeatures",
    "MarketRegimeFeatures",
    "SignalMetadataFeatures",
]
```

**Acceptance Criteria**:
- [ ] All extractors implement base interface
- [ ] Technical features extract from OHLCV data
- [ ] Market features work with benchmark data
- [ ] Signal features handle missing values
- [ ] All features normalized appropriately
- [ ] Edge cases (insufficient data) handled

**Tests to Write**:
```python
def test_trend_features():
    """Test trend feature extraction."""
    ohlcv = pd.DataFrame({
        "open": [100] * 250,
        "high": [101] * 250,
        "low": [99] * 250,
        "close": list(range(100, 350)),  # Uptrend
        "volume": [1000] * 250,
    })

    extractor = TrendFeatures()
    features = extractor.extract(ohlcv)

    assert "price_vs_ma20" in features
    assert features["price_vs_ma20"] > 0  # Price above MA in uptrend

def test_momentum_features_rsi():
    """Test RSI calculation."""
    pass

def test_signal_features():
    """Test signal metadata extraction."""
    signal_data = {
        "technical_score": 8.0,
        "conviction": "HIGH",
        "entry_price": 100,
        "target_price": 110,
        "stop_price": 95,
    }

    extractor = SignalMetadataFeatures()
    features = extractor.extract(signal_data)

    assert features["technical_score"] == 0.8  # Normalized
    assert features["conviction_high"] == 1.0
    assert features["risk_reward_ratio"] == 2.0  # 10/5
```

---

## Task 4.1.4: Implement Feature Pipeline

**Context**:
The feature pipeline orchestrates all extractors and produces complete feature vectors.

**Objective**:
Create a pipeline that combines all extractors and handles feature engineering.

**Files to Create**:
```
trading_system/ml_refinement/features/
├── pipeline.py
├── feature_registry.py
└── __init__.py
```

**Requirements**:

1. Create `feature_registry.py`:
```python
"""Registry of available features."""

from typing import Dict, List, Type

from trading_system.ml_refinement.features.extractors.base_extractor import (
    BaseFeatureExtractor,
)
from trading_system.ml_refinement.features.extractors.technical_features import (
    MomentumFeatures,
    TrendFeatures,
    VolatilityFeatures,
)
from trading_system.ml_refinement.features.extractors.market_features import (
    MarketRegimeFeatures,
)
from trading_system.ml_refinement.features.extractors.signal_features import (
    SignalMetadataFeatures,
)


class FeatureRegistry:
    """
    Registry of available feature extractors.

    Example:
        registry = FeatureRegistry()
        registry.register(CustomExtractor())

        extractor = registry.get("trend_features")
        all_names = registry.get_all_feature_names()
    """

    # Default extractors
    DEFAULT_EXTRACTORS: List[Type[BaseFeatureExtractor]] = [
        TrendFeatures,
        MomentumFeatures,
        VolatilityFeatures,
        MarketRegimeFeatures,
        SignalMetadataFeatures,
    ]

    def __init__(self):
        """Initialize with default extractors."""
        self._extractors: Dict[str, BaseFeatureExtractor] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register default extractors."""
        for extractor_class in self.DEFAULT_EXTRACTORS:
            extractor = extractor_class()
            self.register(extractor)

    def register(self, extractor: BaseFeatureExtractor):
        """Register an extractor."""
        self._extractors[extractor.name] = extractor

    def get(self, name: str) -> BaseFeatureExtractor:
        """Get extractor by name."""
        if name not in self._extractors:
            raise KeyError(f"Extractor '{name}' not found")
        return self._extractors[name]

    def get_all(self) -> List[BaseFeatureExtractor]:
        """Get all registered extractors."""
        return list(self._extractors.values())

    def get_by_category(self, category: str) -> List[BaseFeatureExtractor]:
        """Get extractors by category."""
        return [e for e in self._extractors.values() if e.category == category]

    def get_all_feature_names(self) -> List[str]:
        """Get all feature names across all extractors."""
        names = []
        for extractor in self._extractors.values():
            names.extend(extractor.feature_names)
        return names

    def get_feature_count(self) -> int:
        """Get total feature count."""
        return len(self.get_all_feature_names())
```

2. Create `pipeline.py`:
```python
"""Feature extraction pipeline."""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from trading_system.ml_refinement.config import FeatureConfig, FeatureSet, FeatureVector
from trading_system.ml_refinement.features.feature_registry import FeatureRegistry
from trading_system.ml_refinement.features.extractors.base_extractor import (
    OHLCVExtractor,
)


class FeaturePipeline:
    """
    Pipeline for extracting features from signals.

    Example:
        pipeline = FeaturePipeline(config)

        # Extract features for a signal
        features = pipeline.extract_features(
            signal_id="sig-123",
            ohlcv_data=ohlcv_df,
            signal_metadata=signal_dict,
            benchmark_data=spy_df,
        )

        # Create feature vector
        fv = pipeline.create_feature_vector(
            signal_id="sig-123",
            features=features,
            target=2.5,  # Optional, fill in later
        )
    """

    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        registry: Optional[FeatureRegistry] = None,
    ):
        """
        Initialize pipeline.

        Args:
            config: Feature configuration.
            registry: Feature registry (uses default if not provided).
        """
        self.config = config or FeatureConfig()
        self.registry = registry or FeatureRegistry()

        # Get extractors based on feature set
        self._extractors = self._get_extractors_for_set()

    def _get_extractors_for_set(self) -> List:
        """Get extractors for configured feature set."""
        if self.config.feature_set == FeatureSet.MINIMAL:
            # Only trend and signal features
            return [
                self.registry.get("trend_features"),
                self.registry.get("signal_metadata"),
            ]
        elif self.config.feature_set == FeatureSet.STANDARD:
            # Standard set
            return [
                self.registry.get("trend_features"),
                self.registry.get("momentum_features"),
                self.registry.get("volatility_features"),
                self.registry.get("signal_metadata"),
            ]
        elif self.config.feature_set == FeatureSet.EXTENDED:
            # All features
            return self.registry.get_all()
        else:
            # Custom - use specified features
            extractors = []
            for name in self.config.custom_features:
                try:
                    extractors.append(self.registry.get(name))
                except KeyError:
                    logger.warning(f"Unknown extractor: {name}")
            return extractors

    def extract_features(
        self,
        signal_id: str,
        ohlcv_data: pd.DataFrame,
        signal_metadata: Dict[str, Any],
        benchmark_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Extract all features for a signal.

        Args:
            signal_id: Unique signal identifier.
            ohlcv_data: OHLCV data for the symbol.
            signal_metadata: Signal metadata dictionary.
            benchmark_data: Optional benchmark OHLCV data.

        Returns:
            Dictionary of feature_name -> value.
        """
        all_features = {}

        for extractor in self._extractors:
            try:
                if isinstance(extractor, OHLCVExtractor):
                    # OHLCV-based extractor
                    if extractor.category == "market" and benchmark_data is not None:
                        features = extractor.extract(
                            ohlcv_data,
                            benchmark_data=benchmark_data,
                        )
                    else:
                        features = extractor.extract(ohlcv_data)
                else:
                    # Metadata-based extractor
                    features = extractor.extract(signal_metadata)

                all_features.update(features)

            except Exception as e:
                logger.warning(
                    f"Error extracting features from {extractor.name}: {e}"
                )
                # Fill with zeros for failed extractor
                for name in extractor.feature_names:
                    all_features[name] = 0.0

        # Apply scaling if configured
        if self.config.scale_features:
            all_features = self._scale_features(all_features)

        logger.debug(
            f"Extracted {len(all_features)} features for signal {signal_id}"
        )

        return all_features

    def create_feature_vector(
        self,
        signal_id: str,
        features: Dict[str, float],
        target: Optional[float] = None,
        timestamp: Optional[str] = None,
    ) -> FeatureVector:
        """
        Create a FeatureVector from extracted features.

        Args:
            signal_id: Signal identifier.
            features: Extracted features dictionary.
            target: Optional target value (R-multiple).
            timestamp: Optional timestamp (defaults to now).

        Returns:
            FeatureVector instance.
        """
        return FeatureVector(
            signal_id=signal_id,
            timestamp=timestamp or datetime.now().isoformat(),
            features=features,
            target=target,
            target_binary=1 if target and target > 0 else (0 if target else None),
        )

    def _scale_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Apply feature scaling.

        Note: In production, scaling parameters should be fitted on training
        data and stored for consistent scaling of new data.
        """
        # For now, just clip extreme values
        scaled = {}
        for name, value in features.items():
            # Clip to reasonable range
            scaled[name] = max(-10.0, min(10.0, value))
        return scaled

    def get_feature_names(self) -> List[str]:
        """Get all feature names produced by this pipeline."""
        names = []
        for extractor in self._extractors:
            names.extend(extractor.feature_names)
        return names

    def get_feature_count(self) -> int:
        """Get total feature count."""
        return len(self.get_feature_names())


class FeatureScaler:
    """
    Feature scaler for normalizing features.

    Fits scaling parameters on training data and applies to new data.
    """

    def __init__(self, method: str = "standard"):
        """
        Initialize scaler.

        Args:
            method: Scaling method ("standard", "minmax", "robust").
        """
        self.method = method
        self.fitted = False
        self._params: Dict[str, Dict] = {}

    def fit(self, feature_data: Dict[str, List[float]]):
        """
        Fit scaling parameters from training data.

        Args:
            feature_data: Dict of feature_name -> list of values.
        """
        import numpy as np

        for name, values in feature_data.items():
            values_arr = np.array(values)

            if self.method == "standard":
                self._params[name] = {
                    "mean": np.mean(values_arr),
                    "std": np.std(values_arr) or 1.0,
                }
            elif self.method == "minmax":
                self._params[name] = {
                    "min": np.min(values_arr),
                    "max": np.max(values_arr) or 1.0,
                }
            elif self.method == "robust":
                self._params[name] = {
                    "median": np.median(values_arr),
                    "iqr": np.percentile(values_arr, 75) - np.percentile(values_arr, 25) or 1.0,
                }

        self.fitted = True

    def transform(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Transform features using fitted parameters.

        Args:
            features: Feature dictionary to transform.

        Returns:
            Transformed features.
        """
        if not self.fitted:
            return features

        transformed = {}
        for name, value in features.items():
            if name not in self._params:
                transformed[name] = value
                continue

            params = self._params[name]

            if self.method == "standard":
                transformed[name] = (value - params["mean"]) / params["std"]
            elif self.method == "minmax":
                range_val = params["max"] - params["min"]
                if range_val > 0:
                    transformed[name] = (value - params["min"]) / range_val
                else:
                    transformed[name] = 0.0
            elif self.method == "robust":
                transformed[name] = (value - params["median"]) / params["iqr"]

        return transformed

    def fit_transform(
        self,
        feature_data: Dict[str, List[float]],
    ) -> List[Dict[str, float]]:
        """Fit and transform in one step."""
        self.fit(feature_data)

        # Reconstruct individual feature dicts
        n_samples = len(list(feature_data.values())[0])
        result = []

        for i in range(n_samples):
            sample = {name: values[i] for name, values in feature_data.items()}
            result.append(self.transform(sample))

        return result
```

3. Create `features/__init__.py`:
```python
"""Feature engineering package."""

from trading_system.ml_refinement.features.feature_registry import FeatureRegistry
from trading_system.ml_refinement.features.pipeline import (
    FeaturePipeline,
    FeatureScaler,
)

__all__ = [
    "FeaturePipeline",
    "FeatureRegistry",
    "FeatureScaler",
]
```

**Acceptance Criteria**:
- [ ] Pipeline combines all extractors
- [ ] Feature sets (minimal, standard, extended) work
- [ ] Custom feature selection works
- [ ] Scaler fits and transforms correctly
- [ ] Feature vector creation works
- [ ] Handles missing/invalid data gracefully

**Tests to Write**:
```python
def test_pipeline_standard_features():
    """Test standard feature set extraction."""
    config = FeatureConfig(feature_set=FeatureSet.STANDARD)
    pipeline = FeaturePipeline(config)

    ohlcv = create_sample_ohlcv(100)
    signal_meta = {"technical_score": 7.5, "conviction": "HIGH"}

    features = pipeline.extract_features(
        signal_id="test",
        ohlcv_data=ohlcv,
        signal_metadata=signal_meta,
    )

    assert len(features) > 0
    assert all(isinstance(v, float) for v in features.values())

def test_feature_scaler():
    """Test feature scaling."""
    scaler = FeatureScaler(method="standard")

    train_data = {
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
    }

    scaler.fit(train_data)

    test_features = {"feature1": 3.0, "feature2": 30.0}
    transformed = scaler.transform(test_features)

    # Mean of both is middle value, so should be ~0
    assert abs(transformed["feature1"]) < 0.1
    assert abs(transformed["feature2"]) < 0.1
```

---

## Task 4.1.5: Implement Base Model Interface

**Context**:
All ML models should follow a consistent interface for training and prediction.

**Objective**:
Create abstract base class and model registry for managing models.

**Files to Create**:
```
trading_system/ml_refinement/models/
├── __init__.py
├── base_model.py
└── model_registry.py
```

**Requirements**:

1. Create `base_model.py`:
```python
"""Base class for ML models."""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

import numpy as np
from loguru import logger

from trading_system.ml_refinement.config import ModelMetadata, ModelType


class BaseModel(ABC):
    """
    Abstract base class for all ML models.

    All models should implement:
    - fit: Train on data
    - predict: Make predictions
    - save/load: Persistence
    """

    def __init__(self, model_type: ModelType, version: str = "1.0"):
        """
        Initialize base model.

        Args:
            model_type: Type of model.
            version: Model version string.
        """
        self.model_type = model_type
        self.version = version
        self.model_id = f"{model_type.value}_{version}_{uuid.uuid4().hex[:8]}"
        self.is_fitted = False

        # Training metadata
        self._train_start_date: Optional[str] = None
        self._train_end_date: Optional[str] = None
        self._train_samples: int = 0
        self._validation_samples: int = 0
        self._feature_names: List[str] = []
        self._feature_importance: Dict[str, float] = {}
        self._train_metrics: Dict[str, float] = {}
        self._validation_metrics: Dict[str, float] = {}

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Train the model.

        Args:
            X_train: Training features (n_samples, n_features).
            y_train: Training targets (n_samples,).
            X_val: Optional validation features.
            y_val: Optional validation targets.
            feature_names: Optional list of feature names.

        Returns:
            Dictionary of training metrics.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features (n_samples, n_features).

        Returns:
            Predictions (n_samples,).
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities (for classification).

        Args:
            X: Features (n_samples, n_features).

        Returns:
            Probabilities (n_samples, n_classes) or (n_samples,) for binary.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> bool:
        """
        Save model to disk.

        Args:
            path: Path to save model.

        Returns:
            True if successful.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bool:
        """
        Load model from disk.

        Args:
            path: Path to load model from.

        Returns:
            True if successful.
        """
        pass

    def get_metadata(self) -> ModelMetadata:
        """Get model metadata."""
        return ModelMetadata(
            model_id=self.model_id,
            model_type=self.model_type,
            version=self.version,
            created_at=datetime.now().isoformat(),
            train_start_date=self._train_start_date or "",
            train_end_date=self._train_end_date or "",
            train_samples=self._train_samples,
            validation_samples=self._validation_samples,
            train_metrics=self._train_metrics,
            validation_metrics=self._validation_metrics,
            feature_names=self._feature_names,
            feature_importance=self._feature_importance,
        )

    def set_training_period(self, start_date: str, end_date: str):
        """Set training period for metadata."""
        self._train_start_date = start_date
        self._train_end_date = end_date


class SignalQualityModel(BaseModel):
    """
    Model for predicting signal quality (probability of success).

    Uses gradient boosting by default for interpretability and performance.
    """

    def __init__(self, version: str = "1.0", **kwargs):
        """
        Initialize signal quality model.

        Args:
            version: Model version.
            **kwargs: Additional model parameters.
        """
        super().__init__(ModelType.SIGNAL_QUALITY, version)

        self.params = {
            "n_estimators": kwargs.get("n_estimators", 100),
            "max_depth": kwargs.get("max_depth", 5),
            "learning_rate": kwargs.get("learning_rate", 0.1),
            "min_samples_leaf": kwargs.get("min_samples_leaf", 20),
            "subsample": kwargs.get("subsample", 0.8),
            "random_state": kwargs.get("random_state", 42),
        }

        self._model = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Train the signal quality model."""
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
        except ImportError:
            logger.error("scikit-learn required for ML models")
            raise

        # Convert to binary classification (win/loss)
        y_train_binary = (y_train > 0).astype(int)

        # Initialize model
        self._model = GradientBoostingClassifier(**self.params)

        # Fit
        self._model.fit(X_train, y_train_binary)

        # Store metadata
        self._train_samples = len(X_train)
        self._feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        # Calculate training metrics
        train_pred = self._model.predict(X_train)
        train_proba = self._model.predict_proba(X_train)[:, 1]

        self._train_metrics = {
            "accuracy": accuracy_score(y_train_binary, train_pred),
            "auc": roc_auc_score(y_train_binary, train_proba),
            "f1": f1_score(y_train_binary, train_pred),
        }

        # Validation metrics
        if X_val is not None and y_val is not None:
            y_val_binary = (y_val > 0).astype(int)
            val_pred = self._model.predict(X_val)
            val_proba = self._model.predict_proba(X_val)[:, 1]

            self._validation_samples = len(X_val)
            self._validation_metrics = {
                "accuracy": accuracy_score(y_val_binary, val_pred),
                "auc": roc_auc_score(y_val_binary, val_proba),
                "f1": f1_score(y_val_binary, val_pred),
            }

        # Feature importance
        if hasattr(self._model, "feature_importances_"):
            importances = self._model.feature_importances_
            self._feature_importance = {
                name: float(imp)
                for name, imp in zip(self._feature_names, importances)
            }

        self.is_fitted = True
        logger.info(
            f"Trained {self.model_type.value} model: "
            f"train_auc={self._train_metrics.get('auc', 0):.3f}"
        )

        return self._train_metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict win/loss (1/0)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of success (win)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self._model.predict_proba(X)[:, 1]

    def save(self, path: str) -> bool:
        """Save model to disk."""
        import pickle

        try:
            model_data = {
                "model": self._model,
                "params": self.params,
                "metadata": self.get_metadata(),
                "feature_names": self._feature_names,
            }

            with open(path, "wb") as f:
                pickle.dump(model_data, f)

            logger.info(f"Saved model to {path}")
            return True

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def load(self, path: str) -> bool:
        """Load model from disk."""
        import pickle

        try:
            with open(path, "rb") as f:
                model_data = pickle.load(f)

            self._model = model_data["model"]
            self.params = model_data["params"]
            self._feature_names = model_data.get("feature_names", [])

            metadata = model_data.get("metadata")
            if metadata:
                self.model_id = metadata.model_id
                self.version = metadata.version
                self._train_metrics = metadata.train_metrics
                self._validation_metrics = metadata.validation_metrics
                self._feature_importance = metadata.feature_importance

            self.is_fitted = True
            logger.info(f"Loaded model from {path}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        sorted_features = sorted(
            self._feature_importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_features[:n]
```

2. Create `model_registry.py`:
```python
"""Model registry for managing model versions."""

from pathlib import Path
from typing import Dict, List, Optional, Type

from loguru import logger

from trading_system.ml_refinement.config import ModelMetadata, ModelType
from trading_system.ml_refinement.models.base_model import BaseModel, SignalQualityModel
from trading_system.ml_refinement.storage.feature_db import FeatureDatabase


class ModelRegistry:
    """
    Registry for managing ML models.

    Example:
        registry = ModelRegistry(model_dir="models/", db=feature_db)

        # Train and register
        model = SignalQualityModel()
        model.fit(X_train, y_train)
        registry.register(model)

        # Activate for production
        registry.activate(model.model_id)

        # Get active model
        active = registry.get_active(ModelType.SIGNAL_QUALITY)
    """

    # Model class mapping
    MODEL_CLASSES: Dict[ModelType, Type[BaseModel]] = {
        ModelType.SIGNAL_QUALITY: SignalQualityModel,
    }

    def __init__(
        self,
        model_dir: str = "models/",
        db: Optional[FeatureDatabase] = None,
    ):
        """
        Initialize registry.

        Args:
            model_dir: Directory for storing model files.
            db: Feature database for metadata storage.
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.db = db

        # In-memory cache of active models
        self._active_models: Dict[ModelType, BaseModel] = {}

    def register(self, model: BaseModel) -> bool:
        """
        Register a trained model.

        Args:
            model: Trained model to register.

        Returns:
            True if successful.
        """
        if not model.is_fitted:
            logger.error("Cannot register unfitted model")
            return False

        # Save model file
        model_path = self.model_dir / f"{model.model_id}.pkl"
        if not model.save(str(model_path)):
            return False

        # Save metadata to database
        if self.db:
            metadata = model.get_metadata()
            self.db.register_model(metadata)

        logger.info(f"Registered model {model.model_id}")
        return True

    def activate(self, model_id: str) -> bool:
        """
        Activate a model for production use.

        Args:
            model_id: ID of model to activate.

        Returns:
            True if successful.
        """
        # Load model
        model_path = self.model_dir / f"{model_id}.pkl"
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False

        # Determine model type from ID
        model_type = None
        for mt in ModelType:
            if model_id.startswith(mt.value):
                model_type = mt
                break

        if not model_type:
            logger.error(f"Cannot determine model type from ID: {model_id}")
            return False

        # Load model
        model_class = self.MODEL_CLASSES.get(model_type)
        if not model_class:
            logger.error(f"No model class for type: {model_type}")
            return False

        model = model_class()
        if not model.load(str(model_path)):
            return False

        # Update database
        if self.db:
            self.db.activate_model(model_id)

        # Cache active model
        self._active_models[model_type] = model

        logger.info(f"Activated model {model_id}")
        return True

    def get_active(self, model_type: ModelType) -> Optional[BaseModel]:
        """
        Get the active model for a type.

        Args:
            model_type: Type of model.

        Returns:
            Active model or None.
        """
        # Check cache first
        if model_type in self._active_models:
            return self._active_models[model_type]

        # Load from database
        if self.db:
            metadata = self.db.get_active_model(model_type.value)
            if metadata:
                return self._load_model(metadata.model_id, model_type)

        return None

    def get_model_history(
        self,
        model_type: ModelType,
        limit: int = 10,
    ) -> List[ModelMetadata]:
        """Get model version history."""
        if not self.db:
            return []

        return self.db.get_model_history(model_type.value, limit)

    def _load_model(
        self,
        model_id: str,
        model_type: ModelType,
    ) -> Optional[BaseModel]:
        """Load a model by ID."""
        model_path = self.model_dir / f"{model_id}.pkl"
        if not model_path.exists():
            return None

        model_class = self.MODEL_CLASSES.get(model_type)
        if not model_class:
            return None

        model = model_class()
        if model.load(str(model_path)):
            self._active_models[model_type] = model
            return model

        return None

    def compare_models(
        self,
        model_type: ModelType,
        metric: str = "auc",
    ) -> List[Dict]:
        """
        Compare all models of a type by metric.

        Args:
            model_type: Type of models to compare.
            metric: Metric to compare by.

        Returns:
            List of models sorted by metric.
        """
        history = self.get_model_history(model_type, limit=50)

        results = []
        for metadata in history:
            val_metric = metadata.validation_metrics.get(metric, 0)
            results.append({
                "model_id": metadata.model_id,
                "version": metadata.version,
                "created_at": metadata.created_at,
                "metric": val_metric,
                "is_active": metadata.is_active,
            })

        return sorted(results, key=lambda x: x["metric"], reverse=True)
```

3. Create `models/__init__.py`:
```python
"""ML models package."""

from trading_system.ml_refinement.models.base_model import (
    BaseModel,
    SignalQualityModel,
)
from trading_system.ml_refinement.models.model_registry import ModelRegistry

__all__ = [
    "BaseModel",
    "SignalQualityModel",
    "ModelRegistry",
]
```

**Acceptance Criteria**:
- [ ] BaseModel defines complete interface
- [ ] SignalQualityModel trains and predicts correctly
- [ ] Model save/load works
- [ ] Registry manages model versions
- [ ] Active model retrieval works
- [ ] Feature importance tracked

**Tests to Write**:
```python
def test_signal_quality_model_fit():
    """Test model training."""
    model = SignalQualityModel()

    X = np.random.randn(100, 10)
    y = np.random.randn(100)  # Will be converted to binary

    metrics = model.fit(X, y)

    assert model.is_fitted
    assert "accuracy" in metrics
    assert "auc" in metrics

def test_model_save_load(tmp_path):
    """Test model persistence."""
    model = SignalQualityModel()
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    model.fit(X, y)

    # Save
    path = str(tmp_path / "model.pkl")
    model.save(path)

    # Load
    loaded = SignalQualityModel()
    loaded.load(path)

    assert loaded.is_fitted

    # Predictions should match
    pred1 = model.predict_proba(X[:5])
    pred2 = loaded.predict_proba(X[:5])
    np.testing.assert_array_almost_equal(pred1, pred2)
```

---

## Summary: Part 1 Tasks

| Task | Description | Key Deliverable |
|------|-------------|-----------------|
| 4.1.1 | Module Structure | Config, FeatureVector, ModelMetadata |
| 4.1.2 | Feature Database | SQLite schema, CRUD operations |
| 4.1.3 | Feature Extractors | Technical, market, signal extractors |
| 4.1.4 | Feature Pipeline | Pipeline, registry, scaler |
| 4.1.5 | Base Model | SignalQualityModel, ModelRegistry |

---

**Part 2 will cover**:
- 4.2.1: Walk-Forward Validation Framework
- 4.2.2: Training Pipeline
- 4.2.3: Model Evaluation Metrics
- 4.2.4: Signal Scoring Integration
- 4.2.5: Retraining Job
- 4.2.6: Parameter Optimization
- 4.2.7: CLI Commands
- 4.2.8: Integration Tests

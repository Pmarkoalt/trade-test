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
            self.connection.execute(
                sql,
                (
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
                ),
            )

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
            cursor = self.connection.execute(
                sql,
                (
                    r_multiple,
                    target_binary,
                    return_pct,
                    datetime.now().isoformat(),
                    signal_id,
                ),
            )

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
        X = np.array([[fd.get(key, 0.0) for key in feature_names] for fd in feature_dicts])
        y = np.array(targets)

        return X, y, feature_names

    def get_training_data_binary(
        self,
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Get training data with binary target (win/loss)."""
        X, y, feature_names = self.get_training_data(start_date, end_date, **kwargs)

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
        result = cursor.fetchone()
        return int(result[0]) if result and result[0] is not None else 0

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
            self.connection.execute(
                sql,
                (
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
                ),
            )

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
            self.connection.execute("UPDATE model_registry SET is_active = 0 WHERE model_type = ?", (model_type,))

            # Activate this model
            self.connection.execute(
                """UPDATE model_registry
                   SET is_active = 1, deployed_at = ?
                   WHERE model_id = ?""",
                (datetime.now().isoformat(), model_id),
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
            self.connection.execute(
                sql,
                (
                    signal_id,
                    model_id,
                    datetime.now().isoformat(),
                    quality_score,
                    predicted_r,
                    confidence,
                ),
            )

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
            cursor = self.connection.execute(
                sql,
                (
                    actual_r,
                    prediction_error,
                    signal_id,
                ),
            )

        return cursor.rowcount > 0

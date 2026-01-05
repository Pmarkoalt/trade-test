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

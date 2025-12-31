-- Performance Tracking Schema
-- Migration: 001_initial_schema
-- Created: 2024-12-30

-- Tracked Signals Table
CREATE TABLE IF NOT EXISTS tracked_signals (
    id TEXT PRIMARY KEY,

    -- Signal details
    symbol TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    direction TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    conviction TEXT NOT NULL,

    -- Prices
    signal_price REAL NOT NULL,
    entry_price REAL NOT NULL,
    target_price REAL NOT NULL,
    stop_price REAL NOT NULL,

    -- Scores
    technical_score REAL,
    news_score REAL,
    combined_score REAL,

    -- Sizing
    position_size_pct REAL,

    -- Status
    status TEXT NOT NULL DEFAULT 'pending',

    -- Timestamps
    created_at TEXT NOT NULL,
    delivered_at TEXT,
    entry_filled_at TEXT,
    exit_filled_at TEXT,

    -- Delivery
    was_delivered INTEGER NOT NULL DEFAULT 0,
    delivery_method TEXT,

    -- Metadata
    reasoning TEXT,
    news_headlines TEXT,  -- JSON array
    tags TEXT,            -- JSON array

    -- Indexes
    created_date TEXT GENERATED ALWAYS AS (date(created_at)) STORED
);

CREATE INDEX IF NOT EXISTS idx_signals_symbol ON tracked_signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_created_date ON tracked_signals(created_date);
CREATE INDEX IF NOT EXISTS idx_signals_status ON tracked_signals(status);
CREATE INDEX IF NOT EXISTS idx_signals_asset_class ON tracked_signals(asset_class);


-- Signal Outcomes Table
CREATE TABLE IF NOT EXISTS signal_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id TEXT NOT NULL UNIQUE,

    -- Actual execution
    actual_entry_price REAL,
    actual_entry_date TEXT,
    actual_exit_price REAL,
    actual_exit_date TEXT,

    -- Trade result
    exit_reason TEXT,
    holding_days INTEGER,

    -- Returns
    return_pct REAL,
    return_dollars REAL,
    r_multiple REAL,

    -- Benchmark
    benchmark_return_pct REAL,
    alpha REAL,

    -- User feedback
    was_followed INTEGER NOT NULL DEFAULT 0,
    user_notes TEXT,

    -- Timestamps
    recorded_at TEXT NOT NULL,

    FOREIGN KEY (signal_id) REFERENCES tracked_signals(id)
);

CREATE INDEX IF NOT EXISTS idx_outcomes_signal_id ON signal_outcomes(signal_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_exit_date ON signal_outcomes(actual_exit_date);


-- Daily Performance Snapshots
CREATE TABLE IF NOT EXISTS daily_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date TEXT NOT NULL UNIQUE,

    -- Cumulative metrics
    total_signals INTEGER NOT NULL DEFAULT 0,
    total_closed INTEGER NOT NULL DEFAULT 0,
    total_wins INTEGER NOT NULL DEFAULT 0,
    total_losses INTEGER NOT NULL DEFAULT 0,

    -- Returns
    cumulative_return_pct REAL,
    cumulative_r REAL,

    -- Rolling metrics (last 30 days)
    rolling_win_rate REAL,
    rolling_avg_r REAL,
    rolling_sharpe REAL,

    -- Equity curve
    starting_equity REAL,
    current_equity REAL,
    high_water_mark REAL,
    current_drawdown_pct REAL,

    -- Timestamp
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_daily_perf_date ON daily_performance(snapshot_date);


-- Strategy Performance (aggregated by signal type)
CREATE TABLE IF NOT EXISTS strategy_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_type TEXT NOT NULL,
    period_type TEXT NOT NULL,  -- 'daily', 'weekly', 'monthly', 'all_time'
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,

    -- Counts
    total_signals INTEGER NOT NULL DEFAULT 0,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,

    -- Metrics
    win_rate REAL,
    avg_return_pct REAL,
    avg_r REAL,
    expectancy_r REAL,
    sharpe_ratio REAL,

    -- Timestamp
    calculated_at TEXT NOT NULL,

    UNIQUE(signal_type, period_type, period_start)
);


-- Metadata table for migrations
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

INSERT OR IGNORE INTO schema_migrations (version, applied_at)
VALUES (1, datetime('now'));


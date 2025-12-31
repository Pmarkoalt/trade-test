"""Database schema definitions for results storage."""

import sqlite3
from typing import List, Optional


def create_schema(conn: sqlite3.Connection) -> None:
    """Create all database tables.

    Args:
        conn: SQLite database connection
    """
    cursor = conn.cursor()

    # Backtest runs table - stores metadata about each backtest run
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_path TEXT,
            strategy_name TEXT,
            split_name TEXT,
            period TEXT NOT NULL,  -- train, validation, holdout
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            starting_equity REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notes TEXT,
            UNIQUE(config_path, strategy_name, split_name, period, start_date, end_date)
        )
    """
    )

    # Results metrics table - stores computed metrics for each run
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS run_metrics (
            metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            sharpe_ratio REAL,
            max_drawdown REAL,
            calmar_ratio REAL,
            total_return REAL,
            total_trades INTEGER,
            winning_trades INTEGER,
            losing_trades INTEGER,
            win_rate REAL,
            avg_r_multiple REAL,
            realized_pnl REAL,
            final_cash REAL,
            final_positions INTEGER,
            ending_equity REAL,
            expectancy REAL,
            profit_factor REAL,
            correlation_to_benchmark REAL,
            percentile_99_daily_loss REAL,
            recovery_factor REAL,
            drawdown_duration INTEGER,
            turnover REAL,
            average_holding_period REAL,
            max_consecutive_losses INTEGER,
            FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id) ON DELETE CASCADE,
            UNIQUE(run_id)
        )
    """
    )

    # Trades table - stores individual closed trades
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            asset_class TEXT NOT NULL,
            entry_date TEXT NOT NULL,
            exit_date TEXT,
            entry_price REAL NOT NULL,
            exit_price REAL,
            quantity INTEGER NOT NULL,
            realized_pnl REAL,
            r_multiple REAL,
            exit_reason TEXT,
            entry_fill_id TEXT,
            exit_fill_id TEXT,
            entry_slippage_bps REAL,
            exit_slippage_bps REAL,
            entry_fee_bps REAL,
            exit_fee_bps REAL,
            entry_total_cost REAL,
            exit_total_cost REAL,
            initial_stop_price REAL,
            adv20_at_entry REAL,
            FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id) ON DELETE CASCADE
        )
    """
    )

    # Equity curve table - stores daily equity values
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS equity_curve (
            equity_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            equity REAL NOT NULL,
            cash REAL,
            open_positions INTEGER,
            gross_exposure REAL,
            FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id) ON DELETE CASCADE,
            UNIQUE(run_id, date)
        )
    """
    )

    # Daily returns table - stores daily return values
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_returns (
            return_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            daily_return REAL NOT NULL,
            FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id) ON DELETE CASCADE,
            UNIQUE(run_id, date)
        )
    """
    )

    # Monthly summary table - stores aggregated monthly metrics
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS monthly_summary (
            monthly_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            month TEXT NOT NULL,  -- YYYY-MM format
            month_start TEXT NOT NULL,
            month_end TEXT NOT NULL,
            start_equity REAL NOT NULL,
            end_equity REAL NOT NULL,
            monthly_return REAL NOT NULL,
            trades_count INTEGER,
            winning_trades INTEGER,
            losing_trades INTEGER,
            win_rate REAL,
            realized_pnl REAL,
            volatility_annualized REAL,
            sharpe_ratio REAL,
            max_drawdown REAL,
            FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id) ON DELETE CASCADE,
            UNIQUE(run_id, month)
        )
    """
    )

    # Create indexes for better query performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_config ON backtest_runs(config_path)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_strategy ON backtest_runs(strategy_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_period ON backtest_runs(period)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_dates ON backtest_runs(start_date, end_date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_run ON trades(run_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_equity_run ON equity_curve(run_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_equity_date ON equity_curve(date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_returns_run ON daily_returns(run_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_monthly_run ON monthly_summary(run_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_monthly_month ON monthly_summary(month)")

    conn.commit()


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Get current schema version.

    Args:
        conn: SQLite database connection

    Returns:
        Schema version number (1 for initial schema)
    """
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
        result = cursor.fetchone()
        if result and result[0] is not None:
            version_val = result[0]
            return int(version_val) if isinstance(version_val, (int, str)) else 1
        return 1
    except sqlite3.OperationalError:
        # Schema version table doesn't exist yet
        return 1


def migrate_schema(conn: sqlite3.Connection, from_version: int, to_version: int) -> None:
    """Migrate database schema from one version to another.

    Args:
        conn: SQLite database connection
        from_version: Current schema version
        to_version: Target schema version
    """
    # For now, we only have version 1, so no migrations needed
    # This is a placeholder for future schema migrations
    pass

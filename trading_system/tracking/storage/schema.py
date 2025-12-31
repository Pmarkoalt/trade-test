"""Database schema definitions."""

from typing import List, Tuple

# Table definitions for validation
TABLES = {
    "tracked_signals": [
        "id",
        "symbol",
        "asset_class",
        "direction",
        "signal_type",
        "conviction",
        "signal_price",
        "entry_price",
        "target_price",
        "stop_price",
        "technical_score",
        "news_score",
        "combined_score",
        "position_size_pct",
        "status",
        "created_at",
        "delivered_at",
        "entry_filled_at",
        "exit_filled_at",
        "was_delivered",
        "delivery_method",
        "reasoning",
        "news_headlines",
        "tags",
    ],
    "signal_outcomes": [
        "id",
        "signal_id",
        "actual_entry_price",
        "actual_entry_date",
        "actual_exit_price",
        "actual_exit_date",
        "exit_reason",
        "holding_days",
        "return_pct",
        "return_dollars",
        "r_multiple",
        "benchmark_return_pct",
        "alpha",
        "was_followed",
        "user_notes",
        "recorded_at",
    ],
    "daily_performance": [
        "id",
        "snapshot_date",
        "total_signals",
        "total_closed",
        "total_wins",
        "total_losses",
        "cumulative_return_pct",
        "cumulative_r",
        "rolling_win_rate",
        "rolling_avg_r",
        "rolling_sharpe",
        "starting_equity",
        "current_equity",
        "high_water_mark",
        "current_drawdown_pct",
        "created_at",
    ],
    "strategy_performance": [
        "id",
        "signal_type",
        "period_type",
        "period_start",
        "period_end",
        "total_signals",
        "wins",
        "losses",
        "win_rate",
        "avg_return_pct",
        "avg_r",
        "expectancy_r",
        "sharpe_ratio",
        "calculated_at",
    ],
}


def get_migration_files() -> List[Tuple[int, str]]:
    """Get ordered list of migration files.

    Returns:
        List of tuples (version_number, file_path) sorted by version
    """
    import os
    from pathlib import Path

    migrations_dir = Path(__file__).parent / "migrations"
    migrations = []

    for f in migrations_dir.glob("*.sql"):
        # Extract version number from filename (e.g., "001_initial_schema.sql" -> 1)
        version = int(f.stem.split("_")[0])
        migrations.append((version, str(f)))

    return sorted(migrations, key=lambda x: x[0])


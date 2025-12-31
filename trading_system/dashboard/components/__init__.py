"""Dashboard components."""

from trading_system.dashboard.components.charts import (
    create_conviction_breakdown,
    create_equity_curve,
    create_monthly_performance_heatmap,
    create_performance_by_day,
    create_returns_distribution,
    create_strategy_comparison_chart,
    create_win_rate_gauge,
)
from trading_system.dashboard.components.tables import (
    render_leaderboard_table,
    render_performance_table,
    render_recent_trades_table,
    render_signals_table,
)
from trading_system.dashboard.components.cards import (
    render_insight_box,
    render_metric_card,
    render_metric_row,
    render_signal_card,
    render_status_badge,
    render_streak_indicator,
)

__all__ = [
    # Charts
    "create_equity_curve",
    "create_win_rate_gauge",
    "create_returns_distribution",
    "create_strategy_comparison_chart",
    "create_performance_by_day",
    "create_conviction_breakdown",
    "create_monthly_performance_heatmap",
    # Tables
    "render_signals_table",
    "render_performance_table",
    "render_leaderboard_table",
    "render_recent_trades_table",
    # Cards
    "render_metric_card",
    "render_metric_row",
    "render_signal_card",
    "render_insight_box",
    "render_status_badge",
    "render_streak_indicator",
]


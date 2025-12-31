"""Overview page for dashboard."""

import streamlit as st

from trading_system.dashboard.config import DashboardConfig
from trading_system.dashboard.services.data_service import DashboardDataService
from trading_system.dashboard.services.cache_service import get_cached_dashboard_data
from trading_system.dashboard.components.charts import (
    create_equity_curve,
    create_win_rate_gauge,
    create_strategy_comparison_chart,
)
from trading_system.dashboard.components.tables import (
    render_recent_trades_table,
    render_leaderboard_table,
)
from trading_system.dashboard.components.cards import (
    render_metric_row,
    render_signal_card,
    render_insight_box,
    render_streak_indicator,
)


def render_overview(config: DashboardConfig):
    """
    Render the overview page.

    Args:
        config: Dashboard configuration.
    """
    st.title("📊 Dashboard Overview")

    # Get data
    service = DashboardDataService(
        tracking_db_path=config.tracking_db_path,
        feature_db_path=config.feature_db_path,
    )

    try:
        data = get_cached_dashboard_data(service)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Make sure the tracking database exists and has data.")
        return

    # Last updated
    st.caption(f"Last updated: {data.fetched_at.strftime('%Y-%m-%d %H:%M:%S')}")

    # Key metrics row
    st.subheader("Key Metrics (30 Days)")

    metrics_30d = data.metrics_30d
    render_metric_row(
        [
            {
                "label": "Total Return",
                "value": f"{metrics_30d.get('total_r', 0):+.1f}R",
                "delta": f"{metrics_30d.get('avg_r', 0):+.2f}R avg",
                "help": "Total R-multiple return",
            },
            {
                "label": "Win Rate",
                "value": f"{metrics_30d.get('win_rate', 0) * 100:.0f}%",
                "help": "Percentage of winning trades",
            },
            {
                "label": "Expectancy",
                "value": f"{metrics_30d.get('expectancy_r', 0):.2f}R",
                "help": "Expected R per trade",
            },
            {
                "label": "Trades",
                "value": str(metrics_30d.get("total_signals", 0)),
                "help": "Number of trades taken",
            },
        ]
    )

    # Streak indicator
    analytics = data.analytics
    if analytics.get("current_streak", 0) >= 3:
        render_streak_indicator(
            streak_count=analytics.get("current_streak", 0),
            streak_type=analytics.get("current_streak_type", ""),
        )

    st.divider()

    # Two columns: Equity curve and signals
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Equity Curve")

        # Get performance timeseries
        perf_service = DashboardDataService(config.tracking_db_path)
        perf_df = perf_service.get_performance_timeseries(days=90)

        if not perf_df.empty:
            fig = create_equity_curve(perf_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No equity curve data available yet")

    with col2:
        st.subheader("Win Rate")
        fig = create_win_rate_gauge(metrics_30d.get("win_rate", 0.5))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Recent signals and trades
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Active Signals")

        if data.active_signals:
            for signal in data.active_signals[:3]:
                render_signal_card(
                    {
                        "symbol": signal.symbol,
                        "direction": signal.direction.value,
                        "conviction": signal.conviction.value,
                        "combined_score": signal.combined_score,
                        "signal_type": signal.signal_type,
                        "entry_price": signal.entry_price,
                        "target_price": signal.target_price,
                        "stop_price": signal.stop_price,
                    }
                )

            if len(data.active_signals) > 3:
                st.caption(f"+ {len(data.active_signals) - 3} more active signals")
        else:
            st.info("No active signals")

    with col2:
        st.subheader("Recent Trades")
        recent_trades = analytics.get("last_10_trades", [])
        render_recent_trades_table(recent_trades, max_rows=5)

    st.divider()

    # Strategy leaderboard
    st.subheader("Strategy Leaderboard")

    leaderboard = data.leaderboard
    if leaderboard.get("entries"):
        col1, col2 = st.columns([1, 2])

        with col1:
            render_leaderboard_table(leaderboard)

        with col2:
            strategy_df = service.get_strategy_comparison()
            if not strategy_df.empty:
                fig = create_strategy_comparison_chart(strategy_df, metric="total_r")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No strategy data available yet")

    st.divider()

    # Insights
    st.subheader("Insights")

    insights = analytics.get("insights", [])
    if insights:
        for insight in insights[:5]:
            if "WARNING" in insight:
                render_insight_box(insight, type="warning")
            elif "Strong" in insight or "Best" in insight:
                render_insight_box(insight, type="success")
            else:
                render_insight_box(insight, type="info")
    else:
        st.info("Generate more trades to get insights")

    # Signal counts
    st.divider()
    st.subheader("Signal Status")

    counts = data.signal_counts
    if counts:
        cols = st.columns(4)
        statuses = ["pending", "active", "closed", "expired"]

        for col, status in zip(cols, statuses):
            with col:
                count = counts.get(status, 0)
                st.metric(status.title(), count)

"""Performance page for dashboard."""

import streamlit as st
import pandas as pd
from datetime import datetime

from trading_system.dashboard.config import DashboardConfig
from trading_system.dashboard.services.data_service import DashboardDataService
from trading_system.dashboard.services.cache_service import (
    get_cached_dashboard_data,
    get_cached_performance_ts,
    get_cached_strategy_comparison,
)
from trading_system.dashboard.components.charts import (
    create_equity_curve,
    create_win_rate_gauge,
    create_returns_distribution,
    create_strategy_comparison_chart,
    create_performance_by_day,
    create_conviction_breakdown,
)
from trading_system.dashboard.components.tables import render_performance_table
from trading_system.dashboard.components.cards import render_metric_row, render_insight_box


def render_performance(config: DashboardConfig):
    """
    Render the performance page.

    Args:
        config: Dashboard configuration.
    """
    st.title("📈 Performance Analytics")

    # Initialize service
    service = DashboardDataService(
        tracking_db_path=config.tracking_db_path,
    )

    # Time period selector
    col1, col2 = st.columns([3, 1])
    with col2:
        period = st.selectbox(
            "Time Period",
            options=[30, 60, 90, 180, 365],
            index=2,
            format_func=lambda x: f"{x} days" if x < 365 else "1 year",
        )

    try:
        data = get_cached_dashboard_data(service)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Determine which metrics to use based on period
    if period <= 30:
        metrics = data.metrics_30d
    elif period <= 90:
        metrics = data.metrics_90d
    else:
        metrics = data.metrics_all

    # Key metrics
    st.subheader("Summary Metrics")

    render_metric_row(
        [
            {
                "label": "Total Return",
                "value": f"{metrics.get('total_r', 0):+.1f}R",
            },
            {
                "label": "Win Rate",
                "value": f"{metrics.get('win_rate', 0) * 100:.1f}%",
            },
            {
                "label": "Expectancy",
                "value": f"{metrics.get('expectancy_r', 0):.2f}R",
            },
            {
                "label": "Sharpe Ratio",
                "value": f"{metrics.get('sharpe_ratio', 0):.2f}",
            },
            {
                "label": "Max Drawdown",
                "value": f"{metrics.get('max_drawdown_pct', 0) * 100:.1f}%",
            },
        ]
    )

    st.divider()

    # Equity curve
    st.subheader("Equity Curve")

    perf_df = get_cached_performance_ts(service, days=period)

    if not perf_df.empty:
        fig = create_equity_curve(perf_df)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No equity data available for this period")

    st.divider()

    # Two columns: Win rate gauge and returns distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Win Rate")
        fig = create_win_rate_gauge(metrics.get("win_rate", 0.5))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Returns Distribution")
        # Get R-multiples from analytics
        analytics = data.analytics
        recent_trades = analytics.get("last_10_trades", [])

        if recent_trades:
            r_values = [t.get("r_multiple", 0) for t in recent_trades]
            fig = create_returns_distribution(r_values)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough trades for distribution")

    st.divider()

    # Detailed metrics table
    st.subheader("Detailed Metrics")
    render_performance_table(metrics)

    st.divider()

    # Analytics breakdowns
    st.subheader("Performance Breakdowns")

    tab1, tab2, tab3 = st.tabs(["By Day", "By Conviction", "By Strategy"])

    with tab1:
        fig = create_performance_by_day(data.analytics)
        if fig.data:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for day analysis")

    with tab2:
        fig = create_conviction_breakdown(data.analytics)
        if fig.data:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for conviction analysis")

    with tab3:
        strategy_df = get_cached_strategy_comparison(service)

        if not strategy_df.empty:
            # Metric selector
            metric = st.selectbox(
                "Compare by",
                options=["total_r", "win_rate", "expectancy"],
                format_func=lambda x: x.replace("_", " ").title(),
            )

            fig = create_strategy_comparison_chart(strategy_df, metric=metric)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No strategy data available")

    st.divider()

    # Insights
    st.subheader("Performance Insights")

    insights = data.analytics.get("insights", [])

    if insights:
        col1, col2 = st.columns(2)

        for i, insight in enumerate(insights):
            with col1 if i % 2 == 0 else col2:
                if "WARNING" in insight:
                    render_insight_box(insight, type="danger")
                elif "Strong" in insight or "Best" in insight or "outperform" in insight:
                    render_insight_box(insight, type="success")
                elif "Consider" in insight:
                    render_insight_box(insight, type="warning")
                else:
                    render_insight_box(insight, type="info")
    else:
        st.info("Generate more trades to unlock insights")

    # Download section
    st.divider()
    st.subheader("Export Data")

    col1, col2 = st.columns(2)

    with col1:
        # Export metrics
        metrics_df = pd.DataFrame([metrics])
        csv = metrics_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Metrics (CSV)",
            data=csv,
            file_name=f"performance_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    with col2:
        # Export equity curve
        if not perf_df.empty:
            csv = perf_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Equity Curve (CSV)",
                data=csv,
                file_name=f"equity_curve_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

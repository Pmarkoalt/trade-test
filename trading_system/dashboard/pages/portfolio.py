"""Portfolio page for the dashboard."""

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from trading_system.dashboard.config import ChartConfig, DashboardConfig
from trading_system.dashboard.components.cards import render_metric_row, render_insight_box
from trading_system.dashboard.services.data_service import DashboardDataService
from trading_system.tracking.models import SignalStatus


def render_portfolio(config: DashboardConfig):
    """Render the portfolio page."""
    st.title("💼 Portfolio")
    st.caption("Current positions and profit/loss tracking")

    # Initialize data service
    service = DashboardDataService(
        tracking_db_path=config.tracking_db_path,
        feature_db_path=config.feature_db_path,
    )

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Active Positions", "Position History", "Allocation"])

    with tab1:
        render_active_positions(service, config)

    with tab2:
        render_position_history(service, config)

    with tab3:
        render_allocation_view(service, config)


def render_active_positions(service: DashboardDataService, config: DashboardConfig):
    """Render active positions section."""
    st.subheader("Active Positions")

    try:
        data = service.get_dashboard_data()
        active_signals = data.active_signals

        if not active_signals:
            st.info("No active positions currently.")
            render_insight_box(
                "Start tracking by generating signals through the trading system.",
                type="info"
            )
            return

        # Portfolio summary metrics
        total_positions = len(active_signals)
        total_exposure = sum(s.position_size_pct for s in active_signals)

        # Calculate unrealized P&L (would need live prices in real implementation)
        buy_count = sum(1 for s in active_signals if s.direction.value == "BUY")
        sell_count = total_positions - buy_count

        render_metric_row([
            {"label": "Active Positions", "value": str(total_positions)},
            {"label": "Total Exposure", "value": f"{total_exposure:.1%}"},
            {"label": "Long Positions", "value": str(buy_count)},
            {"label": "Short Positions", "value": str(sell_count)},
        ])

        st.divider()

        # Positions table
        positions_data = []
        for signal in active_signals:
            # Calculate potential return to target and risk to stop
            if signal.entry_price > 0:
                risk_pct = abs(signal.stop_price - signal.entry_price) / signal.entry_price
                reward_pct = abs(signal.target_price - signal.entry_price) / signal.entry_price
                rr_ratio = reward_pct / risk_pct if risk_pct > 0 else 0
            else:
                risk_pct = 0
                reward_pct = 0
                rr_ratio = 0

            positions_data.append({
                "Symbol": signal.symbol,
                "Direction": signal.direction.value,
                "Entry": f"${signal.entry_price:.2f}",
                "Target": f"${signal.target_price:.2f}",
                "Stop": f"${signal.stop_price:.2f}",
                "Size": f"{signal.position_size_pct:.1%}",
                "R:R": f"{rr_ratio:.1f}",
                "Score": f"{signal.combined_score:.1f}",
                "Conviction": signal.conviction.value,
                "Days Held": (datetime.now() - signal.created_at).days if signal.created_at else 0,
            })

        df = pd.DataFrame(positions_data)

        # Style the dataframe
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                "Direction": st.column_config.TextColumn("Dir", width="small"),
                "Conviction": st.column_config.TextColumn("Conv", width="small"),
            }
        )

        # Insights
        if total_positions > 5:
            render_insight_box(
                f"You have {total_positions} active positions. Consider reviewing concentration risk.",
                type="warning"
            )

        if total_exposure > 0.8:
            render_insight_box(
                f"Portfolio exposure is {total_exposure:.0%}. Limited capacity for new positions.",
                type="warning"
            )

    except Exception as e:
        st.error(f"Error loading active positions: {e}")


def render_position_history(service: DashboardDataService, config: DashboardConfig):
    """Render position history section."""
    st.subheader("Position History")

    # Date range filter
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "From",
            value=date.today() - timedelta(days=config.default_lookback_days),
        )
    with col2:
        end_date = st.date_input("To", value=date.today())

    # Status filter
    status_filter = st.selectbox(
        "Status",
        options=["All", "Closed", "Expired", "Cancelled"],
        index=0,
    )

    try:
        df = service.get_signals_dataframe(days=config.default_lookback_days)

        if df.empty:
            st.info("No position history available.")
            return

        # Filter by status
        if status_filter != "All":
            df = df[df["status"] == status_filter.lower()]

        # Only show closed/expired/cancelled
        df = df[df["status"].isin(["closed", "expired", "cancelled"])]

        if df.empty:
            st.info("No closed positions in the selected period.")
            return

        # Summary metrics for closed positions
        closed_df = df[df["status"] == "closed"]
        if not closed_df.empty:
            # These would need actual outcome data in production
            total_closed = len(closed_df)

            st.markdown("**Closed Positions Summary**")
            render_metric_row([
                {"label": "Total Closed", "value": str(total_closed)},
                {"label": "Avg Hold Days", "value": "N/A"},
                {"label": "Win Rate", "value": "N/A"},
                {"label": "Total R", "value": "N/A"},
            ])

        st.divider()

        # Display table
        display_cols = ["symbol", "direction", "conviction", "status", "entry_price", "target_price", "stop_price", "created_at"]
        display_cols = [c for c in display_cols if c in df.columns]

        st.dataframe(
            df[display_cols],
            use_container_width=True,
            hide_index=True,
        )

    except Exception as e:
        st.error(f"Error loading position history: {e}")


def render_allocation_view(service: DashboardDataService, config: DashboardConfig):
    """Render allocation visualization."""
    st.subheader("Portfolio Allocation")

    try:
        data = service.get_dashboard_data()
        active_signals = data.active_signals

        if not active_signals:
            st.info("No active positions to visualize.")
            return

        chart_config = ChartConfig()

        # By Symbol (Pie chart)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Allocation by Symbol**")
            symbol_data = {}
            for s in active_signals:
                symbol_data[s.symbol] = symbol_data.get(s.symbol, 0) + s.position_size_pct

            fig = px.pie(
                names=list(symbol_data.keys()),
                values=list(symbol_data.values()),
                hole=0.4,
            )
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Allocation by Direction**")
            direction_data = {"BUY": 0, "SELL": 0}
            for s in active_signals:
                direction_data[s.direction.value] += s.position_size_pct

            colors = [chart_config.positive_color, chart_config.negative_color]
            fig = px.pie(
                names=list(direction_data.keys()),
                values=list(direction_data.values()),
                color_discrete_sequence=colors,
                hole=0.4,
            )
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        # By Asset Class
        st.markdown("**Allocation by Asset Class**")
        asset_data = {}
        for s in active_signals:
            asset_class = s.asset_class or "unknown"
            asset_data[asset_class] = asset_data.get(asset_class, 0) + s.position_size_pct

        if asset_data:
            fig = go.Figure(go.Bar(
                x=list(asset_data.keys()),
                y=[v * 100 for v in asset_data.values()],
                marker_color=chart_config.primary_color,
                text=[f"{v:.1%}" for v in asset_data.values()],
                textposition="outside",
            ))
            fig.update_layout(
                height=250,
                xaxis_title="Asset Class",
                yaxis_title="Allocation (%)",
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

        # By Conviction Level
        st.markdown("**Allocation by Conviction Level**")
        conv_data = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for s in active_signals:
            conv_data[s.conviction.value] += s.position_size_pct

        colors_map = {
            "HIGH": "#22c55e",
            "MEDIUM": "#f59e0b",
            "LOW": "#6b7280",
        }

        fig = go.Figure(go.Bar(
            x=list(conv_data.keys()),
            y=[v * 100 for v in conv_data.values()],
            marker_color=[colors_map[k] for k in conv_data.keys()],
            text=[f"{v:.1%}" for v in conv_data.values()],
            textposition="outside",
        ))
        fig.update_layout(
            height=250,
            xaxis_title="Conviction",
            yaxis_title="Allocation (%)",
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Cash allocation
        total_exposure = sum(s.position_size_pct for s in active_signals)
        cash_pct = max(0, 1.0 - total_exposure)

        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Exposure", f"{total_exposure:.1%}")
        with col2:
            st.metric("Cash Reserve", f"{cash_pct:.1%}")
        with col3:
            max_new_position = min(cash_pct, 0.10)  # Assume max 10% per position
            st.metric("Max New Position", f"{max_new_position:.1%}")

    except Exception as e:
        st.error(f"Error loading allocation view: {e}")

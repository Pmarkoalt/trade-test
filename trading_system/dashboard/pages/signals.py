"""Signals page for dashboard."""

import streamlit as st
import pandas as pd

from trading_system.dashboard.config import DashboardConfig
from trading_system.dashboard.services.data_service import DashboardDataService
from trading_system.dashboard.services.cache_service import get_cached_signals_df
from trading_system.dashboard.components.tables import render_signals_table
from trading_system.dashboard.components.cards import render_signal_card


def render_signals(config: DashboardConfig):
    """
    Render the signals page.

    Args:
        config: Dashboard configuration.
    """
    st.title("📊 Signals")

    # Initialize service
    service = DashboardDataService(
        tracking_db_path=config.tracking_db_path,
    )

    # Filters section
    with st.expander("🔍 Filters", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            status_filter = st.selectbox(
                "Status",
                options=["All", "pending", "active", "closed", "expired"],
                index=0,
            )

        with col2:
            days_filter = st.selectbox(
                "Time Period",
                options=[7, 14, 30, 60, 90],
                index=2,
                format_func=lambda x: f"Last {x} days",
            )

        with col3:
            direction_filter = st.selectbox(
                "Direction",
                options=["All", "BUY", "SELL"],
                index=0,
            )

        with col4:
            conviction_filter = st.selectbox(
                "Conviction",
                options=["All", "HIGH", "MEDIUM", "LOW"],
                index=0,
            )

    # Get filtered data
    status = status_filter.lower() if status_filter != "All" else None

    try:
        df = get_cached_signals_df(service, days=days_filter, status=status)
    except Exception as e:
        st.error(f"Error loading signals: {e}")
        return

    if df.empty:
        st.info("No signals found for the selected filters")
        return

    # Apply additional filters
    if direction_filter != "All":
        df = df[df["direction"] == direction_filter]

    if conviction_filter != "All":
        df = df[df["conviction"] == conviction_filter]

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Signals", len(df))

    with col2:
        if "conviction" in df.columns:
            high_conv = len(df[df["conviction"] == "HIGH"])
            st.metric("High Conviction", high_conv)

    with col3:
        buy_count = len(df[df["direction"] == "BUY"])
        st.metric("Buy Signals", buy_count)

    with col4:
        if "combined_score" in df.columns:
            avg_score = df["combined_score"].astype(float).mean()
            st.metric("Avg Score", f"{avg_score:.1f}")

    st.divider()

    # View toggle
    view_mode = st.radio(
        "View Mode",
        options=["Table", "Cards"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if view_mode == "Table":
        # Sort options
        col1, col2 = st.columns([3, 1])
        with col2:
            sort_by = st.selectbox(
                "Sort by",
                options=["created_at", "combined_score", "symbol"],
                format_func=lambda x: x.replace("_", " ").title(),
            )

        # Sort DataFrame
        if sort_by in df.columns:
            ascending = sort_by == "symbol"
            df_sorted = df.sort_values(sort_by, ascending=ascending)
        else:
            df_sorted = df

        render_signals_table(df_sorted, max_rows=config.max_signals_display)

    else:  # Cards view
        # Sort by created_at for cards
        df_sorted = df.sort_values("created_at", ascending=False)

        # Paginate cards
        items_per_page = 10
        total_pages = (len(df_sorted) - 1) // items_per_page + 1

        if total_pages > 1:
            page = st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=1,
            )
        else:
            page = 1

        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page

        for _, row in df_sorted.iloc[start_idx:end_idx].iterrows():
            render_signal_card(row.to_dict())

        if total_pages > 1:
            st.caption(f"Page {page} of {total_pages}")

    st.divider()

    # Signal details expander
    st.subheader("Signal Details")

    # Select signal for details
    signal_ids = df["id"].tolist() if "id" in df.columns else []

    if signal_ids:
        selected_id = st.selectbox(
            "Select Signal",
            options=signal_ids,
            format_func=lambda x: f"{df[df['id'] == x]['symbol'].values[0]} - {x[:12]}...",
        )

        if selected_id:
            details = service.get_signal_details(selected_id)

            if details:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Signal Info**")
                    signal_data = details.get("signal", {})

                    for key in ["symbol", "direction", "conviction", "status", "signal_type"]:
                        if key in signal_data:
                            st.text(f"{key.title()}: {signal_data[key]}")

                with col2:
                    st.markdown("**Prices**")
                    for key in ["entry_price", "target_price", "stop_price"]:
                        if key in signal_data:
                            st.text(f"{key.replace('_', ' ').title()}: ${signal_data[key]:.2f}")

                # Outcome if available
                outcome = details.get("outcome")
                if outcome:
                    st.markdown("**Outcome**")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        r_mult = outcome.get("r_multiple", 0)
                        color = "green" if r_mult > 0 else "red"
                        st.markdown(f"R-Multiple: :{color}[{r_mult:+.2f}R]")

                    with col2:
                        ret_pct = outcome.get("return_pct", 0) * 100
                        st.text(f"Return: {ret_pct:+.1f}%")

                    with col3:
                        reason = outcome.get("exit_reason", "Unknown")
                        st.text(f"Exit: {reason}")

                # Reasoning
                if signal_data.get("reasoning"):
                    st.markdown("**Reasoning**")
                    st.text(signal_data["reasoning"])

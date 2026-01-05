"""Table components for dashboard."""

from typing import List

import pandas as pd
import streamlit as st


def render_signals_table(
    df: pd.DataFrame,
    show_actions: bool = False,
    max_rows: int = 50,
):
    """
    Render signals table.

    Args:
        df: DataFrame with signal data.
        show_actions: Whether to show action buttons.
        max_rows: Maximum rows to display.
    """
    if df.empty:
        st.info("No signals to display")
        return

    # Limit rows
    df_display = df.head(max_rows).copy()

    # Format columns
    if "created_at" in df_display.columns:
        df_display["created_at"] = pd.to_datetime(df_display["created_at"]).dt.strftime("%Y-%m-%d %H:%M")

    if "combined_score" in df_display.columns:
        df_display["combined_score"] = df_display["combined_score"].apply(lambda x: f"{x:.1f}")

    # Define column config
    column_config = {
        "symbol": st.column_config.TextColumn("Symbol", width="small"),
        "direction": st.column_config.TextColumn("Dir", width="small"),
        "conviction": st.column_config.TextColumn("Conv", width="small"),
        "status": st.column_config.TextColumn("Status", width="small"),
        "entry_price": st.column_config.NumberColumn("Entry", format="$%.2f"),
        "target_price": st.column_config.NumberColumn("Target", format="$%.2f"),
        "stop_price": st.column_config.NumberColumn("Stop", format="$%.2f"),
        "combined_score": st.column_config.TextColumn("Score", width="small"),
        "created_at": st.column_config.TextColumn("Created", width="medium"),
    }

    # Select columns to display
    display_cols = [
        col
        for col in [
            "symbol",
            "direction",
            "conviction",
            "status",
            "entry_price",
            "target_price",
            "stop_price",
            "combined_score",
            "created_at",
        ]
        if col in df_display.columns
    ]

    st.dataframe(
        df_display[display_cols],
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
    )

    if len(df) > max_rows:
        st.caption(f"Showing {max_rows} of {len(df)} signals")


def render_performance_table(metrics: dict):
    """
    Render performance metrics table.

    Args:
        metrics: Performance metrics dictionary.
    """
    # Create two columns for metrics
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Returns**")
        data = {
            "Metric": ["Total Return", "Avg Return", "Avg Winner", "Avg Loser"],
            "Value": [
                f"{metrics.get('total_r', 0):.2f}R",
                f"{metrics.get('avg_r', 0):.2f}R",
                f"{metrics.get('avg_winner_r', 0):.2f}R",
                f"{metrics.get('avg_loser_r', 0):.2f}R",
            ],
        }
        st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)

    with col2:
        st.markdown("**Risk Metrics**")
        data = {
            "Metric": ["Win Rate", "Expectancy", "Sharpe", "Max DD"],
            "Value": [
                f"{metrics.get('win_rate', 0) * 100:.1f}%",
                f"{metrics.get('expectancy_r', 0):.2f}R",
                f"{metrics.get('sharpe_ratio', 0):.2f}",
                f"{metrics.get('max_drawdown_pct', 0) * 100:.1f}%",
            ],
        }
        st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)


def render_leaderboard_table(leaderboard: dict):
    """
    Render strategy leaderboard table.

    Args:
        leaderboard: Leaderboard dictionary with entries.
    """
    entries = leaderboard.get("entries", [])
    if not entries:
        st.info("No leaderboard data")
        return

    data = []
    for entry in entries:
        # Trend indicator
        if entry.get("trend") == "up":
            trend = "📈"
        elif entry.get("trend") == "down":
            trend = "📉"
        else:
            trend = "➡️"

        data.append(
            {
                "Rank": f"#{entry.get('rank', 0)}",
                "Strategy": entry.get("display_name", entry.get("strategy_name", "")),
                "Total R": f"{entry.get('total_r', 0):+.1f}R",
                "Win Rate": f"{entry.get('win_rate', 0) * 100:.0f}%",
                "Trades": entry.get("trade_count", 0),
                "Trend": trend,
            }
        )

    df = pd.DataFrame(data)
    st.dataframe(df, hide_index=True, use_container_width=True)


def render_recent_trades_table(trades: List[dict], max_rows: int = 10):
    """
    Render recent trades table.

    Args:
        trades: List of trade dictionaries.
        max_rows: Maximum rows to display.
    """
    if not trades:
        st.info("No recent trades")
        return

    data = []
    for trade in trades[:max_rows]:
        r_mult = trade.get("r_multiple", 0)
        "green" if r_mult > 0 else "red"

        data.append(
            {
                "Symbol": trade.get("symbol", ""),
                "Direction": trade.get("direction", ""),
                "Result": f"{r_mult:+.2f}R",
                "Exit": trade.get("exit_reason", "Manual"),
                "Date": trade.get("exit_date", ""),
            }
        )

    df = pd.DataFrame(data)
    st.dataframe(df, hide_index=True, use_container_width=True)

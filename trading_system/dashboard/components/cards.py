"""Card components for dashboard."""

from typing import Optional

import streamlit as st


def render_metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_color: str = "normal",
    help_text: Optional[str] = None,
):
    """
    Render a metric card using Streamlit's metric.

    Args:
        label: Metric label.
        value: Metric value.
        delta: Optional delta value.
        delta_color: Delta color ("normal", "inverse", "off").
        help_text: Optional help tooltip.
    """
    st.metric(
        label=label,
        value=value,
        delta=delta,
        delta_color=delta_color,
        help=help_text,
    )


def render_metric_row(metrics: list):
    """
    Render a row of metric cards.

    Args:
        metrics: List of metric dicts with keys: label, value, delta (optional).
    """
    cols = st.columns(len(metrics))

    for col, metric in zip(cols, metrics):
        with col:
            render_metric_card(
                label=metric.get("label", ""),
                value=metric.get("value", ""),
                delta=metric.get("delta"),
                delta_color=metric.get("delta_color", "normal"),
                help_text=metric.get("help"),
            )


def render_signal_card(signal: dict):
    """
    Render a signal card with details.

    Args:
        signal: Signal dictionary.
    """
    direction = signal.get("direction", "BUY")
    conviction = signal.get("conviction", "MEDIUM")
    symbol = signal.get("symbol", "???")

    # Card styling based on direction
    border_color = "#22c55e" if direction == "BUY" else "#ef4444"

    # Conviction badge color
    conv_colors = {
        "HIGH": ("#dcfce7", "#166534"),
        "MEDIUM": ("#fef3c7", "#92400e"),
        "LOW": ("#fee2e2", "#991b1b"),
    }
    bg_color, text_color = conv_colors.get(conviction, ("#f3f4f6", "#374151"))

    st.markdown(
        f"""
        <div style="
            border: 1px solid #e5e7eb;
            border-left: 4px solid {border_color};
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            background: white;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.25rem; font-weight: bold;">{symbol}</span>
                <span style="
                    background: {bg_color};
                    color: {text_color};
                    padding: 0.25rem 0.75rem;
                    border-radius: 9999px;
                    font-size: 0.75rem;
                    font-weight: 600;
                ">{conviction}</span>
            </div>
            <div style="color: #666; font-size: 0.875rem;">
                <strong>{direction}</strong> · Score: {signal.get('combined_score', 0):.1f} · {signal.get('signal_type', '')}
            </div>
            <div style="margin-top: 0.5rem; font-size: 0.875rem;">
                Entry: ${signal.get('entry_price', 0):.2f} ·
                Target: ${signal.get('target_price', 0):.2f} ·
                Stop: ${signal.get('stop_price', 0):.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight_box(
    text: str,
    type: str = "info",
):
    """
    Render an insight box.

    Args:
        text: Insight text.
        type: Box type ("success", "warning", "danger", "info").
    """
    colors = {
        "success": ("#f0fdf4", "#22c55e"),
        "warning": ("#fffbeb", "#f59e0b"),
        "danger": ("#fef2f2", "#ef4444"),
        "info": ("#eff6ff", "#3b82f6"),
    }
    bg_color, border_color = colors.get(type, colors["info"])

    st.markdown(
        f"""
        <div style="
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 0.75rem;
            background: {bg_color};
            border-left: 4px solid {border_color};
        ">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_badge(status: str) -> str:
    """
    Render a status badge HTML.

    Args:
        status: Status string.

    Returns:
        HTML string for badge.
    """
    status_colors = {
        "pending": ("#fef3c7", "#92400e"),
        "active": ("#dbeafe", "#1d4ed8"),
        "closed": ("#f3f4f6", "#6b7280"),
        "expired": ("#fee2e2", "#991b1b"),
    }

    bg_color, text_color = status_colors.get(status.lower(), ("#f3f4f6", "#374151"))

    return f"""
        <span style="
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 500;
            background: {bg_color};
            color: {text_color};
        ">{status.upper()}</span>
    """


def render_streak_indicator(
    streak_count: int,
    streak_type: str,
):
    """
    Render streak indicator.

    Args:
        streak_count: Number of consecutive wins/losses.
        streak_type: "win" or "loss".
    """
    if streak_count < 3:
        return

    if streak_type == "win":
        emoji = "🔥"
        color = "#22c55e"
        text = f"{streak_count}-trade winning streak!"
    else:
        emoji = "⚠️"
        color = "#ef4444"
        text = f"{streak_count}-trade losing streak"

    st.markdown(
        f"""
        <div style="
            padding: 0.75rem 1rem;
            border-radius: 8px;
            background: {'#f0fdf4' if streak_type == 'win' else '#fef2f2'};
            text-align: center;
            margin-bottom: 1rem;
        ">
            <span style="font-size: 1.5rem; margin-right: 0.5rem;">{emoji}</span>
            <span style="font-weight: 600; color: {color};">{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

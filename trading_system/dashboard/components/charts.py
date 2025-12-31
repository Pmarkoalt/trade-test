"""Chart components for dashboard."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trading_system.dashboard.config import ChartConfig


def create_equity_curve(
    df: pd.DataFrame,
    config: Optional[ChartConfig] = None,
) -> go.Figure:
    """
    Create equity curve chart.

    Args:
        df: DataFrame with 'date', 'equity', 'drawdown_pct' columns.
        config: Chart configuration.

    Returns:
        Plotly figure.
    """
    config = config or ChartConfig()

    # Create subplots: equity and drawdown
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Equity Curve", "Drawdown"),
    )

    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["equity"],
            mode="lines",
            name="Equity",
            line=dict(color=config.primary_color, width=2),
            fill="tozeroy",
            fillcolor=f"rgba(59, 130, 246, 0.1)",
        ),
        row=1,
        col=1,
    )

    # High water mark
    if "high_water_mark" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["high_water_mark"],
                mode="lines",
                name="High Water Mark",
                line=dict(color=config.neutral_color, width=1, dash="dash"),
            ),
            row=1,
            col=1,
        )

    # Drawdown
    if "drawdown_pct" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["drawdown_pct"] * 100,
                mode="lines",
                name="Drawdown",
                line=dict(color=config.negative_color, width=1),
                fill="tozeroy",
                fillcolor=f"rgba(239, 68, 68, 0.2)",
            ),
            row=2,
            col=1,
        )

    # Update layout
    fig.update_layout(
        height=config.height,
        template=config.template,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=20, t=40, b=20),
    )

    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    return fig


def create_win_rate_gauge(
    win_rate: float,
    config: Optional[ChartConfig] = None,
) -> go.Figure:
    """
    Create win rate gauge chart.

    Args:
        win_rate: Win rate (0-1).
        config: Chart configuration.

    Returns:
        Plotly figure.
    """
    config = config or ChartConfig()

    # Determine color based on value
    if win_rate >= 0.55:
        color = config.positive_color
    elif win_rate >= 0.45:
        color = "#f59e0b"  # Warning yellow
    else:
        color = config.negative_color

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=win_rate * 100,
            number={"suffix": "%", "font": {"size": 40}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 45], "color": "rgba(239, 68, 68, 0.2)"},
                    {"range": [45, 55], "color": "rgba(245, 158, 11, 0.2)"},
                    {"range": [55, 100], "color": "rgba(34, 197, 94, 0.2)"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 2},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
            title={"text": "Win Rate"},
        )
    )

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig


def create_returns_distribution(
    returns: List[float],
    config: Optional[ChartConfig] = None,
) -> go.Figure:
    """
    Create returns distribution histogram.

    Args:
        returns: List of return values (R-multiples).
        config: Chart configuration.

    Returns:
        Plotly figure.
    """
    config = config or ChartConfig()

    fig = go.Figure()

    # Separate winners and losers
    winners = [r for r in returns if r > 0]
    losers = [r for r in returns if r <= 0]

    # Add histograms
    fig.add_trace(
        go.Histogram(
            x=winners,
            name="Winners",
            marker_color=config.positive_color,
            opacity=0.7,
            nbinsx=20,
        )
    )

    fig.add_trace(
        go.Histogram(
            x=losers,
            name="Losers",
            marker_color=config.negative_color,
            opacity=0.7,
            nbinsx=20,
        )
    )

    # Add mean line
    mean_r = np.mean(returns) if returns else 0

    fig.add_vline(
        x=mean_r,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Mean: {mean_r:.2f}R",
    )

    fig.update_layout(
        height=config.height,
        template=config.template,
        title="Return Distribution (R-Multiples)",
        xaxis_title="R-Multiple",
        yaxis_title="Count",
        barmode="overlay",
        showlegend=True,
    )

    return fig


def create_strategy_comparison_chart(
    df: pd.DataFrame,
    metric: str = "total_r",
    config: Optional[ChartConfig] = None,
) -> go.Figure:
    """
    Create strategy comparison bar chart.

    Args:
        df: DataFrame with strategy metrics.
        metric: Metric to compare ("total_r", "win_rate", "expectancy").
        config: Chart configuration.

    Returns:
        Plotly figure.
    """
    config = config or ChartConfig()

    # Sort by metric
    df_sorted = df.sort_values(metric, ascending=True)

    # Color based on positive/negative
    colors = [config.positive_color if v > 0 else config.negative_color for v in df_sorted[metric]]

    fig = go.Figure(
        go.Bar(
            x=df_sorted[metric],
            y=df_sorted["strategy"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.2f}" for v in df_sorted[metric]],
            textposition="outside",
        )
    )

    # Title based on metric
    titles = {
        "total_r": "Total Return (R)",
        "win_rate": "Win Rate (%)",
        "expectancy": "Expectancy (R)",
    }

    fig.update_layout(
        height=max(300, len(df) * 40),
        template=config.template,
        title=f"Strategy Comparison: {titles.get(metric, metric)}",
        xaxis_title=titles.get(metric, metric),
        yaxis_title="Strategy",
        showlegend=False,
    )

    return fig


def create_performance_by_day(
    analytics: dict,
    config: Optional[ChartConfig] = None,
) -> go.Figure:
    """
    Create performance by day of week chart.

    Args:
        analytics: Analytics dict with performance_by_day_of_week.
        config: Chart configuration.

    Returns:
        Plotly figure.
    """
    config = config or ChartConfig()

    by_day = analytics.get("performance_by_day_of_week", {})
    if not by_day:
        return go.Figure()

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    win_rates = [by_day.get(d, {}).get("win_rate", 0) * 100 for d in days]
    avg_r = [by_day.get(d, {}).get("avg_r", 0) for d in days]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Win Rate by Day", "Avg R by Day"),
    )

    # Win rate bars
    fig.add_trace(
        go.Bar(
            x=days,
            y=win_rates,
            marker_color=config.primary_color,
            name="Win Rate",
        ),
        row=1,
        col=1,
    )

    # Avg R bars
    colors = [config.positive_color if r > 0 else config.negative_color for r in avg_r]
    fig.add_trace(
        go.Bar(
            x=days,
            y=avg_r,
            marker_color=colors,
            name="Avg R",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=300,
        template=config.template,
        showlegend=False,
    )

    fig.update_yaxes(title_text="Win Rate (%)", row=1, col=1)
    fig.update_yaxes(title_text="Avg R", row=1, col=2)

    return fig


def create_conviction_breakdown(
    analytics: dict,
    config: Optional[ChartConfig] = None,
) -> go.Figure:
    """
    Create conviction level breakdown chart.

    Args:
        analytics: Analytics dict with performance_by_conviction.
        config: Chart configuration.

    Returns:
        Plotly figure.
    """
    config = config or ChartConfig()

    by_conv = analytics.get("performance_by_conviction", {})
    if not by_conv:
        return go.Figure()

    levels = ["HIGH", "MEDIUM", "LOW"]
    win_rates = [by_conv.get(l, {}).get("win_rate", 0) * 100 for l in levels]
    avg_r = [by_conv.get(l, {}).get("avg_r", 0) for l in levels]
    totals = [by_conv.get(l, {}).get("total", 0) for l in levels]

    fig = go.Figure()

    # Grouped bar chart
    fig.add_trace(
        go.Bar(
            name="Win Rate (%)",
            x=levels,
            y=win_rates,
            marker_color=config.primary_color,
            text=[f"{v:.0f}%" for v in win_rates],
            textposition="outside",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Avg R",
            x=levels,
            y=avg_r,
            marker_color=config.positive_color,
            text=[f"{v:.2f}R" for v in avg_r],
            textposition="outside",
        )
    )

    fig.update_layout(
        height=300,
        template=config.template,
        title="Performance by Conviction Level",
        barmode="group",
        showlegend=True,
    )

    return fig


def create_monthly_performance_heatmap(
    performance_data: List[dict],
    config: Optional[ChartConfig] = None,
) -> go.Figure:
    """
    Create monthly performance heatmap.

    Args:
        performance_data: List of monthly performance dicts.
        config: Chart configuration.

    Returns:
        Plotly figure.
    """
    config = config or ChartConfig()

    if not performance_data:
        return go.Figure()

    # Organize by year and month
    # Create matrix (years x months)
    years = sorted(set(d.get("year", 2024) for d in performance_data))
    months = list(range(1, 13))

    z = np.zeros((len(years), 12))
    z[:] = np.nan

    for d in performance_data:
        year = d.get("year", 2024)
        month = d.get("month", 1)
        r_value = d.get("total_r", 0)

        year_idx = years.index(year)
        month_idx = month - 1
        z[year_idx, month_idx] = r_value

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=month_names,
            y=[str(y) for y in years],
            colorscale="RdYlGn",
            zmid=0,
            text=[[f"{v:.1f}R" if not np.isnan(v) else "" for v in row] for row in z],
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}R<extra></extra>",
        )
    )

    fig.update_layout(
        height=max(200, len(years) * 60),
        template=config.template,
        title="Monthly Performance (R-Multiples)",
    )

    return fig

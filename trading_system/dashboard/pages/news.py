"""News page for the dashboard."""

import asyncio
import os
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from trading_system.dashboard.config import ChartConfig, DashboardConfig
from trading_system.dashboard.components.cards import render_insight_box


def render_news(config: DashboardConfig):
    """Render the news page."""
    st.title("📰 News & Sentiment")
    st.caption("Market news feed and sentiment analysis")

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["News Feed", "Sentiment Analysis", "Watchlist News"])

    with tab1:
        render_news_feed(config)

    with tab2:
        render_sentiment_analysis(config)

    with tab3:
        render_watchlist_news(config)


def render_news_feed(config: DashboardConfig):
    """Render the main news feed."""
    st.subheader("Latest Market News")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        category = st.selectbox(
            "Category",
            options=["All", "Technology", "Finance", "Crypto", "Economy"],
            index=0,
        )

    with col2:
        sentiment_filter = st.selectbox(
            "Sentiment",
            options=["All", "Positive", "Neutral", "Negative"],
            index=0,
        )

    with col3:
        timeframe = st.selectbox(
            "Timeframe",
            options=["Last 24 Hours", "Last 48 Hours", "Last Week"],
            index=0,
        )

    st.divider()

    # Try to fetch news if API keys are configured
    news_articles = _get_sample_news()  # Use sample data for demo

    if not news_articles:
        st.info("No news articles available. Configure API keys to enable live news.")
        _render_api_setup_guide()
        return

    # Filter articles
    filtered_articles = news_articles
    if sentiment_filter != "All":
        filtered_articles = [
            a for a in filtered_articles
            if a.get("sentiment_label", "").lower() == sentiment_filter.lower()
        ]

    # Display articles
    for article in filtered_articles:
        _render_news_card(article)

    if len(filtered_articles) == 0:
        st.info("No articles match the selected filters.")


def render_sentiment_analysis(config: DashboardConfig):
    """Render sentiment analysis dashboard."""
    st.subheader("Sentiment Analysis")

    # Sample sentiment data
    sentiment_data = _get_sample_sentiment_data()

    if not sentiment_data:
        st.info("No sentiment data available.")
        return

    chart_config = ChartConfig()

    # Overall sentiment gauge
    col1, col2, col3 = st.columns(3)

    with col1:
        overall_sentiment = sentiment_data.get("overall", 0.15)
        _render_sentiment_gauge(overall_sentiment, "Overall Market Sentiment")

    with col2:
        equity_sentiment = sentiment_data.get("equity", 0.22)
        _render_sentiment_gauge(equity_sentiment, "Equity Sentiment")

    with col3:
        crypto_sentiment = sentiment_data.get("crypto", -0.05)
        _render_sentiment_gauge(crypto_sentiment, "Crypto Sentiment")

    st.divider()

    # Sentiment by symbol
    st.markdown("**Sentiment by Symbol**")

    symbol_sentiments = sentiment_data.get("by_symbol", {
        "AAPL": 0.45,
        "MSFT": 0.32,
        "GOOGL": 0.18,
        "NVDA": 0.55,
        "AMZN": -0.12,
        "BTC": -0.08,
        "ETH": 0.15,
    })

    # Create horizontal bar chart
    symbols = list(symbol_sentiments.keys())
    scores = list(symbol_sentiments.values())
    colors = [chart_config.positive_color if s > 0 else chart_config.negative_color for s in scores]

    fig = go.Figure(go.Bar(
        x=scores,
        y=symbols,
        orientation='h',
        marker_color=colors,
        text=[f"{s:+.2f}" for s in scores],
        textposition='outside',
    ))

    fig.update_layout(
        height=max(250, len(symbols) * 35),
        xaxis_title="Sentiment Score",
        xaxis_range=[-1, 1],
        margin=dict(l=60, r=40, t=20, b=40),
    )

    # Add zero line
    fig.add_vline(x=0, line_dash="dash", line_color="gray")

    st.plotly_chart(fig, use_container_width=True)

    # Sentiment trend over time
    st.markdown("**Sentiment Trend (Last 7 Days)**")

    trend_data = sentiment_data.get("trend", _generate_sample_trend())
    df_trend = pd.DataFrame(trend_data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_trend["date"],
        y=df_trend["sentiment"],
        mode="lines+markers",
        name="Sentiment",
        line=dict(color=chart_config.primary_color, width=2),
        fill="tozeroy",
        fillcolor="rgba(59, 130, 246, 0.1)",
    ))

    fig.update_layout(
        height=300,
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        yaxis_range=[-1, 1],
        margin=dict(l=40, r=20, t=20, b=40),
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    st.plotly_chart(fig, use_container_width=True)

    # Insights
    if overall_sentiment > 0.3:
        render_insight_box(
            "Market sentiment is strongly positive. Consider taking profits on long positions.",
            type="success"
        )
    elif overall_sentiment < -0.3:
        render_insight_box(
            "Market sentiment is negative. Look for oversold opportunities.",
            type="warning"
        )


def render_watchlist_news(config: DashboardConfig):
    """Render news for watchlist symbols."""
    st.subheader("Watchlist News")

    # Watchlist input
    default_watchlist = "AAPL, MSFT, GOOGL, NVDA, BTC, ETH"
    watchlist_input = st.text_input(
        "Symbols (comma-separated)",
        value=default_watchlist,
        help="Enter symbols to track news for",
    )

    symbols = [s.strip().upper() for s in watchlist_input.split(",") if s.strip()]

    if not symbols:
        st.info("Enter symbols to see relevant news.")
        return

    st.divider()

    # Lookback selector
    lookback = st.selectbox(
        "Lookback",
        options=["24 hours", "48 hours", "1 week"],
        index=1,
    )

    # Get news for each symbol
    for symbol in symbols:
        with st.expander(f"📰 {symbol}", expanded=True):
            news = _get_sample_news_for_symbol(symbol)

            if not news:
                st.caption("No recent news for this symbol.")
                continue

            for article in news[:3]:  # Show top 3
                _render_compact_news_card(article)

            # Show sentiment summary
            avg_sentiment = sum(a.get("sentiment_score", 0) for a in news) / len(news)
            sentiment_label = "Positive" if avg_sentiment > 0.05 else "Negative" if avg_sentiment < -0.05 else "Neutral"
            sentiment_color = "#22c55e" if avg_sentiment > 0.05 else "#ef4444" if avg_sentiment < -0.05 else "#6b7280"

            st.markdown(
                f"<span style='color: {sentiment_color}; font-weight: 600;'>"
                f"Overall: {sentiment_label} ({avg_sentiment:+.2f})</span>",
                unsafe_allow_html=True
            )


def _render_news_card(article: dict):
    """Render a news article card."""
    sentiment = article.get("sentiment_score", 0)
    sentiment_label = article.get("sentiment_label", "neutral")

    # Sentiment colors
    if sentiment > 0.1:
        border_color = "#22c55e"
        badge_bg = "#dcfce7"
        badge_text = "#166534"
    elif sentiment < -0.1:
        border_color = "#ef4444"
        badge_bg = "#fee2e2"
        badge_text = "#991b1b"
    else:
        border_color = "#6b7280"
        badge_bg = "#f3f4f6"
        badge_text = "#374151"

    # Format time
    published = article.get("published_at", datetime.now())
    if isinstance(published, str):
        try:
            published = datetime.fromisoformat(published.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            published = datetime.now()

    time_ago = _format_time_ago(published)

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
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
                <div style="flex: 1;">
                    <span style="font-size: 1rem; font-weight: 600; color: #1f2937;">
                        {article.get('title', 'Untitled')}
                    </span>
                </div>
                <span style="
                    background: {badge_bg};
                    color: {badge_text};
                    padding: 0.25rem 0.5rem;
                    border-radius: 4px;
                    font-size: 0.75rem;
                    font-weight: 500;
                    margin-left: 0.5rem;
                    white-space: nowrap;
                ">{sentiment_label.upper()}</span>
            </div>
            <div style="color: #6b7280; font-size: 0.875rem; margin-bottom: 0.5rem;">
                {article.get('summary', '')[:200]}{'...' if len(article.get('summary', '')) > 200 else ''}
            </div>
            <div style="display: flex; justify-content: space-between; color: #9ca3af; font-size: 0.75rem;">
                <span>{article.get('source', 'Unknown')}</span>
                <span>{time_ago}</span>
            </div>
            <div style="margin-top: 0.5rem;">
                {''.join([f'<span style="background: #e5e7eb; padding: 0.125rem 0.375rem; border-radius: 4px; font-size: 0.7rem; margin-right: 0.25rem;">{s}</span>' for s in article.get('symbols', [])[:5]])}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def _render_compact_news_card(article: dict):
    """Render a compact news card."""
    sentiment = article.get("sentiment_score", 0)
    emoji = "🟢" if sentiment > 0.1 else "🔴" if sentiment < -0.1 else "⚪"

    published = article.get("published_at", datetime.now())
    time_ago = _format_time_ago(published) if isinstance(published, datetime) else "Recently"

    st.markdown(
        f"""
        <div style="padding: 0.5rem 0; border-bottom: 1px solid #f3f4f6;">
            <div style="display: flex; align-items: flex-start;">
                <span style="margin-right: 0.5rem;">{emoji}</span>
                <div>
                    <div style="font-size: 0.875rem; color: #374151;">{article.get('title', 'Untitled')}</div>
                    <div style="font-size: 0.75rem; color: #9ca3af;">{article.get('source', '')} · {time_ago}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def _render_sentiment_gauge(value: float, title: str):
    """Render a sentiment gauge."""
    # Normalize to 0-100 for gauge
    gauge_value = (value + 1) * 50  # -1 to 1 -> 0 to 100

    if value > 0.2:
        color = "#22c55e"
        label = "Positive"
    elif value < -0.2:
        color = "#ef4444"
        label = "Negative"
    else:
        color = "#f59e0b"
        label = "Neutral"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "", "font": {"size": 28}, "valueformat": "+.2f"},
        gauge={
            "axis": {"range": [-1, 1], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [-1, -0.2], "color": "rgba(239, 68, 68, 0.2)"},
                {"range": [-0.2, 0.2], "color": "rgba(245, 158, 11, 0.2)"},
                {"range": [0.2, 1], "color": "rgba(34, 197, 94, 0.2)"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 2},
                "thickness": 0.75,
                "value": 0,
            },
        },
        title={"text": title, "font": {"size": 14}},
    ))

    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=40, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_api_setup_guide():
    """Render API setup instructions."""
    with st.expander("📋 How to enable live news"):
        st.markdown("""
        To enable live news fetching, set up the following API keys:

        **NewsAPI.org** (Free tier: 100 requests/day)
        1. Sign up at [newsapi.org](https://newsapi.org)
        2. Get your API key
        3. Set environment variable: `NEWSAPI_API_KEY`

        **Alpha Vantage** (Free tier: 5 calls/min)
        1. Sign up at [alphavantage.co](https://www.alphavantage.co)
        2. Get your API key
        3. Set environment variable: `ALPHA_VANTAGE_API_KEY`

        After setting the keys, restart the dashboard.
        """)


def _format_time_ago(dt: datetime) -> str:
    """Format datetime as time ago string."""
    now = datetime.now()
    if dt.tzinfo:
        now = datetime.now(dt.tzinfo)

    delta = now - dt

    if delta.days > 7:
        return dt.strftime("%b %d")
    elif delta.days > 1:
        return f"{delta.days} days ago"
    elif delta.days == 1:
        return "Yesterday"
    elif delta.seconds > 3600:
        hours = delta.seconds // 3600
        return f"{hours}h ago"
    elif delta.seconds > 60:
        minutes = delta.seconds // 60
        return f"{minutes}m ago"
    else:
        return "Just now"


def _get_sample_news() -> List[dict]:
    """Get sample news data for demo purposes."""
    return [
        {
            "title": "Tech Stocks Rally on Strong Earnings Reports",
            "summary": "Major technology companies reported better-than-expected earnings, driving a broad rally in the tech sector. Apple and Microsoft led gains with impressive revenue growth.",
            "source": "Reuters",
            "published_at": datetime.now() - timedelta(hours=2),
            "sentiment_score": 0.65,
            "sentiment_label": "positive",
            "symbols": ["AAPL", "MSFT", "GOOGL"],
        },
        {
            "title": "Federal Reserve Signals Cautious Approach to Rate Cuts",
            "summary": "Fed officials indicated they will take a data-dependent approach to any future rate cuts, suggesting patience in the face of persistent inflation concerns.",
            "source": "Bloomberg",
            "published_at": datetime.now() - timedelta(hours=5),
            "sentiment_score": -0.15,
            "sentiment_label": "neutral",
            "symbols": [],
        },
        {
            "title": "Bitcoin Drops Below Key Support Level",
            "summary": "Bitcoin fell below the $42,000 support level amid profit-taking and concerns over regulatory developments in major markets.",
            "source": "CoinDesk",
            "published_at": datetime.now() - timedelta(hours=8),
            "sentiment_score": -0.45,
            "sentiment_label": "negative",
            "symbols": ["BTC"],
        },
        {
            "title": "NVIDIA Announces New AI Chip Architecture",
            "summary": "NVIDIA unveiled its next-generation AI accelerator with significant performance improvements, reinforcing its leadership in the AI hardware market.",
            "source": "TechCrunch",
            "published_at": datetime.now() - timedelta(hours=12),
            "sentiment_score": 0.72,
            "sentiment_label": "positive",
            "symbols": ["NVDA"],
        },
        {
            "title": "Oil Prices Stabilize After Volatile Week",
            "summary": "Crude oil prices found stability after a week of volatility driven by geopolitical tensions and supply concerns.",
            "source": "WSJ",
            "published_at": datetime.now() - timedelta(hours=18),
            "sentiment_score": 0.05,
            "sentiment_label": "neutral",
            "symbols": ["XOM", "CVX"],
        },
    ]


def _get_sample_news_for_symbol(symbol: str) -> List[dict]:
    """Get sample news for a specific symbol."""
    all_news = _get_sample_news()
    return [n for n in all_news if symbol in n.get("symbols", [])] or [
        {
            "title": f"Latest developments for {symbol}",
            "source": "Market News",
            "published_at": datetime.now() - timedelta(hours=6),
            "sentiment_score": 0.12,
        }
    ]


def _get_sample_sentiment_data() -> dict:
    """Get sample sentiment data."""
    return {
        "overall": 0.18,
        "equity": 0.25,
        "crypto": -0.08,
        "by_symbol": {
            "AAPL": 0.42,
            "MSFT": 0.35,
            "GOOGL": 0.22,
            "NVDA": 0.68,
            "AMZN": 0.15,
            "META": 0.28,
            "BTC": -0.12,
            "ETH": 0.08,
        },
        "trend": _generate_sample_trend(),
    }


def _generate_sample_trend() -> List[dict]:
    """Generate sample sentiment trend data."""
    import random
    trend = []
    base = 0.1
    for i in range(7):
        date = datetime.now() - timedelta(days=6-i)
        base += random.uniform(-0.15, 0.15)
        base = max(-0.8, min(0.8, base))
        trend.append({
            "date": date.strftime("%Y-%m-%d"),
            "sentiment": round(base, 2),
        })
    return trend

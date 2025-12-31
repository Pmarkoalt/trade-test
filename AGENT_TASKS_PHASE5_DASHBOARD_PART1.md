# Agent Tasks: Phase 5 - Dashboard (Part 1: Core Dashboard & Views)

**Phase Goal**: Build a web dashboard for monitoring signals, performance, and system status
**Duration**: 1-2 weeks (Part 1)
**Prerequisites**: Phases 1-4 complete (MVP, News, Tracking, ML)

---

## Phase 5 Part 1 Overview

### What We're Building
1. **Streamlit Dashboard App** - Main web application
2. **Signals View** - Current and historical signals
3. **Performance View** - Performance metrics and charts
4. **Portfolio View** - Current positions and P&L
5. **Data Services** - Backend services for dashboard

### Architecture Addition

```
trading_system/
├── dashboard/                       # NEW: Web dashboard
│   ├── __init__.py
│   ├── app.py                       # Main Streamlit app
│   ├── config.py                    # Dashboard configuration
│   ├── pages/
│   │   ├── __init__.py
│   │   ├── signals.py               # Signals view
│   │   ├── performance.py           # Performance view
│   │   ├── portfolio.py             # Portfolio view
│   │   └── overview.py              # Dashboard overview
│   ├── components/
│   │   ├── __init__.py
│   │   ├── charts.py                # Chart components
│   │   ├── tables.py                # Table components
│   │   ├── cards.py                 # Metric cards
│   │   └── filters.py               # Filter components
│   ├── services/
│   │   ├── __init__.py
│   │   ├── data_service.py          # Data fetching
│   │   ├── cache_service.py         # Caching layer
│   │   └── auth_service.py          # Authentication (optional)
│   └── static/
│       └── styles.css               # Custom CSS
```

---

## Task 5.1.1: Create Dashboard Module Structure

**Context**:
The dashboard provides a web interface for monitoring the trading system.

**Objective**:
Create the directory structure, configuration, and main app entry point.

**Files to Create**:
```
trading_system/dashboard/
├── __init__.py
├── app.py
├── config.py
├── pages/
│   └── __init__.py
├── components/
│   └── __init__.py
├── services/
│   └── __init__.py
└── static/
    └── styles.css
```

**Requirements**:

1. Create `config.py`:
```python
"""Dashboard configuration."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class DashboardConfig:
    """Configuration for the dashboard."""

    # App settings
    title: str = "Trading Assistant"
    page_icon: str = "📈"
    layout: str = "wide"

    # Data sources
    tracking_db_path: str = "tracking.db"
    feature_db_path: str = "features.db"

    # Refresh settings
    auto_refresh: bool = True
    refresh_interval_seconds: int = 300  # 5 minutes

    # Display settings
    default_lookback_days: int = 30
    max_signals_display: int = 100
    chart_height: int = 400

    # Theme
    theme: str = "light"  # "light" or "dark"

    # Authentication (optional)
    require_auth: bool = False
    auth_password_hash: str = ""

    # Performance thresholds for color coding
    win_rate_good: float = 0.55
    win_rate_warning: float = 0.45
    sharpe_good: float = 1.5
    sharpe_warning: float = 0.5


@dataclass
class ChartConfig:
    """Configuration for charts."""

    # Colors
    positive_color: str = "#22c55e"
    negative_color: str = "#ef4444"
    neutral_color: str = "#6b7280"
    primary_color: str = "#3b82f6"

    # Plotly template
    template: str = "plotly_white"

    # Default sizes
    height: int = 400
    width: Optional[int] = None  # None = auto


# Page definitions
PAGES = {
    "overview": {
        "title": "Overview",
        "icon": "🏠",
        "description": "Dashboard overview",
    },
    "signals": {
        "title": "Signals",
        "icon": "📊",
        "description": "View and analyze signals",
    },
    "performance": {
        "title": "Performance",
        "icon": "📈",
        "description": "Performance metrics and analytics",
    },
    "portfolio": {
        "title": "Portfolio",
        "icon": "💼",
        "description": "Current positions and P&L",
    },
    "news": {
        "title": "News",
        "icon": "📰",
        "description": "News feed and sentiment",
    },
    "settings": {
        "title": "Settings",
        "icon": "⚙️",
        "description": "System settings",
    },
}
```

2. Create `app.py`:
```python
"""Main Streamlit dashboard application."""

import streamlit as st
from pathlib import Path

from trading_system.dashboard.config import DashboardConfig, PAGES


def setup_page_config(config: DashboardConfig):
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title=config.title,
        page_icon=config.page_icon,
        layout=config.layout,
        initial_sidebar_state="expanded",
    )


def load_custom_css():
    """Load custom CSS styles."""
    css_path = Path(__file__).parent / "static" / "styles.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_sidebar(config: DashboardConfig) -> str:
    """Render sidebar navigation."""
    with st.sidebar:
        st.title(f"{config.page_icon} {config.title}")
        st.divider()

        # Navigation
        selected_page = st.radio(
            "Navigation",
            options=list(PAGES.keys()),
            format_func=lambda x: f"{PAGES[x]['icon']} {PAGES[x]['title']}",
            label_visibility="collapsed",
        )

        st.divider()

        # Quick stats (placeholder)
        st.caption("Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Win Rate", "65%", "+2%")
        with col2:
            st.metric("Total R", "+12.5R", "+1.2R")

        st.divider()

        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        # Auto-refresh status
        if config.auto_refresh:
            st.caption(f"Auto-refresh: {config.refresh_interval_seconds}s")

        # Footer
        st.divider()
        st.caption("Trading Assistant v1.0")

    return selected_page


def main():
    """Main application entry point."""
    config = DashboardConfig()

    # Setup
    setup_page_config(config)
    load_custom_css()

    # Authentication check (if enabled)
    if config.require_auth:
        if not check_authentication(config):
            render_login_page()
            return

    # Render sidebar and get selected page
    selected_page = render_sidebar(config)

    # Render selected page
    if selected_page == "overview":
        from trading_system.dashboard.pages.overview import render_overview
        render_overview(config)
    elif selected_page == "signals":
        from trading_system.dashboard.pages.signals import render_signals
        render_signals(config)
    elif selected_page == "performance":
        from trading_system.dashboard.pages.performance import render_performance
        render_performance(config)
    elif selected_page == "portfolio":
        from trading_system.dashboard.pages.portfolio import render_portfolio
        render_portfolio(config)
    elif selected_page == "news":
        from trading_system.dashboard.pages.news import render_news
        render_news(config)
    elif selected_page == "settings":
        from trading_system.dashboard.pages.settings import render_settings
        render_settings(config)


def check_authentication(config: DashboardConfig) -> bool:
    """Check if user is authenticated."""
    return st.session_state.get("authenticated", False)


def render_login_page():
    """Render login page."""
    st.title("🔐 Login")

    password = st.text_input("Password", type="password")

    if st.button("Login"):
        # Simple password check (use proper auth in production)
        if password:  # Add real validation
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid password")


if __name__ == "__main__":
    main()
```

3. Create `static/styles.css`:
```css
/* Custom styles for Trading Assistant Dashboard */

/* Main container */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 1.5rem;
    color: white;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.metric-card-value {
    font-size: 2rem;
    font-weight: bold;
    margin-bottom: 0.25rem;
}

.metric-card-label {
    font-size: 0.875rem;
    opacity: 0.9;
}

/* Signal cards */
.signal-card {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
    background: white;
}

.signal-card.buy {
    border-left: 4px solid #22c55e;
}

.signal-card.sell {
    border-left: 4px solid #ef4444;
}

.signal-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}

.signal-card-symbol {
    font-size: 1.25rem;
    font-weight: bold;
}

.signal-card-conviction {
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
}

.conviction-high {
    background: #dcfce7;
    color: #166534;
}

.conviction-medium {
    background: #fef3c7;
    color: #92400e;
}

.conviction-low {
    background: #fee2e2;
    color: #991b1b;
}

/* Tables */
.data-table {
    width: 100%;
    border-collapse: collapse;
}

.data-table th {
    background: #f9fafb;
    padding: 0.75rem;
    text-align: left;
    font-weight: 600;
    border-bottom: 2px solid #e5e7eb;
}

.data-table td {
    padding: 0.75rem;
    border-bottom: 1px solid #e5e7eb;
}

.data-table tr:hover {
    background: #f9fafb;
}

/* Positive/Negative colors */
.positive {
    color: #22c55e;
}

.negative {
    color: #ef4444;
}

/* Status badges */
.status-badge {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 500;
}

.status-active {
    background: #dbeafe;
    color: #1d4ed8;
}

.status-closed {
    background: #f3f4f6;
    color: #6b7280;
}

.status-pending {
    background: #fef3c7;
    color: #92400e;
}

/* Charts container */
.chart-container {
    background: white;
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

/* Section headers */
.section-header {
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e5e7eb;
}

/* Filters bar */
.filters-bar {
    background: #f9fafb;
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
}

/* Insight boxes */
.insight-box {
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 0.75rem;
}

.insight-box.success {
    background: #f0fdf4;
    border-left: 4px solid #22c55e;
}

.insight-box.warning {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
}

.insight-box.danger {
    background: #fef2f2;
    border-left: 4px solid #ef4444;
}

.insight-box.info {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
}
```

4. Create `__init__.py`:
```python
"""Dashboard module for Trading Assistant."""

from trading_system.dashboard.config import DashboardConfig, ChartConfig, PAGES

__all__ = [
    "DashboardConfig",
    "ChartConfig",
    "PAGES",
]
```

**Acceptance Criteria**:
- [ ] All directories and files created
- [ ] Streamlit app runs without errors
- [ ] Sidebar navigation works
- [ ] Custom CSS loads correctly
- [ ] Config properly structured

**Tests to Write**:
```python
def test_dashboard_config():
    """Test dashboard config defaults."""
    config = DashboardConfig()
    assert config.title == "Trading Assistant"
    assert config.refresh_interval_seconds == 300

def test_pages_defined():
    """Test all pages are defined."""
    from trading_system.dashboard.config import PAGES
    assert "overview" in PAGES
    assert "signals" in PAGES
    assert "performance" in PAGES
```

---

## Task 5.1.2: Create Data Service

**Context**:
The dashboard needs a service layer to fetch and cache data.

**Objective**:
Create data service for accessing tracking data, signals, and performance metrics.

**Files to Create**:
```
trading_system/dashboard/services/
├── __init__.py
├── data_service.py
└── cache_service.py
```

**Requirements**:

1. Create `data_service.py`:
```python
"""Data service for dashboard."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from trading_system.tracking.storage.sqlite_store import SQLiteTrackingStore
from trading_system.tracking.performance_calculator import PerformanceCalculator
from trading_system.tracking.analytics.signal_analytics import SignalAnalyzer
from trading_system.tracking.reports.leaderboard import LeaderboardGenerator
from trading_system.tracking.models import SignalStatus, TrackedSignal


@dataclass
class DashboardData:
    """Container for dashboard data."""

    # Signals
    recent_signals: List[TrackedSignal]
    pending_signals: List[TrackedSignal]
    active_signals: List[TrackedSignal]

    # Performance
    metrics_30d: dict
    metrics_90d: dict
    metrics_all: dict

    # Analytics
    analytics: dict
    leaderboard: dict

    # Counts
    signal_counts: dict

    # Timestamp
    fetched_at: datetime


class DashboardDataService:
    """
    Service for fetching dashboard data.

    Example:
        service = DashboardDataService(tracking_db_path="tracking.db")
        data = service.get_dashboard_data()

        print(f"Recent signals: {len(data.recent_signals)}")
        print(f"Win rate: {data.metrics_30d['win_rate']:.0%}")
    """

    def __init__(
        self,
        tracking_db_path: str = "tracking.db",
        feature_db_path: str = "features.db",
    ):
        """
        Initialize data service.

        Args:
            tracking_db_path: Path to tracking database.
            feature_db_path: Path to feature database.
        """
        self.tracking_db_path = tracking_db_path
        self.feature_db_path = feature_db_path

    def get_dashboard_data(self) -> DashboardData:
        """Get all dashboard data."""
        store = SQLiteTrackingStore(self.tracking_db_path)

        try:
            # Signals
            recent_signals = store.get_recent_signals(days=30)
            pending_signals = store.get_signals_by_status(SignalStatus.PENDING, limit=50)
            active_signals = store.get_signals_by_status(SignalStatus.ACTIVE, limit=50)

            # Performance calculator
            calculator = PerformanceCalculator(store)

            metrics_30d = calculator.calculate_rolling_metrics(window_days=30)
            metrics_90d = calculator.calculate_rolling_metrics(window_days=90)
            metrics_all = calculator.calculate_metrics()

            # Analytics
            analyzer = SignalAnalyzer(store)
            analytics = analyzer.analyze()

            # Leaderboard
            lb_generator = LeaderboardGenerator(store)
            leaderboard = lb_generator.generate_monthly()

            # Counts
            signal_counts = store.count_signals_by_status()

            return DashboardData(
                recent_signals=recent_signals,
                pending_signals=pending_signals,
                active_signals=active_signals,
                metrics_30d=metrics_30d,
                metrics_90d=metrics_90d,
                metrics_all=metrics_all.__dict__,
                analytics=analytics.__dict__,
                leaderboard=leaderboard.__dict__,
                signal_counts=signal_counts,
                fetched_at=datetime.now(),
            )

        finally:
            store.close()

    def get_signals_dataframe(
        self,
        days: int = 30,
        status: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get signals as a DataFrame."""
        store = SQLiteTrackingStore(self.tracking_db_path)

        try:
            if status:
                signals = store.get_signals_by_status(
                    SignalStatus(status),
                    limit=500,
                )
            else:
                signals = store.get_recent_signals(days=days)

            if not signals:
                return pd.DataFrame()

            # Convert to DataFrame
            data = []
            for signal in signals:
                data.append({
                    "id": signal.id,
                    "symbol": signal.symbol,
                    "direction": signal.direction.value,
                    "signal_type": signal.signal_type,
                    "conviction": signal.conviction.value,
                    "status": signal.status.value,
                    "entry_price": signal.entry_price,
                    "target_price": signal.target_price,
                    "stop_price": signal.stop_price,
                    "combined_score": signal.combined_score,
                    "technical_score": signal.technical_score,
                    "news_score": signal.news_score,
                    "created_at": signal.created_at,
                    "was_delivered": signal.was_delivered,
                })

            df = pd.DataFrame(data)
            return df

        finally:
            store.close()

    def get_performance_timeseries(
        self,
        days: int = 90,
    ) -> pd.DataFrame:
        """Get performance timeseries for charting."""
        store = SQLiteTrackingStore(self.tracking_db_path)

        try:
            calculator = PerformanceCalculator(store)

            # Get equity curve
            curve = calculator.get_equity_curve(
                starting_equity=100000,
                start_date=date.today() - timedelta(days=days),
                end_date=date.today(),
            )

            if not curve:
                return pd.DataFrame()

            df = pd.DataFrame(curve)
            df["date"] = pd.to_datetime(df["date"])
            return df

        finally:
            store.close()

    def get_signal_details(self, signal_id: str) -> Optional[dict]:
        """Get detailed information for a single signal."""
        store = SQLiteTrackingStore(self.tracking_db_path)

        try:
            signal = store.get_signal(signal_id)
            if not signal:
                return None

            outcome = store.get_outcome(signal_id)

            return {
                "signal": signal.__dict__,
                "outcome": outcome.__dict__ if outcome else None,
            }

        finally:
            store.close()

    def get_daily_summary(self, target_date: date = None) -> dict:
        """Get summary for a specific date."""
        target_date = target_date or date.today()
        store = SQLiteTrackingStore(self.tracking_db_path)

        try:
            signals = store.get_signals_by_date_range(
                start_date=target_date,
                end_date=target_date,
            )

            return {
                "date": target_date.isoformat(),
                "total_signals": len(signals),
                "by_conviction": {
                    "HIGH": len([s for s in signals if s.conviction.value == "HIGH"]),
                    "MEDIUM": len([s for s in signals if s.conviction.value == "MEDIUM"]),
                    "LOW": len([s for s in signals if s.conviction.value == "LOW"]),
                },
                "by_direction": {
                    "BUY": len([s for s in signals if s.direction.value == "BUY"]),
                    "SELL": len([s for s in signals if s.direction.value == "SELL"]),
                },
                "signals": [s.__dict__ for s in signals],
            }

        finally:
            store.close()

    def get_strategy_comparison(self) -> pd.DataFrame:
        """Get strategy performance comparison."""
        store = SQLiteTrackingStore(self.tracking_db_path)

        try:
            lb_gen = LeaderboardGenerator(store)
            leaderboard = lb_gen.generate_all_time()

            if not leaderboard.entries:
                return pd.DataFrame()

            data = []
            for entry in leaderboard.entries:
                data.append({
                    "rank": entry.rank,
                    "strategy": entry.display_name,
                    "total_r": entry.total_r,
                    "win_rate": entry.win_rate,
                    "expectancy": entry.expectancy_r,
                    "trades": entry.trade_count,
                    "trend": entry.trend,
                })

            return pd.DataFrame(data)

        finally:
            store.close()
```

2. Create `cache_service.py`:
```python
"""Caching service for dashboard."""

import streamlit as st
from datetime import datetime, timedelta
from typing import Any, Callable, Optional
from functools import wraps

from loguru import logger


class CacheService:
    """
    Caching service using Streamlit's cache.

    Uses st.cache_data for data caching with TTL support.
    """

    @staticmethod
    def cache_data(ttl_seconds: int = 300):
        """
        Decorator for caching data with TTL.

        Args:
            ttl_seconds: Time-to-live in seconds.

        Returns:
            Cached function decorator.
        """
        def decorator(func: Callable):
            @st.cache_data(ttl=ttl_seconds)
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator

    @staticmethod
    def clear_cache():
        """Clear all cached data."""
        st.cache_data.clear()
        logger.info("Dashboard cache cleared")

    @staticmethod
    def get_cached_or_compute(
        key: str,
        compute_func: Callable,
        ttl_seconds: int = 300,
    ) -> Any:
        """
        Get cached value or compute it.

        Args:
            key: Cache key.
            compute_func: Function to compute value if not cached.
            ttl_seconds: Time-to-live in seconds.

        Returns:
            Cached or computed value.
        """
        # Use session state for manual caching
        cache_key = f"cache_{key}"
        expiry_key = f"expiry_{key}"

        now = datetime.now()

        # Check if cached and not expired
        if cache_key in st.session_state:
            expiry = st.session_state.get(expiry_key, now)
            if now < expiry:
                return st.session_state[cache_key]

        # Compute and cache
        value = compute_func()
        st.session_state[cache_key] = value
        st.session_state[expiry_key] = now + timedelta(seconds=ttl_seconds)

        return value


# Cached data fetchers using Streamlit decorators
@st.cache_data(ttl=300)
def get_cached_dashboard_data(_service) -> dict:
    """Get cached dashboard data."""
    return _service.get_dashboard_data()


@st.cache_data(ttl=300)
def get_cached_signals_df(_service, days: int, status: str = None):
    """Get cached signals DataFrame."""
    return _service.get_signals_dataframe(days=days, status=status)


@st.cache_data(ttl=300)
def get_cached_performance_ts(_service, days: int):
    """Get cached performance timeseries."""
    return _service.get_performance_timeseries(days=days)


@st.cache_data(ttl=300)
def get_cached_strategy_comparison(_service):
    """Get cached strategy comparison."""
    return _service.get_strategy_comparison()
```

3. Create `services/__init__.py`:
```python
"""Dashboard services."""

from trading_system.dashboard.services.data_service import (
    DashboardData,
    DashboardDataService,
)
from trading_system.dashboard.services.cache_service import (
    CacheService,
    get_cached_dashboard_data,
    get_cached_signals_df,
    get_cached_performance_ts,
    get_cached_strategy_comparison,
)

__all__ = [
    "DashboardData",
    "DashboardDataService",
    "CacheService",
    "get_cached_dashboard_data",
    "get_cached_signals_df",
    "get_cached_performance_ts",
    "get_cached_strategy_comparison",
]
```

**Acceptance Criteria**:
- [ ] Data service fetches all required data
- [ ] DataFrame conversion works correctly
- [ ] Caching reduces database calls
- [ ] Cache TTL works as expected
- [ ] Error handling for missing data

---

## Task 5.1.3: Create Chart Components

**Context**:
Charts are essential for visualizing performance and signals.

**Objective**:
Create reusable Plotly chart components.

**Files to Create**:
```
trading_system/dashboard/components/charts.py
```

**Requirements**:

```python
"""Chart components for dashboard."""

from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trading_system.dashboard.config import ChartConfig


def create_equity_curve(
    df: pd.DataFrame,
    config: ChartConfig = None,
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
        rows=2, cols=1,
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
        row=1, col=1,
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
            row=1, col=1,
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
            row=2, col=1,
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
    config: ChartConfig = None,
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

    fig = go.Figure(go.Indicator(
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
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig


def create_returns_distribution(
    returns: List[float],
    config: ChartConfig = None,
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
    fig.add_trace(go.Histogram(
        x=winners,
        name="Winners",
        marker_color=config.positive_color,
        opacity=0.7,
        nbinsx=20,
    ))

    fig.add_trace(go.Histogram(
        x=losers,
        name="Losers",
        marker_color=config.negative_color,
        opacity=0.7,
        nbinsx=20,
    ))

    # Add mean line
    import numpy as np
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
    config: ChartConfig = None,
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
    colors = [
        config.positive_color if v > 0 else config.negative_color
        for v in df_sorted[metric]
    ]

    fig = go.Figure(go.Bar(
        x=df_sorted[metric],
        y=df_sorted["strategy"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.2f}" for v in df_sorted[metric]],
        textposition="outside",
    ))

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
    config: ChartConfig = None,
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
        rows=1, cols=2,
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
        row=1, col=1,
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
        row=1, col=2,
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
    config: ChartConfig = None,
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
    fig.add_trace(go.Bar(
        name="Win Rate (%)",
        x=levels,
        y=win_rates,
        marker_color=config.primary_color,
        text=[f"{v:.0f}%" for v in win_rates],
        textposition="outside",
    ))

    fig.add_trace(go.Bar(
        name="Avg R",
        x=levels,
        y=avg_r,
        marker_color=config.positive_color,
        text=[f"{v:.2f}R" for v in avg_r],
        textposition="outside",
    ))

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
    config: ChartConfig = None,
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
    import numpy as np

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

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=month_names,
        y=[str(y) for y in years],
        colorscale="RdYlGn",
        zmid=0,
        text=[[f"{v:.1f}R" if not np.isnan(v) else "" for v in row] for row in z],
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}R<extra></extra>",
    ))

    fig.update_layout(
        height=max(200, len(years) * 60),
        template=config.template,
        title="Monthly Performance (R-Multiples)",
    )

    return fig
```

**Acceptance Criteria**:
- [ ] Equity curve renders with drawdown subplot
- [ ] Win rate gauge shows proper colors
- [ ] Returns distribution shows winners/losers
- [ ] Strategy comparison bar chart works
- [ ] Day-of-week analysis chart renders
- [ ] All charts handle empty data gracefully

---

## Task 5.1.4: Create Table and Card Components

**Context**:
Tables and cards display signal and metric information.

**Objective**:
Create reusable table and card components.

**Files to Create**:
```
trading_system/dashboard/components/
├── tables.py
└── cards.py
```

**Requirements**:

1. Create `tables.py`:
```python
"""Table components for dashboard."""

from typing import List, Optional

import pandas as pd
import streamlit as st

from trading_system.dashboard.config import ChartConfig


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
        col for col in [
            "symbol", "direction", "conviction", "status",
            "entry_price", "target_price", "stop_price",
            "combined_score", "created_at"
        ] if col in df_display.columns
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

        data.append({
            "Rank": f"#{entry.get('rank', 0)}",
            "Strategy": entry.get("display_name", entry.get("strategy_name", "")),
            "Total R": f"{entry.get('total_r', 0):+.1f}R",
            "Win Rate": f"{entry.get('win_rate', 0) * 100:.0f}%",
            "Trades": entry.get("trade_count", 0),
            "Trend": trend,
        })

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
        result_color = "green" if r_mult > 0 else "red"

        data.append({
            "Symbol": trade.get("symbol", ""),
            "Direction": trade.get("direction", ""),
            "Result": f"{r_mult:+.2f}R",
            "Exit": trade.get("exit_reason", "Manual"),
            "Date": trade.get("exit_date", ""),
        })

    df = pd.DataFrame(data)
    st.dataframe(df, hide_index=True, use_container_width=True)
```

2. Create `cards.py`:
```python
"""Card components for dashboard."""

from typing import Optional, Tuple

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
```

3. Create `components/__init__.py`:
```python
"""Dashboard components."""

from trading_system.dashboard.components.charts import (
    create_equity_curve,
    create_win_rate_gauge,
    create_returns_distribution,
    create_strategy_comparison_chart,
    create_performance_by_day,
    create_conviction_breakdown,
    create_monthly_performance_heatmap,
)
from trading_system.dashboard.components.tables import (
    render_signals_table,
    render_performance_table,
    render_leaderboard_table,
    render_recent_trades_table,
)
from trading_system.dashboard.components.cards import (
    render_metric_card,
    render_metric_row,
    render_signal_card,
    render_insight_box,
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
```

**Acceptance Criteria**:
- [ ] Signals table renders with proper formatting
- [ ] Performance table shows key metrics
- [ ] Leaderboard table displays ranks and trends
- [ ] Signal cards show all relevant info
- [ ] Insight boxes render with correct colors
- [ ] Status badges display properly

---

## Task 5.1.5: Create Overview Page

**Context**:
The overview page provides a high-level dashboard summary.

**Objective**:
Create the main overview/home page.

**Files to Create**:
```
trading_system/dashboard/pages/overview.py
```

**Requirements**:

```python
"""Overview page for dashboard."""

import streamlit as st
from datetime import datetime, timedelta

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
    render_metric_row([
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
            "value": str(metrics_30d.get('total_signals', 0)),
            "help": "Number of trades taken",
        },
    ])

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
                render_signal_card({
                    "symbol": signal.symbol,
                    "direction": signal.direction.value,
                    "conviction": signal.conviction.value,
                    "combined_score": signal.combined_score,
                    "signal_type": signal.signal_type,
                    "entry_price": signal.entry_price,
                    "target_price": signal.target_price,
                    "stop_price": signal.stop_price,
                })

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
```

**Acceptance Criteria**:
- [ ] Key metrics display correctly
- [ ] Equity curve renders
- [ ] Win rate gauge shows
- [ ] Active signals display
- [ ] Recent trades show
- [ ] Strategy leaderboard works
- [ ] Insights render with correct styling
- [ ] Handles missing data gracefully

---

## Task 5.1.6: Create Signals Page

**Context**:
The signals page shows current and historical signals with filtering.

**Objective**:
Create a comprehensive signals view with filters and details.

**Files to Create**:
```
trading_system/dashboard/pages/signals.py
```

**Requirements**:

```python
"""Signals page for dashboard."""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

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
```

**Acceptance Criteria**:
- [ ] Filters work correctly
- [ ] Table view displays properly
- [ ] Card view renders all signals
- [ ] Pagination works
- [ ] Signal details show complete info
- [ ] Outcome data displays if available

---

## Task 5.1.7: Create Performance Page

**Context**:
The performance page provides detailed performance analytics.

**Objective**:
Create a comprehensive performance analytics view.

**Files to Create**:
```
trading_system/dashboard/pages/performance.py
```

**Requirements**:

```python
"""Performance page for dashboard."""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

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

    render_metric_row([
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
    ])

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
```

**Acceptance Criteria**:
- [ ] Summary metrics display correctly
- [ ] Equity curve renders for different periods
- [ ] Win rate gauge shows accurate value
- [ ] Returns distribution chart works
- [ ] Performance breakdowns by day/conviction/strategy work
- [ ] Data export functionality works
- [ ] Insights display with correct styling

---

## Summary: Part 1 Tasks

| Task | Description | Key Deliverable |
|------|-------------|-----------------|
| 5.1.1 | Module Structure | Config, app.py, CSS, directory layout |
| 5.1.2 | Data Service | DashboardDataService, caching |
| 5.1.3 | Chart Components | Equity curve, gauges, distributions |
| 5.1.4 | Table/Card Components | Signals table, metric cards, badges |
| 5.1.5 | Overview Page | Main dashboard with key metrics |
| 5.1.6 | Signals Page | Signal list with filters and details |
| 5.1.7 | Performance Page | Performance analytics and breakdowns |

---

**Part 2 will cover**:
- 5.2.1: Portfolio Page (positions, P&L)
- 5.2.2: News Page (news feed, sentiment)
- 5.2.3: Settings Page (configuration)
- 5.2.4: Authentication (optional)
- 5.2.5: Deployment Configuration
- 5.2.6: CLI Command for Dashboard
- 5.2.7: Advanced Features (alerts, mobile)
- 5.2.8: Integration Tests

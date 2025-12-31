"""Dashboard configuration."""

from dataclasses import dataclass
from typing import Dict, Optional


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
PAGES: Dict[str, Dict[str, str]] = {
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


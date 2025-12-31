"""Tests for dashboard configuration."""

import pytest

from trading_system.dashboard.config import DashboardConfig, ChartConfig, PAGES


def test_dashboard_config():
    """Test dashboard config defaults."""
    config = DashboardConfig()
    assert config.title == "Trading Assistant"
    assert config.refresh_interval_seconds == 300
    assert config.page_icon == "📈"
    assert config.layout == "wide"
    assert config.auto_refresh is True
    assert config.default_lookback_days == 30
    assert config.max_signals_display == 100
    assert config.chart_height == 400
    assert config.theme == "light"
    assert config.require_auth is False
    assert config.win_rate_good == 0.55
    assert config.win_rate_warning == 0.45
    assert config.sharpe_good == 1.5
    assert config.sharpe_warning == 0.5


def test_chart_config():
    """Test chart config defaults."""
    config = ChartConfig()
    assert config.positive_color == "#22c55e"
    assert config.negative_color == "#ef4444"
    assert config.neutral_color == "#6b7280"
    assert config.primary_color == "#3b82f6"
    assert config.template == "plotly_white"
    assert config.height == 400
    assert config.width is None


def test_pages_defined():
    """Test all pages are defined."""
    assert "overview" in PAGES
    assert "signals" in PAGES
    assert "performance" in PAGES
    assert "portfolio" in PAGES
    assert "news" in PAGES
    assert "settings" in PAGES


def test_pages_structure():
    """Test that all pages have required keys."""
    required_keys = {"title", "icon", "description"}
    for page_key, page_data in PAGES.items():
        assert isinstance(page_data, dict)
        assert required_keys.issubset(page_data.keys()), f"Page {page_key} missing required keys"
        assert isinstance(page_data["title"], str)
        assert isinstance(page_data["icon"], str)
        assert isinstance(page_data["description"], str)


def test_dashboard_config_custom_values():
    """Test dashboard config with custom values."""
    config = DashboardConfig(
        title="Custom Title",
        refresh_interval_seconds=600,
        auto_refresh=False,
        require_auth=True,
    )
    assert config.title == "Custom Title"
    assert config.refresh_interval_seconds == 600
    assert config.auto_refresh is False
    assert config.require_auth is True


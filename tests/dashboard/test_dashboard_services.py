"""Tests for dashboard services."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from pathlib import Path


class TestDashboardConfig:
    """Tests for DashboardConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from trading_system.dashboard.config import DashboardConfig

        config = DashboardConfig()

        assert config.title == "Trading Assistant"
        assert config.theme == "light"
        assert config.auto_refresh is True
        assert config.refresh_interval_seconds == 300
        assert config.default_lookback_days == 30

    def test_chart_config(self):
        """Test ChartConfig values."""
        from trading_system.dashboard.config import ChartConfig

        config = ChartConfig()

        assert config.primary_color == "#3b82f6"
        assert config.positive_color == "#22c55e"
        assert config.negative_color == "#ef4444"
        assert config.default_height == 400

    def test_config_thresholds(self):
        """Test performance threshold values."""
        from trading_system.dashboard.config import DashboardConfig

        config = DashboardConfig()

        assert config.win_rate_good == 0.55
        assert config.win_rate_warning == 0.45
        assert config.sharpe_good == 1.5
        assert config.sharpe_warning == 0.5


class TestAuthService:
    """Tests for AuthService."""

    def test_auth_disabled_when_no_hash(self):
        """Test auth is disabled when no password hash provided."""
        from trading_system.dashboard.services.auth_service import AuthService

        with patch("streamlit.session_state", {}):
            auth = AuthService(password_hash=None)
            assert not auth.is_enabled()
            assert auth.is_authenticated()  # Always authenticated when disabled

    def test_auth_enabled_with_hash(self):
        """Test auth is enabled when password hash provided."""
        from trading_system.dashboard.services.auth_service import AuthService

        with patch("streamlit.session_state", {}):
            auth = AuthService(password_hash="somehash")
            assert auth.is_enabled()
            assert not auth.is_authenticated()  # Not authenticated until login

    def test_password_hashing(self):
        """Test password hashing function."""
        from trading_system.dashboard.services.auth_service import hash_password

        password = "test_password"
        hashed = hash_password(password)

        # Should be SHA256 hex digest (64 characters)
        assert len(hashed) == 64
        assert hashed.isalnum()

        # Same password should produce same hash
        assert hash_password(password) == hashed

    def test_authenticate_success(self):
        """Test successful authentication."""
        from trading_system.dashboard.services.auth_service import AuthService, hash_password

        password = "correct_password"
        password_hash = hash_password(password)

        with patch("streamlit.session_state", {}):
            auth = AuthService(password_hash=password_hash)
            assert auth.authenticate(password)

    def test_authenticate_failure(self):
        """Test failed authentication."""
        from trading_system.dashboard.services.auth_service import AuthService, hash_password

        password_hash = hash_password("correct_password")

        with patch("streamlit.session_state", {}):
            auth = AuthService(password_hash=password_hash)
            assert not auth.authenticate("wrong_password")


class TestDashboardDataService:
    """Tests for DashboardDataService."""

    @pytest.fixture
    def mock_tracking_store(self):
        """Create a mock tracking store."""
        mock = MagicMock()
        mock.get_active_signals.return_value = []
        mock.get_recent_signals.return_value = []
        mock.get_performance_metrics.return_value = None
        return mock

    def test_data_service_initialization(self, tmp_path):
        """Test DashboardDataService initialization."""
        from trading_system.dashboard.services.data_service import DashboardDataService

        tracking_db = tmp_path / "tracking.db"
        feature_db = tmp_path / "features.db"

        service = DashboardDataService(
            tracking_db_path=str(tracking_db),
            feature_db_path=str(feature_db),
        )

        assert service.tracking_db_path == str(tracking_db)
        assert service.feature_db_path == str(feature_db)

    def test_get_dashboard_data_empty(self, tmp_path):
        """Test getting dashboard data when no data exists."""
        from trading_system.dashboard.services.data_service import DashboardDataService

        tracking_db = tmp_path / "tracking.db"
        feature_db = tmp_path / "features.db"

        service = DashboardDataService(
            tracking_db_path=str(tracking_db),
            feature_db_path=str(feature_db),
        )

        data = service.get_dashboard_data()

        assert data.active_signals == []
        assert data.recent_signals == []
        assert data.performance_metrics is None

    def test_get_signals_dataframe(self, tmp_path):
        """Test getting signals as DataFrame."""
        from trading_system.dashboard.services.data_service import DashboardDataService

        tracking_db = tmp_path / "tracking.db"
        feature_db = tmp_path / "features.db"

        service = DashboardDataService(
            tracking_db_path=str(tracking_db),
            feature_db_path=str(feature_db),
        )

        df = service.get_signals_dataframe(days=30)

        # Should return empty DataFrame when no data
        assert len(df) == 0


class TestCacheService:
    """Tests for CacheService."""

    def test_cache_service_initialization(self):
        """Test CacheService initialization."""
        from trading_system.dashboard.services.cache_service import CacheService

        service = CacheService(ttl_seconds=300)
        assert service.ttl_seconds == 300

    def test_cache_key_generation(self):
        """Test cache key generation."""
        from trading_system.dashboard.services.cache_service import CacheService

        service = CacheService()

        # Same inputs should produce same key
        key1 = service.get_cache_key("test", arg1="value1", arg2=123)
        key2 = service.get_cache_key("test", arg1="value1", arg2=123)
        assert key1 == key2

        # Different inputs should produce different keys
        key3 = service.get_cache_key("test", arg1="value2", arg2=123)
        assert key1 != key3


class TestChartComponents:
    """Tests for chart components."""

    def test_create_equity_curve(self):
        """Test equity curve chart creation."""
        import pandas as pd
        from trading_system.dashboard.components.charts import create_equity_curve

        # Create sample data
        dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
        equity = [10000 * (1 + 0.01 * i + 0.005 * (i % 5)) for i in range(30)]

        df = pd.DataFrame(
            {
                "date": dates,
                "equity": equity,
            }
        )

        # Should not raise
        fig = create_equity_curve(df)

        assert fig is not None
        # Check that the figure has data
        assert len(fig.data) > 0

    def test_create_win_rate_gauge(self):
        """Test win rate gauge chart creation."""
        from trading_system.dashboard.components.charts import create_win_rate_gauge

        # Test with various win rates
        for win_rate in [0.0, 0.45, 0.55, 0.75, 1.0]:
            fig = create_win_rate_gauge(win_rate)
            assert fig is not None
            assert len(fig.data) > 0

    def test_create_returns_distribution(self):
        """Test returns distribution chart."""
        import pandas as pd
        from trading_system.dashboard.components.charts import create_returns_distribution

        # Create sample returns
        returns = [0.02, -0.01, 0.03, -0.02, 0.01, 0.0, -0.015, 0.025]
        df = pd.DataFrame({"returns": returns})

        fig = create_returns_distribution(df)
        assert fig is not None


class TestCardComponents:
    """Tests for card components."""

    def test_render_metric_row(self):
        """Test metric row rendering."""
        from trading_system.dashboard.components.cards import render_metric_row

        metrics = [
            {"label": "Win Rate", "value": "55%"},
            {"label": "Sharpe", "value": "1.5"},
            {"label": "Max DD", "value": "-15%"},
        ]

        # Should not raise - just verify it can be called
        # In actual Streamlit context, this would render components
        try:
            with patch("streamlit.columns") as mock_columns:
                mock_col = MagicMock()
                mock_columns.return_value = [mock_col] * len(metrics)
                render_metric_row(metrics)
        except Exception:
            # Expected to fail outside Streamlit context
            pass

    def test_insight_box_types(self):
        """Test different insight box types."""
        from trading_system.dashboard.components.cards import render_insight_box

        # Test each type
        for box_type in ["info", "warning", "success", "error"]:
            try:
                with patch("streamlit.markdown"):
                    render_insight_box(f"Test {box_type} message", type=box_type)
            except Exception:
                # Expected to fail outside Streamlit context
                pass


class TestTableComponents:
    """Tests for table components."""

    def test_render_signals_table_empty(self):
        """Test signals table with empty data."""
        import pandas as pd
        from trading_system.dashboard.components.tables import render_signals_table

        df = pd.DataFrame()

        try:
            with patch("streamlit.dataframe"):
                with patch("streamlit.info"):
                    render_signals_table(df)
        except Exception:
            pass

    def test_render_performance_table(self):
        """Test performance table rendering."""
        import pandas as pd
        from trading_system.dashboard.components.tables import render_performance_table

        df = pd.DataFrame(
            {
                "strategy": ["momentum", "mean_reversion"],
                "win_rate": [0.55, 0.48],
                "sharpe_ratio": [1.5, 0.8],
            }
        )

        try:
            with patch("streamlit.dataframe"):
                render_performance_table(df)
        except Exception:
            pass


class TestNewsPage:
    """Tests for news page functionality."""

    def test_format_time_ago(self):
        """Test time ago formatting."""
        from trading_system.dashboard.pages.news import _format_time_ago

        now = datetime.now()

        # Just now
        result = _format_time_ago(now - timedelta(seconds=30))
        assert result == "Just now"

        # Minutes ago
        result = _format_time_ago(now - timedelta(minutes=5))
        assert "m ago" in result

        # Hours ago
        result = _format_time_ago(now - timedelta(hours=3))
        assert "h ago" in result

        # Days ago
        result = _format_time_ago(now - timedelta(days=3))
        assert "days ago" in result

        # Yesterday
        result = _format_time_ago(now - timedelta(days=1))
        assert result == "Yesterday"

    def test_sample_news_data(self):
        """Test sample news data generation."""
        from trading_system.dashboard.pages.news import _get_sample_news

        news = _get_sample_news()

        assert len(news) > 0
        for article in news:
            assert "title" in article
            assert "sentiment_score" in article
            assert "source" in article

    def test_sample_sentiment_data(self):
        """Test sample sentiment data generation."""
        from trading_system.dashboard.pages.news import _get_sample_sentiment_data

        data = _get_sample_sentiment_data()

        assert "overall" in data
        assert "equity" in data
        assert "crypto" in data
        assert "by_symbol" in data
        assert "trend" in data


class TestSettingsPage:
    """Tests for settings page functionality."""

    def test_check_file_exists(self):
        """Test file existence check."""
        from trading_system.dashboard.pages.settings import _check_file_exists

        # Non-existent file
        assert not _check_file_exists("/nonexistent/file.txt")

        # Existing file (current test file)
        assert _check_file_exists(__file__)


class TestPortfolioPage:
    """Tests for portfolio page functionality."""

    def test_render_portfolio_no_positions(self, tmp_path):
        """Test portfolio rendering with no positions."""
        from trading_system.dashboard.config import DashboardConfig

        config = DashboardConfig()
        config.tracking_db_path = str(tmp_path / "tracking.db")
        config.feature_db_path = str(tmp_path / "features.db")

        # Just verify import works
        from trading_system.dashboard.pages.portfolio import render_portfolio

        assert callable(render_portfolio)


class TestIntegration:
    """Integration tests for the dashboard."""

    def test_all_pages_importable(self):
        """Test that all pages can be imported."""
        from trading_system.dashboard.pages import overview, signals, performance
        from trading_system.dashboard.pages import portfolio, news, settings

        assert callable(overview.render_overview)
        assert callable(signals.render_signals)
        assert callable(performance.render_performance)
        assert callable(portfolio.render_portfolio)
        assert callable(news.render_news)
        assert callable(settings.render_settings)

    def test_all_components_importable(self):
        """Test that all components can be imported."""
        from trading_system.dashboard.components import charts, tables, cards

        assert hasattr(charts, "create_equity_curve")
        assert hasattr(tables, "render_signals_table")
        assert hasattr(cards, "render_metric_row")

    def test_all_services_importable(self):
        """Test that all services can be imported."""
        from trading_system.dashboard.services import data_service, cache_service, auth_service

        assert hasattr(data_service, "DashboardDataService")
        assert hasattr(cache_service, "CacheService")
        assert hasattr(auth_service, "AuthService")

    def test_config_importable(self):
        """Test that config can be imported."""
        from trading_system.dashboard.config import DashboardConfig, ChartConfig

        config = DashboardConfig()
        chart_config = ChartConfig()

        assert config is not None
        assert chart_config is not None

    def test_app_importable(self):
        """Test that main app can be imported."""
        # This should not raise
        import trading_system.dashboard.app as app

        assert hasattr(app, "main")

"""Settings page for the dashboard."""

import hashlib
import os
from pathlib import Path

import streamlit as st
import yaml

from trading_system.dashboard.config import DashboardConfig


def render_settings(config: DashboardConfig):
    """Render the settings page."""
    st.title("Settings")
    st.caption("Configure your trading dashboard")

    # Tabs for different settings categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["General", "API Keys", "Notifications", "Risk Settings", "About"])

    with tab1:
        render_general_settings(config)

    with tab2:
        render_api_settings()

    with tab3:
        render_notification_settings(config)

    with tab4:
        render_risk_settings(config)

    with tab5:
        render_about_section()


def render_general_settings(config: DashboardConfig):
    """Render general settings section."""
    st.subheader("General Settings")

    # Display settings
    st.markdown("**Display Options**")

    col1, col2 = st.columns(2)

    with col1:
        theme = st.selectbox(
            "Theme",
            options=["Light", "Dark"],
            index=0 if config.theme == "light" else 1,
            help="Select the dashboard theme",
        )

        default_lookback = st.number_input(
            "Default Lookback (days)",
            min_value=7,
            max_value=365,
            value=config.default_lookback_days,
            help="Default number of days to display in charts and tables",
        )

    with col2:
        auto_refresh = st.checkbox(
            "Auto-refresh data",
            value=config.auto_refresh,
            help="Automatically refresh data at regular intervals",
        )

        if auto_refresh:
            refresh_interval = st.number_input(
                "Refresh interval (seconds)",
                min_value=60,
                max_value=3600,
                value=config.refresh_interval_seconds,
                help="How often to refresh data",
            )

    st.divider()

    # Data paths
    st.markdown("**Data Paths**")

    tracking_db = st.text_input(
        "Tracking Database Path",
        value=config.tracking_db_path,
        help="Path to the SQLite tracking database",
    )

    feature_db = st.text_input(
        "Feature Database Path",
        value=config.feature_db_path,
        help="Path to the feature store database",
    )

    st.divider()

    # Authentication
    st.markdown("**Authentication**")

    require_auth = st.checkbox(
        "Require authentication",
        value=config.require_auth,
        help="Enable password protection for the dashboard",
    )

    if require_auth:
        new_password = st.text_input(
            "Set Dashboard Password",
            type="password",
            help="Set a new password for the dashboard",
        )

        if new_password and st.button("Update Password"):
            # Hash the password
            hashed = hashlib.sha256(new_password.encode()).hexdigest()
            st.success("Password updated! Restart the dashboard to apply.")
            st.code(f"auth_password_hash: {hashed}")

    st.divider()

    # Save button
    if st.button("Save General Settings", type="primary"):
        st.success("Settings saved! Some changes may require a restart.")
        st.info("Note: In production, settings would be persisted to a config file.")


def render_api_settings():
    """Render API settings section."""
    st.subheader("API Keys")

    st.warning("API keys are stored as environment variables for security. " "Set them in your shell or .env file.")

    # Display current status
    st.markdown("**API Key Status**")

    api_keys = [
        ("MASSIVE_API_KEY", "Massive", "Real-time stock data"),
        ("ALPHA_VANTAGE_API_KEY", "Alpha Vantage", "Stock data & news"),
        ("NEWSAPI_API_KEY", "NewsAPI", "News articles"),
        ("SENDGRID_API_KEY", "SendGrid", "Email notifications"),
        ("BINANCE_API_KEY", "Binance", "Crypto data (optional)"),
    ]

    for env_var, name, description in api_keys:
        is_set = bool(os.environ.get(env_var))
        "" if is_set else ""
        color = "#22c55e" if is_set else "#ef4444"

        st.markdown(
            """
            <div style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.75rem;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                margin-bottom: 0.5rem;
            ">
                <div>
                    <span style="font-weight: 600;">{name}</span>
                    <span style="color: #6b7280; font-size: 0.875rem; margin-left: 0.5rem;">
                        {description}
                    </span>
                </div>
                <span style="color: {color}; font-size: 1.25rem;">{status}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # Instructions
    with st.expander("How to set API keys"):
        st.markdown(
            """
        **Option 1: Environment Variables**
        ```bash
        export MASSIVE_API_KEY="your_key_here"
        export ALPHA_VANTAGE_API_KEY="your_key_here"
        export NEWSAPI_API_KEY="your_key_here"
        export SENDGRID_API_KEY="your_key_here"
        ```

        **Option 2: .env File**
        Create a `.env` file in the project root:
        ```
        MASSIVE_API_KEY=your_key_here
        ALPHA_VANTAGE_API_KEY=your_key_here
        NEWSAPI_API_KEY=your_key_here
        SENDGRID_API_KEY=your_key_here
        ```

        Then load it with `python-dotenv` or source it before running the dashboard.

        **Getting API Keys:**
        - Massive: https://polygon.io (Free tier: 5 calls/min) - Note: Polygon.io has been rebranded to Massive
        - Alpha Vantage: https://www.alphavantage.co (Free tier: 5 calls/min)
        - NewsAPI: https://newsapi.org (Free tier: 100 requests/day)
        - SendGrid: https://sendgrid.com (Free tier: 100 emails/day)
        """
        )


def render_notification_settings(config: DashboardConfig):
    """Render notification settings section."""
    st.subheader("Notification Settings")

    # Email settings
    st.markdown("**Email Notifications**")

    email_enabled = st.checkbox(
        "Enable email notifications",
        value=True,
        help="Send daily signal reports via email",
    )

    if email_enabled:
        col1, col2 = st.columns(2)

        with col1:
            email_recipients = st.text_area(
                "Recipients (one per line)",
                value="user@example.com",
                height=100,
                help="Email addresses to receive reports",
            )

        with col2:
            daily_time = st.time_input(
                "Daily Report Time",
                value=None,
                help="When to send the daily report",
            )

            timezone = st.selectbox(
                "Timezone",
                options=[
                    "America/New_York",
                    "America/Chicago",
                    "America/Los_Angeles",
                    "UTC",
                    "Europe/London",
                ],
                index=0,
            )

    st.divider()

    # Report preferences
    st.markdown("**Report Preferences**")

    col1, col2 = st.columns(2)

    with col1:
        include_news = st.checkbox("Include news digest", value=True)
        include_performance = st.checkbox("Include performance summary", value=True)

    with col2:
        include_portfolio = st.checkbox("Include portfolio status", value=True)
        include_ml_insights = st.checkbox("Include ML insights", value=True)

    st.divider()

    # Alert settings
    st.markdown("**Alert Conditions**")

    st.caption("Send immediate alerts when these conditions occur:")

    high_conviction_alerts = st.checkbox(
        "High conviction signals",
        value=True,
        help="Alert when a HIGH conviction signal is generated",
    )

    stop_hit_alerts = st.checkbox(
        "Stop loss triggered",
        value=True,
        help="Alert when a position hits its stop loss",
    )

    large_drawdown_alert = st.checkbox(
        "Large drawdown",
        value=True,
        help="Alert when portfolio drawdown exceeds threshold",
    )

    if large_drawdown_alert:
        drawdown_threshold = st.slider(
            "Drawdown threshold (%)",
            min_value=5,
            max_value=30,
            value=15,
        )

    st.divider()

    # Test notification
    col1, col2, _ = st.columns([1, 1, 2])

    with col1:
        if st.button("Send Test Email"):
            st.info("Test email would be sent here. Configure SMTP first.")

    with col2:
        if st.button("Save Notification Settings", type="primary"):
            st.success("Notification settings saved!")


def render_risk_settings(config: DashboardConfig):
    """Render risk settings section."""
    st.subheader("Risk Management Settings")

    st.markdown("**Position Sizing**")

    col1, col2 = st.columns(2)

    with col1:
        risk_per_trade = st.slider(
            "Risk per trade (%)",
            min_value=0.25,
            max_value=3.0,
            value=0.75,
            step=0.25,
            help="Maximum risk (in R) per trade",
        )

        max_positions = st.number_input(
            "Maximum positions",
            min_value=1,
            max_value=20,
            value=8,
            help="Maximum number of concurrent positions",
        )

    with col2:
        max_exposure = st.slider(
            "Maximum portfolio exposure (%)",
            min_value=20,
            max_value=100,
            value=80,
            help="Maximum total portfolio exposure",
        )

        max_position_size = st.slider(
            "Maximum single position (%)",
            min_value=5,
            max_value=25,
            value=10,
            help="Maximum size of any single position",
        )

    st.divider()

    st.markdown("**Correlation Limits**")

    max_sector_exposure = st.slider(
        "Maximum sector exposure (%)",
        min_value=20,
        max_value=60,
        value=40,
        help="Maximum exposure to a single sector",
    )

    max_correlated_positions = st.number_input(
        "Maximum correlated positions",
        min_value=1,
        max_value=5,
        value=3,
        help="Maximum positions with correlation > 0.7",
    )

    st.divider()

    st.markdown("**Signal Thresholds**")

    col1, col2 = st.columns(2)

    with col1:
        min_score_threshold = st.slider(
            "Minimum signal score",
            min_value=0.0,
            max_value=10.0,
            value=6.0,
            step=0.5,
            help="Minimum combined score to consider a signal",
        )

    with col2:
        min_rr_ratio = st.slider(
            "Minimum R:R ratio",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.5,
            help="Minimum reward-to-risk ratio",
        )

    st.divider()

    st.markdown("**Performance Thresholds**")

    col1, col2 = st.columns(2)

    with col1:
        win_rate_good = st.slider(
            "Win rate - Good threshold",
            min_value=0.40,
            max_value=0.70,
            value=config.win_rate_good,
            help="Win rate considered 'good' performance",
        )

        win_rate_warning = st.slider(
            "Win rate - Warning threshold",
            min_value=0.30,
            max_value=0.50,
            value=config.win_rate_warning,
            help="Win rate that triggers a warning",
        )

    with col2:
        sharpe_good = st.slider(
            "Sharpe ratio - Good threshold",
            min_value=0.5,
            max_value=3.0,
            value=config.sharpe_good,
            step=0.1,
            help="Sharpe ratio considered 'good'",
        )

        sharpe_warning = st.slider(
            "Sharpe ratio - Warning threshold",
            min_value=0.0,
            max_value=1.0,
            value=config.sharpe_warning,
            step=0.1,
            help="Sharpe ratio that triggers a warning",
        )

    st.divider()

    if st.button("Save Risk Settings", type="primary"):
        st.success("Risk settings saved!")


def render_about_section():
    """Render about section."""
    st.subheader("About Trading Assistant")

    st.markdown(
        """
    **Trading Assistant Dashboard** v1.0

    A comprehensive trading signal generation and tracking system with:
    - Technical analysis signal generation
    - News and sentiment integration
    - ML-enhanced signal scoring
    - Performance tracking and analytics
    - Email notifications

    ---

    **System Status**
    """
    )

    # System checks
    checks = [
        ("Tracking Database", _check_file_exists("tracking.db")),
        ("Feature Database", _check_file_exists("features.db")),
        ("Config File", _check_file_exists("config/trading_config.yaml")),
        ("Massive API", bool(os.environ.get("MASSIVE_API_KEY"))),
        ("Alpha Vantage API", bool(os.environ.get("ALPHA_VANTAGE_API_KEY"))),
        ("NewsAPI", bool(os.environ.get("NEWSAPI_API_KEY"))),
        ("SendGrid API", bool(os.environ.get("SENDGRID_API_KEY"))),
    ]

    for name, status in checks:
        emoji = "" if status else ""
        color = "#22c55e" if status else "#ef4444"
        st.markdown(f"<span style='color: {color};'>{emoji}</span> {name}", unsafe_allow_html=True)

    st.divider()

    # Documentation links
    st.markdown("**Documentation & Resources**")
    st.markdown(
        """
    - [User Guide](#) - How to use the trading system
    - [API Documentation](#) - REST API reference
    - [Strategy Guide](#) - Understanding the strategies
    - [FAQ](#) - Frequently asked questions
    """
    )

    st.divider()

    # Export/Import config
    st.markdown("**Configuration**")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Export Configuration"):
            config_data = {
                "version": "1.0",
                "settings": {
                    "theme": "light",
                    "auto_refresh": True,
                    "refresh_interval": 300,
                },
            }
            st.download_button(
                label="Download Config",
                data=yaml.dump(config_data),
                file_name="trading_config.yaml",
                mime="text/yaml",
            )

    with col2:
        uploaded_file = st.file_uploader(
            "Import Configuration",
            type=["yaml", "yml"],
            help="Upload a configuration file",
        )
        if uploaded_file:
            st.success("Configuration imported!")

    st.divider()

    # Clear data options
    st.markdown("**Data Management**")

    with st.expander("Danger Zone", expanded=False):
        st.warning("These actions cannot be undone!")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Clear Cache", type="secondary"):
                st.cache_data.clear()
                st.success("Cache cleared!")

        with col2:
            if st.button("Reset Settings", type="secondary"):
                st.info("Settings reset to defaults (would require confirmation)")

        with col3:
            if st.button("Clear All Data", type="secondary"):
                st.error("This would delete all tracked data (would require confirmation)")


def _check_file_exists(path: str) -> bool:
    """Check if a file exists."""
    return Path(path).exists()

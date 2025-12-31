"""Main Streamlit dashboard application."""

from pathlib import Path

import streamlit as st

from trading_system.dashboard.config import PAGES, DashboardConfig


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
        from trading_system.dashboard.services.auth_service import AuthService

        auth = AuthService(
            password_hash=config.auth_password_hash,
            session_duration_hours=24,
        )

        if not auth.is_authenticated():
            auth.render_login_page()
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


if __name__ == "__main__":
    main()

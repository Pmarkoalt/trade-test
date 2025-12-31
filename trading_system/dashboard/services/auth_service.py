"""Authentication service for the dashboard."""

import hashlib
import os
import secrets
from datetime import datetime, timedelta
from typing import Optional

import streamlit as st


class AuthService:
    """
    Authentication service for the trading dashboard.

    Provides simple password-based authentication with session management.

    Example:
        auth = AuthService(password_hash="sha256_hash_of_password")

        if not auth.is_authenticated():
            auth.render_login_page()
            return

        # Authenticated content here
    """

    def __init__(
        self,
        password_hash: Optional[str] = None,
        session_duration_hours: int = 24,
    ):
        """
        Initialize authentication service.

        Args:
            password_hash: SHA256 hash of the password. If None, auth is disabled.
            session_duration_hours: How long a session remains valid.
        """
        self.password_hash = password_hash or os.environ.get("DASHBOARD_PASSWORD_HASH")
        self.session_duration = timedelta(hours=session_duration_hours)

        # Initialize session state
        if "auth_token" not in st.session_state:
            st.session_state.auth_token = None
        if "auth_timestamp" not in st.session_state:
            st.session_state.auth_timestamp = None

    def is_enabled(self) -> bool:
        """Check if authentication is enabled."""
        return bool(self.password_hash)

    def is_authenticated(self) -> bool:
        """
        Check if the current session is authenticated.

        Returns:
            True if authenticated, False otherwise.
        """
        if not self.is_enabled():
            return True  # Auth disabled, allow all

        # Check for valid session
        if st.session_state.auth_token and st.session_state.auth_timestamp:
            # Check if session has expired
            elapsed = datetime.now() - st.session_state.auth_timestamp
            if elapsed < self.session_duration:
                return True
            else:
                # Session expired, clear it
                self.logout()

        return False

    def authenticate(self, password: str) -> bool:
        """
        Authenticate with a password.

        Args:
            password: The password to verify.

        Returns:
            True if authentication successful, False otherwise.
        """
        if not self.is_enabled():
            return True

        # Hash the provided password
        provided_hash = hashlib.sha256(password.encode()).hexdigest()

        if provided_hash == self.password_hash:
            # Create session
            st.session_state.auth_token = secrets.token_hex(32)
            st.session_state.auth_timestamp = datetime.now()
            return True

        return False

    def logout(self):
        """Clear the current session."""
        st.session_state.auth_token = None
        st.session_state.auth_timestamp = None

    def render_login_page(self):
        """Render the login page."""
        st.set_page_config(
            page_title="Login - Trading Assistant",
            page_icon="🔐",
            layout="centered",
        )

        # Center the login form
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.markdown(
                """
                <div style="text-align: center; margin-top: 4rem;">
                    <span style="font-size: 4rem;">📈</span>
                    <h1 style="margin-top: 1rem;">Trading Assistant</h1>
                    <p style="color: #6b7280;">Enter your password to continue</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # Login form
            with st.form("login_form"):
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password",
                )

                submitted = st.form_submit_button(
                    "Login",
                    use_container_width=True,
                    type="primary",
                )

                if submitted:
                    if self.authenticate(password):
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid password. Please try again.")

            # Footer
            st.markdown(
                """
                <div style="text-align: center; margin-top: 2rem; color: #9ca3af; font-size: 0.875rem;">
                    Trading Assistant Dashboard v1.0
                </div>
                """,
                unsafe_allow_html=True,
            )

    def render_logout_button(self, sidebar: bool = True):
        """
        Render a logout button.

        Args:
            sidebar: If True, render in sidebar. Otherwise in main area.
        """
        if not self.is_enabled():
            return

        container = st.sidebar if sidebar else st

        if container.button("🚪 Logout", use_container_width=True):
            self.logout()
            st.rerun()

    def get_session_info(self) -> Optional[dict]:
        """
        Get current session information.

        Returns:
            Dict with session info, or None if not authenticated.
        """
        if not self.is_authenticated():
            return None

        elapsed = datetime.now() - st.session_state.auth_timestamp
        remaining = self.session_duration - elapsed

        return {
            "token": st.session_state.auth_token[:8] + "...",  # Truncated for display
            "started_at": st.session_state.auth_timestamp,
            "expires_in": remaining,
            "expires_in_minutes": int(remaining.total_seconds() / 60),
        }


def hash_password(password: str) -> str:
    """
    Hash a password for storage.

    Args:
        password: The password to hash.

    Returns:
        SHA256 hash of the password.

    Example:
        hash = hash_password("my_secure_password")
        print(f"Set DASHBOARD_PASSWORD_HASH={hash}")
    """
    return hashlib.sha256(password.encode()).hexdigest()


def require_auth(
    password_hash: Optional[str] = None,
    session_duration_hours: int = 24,
):
    """
    Decorator-style function to require authentication.

    Call this at the start of your Streamlit app to require authentication.

    Args:
        password_hash: SHA256 hash of the password.
        session_duration_hours: Session duration in hours.

    Returns:
        True if authenticated, False otherwise (and renders login page).

    Example:
        if not require_auth():
            st.stop()  # Stop execution if not authenticated

        # Rest of your app here
    """
    auth = AuthService(
        password_hash=password_hash,
        session_duration_hours=session_duration_hours,
    )

    if not auth.is_enabled():
        return True

    if auth.is_authenticated():
        return True

    auth.render_login_page()
    return False

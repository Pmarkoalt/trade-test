"""Caching service for dashboard."""

from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Optional

from loguru import logger

try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

    # Create a mock st object for when streamlit is not available
    class MockStreamlit:
        class cache_data:
            @staticmethod
            def __call__(ttl=None):
                def decorator(func):
                    return func

                return decorator

            @staticmethod
            def clear():
                pass

        class session_state:
            pass

    st = MockStreamlit()


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
            if STREAMLIT_AVAILABLE:

                @st.cache_data(ttl=ttl_seconds)
                @wraps(func)
                def wrapper(*args, **kwargs):
                    return func(*args, **kwargs)

            else:
                # Fallback when streamlit is not available
                @wraps(func)
                def wrapper(*args, **kwargs):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def clear_cache():
        """Clear all cached data."""
        if STREAMLIT_AVAILABLE:
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
        if not STREAMLIT_AVAILABLE:
            # Fallback: just compute
            return compute_func()

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
if STREAMLIT_AVAILABLE:

    @st.cache_data(ttl=300)
    def get_cached_dashboard_data(_service) -> dict:
        """Get cached dashboard data."""
        result = _service.get_dashboard_data()
        return dict(result) if result else {}

    @st.cache_data(ttl=300)
    def get_cached_signals_df(_service, days: int, status: Optional[str] = None):
        """Get cached signals DataFrame."""
        return _service.get_signals_dataframe(days=days, status=status)

    @st.cache_data(ttl=300)
    def get_cached_performance_ts(_service, days: int):
        """Get cached performance timeseries."""
        return _service.get_performance_timeseries(days=days)

    @st.cache_data(ttl=300)
    def get_cached_strategy_comparison(_service) -> dict:
        """Get cached strategy comparison."""
        result = _service.get_strategy_comparison()
        return dict(result) if result else {}

else:
    # Fallback functions when streamlit is not available
    def get_cached_dashboard_data(_service) -> dict:
        """Get cached dashboard data."""
        result = _service.get_dashboard_data()
        return dict(result) if result else {}

    def get_cached_signals_df(_service, days: int, status: Optional[str] = None):
        """Get cached signals DataFrame."""
        return _service.get_signals_dataframe(days=days, status=status)

    def get_cached_performance_ts(_service, days: int):
        """Get cached performance timeseries."""
        return _service.get_performance_timeseries(days=days)

    def get_cached_strategy_comparison(_service):
        """Get cached strategy comparison."""
        return _service.get_strategy_comparison()

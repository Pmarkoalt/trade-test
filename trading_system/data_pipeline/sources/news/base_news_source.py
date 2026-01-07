"""Abstract base class for news data sources."""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Optional

from .models import NewsFetchResult

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when rate limit is exceeded and cannot wait."""


class NewsAPIError(Exception):
    """Base exception for news API errors."""

    def __init__(self, message: str, source: str, status_code: Optional[int] = None):
        self.message = message
        self.source = source
        self.status_code = status_code
        super().__init__(f"{source}: {message}")


class BaseNewsSource(ABC):
    """Abstract base class for news data sources."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit_per_minute: int = 10,
        max_retries: int = 3,
        retry_delay_seconds: float = 2.0,
    ):
        self.api_key = api_key
        self.rate_limit_per_minute = rate_limit_per_minute
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self._last_request_time: Optional[datetime] = None
        self._request_count: int = 0
        self._daily_request_count: int = 0
        self._daily_reset_time: Optional[datetime] = None

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the name of this news source."""

    @abstractmethod
    async def fetch_articles(
        self,
        symbols: List[str],
        lookback_hours: int = 24,
        max_articles_per_symbol: int = 10,
    ) -> NewsFetchResult:
        """Fetch news articles for given symbols.

        Args:
            symbols: List of stock/crypto symbols to search for
            lookback_hours: How far back to search (default 24 hours)
            max_articles_per_symbol: Maximum articles to return per symbol

        Returns:
            NewsFetchResult with fetched articles
        """

    @abstractmethod
    async def fetch_market_news(
        self,
        category: str = "general",  # 'general', 'technology', 'crypto', etc.
        lookback_hours: int = 24,
        max_articles: int = 20,
    ) -> NewsFetchResult:
        """Fetch general market news.

        Args:
            category: News category to fetch
            lookback_hours: How far back to search
            max_articles: Maximum articles to return

        Returns:
            NewsFetchResult with fetched articles
        """

    async def _respect_rate_limit(self, max_wait_seconds: int = 120) -> None:
        """Wait if necessary to respect rate limits.

        Args:
            max_wait_seconds: Maximum time to wait for rate limit reset

        Raises:
            RateLimitError: If wait time exceeds max_wait_seconds
        """
        now = datetime.now()

        # Reset daily counter if a day has passed
        if self._daily_reset_time is None or now - self._daily_reset_time > timedelta(days=1):
            self._daily_request_count = 0
            self._daily_reset_time = now

        if self._last_request_time is None:
            self._last_request_time = now
            self._request_count = 1
            self._daily_request_count += 1
            return

        # Reset per-minute counter if a minute has passed
        if now - self._last_request_time > timedelta(minutes=1):
            self._request_count = 1
            self._last_request_time = now
            self._daily_request_count += 1
            return

        # Wait if we've hit the rate limit
        if self._request_count >= self.rate_limit_per_minute:
            wait_seconds = 60 - (now - self._last_request_time).seconds
            if wait_seconds > max_wait_seconds:
                raise RateLimitError(
                    f"Rate limit exceeded for {self.source_name}. "
                    f"Would need to wait {wait_seconds}s (max: {max_wait_seconds}s)"
                )
            if wait_seconds > 0:
                logger.info(f"{self.source_name}: Rate limit reached, waiting {wait_seconds}s")
                await asyncio.sleep(wait_seconds)
            self._request_count = 0
            self._last_request_time = datetime.now()

        self._request_count += 1
        self._daily_request_count += 1

    async def _retry_with_backoff(self, coro_factory, operation_name: str = "request"):
        """Execute a coroutine with exponential backoff retry.

        Args:
            coro: Coroutine to execute
            operation_name: Name for logging

        Returns:
            Result of the coroutine

        Raises:
            Exception: If all retries fail
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return await coro_factory()
            except RateLimitError:
                raise  # Don't retry rate limit errors
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay_seconds * (2**attempt)
                    logger.warning(
                        f"{self.source_name}: {operation_name} failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)

        logger.error(f"{self.source_name}: {operation_name} failed after {self.max_retries} attempts")
        raise last_error

    def get_rate_limit_status(self) -> dict:
        """Get current rate limit status.

        Returns:
            Dictionary with rate limit information
        """
        return {
            "source": self.source_name,
            "requests_this_minute": self._request_count,
            "limit_per_minute": self.rate_limit_per_minute,
            "requests_today": self._daily_request_count,
            "last_request": self._last_request_time.isoformat() if self._last_request_time else None,
        }

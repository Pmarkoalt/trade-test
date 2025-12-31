"""Abstract base class for news data sources."""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Optional

from .models import NewsArticle, NewsFetchResult


class BaseNewsSource(ABC):
    """Abstract base class for news data sources."""

    def __init__(self, api_key: Optional[str] = None, rate_limit_per_minute: int = 10):
        self.api_key = api_key
        self.rate_limit_per_minute = rate_limit_per_minute
        self._last_request_time: Optional[datetime] = None
        self._request_count: int = 0

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the name of this news source."""
        pass

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
        pass

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
        pass

    async def _respect_rate_limit(self) -> None:
        """Wait if necessary to respect rate limits."""
        if self._last_request_time is None:
            self._last_request_time = datetime.now()
            self._request_count = 1
            return

        # Reset counter if a minute has passed
        if datetime.now() - self._last_request_time > timedelta(minutes=1):
            self._request_count = 1
            self._last_request_time = datetime.now()
            return

        # Wait if we've hit the rate limit
        if self._request_count >= self.rate_limit_per_minute:
            wait_seconds = 60 - (datetime.now() - self._last_request_time).seconds
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
            self._request_count = 0
            self._last_request_time = datetime.now()

        self._request_count += 1


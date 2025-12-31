"""News aggregator that combines multiple news sources."""

import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .alpha_vantage_news import AlphaVantageNewsClient
from .base_news_source import BaseNewsSource
from .models import NewsArticle
from .newsapi_client import NewsAPIClient

logger = logging.getLogger(__name__)


class NewsAggregator:
    """Aggregate news from multiple sources."""

    def __init__(
        self,
        newsapi_key: Optional[str] = None,
        alpha_vantage_key: Optional[str] = None,
        enable_caching: bool = True,
        cache_ttl_minutes: int = 30,
    ):
        """Initialize news aggregator.

        Args:
            newsapi_key: NewsAPI.org API key
            alpha_vantage_key: Alpha Vantage API key
            enable_caching: Whether to cache results
            cache_ttl_minutes: Cache TTL in minutes
        """
        self.sources: List[BaseNewsSource] = []
        self._cache: Dict[str, Tuple[datetime, List[NewsArticle]]] = {}
        self.enable_caching = enable_caching
        self.cache_ttl_minutes = cache_ttl_minutes

        # Initialize available sources
        if newsapi_key:
            self.sources.append(NewsAPIClient(newsapi_key))
            logger.info("NewsAPI source enabled")
        if alpha_vantage_key:
            self.sources.append(AlphaVantageNewsClient(alpha_vantage_key))
            logger.info("Alpha Vantage News source enabled")

        if not self.sources:
            logger.warning("No news sources configured!")

    async def fetch_news_for_symbols(
        self,
        symbols: List[str],
        lookback_hours: int = 48,
        max_articles_per_symbol: int = 10,
    ) -> List[NewsArticle]:
        """Fetch news for symbols from all sources.

        Args:
            symbols: List of symbols to fetch news for
            lookback_hours: How far back to search
            max_articles_per_symbol: Max articles per symbol

        Returns:
            Deduplicated and sorted list of articles
        """
        # Check cache
        cache_key = self._make_cache_key(symbols, lookback_hours)
        cached = self._get_cached(cache_key)
        if cached:
            logger.info(f"Returning {len(cached)} cached articles")
            return cached

        # Fetch from all sources
        all_articles: List[NewsArticle] = []

        for source in self.sources:
            try:
                result = await source.fetch_articles(
                    symbols=symbols,
                    lookback_hours=lookback_hours,
                    max_articles_per_symbol=max_articles_per_symbol,
                )

                if result.success:
                    all_articles.extend(result.articles)
                    logger.info(f"Fetched {len(result.articles)} articles from {source.source_name}")
                else:
                    logger.warning(f"Failed to fetch from {source.source_name}: {result.error_message}")

            except Exception as e:
                logger.error(f"Error fetching from {source.source_name}: {e}")

        # Deduplicate
        deduplicated = self._deduplicate_articles(all_articles)

        # Sort by published date (newest first)
        deduplicated.sort(key=lambda a: a.published_at, reverse=True)

        # Cache result
        if self.enable_caching:
            self._cache[cache_key] = (datetime.now(), deduplicated)

        logger.info(f"Returning {len(deduplicated)} articles after deduplication")
        return deduplicated

    async def fetch_market_overview(
        self,
        categories: Optional[List[str]] = None,
        max_articles: int = 30,
    ) -> List[NewsArticle]:
        """Fetch general market news.

        Args:
            categories: News categories to fetch (default: ["business", "technology"])
            max_articles: Maximum total articles

        Returns:
            List of market news articles
        """
        if categories is None:
            categories = ["business", "technology"]

        all_articles: List[NewsArticle] = []

        for source in self.sources:
            for category in categories:
                try:
                    result = await source.fetch_market_news(
                        category=category,
                        max_articles=max_articles // len(categories),
                    )
                    if result.success:
                        all_articles.extend(result.articles)
                except Exception as e:
                    logger.error(f"Error fetching {category} from {source.source_name}: {e}")

        deduplicated = self._deduplicate_articles(all_articles)
        deduplicated.sort(key=lambda a: a.published_at, reverse=True)

        return deduplicated[:max_articles]

    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on title similarity.

        Args:
            articles: List of articles to deduplicate

        Returns:
            List of unique articles
        """
        seen_hashes = set()
        unique_articles = []

        for article in articles:
            # Create hash from normalized title
            title_normalized = article.title.lower().strip()
            title_hash = hashlib.md5(title_normalized.encode()).hexdigest()[:16]

            if title_hash not in seen_hashes:
                seen_hashes.add(title_hash)
                unique_articles.append(article)

        return unique_articles

    def _make_cache_key(self, symbols: List[str], lookback_hours: int) -> str:
        """Create cache key from symbols and lookback hours.

        Args:
            symbols: List of symbols
            lookback_hours: Lookback hours

        Returns:
            Cache key string
        """
        symbols_str = ",".join(sorted(symbols))
        return f"{symbols_str}_{lookback_hours}"

    def _get_cached(self, key: str) -> Optional[List[NewsArticle]]:
        """Get cached result if not expired.

        Args:
            key: Cache key

        Returns:
            Cached articles or None if expired/missing
        """
        if not self.enable_caching:
            return None

        if key not in self._cache:
            return None

        cached_time, articles = self._cache[key]
        age_minutes = (datetime.now() - cached_time).total_seconds() / 60

        if age_minutes > self.cache_ttl_minutes:
            del self._cache[key]
            return None

        return articles

    def clear_cache(self):
        """Clear the article cache."""
        self._cache.clear()


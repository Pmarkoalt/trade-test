"""Polygon ("Massive") news client implementation.

This integrates Polygon's News endpoint as an optional third news source.
It is intentionally lightweight and follows the same interface as other
news sources (NewsAPIClient, AlphaVantageNewsClient).

Note: The project already uses "Massive" as the Polygon OHLCV provider.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

try:
    import aiohttp
except ImportError:
    aiohttp = None

from .base_news_source import BaseNewsSource, NewsAPIError, RateLimitError
from .models import NewsArticle, NewsFetchResult

logger = logging.getLogger(__name__)


class PolygonNewsClient(BaseNewsSource):
    """Client for Polygon News API."""

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: str, rate_limit_per_minute: int = 5):
        if aiohttp is None:
            raise ImportError("aiohttp is required for PolygonNewsClient. Install it with: pip install aiohttp")
        super().__init__(api_key=api_key, rate_limit_per_minute=rate_limit_per_minute)
        if not api_key:
            raise ValueError("Polygon API key is required")

    @property
    def source_name(self) -> str:
        return "Polygon"

    async def fetch_articles(
        self,
        symbols: List[str],
        lookback_hours: int = 24,
        max_articles_per_symbol: int = 10,
    ) -> NewsFetchResult:
        all_articles: List[NewsArticle] = []
        errors: List[str] = []

        for symbol in symbols:
            try:
                await self._respect_rate_limit()
                articles = await self._retry_with_backoff(
                    lambda: self._fetch_for_symbol(symbol, lookback_hours, max_articles_per_symbol),
                    operation_name=f"fetch_articles({symbol})",
                )
                all_articles.extend(articles)
            except RateLimitError as e:
                errors.append(f"{symbol}: {e}")
            except Exception as e:
                errors.append(f"{symbol}: {e}")

        return NewsFetchResult(
            articles=all_articles,
            source=self.source_name,
            symbols_requested=symbols,
            success=len(errors) == 0,
            error_message="; ".join(errors) if errors else None,
        )

    async def fetch_market_news(
        self,
        category: str = "general",
        lookback_hours: int = 24,
        max_articles: int = 20,
    ) -> NewsFetchResult:
        # Polygon doesn't support the same categories as NewsAPI; we fetch general market news.
        try:
            await self._respect_rate_limit()
            articles = await self._retry_with_backoff(
                lambda: self._fetch_general(lookback_hours, max_articles),
                operation_name="fetch_market_news",
            )
            return NewsFetchResult(
                articles=articles,
                source=self.source_name,
                symbols_requested=[],
                success=True,
            )
        except Exception as e:
            return NewsFetchResult(
                articles=[],
                source=self.source_name,
                symbols_requested=[],
                success=False,
                error_message=str(e),
            )

    async def _fetch_for_symbol(self, symbol: str, lookback_hours: int, limit: int) -> List[NewsArticle]:
        from_dt = datetime.utcnow() - timedelta(hours=lookback_hours)

        params = {
            "ticker": symbol.upper(),
            "published_utc.gte": from_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "order": "desc",
            "sort": "published_utc",
            "limit": min(limit, 50),
            "apiKey": self.api_key,
        }

        url = f"{self.BASE_URL}/v2/reference/news"

        if aiohttp is None:
            raise ImportError("aiohttp is required for PolygonNewsClient. Install it with: pip install aiohttp")

        async with aiohttp.ClientSession() as session:
            headers = {"Accept-Encoding": "gzip, deflate"}
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 429:
                    raise RateLimitError("Polygon rate limit exceeded")
                if response.status != 200:
                    text = await response.text()
                    raise NewsAPIError(text, source=self.source_name, status_code=response.status)

                data: Dict[str, Any] = await response.json()
                results = data.get("results", []) or []
                return self._parse_articles(results, default_symbol=symbol.upper())

    async def _fetch_general(self, lookback_hours: int, limit: int) -> List[NewsArticle]:
        from_dt = datetime.utcnow() - timedelta(hours=lookback_hours)
        params = {
            "published_utc.gte": from_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "order": "desc",
            "sort": "published_utc",
            "limit": min(limit, 50),
            "apiKey": self.api_key,
        }
        url = f"{self.BASE_URL}/v2/reference/news"

        if aiohttp is None:
            raise ImportError("aiohttp is required for PolygonNewsClient. Install it with: pip install aiohttp")

        async with aiohttp.ClientSession() as session:
            headers = {"Accept-Encoding": "gzip, deflate"}
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 429:
                    raise RateLimitError("Polygon rate limit exceeded")
                if response.status != 200:
                    text = await response.text()
                    raise NewsAPIError(text, source=self.source_name, status_code=response.status)

                data: Dict[str, Any] = await response.json()
                results = data.get("results", []) or []
                return self._parse_articles(results, default_symbol=None)

    def _parse_articles(self, raw_articles: List[Dict[str, Any]], default_symbol: Optional[str]) -> List[NewsArticle]:
        articles: List[NewsArticle] = []

        for raw in raw_articles:
            try:
                published_str = raw.get("published_utc") or raw.get("publishedAt") or ""
                published_at = None
                if published_str:
                    # Polygon returns ISO strings like 2024-01-01T12:00:00Z
                    published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00"))

                article_id = raw.get("id") or raw.get("article_id") or f"polygon_{abs(hash(raw.get('article_url', '') or raw.get('id', '')))}"
                url = raw.get("article_url") or raw.get("url") or ""
                title = raw.get("title") or ""
                summary = raw.get("description") or raw.get("summary")

                symbols = []
                tickers = raw.get("tickers")
                if isinstance(tickers, list):
                    symbols.extend([t.upper() for t in tickers if isinstance(t, str)])
                if default_symbol and default_symbol not in symbols:
                    symbols.append(default_symbol)

                article = NewsArticle(
                    id=str(article_id),
                    source=(raw.get("publisher", {}) or {}).get("name", "Polygon"),
                    title=title,
                    summary=summary,
                    url=url,
                    published_at=published_at,
                    symbols=symbols,
                )

                articles.append(article)

            except Exception as e:
                logger.warning(f"Error parsing Polygon article: {e}")
                continue

        return articles

"""NewsAPI client implementation."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

try:
    import aiohttp
except ImportError:
    aiohttp = None

from .base_news_source import BaseNewsSource
from .models import NewsArticle, NewsFetchResult

logger = logging.getLogger(__name__)

# Company name mappings for better search results
SYMBOL_TO_COMPANY = {
    # Top equities
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google Alphabet",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
    "META": "Meta Facebook",
    "TSLA": "Tesla",
    "JPM": "JPMorgan",
    "V": "Visa",
    "JNJ": "Johnson Johnson",
    # Add more as needed...
    # Crypto
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "BNB": "Binance",
    "XRP": "Ripple XRP",
    "ADA": "Cardano",
    "SOL": "Solana",
    "DOT": "Polkadot",
    "MATIC": "Polygon MATIC",
    "LTC": "Litecoin",
    "LINK": "Chainlink",
}


class NewsAPIClient(BaseNewsSource):
    """Client for NewsAPI.org."""

    BASE_URL = "https://newsapi.org/v2"

    def __init__(self, api_key: str, rate_limit_per_minute: int = 5):
        """Initialize NewsAPI client.

        Args:
            api_key: NewsAPI.org API key
            rate_limit_per_minute: Max requests per minute (free tier: 100/day)

        Raises:
            ImportError: If aiohttp is not installed
        """
        if aiohttp is None:
            raise ImportError("aiohttp is required for NewsAPIClient. Install it with: pip install aiohttp")
        super().__init__(api_key, rate_limit_per_minute)
        if not api_key:
            raise ValueError("NewsAPI API key is required")

    @property
    def source_name(self) -> str:
        """Return the name of this news source."""
        return "NewsAPI"

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
        all_articles = []
        errors = []

        for symbol in symbols:
            try:
                await self._respect_rate_limit()

                # Build search query
                company_name = SYMBOL_TO_COMPANY.get(symbol, symbol)
                query = f'"{company_name}" OR "{symbol}"'

                # Calculate date range
                from_date = datetime.now() - timedelta(hours=lookback_hours)

                # Make API request
                articles = await self._search_everything(
                    query=query,
                    from_date=from_date,
                    page_size=max_articles_per_symbol,
                    sort_by="publishedAt",
                )

                # Tag articles with symbol
                for article in articles:
                    if symbol not in article.symbols:
                        article.symbols.append(symbol)

                all_articles.extend(articles)

            except Exception as e:
                logger.error(f"Error fetching news for {symbol}: {e}")
                errors.append(f"{symbol}: {str(e)}")

        return NewsFetchResult(
            articles=all_articles,
            source=self.source_name,
            symbols_requested=symbols,
            success=len(errors) == 0,
            error_message="; ".join(errors) if errors else None,
        )

    async def fetch_market_news(
        self,
        category: str = "business",
        lookback_hours: int = 24,
        max_articles: int = 20,
    ) -> NewsFetchResult:
        """Fetch general market news.

        Args:
            category: News category to fetch
            lookback_hours: How far back to search (not used for top-headlines)
            max_articles: Maximum articles to return

        Returns:
            NewsFetchResult with fetched articles
        """
        try:
            await self._respect_rate_limit()

            articles = await self._get_top_headlines(
                category=category,
                page_size=max_articles,
            )

            return NewsFetchResult(
                articles=articles,
                source=self.source_name,
                symbols_requested=[],
                success=True,
            )

        except Exception as e:
            logger.error(f"Error fetching market news: {e}")
            return NewsFetchResult(
                articles=[],
                source=self.source_name,
                symbols_requested=[],
                success=False,
                error_message=str(e),
            )

    async def _search_everything(
        self,
        query: str,
        from_date: datetime,
        page_size: int = 10,
        sort_by: str = "publishedAt",
    ) -> List[NewsArticle]:
        """Search all articles using /everything endpoint.

        Args:
            query: Search query string
            from_date: Start date for search
            page_size: Number of articles to return (max 100)
            sort_by: Sort order (publishedAt, relevancy, popularity)

        Returns:
            List of NewsArticle objects
        """
        params = {
            "q": query,
            "from": from_date.strftime("%Y-%m-%dT%H:%M:%S"),
            "sortBy": sort_by,
            "pageSize": min(page_size, 100),
            "language": "en",
            "apiKey": self.api_key,
        }

        url = f"{self.BASE_URL}/everything?{urlencode(params)}"

        if aiohttp is None:
            raise ImportError("aiohttp is required for NewsAPIClient. Install it with: pip install aiohttp")
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 429:
                    raise Exception("NewsAPI rate limit exceeded")
                if response.status != 200:
                    text = await response.text()
                    raise Exception(f"NewsAPI error {response.status}: {text}")

                data = await response.json()

                if data.get("status") != "ok":
                    raise Exception(f"NewsAPI error: {data.get('message', 'Unknown error')}")

                return self._parse_articles(data.get("articles", []))

    async def _get_top_headlines(
        self,
        category: str = "business",
        country: str = "us",
        page_size: int = 20,
    ) -> List[NewsArticle]:
        """Get top headlines using /top-headlines endpoint.

        Args:
            category: News category (business, entertainment, general, health, science, sports, technology)
            country: ISO 3166-1 alpha-2 country code (default: us)
            page_size: Number of articles to return (max 100)

        Returns:
            List of NewsArticle objects
        """
        params = {
            "category": category,
            "country": country,
            "pageSize": min(page_size, 100),
            "apiKey": self.api_key,
        }

        url = f"{self.BASE_URL}/top-headlines?{urlencode(params)}"

        if aiohttp is None:
            raise ImportError("aiohttp is required for NewsAPIClient. Install it with: pip install aiohttp")
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    text = await response.text()
                    raise Exception(f"NewsAPI error {response.status}: {text}")

                data = await response.json()
                return self._parse_articles(data.get("articles", []))

    def _parse_articles(self, raw_articles: List[Dict[str, Any]]) -> List[NewsArticle]:
        """Parse raw API response into NewsArticle objects.

        Args:
            raw_articles: List of raw article dictionaries from API response

        Returns:
            List of NewsArticle objects
        """
        articles = []

        for raw in raw_articles:
            try:
                # Parse published date
                published_str = raw.get("publishedAt", "")
                if published_str:
                    # Handle both Z and +00:00 timezone formats
                    published_str = published_str.replace("Z", "+00:00")
                    published_at = datetime.fromisoformat(published_str)
                else:
                    published_at = datetime.now()

                article = NewsArticle(
                    id=f"newsapi_{abs(hash(raw.get('url', '')))}",
                    source=raw.get("source", {}).get("name", "Unknown"),
                    title=raw.get("title", ""),
                    summary=raw.get("description", ""),
                    content=raw.get("content", ""),  # Often truncated
                    url=raw.get("url", ""),
                    published_at=published_at,
                )
                articles.append(article)

            except Exception as e:
                logger.warning(f"Error parsing article: {e}")
                continue

        return articles

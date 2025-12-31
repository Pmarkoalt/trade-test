# Agent Tasks: Phase 2 - News Integration

**Phase Goal**: Add news sentiment analysis to enhance signal quality
**Duration**: 2 weeks
**Prerequisites**: Phase 1 MVP complete (live data pipeline, signal generator, email, scheduler)

---

## Phase 2 Overview

### What We're Building
1. **News API Integration** - Fetch relevant news articles for portfolio symbols
2. **Sentiment Analysis** - Score news sentiment using VADER + financial lexicon
3. **Signal Enhancement** - Combine technical and news scores
4. **Email Updates** - Add news digest section to daily reports

### Architecture Addition

```
trading_system/
├── data_pipeline/
│   └── sources/
│       └── news/                    # NEW: News data sources
│           ├── __init__.py
│           ├── base_news_source.py
│           ├── newsapi_client.py
│           ├── alpha_vantage_news.py
│           └── news_aggregator.py
│
├── research/                        # NEW: Research & analysis
│   ├── __init__.py
│   ├── news_analyzer.py
│   ├── sentiment/
│   │   ├── __init__.py
│   │   ├── vader_analyzer.py
│   │   ├── financial_lexicon.py
│   │   └── sentiment_aggregator.py
│   ├── entity_extraction/
│   │   ├── __init__.py
│   │   └── ticker_extractor.py
│   └── relevance/
│       ├── __init__.py
│       └── relevance_scorer.py
│
├── signals/
│   └── generators/
│       └── news_signals.py          # NEW: News-based signals
│
└── output/
    └── email/
        └── templates/
            └── news_digest.html     # NEW: News section template
```

---

## Task 2.1.1: Create News Data Source Module Structure

**Context**:
We need to fetch news articles from multiple sources and standardize the format.

**Objective**:
Create the directory structure and base classes for news data sources.

**Files to Create**:
```
trading_system/data_pipeline/sources/news/
├── __init__.py
├── base_news_source.py
├── models.py
├── newsapi_client.py          # Stub
├── alpha_vantage_news.py      # Stub
└── news_aggregator.py         # Stub
```

**Requirements**:

1. Create `models.py` with news data models:
```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from enum import Enum


class SentimentLabel(str, Enum):
    """Sentiment classification."""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


@dataclass
class NewsArticle:
    """A news article with metadata."""
    # Core fields
    id: str
    source: str                      # e.g., "Reuters", "Bloomberg"
    title: str
    summary: Optional[str] = None
    content: Optional[str] = None    # Full article text (if available)
    url: str = ""
    published_at: datetime = None
    fetched_at: datetime = field(default_factory=datetime.now)

    # Extracted data (populated by analyzers)
    symbols: List[str] = field(default_factory=list)  # Mentioned tickers
    asset_classes: List[str] = field(default_factory=list)  # 'equity', 'crypto'

    # Sentiment (populated by sentiment analyzer)
    sentiment_score: Optional[float] = None      # -1.0 to +1.0
    sentiment_label: Optional[SentimentLabel] = None
    sentiment_confidence: Optional[float] = None  # 0.0 to 1.0

    # Relevance (populated by relevance scorer)
    relevance_score: Optional[float] = None      # 0.0 to 1.0
    event_type: Optional[str] = None             # 'earnings', 'merger', 'product', etc.

    # Processing flags
    is_processed: bool = False
    processing_error: Optional[str] = None

    def __post_init__(self):
        if self.published_at is None:
            self.published_at = datetime.now()


@dataclass
class NewsFetchResult:
    """Result of a news fetch operation."""
    articles: List[NewsArticle]
    source: str
    symbols_requested: List[str]
    fetch_time: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: Optional[str] = None
    rate_limit_remaining: Optional[int] = None
```

2. Create `base_news_source.py` with abstract base class:
```python
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
        max_articles_per_symbol: int = 10
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
        max_articles: int = 20
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
                import asyncio
                await asyncio.sleep(wait_seconds)
            self._request_count = 0
            self._last_request_time = datetime.now()

        self._request_count += 1
```

3. Create `__init__.py` with exports:
```python
from .models import NewsArticle, NewsFetchResult, SentimentLabel
from .base_news_source import BaseNewsSource
from .newsapi_client import NewsAPIClient
from .alpha_vantage_news import AlphaVantageNewsClient
from .news_aggregator import NewsAggregator

__all__ = [
    "NewsArticle",
    "NewsFetchResult",
    "SentimentLabel",
    "BaseNewsSource",
    "NewsAPIClient",
    "AlphaVantageNewsClient",
    "NewsAggregator",
]
```

**Acceptance Criteria**:
- [ ] All files created with proper structure
- [ ] Models are complete with all fields
- [ ] Base class has rate limiting logic
- [ ] Imports work: `from trading_system.data_pipeline.sources.news import NewsArticle`

---

## Task 2.1.2: Implement NewsAPI.org Client

**Context**:
NewsAPI.org provides news from 80,000+ sources. Free tier: 100 requests/day, articles up to 1 month old.

**Objective**:
Create a fully functional NewsAPI client.

**Files to Modify**:
- `trading_system/data_pipeline/sources/news/newsapi_client.py`

**Requirements**:

1. Implement `NewsAPIClient`:
```python
import aiohttp
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from urllib.parse import urlencode
import logging

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
        """
        super().__init__(api_key, rate_limit_per_minute)
        if not api_key:
            raise ValueError("NewsAPI API key is required")

    @property
    def source_name(self) -> str:
        return "NewsAPI"

    async def fetch_articles(
        self,
        symbols: List[str],
        lookback_hours: int = 24,
        max_articles_per_symbol: int = 10
    ) -> NewsFetchResult:
        """Fetch news articles for given symbols."""
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
                    sort_by="publishedAt"
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
            error_message="; ".join(errors) if errors else None
        )

    async def fetch_market_news(
        self,
        category: str = "business",
        lookback_hours: int = 24,
        max_articles: int = 20
    ) -> NewsFetchResult:
        """Fetch general market news."""
        try:
            await self._respect_rate_limit()

            articles = await self._get_top_headlines(
                category=category,
                page_size=max_articles
            )

            return NewsFetchResult(
                articles=articles,
                source=self.source_name,
                symbols_requested=[],
                success=True
            )

        except Exception as e:
            logger.error(f"Error fetching market news: {e}")
            return NewsFetchResult(
                articles=[],
                source=self.source_name,
                symbols_requested=[],
                success=False,
                error_message=str(e)
            )

    async def _search_everything(
        self,
        query: str,
        from_date: datetime,
        page_size: int = 10,
        sort_by: str = "publishedAt"
    ) -> List[NewsArticle]:
        """Search all articles using /everything endpoint."""
        params = {
            "q": query,
            "from": from_date.strftime("%Y-%m-%dT%H:%M:%S"),
            "sortBy": sort_by,
            "pageSize": min(page_size, 100),
            "language": "en",
            "apiKey": self.api_key
        }

        url = f"{self.BASE_URL}/everything?{urlencode(params)}"

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
        page_size: int = 20
    ) -> List[NewsArticle]:
        """Get top headlines using /top-headlines endpoint."""
        params = {
            "category": category,
            "country": country,
            "pageSize": min(page_size, 100),
            "apiKey": self.api_key
        }

        url = f"{self.BASE_URL}/top-headlines?{urlencode(params)}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    text = await response.text()
                    raise Exception(f"NewsAPI error {response.status}: {text}")

                data = await response.json()
                return self._parse_articles(data.get("articles", []))

    def _parse_articles(self, raw_articles: List[Dict[str, Any]]) -> List[NewsArticle]:
        """Parse raw API response into NewsArticle objects."""
        articles = []

        for raw in raw_articles:
            try:
                # Parse published date
                published_str = raw.get("publishedAt", "")
                if published_str:
                    published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                else:
                    published_at = datetime.now()

                article = NewsArticle(
                    id=f"newsapi_{hash(raw.get('url', ''))}",
                    source=raw.get("source", {}).get("name", "Unknown"),
                    title=raw.get("title", ""),
                    summary=raw.get("description", ""),
                    content=raw.get("content", ""),  # Often truncated
                    url=raw.get("url", ""),
                    published_at=published_at
                )
                articles.append(article)

            except Exception as e:
                logger.warning(f"Error parsing article: {e}")
                continue

        return articles
```

2. Create configuration in `trading_system/data_pipeline/config.py` (update existing):
```python
class NewsAPIConfig(BaseModel):
    """NewsAPI.org configuration."""
    api_key: str
    rate_limit_per_minute: int = 5
    max_articles_per_symbol: int = 10
    lookback_hours: int = 48
```

**API Documentation Reference**:
- Endpoint: `https://newsapi.org/v2/everything`
- Parameters: `q`, `from`, `to`, `sortBy`, `pageSize`, `language`
- Response: `{"status": "ok", "articles": [...]}`

**Acceptance Criteria**:
- [ ] Can fetch news for AAPL, MSFT (with valid API key)
- [ ] Can fetch news for BTC, ETH
- [ ] Rate limiting prevents 429 errors
- [ ] Articles are properly parsed
- [ ] Handles API errors gracefully
- [ ] Unit tests pass (with mocked responses)

**Test File to Create**: `tests/test_newsapi_client.py`

---

## Task 2.1.3: Implement Alpha Vantage News Client

**Context**:
Alpha Vantage provides news with sentiment already scored. Free tier: 25 requests/day.

**Objective**:
Create an Alpha Vantage News client as a secondary source.

**Files to Modify**:
- `trading_system/data_pipeline/sources/news/alpha_vantage_news.py`

**Requirements**:

1. Implement `AlphaVantageNewsClient`:
```python
import aiohttp
from datetime import datetime, timedelta
from typing import List, Optional
import logging

from .base_news_source import BaseNewsSource
from .models import NewsArticle, NewsFetchResult, SentimentLabel

logger = logging.getLogger(__name__)


class AlphaVantageNewsClient(BaseNewsSource):
    """Client for Alpha Vantage News API."""

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str, rate_limit_per_minute: int = 5):
        """Initialize Alpha Vantage News client.

        Args:
            api_key: Alpha Vantage API key
            rate_limit_per_minute: Max requests per minute
        """
        super().__init__(api_key, rate_limit_per_minute)
        if not api_key:
            raise ValueError("Alpha Vantage API key is required")

    @property
    def source_name(self) -> str:
        return "AlphaVantage"

    async def fetch_articles(
        self,
        symbols: List[str],
        lookback_hours: int = 24,
        max_articles_per_symbol: int = 10
    ) -> NewsFetchResult:
        """Fetch news articles for given symbols."""
        all_articles = []
        errors = []

        # Alpha Vantage accepts comma-separated tickers
        # Process in batches to respect rate limits
        batch_size = 5
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]

            try:
                await self._respect_rate_limit()

                # Convert crypto symbols to AV format
                tickers = [self._format_ticker(s) for s in batch]
                tickers_str = ",".join(tickers)

                articles = await self._fetch_news_sentiment(
                    tickers=tickers_str,
                    limit=max_articles_per_symbol * len(batch)
                )

                all_articles.extend(articles)

            except Exception as e:
                logger.error(f"Error fetching news for {batch}: {e}")
                errors.append(f"{batch}: {str(e)}")

        return NewsFetchResult(
            articles=all_articles,
            source=self.source_name,
            symbols_requested=symbols,
            success=len(errors) == 0,
            error_message="; ".join(errors) if errors else None
        )

    async def fetch_market_news(
        self,
        category: str = "general",
        lookback_hours: int = 24,
        max_articles: int = 20
    ) -> NewsFetchResult:
        """Fetch general market news."""
        try:
            await self._respect_rate_limit()

            # Map category to AV topics
            topic_map = {
                "general": "financial_markets",
                "technology": "technology",
                "crypto": "blockchain",
                "earnings": "earnings",
                "ipo": "ipo",
                "mergers": "mergers_and_acquisitions"
            }
            topics = topic_map.get(category, "financial_markets")

            articles = await self._fetch_news_sentiment(
                topics=topics,
                limit=max_articles
            )

            return NewsFetchResult(
                articles=articles,
                source=self.source_name,
                symbols_requested=[],
                success=True
            )

        except Exception as e:
            logger.error(f"Error fetching market news: {e}")
            return NewsFetchResult(
                articles=[],
                source=self.source_name,
                symbols_requested=[],
                success=False,
                error_message=str(e)
            )

    async def _fetch_news_sentiment(
        self,
        tickers: Optional[str] = None,
        topics: Optional[str] = None,
        limit: int = 50
    ) -> List[NewsArticle]:
        """Fetch news with sentiment from Alpha Vantage."""
        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": self.api_key,
            "limit": min(limit, 200)
        }

        if tickers:
            params["tickers"] = tickers
        if topics:
            params["topics"] = topics

        url = f"{self.BASE_URL}?" + "&".join(f"{k}={v}" for k, v in params.items())

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    text = await response.text()
                    raise Exception(f"Alpha Vantage error {response.status}: {text}")

                data = await response.json()

                # Check for error messages
                if "Error Message" in data:
                    raise Exception(f"Alpha Vantage error: {data['Error Message']}")
                if "Note" in data:
                    # Rate limit warning
                    logger.warning(f"Alpha Vantage: {data['Note']}")

                return self._parse_articles(data.get("feed", []))

    def _parse_articles(self, raw_articles: list) -> List[NewsArticle]:
        """Parse Alpha Vantage news response."""
        articles = []

        for raw in raw_articles:
            try:
                # Parse time
                time_str = raw.get("time_published", "")
                if time_str:
                    # Format: 20231215T120000
                    published_at = datetime.strptime(time_str[:15], "%Y%m%dT%H%M%S")
                else:
                    published_at = datetime.now()

                # Extract symbols mentioned
                ticker_sentiments = raw.get("ticker_sentiment", [])
                symbols = [ts.get("ticker", "") for ts in ticker_sentiments]
                symbols = [s for s in symbols if s]  # Filter empty

                # Get overall sentiment (Alpha Vantage provides this!)
                overall_sentiment = raw.get("overall_sentiment_score", 0)
                sentiment_label = self._map_sentiment_label(
                    raw.get("overall_sentiment_label", "Neutral")
                )

                article = NewsArticle(
                    id=f"alphavantage_{hash(raw.get('url', ''))}",
                    source=raw.get("source", "Unknown"),
                    title=raw.get("title", ""),
                    summary=raw.get("summary", ""),
                    url=raw.get("url", ""),
                    published_at=published_at,
                    symbols=symbols,
                    # Alpha Vantage provides sentiment!
                    sentiment_score=float(overall_sentiment),
                    sentiment_label=sentiment_label,
                    is_processed=True  # Already has sentiment
                )
                articles.append(article)

            except Exception as e:
                logger.warning(f"Error parsing article: {e}")
                continue

        return articles

    def _format_ticker(self, symbol: str) -> str:
        """Format symbol for Alpha Vantage API."""
        # Crypto needs CRYPTO: prefix
        crypto_symbols = {"BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOT", "MATIC", "LTC", "LINK"}
        if symbol.upper() in crypto_symbols:
            return f"CRYPTO:{symbol.upper()}"
        return symbol.upper()

    def _map_sentiment_label(self, av_label: str) -> SentimentLabel:
        """Map Alpha Vantage sentiment label to our enum."""
        mapping = {
            "Bullish": SentimentLabel.POSITIVE,
            "Somewhat-Bullish": SentimentLabel.POSITIVE,
            "Neutral": SentimentLabel.NEUTRAL,
            "Somewhat-Bearish": SentimentLabel.NEGATIVE,
            "Bearish": SentimentLabel.NEGATIVE,
        }
        return mapping.get(av_label, SentimentLabel.NEUTRAL)
```

**Acceptance Criteria**:
- [ ] Can fetch news for equities
- [ ] Can fetch news for crypto (with CRYPTO: prefix)
- [ ] Sentiment scores are extracted from response
- [ ] Rate limiting works
- [ ] Unit tests pass

---

## Task 2.1.4: Implement News Aggregator

**Context**:
We need to combine results from multiple news sources, deduplicate, and prioritize.

**Objective**:
Create a news aggregator that orchestrates multiple sources.

**Files to Modify**:
- `trading_system/data_pipeline/sources/news/news_aggregator.py`

**Requirements**:

1. Implement `NewsAggregator`:
```python
from datetime import datetime
from typing import List, Dict, Optional
import hashlib
import logging

from .base_news_source import BaseNewsSource
from .models import NewsArticle, NewsFetchResult
from .newsapi_client import NewsAPIClient
from .alpha_vantage_news import AlphaVantageNewsClient

logger = logging.getLogger(__name__)


class NewsAggregator:
    """Aggregate news from multiple sources."""

    def __init__(
        self,
        newsapi_key: Optional[str] = None,
        alpha_vantage_key: Optional[str] = None,
        enable_caching: bool = True,
        cache_ttl_minutes: int = 30
    ):
        """Initialize news aggregator.

        Args:
            newsapi_key: NewsAPI.org API key
            alpha_vantage_key: Alpha Vantage API key
            enable_caching: Whether to cache results
            cache_ttl_minutes: Cache TTL in minutes
        """
        self.sources: List[BaseNewsSource] = []
        self._cache: Dict[str, tuple[datetime, List[NewsArticle]]] = {}
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
        max_articles_per_symbol: int = 10
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
                    max_articles_per_symbol=max_articles_per_symbol
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
        categories: List[str] = None,
        max_articles: int = 30
    ) -> List[NewsArticle]:
        """Fetch general market news.

        Args:
            categories: News categories to fetch
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
                        max_articles=max_articles // len(categories)
                    )
                    if result.success:
                        all_articles.extend(result.articles)
                except Exception as e:
                    logger.error(f"Error fetching {category} from {source.source_name}: {e}")

        deduplicated = self._deduplicate_articles(all_articles)
        deduplicated.sort(key=lambda a: a.published_at, reverse=True)

        return deduplicated[:max_articles]

    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on title similarity."""
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
        """Create cache key."""
        symbols_str = ",".join(sorted(symbols))
        return f"{symbols_str}_{lookback_hours}"

    def _get_cached(self, key: str) -> Optional[List[NewsArticle]]:
        """Get cached result if not expired."""
        if not self.enable_caching:
            return None

        if key not in self._cache:
            return None

        cached_time, articles = self._cache[key]
        age_minutes = (datetime.now() - cached_time).seconds / 60

        if age_minutes > self.cache_ttl_minutes:
            del self._cache[key]
            return None

        return articles

    def clear_cache(self):
        """Clear the article cache."""
        self._cache.clear()
```

**Acceptance Criteria**:
- [ ] Aggregates from multiple sources
- [ ] Deduplicates articles with same title
- [ ] Caching works correctly
- [ ] Handles source failures gracefully
- [ ] Unit tests pass

---

## Task 2.2.1: Create Sentiment Analysis Module Structure

**Context**:
We need to analyze news sentiment to enhance trading signals.

**Objective**:
Create the sentiment analysis module structure.

**Files to Create**:
```
trading_system/research/
├── __init__.py
├── config.py
├── news_analyzer.py              # Main orchestrator (stub)
├── sentiment/
│   ├── __init__.py
│   ├── base_analyzer.py
│   ├── vader_analyzer.py         # Stub
│   ├── financial_lexicon.py      # Financial-specific words
│   └── sentiment_aggregator.py   # Stub
├── entity_extraction/
│   ├── __init__.py
│   └── ticker_extractor.py       # Stub
└── relevance/
    ├── __init__.py
    └── relevance_scorer.py       # Stub
```

**Requirements**:

1. Create `config.py`:
```python
from pydantic import BaseModel
from typing import Optional


class SentimentConfig(BaseModel):
    """Sentiment analysis configuration."""
    use_vader: bool = True
    use_finbert: bool = False  # Phase 4 - requires GPU
    vader_threshold_positive: float = 0.05
    vader_threshold_negative: float = -0.05
    min_confidence: float = 0.5


class RelevanceConfig(BaseModel):
    """Relevance scoring configuration."""
    min_relevance_score: float = 0.3
    title_weight: float = 0.6
    summary_weight: float = 0.4


class ResearchConfig(BaseModel):
    """Overall research module configuration."""
    sentiment: SentimentConfig = SentimentConfig()
    relevance: RelevanceConfig = RelevanceConfig()
    max_articles_per_symbol: int = 10
    lookback_hours: int = 48
```

2. Create `sentiment/base_analyzer.py`:
```python
from abc import ABC, abstractmethod
from typing import Tuple

from trading_system.data_pipeline.sources.news.models import NewsArticle, SentimentLabel


class BaseSentimentAnalyzer(ABC):
    """Abstract base class for sentiment analyzers."""

    @abstractmethod
    def analyze(self, text: str) -> Tuple[float, SentimentLabel, float]:
        """Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (score, label, confidence)
            - score: -1.0 to +1.0
            - label: SentimentLabel enum
            - confidence: 0.0 to 1.0
        """
        pass

    def analyze_article(self, article: NewsArticle) -> NewsArticle:
        """Analyze sentiment of a news article.

        Args:
            article: NewsArticle to analyze

        Returns:
            Article with sentiment fields populated
        """
        # Combine title and summary for analysis
        text = f"{article.title}. {article.summary or ''}"

        score, label, confidence = self.analyze(text)

        article.sentiment_score = score
        article.sentiment_label = label
        article.sentiment_confidence = confidence
        article.is_processed = True

        return article
```

**Acceptance Criteria**:
- [ ] All files created with proper structure
- [ ] Configuration models complete
- [ ] Base class defined
- [ ] Imports work correctly

---

## Task 2.2.2: Implement VADER Sentiment Analyzer with Financial Lexicon

**Context**:
VADER is a rule-based sentiment analyzer. We enhance it with financial-specific terms.

**Objective**:
Implement VADER analyzer with custom financial lexicon.

**Files to Modify**:
- `trading_system/research/sentiment/financial_lexicon.py`
- `trading_system/research/sentiment/vader_analyzer.py`

**Requirements**:

1. Create `financial_lexicon.py`:
```python
"""Financial-specific sentiment lexicon for VADER enhancement."""

# Positive financial terms (score: 0.5 to 3.0)
POSITIVE_TERMS = {
    # Strong positive (2.0 - 3.0)
    "bullish": 2.5,
    "surge": 2.5,
    "soar": 2.5,
    "skyrocket": 3.0,
    "rally": 2.0,
    "boom": 2.5,
    "breakout": 2.0,
    "all-time high": 2.5,
    "record high": 2.5,
    "outperform": 2.0,
    "beat expectations": 2.5,
    "beat estimates": 2.5,
    "exceeds expectations": 2.5,

    # Moderate positive (1.0 - 2.0)
    "upgrade": 1.8,
    "buy rating": 1.5,
    "strong buy": 2.0,
    "growth": 1.2,
    "profit": 1.5,
    "gains": 1.2,
    "recovery": 1.5,
    "rebound": 1.5,
    "momentum": 1.2,
    "uptrend": 1.5,
    "positive": 1.0,
    "optimistic": 1.2,
    "confident": 1.0,
    "expansion": 1.2,
    "innovation": 1.0,

    # Mild positive (0.5 - 1.0)
    "stable": 0.5,
    "steady": 0.5,
    "solid": 0.8,
    "healthy": 0.8,
    "improving": 0.8,
}

# Negative financial terms (score: -0.5 to -3.0)
NEGATIVE_TERMS = {
    # Strong negative (-2.0 to -3.0)
    "bearish": -2.5,
    "crash": -3.0,
    "collapse": -3.0,
    "plunge": -2.5,
    "plummet": -2.5,
    "tank": -2.5,
    "meltdown": -3.0,
    "bankruptcy": -3.0,
    "default": -2.5,
    "fraud": -3.0,
    "scandal": -2.5,
    "miss expectations": -2.5,
    "miss estimates": -2.5,
    "below expectations": -2.0,

    # Moderate negative (-1.0 to -2.0)
    "downgrade": -1.8,
    "sell rating": -1.5,
    "strong sell": -2.0,
    "loss": -1.5,
    "losses": -1.5,
    "decline": -1.5,
    "drop": -1.2,
    "fall": -1.2,
    "slump": -1.5,
    "selloff": -1.8,
    "sell-off": -1.8,
    "downtrend": -1.5,
    "recession": -2.0,
    "layoffs": -1.5,
    "lawsuit": -1.5,
    "investigation": -1.2,
    "probe": -1.0,

    # Mild negative (-0.5 to -1.0)
    "concern": -0.8,
    "concerns": -0.8,
    "uncertainty": -0.8,
    "volatile": -0.5,
    "volatility": -0.5,
    "risk": -0.5,
    "risky": -0.8,
    "weak": -0.8,
    "slowdown": -0.8,
}

# Intensifiers specific to finance
INTENSIFIERS = {
    "significantly": 1.5,
    "dramatically": 1.8,
    "sharply": 1.5,
    "substantially": 1.3,
    "massively": 1.8,
    "slightly": 0.5,
    "marginally": 0.5,
    "modestly": 0.7,
}

# Negation handling (these flip sentiment)
NEGATIONS = {
    "not",
    "no",
    "never",
    "neither",
    "nobody",
    "nothing",
    "nowhere",
    "hardly",
    "barely",
    "scarcely",
    "doesn't",
    "isn't",
    "wasn't",
    "shouldn't",
    "wouldn't",
    "couldn't",
    "won't",
    "can't",
    "don't",
}


def get_financial_lexicon() -> dict:
    """Get combined financial lexicon for VADER.

    Returns:
        Dictionary mapping terms to sentiment scores
    """
    lexicon = {}
    lexicon.update(POSITIVE_TERMS)
    lexicon.update(NEGATIVE_TERMS)
    return lexicon
```

2. Implement `vader_analyzer.py`:
```python
from typing import Tuple
import logging

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from trading_system.data_pipeline.sources.news.models import SentimentLabel
from .base_analyzer import BaseSentimentAnalyzer
from .financial_lexicon import get_financial_lexicon, INTENSIFIERS

logger = logging.getLogger(__name__)


class VADERSentimentAnalyzer(BaseSentimentAnalyzer):
    """VADER-based sentiment analyzer with financial lexicon."""

    def __init__(
        self,
        positive_threshold: float = 0.05,
        negative_threshold: float = -0.05
    ):
        """Initialize VADER analyzer.

        Args:
            positive_threshold: Compound score threshold for positive
            negative_threshold: Compound score threshold for negative
        """
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold

        # Initialize VADER
        self.analyzer = SentimentIntensityAnalyzer()

        # Add financial lexicon
        self._add_financial_lexicon()

        logger.info("VADER analyzer initialized with financial lexicon")

    def _add_financial_lexicon(self):
        """Add financial-specific terms to VADER lexicon."""
        financial_terms = get_financial_lexicon()

        for term, score in financial_terms.items():
            self.analyzer.lexicon[term] = score

        # Also add intensifiers
        for term, multiplier in INTENSIFIERS.items():
            if term not in self.analyzer.lexicon:
                # VADER uses these differently, but we can influence
                pass

        logger.info(f"Added {len(financial_terms)} financial terms to lexicon")

    def analyze(self, text: str) -> Tuple[float, SentimentLabel, float]:
        """Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (score, label, confidence)
        """
        if not text or not text.strip():
            return 0.0, SentimentLabel.NEUTRAL, 0.0

        # Get VADER scores
        scores = self.analyzer.polarity_scores(text)

        compound = scores["compound"]  # -1 to +1
        pos = scores["pos"]
        neg = scores["neg"]
        neu = scores["neu"]

        # Determine label
        if compound >= self.positive_threshold:
            if compound >= 0.5:
                label = SentimentLabel.VERY_POSITIVE
            else:
                label = SentimentLabel.POSITIVE
        elif compound <= self.negative_threshold:
            if compound <= -0.5:
                label = SentimentLabel.VERY_NEGATIVE
            else:
                label = SentimentLabel.NEGATIVE
        else:
            label = SentimentLabel.NEUTRAL

        # Calculate confidence based on how extreme the sentiment is
        # and how much of the text is sentiment-bearing (non-neutral)
        sentiment_portion = pos + neg
        extremity = abs(compound)
        confidence = min((sentiment_portion + extremity) / 2, 1.0)

        return compound, label, confidence

    def analyze_batch(self, texts: list) -> list:
        """Analyze multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of (score, label, confidence) tuples
        """
        return [self.analyze(text) for text in texts]
```

3. Add to `pyproject.toml` dependencies:
```toml
# Add to [project.optional-dependencies]
research = [
    "vaderSentiment>=3.3.0,<4.0.0",
    "nltk>=3.8.0,<4.0.0",  # For additional NLP if needed
]
```

**Acceptance Criteria**:
- [ ] Financial lexicon has 50+ terms
- [ ] VADER analyzer initialized correctly
- [ ] "Apple stock surges on earnings beat" → positive
- [ ] "Market crashes on recession fears" → negative
- [ ] Confidence scores are reasonable
- [ ] Unit tests pass

**Test Examples**:
```python
def test_positive_sentiment():
    analyzer = VADERSentimentAnalyzer()
    score, label, conf = analyzer.analyze("Apple stock surges 10% on record earnings")
    assert label in [SentimentLabel.POSITIVE, SentimentLabel.VERY_POSITIVE]
    assert score > 0.3

def test_negative_sentiment():
    analyzer = VADERSentimentAnalyzer()
    score, label, conf = analyzer.analyze("Tech stocks plunge amid recession fears")
    assert label in [SentimentLabel.NEGATIVE, SentimentLabel.VERY_NEGATIVE]
    assert score < -0.3
```

---

## Task 2.2.3: Implement Ticker Extractor

**Context**:
We need to identify which stocks/crypto are mentioned in news articles.

**Objective**:
Create a ticker extraction utility.

**Files to Modify**:
- `trading_system/research/entity_extraction/ticker_extractor.py`

**Requirements**:

1. Implement `TickerExtractor`:
```python
import re
from typing import List, Set, Tuple
import logging

logger = logging.getLogger(__name__)


# Known tickers and their variations
TICKER_ALIASES = {
    # Equities
    "AAPL": ["apple", "iphone", "ipad", "mac", "tim cook"],
    "MSFT": ["microsoft", "windows", "azure", "satya nadella"],
    "GOOGL": ["google", "alphabet", "youtube", "android", "sundar pichai"],
    "AMZN": ["amazon", "aws", "prime", "jeff bezos", "andy jassy"],
    "NVDA": ["nvidia", "geforce", "jensen huang"],
    "META": ["meta", "facebook", "instagram", "whatsapp", "mark zuckerberg"],
    "TSLA": ["tesla", "elon musk", "model s", "model 3", "model y"],
    "JPM": ["jpmorgan", "jp morgan", "jamie dimon"],
    "V": ["visa"],
    "JNJ": ["johnson & johnson", "johnson and johnson"],
    # ... add more

    # Crypto
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth", "ether"],
    "BNB": ["binance coin", "bnb"],
    "XRP": ["ripple", "xrp"],
    "ADA": ["cardano", "ada"],
    "SOL": ["solana", "sol"],
    "DOT": ["polkadot", "dot"],
    "MATIC": ["polygon", "matic"],
    "LTC": ["litecoin", "ltc"],
    "LINK": ["chainlink", "link"],
}

# Build reverse lookup
ALIAS_TO_TICKER = {}
for ticker, aliases in TICKER_ALIASES.items():
    for alias in aliases:
        ALIAS_TO_TICKER[alias.lower()] = ticker


class TickerExtractor:
    """Extract stock/crypto tickers from text."""

    def __init__(self, valid_tickers: Set[str] = None):
        """Initialize extractor.

        Args:
            valid_tickers: Set of valid tickers to recognize.
                          If None, uses built-in list.
        """
        self.valid_tickers = valid_tickers or set(TICKER_ALIASES.keys())

        # Pattern for explicit ticker mentions (e.g., $AAPL, AAPL:, (AAPL))
        self.ticker_pattern = re.compile(
            r'(?:\$([A-Z]{1,5})|'  # $AAPL
            r'\b([A-Z]{1,5})(?::|(?=\s+(?:stock|shares|price|falls|rises|surges|drops)))|'  # AAPL: or AAPL stock
            r'\(([A-Z]{1,5})\))'  # (AAPL)
        )

    def extract(self, text: str) -> List[str]:
        """Extract tickers from text.

        Args:
            text: Text to search

        Returns:
            List of found ticker symbols (deduplicated)
        """
        if not text:
            return []

        found_tickers = set()
        text_lower = text.lower()

        # 1. Find explicit ticker mentions
        for match in self.ticker_pattern.finditer(text):
            for group in match.groups():
                if group and group.upper() in self.valid_tickers:
                    found_tickers.add(group.upper())

        # 2. Find company name mentions
        for alias, ticker in ALIAS_TO_TICKER.items():
            if ticker in self.valid_tickers and alias in text_lower:
                # Verify it's a word boundary match
                pattern = r'\b' + re.escape(alias) + r'\b'
                if re.search(pattern, text_lower):
                    found_tickers.add(ticker)

        return list(found_tickers)

    def extract_with_context(self, text: str) -> List[Tuple[str, str]]:
        """Extract tickers with surrounding context.

        Args:
            text: Text to search

        Returns:
            List of (ticker, context) tuples
        """
        results = []
        tickers = self.extract(text)

        for ticker in tickers:
            # Find context around ticker mention
            context = self._get_context(text, ticker)
            results.append((ticker, context))

        return results

    def _get_context(self, text: str, ticker: str, window: int = 50) -> str:
        """Get text context around ticker mention.

        Args:
            text: Full text
            ticker: Ticker to find
            window: Characters before/after to include

        Returns:
            Context string
        """
        text_lower = text.lower()
        aliases = [ticker.lower()] + [a for a, t in ALIAS_TO_TICKER.items() if t == ticker]

        for alias in aliases:
            idx = text_lower.find(alias)
            if idx >= 0:
                start = max(0, idx - window)
                end = min(len(text), idx + len(alias) + window)
                context = text[start:end].strip()
                if start > 0:
                    context = "..." + context
                if end < len(text):
                    context = context + "..."
                return context

        return ""
```

**Acceptance Criteria**:
- [ ] Extracts $AAPL format
- [ ] Extracts "Apple" → AAPL
- [ ] Extracts "Bitcoin" → BTC
- [ ] Handles multiple tickers in same text
- [ ] Returns context around mentions
- [ ] Unit tests pass

---

## Task 2.2.4: Implement News Analyzer Orchestrator

**Context**:
We need a main orchestrator that ties together news fetching, sentiment analysis, and signal generation.

**Objective**:
Create the main NewsAnalyzer class.

**Files to Modify**:
- `trading_system/research/news_analyzer.py`

**Requirements**:

1. Implement `NewsAnalyzer`:
```python
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple
import logging

from trading_system.data_pipeline.sources.news import (
    NewsAggregator,
    NewsArticle,
    SentimentLabel
)
from .sentiment.vader_analyzer import VADERSentimentAnalyzer
from .entity_extraction.ticker_extractor import TickerExtractor
from .config import ResearchConfig

logger = logging.getLogger(__name__)


@dataclass
class SymbolNewsSummary:
    """Summary of news for a single symbol."""
    symbol: str
    article_count: int
    avg_sentiment: float
    sentiment_label: SentimentLabel
    positive_count: int
    negative_count: int
    neutral_count: int
    top_headlines: List[str]
    sentiment_trend: str  # 'improving', 'declining', 'stable'
    most_recent_article: Optional[datetime] = None


@dataclass
class NewsAnalysisResult:
    """Complete news analysis result."""
    analysis_date: date
    symbols_analyzed: List[str]
    symbol_summaries: Dict[str, SymbolNewsSummary]
    market_sentiment: float  # Overall market sentiment
    market_sentiment_label: SentimentLabel
    total_articles: int
    articles: List[NewsArticle]  # All processed articles


class NewsAnalyzer:
    """Main news analysis orchestrator."""

    def __init__(
        self,
        config: ResearchConfig,
        newsapi_key: Optional[str] = None,
        alpha_vantage_key: Optional[str] = None
    ):
        """Initialize news analyzer.

        Args:
            config: Research configuration
            newsapi_key: NewsAPI.org API key
            alpha_vantage_key: Alpha Vantage API key
        """
        self.config = config

        # Initialize components
        self.news_aggregator = NewsAggregator(
            newsapi_key=newsapi_key,
            alpha_vantage_key=alpha_vantage_key
        )
        self.sentiment_analyzer = VADERSentimentAnalyzer(
            positive_threshold=config.sentiment.vader_threshold_positive,
            negative_threshold=config.sentiment.vader_threshold_negative
        )
        self.ticker_extractor = TickerExtractor()

        logger.info("NewsAnalyzer initialized")

    async def analyze_symbols(
        self,
        symbols: List[str],
        lookback_hours: int = None
    ) -> NewsAnalysisResult:
        """Analyze news for given symbols.

        Args:
            symbols: List of symbols to analyze
            lookback_hours: How far back to search (default from config)

        Returns:
            Complete analysis result
        """
        lookback = lookback_hours or self.config.lookback_hours

        logger.info(f"Analyzing news for {len(symbols)} symbols, {lookback}h lookback")

        # 1. Fetch news
        articles = await self.news_aggregator.fetch_news_for_symbols(
            symbols=symbols,
            lookback_hours=lookback,
            max_articles_per_symbol=self.config.max_articles_per_symbol
        )

        logger.info(f"Fetched {len(articles)} articles")

        # 2. Process each article
        processed_articles = []
        for article in articles:
            processed = self._process_article(article, symbols)
            processed_articles.append(processed)

        # 3. Generate summaries per symbol
        symbol_summaries = {}
        for symbol in symbols:
            summary = self._generate_symbol_summary(symbol, processed_articles)
            symbol_summaries[symbol] = summary

        # 4. Calculate overall market sentiment
        market_sentiment, market_label = self._calculate_market_sentiment(processed_articles)

        return NewsAnalysisResult(
            analysis_date=date.today(),
            symbols_analyzed=symbols,
            symbol_summaries=symbol_summaries,
            market_sentiment=market_sentiment,
            market_sentiment_label=market_label,
            total_articles=len(processed_articles),
            articles=processed_articles
        )

    def _process_article(
        self,
        article: NewsArticle,
        target_symbols: List[str]
    ) -> NewsArticle:
        """Process a single article.

        Args:
            article: Article to process
            target_symbols: Symbols we're interested in

        Returns:
            Processed article with sentiment and symbols
        """
        # Skip if already processed (e.g., Alpha Vantage provides sentiment)
        if article.is_processed and article.sentiment_score is not None:
            # Still need to extract symbols if not done
            if not article.symbols:
                article.symbols = self.ticker_extractor.extract(
                    f"{article.title} {article.summary or ''}"
                )
            return article

        # Extract symbols mentioned
        text = f"{article.title} {article.summary or ''}"
        found_symbols = self.ticker_extractor.extract(text)

        # Only keep symbols we care about
        article.symbols = [s for s in found_symbols if s in target_symbols]

        # Analyze sentiment
        article = self.sentiment_analyzer.analyze_article(article)

        return article

    def _generate_symbol_summary(
        self,
        symbol: str,
        articles: List[NewsArticle]
    ) -> SymbolNewsSummary:
        """Generate summary for a symbol.

        Args:
            symbol: Symbol to summarize
            articles: All processed articles

        Returns:
            Summary for the symbol
        """
        # Filter to articles mentioning this symbol
        symbol_articles = [a for a in articles if symbol in a.symbols]

        if not symbol_articles:
            return SymbolNewsSummary(
                symbol=symbol,
                article_count=0,
                avg_sentiment=0.0,
                sentiment_label=SentimentLabel.NEUTRAL,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                top_headlines=[],
                sentiment_trend="stable"
            )

        # Calculate sentiment stats
        sentiments = [a.sentiment_score for a in symbol_articles if a.sentiment_score is not None]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

        # Count by label
        positive_count = sum(1 for a in symbol_articles
                            if a.sentiment_label in [SentimentLabel.POSITIVE, SentimentLabel.VERY_POSITIVE])
        negative_count = sum(1 for a in symbol_articles
                            if a.sentiment_label in [SentimentLabel.NEGATIVE, SentimentLabel.VERY_NEGATIVE])
        neutral_count = len(symbol_articles) - positive_count - negative_count

        # Determine overall label
        if avg_sentiment >= 0.3:
            sentiment_label = SentimentLabel.VERY_POSITIVE
        elif avg_sentiment >= 0.05:
            sentiment_label = SentimentLabel.POSITIVE
        elif avg_sentiment <= -0.3:
            sentiment_label = SentimentLabel.VERY_NEGATIVE
        elif avg_sentiment <= -0.05:
            sentiment_label = SentimentLabel.NEGATIVE
        else:
            sentiment_label = SentimentLabel.NEUTRAL

        # Get top headlines (most recent, highest absolute sentiment)
        sorted_articles = sorted(
            symbol_articles,
            key=lambda a: (abs(a.sentiment_score or 0), a.published_at),
            reverse=True
        )
        top_headlines = [a.title for a in sorted_articles[:3]]

        # Calculate trend (compare first half vs second half)
        mid = len(symbol_articles) // 2
        if mid > 0:
            old_sentiment = sum(a.sentiment_score or 0 for a in symbol_articles[mid:]) / (len(symbol_articles) - mid)
            new_sentiment = sum(a.sentiment_score or 0 for a in symbol_articles[:mid]) / mid
            if new_sentiment > old_sentiment + 0.1:
                trend = "improving"
            elif new_sentiment < old_sentiment - 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return SymbolNewsSummary(
            symbol=symbol,
            article_count=len(symbol_articles),
            avg_sentiment=avg_sentiment,
            sentiment_label=sentiment_label,
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            top_headlines=top_headlines,
            sentiment_trend=trend,
            most_recent_article=symbol_articles[0].published_at if symbol_articles else None
        )

    def _calculate_market_sentiment(
        self,
        articles: List[NewsArticle]
    ) -> Tuple[float, SentimentLabel]:
        """Calculate overall market sentiment.

        Args:
            articles: All processed articles

        Returns:
            Tuple of (sentiment score, label)
        """
        sentiments = [a.sentiment_score for a in articles if a.sentiment_score is not None]

        if not sentiments:
            return 0.0, SentimentLabel.NEUTRAL

        avg = sum(sentiments) / len(sentiments)

        if avg >= 0.2:
            label = SentimentLabel.POSITIVE
        elif avg <= -0.2:
            label = SentimentLabel.NEGATIVE
        else:
            label = SentimentLabel.NEUTRAL

        return avg, label

    def get_news_score_for_signal(
        self,
        symbol: str,
        analysis: NewsAnalysisResult
    ) -> Tuple[float, str]:
        """Get news score for use in signal scoring.

        Args:
            symbol: Symbol to get score for
            analysis: News analysis result

        Returns:
            Tuple of (score 0-10, reasoning string)
        """
        summary = analysis.symbol_summaries.get(symbol)

        if not summary or summary.article_count == 0:
            return 5.0, "No recent news"  # Neutral score

        # Convert sentiment (-1 to +1) to score (0 to 10)
        # sentiment of 0 = 5, sentiment of +1 = 10, sentiment of -1 = 0
        base_score = (summary.avg_sentiment + 1) * 5

        # Boost/penalize based on article count (more coverage = more confidence)
        coverage_mult = min(summary.article_count / 5, 1.5)  # Max 1.5x for 5+ articles
        adjusted_score = base_score * (0.7 + 0.3 * coverage_mult)

        # Boost for improving trend
        if summary.sentiment_trend == "improving":
            adjusted_score = min(adjusted_score * 1.1, 10)
        elif summary.sentiment_trend == "declining":
            adjusted_score = adjusted_score * 0.9

        # Generate reasoning
        reasoning = self._generate_reasoning(summary)

        return round(adjusted_score, 1), reasoning

    def _generate_reasoning(self, summary: SymbolNewsSummary) -> str:
        """Generate human-readable reasoning."""
        parts = []

        # Sentiment
        if summary.sentiment_label == SentimentLabel.VERY_POSITIVE:
            parts.append("Very positive news sentiment")
        elif summary.sentiment_label == SentimentLabel.POSITIVE:
            parts.append("Positive news sentiment")
        elif summary.sentiment_label == SentimentLabel.VERY_NEGATIVE:
            parts.append("Very negative news sentiment")
        elif summary.sentiment_label == SentimentLabel.NEGATIVE:
            parts.append("Negative news sentiment")
        else:
            parts.append("Neutral news sentiment")

        # Coverage
        parts.append(f"({summary.article_count} articles)")

        # Trend
        if summary.sentiment_trend == "improving":
            parts.append("with improving trend")
        elif summary.sentiment_trend == "declining":
            parts.append("with declining trend")

        return " ".join(parts)
```

**Acceptance Criteria**:
- [ ] Fetches and processes articles for multiple symbols
- [ ] Generates per-symbol summaries
- [ ] Calculates market sentiment
- [ ] Provides scores for signal enhancement
- [ ] Handles empty results gracefully
- [ ] Integration tests pass

---

## Task 2.3.1: Integrate News Scores into Signal Generator

**Context**:
The signal generator needs to incorporate news sentiment into its scoring.

**Objective**:
Update the signal generator to use news scores.

**Files to Modify**:
- `trading_system/signals/live_signal_generator.py`
- `trading_system/signals/config.py`

**Requirements**:

1. Update `SignalConfig`:
```python
class SignalConfig(BaseModel):
    max_recommendations: int = 5
    min_conviction: str = "MEDIUM"

    # Scoring weights
    technical_weight: float = 0.6
    news_weight: float = 0.4

    # News integration
    news_enabled: bool = True
    news_lookback_hours: int = 48
    min_news_score_for_boost: float = 7.0  # Score above this boosts conviction
    max_news_score_for_penalty: float = 3.0  # Score below this penalizes
```

2. Update `LiveSignalGenerator` to use news:
```python
class LiveSignalGenerator:
    def __init__(
        self,
        strategies: List[StrategyInterface],
        portfolio_config: PortfolioConfig,
        data_fetcher: LiveDataFetcher,
        news_analyzer: Optional[NewsAnalyzer] = None,  # NEW
        signal_config: SignalConfig = None
    ):
        # ... existing init ...
        self.news_analyzer = news_analyzer
        self.signal_config = signal_config or SignalConfig()

    async def generate_daily_signals(
        self,
        current_date: date
    ) -> List[Recommendation]:
        # ... existing data fetching and signal generation ...

        # NEW: Fetch news analysis if enabled
        news_analysis = None
        if self.news_analyzer and self.signal_config.news_enabled:
            symbols = list(ohlcv_data.keys())
            news_analysis = await self.news_analyzer.analyze_symbols(
                symbols=symbols,
                lookback_hours=self.signal_config.news_lookback_hours
            )

        # Score signals (now with news)
        scored_signals = self._score_signals(all_signals, news_analysis)

        # ... rest of pipeline ...

    def _score_signals(
        self,
        signals: List[Signal],
        news_analysis: Optional[NewsAnalysisResult]
    ) -> List[Tuple[Signal, float, Dict]]:
        """Score signals with technical and news components."""
        scored = []

        for signal in signals:
            # Technical score (existing logic)
            technical_score = self._calculate_technical_score(signal)

            # News score
            if news_analysis:
                news_score, news_reasoning = self.news_analyzer.get_news_score_for_signal(
                    signal.symbol,
                    news_analysis
                )
            else:
                news_score = 5.0  # Neutral
                news_reasoning = "News analysis disabled"

            # Combined score
            combined = (
                technical_score * self.signal_config.technical_weight +
                news_score * self.signal_config.news_weight
            )

            metadata = {
                "technical_score": technical_score,
                "news_score": news_score,
                "news_reasoning": news_reasoning,
                "combined_score": combined
            }

            scored.append((signal, combined, metadata))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
```

3. Update `Recommendation` to include news data:
```python
@dataclass
class Recommendation:
    # ... existing fields ...

    # News fields
    news_score: Optional[float] = None
    news_reasoning: Optional[str] = None
    news_headlines: List[str] = field(default_factory=list)
    news_sentiment: Optional[str] = None  # 'positive', 'negative', 'neutral'
```

**Acceptance Criteria**:
- [ ] Signals are scored with both technical and news
- [ ] News-positive signals get higher scores
- [ ] News-negative signals get lower scores
- [ ] Recommendations include news metadata
- [ ] Works correctly when news is disabled
- [ ] Integration tests pass

---

## Task 2.3.2: Update Email Template with News Section

**Context**:
The daily email needs to include news digest and show news scores.

**Objective**:
Update the email template to display news information.

**Files to Modify**:
- `trading_system/output/email/templates/daily_signals.html`
- `trading_system/output/email/report_generator.py`

**Requirements**:

1. Add news section to email template:
```html
<!-- Add after recommendations section -->

═══════════════════════════════════════════════════════
                 📰 NEWS DIGEST
═══════════════════════════════════════════════════════

{% if news_analysis %}
<div class="market-sentiment">
    <h3>Market Sentiment: {{ news_analysis.market_sentiment_label.value | title }}</h3>
    <p>Based on {{ news_analysis.total_articles }} articles analyzed</p>
</div>

<!-- Positive News -->
{% if positive_news %}
<div class="news-section positive">
    <h4>🟢 Positive Sentiment</h4>
    <ul>
    {% for article in positive_news[:5] %}
        <li>
            <strong>{{ article.symbols | join(', ') }}</strong>:
            {{ article.title }}
            <span class="source">({{ article.source }})</span>
        </li>
    {% endfor %}
    </ul>
</div>
{% endif %}

<!-- Negative News -->
{% if negative_news %}
<div class="news-section negative">
    <h4>🔴 Negative Sentiment</h4>
    <ul>
    {% for article in negative_news[:5] %}
        <li>
            <strong>{{ article.symbols | join(', ') }}</strong>:
            {{ article.title }}
            <span class="source">({{ article.source }})</span>
        </li>
    {% endfor %}
    </ul>
</div>
{% endif %}

<!-- Notable Headlines -->
<div class="news-section notable">
    <h4>📌 Notable Headlines</h4>
    <ul>
    {% for symbol, summary in news_analysis.symbol_summaries.items() %}
        {% if summary.top_headlines %}
        <li>
            <strong>{{ symbol }}</strong>
            ({{ summary.sentiment_label.value | title }}):
            {{ summary.top_headlines[0] }}
        </li>
        {% endif %}
    {% endfor %}
    </ul>
</div>
{% else %}
<p>News analysis not available</p>
{% endif %}
```

2. Update each recommendation to show news score:
```html
<!-- In recommendation card -->
<div class="scores">
    <div class="score-item">
        <span class="label">Technical:</span>
        <span class="value">{{ rec.technical_score }}/10</span>
    </div>
    <div class="score-item">
        <span class="label">News:</span>
        <span class="value {% if rec.news_score >= 7 %}positive{% elif rec.news_score <= 3 %}negative{% endif %}">
            {{ rec.news_score }}/10
        </span>
    </div>
    <div class="score-item combined">
        <span class="label">Combined:</span>
        <span class="value">{{ rec.combined_score }}/10</span>
    </div>
</div>

{% if rec.news_reasoning %}
<div class="news-context">
    <span class="icon">📰</span> {{ rec.news_reasoning }}
</div>
{% endif %}

{% if rec.news_headlines %}
<div class="headlines">
    <strong>Recent News:</strong>
    <ul>
    {% for headline in rec.news_headlines[:2] %}
        <li>{{ headline }}</li>
    {% endfor %}
    </ul>
</div>
{% endif %}
```

3. Update `report_generator.py`:
```python
async def generate_daily_report(
    self,
    recommendations: List[Recommendation],
    market_summary: Dict[str, Any],
    news_analysis: Optional[NewsAnalysisResult],  # NEW
    date: date
) -> str:
    """Generate HTML email content."""

    # Separate news by sentiment
    positive_news = []
    negative_news = []

    if news_analysis:
        for article in news_analysis.articles:
            if article.sentiment_label in [SentimentLabel.POSITIVE, SentimentLabel.VERY_POSITIVE]:
                positive_news.append(article)
            elif article.sentiment_label in [SentimentLabel.NEGATIVE, SentimentLabel.VERY_NEGATIVE]:
                negative_news.append(article)

    return self.templates['daily_signals'].render(
        recommendations=recommendations,
        market=market_summary,
        news_analysis=news_analysis,
        positive_news=positive_news[:5],
        negative_news=negative_news[:5],
        date=date.strftime("%B %d, %Y"),
        generated_at=datetime.now().strftime("%I:%M %p ET")
    )
```

**Acceptance Criteria**:
- [ ] Email shows news digest section
- [ ] Recommendations show news scores
- [ ] Positive/negative news are separated
- [ ] Headlines are displayed for relevant symbols
- [ ] Email renders correctly
- [ ] Test email looks good

---

## Task 2.3.3: Update Daily Job to Include News

**Context**:
The daily cron job needs to incorporate news analysis.

**Objective**:
Update the daily signals job to include news.

**Files to Modify**:
- `trading_system/scheduler/jobs/daily_signals_job.py`

**Requirements**:

1. Update job to use news analyzer:
```python
async def daily_signals_job(asset_class: str):
    """Execute daily signal generation with news analysis."""
    logger.info(f"Starting daily signals job for {asset_class}")

    try:
        # Load configuration
        config = load_config()

        # Initialize components
        data_fetcher = LiveDataFetcher(config.data_pipeline)

        # Initialize news analyzer if configured
        news_analyzer = None
        if config.research.enabled:
            news_analyzer = NewsAnalyzer(
                config=config.research,
                newsapi_key=config.api_keys.newsapi,
                alpha_vantage_key=config.api_keys.alpha_vantage
            )

        signal_generator = LiveSignalGenerator(
            strategies=load_strategies(config),
            portfolio_config=config.portfolio,
            data_fetcher=data_fetcher,
            news_analyzer=news_analyzer,
            signal_config=config.signals
        )

        email_service = EmailService(config.email)

        # Fetch data
        symbols = config.universe[asset_class]
        ohlcv_data = await data_fetcher.fetch_daily_data(
            symbols=symbols,
            asset_class=asset_class,
            lookback_days=252
        )

        # Generate signals (now includes news)
        recommendations = await signal_generator.generate_daily_signals(
            current_date=date.today()
        )

        # Get news analysis for email
        news_analysis = None
        if news_analyzer:
            news_analysis = await news_analyzer.analyze_symbols(
                symbols=symbols,
                lookback_hours=config.research.lookback_hours
            )

        # Send email
        await email_service.send_daily_report(
            recommendations=recommendations,
            market_summary=get_market_summary(ohlcv_data),
            news_analysis=news_analysis,  # NEW
            date=date.today()
        )

        logger.info(
            f"Daily signals job completed: "
            f"{len(recommendations)} recommendations, "
            f"{news_analysis.total_articles if news_analysis else 0} articles analyzed"
        )

    except Exception as e:
        logger.error(f"Daily signals job failed: {e}")
        await send_error_alert(e)
        raise
```

**Acceptance Criteria**:
- [ ] Job fetches and analyzes news
- [ ] News is integrated into signal scoring
- [ ] Email includes news section
- [ ] Job handles news API failures gracefully
- [ ] Integration tests pass

---

## Task 2.4.1: Add Configuration and Environment Variables

**Context**:
Need to configure news API keys and settings.

**Objective**:
Update configuration to support news integration.

**Files to Modify**:
- `config/trading_config.yaml`
- `.env.example`
- `trading_system/configs/run_config.py`

**Requirements**:

1. Update `.env.example`:
```bash
# API Keys - Data
POLYGON_API_KEY=your_polygon_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here

# API Keys - News
NEWSAPI_API_KEY=your_newsapi_key_here

# Email (SendGrid)
SENDGRID_API_KEY=your_sendgrid_key_here
EMAIL_RECIPIENTS=your@email.com

# Optional
LOG_LEVEL=INFO
```

2. Update config schema:
```python
class ResearchConfig(BaseModel):
    """News and research configuration."""
    enabled: bool = True
    newsapi_key: Optional[str] = None
    alpha_vantage_key: Optional[str] = None
    lookback_hours: int = 48
    max_articles_per_symbol: int = 10

    sentiment: SentimentConfig = SentimentConfig()


class TradingConfig(BaseModel):
    """Main trading system configuration."""
    # ... existing fields ...
    research: ResearchConfig = ResearchConfig()
```

3. Update `trading_config.yaml`:
```yaml
# Research & News
research:
  enabled: true
  lookback_hours: 48
  max_articles_per_symbol: 10

  sentiment:
    use_vader: true
    vader_threshold_positive: 0.05
    vader_threshold_negative: -0.05

# Signal Generation
signals:
  max_recommendations: 5
  min_conviction: "MEDIUM"
  technical_weight: 0.6
  news_weight: 0.4
  news_enabled: true
```

**Acceptance Criteria**:
- [ ] All config fields documented
- [ ] Environment variables work
- [ ] Config validation catches errors
- [ ] Default values are sensible

---

## Task 2.4.2: Write Integration Tests

**Context**:
Need comprehensive tests for the news integration.

**Objective**:
Create integration tests for Phase 2.

**Files to Create**:
- `tests/test_news_integration.py`

**Requirements**:

```python
import pytest
from datetime import date
from unittest.mock import AsyncMock, patch

from trading_system.research.news_analyzer import NewsAnalyzer
from trading_system.research.config import ResearchConfig
from trading_system.signals.live_signal_generator import LiveSignalGenerator


class TestNewsIntegration:
    """Integration tests for news analysis."""

    @pytest.fixture
    def mock_news_response(self):
        """Mock news API response."""
        return {
            "status": "ok",
            "articles": [
                {
                    "source": {"name": "Reuters"},
                    "title": "Apple stock surges on strong iPhone sales",
                    "description": "Apple Inc reported record quarterly revenue...",
                    "url": "https://example.com/1",
                    "publishedAt": "2024-01-15T10:00:00Z"
                },
                {
                    "source": {"name": "Bloomberg"},
                    "title": "Tech stocks rally amid Fed optimism",
                    "description": "Technology shares led the market higher...",
                    "url": "https://example.com/2",
                    "publishedAt": "2024-01-15T09:00:00Z"
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_news_analyzer_processes_articles(self, mock_news_response):
        """Test that news analyzer processes articles correctly."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value.status = 200
            mock_get.return_value.__aenter__.return_value.json = AsyncMock(
                return_value=mock_news_response
            )

            analyzer = NewsAnalyzer(
                config=ResearchConfig(),
                newsapi_key="test_key"
            )

            result = await analyzer.analyze_symbols(
                symbols=["AAPL"],
                lookback_hours=24
            )

            assert result.total_articles >= 1
            assert "AAPL" in result.symbol_summaries
            assert result.symbol_summaries["AAPL"].article_count > 0

    @pytest.mark.asyncio
    async def test_positive_news_boosts_signal_score(self):
        """Test that positive news increases signal score."""
        # ... test implementation

    @pytest.mark.asyncio
    async def test_negative_news_reduces_signal_score(self):
        """Test that negative news decreases signal score."""
        # ... test implementation

    @pytest.mark.asyncio
    async def test_email_includes_news_section(self):
        """Test that generated email includes news digest."""
        # ... test implementation


class TestSentimentAnalysis:
    """Tests for sentiment analysis."""

    def test_vader_positive_financial_terms(self):
        """Test VADER recognizes positive financial terms."""
        from trading_system.research.sentiment.vader_analyzer import VADERSentimentAnalyzer

        analyzer = VADERSentimentAnalyzer()

        # Test positive terms
        score, label, _ = analyzer.analyze("Apple stock surges on record earnings beat")
        assert score > 0.3
        assert label.value in ["positive", "very_positive"]

    def test_vader_negative_financial_terms(self):
        """Test VADER recognizes negative financial terms."""
        from trading_system.research.sentiment.vader_analyzer import VADERSentimentAnalyzer

        analyzer = VADERSentimentAnalyzer()

        # Test negative terms
        score, label, _ = analyzer.analyze("Stock plunges amid bankruptcy fears")
        assert score < -0.3
        assert label.value in ["negative", "very_negative"]

    def test_ticker_extraction(self):
        """Test ticker extraction from text."""
        from trading_system.research.entity_extraction.ticker_extractor import TickerExtractor

        extractor = TickerExtractor()

        # Test various formats
        assert "AAPL" in extractor.extract("$AAPL is up today")
        assert "AAPL" in extractor.extract("Apple stock rises")
        assert "BTC" in extractor.extract("Bitcoin price surges")
        assert "MSFT" in extractor.extract("Microsoft (MSFT) announces...")
```

**Acceptance Criteria**:
- [ ] All integration tests pass
- [ ] Tests cover happy path and edge cases
- [ ] Mock responses are realistic
- [ ] Tests run in CI/CD

---

## Dependencies to Install

Add to `pyproject.toml`:
```toml
[project.optional-dependencies]
research = [
    "vaderSentiment>=3.3.0,<4.0.0",  # Sentiment analysis
    "nltk>=3.8.0,<4.0.0",            # NLP utilities
]

# Update 'all' group to include research
all = [
    # ... existing ...
    "vaderSentiment>=3.3.0,<4.0.0",
    "nltk>=3.8.0,<4.0.0",
]
```

---

## Summary: Phase 2 Tasks

| Task | Description | Effort |
|------|-------------|--------|
| 2.1.1 | Create news data source structure | 1h |
| 2.1.2 | Implement NewsAPI.org client | 3h |
| 2.1.3 | Implement Alpha Vantage News client | 2h |
| 2.1.4 | Implement News Aggregator | 2h |
| 2.2.1 | Create sentiment analysis structure | 1h |
| 2.2.2 | Implement VADER with financial lexicon | 3h |
| 2.2.3 | Implement ticker extractor | 2h |
| 2.2.4 | Implement News Analyzer orchestrator | 4h |
| 2.3.1 | Integrate news into signal generator | 3h |
| 2.3.2 | Update email template with news | 2h |
| 2.3.3 | Update daily job for news | 1h |
| 2.4.1 | Configuration updates | 1h |
| 2.4.2 | Integration tests | 3h |

**Total Estimated Effort**: ~28 hours (2 weeks)

---

**Document Created**: 2024-12-30
**Phase**: 2 - News Integration
**Prerequisites**: Phase 1 MVP complete

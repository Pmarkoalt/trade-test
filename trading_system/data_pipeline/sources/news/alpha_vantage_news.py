"""Alpha Vantage news client implementation."""

import logging
from datetime import datetime
from typing import List, Optional
from urllib.parse import urlencode

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore

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
            
        Raises:
            ImportError: If aiohttp is not installed
        """
        if aiohttp is None:
            raise ImportError("aiohttp is required for AlphaVantageNewsClient. Install it with: pip install aiohttp")
        super().__init__(api_key, rate_limit_per_minute)
        if not api_key:
            raise ValueError("Alpha Vantage API key is required")

    @property
    def source_name(self) -> str:
        """Return the name of this news source."""
        return "AlphaVantage"

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

        # Alpha Vantage accepts comma-separated tickers
        # Process in batches to respect rate limits
        batch_size = 5
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]

            try:
                await self._respect_rate_limit()

                # Convert crypto symbols to AV format
                tickers = [self._format_ticker(s) for s in batch]
                tickers_str = ",".join(tickers)

                articles = await self._fetch_news_sentiment(
                    tickers=tickers_str,
                    limit=max_articles_per_symbol * len(batch),
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
            error_message="; ".join(errors) if errors else None,
        )

    async def fetch_market_news(
        self,
        category: str = "general",
        lookback_hours: int = 24,
        max_articles: int = 20,
    ) -> NewsFetchResult:
        """Fetch general market news.

        Args:
            category: News category to fetch
            lookback_hours: How far back to search (not used for Alpha Vantage)
            max_articles: Maximum articles to return

        Returns:
            NewsFetchResult with fetched articles
        """
        try:
            await self._respect_rate_limit()

            # Map category to AV topics
            topic_map = {
                "general": "financial_markets",
                "technology": "technology",
                "crypto": "blockchain",
                "earnings": "earnings",
                "ipo": "ipo",
                "mergers": "mergers_and_acquisitions",
            }
            topics = topic_map.get(category, "financial_markets")

            articles = await self._fetch_news_sentiment(
                topics=topics,
                limit=max_articles,
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

    async def _fetch_news_sentiment(
        self,
        tickers: Optional[str] = None,
        topics: Optional[str] = None,
        limit: int = 50,
    ) -> List[NewsArticle]:
        """Fetch news with sentiment from Alpha Vantage.

        Args:
            tickers: Comma-separated ticker symbols (optional)
            topics: Topic filter (optional)
            limit: Maximum number of articles to return (max 200)

        Returns:
            List of NewsArticle objects with sentiment scores
        """
        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": self.api_key,
            "limit": min(limit, 200),
        }

        if tickers:
            params["tickers"] = tickers
        if topics:
            params["topics"] = topics

        url = f"{self.BASE_URL}?{urlencode(params)}"

        if aiohttp is None:
            raise ImportError("aiohttp is required for AlphaVantageNewsClient. Install it with: pip install aiohttp")
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
        """Parse Alpha Vantage news response into NewsArticle objects.

        Args:
            raw_articles: List of raw article dictionaries from API response

        Returns:
            List of NewsArticle objects with sentiment information
        """
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
                symbols = [s.replace("CRYPTO:", "") for s in symbols if s]  # Remove CRYPTO: prefix for storage

                # Get overall sentiment (Alpha Vantage provides this!)
                overall_sentiment = raw.get("overall_sentiment_score")
                sentiment_label = self._map_sentiment_label(
                    raw.get("overall_sentiment_label", "Neutral")
                )

                # Convert sentiment score to float if present
                sentiment_score = None
                if overall_sentiment is not None:
                    try:
                        sentiment_score = float(overall_sentiment)
                    except (ValueError, TypeError):
                        sentiment_score = None

                article = NewsArticle(
                    id=f"alphavantage_{abs(hash(raw.get('url', '')))}",
                    source=raw.get("source", "Unknown"),
                    title=raw.get("title", ""),
                    summary=raw.get("summary", ""),
                    url=raw.get("url", ""),
                    published_at=published_at,
                    symbols=symbols,
                    # Alpha Vantage provides sentiment!
                    sentiment_score=sentiment_score,
                    sentiment_label=sentiment_label,
                    is_processed=True,  # Already has sentiment
                )
                articles.append(article)

            except Exception as e:
                logger.warning(f"Error parsing article: {e}")
                continue

        return articles

    def _format_ticker(self, symbol: str) -> str:
        """Format symbol for Alpha Vantage API.

        Args:
            symbol: Stock or crypto symbol

        Returns:
            Formatted ticker (crypto gets CRYPTO: prefix)
        """
        # Crypto needs CRYPTO: prefix
        crypto_symbols = {"BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOT", "MATIC", "LTC", "LINK"}
        if symbol.upper() in crypto_symbols:
            return f"CRYPTO:{symbol.upper()}"
        return symbol.upper()

    def _map_sentiment_label(self, av_label: str) -> Optional[SentimentLabel]:
        """Map Alpha Vantage sentiment label to our enum.

        Args:
            av_label: Alpha Vantage sentiment label (Bullish, Somewhat-Bullish, etc.)

        Returns:
            SentimentLabel enum value or None
        """
        mapping = {
            "Bullish": SentimentLabel.POSITIVE,
            "Somewhat-Bullish": SentimentLabel.POSITIVE,
            "Neutral": SentimentLabel.NEUTRAL,
            "Somewhat-Bearish": SentimentLabel.NEGATIVE,
            "Bearish": SentimentLabel.NEGATIVE,
        }
        return mapping.get(av_label, SentimentLabel.NEUTRAL)


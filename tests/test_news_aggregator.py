"""Unit tests for News Aggregator."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from trading_system.data_pipeline.sources.news.models import NewsArticle
from trading_system.data_pipeline.sources.news.news_aggregator import NewsAggregator


@pytest.fixture
def sample_articles():
    """Create sample news articles for testing."""
    return [
        NewsArticle(
            id="article1",
            source="Reuters",
            title="Apple announces new product",
            summary="Apple Inc. announced a new product today.",
            url="https://example.com/apple",
            published_at=datetime.now() - timedelta(hours=1),
            symbols=["AAPL"],
        ),
        NewsArticle(
            id="article2",
            source="Bloomberg",
            title="Microsoft earnings beat expectations",
            summary="Microsoft reported strong quarterly earnings.",
            url="https://example.com/msft",
            published_at=datetime.now() - timedelta(hours=2),
            symbols=["MSFT"],
        ),
        NewsArticle(
            id="article3",
            source="Reuters",
            title="Apple announces new product",  # Duplicate title
            summary="Different summary but same title.",
            url="https://example.com/apple2",
            published_at=datetime.now() - timedelta(hours=3),
            symbols=["AAPL"],
        ),
    ]


@pytest.fixture
def mock_newsapi_client():
    """Create a mock NewsAPI client."""
    mock_client = AsyncMock()
    mock_client.source_name = "NewsAPI"
    return mock_client


@pytest.fixture
def mock_alphavantage_client():
    """Create a mock Alpha Vantage client."""
    mock_client = AsyncMock()
    mock_client.source_name = "AlphaVantage"
    return mock_client


class TestNewsAggregator:
    """Test NewsAggregator class."""

    def test_init_with_both_keys(self):
        """Test initialization with both API keys."""
        aggregator = NewsAggregator(newsapi_key="test_newsapi", alpha_vantage_key="test_av")
        assert len(aggregator.sources) == 2
        assert aggregator.enable_caching is True

    def test_init_with_newsapi_only(self):
        """Test initialization with only NewsAPI key."""
        aggregator = NewsAggregator(newsapi_key="test_newsapi")
        assert len(aggregator.sources) == 1
        assert aggregator.sources[0].source_name == "NewsAPI"

    def test_init_with_alphavantage_only(self):
        """Test initialization with only Alpha Vantage key."""
        aggregator = NewsAggregator(alpha_vantage_key="test_av")
        assert len(aggregator.sources) == 1
        assert aggregator.sources[0].source_name == "AlphaVantage"

    def test_init_with_no_keys(self):
        """Test initialization with no API keys."""
        aggregator = NewsAggregator()
        assert len(aggregator.sources) == 0

    def test_init_with_caching_disabled(self):
        """Test initialization with caching disabled."""
        aggregator = NewsAggregator(
            newsapi_key="test_newsapi", enable_caching=False, cache_ttl_minutes=30
        )
        assert aggregator.enable_caching is False

    @pytest.mark.asyncio
    async def test_fetch_news_for_symbols_success(self, sample_articles):
        """Test successful fetch from multiple sources."""
        # Create aggregator with mocked sources
        with patch("trading_system.data_pipeline.sources.news.news_aggregator.NewsAPIClient") as mock_newsapi_class, patch(
            "trading_system.data_pipeline.sources.news.news_aggregator.AlphaVantageNewsClient"
        ) as mock_av_class:
            mock_newsapi = AsyncMock()
            mock_newsapi.source_name = "NewsAPI"
            mock_newsapi.fetch_articles = AsyncMock(
                return_value=type(
                    "obj",
                    (object,),
                    {
                        "success": True,
                        "articles": sample_articles[:2],
                        "error_message": None,
                    },
                )()
            )

            mock_av = AsyncMock()
            mock_av.source_name = "AlphaVantage"
            mock_av.fetch_articles = AsyncMock(
                return_value=type(
                    "obj",
                    (object,),
                    {
                        "success": True,
                        "articles": sample_articles[1:],
                        "error_message": None,
                    },
                )()
            )

            mock_newsapi_class.return_value = mock_newsapi
            mock_av_class.return_value = mock_av

            aggregator = NewsAggregator(newsapi_key="test", alpha_vantage_key="test")

            result = await aggregator.fetch_news_for_symbols(symbols=["AAPL", "MSFT"])

            # Should deduplicate - article3 has duplicate title of article1
            assert len(result) == 2
            # Should be sorted by published_at (newest first)
            assert result[0].published_at >= result[1].published_at

    @pytest.mark.asyncio
    async def test_fetch_news_for_symbols_deduplication(self, sample_articles):
        """Test that duplicate articles are removed."""
        with patch("trading_system.data_pipeline.sources.news.news_aggregator.NewsAPIClient") as mock_newsapi_class:
            mock_newsapi = AsyncMock()
            mock_newsapi.source_name = "NewsAPI"
            mock_newsapi.fetch_articles = AsyncMock(
                return_value=type(
                    "obj",
                    (object,),
                    {
                        "success": True,
                        "articles": sample_articles,  # Includes duplicate title
                        "error_message": None,
                    },
                )()
            )

            mock_newsapi_class.return_value = mock_newsapi

            aggregator = NewsAggregator(newsapi_key="test", enable_caching=False)

            result = await aggregator.fetch_news_for_symbols(symbols=["AAPL"])

            # Should deduplicate articles with same title (article1 and article3)
            assert len(result) == 2

    @pytest.mark.asyncio
    async def test_fetch_news_for_symbols_source_failure(self):
        """Test that source failures are handled gracefully."""
        with patch("trading_system.data_pipeline.sources.news.news_aggregator.NewsAPIClient") as mock_newsapi_class, patch(
            "trading_system.data_pipeline.sources.news.news_aggregator.AlphaVantageNewsClient"
        ) as mock_av_class:
            mock_newsapi = AsyncMock()
            mock_newsapi.source_name = "NewsAPI"
            mock_newsapi.fetch_articles = AsyncMock(
                return_value=type(
                    "obj",
                    (object,),
                    {
                        "success": False,
                        "articles": [],
                        "error_message": "API error",
                    },
                )()
            )

            mock_av = AsyncMock()
            mock_av.source_name = "AlphaVantage"
            mock_av.fetch_articles = AsyncMock(
                return_value=type(
                    "obj",
                    (object,),
                    {
                        "success": True,
                        "articles": [sample_articles[0]],
                        "error_message": None,
                    },
                )()
            )

            mock_newsapi_class.return_value = mock_newsapi
            mock_av_class.return_value = mock_av

            aggregator = NewsAggregator(newsapi_key="test", alpha_vantage_key="test", enable_caching=False)

            result = await aggregator.fetch_news_for_symbols(symbols=["AAPL"])

            # Should still return articles from successful source
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_fetch_news_for_symbols_exception_handling(self, sample_articles):
        """Test that exceptions from sources are handled gracefully."""
        with patch("trading_system.data_pipeline.sources.news.news_aggregator.NewsAPIClient") as mock_newsapi_class, patch(
            "trading_system.data_pipeline.sources.news.news_aggregator.AlphaVantageNewsClient"
        ) as mock_av_class:
            mock_newsapi = AsyncMock()
            mock_newsapi.source_name = "NewsAPI"
            mock_newsapi.fetch_articles = AsyncMock(side_effect=Exception("Connection error"))

            mock_av = AsyncMock()
            mock_av.source_name = "AlphaVantage"
            mock_av.fetch_articles = AsyncMock(
                return_value=type(
                    "obj",
                    (object,),
                    {
                        "success": True,
                        "articles": [sample_articles[0]],
                        "error_message": None,
                    },
                )()
            )

            mock_newsapi_class.return_value = mock_newsapi
            mock_av_class.return_value = mock_av

            aggregator = NewsAggregator(newsapi_key="test", alpha_vantage_key="test", enable_caching=False)

            result = await aggregator.fetch_news_for_symbols(symbols=["AAPL"])

            # Should still return articles from non-failing source
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_fetch_news_for_symbols_caching(self, sample_articles):
        """Test that caching works correctly."""
        with patch("trading_system.data_pipeline.sources.news.news_aggregator.NewsAPIClient") as mock_newsapi_class:
            mock_newsapi = AsyncMock()
            mock_newsapi.source_name = "NewsAPI"
            mock_newsapi.fetch_articles = AsyncMock(
                return_value=type(
                    "obj",
                    (object,),
                    {
                        "success": True,
                        "articles": [sample_articles[0]],
                        "error_message": None,
                    },
                )()
            )

            mock_newsapi_class.return_value = mock_newsapi

            aggregator = NewsAggregator(newsapi_key="test", enable_caching=True, cache_ttl_minutes=60)

            # First call
            result1 = await aggregator.fetch_news_for_symbols(symbols=["AAPL"])

            # Second call should use cache
            result2 = await aggregator.fetch_news_for_symbols(symbols=["AAPL"])

            # fetch_articles should only be called once due to caching
            assert mock_newsapi.fetch_articles.call_count == 1
            assert len(result1) == len(result2)

    @pytest.mark.asyncio
    async def test_fetch_news_for_symbols_cache_expiration(self, sample_articles):
        """Test that cache expiration works correctly."""
        with patch("trading_system.data_pipeline.sources.news.news_aggregator.NewsAPIClient") as mock_newsapi_class:
            mock_newsapi = AsyncMock()
            mock_newsapi.source_name = "NewsAPI"
            mock_newsapi.fetch_articles = AsyncMock(
                return_value=type(
                    "obj",
                    (object,),
                    {
                        "success": True,
                        "articles": [sample_articles[0]],
                        "error_message": None,
                    },
                )()
            )

            mock_newsapi_class.return_value = mock_newsapi

            aggregator = NewsAggregator(newsapi_key="test", enable_caching=True, cache_ttl_minutes=30)

            # First call - populates cache
            await aggregator.fetch_news_for_symbols(symbols=["AAPL"])

            # Manually expire the cache by setting old timestamp
            cache_key = aggregator._make_cache_key(["AAPL"], 48)
            old_time = datetime.now() - timedelta(minutes=31)
            aggregator._cache[cache_key] = (old_time, [sample_articles[0]])

            # Second call should hit API again (cache expired)
            await aggregator.fetch_news_for_symbols(symbols=["AAPL"], lookback_hours=48)

            # fetch_articles should be called twice (once initial, once after cache expired)
            assert mock_newsapi.fetch_articles.call_count == 2

    @pytest.mark.asyncio
    async def test_fetch_market_overview(self, sample_articles):
        """Test fetching market overview news."""
        with patch("trading_system.data_pipeline.sources.news.news_aggregator.NewsAPIClient") as mock_newsapi_class:
            mock_newsapi = AsyncMock()
            mock_newsapi.source_name = "NewsAPI"
            mock_newsapi.fetch_market_news = AsyncMock(
                return_value=type(
                    "obj",
                    (object,),
                    {
                        "success": True,
                        "articles": sample_articles,
                        "error_message": None,
                    },
                )()
            )

            mock_newsapi_class.return_value = mock_newsapi

            aggregator = NewsAggregator(newsapi_key="test", enable_caching=False)

            result = await aggregator.fetch_market_overview(categories=["business"], max_articles=5)

            assert len(result) <= 5
            # Should be sorted by published_at
            if len(result) > 1:
                assert result[0].published_at >= result[1].published_at

    @pytest.mark.asyncio
    async def test_fetch_market_overview_default_categories(self):
        """Test that default categories are used when none provided."""
        with patch("trading_system.data_pipeline.sources.news.news_aggregator.NewsAPIClient") as mock_newsapi_class:
            mock_newsapi = AsyncMock()
            mock_newsapi.source_name = "NewsAPI"
            mock_newsapi.fetch_market_news = AsyncMock(
                return_value=type(
                    "obj",
                    (object,),
                    {
                        "success": True,
                        "articles": [],
                        "error_message": None,
                    },
                )()
            )

            mock_newsapi_class.return_value = mock_newsapi

            aggregator = NewsAggregator(newsapi_key="test", enable_caching=False)

            await aggregator.fetch_market_overview(max_articles=10)

            # Should be called with default categories (business, technology)
            assert mock_newsapi.fetch_market_news.call_count == 2

    def test_clear_cache(self, sample_articles):
        """Test that cache can be cleared."""
        aggregator = NewsAggregator(newsapi_key="test", enable_caching=True)
        aggregator._cache["test_key"] = (datetime.now(), [sample_articles[0]])

        aggregator.clear_cache()

        assert len(aggregator._cache) == 0

    def test_deduplicate_articles(self, sample_articles):
        """Test deduplication logic."""
        aggregator = NewsAggregator(newsapi_key="test", enable_caching=False)

        # article1 and article3 have the same title (case-insensitive, trimmed)
        deduplicated = aggregator._deduplicate_articles(sample_articles)

        assert len(deduplicated) == 2
        # Should keep first occurrence
        assert deduplicated[0].id == "article1"

    def test_make_cache_key(self):
        """Test cache key generation."""
        aggregator = NewsAggregator(newsapi_key="test")

        key1 = aggregator._make_cache_key(["AAPL", "MSFT"], 48)
        key2 = aggregator._make_cache_key(["MSFT", "AAPL"], 48)  # Different order
        key3 = aggregator._make_cache_key(["AAPL", "MSFT"], 24)  # Different lookback

        assert key1 == key2  # Should be the same (sorted)
        assert key1 != key3  # Should be different (different lookback)

    def test_get_cached_not_expired(self, sample_articles):
        """Test getting cached articles that are not expired."""
        aggregator = NewsAggregator(newsapi_key="test", enable_caching=True, cache_ttl_minutes=60)
        articles = [sample_articles[0]]

        aggregator._cache["test_key"] = (datetime.now() - timedelta(minutes=30), articles)

        cached = aggregator._get_cached("test_key")

        assert cached == articles

    def test_get_cached_expired(self, sample_articles):
        """Test that expired cache entries are not returned."""
        aggregator = NewsAggregator(newsapi_key="test", enable_caching=True, cache_ttl_minutes=30)
        articles = [sample_articles[0]]

        aggregator._cache["test_key"] = (datetime.now() - timedelta(minutes=31), articles)

        cached = aggregator._get_cached("test_key")

        assert cached is None
        assert "test_key" not in aggregator._cache  # Should be deleted

    def test_get_cached_disabled(self):
        """Test that caching is disabled when enable_caching is False."""
        aggregator = NewsAggregator(newsapi_key="test", enable_caching=False)
        articles = [sample_articles[0]]

        aggregator._cache["test_key"] = (datetime.now(), articles)

        cached = aggregator._get_cached("test_key")

        assert cached is None


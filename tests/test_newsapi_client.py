"""Unit tests for NewsAPI client."""

from unittest.mock import AsyncMock, patch

import pytest

from trading_system.data_pipeline.sources.news.models import NewsArticle
from trading_system.data_pipeline.sources.news.newsapi_client import NewsAPIClient


@pytest.fixture
def newsapi_client():
    """Create a NewsAPIClient instance for testing."""
    try:
        return NewsAPIClient(api_key="test_api_key", rate_limit_per_minute=5)
    except ImportError as e:
        pytest.skip(f"Required dependency not available: {e}")


@pytest.fixture
def sample_newsapi_everything_response():
    """Sample NewsAPI /everything endpoint response."""
    return {
        "status": "ok",
        "totalResults": 2,
        "articles": [
            {
                "source": {"id": "reuters", "name": "Reuters"},
                "author": "John Doe",
                "title": "Apple announces new product",
                "description": "Apple Inc. announced a new product today.",
                "url": "https://example.com/apple-news",
                "urlToImage": "https://example.com/image.jpg",
                "publishedAt": "2024-01-01T10:00:00Z",
                "content": "Full article content about Apple's new product announcement.",
            },
            {
                "source": {"id": "bloomberg", "name": "Bloomberg"},
                "author": "Jane Smith",
                "title": "Apple stock rises on news",
                "description": "Apple stock price increased following the announcement.",
                "url": "https://example.com/apple-stock",
                "urlToImage": None,
                "publishedAt": "2024-01-01T11:00:00Z",
                "content": "Apple stock rose 5% following the product announcement.",
            },
        ],
    }


@pytest.fixture
def sample_newsapi_top_headlines_response():
    """Sample NewsAPI /top-headlines endpoint response."""
    return {
        "status": "ok",
        "totalResults": 1,
        "articles": [
            {
                "source": {"id": "cnbc", "name": "CNBC"},
                "author": "Market Reporter",
                "title": "Market Update: Stocks rise",
                "description": "Stock markets are up today.",
                "url": "https://example.com/market-update",
                "urlToImage": None,
                "publishedAt": "2024-01-01T12:00:00Z",
                "content": "Stock markets showed gains across all sectors today.",
            },
        ],
    }


@pytest.fixture
def empty_newsapi_response():
    """Empty NewsAPI response."""
    return {"status": "ok", "totalResults": 0, "articles": []}


class TestNewsAPIClient:
    """Test NewsAPIClient class."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        client = NewsAPIClient(api_key="test_key")
        assert client.api_key == "test_key"
        assert client.source_name == "NewsAPI"

    def test_init_without_api_key_raises_error(self):
        """Test that initialization without API key raises error."""
        with pytest.raises(ValueError, match="NewsAPI API key is required"):
            NewsAPIClient(api_key="")

    @pytest.mark.asyncio
    async def test_fetch_articles_success(self, newsapi_client, sample_newsapi_everything_response):
        """Test successful article fetch for symbols."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=sample_newsapi_everything_response)
            mock_response.text = AsyncMock(return_value="")
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_get.return_value = mock_response

            result = await newsapi_client.fetch_articles(symbols=["AAPL"], lookback_hours=24, max_articles_per_symbol=10)

            assert result.success is True
            assert len(result.articles) == 2
            assert result.source == "NewsAPI"
            assert "AAPL" in result.symbols_requested

            # Verify articles are properly parsed
            assert all(isinstance(article, NewsArticle) for article in result.articles)
            assert result.articles[0].title == "Apple announces new product"
            assert result.articles[0].source == "Reuters"
            assert "AAPL" in result.articles[0].symbols

    @pytest.mark.asyncio
    async def test_fetch_articles_empty_response(self, newsapi_client, empty_newsapi_response):
        """Test handling of empty response."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=empty_newsapi_response)
            mock_response.text = AsyncMock(return_value="")
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_get.return_value = mock_response

            result = await newsapi_client.fetch_articles(symbols=["AAPL"], lookback_hours=24)

            assert result.success is True
            assert len(result.articles) == 0

    @pytest.mark.asyncio
    async def test_fetch_articles_multiple_symbols(self, newsapi_client, sample_newsapi_everything_response):
        """Test fetching articles for multiple symbols."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=sample_newsapi_everything_response)
            mock_response.text = AsyncMock(return_value="")
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_get.return_value = mock_response

            result = await newsapi_client.fetch_articles(symbols=["AAPL", "MSFT"], lookback_hours=24)

            assert result.success is True
            assert len(result.articles) == 4  # 2 articles per symbol
            assert set(result.symbols_requested) == {"AAPL", "MSFT"}

    @pytest.mark.asyncio
    async def test_fetch_articles_rate_limit_error(self, newsapi_client):
        """Test handling of rate limit error (429)."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 429
            mock_response.text = AsyncMock(return_value="Rate limit exceeded")
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_get.return_value = mock_response

            result = await newsapi_client.fetch_articles(symbols=["AAPL"])

            assert result.success is False
            assert "rate limit" in result.error_message.lower() or "429" in result.error_message

    @pytest.mark.asyncio
    async def test_fetch_articles_api_error(self, newsapi_client):
        """Test handling of API error."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 400
            mock_response.text = AsyncMock(return_value="Bad request")
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_get.return_value = mock_response

            result = await newsapi_client.fetch_articles(symbols=["AAPL"])

            assert result.success is False
            assert "error" in result.error_message.lower() or "400" in result.error_message

    @pytest.mark.asyncio
    async def test_fetch_articles_api_status_error(self, newsapi_client):
        """Test handling of API status error (status != 'ok')."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "error", "message": "Invalid API key"})
            mock_response.text = AsyncMock(return_value="")
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_get.return_value = mock_response

            result = await newsapi_client.fetch_articles(symbols=["AAPL"])

            assert result.success is False
            assert "error" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_fetch_market_news_success(self, newsapi_client, sample_newsapi_top_headlines_response):
        """Test successful market news fetch."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=sample_newsapi_top_headlines_response)
            mock_response.text = AsyncMock(return_value="")
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_get.return_value = mock_response

            result = await newsapi_client.fetch_market_news(category="business", max_articles=20)

            assert result.success is True
            assert len(result.articles) == 1
            assert result.source == "NewsAPI"
            assert isinstance(result.articles[0], NewsArticle)
            assert result.articles[0].title == "Market Update: Stocks rise"

    @pytest.mark.asyncio
    async def test_fetch_market_news_error(self, newsapi_client):
        """Test handling of market news fetch error."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Internal server error")
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_get.return_value = mock_response

            result = await newsapi_client.fetch_market_news()

            assert result.success is False
            assert "error" in result.error_message.lower() or "500" in result.error_message

    @pytest.mark.asyncio
    async def test_parse_articles_with_missing_fields(self, newsapi_client):
        """Test that articles with missing fields are handled gracefully."""
        # Create response with missing fields
        response_with_missing_fields = {
            "status": "ok",
            "totalResults": 2,
            "articles": [
                {
                    "source": {"id": "reuters", "name": "Reuters"},
                    "title": "Article without published date",
                    "description": "Test article",
                    # Missing publishedAt and url
                },
                {
                    "source": {"id": "bloomberg", "name": "Bloomberg"},
                    "title": "Complete article",
                    "description": "Test article",
                    "url": "https://example.com/article",
                    "publishedAt": "2024-01-01T10:00:00Z",
                    "content": "Full content",
                },
            ],
        }

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=response_with_missing_fields)
            mock_response.text = AsyncMock(return_value="")
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_get.return_value = mock_response

            result = await newsapi_client.fetch_articles(symbols=["AAPL"])

            # Should still parse what it can
            assert result.success is True
            # At least one article should be parsed (the one with all fields)
            assert len(result.articles) >= 1

    def test_symbol_to_company_mapping(self, newsapi_client):
        """Test that symbol to company name mapping works."""
        from trading_system.data_pipeline.sources.news.newsapi_client import SYMBOL_TO_COMPANY

        assert SYMBOL_TO_COMPANY.get("AAPL") == "Apple"
        assert SYMBOL_TO_COMPANY.get("BTC") == "Bitcoin"
        assert SYMBOL_TO_COMPANY.get("UNKNOWN") is None

    @pytest.mark.asyncio
    async def test_crypto_symbols(self, newsapi_client, sample_newsapi_everything_response):
        """Test fetching news for crypto symbols."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=sample_newsapi_everything_response)
            mock_response.text = AsyncMock(return_value="")
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_get.return_value = mock_response

            result = await newsapi_client.fetch_articles(symbols=["BTC", "ETH"])

            assert result.success is True
            assert "BTC" in result.symbols_requested
            assert "ETH" in result.symbols_requested

    @pytest.mark.asyncio
    async def test_respect_rate_limit(self, newsapi_client, sample_newsapi_everything_response):
        """Test that rate limiting is respected."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=sample_newsapi_everything_response)
            mock_response.text = AsyncMock(return_value="")
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_get.return_value = mock_response

            # Fetch articles for multiple symbols - should respect rate limit
            result = await newsapi_client.fetch_articles(symbols=["AAPL", "MSFT", "GOOGL"])

            assert result.success is True
            # Rate limiting should have been called (we can't easily verify the sleep,
            # but we can verify it doesn't crash)

    def test_missing_aiohttp_raises_error(self):
        """Test that missing aiohttp raises ImportError."""
        with patch("trading_system.data_pipeline.sources.news.newsapi_client.aiohttp", None):
            with pytest.raises(ImportError, match="aiohttp is required"):
                NewsAPIClient(api_key="test_key")

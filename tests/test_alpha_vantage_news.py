"""Unit tests for Alpha Vantage News client."""

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from trading_system.data_pipeline.sources.news.models import NewsArticle, SentimentLabel
from trading_system.data_pipeline.sources.news.alpha_vantage_news import AlphaVantageNewsClient


@pytest.fixture
def alpha_vantage_client():
    """Create an AlphaVantageNewsClient instance for testing."""
    return AlphaVantageNewsClient(api_key="test_api_key", rate_limit_per_minute=5)


@pytest.fixture
def sample_alphavantage_response():
    """Sample Alpha Vantage News API response."""
    return {
        "feed": [
            {
                "title": "Apple Inc. announces quarterly earnings",
                "url": "https://example.com/apple-earnings",
                "time_published": "20240115T120000",
                "authors": ["John Doe"],
                "summary": "Apple reported strong quarterly earnings.",
                "source": "Reuters",
                "category_within_source": "Technology",
                "source_domain": "reuters.com",
                "topics": [{"topic": "Earnings", "relevance_score": "0.95"}],
                "overall_sentiment_score": 0.65,
                "overall_sentiment_label": "Bullish",
                "ticker_sentiment": [
                    {
                        "ticker": "AAPL",
                        "relevance_score": "0.98",
                        "ticker_sentiment_score": "0.70",
                        "ticker_sentiment_label": "Bullish",
                    }
                ],
            },
            {
                "title": "Bitcoin price surges",
                "url": "https://example.com/bitcoin-news",
                "time_published": "20240115T130000",
                "authors": ["Jane Smith"],
                "summary": "Bitcoin price increased significantly.",
                "source": "Bloomberg",
                "category_within_source": "Finance",
                "source_domain": "bloomberg.com",
                "topics": [{"topic": "Blockchain", "relevance_score": "0.92"}],
                "overall_sentiment_score": 0.55,
                "overall_sentiment_label": "Somewhat-Bullish",
                "ticker_sentiment": [
                    {
                        "ticker": "CRYPTO:BTC",
                        "relevance_score": "0.95",
                        "ticker_sentiment_score": "0.60",
                        "ticker_sentiment_label": "Somewhat-Bullish",
                    }
                ],
            },
        ]
    }


@pytest.fixture
def sample_alphavantage_response_neutral():
    """Sample Alpha Vantage News API response with neutral sentiment."""
    return {
        "feed": [
            {
                "title": "Market update",
                "url": "https://example.com/market-update",
                "time_published": "20240115T140000",
                "authors": [],
                "summary": "Market remains stable.",
                "source": "CNBC",
                "category_within_source": "Finance",
                "source_domain": "cnbc.com",
                "topics": [{"topic": "Financial Markets", "relevance_score": "0.80"}],
                "overall_sentiment_score": 0.0,
                "overall_sentiment_label": "Neutral",
                "ticker_sentiment": [],
            }
        ]
    }


@pytest.fixture
def sample_alphavantage_error_response():
    """Sample Alpha Vantage error response."""
    return {"Error Message": "Invalid API key"}


@pytest.fixture
def empty_alphavantage_response():
    """Empty Alpha Vantage response."""
    return {"feed": []}


class TestAlphaVantageNewsClient:
    """Test AlphaVantageNewsClient class."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        client = AlphaVantageNewsClient(api_key="test_key")
        assert client.api_key == "test_key"
        assert client.source_name == "AlphaVantage"

    def test_init_without_api_key_raises_error(self):
        """Test that initialization without API key raises error."""
        with pytest.raises(ValueError, match="Alpha Vantage API key is required"):
            AlphaVantageNewsClient(api_key="")

    @pytest.mark.asyncio
    async def test_fetch_articles_success(self, alpha_vantage_client, sample_alphavantage_response):
        """Test successful article fetch for symbols."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=sample_alphavantage_response)
            mock_response.text = AsyncMock(return_value="")
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_get.return_value = mock_response

            result = await alpha_vantage_client.fetch_articles(symbols=["AAPL"], lookback_hours=24)

            assert result.success is True
            assert len(result.articles) == 2  # Response has 2 articles
            assert result.source == "AlphaVantage"
            assert "AAPL" in result.symbols_requested

            # Verify articles are properly parsed
            assert all(isinstance(article, NewsArticle) for article in result.articles)
            assert result.articles[0].title == "Apple Inc. announces quarterly earnings"
            assert result.articles[0].source == "Reuters"
            assert result.articles[0].sentiment_score == 0.65
            assert result.articles[0].sentiment_label == SentimentLabel.POSITIVE
            assert result.articles[0].is_processed is True

    @pytest.mark.asyncio
    async def test_fetch_articles_crypto_symbols(self, alpha_vantage_client, sample_alphavantage_response):
        """Test fetching articles for crypto symbols (with CRYPTO: prefix)."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=sample_alphavantage_response)
            mock_response.text = AsyncMock(return_value="")
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_get.return_value = mock_response

            result = await alpha_vantage_client.fetch_articles(symbols=["BTC"])

            assert result.success is True
            assert "BTC" in result.symbols_requested
            # Verify crypto symbols are stored without CRYPTO: prefix
            btc_article = next((a for a in result.articles if "BTC" in a.symbols), None)
            if btc_article:
                assert "BTC" in btc_article.symbols
                assert "CRYPTO:BTC" not in btc_article.symbols

    @pytest.mark.asyncio
    async def test_fetch_articles_empty_response(self, alpha_vantage_client, empty_alphavantage_response):
        """Test handling of empty response."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=empty_alphavantage_response)
            mock_response.text = AsyncMock(return_value="")
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_get.return_value = mock_response

            result = await alpha_vantage_client.fetch_articles(symbols=["AAPL"])

            assert result.success is True
            assert len(result.articles) == 0

    @pytest.mark.asyncio
    async def test_fetch_articles_error_response(self, alpha_vantage_client, sample_alphavantage_error_response):
        """Test handling of API error response."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=sample_alphavantage_error_response)
            mock_response.text = AsyncMock(return_value="")
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_get.return_value = mock_response

            result = await alpha_vantage_client.fetch_articles(symbols=["AAPL"])

            assert result.success is False
            assert "error" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_fetch_articles_http_error(self, alpha_vantage_client):
        """Test handling of HTTP error."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Internal server error")
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_get.return_value = mock_response

            result = await alpha_vantage_client.fetch_articles(symbols=["AAPL"])

            assert result.success is False
            assert "error" in result.error_message.lower() or "500" in result.error_message

    @pytest.mark.asyncio
    async def test_fetch_market_news_success(self, alpha_vantage_client, sample_alphavantage_response):
        """Test successful market news fetch."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=sample_alphavantage_response)
            mock_response.text = AsyncMock(return_value="")
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_get.return_value = mock_response

            result = await alpha_vantage_client.fetch_market_news(category="general", max_articles=20)

            assert result.success is True
            assert len(result.articles) == 2
            assert result.source == "AlphaVantage"

    @pytest.mark.asyncio
    async def test_fetch_market_news_category_mapping(self, alpha_vantage_client, sample_alphavantage_response):
        """Test that category is properly mapped to topics."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=sample_alphavantage_response)
            mock_response.text = AsyncMock(return_value="")
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_get.return_value = mock_response

            # Test different categories
            result = await alpha_vantage_client.fetch_market_news(category="technology")
            assert result.success is True

            result = await alpha_vantage_client.fetch_market_news(category="crypto")
            assert result.success is True

    @pytest.mark.asyncio
    async def test_sentiment_label_mapping(self, alpha_vantage_client, sample_alphavantage_response_neutral):
        """Test that sentiment labels are properly mapped."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=sample_alphavantage_response_neutral)
            mock_response.text = AsyncMock(return_value="")
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_get.return_value = mock_response

            result = await alpha_vantage_client.fetch_articles(symbols=["AAPL"])

            assert result.success is True
            assert len(result.articles) == 1
            assert result.articles[0].sentiment_label == SentimentLabel.NEUTRAL
            assert result.articles[0].sentiment_score == 0.0

    def test_format_ticker_equity(self, alpha_vantage_client):
        """Test that equity symbols are formatted correctly."""
        assert alpha_vantage_client._format_ticker("AAPL") == "AAPL"
        assert alpha_vantage_client._format_ticker("msft") == "MSFT"

    def test_format_ticker_crypto(self, alpha_vantage_client):
        """Test that crypto symbols get CRYPTO: prefix."""
        assert alpha_vantage_client._format_ticker("BTC") == "CRYPTO:BTC"
        assert alpha_vantage_client._format_ticker("eth") == "CRYPTO:ETH"
        assert alpha_vantage_client._format_ticker("SOL") == "CRYPTO:SOL"

    def test_map_sentiment_label(self, alpha_vantage_client):
        """Test sentiment label mapping."""
        assert alpha_vantage_client._map_sentiment_label("Bullish") == SentimentLabel.POSITIVE
        assert alpha_vantage_client._map_sentiment_label("Somewhat-Bullish") == SentimentLabel.POSITIVE
        assert alpha_vantage_client._map_sentiment_label("Neutral") == SentimentLabel.NEUTRAL
        assert alpha_vantage_client._map_sentiment_label("Somewhat-Bearish") == SentimentLabel.NEGATIVE
        assert alpha_vantage_client._map_sentiment_label("Bearish") == SentimentLabel.NEGATIVE
        assert alpha_vantage_client._map_sentiment_label("Unknown") == SentimentLabel.NEUTRAL

    @pytest.mark.asyncio
    async def test_parse_articles_extracts_symbols(self, alpha_vantage_client, sample_alphavantage_response):
        """Test that symbols are extracted from ticker_sentiment."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=sample_alphavantage_response)
            mock_response.text = AsyncMock(return_value="")
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_get.return_value = mock_response

            result = await alpha_vantage_client.fetch_articles(symbols=["AAPL"])

            assert result.success is True
            # First article should have AAPL in symbols
            assert "AAPL" in result.articles[0].symbols

    @pytest.mark.asyncio
    async def test_parse_articles_time_format(self, alpha_vantage_client, sample_alphavantage_response):
        """Test that time_published is parsed correctly."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=sample_alphavantage_response)
            mock_response.text = AsyncMock(return_value="")
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_get.return_value = mock_response

            result = await alpha_vantage_client.fetch_articles(symbols=["AAPL"])

            assert result.success is True
            # Verify published_at is a datetime
            assert isinstance(result.articles[0].published_at, datetime)
            # Should parse to 2024-01-15 12:00:00
            assert result.articles[0].published_at.year == 2024
            assert result.articles[0].published_at.month == 1
            assert result.articles[0].published_at.day == 15

    @pytest.mark.asyncio
    async def test_multiple_symbols_batching(self, alpha_vantage_client, sample_alphavantage_response):
        """Test that multiple symbols are processed in batches."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=sample_alphavantage_response)
            mock_response.text = AsyncMock(return_value="")
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_get.return_value = mock_response

            # Request 10 symbols (should batch into groups of 5)
            result = await alpha_vantage_client.fetch_articles(
                symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "JNJ"]
            )

            # Should call API multiple times (batched)
            assert mock_get.call_count >= 2

    def test_missing_aiohttp_raises_error(self):
        """Test that missing aiohttp raises ImportError."""
        with patch("trading_system.data_pipeline.sources.news.alpha_vantage_news.aiohttp", None):
            with pytest.raises(ImportError, match="aiohttp is required"):
                AlphaVantageNewsClient(api_key="test_key")

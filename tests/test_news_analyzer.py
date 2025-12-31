"""Tests for news analyzer orchestrator."""

import pytest
from datetime import datetime, date
from unittest.mock import AsyncMock, MagicMock, patch

from trading_system.data_pipeline.sources.news.models import NewsArticle, SentimentLabel, NewsFetchResult
from trading_system.research.news_analyzer import NewsAnalyzer, SymbolNewsSummary, NewsAnalysisResult
from trading_system.research.config import ResearchConfig, SentimentConfig


class TestNewsAnalyzer:
    """Tests for NewsAnalyzer class."""

    @pytest.fixture
    def config(self):
        """Create a test ResearchConfig."""
        return ResearchConfig(lookback_hours=48, max_articles_per_symbol=10, sentiment=SentimentConfig())

    @pytest.fixture
    def analyzer(self, config):
        """Create a NewsAnalyzer instance."""
        return NewsAnalyzer(config=config)

    def test_analyzer_initialization(self, analyzer, config):
        """Test that analyzer initializes correctly."""
        assert analyzer is not None
        assert analyzer.config == config
        assert analyzer.sentiment_analyzer is not None
        assert analyzer.ticker_extractor is not None

    def test_analyzer_initialization_with_keys(self, config):
        """Test analyzer initialization with API keys."""
        analyzer = NewsAnalyzer(config=config, newsapi_key="test-newsapi-key", alpha_vantage_key="test-av-key")
        assert analyzer is not None
        assert len(analyzer.news_aggregator.sources) == 2

    def test_analyzer_initialization_with_config_keys(self):
        """Test analyzer uses keys from config if not provided."""
        config = ResearchConfig(newsapi_key="config-newsapi-key", alpha_vantage_key="config-av-key")
        analyzer = NewsAnalyzer(config=config)
        assert analyzer is not None
        assert len(analyzer.news_aggregator.sources) == 2

    @pytest.mark.asyncio
    async def test_analyze_symbols_empty_result(self, analyzer):
        """Test analyzing symbols with no articles."""
        # Mock the news aggregator to return empty result
        mock_result = NewsFetchResult(articles=[], source="test", symbols_requested=["AAPL"], success=True)
        analyzer.news_aggregator.fetch_articles = AsyncMock(return_value=mock_result)

        result = await analyzer.analyze_symbols(["AAPL"])

        assert result.symbols_analyzed == ["AAPL"]
        assert result.total_articles == 0
        assert len(result.articles) == 0
        assert "AAPL" in result.symbol_summaries
        summary = result.symbol_summaries["AAPL"]
        assert summary.article_count == 0
        assert summary.avg_sentiment == 0.0
        assert summary.sentiment_label == SentimentLabel.NEUTRAL

    @pytest.mark.asyncio
    async def test_analyze_symbols_with_articles(self, analyzer):
        """Test analyzing symbols with articles."""
        # Create mock articles
        articles = [
            NewsArticle(
                id="1",
                source="Test",
                title="Apple stock surges on earnings",
                summary="Apple reported strong earnings",
                published_at=datetime.now(),
                url="http://test.com/1",
            ),
            NewsArticle(
                id="2",
                source="Test",
                title="Microsoft announces new product",
                summary="Microsoft unveiled new technology",
                published_at=datetime.now(),
                url="http://test.com/2",
            ),
        ]

        mock_result = NewsFetchResult(articles=articles, source="test", symbols_requested=["AAPL", "MSFT"], success=True)
        analyzer.news_aggregator.fetch_articles = AsyncMock(return_value=mock_result)

        result = await analyzer.analyze_symbols(["AAPL", "MSFT"])

        assert len(result.symbols_analyzed) == 2
        assert result.total_articles == 2
        assert len(result.articles) == 2
        assert "AAPL" in result.symbol_summaries
        assert "MSFT" in result.symbol_summaries

    @pytest.mark.asyncio
    async def test_analyze_symbols_custom_lookback(self, analyzer):
        """Test analyzing with custom lookback hours."""
        mock_result = NewsFetchResult(articles=[], source="test", symbols_requested=["AAPL"], success=True)
        analyzer.news_aggregator.fetch_articles = AsyncMock(return_value=mock_result)

        await analyzer.analyze_symbols(["AAPL"], lookback_hours=24)

        # Verify fetch_articles was called with custom lookback
        analyzer.news_aggregator.fetch_articles.assert_called_once()
        call_args = analyzer.news_aggregator.fetch_articles.call_args
        assert call_args[1]["lookback_hours"] == 24

    def test_process_article_extracts_symbols(self, analyzer):
        """Test that _process_article extracts symbols."""
        article = NewsArticle(
            id="1", source="Test", title="Apple stock surges", summary="Apple reported earnings", published_at=datetime.now()
        )

        processed = analyzer._process_article(article, ["AAPL", "MSFT"])

        assert "AAPL" in processed.symbols
        assert processed.sentiment_score is not None
        assert processed.sentiment_label is not None
        assert processed.is_processed is True

    def test_process_article_filters_symbols(self, analyzer):
        """Test that _process_article only keeps target symbols."""
        article = NewsArticle(
            id="1",
            source="Test",
            title="Apple and Bitcoin both rose",
            summary="AAPL and BTC surged",
            published_at=datetime.now(),
        )

        processed = analyzer._process_article(article, ["AAPL"])

        assert "AAPL" in processed.symbols
        assert "BTC" not in processed.symbols  # Not in target list

    def test_process_article_skips_already_processed(self, analyzer):
        """Test that _process_article skips already processed articles."""
        article = NewsArticle(
            id="1",
            source="Test",
            title="Apple stock",
            summary="Test",
            published_at=datetime.now(),
            sentiment_score=0.5,
            sentiment_label=SentimentLabel.POSITIVE,
            is_processed=True,
        )

        processed = analyzer._process_article(article, ["AAPL"])

        # Should not re-analyze sentiment, but may extract symbols
        assert processed.sentiment_score == 0.5
        assert processed.sentiment_label == SentimentLabel.POSITIVE

    def test_generate_symbol_summary_empty(self, analyzer):
        """Test generating summary for symbol with no articles."""
        summary = analyzer._generate_symbol_summary("AAPL", [])

        assert summary.symbol == "AAPL"
        assert summary.article_count == 0
        assert summary.avg_sentiment == 0.0
        assert summary.sentiment_label == SentimentLabel.NEUTRAL
        assert summary.positive_count == 0
        assert summary.negative_count == 0
        assert summary.neutral_count == 0
        assert len(summary.top_headlines) == 0
        assert summary.sentiment_trend == "stable"

    def test_generate_symbol_summary_with_articles(self, analyzer):
        """Test generating summary with articles."""
        articles = [
            NewsArticle(
                id="1",
                source="Test",
                title="Positive news",
                summary="Good earnings",
                published_at=datetime.now(),
                symbols=["AAPL"],
                sentiment_score=0.7,
                sentiment_label=SentimentLabel.VERY_POSITIVE,
                is_processed=True,
            ),
            NewsArticle(
                id="2",
                source="Test",
                title="More positive news",
                summary="Strong growth",
                published_at=datetime.now(),
                symbols=["AAPL"],
                sentiment_score=0.5,
                sentiment_label=SentimentLabel.POSITIVE,
                is_processed=True,
            ),
        ]

        summary = analyzer._generate_symbol_summary("AAPL", articles)

        assert summary.symbol == "AAPL"
        assert summary.article_count == 2
        assert summary.avg_sentiment > 0
        assert summary.positive_count == 2
        assert len(summary.top_headlines) > 0

    def test_calculate_market_sentiment_empty(self, analyzer):
        """Test calculating market sentiment with no articles."""
        score, label = analyzer._calculate_market_sentiment([])

        assert score == 0.0
        assert label == SentimentLabel.NEUTRAL

    def test_calculate_market_sentiment_positive(self, analyzer):
        """Test calculating positive market sentiment."""
        articles = [
            NewsArticle(
                id="1",
                source="Test",
                title="Good news",
                published_at=datetime.now(),
                sentiment_score=0.5,
                sentiment_label=SentimentLabel.POSITIVE,
                is_processed=True,
            ),
            NewsArticle(
                id="2",
                source="Test",
                title="More good news",
                published_at=datetime.now(),
                sentiment_score=0.3,
                sentiment_label=SentimentLabel.POSITIVE,
                is_processed=True,
            ),
        ]

        score, label = analyzer._calculate_market_sentiment(articles)

        assert score > 0
        assert label == SentimentLabel.POSITIVE

    def test_get_news_score_no_news(self, analyzer):
        """Test getting news score when no news available."""
        analysis = NewsAnalysisResult(
            analysis_date=date.today(),
            symbols_analyzed=["AAPL"],
            symbol_summaries={},
            market_sentiment=0.0,
            market_sentiment_label=SentimentLabel.NEUTRAL,
            total_articles=0,
            articles=[],
        )

        score, reasoning = analyzer.get_news_score_for_signal("AAPL", analysis)

        assert score == 5.0
        assert "No recent news" in reasoning

    def test_get_news_score_positive(self, analyzer):
        """Test getting news score for positive sentiment."""
        summary = SymbolNewsSummary(
            symbol="AAPL",
            article_count=5,
            avg_sentiment=0.6,
            sentiment_label=SentimentLabel.VERY_POSITIVE,
            positive_count=4,
            negative_count=0,
            neutral_count=1,
            top_headlines=["Headline 1", "Headline 2"],
            sentiment_trend="improving",
        )

        analysis = NewsAnalysisResult(
            analysis_date=date.today(),
            symbols_analyzed=["AAPL"],
            symbol_summaries={"AAPL": summary},
            market_sentiment=0.5,
            market_sentiment_label=SentimentLabel.POSITIVE,
            total_articles=5,
            articles=[],
        )

        score, reasoning = analyzer.get_news_score_for_signal("AAPL", analysis)

        assert score > 5.0  # Should be above neutral
        assert score <= 10.0
        assert "positive" in reasoning.lower()

    def test_get_news_score_negative(self, analyzer):
        """Test getting news score for negative sentiment."""
        summary = SymbolNewsSummary(
            symbol="AAPL",
            article_count=3,
            avg_sentiment=-0.4,
            sentiment_label=SentimentLabel.NEGATIVE,
            positive_count=0,
            negative_count=2,
            neutral_count=1,
            top_headlines=["Headline 1"],
            sentiment_trend="declining",
        )

        analysis = NewsAnalysisResult(
            analysis_date=date.today(),
            symbols_analyzed=["AAPL"],
            symbol_summaries={"AAPL": summary},
            market_sentiment=-0.3,
            market_sentiment_label=SentimentLabel.NEGATIVE,
            total_articles=3,
            articles=[],
        )

        score, reasoning = analyzer.get_news_score_for_signal("AAPL", analysis)

        assert score < 5.0  # Should be below neutral
        assert score >= 0.0
        assert "negative" in reasoning.lower()

    def test_generate_reasoning(self, analyzer):
        """Test generating human-readable reasoning."""
        summary = SymbolNewsSummary(
            symbol="AAPL",
            article_count=5,
            avg_sentiment=0.5,
            sentiment_label=SentimentLabel.POSITIVE,
            positive_count=4,
            negative_count=0,
            neutral_count=1,
            top_headlines=["Headline 1"],
            sentiment_trend="improving",
        )

        reasoning = analyzer._generate_reasoning(summary)

        assert "Positive news sentiment" in reasoning
        assert "(5 articles)" in reasoning
        assert "improving trend" in reasoning

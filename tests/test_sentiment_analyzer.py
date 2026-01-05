"""Tests for sentiment analysis module."""

import pytest

from trading_system.data_pipeline.sources.news.models import SentimentLabel
from trading_system.research.sentiment.financial_lexicon import NEGATIVE_TERMS, POSITIVE_TERMS, get_financial_lexicon
from trading_system.research.sentiment.vader_analyzer import VADERSentimentAnalyzer


class TestFinancialLexicon:
    """Tests for financial lexicon."""

    def test_lexicon_has_50_plus_terms(self):
        """Test that lexicon has at least 50 terms."""
        lexicon = get_financial_lexicon()
        assert len(lexicon) >= 50

    def test_positive_terms_exist(self):
        """Test that positive terms are included."""
        assert len(POSITIVE_TERMS) > 0
        assert "bullish" in POSITIVE_TERMS
        assert "surge" in POSITIVE_TERMS
        assert POSITIVE_TERMS["bullish"] > 0

    def test_negative_terms_exist(self):
        """Test that negative terms are included."""
        assert len(NEGATIVE_TERMS) > 0
        assert "bearish" in NEGATIVE_TERMS
        assert "crash" in NEGATIVE_TERMS
        assert NEGATIVE_TERMS["bearish"] < 0

    def test_get_financial_lexicon_combines_terms(self):
        """Test that get_financial_lexicon combines positive and negative terms."""
        lexicon = get_financial_lexicon()
        assert "bullish" in lexicon
        assert "bearish" in lexicon
        assert lexicon["bullish"] == POSITIVE_TERMS["bullish"]
        assert lexicon["bearish"] == NEGATIVE_TERMS["bearish"]


class TestVADERSentimentAnalyzer:
    """Tests for VADER sentiment analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create a VADER analyzer instance."""
        try:
            return VADERSentimentAnalyzer()
        except ImportError as e:
            pytest.skip(f"Required dependency not available: {e}")

    def test_analyzer_initialization(self, analyzer):
        """Test that analyzer initializes correctly."""
        assert analyzer is not None
        assert analyzer.positive_threshold == 0.05
        assert analyzer.negative_threshold == -0.05

    def test_analyzer_custom_thresholds(self):
        """Test analyzer with custom thresholds."""
        try:
            analyzer = VADERSentimentAnalyzer(positive_threshold=0.1, negative_threshold=-0.1)
            assert analyzer.positive_threshold == 0.1
            assert analyzer.negative_threshold == -0.1
        except ImportError as e:
            pytest.skip(f"Required dependency not available: {e}")

    def test_empty_text_returns_neutral(self, analyzer):
        """Test that empty text returns neutral sentiment."""
        score, label, confidence = analyzer.analyze("")
        assert label == SentimentLabel.NEUTRAL
        assert score == 0.0
        assert confidence == 0.0

    def test_whitespace_only_returns_neutral(self, analyzer):
        """Test that whitespace-only text returns neutral sentiment."""
        score, label, confidence = analyzer.analyze("   ")
        assert label == SentimentLabel.NEUTRAL
        assert score == 0.0
        assert confidence == 0.0

    def test_positive_sentiment(self, analyzer):
        """Test positive sentiment detection."""
        score, label, conf = analyzer.analyze("Apple stock surges 10% on record earnings")
        assert label in [SentimentLabel.POSITIVE, SentimentLabel.VERY_POSITIVE]
        assert score > 0.3
        assert 0.0 <= conf <= 1.0

    def test_negative_sentiment(self, analyzer):
        """Test negative sentiment detection."""
        score, label, conf = analyzer.analyze("Tech stocks plunge amid recession fears")
        assert label in [SentimentLabel.NEGATIVE, SentimentLabel.VERY_NEGATIVE]
        assert score < -0.3
        assert 0.0 <= conf <= 1.0

    def test_very_positive_sentiment(self, analyzer):
        """Test very positive sentiment (high compound score)."""
        score, label, conf = analyzer.analyze("Stock skyrockets to all-time high after beating expectations")
        assert label == SentimentLabel.VERY_POSITIVE
        assert score >= 0.5
        assert 0.0 <= conf <= 1.0

    def test_very_negative_sentiment(self, analyzer):
        """Test very negative sentiment (low compound score)."""
        score, label, conf = analyzer.analyze("Company faces bankruptcy after fraud scandal")
        assert label == SentimentLabel.VERY_NEGATIVE
        assert score <= -0.5
        assert 0.0 <= conf <= 1.0

    def test_neutral_sentiment(self, analyzer):
        """Test neutral sentiment detection."""
        score, label, conf = analyzer.analyze("The market opened at 9:30 AM today.")
        assert label == SentimentLabel.NEUTRAL
        assert -0.05 <= score <= 0.05
        assert 0.0 <= conf <= 1.0

    def test_financial_terms_recognized(self, analyzer):
        """Test that financial-specific terms are recognized."""
        # Test bullish term
        score1, label1, _ = analyzer.analyze("Market shows bullish momentum")
        assert score1 > 0

        # Test bearish term
        score2, label2, _ = analyzer.analyze("Market shows bearish sentiment")
        assert score2 < 0

    def test_confidence_scores_reasonable(self, analyzer):
        """Test that confidence scores are in valid range."""
        test_texts = [
            "Stock surges on earnings beat",
            "Market crashes on bad news",
            "The company reported quarterly results",
        ]

        for text in test_texts:
            score, label, confidence = analyzer.analyze(text)
            assert 0.0 <= confidence <= 1.0

    def test_analyze_batch(self, analyzer):
        """Test batch analysis of multiple texts."""
        texts = [
            "Stock surges on earnings",
            "Market crashes",
            "Neutral market news",
        ]
        results = analyzer.analyze_batch(texts)

        assert len(results) == 3
        for score, label, confidence in results:
            assert isinstance(score, float)
            assert isinstance(label, SentimentLabel)
            assert 0.0 <= confidence <= 1.0

    def test_analyze_article(self, analyzer):
        """Test analyzing a NewsArticle object."""
        from datetime import datetime

        from trading_system.data_pipeline.sources.news.models import NewsArticle

        article = NewsArticle(
            id="test-1",
            source="Test",
            title="Stock surges on earnings beat",
            summary="Company beats expectations",
            published_at=datetime.now(),
        )

        result = analyzer.analyze_article(article)

        assert result.sentiment_score is not None
        assert result.sentiment_label is not None
        assert result.sentiment_confidence is not None
        assert result.is_processed is True
        assert result.sentiment_score > 0  # Should be positive

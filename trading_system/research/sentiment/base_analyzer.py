"""Abstract base class for sentiment analyzers."""

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

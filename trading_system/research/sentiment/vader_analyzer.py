"""VADER sentiment analyzer implementation."""

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

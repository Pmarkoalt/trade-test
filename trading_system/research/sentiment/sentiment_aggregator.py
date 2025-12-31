"""Sentiment aggregator for multiple articles."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from trading_system.data_pipeline.sources.news.models import NewsArticle, SentimentLabel

logger = logging.getLogger(__name__)


@dataclass
class AggregatedSentiment:
    """Aggregated sentiment for a symbol."""

    symbol: str
    score: float  # -1 to 1
    confidence: float  # 0 to 1
    label: SentimentLabel
    article_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    avg_relevance: float
    last_updated: datetime

    @property
    def sentiment_ratio(self) -> float:
        """Ratio of positive to negative articles."""
        if self.negative_count == 0:
            return float("inf") if self.positive_count > 0 else 1.0
        return self.positive_count / self.negative_count


@dataclass
class AggregationConfig:
    """Configuration for sentiment aggregation."""

    # Time decay settings
    use_time_decay: bool = True
    decay_half_life_hours: float = 24.0

    # Relevance weighting
    use_relevance_weighting: bool = True
    min_relevance_threshold: float = 0.3

    # Source weighting (premium sources get higher weight)
    source_weights: Optional[Dict[str, float]] = None

    # Minimum articles required for high confidence
    min_articles_for_high_confidence: int = 5

    # Recency bias (more recent articles get higher weight)
    recency_bias: float = 1.5

    def __post_init__(self):
        if self.source_weights is None:
            self.source_weights = {
                "reuters": 1.2,
                "bloomberg": 1.2,
                "wsj": 1.1,
                "financial times": 1.1,
                "cnbc": 1.0,
                "marketwatch": 1.0,
                "yahoo finance": 0.9,
                "default": 1.0,
            }


class SentimentAggregator:
    """Aggregate sentiment from multiple news articles for symbols."""

    def __init__(self, config: Optional[AggregationConfig] = None):
        """Initialize aggregator.

        Args:
            config: Aggregation configuration
        """
        self.config = config or AggregationConfig()

    def aggregate_for_symbol(
        self,
        articles: List[NewsArticle],
        symbol: str,
        as_of: Optional[datetime] = None,
    ) -> AggregatedSentiment:
        """Aggregate sentiment for a single symbol.

        Args:
            articles: List of news articles
            symbol: Symbol to aggregate for
            as_of: Reference time for decay calculations (default: now)

        Returns:
            AggregatedSentiment for the symbol
        """
        if as_of is None:
            as_of = datetime.utcnow()

        # Filter articles for this symbol
        symbol_articles = [a for a in articles if symbol.upper() in [s.upper() for s in a.symbols]]

        if not symbol_articles:
            return AggregatedSentiment(
                symbol=symbol,
                score=0.0,
                confidence=0.0,
                label=SentimentLabel.NEUTRAL,
                article_count=0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                avg_relevance=0.0,
                last_updated=as_of,
            )

        # Calculate weighted sentiment
        weighted_scores = []
        total_weight = 0.0
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        relevance_sum = 0.0

        for article in symbol_articles:
            weight = self._calculate_article_weight(article, as_of)

            # Skip low relevance articles
            if self.config.use_relevance_weighting:
                if article.relevance_score is None or article.relevance_score < self.config.min_relevance_threshold:
                    continue

            if article.sentiment_score is None:
                continue

            weighted_scores.append((article.sentiment_score, weight))
            total_weight += weight
            if article.relevance_score is not None:
                relevance_sum += article.relevance_score

            # Count by category
            if article.sentiment_score > 0.1:
                positive_count += 1
            elif article.sentiment_score < -0.1:
                negative_count += 1
            else:
                neutral_count += 1

        if total_weight == 0:
            return AggregatedSentiment(
                symbol=symbol,
                score=0.0,
                confidence=0.0,
                label=SentimentLabel.NEUTRAL,
                article_count=len(symbol_articles),
                positive_count=positive_count,
                negative_count=negative_count,
                neutral_count=neutral_count,
                avg_relevance=0.0,
                last_updated=as_of,
            )

        # Calculate weighted average
        weighted_sum = sum(score * weight for score, weight in weighted_scores if score is not None)
        aggregated_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Calculate confidence
        scores_list = [score for score, _ in weighted_scores if score is not None]
        confidence = self._calculate_confidence(
            len(scores_list),
            scores_list,
            total_weight,
        )

        # Determine label
        label = self._score_to_label(aggregated_score)

        return AggregatedSentiment(
            symbol=symbol,
            score=aggregated_score,
            confidence=confidence,
            label=label,
            article_count=len(weighted_scores),
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            avg_relevance=relevance_sum / len(symbol_articles) if symbol_articles else 0.0,
            last_updated=as_of,
        )

    def aggregate_for_symbols(
        self,
        articles: List[NewsArticle],
        symbols: List[str],
        as_of: Optional[datetime] = None,
    ) -> Dict[str, AggregatedSentiment]:
        """Aggregate sentiment for multiple symbols.

        Args:
            articles: List of all news articles
            symbols: List of symbols to aggregate for
            as_of: Reference time for decay calculations

        Returns:
            Dictionary mapping symbols to AggregatedSentiment
        """
        return {symbol: self.aggregate_for_symbol(articles, symbol, as_of) for symbol in symbols}

    def get_market_sentiment(
        self,
        articles: List[NewsArticle],
        as_of: Optional[datetime] = None,
    ) -> Tuple[float, float]:
        """Calculate overall market sentiment.

        Args:
            articles: List of all news articles
            as_of: Reference time

        Returns:
            Tuple of (sentiment_score, confidence)
        """
        if as_of is None:
            as_of = datetime.utcnow()

        if not articles:
            return 0.0, 0.0

        weighted_scores = []
        total_weight = 0.0

        for article in articles:
            if article.sentiment_score is None:
                continue
            weight = self._calculate_article_weight(article, as_of)
            weighted_scores.append((article.sentiment_score, weight))
            total_weight += weight

        if total_weight == 0:
            return 0.0, 0.0

        weighted_sum = sum(score * weight for score, weight in weighted_scores if score is not None)
        score = weighted_sum / total_weight if total_weight > 0 else 0.0

        scores_list = [s for s, _ in weighted_scores if s is not None]
        confidence = self._calculate_confidence(
            len(scores_list),
            scores_list,
            total_weight,
        )

        return score, confidence

    def _calculate_article_weight(
        self,
        article: NewsArticle,
        as_of: datetime,
    ) -> float:
        """Calculate weight for an article.

        Args:
            article: News article
            as_of: Reference time

        Returns:
            Article weight
        """
        weight = 1.0

        # Apply time decay
        if self.config.use_time_decay and article.published_at is not None:
            hours_old = (as_of - article.published_at).total_seconds() / 3600
            decay = 0.5 ** (hours_old / self.config.decay_half_life_hours)
            weight *= decay

        # Apply relevance weighting
        if self.config.use_relevance_weighting and article.relevance_score is not None:
            weight *= article.relevance_score

        # Apply source weighting
        if self.config.source_weights:
            source_lower = article.source.lower()
            source_weight = self.config.source_weights.get(
                source_lower,
                self.config.source_weights.get("default", 1.0),
            )
            weight *= source_weight

        # Apply recency bias
        if self.config.recency_bias != 1.0 and article.published_at is not None:
            hours_old = max(1, (as_of - article.published_at).total_seconds() / 3600)
            recency_factor = 1.0 / (hours_old ** (self.config.recency_bias - 1))
            weight *= min(recency_factor, 2.0)  # Cap at 2x

        return weight

    def _calculate_confidence(
        self,
        article_count: int,
        scores: List[float],
        total_weight: float,
    ) -> float:
        """Calculate confidence in aggregated sentiment.

        Args:
            article_count: Number of articles
            scores: List of sentiment scores
            total_weight: Total weight of all articles

        Returns:
            Confidence score (0 to 1)
        """
        if article_count == 0:
            return 0.0

        # Base confidence from article count
        count_factor = min(article_count / self.config.min_articles_for_high_confidence, 1.0)

        # Consistency factor (lower variance = higher confidence)
        if len(scores) > 1:
            mean = sum(scores) / len(scores)
            variance = sum((s - mean) ** 2 for s in scores) / len(scores)
            std_dev = variance**0.5
            # Convert std_dev to consistency (0 std = 1.0 consistency, 2 std = 0 consistency)
            consistency_factor = max(0, 1 - std_dev / 2)
        else:
            consistency_factor = 0.5  # Neutral if only one article

        # Combine factors
        confidence = count_factor * 0.6 + consistency_factor * 0.4

        return float(min(confidence, 1.0))

    def _score_to_label(self, score: float) -> SentimentLabel:
        """Convert numeric score to sentiment label.

        Args:
            score: Sentiment score (-1 to 1)

        Returns:
            SentimentLabel
        """
        if score >= 0.5:
            return SentimentLabel.VERY_POSITIVE
        elif score >= 0.1:
            return SentimentLabel.POSITIVE
        elif score <= -0.5:
            return SentimentLabel.VERY_NEGATIVE
        elif score <= -0.1:
            return SentimentLabel.NEGATIVE
        else:
            return SentimentLabel.NEUTRAL

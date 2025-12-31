"""Relevance scorer for news articles."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import re
import logging

from trading_system.data_pipeline.sources.news.models import NewsArticle

logger = logging.getLogger(__name__)


@dataclass
class RelevanceConfig:
    """Configuration for relevance scoring."""

    # Weights for different relevance factors
    symbol_mention_weight: float = 0.35
    keyword_weight: float = 0.25
    recency_weight: float = 0.15
    source_quality_weight: float = 0.15
    title_mention_weight: float = 0.10

    # Keywords that indicate trading relevance
    trading_keywords: Set[str] = None

    # Keywords that indicate low relevance (noise)
    noise_keywords: Set[str] = None

    # Premium sources get higher scores
    premium_sources: Set[str] = None

    # Minimum relevance score to consider
    min_relevance_threshold: float = 0.3

    def __post_init__(self):
        if self.trading_keywords is None:
            self.trading_keywords = {
                # Price action
                "price", "stock", "shares", "trading", "market",
                "rally", "surge", "plunge", "drop", "fall", "rise",
                "gain", "loss", "bullish", "bearish",
                # Financial metrics
                "earnings", "revenue", "profit", "loss", "eps",
                "guidance", "forecast", "outlook", "beat", "miss",
                # Corporate actions
                "acquisition", "merger", "buyout", "ipo", "spinoff",
                "dividend", "buyback", "split",
                # Analyst activity
                "upgrade", "downgrade", "rating", "target", "analyst",
                "recommendation", "overweight", "underweight",
                # Regulatory/Legal
                "sec", "fda", "approval", "lawsuit", "investigation",
                "regulation", "compliance",
                # Market dynamics
                "volume", "volatility", "momentum", "breakout",
                "support", "resistance", "trend",
            }

        if self.noise_keywords is None:
            self.noise_keywords = {
                # Non-trading content
                "sponsored", "advertisement", "opinion", "editorial",
                "podcast", "video", "webinar", "newsletter",
                # Social/Entertainment
                "celebrity", "entertainment", "sports", "lifestyle",
                # Generic noise
                "click here", "subscribe", "sign up", "free trial",
            }

        if self.premium_sources is None:
            self.premium_sources = {
                "reuters", "bloomberg", "wsj", "wall street journal",
                "financial times", "ft", "barrons", "cnbc",
                "marketwatch", "seeking alpha", "benzinga",
            }


class RelevanceScorer:
    """Score news articles for trading relevance."""

    def __init__(self, config: Optional[RelevanceConfig] = None):
        """Initialize relevance scorer.

        Args:
            config: Scoring configuration
        """
        self.config = config or RelevanceConfig()

        # Compile regex patterns for efficiency
        self._keyword_pattern = self._compile_keyword_pattern(self.config.trading_keywords)
        self._noise_pattern = self._compile_keyword_pattern(self.config.noise_keywords)

    def _compile_keyword_pattern(self, keywords: Set[str]) -> re.Pattern:
        """Compile keywords into regex pattern.

        Args:
            keywords: Set of keywords

        Returns:
            Compiled regex pattern
        """
        escaped = [re.escape(kw) for kw in keywords]
        pattern = r'\b(' + '|'.join(escaped) + r')\b'
        return re.compile(pattern, re.IGNORECASE)

    def score_article(
        self,
        article: NewsArticle,
        target_symbols: Optional[List[str]] = None,
    ) -> float:
        """Score a single article for relevance.

        Args:
            article: News article to score
            target_symbols: Optional list of symbols we're interested in

        Returns:
            Relevance score (0 to 1)
        """
        scores = {}

        # Symbol mention score
        scores["symbol"] = self._score_symbol_mentions(article, target_symbols)

        # Keyword relevance score
        scores["keyword"] = self._score_keywords(article)

        # Source quality score
        scores["source"] = self._score_source_quality(article)

        # Title mention score (symbols in title are more relevant)
        scores["title"] = self._score_title_mentions(article, target_symbols)

        # Apply noise penalty
        noise_penalty = self._calculate_noise_penalty(article)

        # Calculate weighted score
        weighted_score = (
            self.config.symbol_mention_weight * scores["symbol"]
            + self.config.keyword_weight * scores["keyword"]
            + self.config.source_quality_weight * scores["source"]
            + self.config.title_mention_weight * scores["title"]
        )

        # Apply noise penalty
        final_score = weighted_score * (1 - noise_penalty)

        return min(max(final_score, 0.0), 1.0)

    def score_articles(
        self,
        articles: List[NewsArticle],
        target_symbols: Optional[List[str]] = None,
    ) -> List[float]:
        """Score multiple articles.

        Args:
            articles: List of articles to score
            target_symbols: Optional list of target symbols

        Returns:
            List of relevance scores
        """
        return [self.score_article(article, target_symbols) for article in articles]

    def filter_relevant(
        self,
        articles: List[NewsArticle],
        target_symbols: Optional[List[str]] = None,
        min_score: Optional[float] = None,
    ) -> List[NewsArticle]:
        """Filter articles by relevance score.

        Args:
            articles: List of articles
            target_symbols: Optional target symbols
            min_score: Minimum relevance score (uses config default if None)

        Returns:
            Filtered list of relevant articles
        """
        threshold = min_score or self.config.min_relevance_threshold
        return [
            article for article in articles
            if self.score_article(article, target_symbols) >= threshold
        ]

    def rank_by_relevance(
        self,
        articles: List[NewsArticle],
        target_symbols: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[tuple]:
        """Rank articles by relevance score.

        Args:
            articles: List of articles
            target_symbols: Optional target symbols
            top_k: Return only top K articles

        Returns:
            List of (article, score) tuples sorted by relevance
        """
        scored = [
            (article, self.score_article(article, target_symbols))
            for article in articles
        ]
        sorted_articles = sorted(scored, key=lambda x: x[1], reverse=True)

        if top_k:
            return sorted_articles[:top_k]
        return sorted_articles

    def _score_symbol_mentions(
        self,
        article: NewsArticle,
        target_symbols: Optional[List[str]] = None,
    ) -> float:
        """Score based on symbol mentions.

        Args:
            article: News article
            target_symbols: Target symbols

        Returns:
            Score (0 to 1)
        """
        if not article.symbols:
            return 0.0

        # If no target symbols, any symbol mention is good
        if not target_symbols:
            # More symbols = more relevant (up to a point)
            return min(len(article.symbols) / 3, 1.0)

        # Check overlap with target symbols
        target_set = set(s.upper() for s in target_symbols)
        article_set = set(s.upper() for s in article.symbols)

        matches = len(target_set & article_set)
        if matches == 0:
            return 0.0

        # Score based on match ratio
        return min(matches / len(target_symbols), 1.0)

    def _score_keywords(self, article: NewsArticle) -> float:
        """Score based on trading keyword presence.

        Args:
            article: News article

        Returns:
            Score (0 to 1)
        """
        text = f"{article.title} {article.content or article.summary or ''}"
        text_lower = text.lower()

        # Count keyword matches
        matches = len(self._keyword_pattern.findall(text_lower))

        # Normalize by text length (keywords per 100 words)
        word_count = len(text.split())
        if word_count == 0:
            return 0.0

        keyword_density = (matches / word_count) * 100

        # Score based on density (optimal around 2-5%)
        if keyword_density < 1:
            return keyword_density / 2
        elif keyword_density <= 5:
            return min(keyword_density / 5, 1.0)
        else:
            # Too many keywords might be spam
            return max(1.0 - (keyword_density - 5) / 10, 0.5)

    def _score_source_quality(self, article: NewsArticle) -> float:
        """Score based on source quality.

        Args:
            article: News article

        Returns:
            Score (0 to 1)
        """
        source_lower = article.source.lower()

        # Premium sources get high scores
        for premium in self.config.premium_sources:
            if premium in source_lower:
                return 1.0

        # Unknown sources get medium score
        return 0.5

    def _score_title_mentions(
        self,
        article: NewsArticle,
        target_symbols: Optional[List[str]] = None,
    ) -> float:
        """Score based on symbol mentions in title.

        Args:
            article: News article
            target_symbols: Target symbols

        Returns:
            Score (0 to 1)
        """
        if not target_symbols:
            # Check if any symbols in title
            title_upper = article.title.upper()
            for symbol in article.symbols:
                if symbol.upper() in title_upper:
                    return 1.0
            return 0.0

        # Check for target symbols in title
        title_upper = article.title.upper()
        for symbol in target_symbols:
            if symbol.upper() in title_upper:
                return 1.0

        return 0.0

    def _calculate_noise_penalty(self, article: NewsArticle) -> float:
        """Calculate penalty for noise keywords.

        Args:
            article: News article

        Returns:
            Penalty (0 to 1, higher = more penalty)
        """
        text = f"{article.title} {article.content or article.summary or ''}"
        text_lower = text.lower()

        # Count noise matches
        matches = len(self._noise_pattern.findall(text_lower))

        # Penalty increases with noise keywords
        return min(matches * 0.1, 0.5)  # Max 50% penalty

"""Main orchestrator for news analysis."""

from dataclasses import dataclass
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple
import logging

from trading_system.data_pipeline.sources.news import (
    NewsAggregator,
    NewsArticle,
    NewsFetchResult,
    SentimentLabel,
    NewsAPIClient,
    AlphaVantageNewsClient,
)
from .entity_extraction.ticker_extractor import TickerExtractor
from .config import ResearchConfig

# Import VADER analyzer conditionally (optional dependency)
try:
    from .sentiment.vader_analyzer import VADERSentimentAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    VADERSentimentAnalyzer = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class SymbolNewsSummary:
    """Summary of news for a single symbol."""

    symbol: str
    article_count: int
    avg_sentiment: float
    sentiment_label: SentimentLabel
    positive_count: int
    negative_count: int
    neutral_count: int
    top_headlines: List[str]
    sentiment_trend: str  # 'improving', 'declining', 'stable'
    most_recent_article: Optional[datetime] = None


@dataclass
class NewsAnalysisResult:
    """Complete news analysis result."""

    analysis_date: date
    symbols_analyzed: List[str]
    symbol_summaries: Dict[str, SymbolNewsSummary]
    market_sentiment: float  # Overall market sentiment
    market_sentiment_label: SentimentLabel
    total_articles: int
    articles: List[NewsArticle]  # All processed articles


class NewsAnalyzer:
    """Main news analysis orchestrator."""

    def __init__(
        self,
        config: ResearchConfig,
        newsapi_key: Optional[str] = None,
        alpha_vantage_key: Optional[str] = None
    ):
        """Initialize news analyzer.

        Args:
            config: Research configuration
            newsapi_key: NewsAPI.org API key (falls back to config.newsapi_key)
            alpha_vantage_key: Alpha Vantage API key (falls back to config.alpha_vantage_key)
        """
        self.config = config

        # Use provided keys or fall back to config
        newsapi_key = newsapi_key or config.newsapi_key
        alpha_vantage_key = alpha_vantage_key or config.alpha_vantage_key

        # Build news sources list
        sources = []
        if newsapi_key:
            sources.append(NewsAPIClient(api_key=newsapi_key))
        if alpha_vantage_key:
            sources.append(AlphaVantageNewsClient(api_key=alpha_vantage_key))

        # Initialize components
        if sources:
            self.news_aggregator = NewsAggregator(sources=sources)
        else:
            # Create empty aggregator if no keys provided
            self.news_aggregator = NewsAggregator(sources=[])

        # Initialize sentiment analyzer (optional dependency)
        if not VADER_AVAILABLE or VADERSentimentAnalyzer is None:
            raise ImportError(
                "vaderSentiment is required for NewsAnalyzer. "
                "Install it with: pip install vaderSentiment"
            )
        self.sentiment_analyzer = VADERSentimentAnalyzer(
            positive_threshold=config.sentiment.vader_threshold_positive,
            negative_threshold=config.sentiment.vader_threshold_negative
        )
        self.ticker_extractor = TickerExtractor()

        logger.info("NewsAnalyzer initialized")

    async def analyze_symbols(
        self,
        symbols: List[str],
        lookback_hours: int = None
    ) -> NewsAnalysisResult:
        """Analyze news for given symbols.

        Args:
            symbols: List of symbols to analyze
            lookback_hours: How far back to search (default from config)

        Returns:
            Complete analysis result
        """
        lookback = lookback_hours or self.config.lookback_hours

        logger.info(f"Analyzing news for {len(symbols)} symbols, {lookback}h lookback")

        # 1. Fetch news
        fetch_result = await self.news_aggregator.fetch_articles(
            symbols=symbols,
            lookback_hours=lookback,
            max_articles_per_symbol=self.config.max_articles_per_symbol
        )

        articles = fetch_result.articles if fetch_result.success else []
        logger.info(f"Fetched {len(articles)} articles")

        # 2. Process each article
        processed_articles = []
        for article in articles:
            processed = self._process_article(article, symbols)
            processed_articles.append(processed)

        # 3. Generate summaries per symbol
        symbol_summaries = {}
        for symbol in symbols:
            summary = self._generate_symbol_summary(symbol, processed_articles)
            symbol_summaries[symbol] = summary

        # 4. Calculate overall market sentiment
        market_sentiment, market_label = self._calculate_market_sentiment(processed_articles)

        return NewsAnalysisResult(
            analysis_date=date.today(),
            symbols_analyzed=symbols,
            symbol_summaries=symbol_summaries,
            market_sentiment=market_sentiment,
            market_sentiment_label=market_label,
            total_articles=len(processed_articles),
            articles=processed_articles
        )

    def _process_article(
        self,
        article: NewsArticle,
        target_symbols: List[str]
    ) -> NewsArticle:
        """Process a single article.

        Args:
            article: Article to process
            target_symbols: Symbols we're interested in

        Returns:
            Processed article with sentiment and symbols
        """
        # Skip if already processed (e.g., Alpha Vantage provides sentiment)
        if article.is_processed and article.sentiment_score is not None:
            # Still need to extract symbols if not done
            if not article.symbols:
                article.symbols = self.ticker_extractor.extract(
                    f"{article.title} {article.summary or ''}"
                )
            return article

        # Extract symbols mentioned
        text = f"{article.title} {article.summary or ''}"
        found_symbols = self.ticker_extractor.extract(text)

        # Only keep symbols we care about
        article.symbols = [s for s in found_symbols if s in target_symbols]

        # Analyze sentiment
        article = self.sentiment_analyzer.analyze_article(article)

        return article

    def _generate_symbol_summary(
        self,
        symbol: str,
        articles: List[NewsArticle]
    ) -> SymbolNewsSummary:
        """Generate summary for a symbol.

        Args:
            symbol: Symbol to summarize
            articles: All processed articles

        Returns:
            Summary for the symbol
        """
        # Filter to articles mentioning this symbol
        symbol_articles = [a for a in articles if symbol in a.symbols]

        if not symbol_articles:
            return SymbolNewsSummary(
                symbol=symbol,
                article_count=0,
                avg_sentiment=0.0,
                sentiment_label=SentimentLabel.NEUTRAL,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                top_headlines=[],
                sentiment_trend="stable"
            )

        # Calculate sentiment stats
        sentiments = [a.sentiment_score for a in symbol_articles if a.sentiment_score is not None]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

        # Count by label
        positive_count = sum(1 for a in symbol_articles
                            if a.sentiment_label in [SentimentLabel.POSITIVE, SentimentLabel.VERY_POSITIVE])
        negative_count = sum(1 for a in symbol_articles
                            if a.sentiment_label in [SentimentLabel.NEGATIVE, SentimentLabel.VERY_NEGATIVE])
        neutral_count = len(symbol_articles) - positive_count - negative_count

        # Determine overall label
        if avg_sentiment >= 0.3:
            sentiment_label = SentimentLabel.VERY_POSITIVE
        elif avg_sentiment >= 0.05:
            sentiment_label = SentimentLabel.POSITIVE
        elif avg_sentiment <= -0.3:
            sentiment_label = SentimentLabel.VERY_NEGATIVE
        elif avg_sentiment <= -0.05:
            sentiment_label = SentimentLabel.NEGATIVE
        else:
            sentiment_label = SentimentLabel.NEUTRAL

        # Get top headlines (most recent, highest absolute sentiment)
        sorted_articles = sorted(
            symbol_articles,
            key=lambda a: (abs(a.sentiment_score or 0), a.published_at or datetime.min),
            reverse=True
        )
        top_headlines = [a.title for a in sorted_articles[:3]]

        # Calculate trend (compare first half vs second half)
        mid = len(symbol_articles) // 2
        if mid > 0:
            old_sentiment = sum(a.sentiment_score or 0 for a in symbol_articles[mid:]) / (len(symbol_articles) - mid)
            new_sentiment = sum(a.sentiment_score or 0 for a in symbol_articles[:mid]) / mid
            if new_sentiment > old_sentiment + 0.1:
                trend = "improving"
            elif new_sentiment < old_sentiment - 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Get most recent article date
        most_recent = None
        if symbol_articles:
            dates = [a.published_at for a in symbol_articles if a.published_at]
            if dates:
                most_recent = max(dates)

        return SymbolNewsSummary(
            symbol=symbol,
            article_count=len(symbol_articles),
            avg_sentiment=avg_sentiment,
            sentiment_label=sentiment_label,
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            top_headlines=top_headlines,
            sentiment_trend=trend,
            most_recent_article=most_recent
        )

    def _calculate_market_sentiment(
        self,
        articles: List[NewsArticle]
    ) -> Tuple[float, SentimentLabel]:
        """Calculate overall market sentiment.

        Args:
            articles: All processed articles

        Returns:
            Tuple of (sentiment score, label)
        """
        sentiments = [a.sentiment_score for a in articles if a.sentiment_score is not None]

        if not sentiments:
            return 0.0, SentimentLabel.NEUTRAL

        avg = sum(sentiments) / len(sentiments)

        if avg >= 0.2:
            label = SentimentLabel.POSITIVE
        elif avg <= -0.2:
            label = SentimentLabel.NEGATIVE
        else:
            label = SentimentLabel.NEUTRAL

        return avg, label

    def get_news_score_for_signal(
        self,
        symbol: str,
        analysis: NewsAnalysisResult
    ) -> Tuple[float, str]:
        """Get news score for use in signal scoring.

        Args:
            symbol: Symbol to get score for
            analysis: News analysis result

        Returns:
            Tuple of (score 0-10, reasoning string)
        """
        summary = analysis.symbol_summaries.get(symbol)

        if not summary or summary.article_count == 0:
            return 5.0, "No recent news"  # Neutral score

        # Convert sentiment (-1 to +1) to score (0 to 10)
        # sentiment of 0 = 5, sentiment of +1 = 10, sentiment of -1 = 0
        base_score = (summary.avg_sentiment + 1) * 5

        # Boost/penalize based on article count (more coverage = more confidence)
        coverage_mult = min(summary.article_count / 5, 1.5)  # Max 1.5x for 5+ articles
        adjusted_score = base_score * (0.7 + 0.3 * coverage_mult)

        # Boost for improving trend
        if summary.sentiment_trend == "improving":
            adjusted_score = min(adjusted_score * 1.1, 10)
        elif summary.sentiment_trend == "declining":
            adjusted_score = adjusted_score * 0.9

        # Generate reasoning
        reasoning = self._generate_reasoning(summary)

        return round(adjusted_score, 1), reasoning

    def _generate_reasoning(self, summary: SymbolNewsSummary) -> str:
        """Generate human-readable reasoning."""
        parts = []

        # Sentiment
        if summary.sentiment_label == SentimentLabel.VERY_POSITIVE:
            parts.append("Very positive news sentiment")
        elif summary.sentiment_label == SentimentLabel.POSITIVE:
            parts.append("Positive news sentiment")
        elif summary.sentiment_label == SentimentLabel.VERY_NEGATIVE:
            parts.append("Very negative news sentiment")
        elif summary.sentiment_label == SentimentLabel.NEGATIVE:
            parts.append("Negative news sentiment")
        else:
            parts.append("Neutral news sentiment")

        # Coverage
        parts.append(f"({summary.article_count} articles)")

        # Trend
        if summary.sentiment_trend == "improving":
            parts.append("with improving trend")
        elif summary.sentiment_trend == "declining":
            parts.append("with declining trend")

        return " ".join(parts)

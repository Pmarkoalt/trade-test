"""Sentiment analysis modules."""

from .base_analyzer import BaseSentimentAnalyzer
from .financial_lexicon import get_financial_lexicon, POSITIVE_TERMS, NEGATIVE_TERMS
from .sentiment_aggregator import AggregatedSentiment, AggregationConfig, SentimentAggregator
from .synthetic_generator import SentimentConfig, SentimentMode, SyntheticSentimentGenerator, generate_sentiment_for_backtest

__all__ = [
    "BaseSentimentAnalyzer",
    "get_financial_lexicon",
    "POSITIVE_TERMS",
    "NEGATIVE_TERMS",
    "SentimentAggregator",
    "AggregatedSentiment",
    "AggregationConfig",
    "SyntheticSentimentGenerator",
    "SentimentConfig",
    "SentimentMode",
    "generate_sentiment_for_backtest",
]

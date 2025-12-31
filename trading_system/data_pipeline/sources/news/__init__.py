"""News data source implementations."""

from .alpha_vantage_news import AlphaVantageNewsClient
from .base_news_source import BaseNewsSource
from .models import NewsArticle, NewsFetchResult, SentimentLabel
from .news_aggregator import NewsAggregator
from .newsapi_client import NewsAPIClient

__all__ = [
    "NewsArticle",
    "NewsFetchResult",
    "SentimentLabel",
    "BaseNewsSource",
    "NewsAPIClient",
    "AlphaVantageNewsClient",
    "NewsAggregator",
]


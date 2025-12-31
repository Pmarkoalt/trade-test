"""Configuration models for research module."""

from typing import Optional

from pydantic import BaseModel


class SentimentConfig(BaseModel):
    """Sentiment analysis configuration."""

    use_vader: bool = True
    use_finbert: bool = False  # Phase 4 - requires GPU
    vader_threshold_positive: float = 0.05
    vader_threshold_negative: float = -0.05
    min_confidence: float = 0.5


class RelevanceConfig(BaseModel):
    """Relevance scoring configuration."""

    min_relevance_score: float = 0.3
    title_weight: float = 0.6
    summary_weight: float = 0.4


class ResearchConfig(BaseModel):
    """Overall research module configuration."""

    enabled: bool = True
    newsapi_key: Optional[str] = None
    alpha_vantage_key: Optional[str] = None
    lookback_hours: int = 48
    max_articles_per_symbol: int = 10

    sentiment: SentimentConfig = SentimentConfig()
    relevance: RelevanceConfig = RelevanceConfig()

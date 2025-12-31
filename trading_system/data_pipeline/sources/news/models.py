"""Data models for news articles and news fetch results."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


class SentimentLabel(str, Enum):
    """Sentiment classification."""

    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


@dataclass
class NewsArticle:
    """A news article with metadata."""

    # Core fields
    id: str
    source: str  # e.g., "Reuters", "Bloomberg"
    title: str
    summary: Optional[str] = None
    content: Optional[str] = None  # Full article text (if available)
    url: str = ""
    published_at: Optional[datetime] = None
    fetched_at: datetime = field(default_factory=datetime.now)

    # Extracted data (populated by analyzers)
    symbols: List[str] = field(default_factory=list)  # Mentioned tickers
    asset_classes: List[str] = field(default_factory=list)  # 'equity', 'crypto'

    # Sentiment (populated by sentiment analyzer)
    sentiment_score: Optional[float] = None  # -1.0 to +1.0
    sentiment_label: Optional[SentimentLabel] = None
    sentiment_confidence: Optional[float] = None  # 0.0 to 1.0

    # Relevance (populated by relevance scorer)
    relevance_score: Optional[float] = None  # 0.0 to 1.0
    event_type: Optional[str] = None  # 'earnings', 'merger', 'product', etc.

    # Processing flags
    is_processed: bool = False
    processing_error: Optional[str] = None

    def __post_init__(self):
        if self.published_at is None:
            self.published_at = datetime.now()


@dataclass
class NewsFetchResult:
    """Result of a news fetch operation."""

    articles: List[NewsArticle]
    source: str
    symbols_requested: List[str]
    fetch_time: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: Optional[str] = None
    rate_limit_remaining: Optional[int] = None


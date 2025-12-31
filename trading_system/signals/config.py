"""Configuration for signals module."""

from pydantic import BaseModel


class SignalConfig(BaseModel):
    """Configuration for signal generation."""

    max_recommendations: int = 5
    min_conviction: str = "MEDIUM"  # Minimum to include

    # Scoring weights
    technical_weight: float = 0.6
    news_weight: float = 0.4

    # News integration
    news_enabled: bool = True
    news_lookback_hours: int = 48
    min_news_score_for_boost: float = 7.0  # Score above this boosts conviction
    max_news_score_for_penalty: float = 3.0  # Score below this penalizes


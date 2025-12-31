"""Configuration for signals module."""

from pydantic import BaseModel


class SignalConfig(BaseModel):
    """Configuration for signal generation."""

    max_recommendations: int = 5
    min_conviction: str = "MEDIUM"  # Minimum to include
    technical_weight: float = 1.0
    news_weight: float = 0.0  # Phase 1: technical only


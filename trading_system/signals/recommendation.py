"""Recommendation dataclass for trading signals."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class Recommendation:
    """A trading recommendation to deliver to user."""

    id: str
    symbol: str
    asset_class: str  # 'equity' or 'crypto'
    direction: str  # 'BUY' or 'SELL'
    conviction: str  # 'HIGH', 'MEDIUM', 'LOW'

    # Prices
    current_price: float
    entry_price: float  # Expected fill price (next open)
    target_price: float
    stop_price: float

    # Sizing
    position_size_pct: float
    risk_pct: float

    # Scores
    technical_score: float

    # Context
    signal_type: str  # 'breakout_20d', 'breakout_55d', etc.
    reasoning: str

    # Optional scores
    news_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    combined_score: float = 0.0

    # Optional metadata
    news_headlines: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)
    strategy_name: Optional[str] = None


"""OHLCV bar data model."""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class Bar:
    """Single OHLCV bar for a symbol at a specific date."""

    date: pd.Timestamp  # Date/time of the bar
    symbol: str  # Ticker symbol (e.g., "AAPL", "BTC")
    open: float  # Opening price
    high: float  # High price
    low: float  # Low price
    close: float  # Closing price
    volume: float  # Volume (shares or units)
    dollar_volume: Optional[float] = None  # Computed: close * volume

    def __post_init__(self):
        """Validate OHLC relationships and compute dollar_volume if needed."""
        # Validate prices are positive
        if self.close <= 0 or self.open <= 0 or self.high <= 0 or self.low <= 0:
            raise ValueError(f"Invalid prices: {self.symbol} {self.date}, " "prices must be positive")

        # Validate OHLC relationships
        if not (self.low <= self.open <= self.high):
            raise ValueError(
                f"Invalid OHLC: {self.symbol} {self.date}, "
                f"open ({self.open}) must be between low ({self.low}) and high ({self.high})"
            )

        if not (self.low <= self.close <= self.high):
            raise ValueError(
                f"Invalid OHLC: {self.symbol} {self.date}, "
                f"close ({self.close}) must be between low ({self.low}) and high ({self.high})"
            )

        # Validate volume
        if self.volume < 0:
            raise ValueError(f"Negative volume: {self.symbol} {self.date}, volume={self.volume}")

        # Compute dollar_volume if not provided
        if self.dollar_volume is None or self.dollar_volume == 0:
            self.dollar_volume = self.close * self.volume

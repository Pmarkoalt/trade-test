"""Feature row (indicators) data model."""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class FeatureRow:
    """Computed indicators for a symbol at a specific date."""
    
    date: pd.Timestamp
    symbol: str
    asset_class: str  # "equity" | "crypto"
    
    # Price data
    close: float
    open: float
    high: float
    low: float
    
    # Moving averages
    ma20: Optional[float] = None  # NaN until 20 bars available
    ma50: Optional[float] = None  # NaN until 50 bars available
    ma200: Optional[float] = None  # NaN until 200 bars available
    
    # MA50 slope: (MA50[t] / MA50[t-20]) - 1
    ma50_slope: Optional[float] = None  # NaN until 70 bars available (50 for MA50 + 20 for slope)
    
    # Volatility
    atr14: Optional[float] = None  # NaN until 14 bars available
    
    # Momentum
    roc60: Optional[float] = None  # NaN if close[t-60] missing
    
    # Breakout levels
    highest_close_20d: Optional[float] = None  # Highest close over last 20 days (exclusive of today)
    highest_close_55d: Optional[float] = None  # Highest close over last 55 days (exclusive of today)
    
    # Volume
    adv20: Optional[float] = None  # 20-day average dollar volume
    
    # Returns
    returns_1d: Optional[float] = None  # (close[t] / close[t-1]) - 1
    
    # Benchmark data (for relative strength)
    benchmark_roc60: Optional[float] = None  # SPY or BTC ROC60
    benchmark_returns_1d: Optional[float] = None  # SPY or BTC daily return
    
    # Mean reversion indicators
    zscore: Optional[float] = None  # Z-score: (close - MA) / STD, for mean reversion
    ma_lookback: Optional[float] = None  # Rolling mean for mean reversion (configurable lookback)
    std_lookback: Optional[float] = None  # Rolling std for mean reversion
    
    def is_valid_for_entry(self) -> bool:
        """Check if sufficient data exists for signal generation.
        
        Required indicators:
        - ma20, ma50 (for eligibility)
        - atr14 (for stop calculation)
        - highest_close_20d, highest_close_55d (for entry triggers)
        - adv20 (for capacity check)
        
        Returns:
            True if all required indicators are valid (not None and not NaN)
        """
        required = [
            self.ma20,
            self.ma50,
            self.atr14,
            self.highest_close_20d,
            self.highest_close_55d,
            self.adv20,
        ]
        
        return all(
            x is not None and not np.isnan(x) and np.isfinite(x)
            for x in required
        )
    
    def __post_init__(self):
        """Validate asset_class and price data."""
        if self.asset_class not in ["equity", "crypto"]:
            raise ValueError(
                f"Invalid asset_class: {self.asset_class}, must be 'equity' or 'crypto'"
            )
        
        if self.close <= 0 or self.open <= 0 or self.high <= 0 or self.low <= 0:
            raise ValueError(
                f"Invalid prices: {self.symbol} {self.date}, prices must be positive"
            )


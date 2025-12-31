"""Market data container for bars, features, and benchmarks."""

from typing import Dict, Optional
import pandas as pd

from .bar import Bar
from .features import FeatureRow


class MarketData:
    """Container for all market data (bars, features, benchmarks)."""
    
    def __init__(self):
        self.bars: Dict[str, pd.DataFrame] = {}  # symbol -> DataFrame of bars
        self.features: Dict[str, pd.DataFrame] = {}  # symbol -> DataFrame of features
        self.benchmarks: Dict[str, pd.DataFrame] = {}  # "SPY" or "BTC" -> DataFrame
    
    def get_bar(self, symbol: str, date: pd.Timestamp) -> Optional[Bar]:
        """
        Get bar for symbol at date.
        
        Args:
            symbol: Symbol name
            date: Date timestamp
        
        Returns:
            Bar object or None if not found
        """
        if symbol not in self.bars:
            return None
        df = self.bars[symbol]
        if date not in df.index:
            return None
        row = df.loc[date]
        return Bar(
            date=date,
            symbol=symbol,
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume'],
            dollar_volume=row.get('dollar_volume', row['close'] * row['volume'])
        )
    
    def get_features(self, symbol: str, date: pd.Timestamp) -> Optional[FeatureRow]:
        """
        Get features for symbol at date.
        
        Args:
            symbol: Symbol name
            date: Date timestamp
        
        Returns:
            FeatureRow object or None if not found
        """
        if symbol not in self.features:
            return None
        df = self.features[symbol]
        if date not in df.index:
            return None
        row = df.loc[date]
        # Convert Series to dict and add date if not present
        row_dict = row.to_dict()
        if 'date' not in row_dict:
            row_dict['date'] = date
        return FeatureRow(**row_dict)

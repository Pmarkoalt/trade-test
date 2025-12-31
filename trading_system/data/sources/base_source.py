"""Base data source interface for loading OHLCV data."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import pandas as pd


class DataQualityReport(TypedDict, total=False):
    """Data quality report structure."""

    available: bool
    error: str  # Only present when available=False
    row_count: int  # Only present when available=True
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]]  # Only present when available=True
    missing_dates: List[pd.Timestamp]  # Only present when available=True
    missing_dates_count: int  # Only present when available=True
    duplicate_dates: List[pd.Timestamp]  # Only present when available=True
    duplicate_dates_count: int  # Only present when available=True
    negative_prices: int  # Only present when available=True
    zero_volumes: int  # Only present when available=True
    null_values: Dict[str, int]  # Only present when available=True

logger = logging.getLogger(__name__)


class BaseDataSource(ABC):
    """Abstract base class for all data sources.

    All data sources must implement methods to load OHLCV data for symbols
    within a date range. The data is expected to be in a standard format:
    - Index: date (pd.Timestamp)
    - Columns: open, high, low, close, volume, dollar_volume (optional)
    """

    @abstractmethod
    def load_ohlcv(
        self, symbols: List[str], start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None
    ) -> Dict[str, pd.DataFrame]:
        """Load OHLCV data for one or more symbols.

        Args:
            symbols: List of symbols to load
            start_date: Optional start date filter (inclusive)
            end_date: Optional end date filter (inclusive)

        Returns:
            Dictionary mapping symbol -> DataFrame with:
            - Index: date (pd.Timestamp)
            - Columns: open, high, low, close, volume, dollar_volume
        """
        pass

    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols from this source.

        Returns:
            List of symbol strings
        """
        pass

    @abstractmethod
    def get_date_range(self, symbol: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Get available date range for a symbol.

        Args:
            symbol: Symbol to check

        Returns:
            Tuple of (start_date, end_date) or None if symbol not available
        """
        pass

    def supports_incremental(self) -> bool:
        """Check if this source supports incremental loading.

        Returns:
            True if incremental loading is supported
        """
        return False

    def load_incremental(self, symbol: str, last_update_date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Load data incrementally since last update (if supported).

        Args:
            symbol: Symbol to load
            last_update_date: Last known date (exclusive)

        Returns:
            DataFrame with new data or None if not supported/no new data
        """
        if not self.supports_incremental():
            return None

        # Default implementation: load full range and filter
        end_date = pd.Timestamp.now()
        all_data = self.load_ohlcv([symbol], start_date=last_update_date, end_date=end_date)
        if symbol in all_data:
            df = all_data[symbol]
            # Filter out the last_update_date itself (since it's exclusive)
            if len(df) > 0 and df.index[0] <= last_update_date:
                df = df[df.index > last_update_date]
            return df if len(df) > 0 else None
        return None

    def check_data_quality(
        self, symbol: str, start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None
    ) -> DataQualityReport:
        """Check data quality for a symbol.

        Args:
            symbol: Symbol to check
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            Dictionary with quality metrics:
            - missing_dates: List of missing dates
            - duplicate_dates: List of duplicate dates
            - negative_prices: Count of negative prices
            - zero_volumes: Count of zero volumes
            - null_values: Count of null values per column
        """
        data = self.load_ohlcv([symbol], start_date=start_date, end_date=end_date)
        if symbol not in data:
            return {"available": False, "error": "Symbol not found"}

        df = data[symbol]

        # Check for missing dates (gaps)
        missing_dates = []
        if len(df) > 1:
            date_range = pd.date_range(start=df.index[0], end=df.index[-1], freq="D")
            existing_dates = set(df.index)
            missing_dates = [d for d in date_range if d not in existing_dates]

        # Check for duplicate dates
        duplicate_dates = df.index[df.index.duplicated()].tolist() if df.index.has_duplicates else []

        # Check for negative prices
        price_cols = ["open", "high", "low", "close"]
        negative_prices = 0
        for col in price_cols:
            if col in df.columns:
                negative_prices += (df[col] < 0).sum()

        # Check for zero volumes
        zero_volumes = (df["volume"] == 0).sum() if "volume" in df.columns else 0

        # Check for null values
        null_values = df.isnull().sum().to_dict()

        return {
            "available": True,
            "row_count": len(df),
            "date_range": (df.index[0], df.index[-1]) if len(df) > 0 else None,
            "missing_dates": missing_dates,
            "missing_dates_count": len(missing_dates),
            "duplicate_dates": duplicate_dates,
            "duplicate_dates_count": len(duplicate_dates),
            "negative_prices": int(negative_prices),
            "zero_volumes": int(zero_volumes),
            "null_values": null_values,
        }

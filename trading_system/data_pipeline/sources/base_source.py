"""Abstract base class for data sources."""

from abc import ABC, abstractmethod
from datetime import date
from typing import Optional

import pandas as pd

from trading_system.models.bar import Bar


class BaseDataSource(ABC):
    """Abstract base class for all data sources.

    All data source implementations must inherit from this class and
    implement the required methods.
    """

    @abstractmethod
    async def fetch_daily_bars(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch daily OHLCV bars for a symbol over a date range.

        Args:
            symbol: The symbol to fetch data for (e.g., "AAPL", "BTC")
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            DataFrame with columns: date, symbol, open, high, low, close, volume
            Index should be the date column

        Raises:
            DataFetchError: If data fetching fails
            APIRateLimitError: If API rate limit is exceeded
            DataValidationError: If fetched data fails validation
        """
        pass

    @abstractmethod
    async def fetch_latest_bar(self, symbol: str) -> Optional[Bar]:
        """Fetch the latest OHLCV bar for a symbol.

        Args:
            symbol: The symbol to fetch data for (e.g., "AAPL", "BTC")

        Returns:
            Bar object with the latest data, or None if no data is available

        Raises:
            DataFetchError: If data fetching fails
            APIRateLimitError: If API rate limit is exceeded
            DataValidationError: If fetched data fails validation
        """
        pass

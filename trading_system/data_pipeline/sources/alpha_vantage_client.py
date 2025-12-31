"""Alpha Vantage client for fetching equity data."""

from datetime import date
from typing import Optional

import pandas as pd

from trading_system.data_pipeline.sources.base_source import BaseDataSource
from trading_system.models.bar import Bar


class AlphaVantageClient(BaseDataSource):
    """Client for fetching equity data from Alpha Vantage.

    This is a stub implementation that will be expanded in future tasks.
    """

    def __init__(self, api_key: str):
        """Initialize the Alpha Vantage client.

        Args:
            api_key: Alpha Vantage API key
        """
        self.api_key = api_key

    async def fetch_daily_bars(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch daily OHLCV bars for a symbol over a date range.

        Args:
            symbol: The symbol to fetch data for (e.g., "AAPL")
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            DataFrame with columns: date, symbol, open, high, low, close, volume

        Note:
            This is a stub method that will be implemented in future tasks.
        """
        raise NotImplementedError("This method will be implemented in future tasks")

    async def fetch_latest_bar(self, symbol: str) -> Optional[Bar]:
        """Fetch the latest OHLCV bar for a symbol.

        Args:
            symbol: The symbol to fetch data for (e.g., "AAPL")

        Returns:
            Bar object with the latest data, or None if no data is available

        Note:
            This is a stub method that will be implemented in future tasks.
        """
        raise NotImplementedError("This method will be implemented in future tasks")

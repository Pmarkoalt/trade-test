"""API data source implementations (Alpha Vantage, Massive, etc.)."""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ...exceptions import DataSourceError, DataValidationError
from ..validator import validate_ohlcv
from .base_source import BaseDataSource

logger = logging.getLogger(__name__)


class APIDataSource(BaseDataSource):
    """Base class for API data sources.

    Handles rate limiting and API key management.
    """

    def __init__(self, api_key: str, rate_limit_delay: float = 1.0):
        """Initialize API data source.

        Args:
            api_key: API key for authentication
            rate_limit_delay: Delay between API calls (seconds)
        """
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time = 0.0
        self._available_symbols_cache: Optional[List[str]] = None

    def _rate_limit(self):
        """Enforce rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self._last_request_time = time.time()

    def load_ohlcv(
        self, symbols: List[str], start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None
    ) -> Dict[str, pd.DataFrame]:
        """Load OHLCV data from API.

        Args:
            symbols: List of symbols to load
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Dictionary mapping symbol -> DataFrame
        """
        data = {}
        for symbol in symbols:
            try:
                df = self._fetch_symbol_data(symbol, start_date, end_date)
                if df is not None and not df.empty:
                    # Compute dollar_volume if not present
                    if "dollar_volume" not in df.columns:
                        df["dollar_volume"] = df["close"] * df["volume"]

                    # Validate
                    if validate_ohlcv(df, symbol):
                        data[symbol] = df
                    else:
                        logger.error(f"Validation failed for {symbol}, skipping")
                        raise DataValidationError(f"Data validation failed for {symbol}", symbol=symbol)
            except DataValidationError:
                # Re-raise validation errors
                raise
            except (ConnectionError, TimeoutError) as e:
                logger.error(f"Network error loading {symbol} from API: {e}")
                raise DataSourceError(
                    f"Network error loading {symbol}: {e}", symbol=symbol, source_type=self.__class__.__name__
                ) from e
            except ValueError as e:
                logger.error(f"Data format error for {symbol}: {e}")
                raise DataSourceError(
                    f"Data format error for {symbol}: {e}", symbol=symbol, source_type=self.__class__.__name__
                ) from e
            except Exception as e:
                logger.error(f"Unexpected error loading {symbol} from API: {e}", exc_info=True)
                raise DataSourceError(
                    f"Unexpected error loading {symbol}: {e}", symbol=symbol, source_type=self.__class__.__name__
                ) from e

        return data

    def _fetch_symbol_data(
        self, symbol: str, start_date: Optional[pd.Timestamp], end_date: Optional[pd.Timestamp]
    ) -> Optional[pd.DataFrame]:
        """Fetch data for a single symbol (implemented by subclasses)."""
        raise NotImplementedError("Subclass must implement _fetch_symbol_data")

    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols (may be limited by API).

        Returns:
            List of symbols
        """
        # Most APIs don't provide a list of all available symbols
        # Return empty list or cache if available
        if self._available_symbols_cache is None:
            logger.warning("get_available_symbols not fully supported for API sources")
            self._available_symbols_cache = []
        return self._available_symbols_cache.copy()

    def get_date_range(self, symbol: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Get available date range for a symbol.

        Args:
            symbol: Symbol to check

        Returns:
            Tuple of (start_date, end_date) or None if symbol not available
        """
        # Load data to determine range
        data = self.load_ohlcv([symbol])
        if symbol in data:
            df = data[symbol]
            if len(df) > 0:
                return (df.index[0], df.index[-1])
        return None

    def supports_incremental(self) -> bool:
        """API sources support incremental loading via date filtering."""
        return True


class AlphaVantageSource(APIDataSource):
    """Alpha Vantage API data source."""

    def __init__(self, api_key: str, rate_limit_delay: float = 12.0):
        """Initialize Alpha Vantage source.

        Args:
            api_key: Alpha Vantage API key
            rate_limit_delay: Delay between API calls (default 12s for free tier)
        """
        super().__init__(api_key, rate_limit_delay)
        self.base_url = "https://www.alphavantage.co/query"

    def _fetch_symbol_data(
        self, symbol: str, start_date: Optional[pd.Timestamp], end_date: Optional[pd.Timestamp]
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Alpha Vantage API."""
        try:
            import requests  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError("requests module not available. Install with: pip install requests")

        self._rate_limit()

        # Alpha Vantage TIME_SERIES_DAILY endpoint
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": "full",  # Get full historical data
            "datatype": "csv",
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()

            # Parse CSV response
            from io import StringIO

            df = pd.read_csv(StringIO(response.text), parse_dates=["timestamp"], index_col="timestamp")
            df.index.name = "date"

            # Rename columns to match expected format
            df.rename(
                columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}, inplace=True
            )

            df.sort_index(inplace=True)

            # Filter by date range
            if start_date is not None:
                df = df[df.index >= start_date]
            if end_date is not None:
                df = df[df.index <= end_date]

            return df

        except Exception as e:
            logger.error(f"Alpha Vantage API error for {symbol}: {e}")
            return None


class MassiveSource(APIDataSource):
    """Massive API data source (formerly Polygon.io)."""

    def __init__(self, api_key: str, rate_limit_delay: float = 0.1):
        """Initialize Massive source.

        Args:
            api_key: Massive API key
            rate_limit_delay: Delay between API calls
        """
        super().__init__(api_key, rate_limit_delay)
        self.base_url = "https://api.polygon.io/v2"

    def _fetch_symbol_data(
        self, symbol: str, start_date: Optional[pd.Timestamp], end_date: Optional[pd.Timestamp]
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Massive API."""
        try:
            import requests  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError("requests module not available. Install with: pip install requests")

        self._rate_limit()

        # Massive aggregates endpoint (daily bars)
        # Format dates for API
        if start_date is None:
            start_date = pd.Timestamp("2020-01-01")  # Default start
        if end_date is None:
            end_date = pd.Timestamp.now()

        url = f"{self.base_url}/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        params = {"apiKey": self.api_key, "adjusted": "true", "sort": "asc", "limit": 50000}  # Max limit

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if data.get("status") != "OK" or "results" not in data:
                logger.error(f"Massive API error for {symbol}: {data.get('statusMessage', 'Unknown error')}")
                return None

            results = data["results"]
            if not results:
                logger.warning(f"No data returned for {symbol}")
                return None

            # Convert to DataFrame
            records = []
            for bar in results:
                records.append(
                    {
                        "date": pd.Timestamp(bar["t"], unit="ms"),
                        "open": bar["o"],
                        "high": bar["h"],
                        "low": bar["l"],
                        "close": bar["c"],
                        "volume": bar["v"],
                    }
                )

            df = pd.DataFrame(records)
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)

            return df

        except Exception as e:
            logger.error(f"Massive API error for {symbol}: {e}")
            return None

"""Binance client for fetching cryptocurrency data."""

import asyncio
import logging
from collections import deque
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore

from trading_system.data_pipeline.exceptions import APIRateLimitError, DataFetchError, DataValidationError
from trading_system.data_pipeline.sources.base_source import BaseDataSource
from trading_system.models.bar import Bar

logger = logging.getLogger(__name__)

# Symbol mapping: crypto symbols to Binance trading pairs
SYMBOL_MAP = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "BNB": "BNBUSDT",
    "XRP": "XRPUSDT",
    "ADA": "ADAUSDT",
    "SOL": "SOLUSDT",
    "DOT": "DOTUSDT",
    "MATIC": "MATICUSDT",
    "LTC": "LTCUSDT",
    "LINK": "LINKUSDT",
}


class BinanceClient(BaseDataSource):
    """Client for fetching cryptocurrency data from Binance.

    Handles rate limiting, symbol mapping, and error handling.
    Binance public endpoints don't require API keys.
    """

    BASE_URL = "https://api.binance.com/api/v3"

    def __init__(self, rate_limit_per_minute: int = 20, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """Initialize the Binance client.

        Args:
            rate_limit_per_minute: Maximum API calls per minute (default: 20 for public endpoints)
            api_key: Binance API key (optional for public endpoints)
            api_secret: Binance API secret (optional for public endpoints)
        """
        if aiohttp is None:
            raise ImportError("aiohttp is required for BinanceClient. Install with: pip install aiohttp")

        self.api_key = api_key
        self.api_secret = api_secret
        self.rate_limit_per_minute = rate_limit_per_minute
        self.rate_limit_window = 60.0  # 60 seconds window
        self._call_times: deque = deque()  # Track API call timestamps
        self._session: Optional[aiohttp.ClientSession] = None

    def _map_symbol(self, symbol: str) -> str:
        """Map crypto symbol to Binance trading pair.

        Args:
            symbol: Crypto symbol (e.g., "BTC") or already mapped (e.g., "BTCUSDT")

        Returns:
            Binance trading pair (e.g., "BTCUSDT")

        Raises:
            ValueError: If symbol cannot be mapped
        """
        # If already in correct format, return as-is
        if symbol.endswith("USDT"):
            return symbol

        # Map using SYMBOL_MAP
        if symbol in SYMBOL_MAP:
            return SYMBOL_MAP[symbol]

        # Try appending USDT as fallback
        mapped = f"{symbol}USDT"
        logger.warning(f"Symbol {symbol} not in SYMBOL_MAP, using {mapped} as fallback")
        return mapped

    async def _get_session(self) -> "aiohttp.ClientSession":
        """Get or create aiohttp session."""
        if aiohttp is None:
            raise ImportError("aiohttp is required for BinanceClient. Install it with: pip install aiohttp")
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _close_session(self) -> None:
        """Close aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting by tracking API calls and waiting if needed."""
        now = asyncio.get_event_loop().time()

        # Remove old call times outside the window
        while self._call_times and (now - self._call_times[0]) > self.rate_limit_window:
            self._call_times.popleft()

        # If we're at the limit, wait until the oldest call expires
        if len(self._call_times) >= self.rate_limit_per_minute:
            wait_time = self.rate_limit_window - (now - self._call_times[0]) + 0.1  # Add small buffer
            if wait_time > 0:
                logger.debug(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
                # Clean up old calls after waiting
                now = asyncio.get_event_loop().time()
                while self._call_times and (now - self._call_times[0]) > self.rate_limit_window:
                    self._call_times.popleft()

        # Record this API call
        self._call_times.append(asyncio.get_event_loop().time())

    async def _make_request(self, url: str, params: Optional[dict] = None, max_retries: int = 3) -> list:
        """Make an HTTP request with exponential backoff and error handling.

        Args:
            url: Request URL
            params: Query parameters
            max_retries: Maximum number of retry attempts

        Returns:
            JSON response as list (Binance klines format)

        Raises:
            DataFetchError: If request fails after retries
            APIRateLimitError: If rate limit is exceeded (429 status)
        """
        await self._enforce_rate_limit()

        if params is None:
            params = {}

        session = await self._get_session()
        backoff_delay = 1.0

        for attempt in range(max_retries):
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 429:
                        # Rate limit exceeded
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            wait_time = float(retry_after)
                        else:
                            wait_time = backoff_delay * (2**attempt)

                        logger.warning(f"Rate limit exceeded. Waiting {wait_time:.2f} seconds...")
                        await asyncio.sleep(wait_time)
                        raise APIRateLimitError(f"Rate limit exceeded for Binance API")

                    if response.status == 200:
                        data = await response.json()
                        # Binance returns error as JSON object if there's an error
                        if isinstance(data, dict) and "code" in data:
                            error_msg = data.get("msg", "Unknown error")
                            raise DataFetchError(f"Binance API error: {error_msg}")
                        return data

                    elif response.status >= 500:
                        # Server error, retry with exponential backoff
                        if attempt < max_retries - 1:
                            wait_time = backoff_delay * (2**attempt)
                            logger.warning(f"Server error {response.status}. Retrying in {wait_time:.2f} seconds...")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            raise DataFetchError(f"Server error {response.status} after {max_retries} attempts")

                    else:
                        # Client error (4xx), don't retry
                        error_text = await response.text()
                        raise DataFetchError(f"HTTP {response.status}: {error_text}")

            except aiohttp.ClientError as e:
                if attempt < max_retries - 1:
                    wait_time = backoff_delay * (2**attempt)
                    logger.warning(f"Network error: {e}. Retrying in {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise DataFetchError(f"Network error after {max_retries} attempts: {e}") from e

        raise DataFetchError(f"Request failed after {max_retries} attempts")

    def _parse_response(self, data: list, symbol: str) -> pd.DataFrame:
        """Parse Binance klines response into DataFrame.

        Args:
            data: Binance klines response (array of arrays)
            symbol: Original symbol (before mapping)

        Returns:
            DataFrame with columns: date, symbol, open, high, low, close, volume

        Raises:
            DataValidationError: If response format is invalid
        """
        if not data:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume"])

        rows = []
        for kline in data:
            try:
                # Binance klines format: [open_time, open, high, low, close, volume, ...]
                # Times are in milliseconds UTC
                open_time_ms = int(kline[0])
                date_val = pd.Timestamp.fromtimestamp(open_time_ms / 1000.0, tz="UTC").date()

                rows.append(
                    {
                        "date": date_val,
                        "symbol": symbol,  # Use original symbol, not mapped
                        "open": float(kline[1]),
                        "high": float(kline[2]),
                        "low": float(kline[3]),
                        "close": float(kline[4]),
                        "volume": float(kline[5]),
                    }
                )
            except (IndexError, ValueError, TypeError) as e:
                logger.warning(f"Error parsing kline data: {kline}. Error: {e}")
                continue

        if not rows:
            raise DataValidationError(f"No valid klines found in response for {symbol}")

        df = pd.DataFrame(rows)
        df = df.sort_values("date").reset_index(drop=True)

        # Validate data
        if df.empty:
            raise DataValidationError(f"Empty DataFrame after parsing for {symbol}")

        # Check for required columns
        required_cols = ["date", "symbol", "open", "high", "low", "close", "volume"]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise DataValidationError(f"Missing columns in DataFrame: {missing_cols}")

        return df

    async def fetch_daily_bars(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch daily OHLCV bars for a symbol over a date range.

        Args:
            symbol: The symbol to fetch data for (e.g., "BTC" or "BTCUSDT")
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            DataFrame with columns: date, symbol, open, high, low, close, volume
            Index is reset (not set to date)

        Raises:
            DataFetchError: If data fetching fails
            APIRateLimitError: If API rate limit is exceeded
            DataValidationError: If fetched data fails validation
        """
        # Map symbol to Binance trading pair
        trading_pair = self._map_symbol(symbol)

        # Convert dates to UTC timestamps (milliseconds)
        # Binance uses UTC, so we need to convert start_date and end_date to UTC
        start_dt = datetime.combine(start_date, datetime.min.time())
        start_dt_utc = start_dt.replace(tzinfo=None)  # Treat as UTC
        start_time_ms = int(start_dt_utc.timestamp() * 1000)

        # End date should be end of day (23:59:59)
        end_dt = datetime.combine(end_date, datetime.max.time())
        end_dt_utc = end_dt.replace(tzinfo=None)  # Treat as UTC
        end_time_ms = int(end_dt_utc.timestamp() * 1000)

        url = f"{self.BASE_URL}/klines"
        params = {
            "symbol": trading_pair,
            "interval": "1d",  # Daily candles
            "startTime": str(start_time_ms),
            "endTime": str(end_time_ms),
            "limit": "1000",  # Binance max is 1000 per request
        }

        try:
            # Binance may require multiple requests if date range is large
            all_data = []
            current_start = start_time_ms

            while current_start <= end_time_ms:
                params["startTime"] = str(current_start)
                params["endTime"] = str(end_time_ms)

                klines = await self._make_request(url, params)

                if not klines:
                    break

                all_data.extend(klines)

                # If we got less than 1000 results, we've reached the end
                if len(klines) < 1000:
                    break

                # Update start time to next day after last kline
                last_kline_time = int(klines[-1][0])
                current_start = last_kline_time + 86400000  # Add 1 day in milliseconds

            if not all_data:
                logger.warning(f"No data returned for {symbol} from {start_date} to {end_date}")
                return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume"])

            df = self._parse_response(all_data, symbol)

            # Filter to requested date range (API might return slightly different range)
            df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

            logger.info(f"Fetched {len(df)} bars for {symbol} from {start_date} to {end_date}")
            return df

        except (DataFetchError, APIRateLimitError, DataValidationError):
            raise
        except Exception as e:
            raise DataFetchError(f"Unexpected error fetching data for {symbol}: {e}") from e

    async def fetch_latest_bar(self, symbol: str) -> Optional[Bar]:
        """Fetch the most recent bar for a symbol.

        Args:
            symbol: The symbol to fetch data for (e.g., "BTC" or "BTCUSDT")

        Returns:
            Bar object with the latest data, or None if no data is available

        Raises:
            DataFetchError: If data fetching fails
            APIRateLimitError: If API rate limit is exceeded
            DataValidationError: If fetched data fails validation
        """
        # Map symbol to Binance trading pair
        trading_pair = self._map_symbol(symbol)

        # Fetch last 2 days to ensure we get the latest bar
        end_date = date.today()
        start_date = end_date - timedelta(days=2)

        try:
            df = await self.fetch_daily_bars(symbol, start_date, end_date)

            if df.empty:
                logger.warning(f"No data available for {symbol}")
                return None

            # Get the most recent bar
            latest_row = df.iloc[-1]

            try:
                bar = Bar(
                    date=pd.Timestamp(latest_row["date"]),
                    symbol=symbol,  # Use original symbol
                    open=float(latest_row["open"]),
                    high=float(latest_row["high"]),
                    low=float(latest_row["low"]),
                    close=float(latest_row["close"]),
                    volume=float(latest_row["volume"]),
                )
                return bar
            except ValueError as e:
                raise DataValidationError(f"Invalid bar data for {symbol}: {e}") from e

        except (DataFetchError, APIRateLimitError, DataValidationError):
            raise
        except Exception as e:
            raise DataFetchError(f"Unexpected error fetching latest bar for {symbol}: {e}") from e

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - close session."""
        await self._close_session()
        return False

"""Polygon.io client for fetching equity data."""

import asyncio
import logging
from collections import deque
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore

from trading_system.data_pipeline.exceptions import APIRateLimitError, DataFetchError, DataValidationError
from trading_system.data_pipeline.sources.base_source import BaseDataSource
from trading_system.models.bar import Bar

logger = logging.getLogger(__name__)


class PolygonClient(BaseDataSource):
    """Client for fetching equity data from Polygon.io.

    Handles rate limiting, exponential backoff, and error handling.
    """

    BASE_URL = "https://api.polygon.io/v2/aggs/ticker"

    def __init__(self, api_key: str, rate_limit_per_minute: int = 5):
        """Initialize the Polygon.io client.

        Args:
            api_key: Polygon.io API key
            rate_limit_per_minute: Maximum API calls per minute (default: 5 for free tier)
        """
        if aiohttp is None:
            raise ImportError("aiohttp is required for PolygonClient. Install with: pip install aiohttp")

        self.api_key = api_key
        self.rate_limit_per_minute = rate_limit_per_minute
        self.rate_limit_window = 60.0  # 60 seconds window
        self._call_times: deque = deque()  # Track API call timestamps
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> "aiohttp.ClientSession":
        """Get or create aiohttp session."""
        if aiohttp is None:
            raise ImportError("aiohttp is required for PolygonClient. Install it with: pip install aiohttp")
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

    async def _make_request(self, url: str, params: Optional[Dict[str, str]] = None, max_retries: int = 3) -> Dict:
        """Make an HTTP request with exponential backoff and error handling.

        Args:
            url: Request URL
            params: Query parameters
            max_retries: Maximum number of retry attempts

        Returns:
            JSON response as dictionary

        Raises:
            DataFetchError: If request fails after retries
            APIRateLimitError: If rate limit is exceeded (429 status)
        """
        await self._enforce_rate_limit()

        if params is None:
            params = {}
        params["apiKey"] = self.api_key

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
                        raise APIRateLimitError(f"Rate limit exceeded for Polygon.io API")

                    if response.status == 200:
                        data = await response.json()
                        status = data.get("status", "unknown")
                        if status == "OK":
                            return data
                        elif status == "ERROR":
                            error_msg = data.get("error", "Unknown error")
                            raise DataFetchError(f"Polygon.io API error: {error_msg}")
                        else:
                            raise DataFetchError(f"Unexpected API status: {status}")

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

    def _parse_response(self, data: Dict, symbol: str) -> pd.DataFrame:
        """Parse Polygon.io API response into DataFrame.

        Args:
            data: JSON response from API
            symbol: Symbol name

        Returns:
            DataFrame with columns: date, symbol, open, high, low, close, volume

        Raises:
            DataValidationError: If response format is invalid
        """
        results = data.get("results", [])
        if not results:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume"])

        rows = []
        for item in results:
            try:
                # Polygon returns timestamps in milliseconds
                timestamp_ms = item.get("t", 0)
                date_val = pd.Timestamp.fromtimestamp(timestamp_ms / 1000.0, tz="UTC").date()

                rows.append(
                    {
                        "date": date_val,
                        "symbol": symbol,
                        "open": float(item.get("o", 0)),
                        "high": float(item.get("h", 0)),
                        "low": float(item.get("l", 0)),
                        "close": float(item.get("c", 0)),
                        "volume": float(item.get("v", 0)),
                    }
                )
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Error parsing bar data: {item}. Error: {e}")
                continue

        if not rows:
            raise DataValidationError(f"No valid bars found in response for {symbol}")

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
            symbol: The symbol to fetch data for (e.g., "AAPL")
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
        # Format dates as YYYY-MM-DD
        from_str = start_date.strftime("%Y-%m-%d")
        to_str = end_date.strftime("%Y-%m-%d")

        url = f"{self.BASE_URL}/{symbol}/range/1/day/{from_str}/{to_str}"

        try:
            data = await self._make_request(url)
            df = self._parse_response(data, symbol)

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
            symbol: The symbol to fetch data for (e.g., "AAPL")

        Returns:
            Bar object with the latest data, or None if no data is available

        Raises:
            DataFetchError: If data fetching fails
            APIRateLimitError: If API rate limit is exceeded
            DataValidationError: If fetched data fails validation
        """
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
                    symbol=symbol,
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

    async def fetch_multiple_symbols(self, symbols: List[str], start_date: date, end_date: date) -> Dict[str, pd.DataFrame]:
        """Fetch bars for multiple symbols.

        Args:
            symbols: List of symbols to fetch
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            Dictionary mapping symbol to DataFrame with columns: date, symbol, open, high, low, close, volume

        Raises:
            DataFetchError: If data fetching fails for any symbol
            APIRateLimitError: If API rate limit is exceeded
        """
        results: Dict[str, pd.DataFrame] = {}

        for symbol in symbols:
            try:
                df = await self.fetch_daily_bars(symbol, start_date, end_date)
                results[symbol] = df
            except (DataFetchError, APIRateLimitError, DataValidationError) as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                # Continue with other symbols instead of failing completely
                results[symbol] = pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume"])

        return results

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - close session."""
        await self._close_session()
        return False

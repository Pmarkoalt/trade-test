"""Main orchestrator for fetching live data from various sources."""

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional

import pandas as pd

from trading_system.data_pipeline.cache.data_cache import DataCache
from trading_system.data_pipeline.config import DataPipelineConfig
from trading_system.data_pipeline.exceptions import DataFetchError
from trading_system.data_pipeline.sources.base_source import BaseDataSource
from trading_system.data_pipeline.sources.binance_client import BinanceClient
from trading_system.data_pipeline.sources.massive_client import MassiveClient
from trading_system.models.bar import Bar

logger = logging.getLogger(__name__)


class LiveDataFetcher:
    """Orchestrates data fetching from multiple sources with caching.

    Coordinates data fetching from Massive (equities) and Binance (crypto)
    with intelligent caching to minimize API calls.
    """

    def __init__(self, config: DataPipelineConfig):
        """Initialize the live data fetcher.

        Args:
            config: Configuration for the data pipeline
        """
        self.config = config
        self.cache = DataCache(config.cache_path, config.cache_ttl_hours)

        # Initialize data sources
        self.massive: Optional[MassiveClient] = None
        if config.massive_api_key:
            self.massive = MassiveClient(config.massive_api_key)

        self.binance = BinanceClient()

    def _get_source(self, asset_class: str) -> BaseDataSource:
        """Get appropriate data source for asset class.

        Args:
            asset_class: Asset class ('equity' or 'crypto')

        Returns:
            Data source instance

        Raises:
            DataFetchError: If Massive API key is required but not provided
            ValueError: If asset class is unknown
        """
        if asset_class == "equity":
            if not self.massive:
                raise DataFetchError("Massive API key required for equities. Set massive_api_key in config.")
            return self.massive
        elif asset_class == "crypto":
            return self.binance
        else:
            raise ValueError(f"Unknown asset class: {asset_class}. Must be 'equity' or 'crypto'")

    async def fetch_daily_data(
        self,
        symbols: List[str],
        asset_class: str,
        lookback_days: int = 252,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch daily OHLCV data for multiple symbols.

        Uses cache-first strategy: checks cache for each symbol, then fetches
        only missing data from API. Updates cache with newly fetched data.

        Args:
            symbols: List of symbols to fetch data for
            asset_class: Asset class ('equity' or 'crypto')
            lookback_days: Number of days to look back (default: 252 trading days)

        Returns:
            Dictionary mapping symbol to OHLCV DataFrame with columns:
            date, symbol, open, high, low, close, volume

        Raises:
            DataFetchError: If data fetching fails
            ValueError: If asset class is unknown
        """
        logger.info(f"Fetching {len(symbols)} {asset_class} symbols (lookback: {lookback_days} days)")

        # Calculate date range
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)

        # Get data source
        source = self._get_source(asset_class)

        # Track cache hits and API fetches
        cache_hits = 0
        api_fetches = 0
        results: Dict[str, pd.DataFrame] = {}

        # Process each symbol
        for symbol in symbols:
            try:
                # Generate cache key
                cache_key = self.cache.get_cache_key(symbol, asset_class, start_date, end_date)

                # Try cache first
                cached_df = self.cache.get(cache_key)
                if cached_df is not None and len(cached_df) > 0:
                    logger.debug(f"Cache hit for {symbol}")
                    results[symbol] = cached_df
                    cache_hits += 1
                    continue

                # Cache miss - fetch from API
                logger.debug(f"Cache miss for {symbol}, fetching from API")
                api_fetches += 1

                df = await source.fetch_daily_bars(symbol, start_date, end_date)

                if df is not None and len(df) > 0:
                    # Store in cache
                    self.cache.set(cache_key, df)
                    results[symbol] = df
                    logger.debug(f"Fetched and cached {len(df)} bars for {symbol}")
                else:
                    logger.warning(f"No data returned for {symbol}")

            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                # Continue with other symbols instead of failing completely
                continue

        logger.info(
            f"Completed fetching {asset_class} data: "
            f"{len(results)}/{len(symbols)} symbols, "
            f"cache hits: {cache_hits}, API fetches: {api_fetches}"
        )

        return results

    async def fetch_latest_bars(
        self,
        symbols: List[str],
        asset_class: str,
    ) -> Dict[str, Optional[Bar]]:
        """Fetch most recent bar for each symbol.

        Args:
            symbols: List of symbols to fetch data for
            asset_class: Asset class ('equity' or 'crypto')

        Returns:
            Dictionary mapping symbol to Bar object, or None if no data available

        Raises:
            DataFetchError: If data fetching fails
            ValueError: If asset class is unknown
        """
        logger.info(f"Fetching latest bars for {len(symbols)} {asset_class} symbols")

        # Get data source
        source = self._get_source(asset_class)

        results: Dict[str, Optional[Bar]] = {}

        # Process each symbol
        for symbol in symbols:
            try:
                bar = await source.fetch_latest_bar(symbol)
                results[symbol] = bar

                if bar:
                    logger.debug(f"Fetched latest bar for {symbol}: {bar.date}, close={bar.close}")
                else:
                    logger.warning(f"No latest bar available for {symbol}")

            except Exception as e:
                logger.error(f"Failed to fetch latest bar for {symbol}: {e}")
                results[symbol] = None
                # Continue with other symbols
                continue

        successful = sum(1 for bar in results.values() if bar is not None)
        logger.info(f"Fetched latest bars: {successful}/{len(symbols)} successful")

        return results

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - close data source sessions."""
        # Close Massive session if exists
        if self.massive:
            await self.massive._close_session()

        # Close Binance session
        await self.binance._close_session()

        return False

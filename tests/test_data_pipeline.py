"""Integration tests for data pipeline."""

import os
import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from trading_system.data_pipeline.cache.data_cache import DataCache
from trading_system.data_pipeline.config import DataPipelineConfig
from trading_system.data_pipeline.live_data_fetcher import LiveDataFetcher


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def fetcher(temp_cache_dir):
    """Create a LiveDataFetcher instance for testing."""
    polygon_api_key = os.getenv("POLYGON_API_KEY")
    config = DataPipelineConfig(
        polygon_api_key=polygon_api_key,
        cache_path=temp_cache_dir,
        cache_ttl_hours=24,
    )
    return LiveDataFetcher(config)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    dates = [date.today() - timedelta(days=i) for i in range(30, 0, -1)]
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": ["AAPL"] * 30,
            "open": [100.0 + i * 0.5 for i in range(30)],
            "high": [105.0 + i * 0.5 for i in range(30)],
            "low": [99.0 + i * 0.5 for i in range(30)],
            "close": [104.0 + i * 0.5 for i in range(30)],
            "volume": [1000000 + i * 10000 for i in range(30)],
        }
    )


class TestLiveDataFetcherIntegration:
    """Integration tests for LiveDataFetcher (can use real API if keys are available)."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_fetch_equity_data(self, fetcher, sample_dataframe):
        """Test fetching equity data."""
        # Skip if no Polygon API key
        if not fetcher.polygon:
            pytest.skip("POLYGON_API_KEY not set, skipping integration test")

        # Mock the API call to avoid actual API requests in tests
        with patch.object(fetcher.polygon, "fetch_daily_bars", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = sample_dataframe

            data = await fetcher.fetch_daily_data(symbols=["AAPL", "MSFT"], asset_class="equity", lookback_days=30)

            assert "AAPL" in data
            assert len(data["AAPL"]) >= 20

            # Verify API was called
            assert mock_fetch.call_count == 2  # One for each symbol

        await fetcher.__aexit__(None, None, None)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_fetch_crypto_data(self, fetcher, sample_dataframe):
        """Test fetching crypto data."""
        # Mock the API call
        with patch.object(fetcher.binance, "fetch_daily_bars", new_callable=AsyncMock) as mock_fetch:
            # Create crypto-specific dataframe
            crypto_df = sample_dataframe.copy()
            crypto_df["symbol"] = "BTC"

            mock_fetch.return_value = crypto_df

            data = await fetcher.fetch_daily_data(symbols=["BTC", "ETH"], asset_class="crypto", lookback_days=30)

            assert "BTC" in data
            assert len(data["BTC"]) >= 20

            # Verify API was called
            assert mock_fetch.call_count == 2  # One for each symbol

        await fetcher.__aexit__(None, None, None)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_fetch_equity_data_with_cache(self, fetcher, sample_dataframe):
        """Test fetching equity data with cache."""
        if not fetcher.polygon:
            pytest.skip("POLYGON_API_KEY not set, skipping integration test")

        # Pre-populate cache
        cache_key = fetcher.cache.get_cache_key("AAPL", "equity", date.today() - timedelta(days=30), date.today())
        fetcher.cache.set(cache_key, sample_dataframe)

        # Mock API (should not be called due to cache)
        with patch.object(fetcher.polygon, "fetch_daily_bars", new_callable=AsyncMock) as mock_fetch:
            data = await fetcher.fetch_daily_data(symbols=["AAPL"], asset_class="equity", lookback_days=30)

            assert "AAPL" in data
            pd.testing.assert_frame_equal(data["AAPL"], sample_dataframe)

            # API should not be called due to cache hit
            assert mock_fetch.call_count == 0

        await fetcher.__aexit__(None, None, None)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_fetch_crypto_data_with_cache(self, fetcher, sample_dataframe):
        """Test fetching crypto data with cache."""
        # Pre-populate cache
        cache_key = fetcher.cache.get_cache_key("BTC", "crypto", date.today() - timedelta(days=30), date.today())
        crypto_df = sample_dataframe.copy()
        crypto_df["symbol"] = "BTC"
        fetcher.cache.set(cache_key, crypto_df)

        # Mock API (should not be called due to cache)
        with patch.object(fetcher.binance, "fetch_daily_bars", new_callable=AsyncMock) as mock_fetch:
            data = await fetcher.fetch_daily_data(symbols=["BTC"], asset_class="crypto", lookback_days=30)

            assert "BTC" in data
            pd.testing.assert_frame_equal(data["BTC"], crypto_df)

            # API should not be called due to cache hit
            assert mock_fetch.call_count == 0

        await fetcher.__aexit__(None, None, None)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_fetch_multiple_symbols_mixed_cache(self, fetcher, sample_dataframe):
        """Test fetching multiple symbols with mixed cache hits/misses."""
        if not fetcher.polygon:
            pytest.skip("POLYGON_API_KEY not set, skipping integration test")

        # Pre-populate cache for one symbol
        cache_key_aapl = fetcher.cache.get_cache_key("AAPL", "equity", date.today() - timedelta(days=30), date.today())
        fetcher.cache.set(cache_key_aapl, sample_dataframe)

        # Mock API for the other symbol
        msft_df = sample_dataframe.copy()
        msft_df["symbol"] = "MSFT"
        with patch.object(fetcher.polygon, "fetch_daily_bars", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = msft_df

            data = await fetcher.fetch_daily_data(symbols=["AAPL", "MSFT"], asset_class="equity", lookback_days=30)

            assert "AAPL" in data
            assert "MSFT" in data

            # Only MSFT should trigger API call (AAPL from cache)
            assert mock_fetch.call_count == 1

        await fetcher.__aexit__(None, None, None)


class TestLiveDataFetcherUnit:
    """Unit tests for LiveDataFetcher (mocked, no API keys required)."""

    @pytest.mark.asyncio
    async def test_fetch_equity_data_mocked(self, fetcher, sample_dataframe):
        """Test fetching equity data with mocked API (no API key required)."""
        # Create a fetcher with mocked polygon client
        with patch.object(fetcher, "polygon", None):
            # Should raise error without API key
            with pytest.raises(Exception):
                await fetcher.fetch_daily_data(symbols=["AAPL"], asset_class="equity", lookback_days=30)

        # With polygon client mocked
        fetcher.polygon = AsyncMock()
        fetcher.polygon.fetch_daily_bars = AsyncMock(return_value=sample_dataframe)

        data = await fetcher.fetch_daily_data(symbols=["AAPL"], asset_class="equity", lookback_days=30)

        assert "AAPL" in data
        assert len(data["AAPL"]) == 30

        await fetcher.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_fetch_crypto_data_mocked(self, fetcher, sample_dataframe):
        """Test fetching crypto data with mocked API."""
        crypto_df = sample_dataframe.copy()
        crypto_df["symbol"] = "BTC"

        with patch.object(fetcher.binance, "fetch_daily_bars", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = crypto_df

            data = await fetcher.fetch_daily_data(symbols=["BTC"], asset_class="crypto", lookback_days=30)

            assert "BTC" in data
            assert len(data["BTC"]) == 30

        await fetcher.__aexit__(None, None, None)

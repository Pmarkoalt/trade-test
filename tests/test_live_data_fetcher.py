"""Integration tests for LiveDataFetcher."""

import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from trading_system.data_pipeline.config import DataPipelineConfig  # noqa: E402
from trading_system.data_pipeline.exceptions import DataFetchError  # noqa: E402
from trading_system.data_pipeline.live_data_fetcher import LiveDataFetcher  # noqa: E402
from trading_system.models.bar import Bar  # noqa: E402

# Skip if aiohttp is not installed (required by BinanceClient used in LiveDataFetcher)
pytest.importorskip("aiohttp", reason="aiohttp required for LiveDataFetcher tests")


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config_with_massive(temp_cache_dir):
    """Create config with Massive API key."""
    return DataPipelineConfig(
        massive_api_key="test_massive_key",
        cache_path=temp_cache_dir,
        cache_ttl_hours=24,
    )


@pytest.fixture
def config_no_massive(temp_cache_dir):
    """Create config without Massive API key."""
    return DataPipelineConfig(
        massive_api_key=None,
        cache_path=temp_cache_dir,
        cache_ttl_hours=24,
    )


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0],
            "close": [104.0, 105.0, 106.0],
            "volume": [1000000, 1100000, 1200000],
        }
    )


class TestLiveDataFetcher:
    """Test LiveDataFetcher class."""

    @pytest.mark.asyncio
    async def test_fetch_daily_data_crypto_cache_hit(self, config_no_massive, sample_dataframe):
        """Test fetching crypto data with cache hit."""
        fetcher = LiveDataFetcher(config_no_massive)

        # Pre-populate cache
        cache_key = fetcher.cache.get_cache_key("BTC", "crypto", date(2023, 1, 1), date(2023, 1, 3))
        fetcher.cache.set(cache_key, sample_dataframe)

        # Fetch should use cache
        results = await fetcher.fetch_daily_data(["BTC"], "crypto", lookback_days=3)

        assert "BTC" in results
        pd.testing.assert_frame_equal(results["BTC"], sample_dataframe)

        await fetcher.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_fetch_daily_data_crypto_cache_miss(self, config_no_massive, sample_dataframe):
        """Test fetching crypto data with cache miss."""
        fetcher = LiveDataFetcher(config_no_massive)

        with patch.object(fetcher.binance, "fetch_daily_bars", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = sample_dataframe

            results = await fetcher.fetch_daily_data(["BTC"], "crypto", lookback_days=3)

            assert "BTC" in results
            pd.testing.assert_frame_equal(results["BTC"], sample_dataframe)

            # Verify API was called
            mock_fetch.assert_called_once()

            # Verify data was cached
            cache_key = fetcher.cache.get_cache_key("BTC", "crypto", date.today() - timedelta(days=3), date.today())
            cached_df = fetcher.cache.get(cache_key)
            assert cached_df is not None

        await fetcher.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_fetch_daily_data_equity_with_massive(self, config_with_massive, sample_dataframe):
        """Test fetching equity data with Massive API key."""
        fetcher = LiveDataFetcher(config_with_massive)

        assert fetcher.massive is not None

        with patch.object(fetcher.massive, "fetch_daily_bars", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = sample_dataframe

            results = await fetcher.fetch_daily_data(["AAPL"], "equity", lookback_days=3)

            assert "AAPL" in results
            pd.testing.assert_frame_equal(results["AAPL"], sample_dataframe)

            # Verify API was called
            mock_fetch.assert_called_once()

        await fetcher.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_fetch_daily_data_equity_no_massive_key(self, config_no_massive):
        """Test that equity fetching fails without Massive API key."""
        fetcher = LiveDataFetcher(config_no_massive)

        with pytest.raises(DataFetchError, match="Massive API key required"):
            await fetcher.fetch_daily_data(["AAPL"], "equity", lookback_days=3)

        await fetcher.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_fetch_daily_data_partial_cache_hits(self, config_no_massive, sample_dataframe):
        """Test fetching multiple symbols with partial cache hits."""
        fetcher = LiveDataFetcher(config_no_massive)

        # Pre-populate cache for one symbol
        cache_key_btc = fetcher.cache.get_cache_key("BTC", "crypto", date.today() - timedelta(days=3), date.today())
        fetcher.cache.set(cache_key_btc, sample_dataframe)

        # Mock API fetch for the other symbol
        with patch.object(fetcher.binance, "fetch_daily_bars", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = sample_dataframe

            results = await fetcher.fetch_daily_data(["BTC", "ETH"], "crypto", lookback_days=3)

            # Both should be in results
            assert "BTC" in results
            assert "ETH" in results

            # BTC should come from cache (no API call)
            # ETH should come from API (one API call)
            assert mock_fetch.call_count == 1

        await fetcher.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_fetch_latest_bars_crypto(self, config_no_massive):
        """Test fetching latest bars for crypto."""
        fetcher = LiveDataFetcher(config_no_massive)

        sample_bar = Bar(
            date=pd.Timestamp(date.today()),
            symbol="BTC",
            open=100.0,
            high=105.0,
            low=99.0,
            close=104.0,
            volume=1000000,
        )

        with patch.object(fetcher.binance, "fetch_latest_bar", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = sample_bar

            results = await fetcher.fetch_latest_bars(["BTC"], "crypto")

            assert "BTC" in results
            assert results["BTC"] == sample_bar

        await fetcher.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_fetch_latest_bars_equity(self, config_with_massive):
        """Test fetching latest bars for equity."""
        fetcher = LiveDataFetcher(config_with_massive)

        sample_bar = Bar(
            date=pd.Timestamp(date.today()),
            symbol="AAPL",
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=50000000,
        )

        with patch.object(fetcher.massive, "fetch_latest_bar", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = sample_bar

            results = await fetcher.fetch_latest_bars(["AAPL"], "equity")

            assert "AAPL" in results
            assert results["AAPL"] == sample_bar

        await fetcher.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_fetch_latest_bars_no_data(self, config_no_massive):
        """Test fetching latest bars when no data is available."""
        fetcher = LiveDataFetcher(config_no_massive)

        with patch.object(fetcher.binance, "fetch_latest_bar", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = None

            results = await fetcher.fetch_latest_bars(["BTC"], "crypto")

            assert "BTC" in results
            assert results["BTC"] is None

        await fetcher.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_get_source_equity(self, config_with_massive):
        """Test _get_source for equity."""
        fetcher = LiveDataFetcher(config_with_massive)

        source = fetcher._get_source("equity")
        assert source == fetcher.massive

        await fetcher.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_get_source_crypto(self, config_no_massive):
        """Test _get_source for crypto."""
        fetcher = LiveDataFetcher(config_no_massive)

        source = fetcher._get_source("crypto")
        assert source == fetcher.binance

        await fetcher.__aexit__(None, None, None)

    def test_get_source_invalid_asset_class(self, config_no_massive):
        """Test _get_source with invalid asset class."""
        fetcher = LiveDataFetcher(config_no_massive)

        with pytest.raises(ValueError, match="Unknown asset class"):
            fetcher._get_source("invalid")

    @pytest.mark.asyncio
    async def test_fetch_daily_data_error_handling(self, config_no_massive):
        """Test error handling when API fetch fails."""
        fetcher = LiveDataFetcher(config_no_massive)

        with patch.object(fetcher.binance, "fetch_daily_bars", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = DataFetchError("API error")

            # Should not raise, but return empty dict for failed symbol
            results = await fetcher.fetch_daily_data(["BTC"], "crypto", lookback_days=3)

            # Symbol should not be in results due to error
            assert "BTC" not in results

        await fetcher.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_fetch_daily_data_multiple_symbols_partial_failure(self, config_no_massive, sample_dataframe):
        """Test fetching multiple symbols when one fails."""
        fetcher = LiveDataFetcher(config_no_massive)

        call_count = 0

        async def mock_fetch(symbol, start_date, end_date):
            nonlocal call_count
            call_count += 1
            if symbol == "BTC":
                return sample_dataframe
            else:
                raise DataFetchError(f"Failed to fetch {symbol}")

        with patch.object(fetcher.binance, "fetch_daily_bars", side_effect=mock_fetch):
            results = await fetcher.fetch_daily_data(["BTC", "ETH"], "crypto", lookback_days=3)

            # BTC should succeed
            assert "BTC" in results

            # ETH should fail but not crash the whole operation
            assert "ETH" not in results

        await fetcher.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_context_manager(self, config_no_massive):
        """Test async context manager."""
        async with LiveDataFetcher(config_no_massive) as fetcher:
            assert fetcher is not None
            assert fetcher.binance is not None

        # Sessions should be closed after context exit
        assert fetcher.binance._session is None or fetcher.binance._session.closed

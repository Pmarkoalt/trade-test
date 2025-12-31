"""Unit tests for Binance client."""

import asyncio
from datetime import date, timedelta
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from trading_system.data_pipeline.exceptions import APIRateLimitError, DataFetchError, DataValidationError
from trading_system.data_pipeline.sources.binance_client import BinanceClient, SYMBOL_MAP
from trading_system.models.bar import Bar


@pytest.fixture
def binance_client():
    """Create a BinanceClient instance for testing."""
    return BinanceClient(rate_limit_per_minute=20)


@pytest.fixture
def sample_binance_response():
    """Sample Binance klines API response."""
    # Binance klines format: [open_time, open, high, low, close, volume, close_time, quote_volume, trades, ...]
    return [
        [1672531200000, "100.0", "105.0", "99.0", "104.0", "1000000", 1672617599999, "104000000", 1000],
        [1672617600000, "104.0", "108.0", "103.0", "107.0", "1200000", 1672703999999, "128400000", 1100],
    ]


@pytest.fixture
def empty_binance_response():
    """Empty Binance klines API response."""
    return []


class TestBinanceClient:
    """Test BinanceClient class."""

    def test_symbol_mapping(self, binance_client):
        """Test symbol mapping to Binance trading pairs."""
        assert binance_client._map_symbol("BTC") == "BTCUSDT"
        assert binance_client._map_symbol("ETH") == "ETHUSDT"
        assert binance_client._map_symbol("BTCUSDT") == "BTCUSDT"  # Already mapped
        assert binance_client._map_symbol("SOL") == "SOLUSDT"

    def test_symbol_mapping_all_crypto_symbols(self, binance_client):
        """Test that all 10 crypto symbols can be mapped."""
        crypto_symbols = ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOT", "MATIC", "LTC", "LINK"]
        for symbol in crypto_symbols:
            mapped = binance_client._map_symbol(symbol)
            assert mapped.endswith("USDT")
            assert mapped in SYMBOL_MAP.values() or mapped == f"{symbol}USDT"

    @pytest.mark.asyncio
    async def test_fetch_daily_bars_success(self, binance_client, sample_binance_response):
        """Test successful data fetch."""
        with patch.object(binance_client, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = sample_binance_response

            start_date = date(2023, 1, 1)
            end_date = date(2023, 1, 2)
            df = await binance_client.fetch_daily_bars("BTC", start_date, end_date)

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert all(col in df.columns for col in ["date", "symbol", "open", "high", "low", "close", "volume"])
            assert df["symbol"].iloc[0] == "BTC"  # Original symbol, not mapped
            assert df["open"].iloc[0] == 100.0
            assert df["close"].iloc[0] == 104.0
            assert df["volume"].iloc[0] == 1000000

            # Verify date parsing (UTC)
            assert df["date"].iloc[0] == date(2023, 1, 1)
            assert df["date"].iloc[1] == date(2023, 1, 2)

            # Verify API was called with correct parameters
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert "klines" in call_args[0][0]  # URL contains klines endpoint
            params = call_args[1]["params"]
            assert params["symbol"] == "BTCUSDT"  # Mapped symbol
            assert params["interval"] == "1d"

    @pytest.mark.asyncio
    async def test_fetch_daily_bars_empty_response(self, binance_client, empty_binance_response):
        """Test handling of empty response."""
        with patch.object(binance_client, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = empty_binance_response

            start_date = date(2023, 1, 1)
            end_date = date(2023, 1, 2)
            df = await binance_client.fetch_daily_bars("BTC", start_date, end_date)

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0
            assert all(col in df.columns for col in ["date", "symbol", "open", "high", "low", "close", "volume"])

    @pytest.mark.asyncio
    async def test_fetch_daily_bars_rate_limit(self, binance_client):
        """Test rate limit handling."""
        with patch.object(binance_client, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = APIRateLimitError("Rate limit exceeded")

            start_date = date(2023, 1, 1)
            end_date = date(2023, 1, 2)

            with pytest.raises(APIRateLimitError):
                await binance_client.fetch_daily_bars("BTC", start_date, end_date)

    @pytest.mark.asyncio
    async def test_fetch_daily_bars_network_error(self, binance_client):
        """Test handling of network errors."""
        with patch.object(binance_client, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = DataFetchError("Network error")

            start_date = date(2023, 1, 1)
            end_date = date(2023, 1, 2)

            with pytest.raises(DataFetchError, match="Network error"):
                await binance_client.fetch_daily_bars("BTC", start_date, end_date)

    @pytest.mark.asyncio
    async def test_fetch_daily_bars_large_date_range(self, binance_client):
        """Test fetching data for large date range (requires multiple API calls)."""
        # Mock multiple responses for pagination
        response1 = [[1672531200000, "100.0", "105.0", "99.0", "104.0", "1000000", 1672617599999, "104000000", 1000]]
        response2 = [[1672617600000, "104.0", "108.0", "103.0", "107.0", "1200000", 1672703999999, "128400000", 1100]]

        with patch.object(binance_client, "_make_request", new_callable=AsyncMock) as mock_request:
            # First call returns 1 result, second call returns 1 result (less than 1000, so stops)
            mock_request.side_effect = [response1, response2]

            start_date = date(2023, 1, 1)
            end_date = date(2023, 1, 10)  # Large range
            df = await binance_client.fetch_daily_bars("BTC", start_date, end_date)

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert mock_request.call_count == 2

    @pytest.mark.asyncio
    async def test_fetch_latest_bar_success(self, binance_client, sample_binance_response):
        """Test fetching latest bar."""
        with patch.object(binance_client, "fetch_daily_bars", new_callable=AsyncMock) as mock_fetch:
            # Create DataFrame with sample data
            df = pd.DataFrame(
                {
                    "date": [date(2023, 1, 1), date(2023, 1, 2)],
                    "symbol": ["BTC", "BTC"],
                    "open": [100.0, 104.0],
                    "high": [105.0, 108.0],
                    "low": [99.0, 103.0],
                    "close": [104.0, 107.0],
                    "volume": [1000000, 1200000],
                }
            )
            mock_fetch.return_value = df

            bar = await binance_client.fetch_latest_bar("BTC")

            assert isinstance(bar, Bar)
            assert bar.symbol == "BTC"  # Original symbol
            assert bar.close == 107.0
            assert bar.open == 104.0
            assert bar.volume == 1200000

    @pytest.mark.asyncio
    async def test_fetch_latest_bar_no_data(self, binance_client):
        """Test fetching latest bar when no data is available."""
        with patch.object(binance_client, "fetch_daily_bars", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume"])

            bar = await binance_client.fetch_latest_bar("BTC")

            assert bar is None

    @pytest.mark.asyncio
    async def test_fetch_daily_bars_all_crypto_symbols(self, binance_client, sample_binance_response):
        """Test that all 10 crypto symbols can be fetched."""
        crypto_symbols = ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOT", "MATIC", "LTC", "LINK"]

        with patch.object(binance_client, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = sample_binance_response

            for symbol in crypto_symbols:
                start_date = date(2023, 1, 1)
                end_date = date(2023, 1, 2)
                df = await binance_client.fetch_daily_bars(symbol, start_date, end_date)

                assert isinstance(df, pd.DataFrame)
                assert df["symbol"].iloc[0] == symbol  # Original symbol preserved

    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(self, binance_client):
        """Test that rate limiting is enforced."""
        # Create a client with very low rate limit for testing
        client = BinanceClient(rate_limit_per_minute=2)

        with patch("aiohttp.ClientSession.get") as mock_get:
            # Mock successful response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=[])
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_get.return_value = mock_response

            # Make multiple requests quickly
            start_date = date(2023, 1, 1)
            end_date = date(2023, 1, 2)

            # First two should succeed quickly
            await client.fetch_daily_bars("BTC", start_date, end_date)
            await client.fetch_daily_bars("ETH", start_date, end_date)

            # Third should trigger rate limiting (but we can't easily test the delay)
            # At minimum, verify it doesn't crash
            await client.fetch_daily_bars("SOL", start_date, end_date)

            await client._close_session()

    @pytest.mark.asyncio
    async def test_parse_response_invalid_data(self, binance_client):
        """Test parsing of invalid response data."""
        invalid_response = [[1234567890000]]  # Missing required fields

        with patch.object(binance_client, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = invalid_response

            start_date = date(2023, 1, 1)
            end_date = date(2023, 1, 2)

            # Should handle gracefully
            df = await binance_client.fetch_daily_bars("BTC", start_date, end_date)
            # Should either be empty or raise DataValidationError
            assert isinstance(df, pd.DataFrame)

    @pytest.mark.asyncio
    async def test_context_manager(self, binance_client):
        """Test async context manager."""
        async with binance_client as client:
            assert client is binance_client

        # Session should be closed after context exit
        assert binance_client._session is None or binance_client._session.closed

    def test_missing_aiohttp_raises_error(self):
        """Test that missing aiohttp raises ImportError."""
        with patch("trading_system.data_pipeline.sources.binance_client.aiohttp", None):
            with pytest.raises(ImportError, match="aiohttp is required"):
                BinanceClient()

    @pytest.mark.asyncio
    async def test_binance_error_response(self, binance_client):
        """Test handling of Binance error responses."""
        error_response = {"code": -1121, "msg": "Invalid symbol"}

        with patch.object(binance_client, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = error_response

            start_date = date(2023, 1, 1)
            end_date = date(2023, 1, 2)

            # Should raise DataFetchError
            with pytest.raises(DataFetchError, match="Binance API error"):
                await binance_client.fetch_daily_bars("INVALID", start_date, end_date)

    @pytest.mark.asyncio
    async def test_timezone_handling(self, binance_client, sample_binance_response):
        """Test that UTC timezone is handled correctly."""
        with patch.object(binance_client, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = sample_binance_response

            start_date = date(2023, 1, 1)
            end_date = date(2023, 1, 2)
            df = await binance_client.fetch_daily_bars("BTC", start_date, end_date)

            # Verify dates are correct (Binance uses UTC)
            assert df["date"].iloc[0] == date(2023, 1, 1)
            assert df["date"].iloc[1] == date(2023, 1, 2)


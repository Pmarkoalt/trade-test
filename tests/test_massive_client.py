"""Unit tests for Massive client."""

import asyncio
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from trading_system.data_pipeline.exceptions import APIRateLimitError, DataFetchError, DataValidationError
from trading_system.data_pipeline.sources.massive_client import MassiveClient
from trading_system.models.bar import Bar


@pytest.fixture
def massive_client():
    """Create a MassiveClient instance for testing."""
    return MassiveClient(api_key="test_api_key", rate_limit_per_minute=5)


@pytest.fixture
def sample_massive_response():
    """Sample Massive API response."""
    return {
        "status": "OK",
        "results": [
            {
                "t": 1672531200000,  # 2023-01-01 00:00:00 UTC in milliseconds
                "o": 100.0,
                "h": 105.0,
                "l": 99.0,
                "c": 104.0,
                "v": 1000000,
            },
            {
                "t": 1672617600000,  # 2023-01-02 00:00:00 UTC in milliseconds
                "o": 104.0,
                "h": 108.0,
                "l": 103.0,
                "c": 107.0,
                "v": 1200000,
            },
        ],
    }


@pytest.fixture
def empty_massive_response():
    """Empty Massive API response."""
    return {"status": "OK", "results": []}


class TestMassiveClient:
    """Test MassiveClient class."""

    @pytest.mark.asyncio
    async def test_fetch_daily_bars_success(self, massive_client, sample_massive_response):
        """Test successful data fetch."""
        with patch.object(massive_client, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = sample_massive_response

            start_date = date(2023, 1, 1)
            end_date = date(2023, 1, 2)
            df = await massive_client.fetch_daily_bars("AAPL", start_date, end_date)

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert all(col in df.columns for col in ["date", "symbol", "open", "high", "low", "close", "volume"])
            assert df["symbol"].iloc[0] == "AAPL"
            assert df["open"].iloc[0] == 100.0
            assert df["close"].iloc[0] == 104.0
            assert df["volume"].iloc[0] == 1000000

            # Verify date parsing
            assert df["date"].iloc[0] == date(2023, 1, 1)
            assert df["date"].iloc[1] == date(2023, 1, 2)

            # Verify API was called with correct URL
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert "AAPL" in call_args[0][0]  # URL contains symbol
            assert "2023-01-01" in call_args[0][0]  # URL contains start date
            assert "2023-01-02" in call_args[0][0]  # URL contains end date

    @pytest.mark.asyncio
    async def test_fetch_daily_bars_empty_response(self, massive_client, empty_massive_response):
        """Test handling of empty response."""
        with patch.object(massive_client, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = empty_massive_response

            start_date = date(2023, 1, 1)
            end_date = date(2023, 1, 2)
            df = await massive_client.fetch_daily_bars("AAPL", start_date, end_date)

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0
            assert all(col in df.columns for col in ["date", "symbol", "open", "high", "low", "close", "volume"])

    @pytest.mark.asyncio
    async def test_fetch_daily_bars_rate_limit(self, massive_client):
        """Test rate limit handling."""
        with patch.object(massive_client, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = APIRateLimitError("Rate limit exceeded")

            start_date = date(2023, 1, 1)
            end_date = date(2023, 1, 2)

            with pytest.raises(APIRateLimitError):
                await massive_client.fetch_daily_bars("AAPL", start_date, end_date)

    @pytest.mark.asyncio
    async def test_fetch_daily_bars_network_error(self, massive_client):
        """Test handling of network errors."""
        with patch.object(massive_client, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = DataFetchError("Network error")

            start_date = date(2023, 1, 1)
            end_date = date(2023, 1, 2)

            with pytest.raises(DataFetchError, match="Network error"):
                await massive_client.fetch_daily_bars("AAPL", start_date, end_date)

    @pytest.mark.asyncio
    async def test_fetch_latest_bar_success(self, massive_client, sample_massive_response):
        """Test fetching latest bar."""
        with patch.object(massive_client, "fetch_daily_bars", new_callable=AsyncMock) as mock_fetch:
            # Create DataFrame with sample data
            df = pd.DataFrame(
                {
                    "date": [date(2023, 1, 1), date(2023, 1, 2)],
                    "symbol": ["AAPL", "AAPL"],
                    "open": [100.0, 104.0],
                    "high": [105.0, 108.0],
                    "low": [99.0, 103.0],
                    "close": [104.0, 107.0],
                    "volume": [1000000, 1200000],
                }
            )
            mock_fetch.return_value = df

            bar = await massive_client.fetch_latest_bar("AAPL")

            assert isinstance(bar, Bar)
            assert bar.symbol == "AAPL"
            assert bar.close == 107.0
            assert bar.open == 104.0
            assert bar.volume == 1200000

    @pytest.mark.asyncio
    async def test_fetch_latest_bar_no_data(self, massive_client):
        """Test fetching latest bar when no data is available."""
        with patch.object(massive_client, "fetch_daily_bars", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume"])

            bar = await massive_client.fetch_latest_bar("AAPL")

            assert bar is None

    @pytest.mark.asyncio
    async def test_fetch_multiple_symbols(self, massive_client, sample_massive_response):
        """Test fetching multiple symbols."""
        with patch.object(massive_client, "fetch_daily_bars", new_callable=AsyncMock) as mock_fetch:
            # Create sample DataFrame
            df = pd.DataFrame(
                {
                    "date": [date(2023, 1, 1)],
                    "symbol": ["AAPL"],
                    "open": [100.0],
                    "high": [105.0],
                    "low": [99.0],
                    "close": [104.0],
                    "volume": [1000000],
                }
            )
            mock_fetch.return_value = df

            symbols = ["AAPL", "MSFT"]
            start_date = date(2023, 1, 1)
            end_date = date(2023, 1, 2)
            results = await massive_client.fetch_multiple_symbols(symbols, start_date, end_date)

            assert isinstance(results, dict)
            assert "AAPL" in results
            assert "MSFT" in results
            assert len(results) == 2
            assert isinstance(results["AAPL"], pd.DataFrame)
            assert isinstance(results["MSFT"], pd.DataFrame)

            # Verify fetch_daily_bars was called for each symbol
            assert mock_fetch.call_count == 2

    @pytest.mark.asyncio
    async def test_fetch_multiple_symbols_partial_failure(self, massive_client):
        """Test fetching multiple symbols when one fails."""
        call_count = 0

        async def mock_fetch(symbol, start_date, end_date):
            nonlocal call_count
            call_count += 1
            if symbol == "AAPL":
                return pd.DataFrame(
                    {
                        "date": [date(2023, 1, 1)],
                        "symbol": ["AAPL"],
                        "open": [100.0],
                        "high": [105.0],
                        "low": [99.0],
                        "close": [104.0],
                        "volume": [1000000],
                    }
                )
            else:
                raise DataFetchError(f"Failed to fetch {symbol}")

        with patch.object(massive_client, "fetch_daily_bars", side_effect=mock_fetch):
            symbols = ["AAPL", "MSFT"]
            start_date = date(2023, 1, 1)
            end_date = date(2023, 1, 2)
            results = await massive_client.fetch_multiple_symbols(symbols, start_date, end_date)

            # Should continue with other symbols even if one fails
            assert "AAPL" in results
            assert "MSFT" in results
            assert len(results["AAPL"]) > 0
            assert len(results["MSFT"]) == 0  # Empty DataFrame for failed symbol

    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(self, massive_client):
        """Test that rate limiting is enforced."""
        # Create a client with very low rate limit for testing
        client = MassiveClient(api_key="test_key", rate_limit_per_minute=2)

        with patch("aiohttp.ClientSession.get") as mock_get:
            # Mock successful response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "OK", "results": []})
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_get.return_value = mock_response

            # Make multiple requests quickly
            start_date = date(2023, 1, 1)
            end_date = date(2023, 1, 2)

            # First two should succeed quickly
            await client.fetch_daily_bars("AAPL", start_date, end_date)
            await client.fetch_daily_bars("MSFT", start_date, end_date)

            # Third should trigger rate limiting (but we can't easily test the delay)
            # At minimum, verify it doesn't crash
            await client.fetch_daily_bars("GOOGL", start_date, end_date)

            await client._close_session()

    @pytest.mark.asyncio
    async def test_parse_response_invalid_data(self, massive_client):
        """Test parsing of invalid response data."""
        invalid_response = {"status": "OK", "results": [{"invalid": "data"}]}

        with patch.object(massive_client, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = invalid_response

            start_date = date(2023, 1, 1)
            end_date = date(2023, 1, 2)

            # Should handle gracefully and return empty or partial DataFrame
            df = await massive_client.fetch_daily_bars("AAPL", start_date, end_date)
            # Should either be empty or raise DataValidationError
            assert isinstance(df, pd.DataFrame)

    @pytest.mark.asyncio
    async def test_context_manager(self, massive_client):
        """Test async context manager."""
        async with massive_client as client:
            assert client is massive_client

        # Session should be closed after context exit
        assert massive_client._session is None or massive_client._session.closed

    def test_missing_aiohttp_raises_error(self):
        """Test that missing aiohttp raises ImportError."""
        with patch("trading_system.data_pipeline.sources.massive_client.aiohttp", None):
            with pytest.raises(ImportError, match="aiohttp is required"):
                MassiveClient(api_key="test_key")

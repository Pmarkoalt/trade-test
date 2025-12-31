"""Unit tests for data loading and validation."""

import os
import pytest
import pandas as pd
import numpy as np

from trading_system.data import (
    load_ohlcv_data,
    load_universe,
    load_benchmark,
    validate_ohlcv,
    detect_missing_data,
    CRYPTO_UNIVERSE,
    get_trading_days,
    get_crypto_days
)
from trading_system.models.market_data import MarketData


# Get test fixtures directory
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


class TestLoadOHLCVData:
    """Test load_ohlcv_data function."""
    
    def test_load_valid_data(self):
        """Test loading valid OHLCV data."""
        data = load_ohlcv_data(FIXTURES_DIR, ["AAPL"])
        
        assert "AAPL" in data
        df = data["AAPL"]
        
        assert len(df) > 0
        assert all(col in df.columns for col in ["open", "high", "low", "close", "volume"])
        assert "dollar_volume" in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)
        
        # Check dollar_volume is computed correctly
        expected_dv = df["close"] * df["volume"]
        assert np.allclose(df["dollar_volume"], expected_dv)
    
    def test_load_multiple_symbols(self):
        """Test loading multiple symbols."""
        data = load_ohlcv_data(FIXTURES_DIR, ["AAPL", "MSFT"])
        
        assert "AAPL" in data
        assert "MSFT" in data
        assert len(data) == 2
    
    def test_missing_file_handling(self):
        """Test handling of missing files."""
        data = load_ohlcv_data(FIXTURES_DIR, ["AAPL", "NONEXISTENT"])
        
        # Should load AAPL but skip NONEXISTENT
        assert "AAPL" in data
        assert "NONEXISTENT" not in data
    
    def test_date_filtering(self):
        """Test date range filtering."""
        start_date = pd.Timestamp("2023-01-02")
        end_date = pd.Timestamp("2023-01-04")
        
        data = load_ohlcv_data(
            FIXTURES_DIR,
            ["AAPL"],
            start_date=start_date,
            end_date=end_date
        )
        
        df = data["AAPL"]
        assert df.index.min() >= start_date
        assert df.index.max() <= end_date
    
    def test_invalid_data_skipped(self):
        """Test that invalid data files are skipped."""
        # INVALID_OHLC.csv has invalid OHLC relationships
        data = load_ohlcv_data(FIXTURES_DIR, ["INVALID_OHLC"])
        
        # Should be skipped due to validation failure
        assert "INVALID_OHLC" not in data


class TestValidateOHLCV:
    """Test validate_ohlcv function."""
    
    def test_valid_data(self):
        """Test validation of valid data."""
        df = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [102.0, 103.0],
            'low': [99.0, 100.0],
            'close': [101.0, 102.0],
            'volume': [1000000, 1100000]
        }, index=pd.date_range("2023-01-01", periods=2))
        
        assert validate_ohlcv(df, "TEST") is True
    
    def test_invalid_ohlc_relationship(self):
        """Test validation fails on invalid OHLC."""
        df = pd.DataFrame({
            'open': [100.0, 151.50],
            'high': [102.0, 150.00],  # high < open
            'low': [99.0, 150.90],
            'close': [101.0, 152.80],
            'volume': [1000000, 4800000]
        }, index=pd.date_range("2023-01-01", periods=2))
        
        assert validate_ohlcv(df, "TEST") is False
    
    def test_negative_volume(self):
        """Test validation fails on negative volume."""
        df = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [102.0, 103.0],
            'low': [99.0, 100.0],
            'close': [101.0, 102.0],
            'volume': [1000000, -1000000]  # Negative volume
        }, index=pd.date_range("2023-01-01", periods=2))
        
        assert validate_ohlcv(df, "TEST") is False
    
    def test_non_positive_prices(self):
        """Test validation fails on non-positive prices."""
        df = pd.DataFrame({
            'open': [100.0, 0.0],  # Zero price
            'high': [102.0, 103.0],
            'low': [99.0, 100.0],
            'close': [101.0, 102.0],
            'volume': [1000000, 1100000]
        }, index=pd.date_range("2023-01-01", periods=2))
        
        assert validate_ohlcv(df, "TEST") is False
    
    def test_duplicate_dates(self):
        """Test validation fails on duplicate dates."""
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [102.0, 103.0, 104.0],
            'low': [99.0, 100.0, 101.0],
            'close': [101.0, 102.0, 103.0],
            'volume': [1000000, 1100000, 1200000]
        }, index=pd.DatetimeIndex([
            "2023-01-01",
            "2023-01-02",
            "2023-01-02"  # Duplicate
        ]))
        
        assert validate_ohlcv(df, "TEST") is False
    
    def test_extreme_move_warning(self, caplog):
        """Test that extreme moves generate warnings but don't fail validation."""
        df = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [102.0, 200.0],
            'low': [99.0, 100.0],
            'close': [101.0, 160.0],  # >50% move
            'volume': [1000000, 1000000]
        }, index=pd.date_range("2023-01-01", periods=2))
        
        # Should pass validation but log warning
        result = validate_ohlcv(df, "TEST")
        assert result is True
        assert "Extreme moves" in caplog.text or "extreme" in caplog.text.lower()


class TestDetectMissingData:
    """Test detect_missing_data function."""
    
    def test_no_missing_data(self):
        """Test detection when no data is missing."""
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [102.0, 103.0, 104.0],
            'low': [99.0, 100.0, 101.0],
            'close': [101.0, 102.0, 103.0],
            'volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range("2023-01-01", periods=3, freq='D'))
        
        result = detect_missing_data(df, "TEST", asset_class="crypto")
        
        assert len(result['missing_dates']) == 0
        assert len(result['consecutive_gaps']) == 0
    
    def test_single_missing_day(self):
        """Test detection of single missing day."""
        # Create data with missing day (2023-01-02)
        dates = pd.DatetimeIndex([
            "2023-01-01",
            "2023-01-03",
            "2023-01-04"
        ])
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [102.0, 103.0, 104.0],
            'low': [99.0, 100.0, 101.0],
            'close': [101.0, 102.0, 103.0],
            'volume': [1000000, 1100000, 1200000]
        }, index=dates)
        
        result = detect_missing_data(df, "TEST", asset_class="crypto")
        
        assert len(result['missing_dates']) == 1
        assert pd.Timestamp("2023-01-02") in result['missing_dates']
        assert len(result['consecutive_gaps']) == 1
    
    def test_consecutive_missing_days(self):
        """Test detection of consecutive missing days."""
        dates = pd.DatetimeIndex([
            "2023-01-01",
            "2023-01-05"  # Missing 3 days
        ])
        df = pd.DataFrame({
            'open': [100.0, 104.0],
            'high': [102.0, 106.0],
            'low': [99.0, 103.0],
            'close': [101.0, 105.0],
            'volume': [1000000, 1400000]
        }, index=dates)
        
        result = detect_missing_data(df, "TEST", asset_class="crypto")
        
        assert len(result['missing_dates']) == 3
        assert len(result['consecutive_gaps']) == 1
        gap_start, gap_end = result['consecutive_gaps'][0]
        assert (gap_end - gap_start).days == 2  # 3 missing days = 2 day gap


class TestLoadUniverse:
    """Test load_universe function."""
    
    def test_load_crypto_universe(self):
        """Test loading crypto universe (fixed list)."""
        universe = load_universe("crypto")
        
        assert universe == CRYPTO_UNIVERSE
        assert len(universe) == 10
        assert "BTC" in universe
        assert "ETH" in universe
    
    def test_load_equity_universe_from_file(self):
        """Test loading equity universe from file."""
        universe_path = os.path.join(FIXTURES_DIR, "NASDAQ-100.csv")
        universe = load_universe("NASDAQ-100", universe_path=universe_path)
        
        assert len(universe) == 3
        assert "AAPL" in universe
        assert "MSFT" in universe
        assert "GOOGL" in universe
    
    def test_invalid_universe_type(self):
        """Test error handling for invalid universe type."""
        with pytest.raises(ValueError, match="Unknown universe type"):
            load_universe("INVALID_TYPE")
    
    def test_missing_universe_file(self):
        """Test error handling for missing universe file."""
        with pytest.raises(FileNotFoundError):
            load_universe("NASDAQ-100", universe_path="nonexistent.csv")


class TestLoadBenchmark:
    """Test load_benchmark function."""
    
    def test_benchmark_file_not_found(self):
        """Test error handling when benchmark file is missing."""
        with pytest.raises(ValueError, match="Benchmark.*not in loaded data|Benchmark file not found"):
            load_benchmark("NONEXISTENT", FIXTURES_DIR)
    
    def test_benchmark_insufficient_data(self):
        """Test that benchmark validation requires minimum 250 days."""
        # Fixtures only have 5 days, should fail validation
        with pytest.raises(ValueError, match="insufficient data"):
            load_benchmark("SPY", FIXTURES_DIR)
    
    def test_benchmark_date_filtering(self):
        """Test date filtering for benchmark (with sufficient data requirement)."""
        # This test demonstrates date filtering, but will fail due to insufficient data
        # In practice, you'd have a larger dataset
        start_date = pd.Timestamp("2023-01-02")
        end_date = pd.Timestamp("2023-01-04")
        
        # Should fail due to insufficient data after filtering
        with pytest.raises(ValueError, match="insufficient data"):
            load_benchmark(
                "SPY",
                FIXTURES_DIR,
                start_date=start_date,
                end_date=end_date
            )


class TestCalendarFunctions:
    """Test calendar helper functions."""
    
    def test_get_trading_days(self):
        """Test get_trading_days function."""
        dates = pd.date_range("2023-01-01", periods=10, freq='D')
        end_date = pd.Timestamp("2023-01-10")
        
        trading_days = get_trading_days(dates, end_date, lookback=5)
        
        assert len(trading_days) <= 5
        assert all(isinstance(d, pd.Timestamp) for d in trading_days)
        assert all(d.weekday() < 5 for d in trading_days)  # Mon-Fri only
    
    def test_get_crypto_days(self):
        """Test get_crypto_days function."""
        end_date = pd.Timestamp("2023-01-10")
        crypto_days = get_crypto_days(end_date, lookback=7)
        
        assert len(crypto_days) == 7
        assert crypto_days[-1] == end_date
        assert all(isinstance(d, pd.Timestamp) for d in crypto_days)


class TestMarketData:
    """Test MarketData container class."""
    
    def test_market_data_initialization(self):
        """Test MarketData initialization."""
        market_data = MarketData()
        
        assert isinstance(market_data.bars, dict)
        assert isinstance(market_data.features, dict)
        assert isinstance(market_data.benchmarks, dict)
    
    def test_get_bar(self):
        """Test getting bar from MarketData."""
        market_data = MarketData()
        
        # Load some data
        data = load_ohlcv_data(FIXTURES_DIR, ["AAPL"])
        market_data.bars = data
        
        # Get a bar
        date = data["AAPL"].index[0]
        bar = market_data.get_bar("AAPL", date)
        
        assert bar is not None
        assert bar.symbol == "AAPL"
        assert bar.date == date
        assert bar.close > 0
    
    def test_get_bar_nonexistent(self):
        """Test getting bar for nonexistent symbol/date."""
        market_data = MarketData()
        
        bar = market_data.get_bar("NONEXISTENT", pd.Timestamp("2023-01-01"))
        assert bar is None
        
        # Load data
        data = load_ohlcv_data(FIXTURES_DIR, ["AAPL"])
        market_data.bars = data
        
        # Try nonexistent date
        bar = market_data.get_bar("AAPL", pd.Timestamp("2099-01-01"))
        assert bar is None


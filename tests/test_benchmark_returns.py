"""Tests for benchmark returns extraction in BacktestRunner.

This test suite verifies that benchmark returns extraction works correctly
for both equity (SPY) and crypto (BTC) benchmarks, and handles edge cases
gracefully.
"""

import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from trading_system.integration.runner import BacktestRunner
from trading_system.models.market_data import MarketData
from trading_system.configs.run_config import RunConfig

# Get test fixtures directory
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


class TestBenchmarkReturnsExtraction:
    """Test benchmark returns extraction functionality."""
    
    @pytest.fixture
    def sample_config(self):
        """Create a minimal RunConfig for testing."""
        # Create a minimal config object
        config = Mock(spec=RunConfig)
        return config
    
    @pytest.fixture
    def runner(self, sample_config):
        """Create a BacktestRunner instance for testing."""
        runner = BacktestRunner(sample_config)
        return runner
    
    @pytest.fixture
    def equity_benchmark_data(self):
        """Create sample SPY benchmark data."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        # Create realistic price data with some variation
        base_price = 380.0
        returns = [0.0, 0.01, -0.005, 0.02, 0.005, -0.01, 0.015, -0.005, 0.01, 0.005]
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [80000000] * len(dates),
            'dollar_volume': [p * 80000000 for p in prices]
        }, index=dates)
        return df
    
    @pytest.fixture
    def crypto_benchmark_data(self):
        """Create sample BTC benchmark data."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        # Create realistic price data with some variation
        base_price = 16500.0
        returns = [0.0, 0.02, -0.01, 0.03, 0.01, -0.02, 0.025, -0.015, 0.02, 0.01]
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': [1500.5] * len(dates),
            'dollar_volume': [p * 1500.5 for p in prices]
        }, index=dates)
        return df
    
    def test_extract_spy_benchmark_returns(self, runner, equity_benchmark_data):
        """Test extraction of SPY benchmark returns for equity."""
        # Setup market data
        runner.market_data = MarketData()
        runner.market_data.benchmarks['SPY'] = equity_benchmark_data
        
        # Extract returns for all dates
        dates = list(equity_benchmark_data.index)
        returns = runner._extract_benchmark_returns(dates, benchmark_symbol='SPY')
        
        # Should return list of returns (one less than dates)
        assert returns is not None
        assert isinstance(returns, list)
        assert len(returns) == len(dates) - 1  # First day has no return
        
        # Verify return calculations
        # Day 1: price goes from 380.0 to 383.8 (1% return)
        assert abs(returns[0] - 0.01) < 1e-10
        
        # Day 2: price goes from 383.8 to 381.88 (approx -0.5% return)
        expected_return_2 = (equity_benchmark_data.loc[dates[2], 'close'] / 
                            equity_benchmark_data.loc[dates[1], 'close']) - 1.0
        assert abs(returns[1] - expected_return_2) < 1e-10
        
        # All returns should be finite
        assert all(np.isfinite(r) for r in returns)
    
    def test_extract_btc_benchmark_returns(self, runner, crypto_benchmark_data):
        """Test extraction of BTC benchmark returns for crypto."""
        # Setup market data
        runner.market_data = MarketData()
        runner.market_data.benchmarks['BTC'] = crypto_benchmark_data
        
        # Extract returns for all dates
        dates = list(crypto_benchmark_data.index)
        returns = runner._extract_benchmark_returns(dates, benchmark_symbol='BTC')
        
        # Should return list of returns (one less than dates)
        assert returns is not None
        assert isinstance(returns, list)
        assert len(returns) == len(dates) - 1  # First day has no return
        
        # Verify return calculations
        # Day 1: price goes from 16500.0 to 16830.0 (2% return)
        assert abs(returns[0] - 0.02) < 1e-10
        
        # Day 2: price goes from 16830.0 to 16661.7 (approx -1% return)
        expected_return_2 = (crypto_benchmark_data.loc[dates[2], 'close'] / 
                            crypto_benchmark_data.loc[dates[1], 'close']) - 1.0
        assert abs(returns[1] - expected_return_2) < 1e-10
        
        # All returns should be finite
        assert all(np.isfinite(r) for r in returns)
    
    def test_missing_benchmark_symbol(self, runner, equity_benchmark_data):
        """Test that missing benchmark symbol returns None gracefully."""
        # Setup market data with SPY but request BTC
        runner.market_data = MarketData()
        runner.market_data.benchmarks['SPY'] = equity_benchmark_data
        
        dates = list(equity_benchmark_data.index)
        returns = runner._extract_benchmark_returns(dates, benchmark_symbol='BTC')
        
        # Should return None when benchmark not found
        assert returns is None
    
    def test_market_data_is_none(self, runner):
        """Test that None market_data returns None."""
        runner.market_data = None
        
        dates = [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-02')]
        returns = runner._extract_benchmark_returns(dates, benchmark_symbol='SPY')
        
        # Should return None when market_data is None
        assert returns is None
    
    def test_missing_dates_in_benchmark(self, runner, equity_benchmark_data):
        """Test handling of missing dates in benchmark data."""
        # Setup market data
        runner.market_data = MarketData()
        runner.market_data.benchmarks['SPY'] = equity_benchmark_data
        
        # Request dates that include some not in benchmark
        dates = list(equity_benchmark_data.index[:5])  # First 5 dates (present)
        dates.append(pd.Timestamp('2024-01-15'))  # Missing date
        dates.append(pd.Timestamp('2024-01-16'))  # Missing date
        dates.extend(list(equity_benchmark_data.index[5:]))  # Remaining dates (present)
        
        returns = runner._extract_benchmark_returns(dates, benchmark_symbol='SPY')
        
        # Should still return a list
        assert returns is not None
        assert isinstance(returns, list)
        
        # Should handle missing dates by using previous return or 0
        # Length should be len(dates) - 1
        assert len(returns) == len(dates) - 1
        
        # All returns should be finite
        assert all(np.isfinite(r) for r in returns)
    
    def test_empty_dates_list(self, runner, equity_benchmark_data):
        """Test handling of empty dates list."""
        runner.market_data = MarketData()
        runner.market_data.benchmarks['SPY'] = equity_benchmark_data
        
        returns = runner._extract_benchmark_returns([], benchmark_symbol='SPY')
        
        # Should return None for empty list
        # The implementation returns None when returns list is empty
        assert returns is None
    
    def test_single_date(self, runner, equity_benchmark_data):
        """Test handling of single date."""
        runner.market_data = MarketData()
        runner.market_data.benchmarks['SPY'] = equity_benchmark_data
        
        dates = [equity_benchmark_data.index[0]]
        returns = runner._extract_benchmark_returns(dates, benchmark_symbol='SPY')
        
        # Single date has no return, so after removing first return (which is 0.0),
        # list will be empty and should return None
        assert returns is None
    
    def test_zero_close_price_handling(self, runner):
        """Test handling of zero or negative close prices."""
        # Create benchmark data with zero close
        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        df = pd.DataFrame({
            'open': [100.0, 101.0, 0.0],
            'high': [102.0, 103.0, 1.0],
            'low': [99.0, 100.0, 0.0],
            'close': [100.0, 101.0, 0.0],  # Zero close on day 3
            'volume': [1000000, 1100000, 1200000],
            'dollar_volume': [100000000, 111100000, 0]
        }, index=dates)
        
        runner.market_data = MarketData()
        runner.market_data.benchmarks['SPY'] = df
        
        returns = runner._extract_benchmark_returns(list(dates), benchmark_symbol='SPY')
        
        # Should handle zero price gracefully
        # Day 2 return should be calculated (101/100 - 1 = 0.01)
        assert returns is not None
        assert len(returns) == 2  # 3 dates - 1
        
        # First return should be valid
        assert abs(returns[0] - 0.01) < 1e-10
        
        # Second return with zero close should be -1.0 (or handled somehow)
        # When prev_close is 101.0 and close is 0.0, return is -1.0
        assert returns[1] == -1.0 or np.isfinite(returns[1])
    
    def test_consecutive_missing_dates(self, runner, equity_benchmark_data):
        """Test handling of consecutive missing dates."""
        runner.market_data = MarketData()
        runner.market_data.benchmarks['SPY'] = equity_benchmark_data
        
        # Create dates with consecutive gaps
        dates = [
            equity_benchmark_data.index[0],  # Present
            pd.Timestamp('2024-01-15'),  # Missing
            pd.Timestamp('2024-01-16'),  # Missing
            pd.Timestamp('2024-01-17'),  # Missing
            equity_benchmark_data.index[1],  # Present
        ]
        
        returns = runner._extract_benchmark_returns(dates, benchmark_symbol='SPY')
        
        # Should handle missing dates
        assert returns is not None
        assert len(returns) == len(dates) - 1
        assert all(np.isfinite(r) for r in returns)
    
    def test_returns_match_manual_calculation(self, runner):
        """Test that extracted returns match manual calculation."""
        # Create simple benchmark data
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        prices = [100.0, 105.0, 102.0, 108.0, 110.0]  # 5%, -2.86%, 5.88%, 1.85%
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000000] * len(dates),
            'dollar_volume': [p * 1000000 for p in prices]
        }, index=dates)
        
        runner.market_data = MarketData()
        runner.market_data.benchmarks['SPY'] = df
        
        returns = runner._extract_benchmark_returns(list(dates), benchmark_symbol='SPY')
        
        # Manual calculation
        expected_returns = [
            (105.0 / 100.0) - 1.0,  # 0.05
            (102.0 / 105.0) - 1.0,  # -0.02857...
            (108.0 / 102.0) - 1.0,  # 0.05882...
            (110.0 / 108.0) - 1.0,  # 0.01851...
        ]
        
        assert returns is not None
        assert len(returns) == 4
        for i, expected in enumerate(expected_returns):
            assert abs(returns[i] - expected) < 1e-10
    
    def test_equity_vs_crypto_benchmarks(self, runner, equity_benchmark_data, crypto_benchmark_data):
        """Test that both equity and crypto benchmarks work correctly."""
        runner.market_data = MarketData()
        runner.market_data.benchmarks['SPY'] = equity_benchmark_data
        runner.market_data.benchmarks['BTC'] = crypto_benchmark_data
        
        # Test SPY
        spy_dates = list(equity_benchmark_data.index)
        spy_returns = runner._extract_benchmark_returns(spy_dates, benchmark_symbol='SPY')
        assert spy_returns is not None
        assert len(spy_returns) == len(spy_dates) - 1
        
        # Test BTC
        btc_dates = list(crypto_benchmark_data.index)
        btc_returns = runner._extract_benchmark_returns(btc_dates, benchmark_symbol='BTC')
        assert btc_returns is not None
        assert len(btc_returns) == len(btc_dates) - 1
        
        # Both should have valid returns
        assert all(np.isfinite(r) for r in spy_returns)
        assert all(np.isfinite(r) for r in btc_returns)
    
    def test_returns_length_matches_dates(self, runner, equity_benchmark_data):
        """Test that returns length is always one less than dates length."""
        runner.market_data = MarketData()
        runner.market_data.benchmarks['SPY'] = equity_benchmark_data
        
        # Test with various date ranges
        for n_dates in range(2, 11):
            dates = list(equity_benchmark_data.index[:n_dates])
            returns = runner._extract_benchmark_returns(dates, benchmark_symbol='SPY')
            
            if n_dates > 1:
                assert returns is not None
                assert len(returns) == n_dates - 1
    
    def test_negative_returns_handled(self, runner):
        """Test that negative returns are handled correctly."""
        # Create data with negative returns
        dates = pd.date_range('2024-01-01', periods=4, freq='D')
        prices = [100.0, 95.0, 90.0, 85.0]  # -5%, -5.26%, -5.56%
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000000] * len(dates),
            'dollar_volume': [p * 1000000 for p in prices]
        }, index=dates)
        
        runner.market_data = MarketData()
        runner.market_data.benchmarks['SPY'] = df
        
        returns = runner._extract_benchmark_returns(list(dates), benchmark_symbol='SPY')
        
        assert returns is not None
        assert len(returns) == 3
        # All returns should be negative
        assert all(r < 0 for r in returns)
        # Verify values
        assert abs(returns[0] - (-0.05)) < 1e-10  # -5%
        assert abs(returns[1] - ((90.0/95.0) - 1.0)) < 1e-10
        assert abs(returns[2] - ((85.0/90.0) - 1.0)) < 1e-10
    
    def test_with_real_benchmark_data(self, runner):
        """Test extraction using actual benchmark CSV files from fixtures."""
        from trading_system.data import load_ohlcv_data
        
        # Load benchmark data from fixtures
        benchmark_dir = os.path.join(FIXTURES_DIR, "benchmarks")
        
        # Try to load SPY benchmark
        try:
            spy_data = load_ohlcv_data(benchmark_dir, ["SPY"])
            if "SPY" in spy_data:
                runner.market_data = MarketData()
                runner.market_data.benchmarks['SPY'] = spy_data["SPY"]
                
                # Extract returns for available dates
                dates = list(spy_data["SPY"].index)
                returns = runner._extract_benchmark_returns(dates, benchmark_symbol='SPY')
                
                # Should return valid returns
                assert returns is not None
                assert len(returns) == len(dates) - 1
                assert all(np.isfinite(r) for r in returns)
                
                # Verify returns are reasonable (between -1 and 1 for daily returns)
                assert all(-1.0 <= r <= 1.0 for r in returns), "Returns should be between -100% and +100%"
        except (FileNotFoundError, ValueError, KeyError):
            # Skip if benchmark files not available or insufficient data
            pytest.skip("Benchmark data not available for integration test")
        
        # Try to load BTC benchmark
        try:
            btc_data = load_ohlcv_data(benchmark_dir, ["BTC"])
            if "BTC" in btc_data:
                runner.market_data = MarketData()
                runner.market_data.benchmarks['BTC'] = btc_data["BTC"]
                
                # Extract returns for available dates
                dates = list(btc_data["BTC"].index)
                returns = runner._extract_benchmark_returns(dates, benchmark_symbol='BTC')
                
                # Should return valid returns
                assert returns is not None
                assert len(returns) == len(dates) - 1
                assert all(np.isfinite(r) for r in returns)
                
                # Verify returns are reasonable
                assert all(-1.0 <= r <= 1.0 for r in returns), "Returns should be between -100% and +100%"
        except (FileNotFoundError, ValueError, KeyError):
            # Skip if benchmark files not available or insufficient data
            pytest.skip("Benchmark data not available for integration test")


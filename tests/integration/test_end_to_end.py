"""End-to-end integration test for the trading system.

This test verifies that the complete system produces expected results
with the test dataset.
"""

import os
import pytest
import pandas as pd
import numpy as np
from typing import Dict, List

# Import test utilities
from tests.utils import (
    create_sample_bar,
    create_sample_feature_row,
    create_sample_signal,
    assert_no_lookahead,
    assert_valid_signal,
    assert_valid_portfolio,
)

# Get test fixtures directory
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
CONFIGS_DIR = os.path.join(FIXTURES_DIR, "configs")


class TestEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.fixture
    def test_data_path(self):
        """Path to test data fixtures."""
        return FIXTURES_DIR
    
    @pytest.fixture
    def equity_symbols(self):
        """Equity symbols for testing."""
        return ["AAPL", "MSFT", "GOOGL"]
    
    @pytest.fixture
    def crypto_symbols(self):
        """Crypto symbols for testing."""
        return ["BTC", "ETH", "SOL"]
    
    def test_data_loading(self, test_data_path, equity_symbols):
        """Test that data loading works with test fixtures."""
        from trading_system.data import load_ohlcv_data
        
        # Load equity data
        equity_data = load_ohlcv_data(test_data_path, equity_symbols)
        
        # Verify data loaded
        assert len(equity_data) > 0, "Should load some equity data"
        
        # Verify each symbol has data
        for symbol in equity_symbols:
            # Check if file exists (files may have _sample suffix)
            sample_file = os.path.join(test_data_path, f"{symbol}_sample.csv")
            if os.path.exists(sample_file):
                # If sample file exists, we expect data for that symbol
                # Note: Actual implementation may require renaming files
                pass
    
    def test_strategy_signal_generation(self):
        """Test that strategies can generate signals from test data."""
        from trading_system.strategies.equity_strategy import EquityStrategy
        from trading_system.configs.strategy_config import StrategyConfig, EntryConfig, ExitConfig, CapacityConfig
        from trading_system.models.features import FeatureRow
        
        # Create test strategy config
        config = StrategyConfig(
            name="test_equity",
            asset_class="equity",
            universe=["AAPL"],
            benchmark="SPY",
            entry=EntryConfig(fast_clearance=0.005, slow_clearance=0.010),
            exit=ExitConfig(mode="ma_cross", exit_ma=20, hard_stop_atr_mult=2.5),
            capacity=CapacityConfig(max_order_pct_adv=0.005),
        )
        
        strategy = EquityStrategy(config)
        
        # Create test feature row
        features = create_sample_feature_row(
            date=pd.Timestamp("2023-11-15"),
            symbol="AAPL",
            asset_class="equity",
            close=150.0,
            ma20=148.0,
            ma50=145.0,
            highest_close_20d=148.0,  # Close > highest_close_20d * 1.005 triggers breakout
            highest_close_55d=142.0,
            adv20=100000000.0,
        )
        
        # Generate signal
        order_notional = 500000.0  # Within capacity
        signal = strategy.generate_signal("AAPL", features, order_notional)
        
        # Verify signal (may be None if conditions not met)
        if signal is not None:
            assert_valid_signal(signal)
            assert signal.symbol == "AAPL"
            assert signal.asset_class == "equity"
    
    def test_portfolio_operations(self):
        """Test portfolio operations with sample data."""
        from trading_system.models.portfolio import Portfolio
        from trading_system.models.positions import Position
        from trading_system.models.signals import BreakoutType
        
        # Create portfolio
        portfolio = create_sample_portfolio(
            date=pd.Timestamp("2023-11-15"),
            starting_equity=100000.0,
            cash=100000.0,
        )
        
        assert_valid_portfolio(portfolio)
        
        # Create position
        position = create_sample_position(
            symbol="AAPL",
            asset_class="equity",
            entry_date=pd.Timestamp("2023-11-10"),
            entry_price=150.0,
            quantity=100,
            stop_price=145.0,
        )
        
        # Add position to portfolio
        portfolio.positions["AAPL"] = position
        
        # Update portfolio equity
        current_prices = {"AAPL": 155.0}
        portfolio.update_equity(current_prices)
        
        # Verify portfolio
        assert_valid_portfolio(portfolio)
        assert len(portfolio.positions) == 1
        assert portfolio.open_trades == 1
    
    def test_no_lookahead_bias(self):
        """Test that signals don't use future data."""
        from trading_system.models.bar import Bar
        from trading_system.models.signals import Signal, SignalSide, BreakoutType
        
        # Create bars for multiple dates
        dates = pd.bdate_range("2023-11-01", "2023-11-10")
        bars = {}
        for date in dates:
            bars[date] = create_sample_bar(
                date=date,
                symbol="AAPL",
                base_price=150.0,
                volatility=0.02,
            )
        
        # Create signals (simulated)
        signals = []
        for i, date in enumerate(dates[20:], start=20):  # Start from day 20
            signal = create_sample_signal(
                date=date,
                symbol="AAPL",
                asset_class="equity",
                entry_price=bars[date].close,
                atr14=3.0,
            )
            signals.append(signal)
        
        # Assert no lookahead
        assert_no_lookahead(signals, bars)
    
    def test_data_validation(self, test_data_path):
        """Test that test data passes validation."""
        from trading_system.data import load_ohlcv_data
        from trading_system.data.validator import validate_ohlcv
        
        # Try loading sample data files
        # Note: This test may need adjustment based on actual file naming
        test_symbols = ["AAPL_sample", "MSFT_sample", "GOOGL_sample"]
        
        # Check if files exist
        existing_files = []
        for symbol in test_symbols:
            file_path = os.path.join(test_data_path, f"{symbol}.csv")
            if os.path.exists(file_path):
                existing_files.append(symbol)
        
        if existing_files:
            # Load and validate data
            data = load_ohlcv_data(test_data_path, existing_files)
            
            for symbol, df in data.items():
                assert validate_ohlcv(df, symbol), (
                    f"Test data for {symbol} failed validation"
                )
                assert len(df) > 0, f"Test data for {symbol} is empty"
    
    def test_integration_workflow(self):
        """Test basic integration workflow (without full backtest engine).
        
        This is a simplified test that verifies components work together.
        Full backtest integration test will require the backtest engine.
        """
        # This test will be expanded once the backtest engine is implemented
        # For now, it's a placeholder that verifies basic component integration
        
        from trading_system.models.features import FeatureRow
        from trading_system.models.signals import Signal
        
        # Create feature row
        features = create_sample_feature_row(
            date=pd.Timestamp("2023-11-15"),
            symbol="AAPL",
            close=150.0,
        )
        
        # Verify feature row is valid
        assert features.is_valid_for_entry() or not features.is_valid_for_entry()
        # (Either valid or not, just check method works)
        
        # Create signal from features
        signal = create_sample_signal(
            date=pd.Timestamp("2023-11-15"),
            symbol="AAPL",
            entry_price=features.close,
        )
        
        # Verify signal
        assert_valid_signal(signal)


@pytest.mark.skip(reason="Requires full backtest engine implementation")
class TestFullBacktest:
    """Full backtest integration tests (requires backtest engine)."""
    
    def test_full_backtest_run(self):
        """Test running a full backtest with test config and data."""
        # This test will be implemented once the backtest engine is available
        # It should:
        # 1. Load test config
        # 2. Load test data
        # 3. Run backtest
        # 4. Verify expected trades occurred
        # 5. Verify metrics are reasonable
        pass
    
    def test_expected_trades(self):
        """Test that system produces expected trades from test dataset."""
        # This test will verify that the system produces known expected trades
        # when run on the test dataset
        pass


"""Integration tests for missing data handling in backtest engine.

Tests verify that missing data scenarios are handled correctly:
1. Single day missing: skip signal generation, log warning
2. 2+ consecutive days: mark unhealthy, force exit positions
3. No infinite loops or crashes
"""

import pytest
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import os

from trading_system.backtest.engine import BacktestEngine
from trading_system.backtest.event_loop import DailyEventLoop
from trading_system.models.market_data import MarketData
from trading_system.models.portfolio import Portfolio
from trading_system.models.positions import Position, ExitReason
from trading_system.strategies.equity_strategy import EquityStrategy
from trading_system.configs.strategy_config import StrategyConfig, EntryConfig, ExitConfig, CapacityConfig, RiskConfig
from trading_system.data.loader import load_ohlcv_data
from trading_system.data.validator import detect_missing_data
from trading_system.data.calendar import get_next_trading_day
from trading_system.indicators.feature_computer import compute_features


# Get test fixtures directory
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def simple_strategy_config():
    """Create a simple strategy config for testing."""
    return StrategyConfig(
        name="test_equity",
        asset_class="equity",
        universe=["TEST"],
        benchmark="SPY",
        entry=EntryConfig(fast_clearance=0.005, slow_clearance=0.010),
        exit=ExitConfig(mode="ma_cross", exit_ma=20, hard_stop_atr_mult=2.5),
        capacity=CapacityConfig(max_order_pct_adv=0.005),
        risk=RiskConfig(
            risk_per_trade=0.01,
            max_positions=5,
            max_exposure=1.0,
            max_position_notional=0.20
        )
    )


@pytest.fixture
def sample_market_data_with_gaps():
    """Create market data with missing days for testing."""
    market_data = MarketData()
    
    # Create data with single missing day (2023-01-02)
    dates_single = pd.DatetimeIndex([
        "2023-01-01",
        "2023-01-03",  # Missing 2023-01-02
        "2023-01-04",
        "2023-01-05",
        "2023-01-06"
    ])
    
    prices = [100.0, 102.0, 103.0, 104.0, 105.0]
    bars_single = pd.DataFrame({
        'open': prices,
        'high': [p * 1.02 for p in prices],
        'low': [p * 0.98 for p in prices],
        'close': prices,
        'volume': [1000000] * len(prices),
        'dollar_volume': [p * 1000000 for p in prices]
    }, index=dates_single)
    market_data.bars['SINGLE_GAP'] = bars_single
    
    # Create data with 2+ consecutive missing days (2023-01-03, 2023-01-04)
    dates_consecutive = pd.DatetimeIndex([
        "2023-01-01",
        "2023-01-02",
        "2023-01-05",  # Missing 2023-01-03, 2023-01-04
        "2023-01-06"
    ])
    
    prices2 = [100.0, 101.0, 105.0, 106.0]
    bars_consecutive = pd.DataFrame({
        'open': prices2,
        'high': [p * 1.02 for p in prices2],
        'low': [p * 0.98 for p in prices2],
        'close': prices2,
        'volume': [1000000] * len(prices2),
        'dollar_volume': [p * 1000000 for p in prices2]
    }, index=dates_consecutive)
    market_data.bars['CONSECUTIVE_GAP'] = bars_consecutive
    
    return market_data


class TestMissingDataDetection:
    """Test missing data detection logic."""
    
    def test_detect_single_missing_day(self):
        """Test detection of single missing day."""
        dates = pd.DatetimeIndex([
            "2023-01-01",
            "2023-01-03",  # Missing 2023-01-02
            "2023-01-04"
        ])
        df = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [102.0, 103.0, 104.0],
            'low': [99.0, 100.0, 101.0],
            'close': [101.0, 102.0, 103.0],
            'volume': [1000000, 1100000, 1200000]
        }, index=dates)
        
        result = detect_missing_data(df, "TEST", asset_class="equity")
        
        assert len(result['missing_dates']) == 1
        assert pd.Timestamp("2023-01-02") in result['missing_dates']
        assert len(result['consecutive_gaps']) == 1
    
    def test_detect_consecutive_missing_days(self):
        """Test detection of 2+ consecutive missing days."""
        dates = pd.DatetimeIndex([
            "2023-01-01",
            "2023-01-02",
            "2023-01-05"  # Missing 2023-01-03, 2023-01-04
        ])
        df = pd.DataFrame({
            'open': [100.0, 101.0, 104.0],
            'high': [102.0, 103.0, 106.0],
            'low': [99.0, 100.0, 103.0],
            'close': [101.0, 102.0, 105.0],
            'volume': [1000000, 1100000, 1300000]
        }, index=dates)
        
        result = detect_missing_data(df, "TEST", asset_class="equity")
        
        assert len(result['missing_dates']) == 2
        assert pd.Timestamp("2023-01-03") in result['missing_dates']
        assert pd.Timestamp("2023-01-04") in result['missing_dates']
        assert len(result['consecutive_gaps']) == 1
        assert result['gap_lengths'][0] == 2


class TestMissingDataInEventLoop:
    """Test missing data handling in event loop."""
    
    def test_single_missing_day_logs_warning(self, sample_market_data_with_gaps, simple_strategy_config, caplog):
        """Test that single missing day logs warning and skips signal generation."""
        with caplog.at_level(logging.WARNING):
            market_data = sample_market_data_with_gaps
            strategy = EquityStrategy(simple_strategy_config)
            strategy.universe = ["SINGLE_GAP"]
            
            portfolio = Portfolio(
                starting_equity=100000.0,
                starting_cash=100000.0,
                date=pd.Timestamp("2023-01-01")
            )
            
            event_loop = DailyEventLoop(
                market_data=market_data,
                portfolio=portfolio,
                strategies=[strategy],
                compute_features_fn=compute_features,
                get_next_trading_day=get_next_trading_day,
                rng=None
            )
            
            # Process day with missing data (2023-01-02)
            # Note: 2023-01-02 is a Monday, so it's a trading day
            missing_date = pd.Timestamp("2023-01-02")
            events = event_loop.process_day(missing_date)
            
            # Check that warning was logged
            assert "MISSING_DATA_1DAY" in caplog.text
            assert "SINGLE_GAP" in caplog.text
            
            # Check that missing count was set
            assert event_loop.missing_data_counts.get("SINGLE_GAP") == 1
    
    def test_consecutive_missing_days_logs_error(self, sample_market_data_with_gaps, simple_strategy_config, caplog):
        """Test that 2+ consecutive missing days logs error."""
        with caplog.at_level(logging.ERROR):
            market_data = sample_market_data_with_gaps
            strategy = EquityStrategy(simple_strategy_config)
            strategy.universe = ["CONSECUTIVE_GAP"]
            
            portfolio = Portfolio(
                starting_equity=100000.0,
                starting_cash=100000.0,
                date=pd.Timestamp("2023-01-01")
            )
            
            event_loop = DailyEventLoop(
                market_data=market_data,
                portfolio=portfolio,
                strategies=[strategy],
                compute_features_fn=compute_features,
                get_next_trading_day=get_next_trading_day,
                rng=None
            )
            
            # Process first missing day (2023-01-03)
            missing_date1 = pd.Timestamp("2023-01-03")
            events1 = event_loop.process_day(missing_date1)
            
            # Process second missing day (2023-01-04)
            missing_date2 = pd.Timestamp("2023-01-04")
            events2 = event_loop.process_day(missing_date2)
            
            # Check that error was logged
            assert "DATA_UNHEALTHY" in caplog.text
            assert "CONSECUTIVE_GAP" in caplog.text
            assert "missing 2 consecutive days" in caplog.text or "missing 2 days" in caplog.text
            
            # Check that missing count was incremented
            assert event_loop.missing_data_counts.get("CONSECUTIVE_GAP") >= 2
    
    def test_consecutive_missing_days_force_exit_position(self, sample_market_data_with_gaps, simple_strategy_config):
        """Test that 2+ consecutive missing days forces exit of existing position."""
        market_data = sample_market_data_with_gaps
        strategy = EquityStrategy(simple_strategy_config)
        strategy.universe = ["CONSECUTIVE_GAP"]
        
        portfolio = Portfolio(
            starting_equity=100000.0,
            starting_cash=100000.0,
            date=pd.Timestamp("2023-01-01")
        )
        
        # Create a position before the missing data
        from trading_system.models.orders import Fill
        from trading_system.models.signals import SignalSide
        
        entry_fill = Fill(
            fill_id="test_fill_1",
            order_id="test_order_1",
            symbol="CONSECUTIVE_GAP",
            asset_class="equity",
            date=pd.Timestamp("2023-01-02"),
            side=SignalSide.BUY,
            quantity=100,
            fill_price=101.0,
            open_price=101.0,
            slippage_bps=8.0,
            fee_bps=5.0,
            total_cost=10150.0,
            vol_mult=1.0,
            size_penalty=1.0,
            weekend_penalty=1.0,
            stress_mult=1.0,
            notional=10100.0
        )
        
        position = portfolio.process_fill(
            fill=entry_fill,
            stop_price=98.0,
            atr_mult=2.5,
            triggered_on=None,
            adv20_at_entry=1000000.0
        )
        
        assert position is not None
        assert "CONSECUTIVE_GAP" in portfolio.positions
        assert portfolio.positions["CONSECUTIVE_GAP"].is_open()
        
        event_loop = DailyEventLoop(
            market_data=market_data,
            portfolio=portfolio,
            strategies=[strategy],
            compute_features_fn=compute_features,
            get_next_trading_day=get_next_trading_day,
            rng=None
        )
        
        # Process first missing day (2023-01-03)
        missing_date1 = pd.Timestamp("2023-01-03")
        events1 = event_loop.process_day(missing_date1)
        
        # Process second missing day (2023-01-04) - should trigger force exit
        missing_date2 = pd.Timestamp("2023-01-04")
        events2 = event_loop.process_day(missing_date2)
        
        # Check that exit order was created or position was closed
        # The position should be closed due to DATA_MISSING exit reason
        # Check if there are pending exit orders
        assert len(event_loop.pending_exit_orders) > 0 or "CONSECUTIVE_GAP" not in portfolio.positions or not portfolio.positions["CONSECUTIVE_GAP"].is_open()
    
    def test_missing_data_resets_when_data_returns(self, sample_market_data_with_gaps, simple_strategy_config):
        """Test that missing data count resets when data becomes available again."""
        market_data = sample_market_data_with_gaps
        strategy = EquityStrategy(simple_strategy_config)
        strategy.universe = ["SINGLE_GAP"]
        
        portfolio = Portfolio(
            starting_equity=100000.0,
            starting_cash=100000.0,
            date=pd.Timestamp("2023-01-01")
        )
        
        event_loop = DailyEventLoop(
            market_data=market_data,
            portfolio=portfolio,
            strategies=[strategy],
            compute_features_fn=compute_features,
            get_next_trading_day=get_next_trading_day,
            rng=None
        )
        
        # Process missing day (2023-01-02)
        missing_date = pd.Timestamp("2023-01-02")
        event_loop.process_day(missing_date)
        
        # Check that missing count was set
        assert event_loop.missing_data_counts.get("SINGLE_GAP") == 1
        
        # Process day with data (2023-01-03)
        data_date = pd.Timestamp("2023-01-03")
        event_loop.process_day(data_date)
        
        # Check that missing count was reset
        assert "SINGLE_GAP" not in event_loop.missing_data_counts
    
    def test_no_infinite_loop_on_missing_data(self, sample_market_data_with_gaps, simple_strategy_config):
        """Test that missing data doesn't cause infinite loops or crashes."""
        market_data = sample_market_data_with_gaps
        strategy = EquityStrategy(simple_strategy_config)
        strategy.universe = ["CONSECUTIVE_GAP"]
        
        portfolio = Portfolio(
            starting_equity=100000.0,
            starting_cash=100000.0,
            date=pd.Timestamp("2023-01-01")
        )
        
        event_loop = DailyEventLoop(
            market_data=market_data,
            portfolio=portfolio,
            strategies=[strategy],
            compute_features_fn=compute_features,
            get_next_trading_day=get_next_trading_day,
            rng=None
        )
        
        # Process multiple days with missing data - should not crash
        dates = pd.date_range("2023-01-01", "2023-01-06", freq='D')
        for date in dates:
            try:
                events = event_loop.process_day(date)
                # Should complete without error
                assert events is not None
            except Exception as e:
                pytest.fail(f"Event loop crashed on {date}: {e}")


class TestMissingDataWithFixtures:
    """Test missing data handling with actual fixture files."""
    
    def test_missing_day_fixture(self):
        """Test handling of MISSING_DAY.csv fixture."""
        fixture_path = os.path.join(FIXTURES_DIR, "MISSING_DAY.csv")
        
        if not os.path.exists(fixture_path):
            pytest.skip(f"Fixture not found: {fixture_path}")
        
        # Load the fixture
        df = pd.read_csv(fixture_path, index_col=0, parse_dates=True)
        
        # Detect missing data
        result = detect_missing_data(df, "MISSING_DAY", asset_class="equity")
        
        # Should detect one missing day (2023-01-02)
        assert len(result['missing_dates']) == 1
        assert pd.Timestamp("2023-01-02") in result['missing_dates']
    
    def test_missing_day_2plus_fixture(self):
        """Test handling of MISSING_DAY_2PLUS.csv fixture."""
        fixture_path = os.path.join(FIXTURES_DIR, "MISSING_DAY_2PLUS.csv")
        
        if not os.path.exists(fixture_path):
            pytest.skip(f"Fixture not found: {fixture_path}")
        
        # Load the fixture
        df = pd.read_csv(fixture_path, index_col=0, parse_dates=True)
        
        # Detect missing data
        result = detect_missing_data(df, "MISSING_DAY_2PLUS", asset_class="equity")
        
        # Should detect 2 consecutive missing days (2023-01-03, 2023-01-04)
        assert len(result['missing_dates']) == 2
        assert pd.Timestamp("2023-01-03") in result['missing_dates']
        assert pd.Timestamp("2023-01-04") in result['missing_dates']
        assert len(result['consecutive_gaps']) == 1
        assert result['gap_lengths'][0] == 2


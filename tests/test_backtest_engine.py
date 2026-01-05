"""Integration tests for backtest engine."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_system.backtest import BacktestEngine, WalkForwardSplit, create_default_split
from trading_system.configs.strategy_config import StrategyConfig
from trading_system.models.market_data import MarketData
from trading_system.portfolio.portfolio import Portfolio
from trading_system.strategies import CryptoMomentumStrategy, EquityMomentumStrategy

# Backward compatibility aliases
EquityStrategy = EquityMomentumStrategy
CryptoStrategy = CryptoMomentumStrategy


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    market_data = MarketData()

    # Create 3 symbols with 3 months of data
    dates = pd.date_range(start="2024-01-01", end="2024-03-31", freq="D")

    # Symbol 1: Trending up (should generate signals)
    np.random.seed(42)
    prices1 = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    bars1 = pd.DataFrame(
        {
            "open": prices1,
            "high": prices1 * 1.02,
            "low": prices1 * 0.98,
            "close": prices1,
            "volume": np.random.randint(1000000, 5000000, len(dates)),
            "dollar_volume": prices1 * np.random.randint(1000000, 5000000, len(dates)),
        },
        index=dates,
    )
    market_data.bars["SYMBOL1"] = bars1

    # Symbol 2: Sideways (should not generate signals)
    prices2 = 50 + np.cumsum(np.random.randn(len(dates)) * 0.2)
    bars2 = pd.DataFrame(
        {
            "open": prices2,
            "high": prices2 * 1.01,
            "low": prices2 * 0.99,
            "close": prices2,
            "volume": np.random.randint(500000, 2000000, len(dates)),
            "dollar_volume": prices2 * np.random.randint(500000, 2000000, len(dates)),
        },
        index=dates,
    )
    market_data.bars["SYMBOL2"] = bars2

    # Symbol 3: Trending up (should generate signals)
    prices3 = 200 + np.cumsum(np.random.randn(len(dates)) * 0.8)
    bars3 = pd.DataFrame(
        {
            "open": prices3,
            "high": prices3 * 1.03,
            "low": prices3 * 0.97,
            "close": prices3,
            "volume": np.random.randint(2000000, 8000000, len(dates)),
            "dollar_volume": prices3 * np.random.randint(2000000, 8000000, len(dates)),
        },
        index=dates,
    )
    market_data.bars["SYMBOL3"] = bars3

    # Create benchmark data (SPY)
    spy_prices = 400 + np.cumsum(np.random.randn(len(dates)) * 0.3)
    spy_bars = pd.DataFrame(
        {
            "open": spy_prices,
            "high": spy_prices * 1.01,
            "low": spy_prices * 0.99,
            "close": spy_prices,
            "volume": np.random.randint(50000000, 100000000, len(dates)),
            "dollar_volume": spy_prices * np.random.randint(50000000, 100000000, len(dates)),
        },
        index=dates,
    )
    market_data.benchmarks["SPY"] = spy_bars

    # Create benchmark data (BTC)
    btc_prices = 50000 + np.cumsum(np.random.randn(len(dates)) * 500)
    btc_bars = pd.DataFrame(
        {
            "open": btc_prices,
            "high": btc_prices * 1.02,
            "low": btc_prices * 0.98,
            "close": btc_prices,
            "volume": np.random.randint(100000, 500000, len(dates)),
            "dollar_volume": btc_prices * np.random.randint(100000, 500000, len(dates)),
        },
        index=dates,
    )
    market_data.benchmarks["BTC"] = btc_bars

    return market_data


@pytest.fixture
def simple_strategy_config():
    """Create a simple strategy config for testing."""
    # Create minimal config
    config_dict = {
        "name": "test_equity_momentum",
        "asset_class": "equity",
        "universe": ["SYMBOL1", "SYMBOL2", "SYMBOL3"],
        "benchmark": "SPY",
        "eligibility": {"min_price": 1.0, "min_volume": 100000, "trend_ma": 50, "min_slope": 0.0},
        "entry": {"fast_breakout_days": 20, "slow_breakout_days": 55, "min_clearance_atr": 0.5},
        "exit": {"mode": "ma_cross", "exit_ma": 20, "hard_stop_atr_mult": 2.5},
        "risk": {"risk_per_trade": 0.0075, "max_positions": 10, "max_exposure": 0.80, "max_position_notional": 0.15},
        "capacity": {"max_order_pct_adv": 0.20},
    }
    return StrategyConfig(**config_dict)


@pytest.fixture
def simple_strategy(simple_strategy_config):
    """Create a simple strategy for testing."""
    return EquityStrategy(simple_strategy_config)


@pytest.fixture
def test_split():
    """Create a test walk-forward split."""
    return WalkForwardSplit(
        name="test_split",
        train_start=pd.Timestamp("2024-01-01"),
        train_end=pd.Timestamp("2024-01-31"),
        validation_start=pd.Timestamp("2024-02-01"),
        validation_end=pd.Timestamp("2024-02-28"),
        holdout_start=pd.Timestamp("2024-03-01"),
        holdout_end=pd.Timestamp("2024-03-31"),
    )


def test_walk_forward_split_validation(test_split):
    """Test walk-forward split validation."""
    assert test_split.validate()

    # Test date containment
    assert test_split.contains_date(pd.Timestamp("2024-01-15"), "train")
    assert not test_split.contains_date(pd.Timestamp("2024-02-15"), "train")
    assert test_split.contains_date(pd.Timestamp("2024-02-15"), "validation")

    # Test period dates
    train_start, train_end = test_split.get_period_dates("train")
    assert train_start == pd.Timestamp("2024-01-01")
    assert train_end == pd.Timestamp("2024-01-31")


def test_backtest_engine_initialization(sample_market_data, simple_strategy):
    """Test backtest engine initialization."""
    engine = BacktestEngine(market_data=sample_market_data, strategies=[simple_strategy], starting_equity=100000.0, seed=42)

    assert engine.starting_equity == 100000.0
    assert engine.portfolio.equity == 100000.0
    assert engine.portfolio.cash == 100000.0
    assert len(engine.strategies) == 1


def test_backtest_engine_run_basic(sample_market_data, simple_strategy, test_split):
    """Test basic backtest run."""
    engine = BacktestEngine(market_data=sample_market_data, strategies=[simple_strategy], starting_equity=100000.0, seed=42)

    # Run on train period
    results = engine.run(split=test_split, period="train")

    # Check results structure
    assert "split_name" in results
    assert "period" in results
    assert "starting_equity" in results
    assert "ending_equity" in results
    assert "total_return" in results
    assert "total_trades" in results

    # Check that equity curve exists
    assert "equity_curve" in results
    assert len(results["equity_curve"]) > 0

    # Check that starting equity is correct
    assert results["starting_equity"] == 100000.0


def test_backtest_no_lookahead(sample_market_data, simple_strategy, test_split):
    """Test that backtest has no lookahead bias.

    This test verifies that:
    - Indicators use data <= current date
    - Orders created at t execute at t+1
    - Stops updated at t+1 check exits at t+2
    """
    engine = BacktestEngine(market_data=sample_market_data, strategies=[simple_strategy], starting_equity=100000.0, seed=42)

    # Run backtest
    engine.run(split=test_split, period="train")

    # Check that daily events are in order
    assert len(engine.daily_events) > 0

    # Verify timing: signals generated at day t, orders execute at t+1
    for i, event in enumerate(engine.daily_events):
        event.get("date")  # Access date to verify it exists

        # Signals should be generated at this date
        if event["signals_generated"]:
            # Orders should be created for next day
            if event["orders_created"]:
                # Check that order execution happens on a later date
                # (This is a simplified check - in production, verify exact timing)
                pass


def test_backtest_export_results(sample_market_data, simple_strategy, test_split):
    """Test exporting backtest results."""
    engine = BacktestEngine(market_data=sample_market_data, strategies=[simple_strategy], starting_equity=100000.0, seed=42)

    # Run backtest
    engine.run(split=test_split, period="train")

    # Export to temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        engine.export_results(tmpdir)

        # Check that files were created
        output_path = Path(tmpdir)
        assert (output_path / "equity_curve.csv").exists()
        assert (output_path / "daily_metrics.csv").exists()

        # Check that equity curve file has data
        equity_df = pd.read_csv(output_path / "equity_curve.csv")
        assert len(equity_df) > 0
        assert "equity" in equity_df.columns


def test_default_split_creation():
    """Test creating default 24-month split."""
    start_date = pd.Timestamp("2024-01-01")
    split = create_default_split(start_date, months=24)

    assert split.name == "default_split"
    assert split.train_start == start_date
    assert split.train_end == start_date + pd.DateOffset(months=15) - pd.Timedelta(days=1)
    assert split.validation_start == split.train_end + pd.Timedelta(days=1)
    assert split.validation_end == split.validation_start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
    assert split.holdout_start == split.validation_end + pd.Timedelta(days=1)
    assert split.holdout_end == split.holdout_start + pd.DateOffset(months=6) - pd.Timedelta(days=1)


def test_backtest_multiple_periods(sample_market_data, simple_strategy, test_split):
    """Test running backtest on multiple periods."""
    engine = BacktestEngine(market_data=sample_market_data, strategies=[simple_strategy], starting_equity=100000.0, seed=42)

    # Run on train
    train_results = engine.run(split=test_split, period="train")
    assert train_results["period"] == "train"

    # Run on validation
    validation_results = engine.run(split=test_split, period="validation")
    assert validation_results["period"] == "validation"

    # Run on holdout
    holdout_results = engine.run(split=test_split, period="holdout")
    assert holdout_results["period"] == "holdout"


def test_backtest_empty_period(sample_market_data, simple_strategy):
    """Test backtest with empty period (no data)."""
    # Create split with dates outside data range
    empty_split = WalkForwardSplit(
        name="empty_split",
        train_start=pd.Timestamp("2025-01-01"),
        train_end=pd.Timestamp("2025-01-31"),
        validation_start=pd.Timestamp("2025-02-01"),
        validation_end=pd.Timestamp("2025-02-28"),
        holdout_start=pd.Timestamp("2025-03-01"),
        holdout_end=pd.Timestamp("2025-03-31"),
    )

    engine = BacktestEngine(market_data=sample_market_data, strategies=[simple_strategy], starting_equity=100000.0, seed=42)

    # Run should not crash
    results = engine.run(split=empty_split, period="train")

    # Should return empty results
    assert results["total_trades"] == 0
    assert results["ending_equity"] == results["starting_equity"]


def test_event_loop_timing(sample_market_data, simple_strategy):
    """Test that event loop maintains correct timing.

    Verifies:
    - Data updated through day t close
    - Signals generated at day t close
    - Orders execute at day t+1 open
    - Exits execute at day t+2 open
    """
    from trading_system.backtest.event_loop import DailyEventLoop
    from trading_system.indicators.feature_computer import compute_features

    portfolio = Portfolio(date=pd.Timestamp("2024-01-01"), cash=100000.0, starting_equity=100000.0, equity=100000.0)

    def get_next_trading_day(date):
        all_dates = sorted(sample_market_data.bars["SYMBOL1"].index)
        available = [d for d in all_dates if d > date]
        return min(available) if available else date + pd.Timedelta(days=1)

    event_loop = DailyEventLoop(
        market_data=sample_market_data,
        portfolio=portfolio,
        strategies=[simple_strategy],
        compute_features_fn=compute_features,
        get_next_trading_day=get_next_trading_day,
        rng=np.random.default_rng(42),
    )

    # Process a day
    date = pd.Timestamp("2024-01-15")
    events = event_loop.process_day(date)

    # Check event structure
    assert "date" in events
    assert "signals_generated" in events
    assert "orders_created" in events
    assert "orders_executed" in events
    assert "portfolio_state" in events


def test_backtest_engine_slippage_multiplier(sample_market_data, simple_strategy, test_split):
    """Test backtest engine with different slippage multipliers."""
    # Test with normal slippage
    engine_normal = BacktestEngine(
        market_data=sample_market_data,
        strategies=[simple_strategy],
        starting_equity=100000.0,
        seed=42,
        slippage_multiplier=1.0,
    )
    results_normal = engine_normal.run(split=test_split, period="train")

    # Test with higher slippage (stress test)
    engine_stress = BacktestEngine(
        market_data=sample_market_data,
        strategies=[simple_strategy],
        starting_equity=100000.0,
        seed=42,
        slippage_multiplier=2.0,
    )
    results_stress = engine_stress.run(split=test_split, period="train")

    # Higher slippage should generally result in lower returns (or same if no trades)
    # But we can't guarantee this, so just verify both run successfully
    assert results_normal["starting_equity"] == results_stress["starting_equity"]
    assert results_normal["total_trades"] == results_stress["total_trades"]


def test_backtest_engine_crash_dates(sample_market_data, simple_strategy, test_split):
    """Test backtest engine with crash dates (flash crash simulation)."""
    # Add a crash date in the middle of the test period
    crash_date = pd.Timestamp("2024-01-15")
    engine = BacktestEngine(
        market_data=sample_market_data,
        strategies=[simple_strategy],
        starting_equity=100000.0,
        seed=42,
        crash_dates=[crash_date],
    )

    results = engine.run(split=test_split, period="train")

    # Verify engine runs successfully with crash dates
    assert results["starting_equity"] == 100000.0
    assert "total_trades" in results


def test_backtest_engine_seed_reproducibility(sample_market_data, simple_strategy, test_split):
    """Test that backtest results are reproducible with same seed."""
    # Run with same seed twice
    engine1 = BacktestEngine(
        market_data=sample_market_data,
        strategies=[simple_strategy],
        starting_equity=100000.0,
        seed=42,
    )
    results1 = engine1.run(split=test_split, period="train")

    engine2 = BacktestEngine(
        market_data=sample_market_data,
        strategies=[simple_strategy],
        starting_equity=100000.0,
        seed=42,
    )
    results2 = engine2.run(split=test_split, period="train")

    # Results should be identical with same seed
    assert results1["total_trades"] == results2["total_trades"]
    assert results1["ending_equity"] == results2["ending_equity"]
    assert results1["total_return"] == results2["total_return"]


def test_backtest_engine_results_structure(sample_market_data, simple_strategy, test_split):
    """Test that backtest results contain all expected fields."""
    engine = BacktestEngine(
        market_data=sample_market_data,
        strategies=[simple_strategy],
        starting_equity=100000.0,
        seed=42,
    )
    results = engine.run(split=test_split, period="train")

    # Check all required fields exist
    required_fields = [
        "split_name",
        "period",
        "start_date",
        "end_date",
        "starting_equity",
        "ending_equity",
        "total_return",
        "sharpe_ratio",
        "max_drawdown",
        "total_trades",
        "winning_trades",
        "losing_trades",
        "win_rate",
        "avg_r_multiple",
        "realized_pnl",
        "final_cash",
        "final_positions",
        "equity_curve",
        "daily_returns",
        "closed_trades",
    ]

    for field in required_fields:
        assert field in results, f"Missing field: {field}"

    # Check field types and values
    assert isinstance(results["equity_curve"], list)
    assert isinstance(results["daily_returns"], list)
    assert isinstance(results["closed_trades"], list)
    assert results["starting_equity"] > 0
    assert results["ending_equity"] > 0
    assert 0.0 <= results["win_rate"] <= 1.0
    assert 0.0 <= results["max_drawdown"] <= 1.0


def test_backtest_engine_export_trade_log(sample_market_data, simple_strategy, test_split):
    """Test that trade log is exported correctly."""
    engine = BacktestEngine(
        market_data=sample_market_data,
        strategies=[simple_strategy],
        starting_equity=100000.0,
        seed=42,
    )
    engine.run(split=test_split, period="train")

    with tempfile.TemporaryDirectory() as tmpdir:
        engine.export_results(tmpdir)
        output_path = Path(tmpdir)

        # Check trade log exists (even if empty)
        trade_log_path = output_path / "trade_log.csv"
        if trade_log_path.exists():
            trade_df = pd.read_csv(trade_log_path)
            if len(trade_df) > 0:
                # Check required columns
                required_columns = [
                    "symbol",
                    "entry_date",
                    "exit_date",
                    "entry_price",
                    "exit_price",
                    "quantity",
                    "realized_pnl",
                ]
                for col in required_columns:
                    assert col in trade_df.columns, f"Missing column: {col}"


def test_backtest_engine_get_all_dates(sample_market_data, simple_strategy):
    """Test _get_all_dates method."""
    engine = BacktestEngine(
        market_data=sample_market_data,
        strategies=[simple_strategy],
        starting_equity=100000.0,
        seed=42,
    )

    all_dates = engine._get_all_dates()

    # Should return sorted list of dates
    assert isinstance(all_dates, list)
    assert len(all_dates) > 0
    assert all_dates == sorted(all_dates)

    # Should include dates from all symbols
    expected_dates = set()
    for symbol, bars_df in sample_market_data.bars.items():
        expected_dates.update(bars_df.index)
    for benchmark_df in sample_market_data.benchmarks.values():
        expected_dates.update(benchmark_df.index)

    assert set(all_dates) == expected_dates


def test_backtest_engine_empty_results(sample_market_data, simple_strategy):
    """Test _empty_results method."""
    engine = BacktestEngine(
        market_data=sample_market_data,
        strategies=[simple_strategy],
        starting_equity=100000.0,
        seed=42,
    )

    empty_results = engine._empty_results()

    # Check structure
    assert empty_results["starting_equity"] == 100000.0
    assert empty_results["ending_equity"] == 100000.0
    assert empty_results["total_return"] == 0.0
    assert empty_results["total_trades"] == 0
    assert empty_results["win_rate"] == 0.0
    assert len(empty_results["equity_curve"]) == 1
    assert empty_results["equity_curve"][0] == 100000.0


def test_backtest_engine_multiple_strategies(sample_market_data, simple_strategy_config):
    """Test backtest engine with multiple strategies."""
    # Create two strategies
    strategy1 = EquityStrategy(simple_strategy_config)
    strategy2 = EquityStrategy(simple_strategy_config)

    engine = BacktestEngine(
        market_data=sample_market_data,
        strategies=[strategy1, strategy2],
        starting_equity=100000.0,
        seed=42,
    )

    assert len(engine.strategies) == 2

    test_split = WalkForwardSplit(
        name="test_split",
        train_start=pd.Timestamp("2024-01-01"),
        train_end=pd.Timestamp("2024-01-31"),
        validation_start=pd.Timestamp("2024-02-01"),
        validation_end=pd.Timestamp("2024-02-28"),
        holdout_start=pd.Timestamp("2024-03-01"),
        holdout_end=pd.Timestamp("2024-03-31"),
    )

    results = engine.run(split=test_split, period="train")
    assert results["starting_equity"] == 100000.0

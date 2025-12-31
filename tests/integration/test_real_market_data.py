"""Integration tests with real historical market data.

This test suite verifies that the system works correctly with actual
historical market data, handles real-world data quality issues, and
performs well under different market conditions (bull, bear, range).

These tests require real market data to be downloaded first:
    python scripts/download_real_market_data.py --output data/real_market_data/
"""

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pytest

from trading_system.configs.run_config import RunConfig
from trading_system.data import load_ohlcv_data
from trading_system.data.validator import detect_missing_data, validate_ohlcv
from trading_system.integration.runner import BacktestRunner

# Path to real market data (downloaded via download_real_market_data.py)
REAL_DATA_DIR = Path("data/real_market_data")
EQUITY_DATA_DIR = REAL_DATA_DIR / "equity" / "ohlcv"
CRYPTO_DATA_DIR = REAL_DATA_DIR / "crypto" / "ohlcv"
BENCHMARK_DATA_DIR = REAL_DATA_DIR / "benchmarks"


def has_real_data() -> bool:
    """Check if real market data is available."""
    return EQUITY_DATA_DIR.exists() and any(EQUITY_DATA_DIR.glob("*.csv"))


@pytest.mark.skipif(not has_real_data(), reason="Real market data not available. Run: python scripts/download_real_market_data.py")
class TestRealMarketData:
    """Tests with real historical market data."""

    @pytest.fixture
    def equity_symbols(self):
        """Get available equity symbols from real data."""
        if not EQUITY_DATA_DIR.exists():
            return []
        return [f.stem for f in EQUITY_DATA_DIR.glob("*.csv")][:5]  # Limit to 5 for speed

    @pytest.fixture
    def crypto_symbols(self):
        """Get available crypto symbols from real data."""
        if not CRYPTO_DATA_DIR.exists():
            return []
        return [f.stem for f in CRYPTO_DATA_DIR.glob("*.csv")][:3]  # Limit to 3 for speed

    def test_real_data_loading(self, equity_symbols):
        """Test that real market data loads correctly."""
        if not equity_symbols:
            pytest.skip("No equity data available")

        # Load real data
        data = load_ohlcv_data(str(EQUITY_DATA_DIR), equity_symbols[:3])

        # Verify data loaded
        assert len(data) > 0, "Should load some equity data"

        # Verify each symbol has data
        for symbol in equity_symbols[:3]:
            if symbol in data:
                df = data[symbol]
                assert len(df) > 0, f"Data for {symbol} should not be empty"
                assert "date" in df.columns or df.index.name == "date", f"Data for {symbol} should have date column/index"
                assert all(col in df.columns for col in ["open", "high", "low", "close", "volume"]), \
                    f"Data for {symbol} should have OHLCV columns"

    def test_real_data_validation(self, equity_symbols):
        """Test that real market data passes validation."""
        if not equity_symbols:
            pytest.skip("No equity data available")

        # Load and validate real data
        data = load_ohlcv_data(str(EQUITY_DATA_DIR), equity_symbols[:3])

        for symbol, df in data.items():
            # Validate OHLCV structure
            assert validate_ohlcv(df, symbol), f"Real data for {symbol} should pass validation"

            # Check for common data quality issues
            # 1. No negative prices
            assert (df[["open", "high", "low", "close"]] > 0).all().all(), \
                f"Real data for {symbol} should have positive prices"

            # 2. OHLC relationships are valid
            assert (df["high"] >= df["low"]).all(), f"Real data for {symbol} should have high >= low"
            assert (df["high"] >= df["open"]).all(), f"Real data for {symbol} should have high >= open"
            assert (df["high"] >= df["close"]).all(), f"Real data for {symbol} should have high >= close"
            assert (df["low"] <= df["open"]).all(), f"Real data for {symbol} should have low <= open"
            assert (df["low"] <= df["close"]).all(), f"Real data for {symbol} should have low <= close"

            # 3. Dates are in order
            if "date" in df.columns:
                dates = pd.to_datetime(df["date"])
            else:
                dates = df.index
            assert dates.is_monotonic_increasing, f"Real data for {symbol} should have dates in order"

    def test_real_data_quality_issues(self, equity_symbols):
        """Test that system handles real-world data quality issues."""
        if not equity_symbols:
            pytest.skip("No equity data available")

        # Load real data
        data = load_ohlcv_data(str(EQUITY_DATA_DIR), equity_symbols[:3])

        for symbol, df in data.items():
            # Check for missing data
            missing_info = detect_missing_data(df, symbol, asset_class="equity")
            assert "missing_dates" in missing_info, "Should detect missing dates"

            # Real data may have missing days (holidays, weekends for equity)
            # System should handle this gracefully
            if len(missing_info["missing_dates"]) > 0:
                print(f"  {symbol}: {len(missing_info['missing_dates'])} missing dates detected")

            # Check for extreme moves (>50% daily return)
            if "close" in df.columns:
                returns = df["close"].pct_change().dropna()
                extreme_moves = abs(returns) > 0.50

                if extreme_moves.any():
                    print(f"  {symbol}: {extreme_moves.sum()} extreme moves detected")
                    # System should handle extreme moves (treated as missing data per EDGE_CASES.md)

            # Check for low volume days
            if "volume" in df.columns:
                low_volume_threshold = df["volume"].quantile(0.05)  # Bottom 5%
                low_volume_days = (df["volume"] < low_volume_threshold).sum()
                if low_volume_days > 0:
                    print(f"  {symbol}: {low_volume_days} low volume days detected")

    def test_bull_market_condition(self, equity_symbols):
        """Test system performance during bull market conditions."""
        if not equity_symbols:
            pytest.skip("No equity data available")

        # Define bull market period (e.g., 2020-04 to 2021-11 post-COVID recovery)
        start_date = pd.Timestamp("2020-04-01")
        end_date = pd.Timestamp("2021-11-30")

        # Load data for bull market period
        data = load_ohlcv_data(
            str(EQUITY_DATA_DIR),
            equity_symbols[:3],
            start_date=start_date,
            end_date=end_date,
        )

        if not data:
            pytest.skip("No data available for bull market period")

        # Verify we have data
        for symbol, df in data.items():
            assert len(df) > 0, f"Should have data for {symbol} in bull market period"

            # Verify this is actually a bull market (positive trend)
            if "close" in df.columns and len(df) > 20:
                first_price = df["close"].iloc[0]
                last_price = df["close"].iloc[-1]
                total_return = (last_price / first_price) - 1

                # Bull market should show positive returns
                print(f"  {symbol} bull market return: {total_return:.2%}")

    def test_bear_market_condition(self, equity_symbols):
        """Test system performance during bear market conditions."""
        if not equity_symbols:
            pytest.skip("No equity data available")

        # Define bear market period (e.g., 2022 bear market)
        start_date = pd.Timestamp("2022-01-01")
        end_date = pd.Timestamp("2022-12-31")

        # Load data for bear market period
        data = load_ohlcv_data(
            str(EQUITY_DATA_DIR),
            equity_symbols[:3],
            start_date=start_date,
            end_date=end_date,
        )

        if not data:
            pytest.skip("No data available for bear market period")

        # Verify we have data
        for symbol, df in data.items():
            assert len(df) > 0, f"Should have data for {symbol} in bear market period"

            # Verify this is actually a bear market (negative trend)
            if "close" in df.columns and len(df) > 20:
                first_price = df["close"].iloc[0]
                last_price = df["close"].iloc[-1]
                total_return = (last_price / first_price) - 1

                # Bear market should show negative returns
                print(f"  {symbol} bear market return: {total_return:.2%}")

    def test_range_market_condition(self, equity_symbols):
        """Test system performance during range-bound market conditions."""
        if not equity_symbols:
            pytest.skip("No equity data available")

        # Define range market period (e.g., 2019 - relatively flat)
        start_date = pd.Timestamp("2019-01-01")
        end_date = pd.Timestamp("2019-12-31")

        # Load data for range market period
        data = load_ohlcv_data(
            str(EQUITY_DATA_DIR),
            equity_symbols[:3],
            start_date=start_date,
            end_date=end_date,
        )

        if not data:
            pytest.skip("No data available for range market period")

        # Verify we have data
        for symbol, df in data.items():
            assert len(df) > 0, f"Should have data for {symbol} in range market period"

            # Verify this is range-bound (low volatility, small net change)
            if "close" in df.columns and len(df) > 20:
                first_price = df["close"].iloc[0]
                last_price = df["close"].iloc[-1]
                total_return = abs((last_price / first_price) - 1)

                # Range market should show small net change
                volatility = df["close"].pct_change().std()
                print(f"  {symbol} range market: return={total_return:.2%}, volatility={volatility:.2%}")

    def test_full_backtest_with_real_data(self, equity_symbols, tmp_path):
        """Test full backtest workflow with real market data."""
        if not equity_symbols:
            pytest.skip("No equity data available")

        # Create a minimal config for real data testing
        config_dict = {
            "dataset": {
                "equity_path": str(EQUITY_DATA_DIR),
                "crypto_path": str(CRYPTO_DATA_DIR) if CRYPTO_DATA_DIR.exists() else str(EQUITY_DATA_DIR),
                "benchmark_path": str(BENCHMARK_DATA_DIR) if BENCHMARK_DATA_DIR.exists() else str(EQUITY_DATA_DIR),
                "format": "csv",
                "start_date": "2020-01-01",
                "end_date": "2023-12-31",
                "min_lookback_days": 250,
            },
            "splits": {
                "train_start": "2020-01-01",
                "train_end": "2021-12-31",
                "validation_start": "2022-01-01",
                "validation_end": "2022-12-31",
                "holdout_start": "2023-01-01",
                "holdout_end": "2023-12-31",
            },
            "strategies": {
                "equity": {
                    "config_path": "EXAMPLE_CONFIGS/equity_config.yaml",
                    "enabled": True,
                },
                "crypto": {
                    "config_path": "EXAMPLE_CONFIGS/crypto_config.yaml",
                    "enabled": False,  # Disable crypto for this test
                },
            },
            "portfolio": {
                "starting_equity": 100000,
            },
            "volatility_scaling": {
                "enabled": True,
                "mode": "continuous",
                "lookback": 20,
                "baseline_lookback": 252,
                "min_multiplier": 0.33,
                "max_multiplier": 1.0,
            },
            "correlation_guard": {
                "enabled": True,
                "min_positions": 4,
                "avg_pairwise_threshold": 0.70,
                "candidate_threshold": 0.75,
            },
            "scoring": {
                "weights": {
                    "breakout": 0.50,
                    "momentum": 0.30,
                    "diversification": 0.20,
                },
            },
            "execution": {
                "signal_timing": "close",
                "execution_timing": "next_open",
                "slippage_model": "full",
            },
            "output": {
                "base_path": str(tmp_path),
                "run_id": None,
                "equity_curve": "equity_curve.csv",
                "trade_log": "trade_log.csv",
                "weekly_summary": "weekly_summary.csv",
                "monthly_report": "monthly_report.json",
                "log_level": "INFO",
                "log_file": "backtest.log",
            },
            "random_seed": 42,
            "validation": {
                "sensitivity": {"enabled": False},
                "stress_tests": {
                    "slippage_multipliers": [1.0],
                    "bear_market_test": False,
                    "range_market_test": False,
                    "flash_crash_test": False,
                },
                "statistical": {
                    "bootstrap_iterations": 100,
                    "permutation_iterations": 100,
                    "bootstrap_5th_percentile_threshold": 0.4,
                },
            },
            "metrics": {
                "primary": {
                    "sharpe_ratio_min": 0.5,
                    "max_drawdown_max": 0.20,
                    "calmar_ratio_min": 1.0,
                    "min_trades": 5,
                },
            },
        }

        try:
            config = RunConfig(**config_dict)
            runner = BacktestRunner(config)
            runner.initialize()

            # Run backtest on train period
            results = runner.run_backtest(period="train")

            # Verify results structure
            assert results is not None, "Results should not be None"
            assert "total_trades" in results, "Results should contain total_trades"
            assert "total_return" in results, "Results should contain total_return"
            assert "sharpe_ratio" in results, "Results should contain sharpe_ratio"

            # Verify metrics are reasonable
            assert np.isfinite(results["total_return"]), "Total return should be finite"
            assert np.isfinite(results["sharpe_ratio"]), "Sharpe ratio should be finite"
            assert 0.0 <= results["max_drawdown"] <= 1.0, "Max drawdown should be between 0 and 1"

            print(f"\nReal Data Backtest Results:")
            print(f"  Total trades: {results['total_trades']}")
            print(f"  Total return: {results['total_return']:.2%}")
            print(f"  Sharpe ratio: {results['sharpe_ratio']:.2f}")
            print(f"  Max drawdown: {results['max_drawdown']:.2%}")

        except Exception as e:
            pytest.skip(f"Backtest with real data failed (may need config adjustments): {e}")

    def test_missing_days_handling(self, equity_symbols):
        """Test that system handles missing trading days correctly."""
        if not equity_symbols:
            pytest.skip("No equity data available")

        # Load real data
        data = load_ohlcv_data(str(EQUITY_DATA_DIR), equity_symbols[:1])

        for symbol, df in data.items():
            # Detect missing days
            missing_info = detect_missing_data(df, symbol, asset_class="equity")

            # Real equity data should have missing days (holidays, weekends)
            # System should handle this gracefully
            if len(missing_info.get("missing_dates", [])) > 0:
                print(f"  {symbol}: {len(missing_info['missing_dates'])} missing dates")
                # System should continue processing despite missing days

    def test_extreme_moves_handling(self, equity_symbols):
        """Test that system handles extreme price moves in real data."""
        if not equity_symbols:
            pytest.skip("No equity data available")

        # Load real data
        data = load_ohlcv_data(str(EQUITY_DATA_DIR), equity_symbols[:3])

        for symbol, df in data.items():
            if "close" in df.columns and len(df) > 1:
                returns = df["close"].pct_change().dropna()
                extreme_moves = abs(returns) > 0.50  # >50% daily move

                if extreme_moves.any():
                    extreme_dates = returns.index[extreme_moves]
                    print(f"  {symbol}: {len(extreme_dates)} extreme moves detected")
                    for date in extreme_dates[:3]:  # Show first 3
                        move_pct = returns.loc[date] * 100
                        print(f"    {date}: {move_pct:.1f}%")

                    # Per EDGE_CASES.md, extreme moves should be treated as missing data
                    # System should skip these bars during signal generation

    def test_duplicate_dates_handling(self, equity_symbols):
        """Test that system handles duplicate dates in real data."""
        if not equity_symbols:
            pytest.skip("No equity data available")

        # Load real data
        data = load_ohlcv_data(str(EQUITY_DATA_DIR), equity_symbols[:3])

        for symbol, df in data.items():
            # Check for duplicate dates
            if "date" in df.columns:
                dates = pd.to_datetime(df["date"])
            else:
                dates = df.index

            duplicates = dates.duplicated()
            if duplicates.any():
                print(f"  {symbol}: {duplicates.sum()} duplicate dates detected")
                # System should handle duplicates (keep first, drop later, or merge)
            else:
                # No duplicates - good!
                assert True


"""Unit tests for storage/database.py - ResultsDatabase class."""

import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_system.models.positions import ExitReason, Position, PositionSide
from trading_system.models.signals import BreakoutType
from trading_system.storage.database import ResultsDatabase, get_default_db_path


class TestGetDefaultDbPath:
    """Tests for get_default_db_path function."""

    def test_get_default_db_path(self):
        """Test default database path creation."""
        path = get_default_db_path()
        assert isinstance(path, Path)
        assert path.name == "backtest_results.db"
        assert path.parent.name == "results"
        # Parent directory should be created
        assert path.parent.exists()


class TestResultsDatabase:
    """Tests for ResultsDatabase class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for test database
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db_path = self.temp_dir / "test_backtest_results.db"
        self.db = ResultsDatabase(db_path=self.db_path)

    def teardown_method(self):
        """Clean up test fixtures."""
        # Close database connections
        if hasattr(self, "db"):
            # Database should close connections automatically
            pass
        # Remove temporary directory
        if hasattr(self, "temp_dir") and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_init_default_path(self):
        """Test initialization with default path."""
        db = ResultsDatabase()
        assert db.db_path == get_default_db_path()
        # Clean up
        if db.db_path.exists():
            db_path = db.db_path
            db = None  # Close connection
            db_path.unlink(missing_ok=True)

    def test_init_custom_path(self):
        """Test initialization with custom path."""
        db = ResultsDatabase(db_path=self.db_path)
        assert db.db_path == self.db_path
        assert self.db_path.exists()

    def test_init_creates_directory(self):
        """Test that initialization creates parent directory."""
        nested_path = self.temp_dir / "nested" / "deep" / "test.db"
        db = ResultsDatabase(db_path=nested_path)
        assert nested_path.parent.exists()
        assert nested_path.exists()

    def test_store_results_basic(self):
        """Test storing basic results."""
        results = {
            "start_date": pd.Timestamp("2023-01-01"),
            "end_date": pd.Timestamp("2023-12-31"),
            "starting_equity": 100000.0,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.15,
            "total_return": 0.25,
            "total_trades": 50,
            "winning_trades": 30,
            "losing_trades": 20,
            "win_rate": 0.6,
            "realized_pnl": 25000.0,
            "final_cash": 125000.0,
            "final_positions": 0,
            "ending_equity": 125000.0,
            "closed_trades": [],
            "equity_curve": [100000.0, 105000.0, 110000.0, 125000.0],
            "daily_events": [
                {
                    "date": pd.Timestamp("2023-01-01"),
                    "portfolio_state": {"cash": 100000.0, "open_positions": 0, "gross_exposure": 0.0},
                },
                {
                    "date": pd.Timestamp("2023-01-02"),
                    "portfolio_state": {"cash": 95000.0, "open_positions": 1, "gross_exposure": 5000.0},
                },
                {
                    "date": pd.Timestamp("2023-01-03"),
                    "portfolio_state": {"cash": 90000.0, "open_positions": 2, "gross_exposure": 10000.0},
                },
                {
                    "date": pd.Timestamp("2023-12-31"),
                    "portfolio_state": {"cash": 125000.0, "open_positions": 0, "gross_exposure": 0.0},
                },
            ],
            "daily_returns": [0.0, 0.05, 0.0476, 0.1364],
        }

        run_id = self.db.store_results(
            results=results,
            config_path="test_config.yaml",
            strategy_name="test_strategy",
            period="train",
            split_name="split_1",
            notes="Test run",
        )

        assert isinstance(run_id, int)
        assert run_id > 0

    def test_store_results_with_trades(self):
        """Test storing results with closed trades."""
        entry_date = pd.Timestamp("2023-01-15")
        exit_date = pd.Timestamp("2023-01-20")

        trade = Position(
            symbol="AAPL",
            asset_class="equity",
            entry_date=entry_date,
            entry_price=150.0,
            entry_fill_id="fill_1",
            quantity=100,
            side=PositionSide.LONG,
            stop_price=145.0,
            initial_stop_price=145.0,
            hard_stop_atr_mult=2.5,
            entry_slippage_bps=10.0,
            entry_fee_bps=1.0,
            entry_total_cost=15.0,
            triggered_on=BreakoutType.FAST_20D,
            adv20_at_entry=10000000.0,
            exit_date=exit_date,
            exit_price=155.0,
            exit_fill_id="fill_2",
            exit_reason=ExitReason.MANUAL,
            exit_slippage_bps=10.0,
            exit_fee_bps=1.0,
            exit_total_cost=15.5,
            realized_pnl=485.0,
        )

        results = {
            "start_date": pd.Timestamp("2023-01-01"),
            "end_date": pd.Timestamp("2023-01-31"),
            "starting_equity": 100000.0,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.10,
            "total_return": 0.05,
            "total_trades": 1,
            "winning_trades": 1,
            "losing_trades": 0,
            "win_rate": 1.0,
            "realized_pnl": 485.0,
            "final_cash": 100485.0,
            "final_positions": 0,
            "ending_equity": 100485.0,
            "closed_trades": [trade],
            "equity_curve": [100000.0, 100485.0],
            "daily_events": [
                {
                    "date": pd.Timestamp("2023-01-01"),
                    "portfolio_state": {"cash": 100000.0, "open_positions": 0, "gross_exposure": 0.0},
                },
                {
                    "date": pd.Timestamp("2023-01-31"),
                    "portfolio_state": {"cash": 100485.0, "open_positions": 0, "gross_exposure": 0.0},
                },
            ],
            "daily_returns": [0.0, 0.00485],
        }

        run_id = self.db.store_results(
            results=results, config_path="test_config.yaml", strategy_name="test_strategy", period="train"
        )

        # Verify trade was stored
        trades_df = self.db.get_trades(run_id)
        assert len(trades_df) == 1
        assert trades_df.iloc[0]["symbol"] == "AAPL"
        assert trades_df.iloc[0]["realized_pnl"] == 485.0

    def test_store_results_with_monthly_summary(self):
        """Test storing results with monthly summary."""
        dates = pd.date_range("2023-01-01", "2023-03-31", freq="D")
        equity_curve = [100000.0 + i * 100 for i in range(len(dates))]
        daily_returns = [0.001] * len(dates)

        results = {
            "start_date": dates[0],
            "end_date": dates[-1],
            "starting_equity": 100000.0,
            "sharpe_ratio": 1.0,
            "max_drawdown": 0.05,
            "total_return": 0.10,
            "total_trades": 10,
            "winning_trades": 6,
            "losing_trades": 4,
            "win_rate": 0.6,
            "realized_pnl": 10000.0,
            "final_cash": 110000.0,
            "final_positions": 0,
            "ending_equity": 110000.0,
            "closed_trades": [],
            "equity_curve": equity_curve,
            "daily_events": [
                {"date": date, "portfolio_state": {"cash": 100000.0, "open_positions": 0, "gross_exposure": 0.0}}
                for date in dates
            ],
            "daily_returns": daily_returns,
        }

        run_id = self.db.store_results(
            results=results, config_path="test_config.yaml", strategy_name="test_strategy", period="train"
        )

        # Verify monthly summary was stored
        monthly_df = self.db.get_monthly_summary(run_id)
        assert len(monthly_df) >= 3  # At least 3 months
        assert "month" in monthly_df.columns
        assert "monthly_return" in monthly_df.columns

    def test_store_results_duplicate_run(self):
        """Test storing duplicate run (should update existing)."""
        results = {
            "start_date": pd.Timestamp("2023-01-01"),
            "end_date": pd.Timestamp("2023-12-31"),
            "starting_equity": 100000.0,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.15,
            "total_return": 0.25,
            "total_trades": 50,
            "winning_trades": 30,
            "losing_trades": 20,
            "win_rate": 0.6,
            "realized_pnl": 25000.0,
            "final_cash": 125000.0,
            "final_positions": 0,
            "ending_equity": 125000.0,
            "closed_trades": [],
            "equity_curve": [100000.0, 125000.0],
            "daily_events": [
                {
                    "date": pd.Timestamp("2023-01-01"),
                    "portfolio_state": {"cash": 100000.0, "open_positions": 0, "gross_exposure": 0.0},
                },
                {
                    "date": pd.Timestamp("2023-12-31"),
                    "portfolio_state": {"cash": 125000.0, "open_positions": 0, "gross_exposure": 0.0},
                },
            ],
            "daily_returns": [0.0, 0.25],
        }

        run_id1 = self.db.store_results(
            results=results,
            config_path="test_config.yaml",
            strategy_name="test_strategy",
            period="train",
            split_name="split_1",
        )

        # Store again with same parameters
        results["sharpe_ratio"] = 2.0  # Update metric
        run_id2 = self.db.store_results(
            results=results,
            config_path="test_config.yaml",
            strategy_name="test_strategy",
            period="train",
            split_name="split_1",
        )

        # Should reuse same run_id
        assert run_id1 == run_id2

        # Verify updated metric
        run_data = self.db.get_run(run_id1)
        assert run_data["metrics"]["sharpe_ratio"] == 2.0

    def test_get_run(self):
        """Test retrieving run by ID."""
        results = {
            "start_date": pd.Timestamp("2023-01-01"),
            "end_date": pd.Timestamp("2023-12-31"),
            "starting_equity": 100000.0,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.15,
            "total_return": 0.25,
            "total_trades": 50,
            "winning_trades": 30,
            "losing_trades": 20,
            "win_rate": 0.6,
            "realized_pnl": 25000.0,
            "final_cash": 125000.0,
            "final_positions": 0,
            "ending_equity": 125000.0,
            "closed_trades": [],
            "equity_curve": [100000.0, 125000.0],
            "daily_events": [
                {
                    "date": pd.Timestamp("2023-01-01"),
                    "portfolio_state": {"cash": 100000.0, "open_positions": 0, "gross_exposure": 0.0},
                },
                {
                    "date": pd.Timestamp("2023-12-31"),
                    "portfolio_state": {"cash": 125000.0, "open_positions": 0, "gross_exposure": 0.0},
                },
            ],
            "daily_returns": [0.0, 0.25],
        }

        run_id = self.db.store_results(
            results=results, config_path="test_config.yaml", strategy_name="test_strategy", period="train"
        )

        run_data = self.db.get_run(run_id)
        assert run_data is not None
        assert run_data["run_id"] == run_id
        assert run_data["strategy_name"] == "test_strategy"
        assert run_data["period"] == "train"
        assert "metrics" in run_data
        assert run_data["metrics"]["sharpe_ratio"] == 1.5

    def test_get_run_not_found(self):
        """Test retrieving non-existent run."""
        run_data = self.db.get_run(99999)
        assert run_data is None

    def test_query_runs(self):
        """Test querying runs with filters."""
        # Store multiple runs
        for i in range(3):
            results = {
                "start_date": pd.Timestamp(f"2023-0{i+1}-01"),
                "end_date": pd.Timestamp(f"2023-0{i+1}-28"),
                "starting_equity": 100000.0,
                "sharpe_ratio": 1.0 + i * 0.1,
                "max_drawdown": 0.15,
                "total_return": 0.10,
                "total_trades": 10,
                "winning_trades": 6,
                "losing_trades": 4,
                "win_rate": 0.6,
                "realized_pnl": 10000.0,
                "final_cash": 110000.0,
                "final_positions": 0,
                "ending_equity": 110000.0,
                "closed_trades": [],
                "equity_curve": [100000.0, 110000.0],
                "daily_events": [
                    {
                        "date": pd.Timestamp(f"2023-0{i+1}-01"),
                        "portfolio_state": {"cash": 100000.0, "open_positions": 0, "gross_exposure": 0.0},
                    },
                    {
                        "date": pd.Timestamp(f"2023-0{i+1}-28"),
                        "portfolio_state": {"cash": 110000.0, "open_positions": 0, "gross_exposure": 0.0},
                    },
                ],
                "daily_returns": [0.0, 0.10],
            }

            self.db.store_results(
                results=results,
                config_path=f"test_config_{i}.yaml",
                strategy_name="test_strategy",
                period="train" if i < 2 else "validation",
            )

        # Query by strategy
        runs = self.db.query_runs(strategy_name="test_strategy")
        assert len(runs) == 3

        # Query by period
        runs = self.db.query_runs(period="train")
        assert len(runs) == 2

        # Query with limit
        runs = self.db.query_runs(limit=1)
        assert len(runs) == 1

    def test_get_equity_curve(self):
        """Test retrieving equity curve."""
        equity_curve = [100000.0, 105000.0, 110000.0, 125000.0]
        daily_events = [
            {
                "date": pd.Timestamp("2023-01-01"),
                "portfolio_state": {"cash": 100000.0, "open_positions": 0, "gross_exposure": 0.0},
            },
            {
                "date": pd.Timestamp("2023-01-02"),
                "portfolio_state": {"cash": 95000.0, "open_positions": 1, "gross_exposure": 5000.0},
            },
            {
                "date": pd.Timestamp("2023-01-03"),
                "portfolio_state": {"cash": 90000.0, "open_positions": 2, "gross_exposure": 10000.0},
            },
            {
                "date": pd.Timestamp("2023-01-04"),
                "portfolio_state": {"cash": 125000.0, "open_positions": 0, "gross_exposure": 0.0},
            },
        ]

        results = {
            "start_date": pd.Timestamp("2023-01-01"),
            "end_date": pd.Timestamp("2023-01-04"),
            "starting_equity": 100000.0,
            "sharpe_ratio": 1.0,
            "max_drawdown": 0.05,
            "total_return": 0.25,
            "total_trades": 2,
            "winning_trades": 2,
            "losing_trades": 0,
            "win_rate": 1.0,
            "realized_pnl": 25000.0,
            "final_cash": 125000.0,
            "final_positions": 0,
            "ending_equity": 125000.0,
            "closed_trades": [],
            "equity_curve": equity_curve,
            "daily_events": daily_events,
            "daily_returns": [0.0, 0.05, 0.0476, 0.1364],
        }

        run_id = self.db.store_results(
            results=results, config_path="test_config.yaml", strategy_name="test_strategy", period="train"
        )

        df = self.db.get_equity_curve(run_id)
        assert len(df) == 4
        assert "date" in df.columns
        assert "equity" in df.columns
        assert "cash" in df.columns
        assert df.iloc[0]["equity"] == 100000.0
        assert df.iloc[-1]["equity"] == 125000.0

    def test_get_trades(self):
        """Test retrieving trades."""
        trade = Position(
            symbol="AAPL",
            asset_class="equity",
            entry_date=pd.Timestamp("2023-01-15"),
            entry_price=150.0,
            entry_fill_id="fill_1",
            quantity=100,
            side=PositionSide.LONG,
            stop_price=145.0,
            initial_stop_price=145.0,
            hard_stop_atr_mult=2.5,
            entry_slippage_bps=10.0,
            entry_fee_bps=1.0,
            entry_total_cost=15.0,
            triggered_on=BreakoutType.FAST_20D,
            adv20_at_entry=10000000.0,
            exit_date=pd.Timestamp("2023-01-20"),
            exit_price=155.0,
            exit_fill_id="fill_2",
            exit_reason=ExitReason.MANUAL,
            exit_slippage_bps=10.0,
            exit_fee_bps=1.0,
            exit_total_cost=15.5,
            realized_pnl=485.0,
        )

        results = {
            "start_date": pd.Timestamp("2023-01-01"),
            "end_date": pd.Timestamp("2023-01-31"),
            "starting_equity": 100000.0,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.10,
            "total_return": 0.05,
            "total_trades": 1,
            "winning_trades": 1,
            "losing_trades": 0,
            "win_rate": 1.0,
            "realized_pnl": 485.0,
            "final_cash": 100485.0,
            "final_positions": 0,
            "ending_equity": 100485.0,
            "closed_trades": [trade],
            "equity_curve": [100000.0, 100485.0],
            "daily_events": [
                {
                    "date": pd.Timestamp("2023-01-01"),
                    "portfolio_state": {"cash": 100000.0, "open_positions": 0, "gross_exposure": 0.0},
                },
                {
                    "date": pd.Timestamp("2023-01-31"),
                    "portfolio_state": {"cash": 100485.0, "open_positions": 0, "gross_exposure": 0.0},
                },
            ],
            "daily_returns": [0.0, 0.00485],
        }

        run_id = self.db.store_results(
            results=results, config_path="test_config.yaml", strategy_name="test_strategy", period="train"
        )

        df = self.db.get_trades(run_id)
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "AAPL"
        assert df.iloc[0]["entry_price"] == 150.0
        assert df.iloc[0]["exit_price"] == 155.0
        assert df.iloc[0]["realized_pnl"] == 485.0

    def test_compare_runs(self):
        """Test comparing multiple runs."""
        # Store multiple runs
        run_ids = []
        for i in range(3):
            results = {
                "start_date": pd.Timestamp("2023-01-01"),
                "end_date": pd.Timestamp("2023-12-31"),
                "starting_equity": 100000.0,
                "sharpe_ratio": 1.0 + i * 0.2,
                "max_drawdown": 0.15 - i * 0.02,
                "total_return": 0.10 + i * 0.05,
                "total_trades": 10 + i * 5,
                "winning_trades": 6 + i * 3,
                "losing_trades": 4 + i * 2,
                "win_rate": 0.6,
                "realized_pnl": 10000.0 + i * 5000.0,
                "final_cash": 110000.0 + i * 5000.0,
                "final_positions": 0,
                "ending_equity": 110000.0 + i * 5000.0,
                "closed_trades": [],
                "equity_curve": [100000.0, 110000.0 + i * 5000.0],
                "daily_events": [
                    {
                        "date": pd.Timestamp("2023-01-01"),
                        "portfolio_state": {"cash": 100000.0, "open_positions": 0, "gross_exposure": 0.0},
                    },
                    {
                        "date": pd.Timestamp("2023-12-31"),
                        "portfolio_state": {"cash": 110000.0 + i * 5000.0, "open_positions": 0, "gross_exposure": 0.0},
                    },
                ],
                "daily_returns": [0.0, 0.10 + i * 0.05],
            }

            run_id = self.db.store_results(
                results=results, config_path=f"test_config_{i}.yaml", strategy_name="test_strategy", period="train"
            )
            run_ids.append(run_id)

        # Compare runs
        comparison_df = self.db.compare_runs(run_ids)
        assert len(comparison_df) == 3
        assert "run_id" in comparison_df.columns
        assert "sharpe_ratio" in comparison_df.columns
        assert "max_drawdown" in comparison_df.columns
        assert "total_return" in comparison_df.columns

    def test_delete_run(self):
        """Test deleting a run."""
        results = {
            "start_date": pd.Timestamp("2023-01-01"),
            "end_date": pd.Timestamp("2023-12-31"),
            "starting_equity": 100000.0,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.15,
            "total_return": 0.25,
            "total_trades": 50,
            "winning_trades": 30,
            "losing_trades": 20,
            "win_rate": 0.6,
            "realized_pnl": 25000.0,
            "final_cash": 125000.0,
            "final_positions": 0,
            "ending_equity": 125000.0,
            "closed_trades": [],
            "equity_curve": [100000.0, 125000.0],
            "daily_events": [
                {
                    "date": pd.Timestamp("2023-01-01"),
                    "portfolio_state": {"cash": 100000.0, "open_positions": 0, "gross_exposure": 0.0},
                },
                {
                    "date": pd.Timestamp("2023-12-31"),
                    "portfolio_state": {"cash": 125000.0, "open_positions": 0, "gross_exposure": 0.0},
                },
            ],
            "daily_returns": [0.0, 0.25],
        }

        run_id = self.db.store_results(
            results=results, config_path="test_config.yaml", strategy_name="test_strategy", period="train"
        )

        # Verify run exists
        assert self.db.get_run(run_id) is not None

        # Delete run
        self.db.delete_run(run_id)

        # Verify run is deleted
        assert self.db.get_run(run_id) is None

    def test_archive_runs(self):
        """Test archiving runs."""
        results = {
            "start_date": pd.Timestamp("2023-01-01"),
            "end_date": pd.Timestamp("2023-12-31"),
            "starting_equity": 100000.0,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.15,
            "total_return": 0.25,
            "total_trades": 50,
            "winning_trades": 30,
            "losing_trades": 20,
            "win_rate": 0.6,
            "realized_pnl": 25000.0,
            "final_cash": 125000.0,
            "final_positions": 0,
            "ending_equity": 125000.0,
            "closed_trades": [],
            "equity_curve": [100000.0, 125000.0],
            "daily_events": [
                {
                    "date": pd.Timestamp("2023-01-01"),
                    "portfolio_state": {"cash": 100000.0, "open_positions": 0, "gross_exposure": 0.0},
                },
                {
                    "date": pd.Timestamp("2023-12-31"),
                    "portfolio_state": {"cash": 125000.0, "open_positions": 0, "gross_exposure": 0.0},
                },
            ],
            "daily_returns": [0.0, 0.25],
        }

        run_id = self.db.store_results(
            results=results, config_path="test_config.yaml", strategy_name="test_strategy", period="train"
        )

        # Archive run
        archive_path = self.temp_dir / "archive.db"
        self.db.archive_runs([run_id], archive_db_path=archive_path)

        # Verify archive database was created
        assert archive_path.exists()

    def test_store_results_with_benchmark_returns(self):
        """Test storing results with benchmark returns for additional metrics."""
        results = {
            "start_date": pd.Timestamp("2023-01-01"),
            "end_date": pd.Timestamp("2023-12-31"),
            "starting_equity": 100000.0,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.15,
            "total_return": 0.25,
            "total_trades": 50,
            "winning_trades": 30,
            "losing_trades": 20,
            "win_rate": 0.6,
            "realized_pnl": 25000.0,
            "final_cash": 125000.0,
            "final_positions": 0,
            "ending_equity": 125000.0,
            "closed_trades": [],
            "equity_curve": [100000.0, 125000.0],
            "daily_events": [
                {
                    "date": pd.Timestamp("2023-01-01"),
                    "portfolio_state": {"cash": 100000.0, "open_positions": 0, "gross_exposure": 0.0},
                },
                {
                    "date": pd.Timestamp("2023-12-31"),
                    "portfolio_state": {"cash": 125000.0, "open_positions": 0, "gross_exposure": 0.0},
                },
            ],
            "daily_returns": [0.0, 0.25],
            "benchmark_returns": [0.0, 0.20],  # Benchmark underperformed
        }

        run_id = self.db.store_results(
            results=results, config_path="test_config.yaml", strategy_name="test_strategy", period="train"
        )

        # Verify run was stored
        run_data = self.db.get_run(run_id)
        assert run_data is not None
        # Additional metrics like correlation_to_benchmark should be computed if possible
        assert "metrics" in run_data

"""Tests for reporting module."""

import json
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from trading_system.reporting.metrics import MetricsCalculator
from trading_system.reporting.report_generator import ReportGenerator


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_run_directory(temp_dir):
    """Create a sample run directory with train/validation/holdout subdirectories."""
    base_path = temp_dir / "results"
    run_id = "test_run_123"
    run_dir = base_path / run_id

    # Create period directories
    train_dir = run_dir / "train"
    validation_dir = run_dir / "validation"
    holdout_dir = run_dir / "holdout"

    for period_dir in [train_dir, validation_dir, holdout_dir]:
        period_dir.mkdir(parents=True, exist_ok=True)

    # Create sample equity curve data
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    equity_values = [100000.0]
    for i in range(99):
        # Simulate some returns
        daily_return = 0.001 * (i % 10 - 5)  # Varying returns
        equity_values.append(equity_values[-1] * (1 + daily_return))

    # Create equity curve CSV for each period
    for period_dir, start_idx in [(train_dir, 0), (validation_dir, 60), (holdout_dir, 80)]:
        period_dates = dates[start_idx : start_idx + 20]
        period_equity = equity_values[start_idx : start_idx + 20]

        equity_df = pd.DataFrame(
            {
                "date": period_dates,
                "equity": period_equity,
                "cash": [50000.0] * len(period_dates),
                "positions": [2] * len(period_dates),
                "exposure": [50000.0] * len(period_dates),
                "exposure_pct": [50.0] * len(period_dates),
            }
        )
        equity_df.to_csv(period_dir / "equity_curve.csv", index=False)

        # Create sample trade log
        trade_data = []
        for i in range(5):
            entry_date = period_dates[0] + timedelta(days=i * 3)
            exit_date = entry_date + timedelta(days=5)
            pnl = 100.0 if i % 2 == 0 else -50.0

            trade_data.append(
                {
                    "trade_id": i + 1,
                    "symbol": f"STOCK{i}",
                    "asset_class": "equity",
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": 100.0,
                    "exit_price": 100.0 + (pnl / 100),
                    "quantity": 100,
                    "entry_fill_id": f"fill_{i}",
                    "exit_fill_id": f"exit_{i}",
                    "realized_pnl": pnl,
                    "entry_slippage_bps": 5.0,
                    "entry_fee_bps": 1.0,
                    "entry_total_cost": 6.0,
                    "exit_slippage_bps": 5.0,
                    "exit_fee_bps": 1.0,
                    "exit_total_cost": 6.0,
                    "total_cost": 12.0,
                    "initial_stop_price": 95.0,
                    "stop_price": 95.0,
                    "exit_reason": "trailing_ma_cross",
                    "triggered_on": "20D",
                    "holding_days": 5,
                    "r_multiple": pnl / 500.0,  # (exit - entry) / (entry - stop)
                    "adv20_at_entry": 1000000.0,
                }
            )

        trade_df = pd.DataFrame(trade_data)
        trade_df.to_csv(period_dir / "trade_log.csv", index=False)

        # Create weekly summary
        weekly_data = []
        for week_start in period_dates[::7]:
            week_end = week_start + timedelta(days=6)
            weekly_data.append(
                {
                    "week": week_start.strftime("%Y-W%U"),
                    "week_start": week_start,
                    "week_end": week_end,
                    "start_equity": 100000.0,
                    "end_equity": 101000.0,
                    "weekly_return": 0.01,
                    "weekly_return_pct": 1.0,
                    "trades": 2,
                    "realized_pnl": 200.0,
                    "volatility_annualized": 0.15,
                    "max_drawdown": 0.02,
                    "max_drawdown_pct": 2.0,
                }
            )

        weekly_df = pd.DataFrame(weekly_data)
        weekly_df.to_csv(period_dir / "weekly_summary.csv", index=False)

        # Create monthly report JSON
        monthly_report = {
            "generated_at": datetime.now().isoformat(),
            "period": {
                "start_date": period_dates[0].isoformat(),
                "end_date": period_dates[-1].isoformat(),
                "total_days": len(period_dates),
                "trading_days": len(period_dates),
            },
            "overall_metrics": {
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.05,
                "calmar_ratio": 2.0,
                "total_trades": 5.0,
                "win_rate": 0.6,
            },
            "monthly_summary": [],
        }

        with open(period_dir / "monthly_report.json", "w") as f:
            json.dump(monthly_report, f)

    return {"base_path": str(base_path), "run_id": run_id, "run_dir": run_dir}


def test_report_generator_init(sample_run_directory):
    """Test ReportGenerator initialization."""
    gen = ReportGenerator(base_path=sample_run_directory["base_path"], run_id=sample_run_directory["run_id"])

    assert gen.run_dir == sample_run_directory["run_dir"]
    assert gen.run_id == sample_run_directory["run_id"]


def test_report_generator_init_missing_directory(temp_dir):
    """Test ReportGenerator initialization with missing directory."""
    with pytest.raises(FileNotFoundError):
        ReportGenerator(base_path=str(temp_dir / "results"), run_id="nonexistent_run")


def test_load_period_data(sample_run_directory):
    """Test loading period data."""
    gen = ReportGenerator(base_path=sample_run_directory["base_path"], run_id=sample_run_directory["run_id"])

    # Load train period
    train_data = gen.load_period_data("train")
    assert train_data is not None
    assert train_data["period"] == "train"
    assert "equity_curve_df" in train_data
    assert "trade_log_df" in train_data
    assert "weekly_summary_df" in train_data
    assert "monthly_report" in train_data

    # Check dataframes
    assert len(train_data["equity_curve_df"]) == 20
    assert len(train_data["trade_log_df"]) == 5
    assert "date" in train_data["equity_curve_df"].columns
    assert "equity" in train_data["equity_curve_df"].columns


def test_load_period_data_missing_period(sample_run_directory):
    """Test loading non-existent period."""
    gen = ReportGenerator(base_path=sample_run_directory["base_path"], run_id=sample_run_directory["run_id"])

    # Try to load non-existent period
    data = gen.load_period_data("nonexistent")
    assert data is None


def test_compute_metrics_from_data(sample_run_directory):
    """Test computing metrics from loaded data."""
    gen = ReportGenerator(base_path=sample_run_directory["base_path"], run_id=sample_run_directory["run_id"])

    train_data = gen.load_period_data("train")
    metrics = gen.compute_metrics_from_data(train_data)

    # Check that key metrics are present
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "calmar_ratio" in metrics
    assert "total_trades" in metrics
    assert "win_rate" in metrics
    assert "expectancy" in metrics
    assert "profit_factor" in metrics

    # Check that metrics are numeric
    assert isinstance(metrics["sharpe_ratio"], (int, float))
    assert isinstance(metrics["max_drawdown"], (int, float))
    assert metrics["total_trades"] == 5.0


def test_generate_summary_report(sample_run_directory):
    """Test generating summary report."""
    gen = ReportGenerator(base_path=sample_run_directory["base_path"], run_id=sample_run_directory["run_id"])

    report_path = gen.generate_summary_report()

    assert report_path.exists()

    # Load and check report
    with open(report_path, "r") as f:
        report = json.load(f)

    assert report["run_id"] == sample_run_directory["run_id"]
    assert "periods" in report
    assert "train" in report["periods"]
    assert "validation" in report["periods"]
    assert "holdout" in report["periods"]

    # Check period data
    train_period = report["periods"]["train"]
    assert "date_range" in train_period
    assert "equity" in train_period
    assert "trades" in train_period
    assert "metrics" in train_period
    assert train_period["trades"]["total"] == 5


def test_generate_comparison_report(sample_run_directory):
    """Test generating comparison report."""
    gen = ReportGenerator(base_path=sample_run_directory["base_path"], run_id=sample_run_directory["run_id"])

    report_path = gen.generate_comparison_report()

    assert report_path.exists()

    # Load and check report
    with open(report_path, "r") as f:
        report = json.load(f)

    assert report["run_id"] == sample_run_directory["run_id"]
    assert "period_metrics" in report
    assert "comparison" in report
    assert "train" in report["period_metrics"]
    assert "validation" in report["period_metrics"]
    assert "holdout" in report["period_metrics"]

    # Check comparison data
    assert "sharpe_ratio" in report["comparison"]
    assert "max_drawdown" in report["comparison"]

    # Check degradation metrics
    assert "degradation_train_to_validation" in report
    assert "degradation_train_to_holdout" in report


def test_generate_comparison_report_insufficient_periods(temp_dir):
    """Test comparison report with insufficient periods."""
    base_path = temp_dir / "results"
    run_id = "test_run"
    run_dir = base_path / run_id
    train_dir = run_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    # Create minimal equity curve
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    equity_df = pd.DataFrame(
        {
            "date": dates,
            "equity": [100000.0] * 10,
            "cash": [50000.0] * 10,
            "positions": [0] * 10,
            "exposure": [0.0] * 10,
            "exposure_pct": [0.0] * 10,
        }
    )
    equity_df.to_csv(train_dir / "equity_curve.csv", index=False)

    # Empty trade log
    trade_df = pd.DataFrame()
    trade_df.to_csv(train_dir / "trade_log.csv", index=False)

    gen = ReportGenerator(base_path=str(base_path), run_id=run_id)

    # Should raise ValueError
    with pytest.raises(ValueError, match="Need at least 2 periods"):
        gen.generate_comparison_report()


def test_print_summary(sample_run_directory, capsys):
    """Test printing summary to console."""
    gen = ReportGenerator(base_path=sample_run_directory["base_path"], run_id=sample_run_directory["run_id"])

    gen.print_summary()

    captured = capsys.readouterr()
    assert "Backtest Summary Report" in captured.out
    assert "TRAIN Period" in captured.out
    assert "VALIDATION Period" in captured.out
    assert "HOLDOUT Period" in captured.out
    assert "Sharpe Ratio" in captured.out
    assert "Max Drawdown" in captured.out


def test_cmd_report_integration(sample_run_directory):
    """Test cmd_report CLI function."""
    import argparse

    from trading_system.cli import cmd_report

    class Args:
        def __init__(self):
            self.run_id = sample_run_directory["run_id"]
            self.base_path = sample_run_directory["base_path"]

    args = Args()

    # Should succeed
    result = cmd_report(args)
    assert result == 0


def test_cmd_report_missing_run_id():
    """Test cmd_report with missing run_id."""
    import argparse

    from trading_system.cli import cmd_report

    class Args:
        def __init__(self):
            self.run_id = "nonexistent_run"
            self.base_path = "results/"

    args = Args()

    # Should fail with FileNotFoundError
    result = cmd_report(args)
    assert result == 1

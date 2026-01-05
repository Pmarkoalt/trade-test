"""Unit tests for reporting CSV and JSON writers."""

import json
import shutil
import tempfile
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from trading_system.models.positions import ExitReason, Position, PositionSide
from trading_system.models.signals import BreakoutType
from trading_system.reporting.csv_writer import CSVWriter
from trading_system.reporting.json_writer import JSONWriter


class TestCSVWriter:
    """Tests for CSVWriter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.writer = CSVWriter(output_dir=str(self.temp_dir))

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, "temp_dir") and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_csv_writer_initialization(self):
        """Test CSVWriter initialization."""
        assert self.writer.output_dir.exists()
        assert self.writer.output_dir == self.temp_dir

    def test_write_equity_curve(self):
        """Test writing equity curve CSV."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        equity_curve = [100000.0 + i * 100 for i in range(10)]
        cash_history = [50000.0] * 10
        positions_count_history = [2] * 10
        exposure_history = [50000.0] * 10

        output_path = self.writer.write_equity_curve(
            equity_curve, dates, cash_history, positions_count_history, exposure_history
        )

        assert Path(output_path).exists()

        # Read and verify
        df = pd.read_csv(output_path)
        assert len(df) == 10
        assert "date" in df.columns
        assert "equity" in df.columns
        assert "cash" in df.columns
        assert "positions" in df.columns
        assert "exposure" in df.columns
        assert "exposure_pct" in df.columns

    def test_write_equity_curve_length_mismatch(self):
        """Test writing equity curve with length mismatch."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        equity_curve = [100000.0] * 5  # Wrong length

        with pytest.raises(ValueError, match="must match"):
            self.writer.write_equity_curve(equity_curve, dates, [], [], [])

    def test_write_trade_log_empty(self):
        """Test writing empty trade log."""
        output_path = self.writer.write_trade_log([])

        assert Path(output_path).exists()

        # Read and verify
        df = pd.read_csv(output_path)
        assert len(df) == 0
        assert "trade_id" in df.columns
        assert "symbol" in df.columns

    def test_write_trade_log_with_trades(self):
        """Test writing trade log with trades."""
        # Create sample positions
        positions = []
        for i in range(3):
            pos = Position(
                symbol=f"STOCK{i}",
                asset_class="equity",
                side=PositionSide.LONG,
                quantity=100,
                entry_price=100.0 + i * 10,
                entry_date=pd.Timestamp("2023-01-01") + timedelta(days=i),
                entry_fill_id=f"fill_{i}",
                initial_stop_price=95.0,
                stop_price=95.0,
                hard_stop_atr_mult=2.5,
                entry_slippage_bps=5.0,
                entry_fee_bps=1.0,
                entry_total_cost=6.0,
                triggered_on=BreakoutType.FAST_20D,
                adv20_at_entry=1000000.0,
                exit_price=105.0 + i * 10,
                exit_date=pd.Timestamp("2023-01-01") + timedelta(days=i + 5),
                exit_fill_id=f"exit_{i}",
                exit_reason=ExitReason.TRAILING_MA_CROSS,
                realized_pnl=500.0 + i * 100,
                exit_slippage_bps=5.0,
                exit_fee_bps=1.0,
                exit_total_cost=6.0,
            )
            positions.append(pos)

        output_path = self.writer.write_trade_log(positions)

        assert Path(output_path).exists()

        # Read and verify
        df = pd.read_csv(output_path)
        assert len(df) == 3
        assert df["trade_id"].iloc[0] == 1
        assert df["symbol"].iloc[0] == "STOCK0"
        assert "realized_pnl" in df.columns
        assert "holding_days" in df.columns
        assert "r_multiple" in df.columns

    def test_write_weekly_summary(self):
        """Test writing weekly summary CSV."""
        dates = pd.date_range("2023-01-01", periods=14, freq="D")  # 2 weeks
        equity_curve = [100000.0 + i * 50 for i in range(14)]
        daily_returns = [0.001] * 13  # One less than dates

        # Create sample positions
        positions = []
        for i in range(2):
            pos = Position(
                symbol=f"STOCK{i}",
                asset_class="equity",
                side=PositionSide.LONG,
                quantity=100,
                entry_price=100.0,
                entry_date=pd.Timestamp("2023-01-01") + timedelta(days=i * 3),
                entry_fill_id=f"fill_{i}",
                initial_stop_price=95.0,
                stop_price=95.0,
                hard_stop_atr_mult=2.5,
                entry_slippage_bps=5.0,
                entry_fee_bps=1.0,
                entry_total_cost=6.0,
                triggered_on=BreakoutType.FAST_20D,
                adv20_at_entry=1000000.0,
                exit_price=105.0,
                exit_date=pd.Timestamp("2023-01-01") + timedelta(days=i * 3 + 5),
                exit_fill_id=f"exit_{i}",
                exit_reason=ExitReason.TRAILING_MA_CROSS,
                realized_pnl=500.0,
                exit_slippage_bps=5.0,
                exit_fee_bps=1.0,
                exit_total_cost=6.0,
            )
            positions.append(pos)

        output_path = self.writer.write_weekly_summary(equity_curve, dates, daily_returns, positions)

        assert Path(output_path).exists()

        # Read and verify
        df = pd.read_csv(output_path)
        assert len(df) >= 1  # At least 1 week
        assert "week" in df.columns
        assert "week_start" in df.columns
        assert "week_end" in df.columns
        assert "weekly_return" in df.columns
        assert "trades" in df.columns

    def test_write_weekly_summary_length_mismatch(self):
        """Test writing weekly summary with length mismatch."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        equity_curve = [100000.0] * 5  # Wrong length

        with pytest.raises(ValueError, match="must match"):
            self.writer.write_weekly_summary(equity_curve, dates, [], [])


class TestJSONWriter:
    """Tests for JSONWriter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.writer = JSONWriter(output_dir=str(self.temp_dir))

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, "temp_dir") and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_json_writer_initialization(self):
        """Test JSONWriter initialization."""
        assert self.writer.output_dir.exists()
        assert self.writer.output_dir == self.temp_dir

    def test_write_monthly_report(self):
        """Test writing monthly report JSON."""
        dates = pd.date_range("2023-01-01", periods=60, freq="D")  # ~2 months
        equity_curve = [100000.0 + i * 50 for i in range(60)]
        daily_returns = [0.001] * 59  # One less than dates

        # Create sample positions
        positions = []
        for i in range(5):
            pos = Position(
                symbol=f"STOCK{i}",
                asset_class="equity",
                side=PositionSide.LONG,
                quantity=100,
                entry_price=100.0,
                entry_date=pd.Timestamp("2023-01-01") + timedelta(days=i * 10),
                entry_fill_id=f"fill_{i}",
                initial_stop_price=95.0,
                stop_price=95.0,
                hard_stop_atr_mult=2.5,
                entry_slippage_bps=5.0,
                entry_fee_bps=1.0,
                entry_total_cost=6.0,
                triggered_on=BreakoutType.FAST_20D,
                adv20_at_entry=1000000.0,
                exit_price=105.0 if i % 2 == 0 else 95.0,  # Mix of wins and losses
                exit_date=pd.Timestamp("2023-01-01") + timedelta(days=i * 10 + 5),
                exit_fill_id=f"exit_{i}",
                exit_reason=ExitReason.TRAILING_MA_CROSS,
                realized_pnl=500.0 if i % 2 == 0 else -500.0,
                exit_slippage_bps=5.0,
                exit_fee_bps=1.0,
                exit_total_cost=6.0,
            )
            positions.append(pos)

        output_path = self.writer.write_monthly_report(equity_curve, dates, daily_returns, positions)

        assert Path(output_path).exists()

        # Read and verify
        with open(output_path, "r") as f:
            report = json.load(f)

        assert "generated_at" in report
        assert "period" in report
        assert "overall_metrics" in report
        assert "monthly_summary" in report

        # Check period data
        assert "start_date" in report["period"]
        assert "end_date" in report["period"]
        assert "total_days" in report["period"]

        # Check metrics
        assert "sharpe_ratio" in report["overall_metrics"]
        assert "max_drawdown" in report["overall_metrics"]
        assert "total_trades" in report["overall_metrics"]

        # Check monthly summary
        assert len(report["monthly_summary"]) >= 1

    def test_write_monthly_report_length_mismatch(self):
        """Test writing monthly report with length mismatch."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        equity_curve = [100000.0] * 5  # Wrong length

        with pytest.raises(ValueError, match="must match"):
            self.writer.write_monthly_report(equity_curve, dates, [], [])

    def test_write_scenario_comparison(self):
        """Test writing scenario comparison JSON."""
        scenarios = {
            "baseline": {
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.12,
                "calmar_ratio": 2.0,
                "total_trades": 100.0,
                "win_rate": 0.55,
            },
            "2x_slippage": {
                "sharpe_ratio": 0.8,
                "max_drawdown": 0.18,
                "calmar_ratio": 1.2,
                "total_trades": 100.0,
                "win_rate": 0.50,
            },
            "3x_slippage": {
                "sharpe_ratio": 0.5,
                "max_drawdown": 0.25,
                "calmar_ratio": 0.8,
                "total_trades": 100.0,
                "win_rate": 0.45,
            },
        }

        output_path = self.writer.write_scenario_comparison(scenarios)

        assert Path(output_path).exists()

        # Read and verify
        with open(output_path, "r") as f:
            comparison = json.load(f)

        assert "generated_at" in comparison
        assert "scenarios" in comparison
        assert "comparison" in comparison

        # Check scenarios
        assert "baseline" in comparison["scenarios"]
        assert "2x_slippage" in comparison["scenarios"]
        assert "3x_slippage" in comparison["scenarios"]

        # Check comparison
        assert "sharpe_ratio" in comparison["comparison"]
        assert "max_drawdown" in comparison["comparison"]

        # Check best/worst
        sharpe_comp = comparison["comparison"]["sharpe_ratio"]
        assert "best" in sharpe_comp
        assert "worst" in sharpe_comp
        assert sharpe_comp["best"]["scenario"] == "baseline"
        assert sharpe_comp["worst"]["scenario"] == "3x_slippage"

    def test_write_scenario_comparison_empty(self):
        """Test writing scenario comparison with empty scenarios."""
        output_path = self.writer.write_scenario_comparison({})

        assert Path(output_path).exists()

        # Read and verify
        with open(output_path, "r") as f:
            comparison = json.load(f)

        assert "scenarios" in comparison
        assert comparison["scenarios"] == {}
        assert "comparison" in comparison

"""Unit tests for reporting module."""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

from trading_system.reporting.metrics import MetricsCalculator
from trading_system.reporting.csv_writer import CSVWriter
from trading_system.reporting.json_writer import JSONWriter
from trading_system.models.positions import Position, ExitReason
from trading_system.models.signals import BreakoutType


class TestMetricsCalculator:
    """Tests for metrics calculation."""
    
    def test_sharpe_ratio_basic(self):
        """Test Sharpe ratio calculation."""
        # Create simple equity curve with positive returns
        equity_curve = [100000, 101000, 102000, 103000]
        daily_returns = [0.01, 0.0099, 0.0098]
        closed_trades = []
        
        calc = MetricsCalculator(equity_curve, daily_returns, closed_trades)
        sharpe = calc.sharpe_ratio()
        
        # Should be positive for positive returns
        assert sharpe > 0
        assert isinstance(sharpe, float)
    
    def test_sharpe_ratio_zero_std(self):
        """Test Sharpe ratio with zero standard deviation."""
        equity_curve = [100000, 100000, 100000]
        daily_returns = [0.0, 0.0]
        closed_trades = []
        
        calc = MetricsCalculator(equity_curve, daily_returns, closed_trades)
        sharpe = calc.sharpe_ratio()
        
        assert sharpe == 0.0
    
    def test_max_drawdown(self):
        """Test maximum drawdown calculation."""
        # Create equity curve with drawdown
        equity_curve = [100000, 110000, 105000, 95000, 100000]
        daily_returns = [0.1, -0.045, -0.095, 0.053]
        closed_trades = []
        
        calc = MetricsCalculator(equity_curve, daily_returns, closed_trades)
        max_dd = calc.max_drawdown()
        
        # Max drawdown should be from 110000 to 95000 = 13.64%
        assert max_dd > 0.13
        assert max_dd < 0.14
        assert isinstance(max_dd, float)
    
    def test_max_drawdown_no_drawdown(self):
        """Test max drawdown with no drawdowns."""
        equity_curve = [100000, 101000, 102000, 103000]
        daily_returns = [0.01, 0.0099, 0.0098]
        closed_trades = []
        
        calc = MetricsCalculator(equity_curve, daily_returns, closed_trades)
        max_dd = calc.max_drawdown()
        
        assert max_dd == 0.0
    
    def test_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        equity_curve = [100000, 110000, 105000, 100000]
        daily_returns = [0.1, -0.045, -0.048]
        closed_trades = []
        dates = [pd.Timestamp('2023-01-01') + pd.Timedelta(days=i) for i in range(4)]
        
        calc = MetricsCalculator(equity_curve, daily_returns, closed_trades, dates=dates)
        calmar = calc.calmar_ratio()
        
        assert calmar > 0
        assert isinstance(calmar, float)
    
    def test_total_trades(self):
        """Test total trades count."""
        equity_curve = [100000, 101000]
        daily_returns = [0.01]
        closed_trades = [
            self._create_closed_trade("AAPL", 100.0, 105.0, 100),
            self._create_closed_trade("MSFT", 200.0, 210.0, 50)
        ]
        
        calc = MetricsCalculator(equity_curve, daily_returns, closed_trades)
        total = calc.total_trades()
        
        assert total == 2
    
    def test_expectancy(self):
        """Test expectancy (R-multiple) calculation."""
        equity_curve = [100000, 101000]
        daily_returns = [0.01]
        
        # Create trades with different R-multiples
        # Trade 1: entry=100, stop=95, exit=105 -> R = (105-100)/(100-95) = 1.0
        # Trade 2: entry=100, stop=95, exit=110 -> R = (110-100)/(100-95) = 2.0
        closed_trades = [
            self._create_closed_trade("AAPL", 100.0, 105.0, 100, initial_stop=95.0),
            self._create_closed_trade("MSFT", 100.0, 110.0, 50, initial_stop=95.0)
        ]
        
        calc = MetricsCalculator(equity_curve, daily_returns, closed_trades)
        expectancy = calc.expectancy()
        
        # Average R-multiple should be (1.0 + 2.0) / 2 = 1.5
        assert abs(expectancy - 1.5) < 0.01
    
    def test_profit_factor(self):
        """Test profit factor calculation."""
        equity_curve = [100000, 101000]
        daily_returns = [0.01]
        
        # Create winning and losing trades
        closed_trades = [
            self._create_closed_trade("AAPL", 100.0, 110.0, 100, realized_pnl=1000.0),
            self._create_closed_trade("MSFT", 100.0, 90.0, 50, realized_pnl=-500.0),
            self._create_closed_trade("GOOGL", 200.0, 220.0, 25, realized_pnl=500.0)
        ]
        
        calc = MetricsCalculator(equity_curve, daily_returns, closed_trades)
        pf = calc.profit_factor()
        
        # Gross profit = 1000 + 500 = 1500
        # Gross loss = 500
        # Profit factor = 1500 / 500 = 3.0
        assert abs(pf - 3.0) < 0.01
    
    def test_profit_factor_no_losses(self):
        """Test profit factor with no losses."""
        equity_curve = [100000, 101000]
        daily_returns = [0.01]
        closed_trades = [
            self._create_closed_trade("AAPL", 100.0, 110.0, 100, realized_pnl=1000.0)
        ]
        
        calc = MetricsCalculator(equity_curve, daily_returns, closed_trades)
        pf = calc.profit_factor()
        
        # Should be infinity if no losses
        assert pf == float('inf')
    
    def test_correlation_to_benchmark(self):
        """Test correlation to benchmark."""
        equity_curve = [100000, 101000, 102000]
        daily_returns = [0.01, 0.0099]
        benchmark_returns = [0.008, 0.009]
        closed_trades = []
        
        calc = MetricsCalculator(
            equity_curve, daily_returns, closed_trades,
            benchmark_returns=benchmark_returns
        )
        corr = calc.correlation_to_benchmark()
        
        assert corr is not None
        assert -1.0 <= corr <= 1.0
    
    def test_correlation_to_benchmark_no_benchmark(self):
        """Test correlation when benchmark not provided."""
        equity_curve = [100000, 101000]
        daily_returns = [0.01]
        closed_trades = []
        
        calc = MetricsCalculator(equity_curve, daily_returns, closed_trades)
        corr = calc.correlation_to_benchmark()
        
        assert corr is None
    
    def test_percentile_daily_loss(self):
        """Test percentile daily loss calculation."""
        # Create returns with some losses
        equity_curve = [100000, 99000, 98000, 100000, 101000]
        daily_returns = [-0.01, -0.0101, 0.0204, 0.01]
        closed_trades = []
        
        calc = MetricsCalculator(equity_curve, daily_returns, closed_trades)
        p99_loss = calc.percentile_daily_loss(99.0)
        
        assert p99_loss >= 0
        assert isinstance(p99_loss, float)
    
    def test_recovery_factor(self):
        """Test recovery factor calculation."""
        equity_curve = [100000, 110000, 95000, 105000]
        daily_returns = [0.1, -0.136, 0.105]
        closed_trades = []
        
        calc = MetricsCalculator(equity_curve, daily_returns, closed_trades)
        recovery = calc.recovery_factor()
        
        assert recovery > 0
        assert isinstance(recovery, float)
    
    def test_drawdown_duration(self):
        """Test drawdown duration calculation."""
        # Create equity curve with drawdown period
        equity_curve = [100000, 110000, 105000, 95000, 90000, 95000, 100000]
        daily_returns = [0.1, -0.045, -0.095, -0.053, 0.056, 0.053]
        closed_trades = []
        
        calc = MetricsCalculator(equity_curve, daily_returns, closed_trades)
        duration = calc.drawdown_duration()
        
        assert duration >= 0
        assert isinstance(duration, int)
    
    def test_turnover(self):
        """Test turnover calculation."""
        equity_curve = [100000] * 63  # ~3 months of trading days
        daily_returns = [0.0] * 62
        closed_trades = [
            self._create_closed_trade(f"SYM{i}", 100.0, 105.0, 100)
            for i in range(10)
        ]
        
        calc = MetricsCalculator(equity_curve, daily_returns, closed_trades)
        turnover = calc.turnover()
        
        # Should be approximately 10 trades / 3 months = ~3.33 trades/month
        assert turnover > 0
        assert isinstance(turnover, float)
    
    def test_average_holding_period(self):
        """Test average holding period calculation."""
        equity_curve = [100000, 101000]
        daily_returns = [0.01]
        
        base_date = pd.Timestamp('2023-01-01')
        closed_trades = [
            self._create_closed_trade(
                "AAPL", 100.0, 105.0, 100,
                entry_date=base_date,
                exit_date=base_date + pd.Timedelta(days=5)
            ),
            self._create_closed_trade(
                "MSFT", 200.0, 210.0, 50,
                entry_date=base_date,
                exit_date=base_date + pd.Timedelta(days=10)
            )
        ]
        
        calc = MetricsCalculator(equity_curve, daily_returns, closed_trades)
        avg_holding = calc.average_holding_period()
        
        # Average should be (5 + 10) / 2 = 7.5 days
        assert abs(avg_holding - 7.5) < 0.1
    
    def test_max_consecutive_losses(self):
        """Test max consecutive losses calculation."""
        equity_curve = [100000, 101000]
        daily_returns = [0.01]
        
        closed_trades = [
            self._create_closed_trade("AAPL", 100.0, 90.0, 100, realized_pnl=-1000.0),
            self._create_closed_trade("MSFT", 200.0, 190.0, 50, realized_pnl=-500.0),
            self._create_closed_trade("GOOGL", 300.0, 310.0, 25, realized_pnl=250.0),
            self._create_closed_trade("TSLA", 400.0, 380.0, 20, realized_pnl=-400.0)
        ]
        
        calc = MetricsCalculator(equity_curve, daily_returns, closed_trades)
        max_consec = calc.max_consecutive_losses()
        
        # Should be 2 (first two trades are consecutive losses)
        assert max_consec == 2
    
    def test_win_rate(self):
        """Test win rate calculation."""
        equity_curve = [100000, 101000]
        daily_returns = [0.01]
        
        closed_trades = [
            self._create_closed_trade("AAPL", 100.0, 110.0, 100, realized_pnl=1000.0),
            self._create_closed_trade("MSFT", 200.0, 190.0, 50, realized_pnl=-500.0),
            self._create_closed_trade("GOOGL", 300.0, 310.0, 25, realized_pnl=250.0)
        ]
        
        calc = MetricsCalculator(equity_curve, daily_returns, closed_trades)
        win_rate = calc.win_rate()
        
        # 2 wins out of 3 trades = 0.667
        assert abs(win_rate - 2/3) < 0.01
    
    def test_compute_all_metrics(self):
        """Test computing all metrics at once."""
        equity_curve = [100000, 110000, 105000, 100000]
        daily_returns = [0.1, -0.045, -0.048]
        closed_trades = [
            self._create_closed_trade("AAPL", 100.0, 105.0, 100, initial_stop=95.0)
        ]
        
        calc = MetricsCalculator(equity_curve, daily_returns, closed_trades)
        all_metrics = calc.compute_all_metrics()
        
        assert isinstance(all_metrics, dict)
        assert "sharpe_ratio" in all_metrics
        assert "max_drawdown" in all_metrics
        assert "calmar_ratio" in all_metrics
        assert "total_trades" in all_metrics
        assert "expectancy" in all_metrics
        assert "profit_factor" in all_metrics
        assert "win_rate" in all_metrics
    
    def _create_closed_trade(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        quantity: int,
        initial_stop: Optional[float] = None,
        entry_date: Optional[pd.Timestamp] = None,
        exit_date: Optional[pd.Timestamp] = None,
        realized_pnl: Optional[float] = None
    ) -> Position:
        """Helper to create a closed trade."""
        if initial_stop is None:
            initial_stop = entry_price * 0.95  # 5% stop
        
        if entry_date is None:
            entry_date = pd.Timestamp('2023-01-01')
        
        if exit_date is None:
            exit_date = entry_date + pd.Timedelta(days=5)
        
        if realized_pnl is None:
            realized_pnl = (exit_price - entry_price) * quantity
        
        trade = Position(
            symbol=symbol,
            asset_class="equity",
            entry_date=entry_date,
            entry_price=entry_price,
            entry_fill_id=f"fill_{symbol}_entry",
            quantity=quantity,
            stop_price=initial_stop,
            initial_stop_price=initial_stop,
            hard_stop_atr_mult=2.5,
            entry_slippage_bps=5.0,
            entry_fee_bps=1.0,
            entry_total_cost=entry_price * quantity * 0.0006,  # 6 bps total
            triggered_on=BreakoutType.FAST_20D,
            adv20_at_entry=1000000.0,
            exit_date=exit_date,
            exit_price=exit_price,
            exit_fill_id=f"fill_{symbol}_exit",
            exit_reason=ExitReason.TRAILING_MA_CROSS,
            exit_slippage_bps=5.0,
            exit_fee_bps=1.0,
            exit_total_cost=exit_price * quantity * 0.0006,
            realized_pnl=realized_pnl
        )
        
        return trade


class TestCSVWriter:
    """Tests for CSV writer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.writer = CSVWriter(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_write_equity_curve(self):
        """Test writing equity curve CSV."""
        equity_curve = [100000, 101000, 102000]
        dates = [
            pd.Timestamp('2023-01-01'),
            pd.Timestamp('2023-01-02'),
            pd.Timestamp('2023-01-03')
        ]
        cash_history = [50000, 49000, 48000]
        positions_count = [2, 2, 2]
        exposure = [50000, 52000, 54000]
        
        output_path = self.writer.write_equity_curve(
            equity_curve, dates, cash_history, positions_count, exposure
        )
        
        assert Path(output_path).exists()
        
        # Read and verify
        df = pd.read_csv(output_path)
        assert len(df) == 3
        assert "date" in df.columns
        assert "equity" in df.columns
        assert "cash" in df.columns
        assert "positions" in df.columns
        assert "exposure" in df.columns
    
    def test_write_trade_log(self):
        """Test writing trade log CSV."""
        closed_trades = [
            self._create_closed_trade("AAPL", 100.0, 105.0, 100),
            self._create_closed_trade("MSFT", 200.0, 210.0, 50)
        ]
        
        output_path = self.writer.write_trade_log(closed_trades)
        
        assert Path(output_path).exists()
        
        # Read and verify
        df = pd.read_csv(output_path)
        assert len(df) == 2
        assert "symbol" in df.columns
        assert "entry_date" in df.columns
        assert "exit_date" in df.columns
        assert "realized_pnl" in df.columns
    
    def test_write_trade_log_empty(self):
        """Test writing empty trade log."""
        output_path = self.writer.write_trade_log([])
        
        assert Path(output_path).exists()
        
        df = pd.read_csv(output_path)
        assert len(df) == 0
    
    def test_write_weekly_summary(self):
        """Test writing weekly summary CSV."""
        # Create 3 weeks of data
        dates = [
            pd.Timestamp('2023-01-02') + pd.Timedelta(days=i)
            for i in range(15)  # 3 weeks
        ]
        equity_curve = [100000 + i * 100 for i in range(15)]
        daily_returns = [0.001] * 14
        
        closed_trades = [
            self._create_closed_trade(
                "AAPL", 100.0, 105.0, 100,
                exit_date=pd.Timestamp('2023-01-05')
            )
        ]
        
        output_path = self.writer.write_weekly_summary(
            equity_curve, dates, daily_returns, closed_trades
        )
        
        assert Path(output_path).exists()
        
        # Read and verify
        df = pd.read_csv(output_path)
        assert len(df) > 0
        assert "week" in df.columns
        assert "weekly_return" in df.columns
        assert "trades" in df.columns
    
    def _create_closed_trade(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        quantity: int
    ) -> Position:
        """Helper to create a closed trade."""
        entry_date = pd.Timestamp('2023-01-01')
        exit_date = entry_date + pd.Timedelta(days=5)
        initial_stop = entry_price * 0.95
        
        return Position(
            symbol=symbol,
            asset_class="equity",
            entry_date=entry_date,
            entry_price=entry_price,
            entry_fill_id=f"fill_{symbol}_entry",
            quantity=quantity,
            stop_price=initial_stop,
            initial_stop_price=initial_stop,
            hard_stop_atr_mult=2.5,
            entry_slippage_bps=5.0,
            entry_fee_bps=1.0,
            entry_total_cost=entry_price * quantity * 0.0006,
            triggered_on=BreakoutType.FAST_20D,
            adv20_at_entry=1000000.0,
            exit_date=exit_date,
            exit_price=exit_price,
            exit_fill_id=f"fill_{symbol}_exit",
            exit_reason=ExitReason.TRAILING_MA_CROSS,
            exit_slippage_bps=5.0,
            exit_fee_bps=1.0,
            exit_total_cost=exit_price * quantity * 0.0006,
            realized_pnl=(exit_price - entry_price) * quantity
        )


class TestJSONWriter:
    """Tests for JSON writer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.writer = JSONWriter(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_write_monthly_report(self):
        """Test writing monthly report JSON."""
        # Create 2 months of data
        dates = [
            pd.Timestamp('2023-01-01') + pd.Timedelta(days=i)
            for i in range(42)  # ~2 months
        ]
        equity_curve = [100000 + i * 50 for i in range(42)]
        daily_returns = [0.0005] * 41
        
        closed_trades = [
            self._create_closed_trade(
                "AAPL", 100.0, 105.0, 100,
                exit_date=pd.Timestamp('2023-01-15')
            )
        ]
        
        output_path = self.writer.write_monthly_report(
            equity_curve, dates, daily_returns, closed_trades
        )
        
        assert Path(output_path).exists()
        
        # Read and verify
        with open(output_path, 'r') as f:
            report = json.load(f)
        
        assert "generated_at" in report
        assert "period" in report
        assert "overall_metrics" in report
        assert "monthly_summary" in report
        assert len(report["monthly_summary"]) > 0
    
    def test_write_scenario_comparison(self):
        """Test writing scenario comparison JSON."""
        scenarios = {
            "baseline": {
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.12,
                "calmar_ratio": 1.5,
                "total_trades": 50
            },
            "2x_slippage": {
                "sharpe_ratio": 0.8,
                "max_drawdown": 0.18,
                "calmar_ratio": 1.0,
                "total_trades": 50
            }
        }
        
        output_path = self.writer.write_scenario_comparison(scenarios)
        
        assert Path(output_path).exists()
        
        # Read and verify
        with open(output_path, 'r') as f:
            comparison = json.load(f)
        
        assert "generated_at" in comparison
        assert "scenarios" in comparison
        assert "comparison" in comparison
        assert len(comparison["scenarios"]) == 2
    
    def _create_closed_trade(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        quantity: int,
        exit_date: Optional[pd.Timestamp] = None
    ) -> Position:
        """Helper to create a closed trade."""
        entry_date = pd.Timestamp('2023-01-01')
        if exit_date is None:
            exit_date = entry_date + pd.Timedelta(days=5)
        initial_stop = entry_price * 0.95
        
        return Position(
            symbol=symbol,
            asset_class="equity",
            entry_date=entry_date,
            entry_price=entry_price,
            entry_fill_id=f"fill_{symbol}_entry",
            quantity=quantity,
            stop_price=initial_stop,
            initial_stop_price=initial_stop,
            hard_stop_atr_mult=2.5,
            entry_slippage_bps=5.0,
            entry_fee_bps=1.0,
            entry_total_cost=entry_price * quantity * 0.0006,
            triggered_on=BreakoutType.FAST_20D,
            adv20_at_entry=1000000.0,
            exit_date=exit_date,
            exit_price=exit_price,
            exit_fill_id=f"fill_{symbol}_exit",
            exit_reason=ExitReason.TRAILING_MA_CROSS,
            exit_slippage_bps=5.0,
            exit_fee_bps=1.0,
            exit_total_cost=exit_price * quantity * 0.0006,
            realized_pnl=(exit_price - entry_price) * quantity
        )


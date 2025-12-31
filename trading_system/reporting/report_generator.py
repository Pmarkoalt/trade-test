"""Report generator for loading and analyzing backtest results."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None
    Table = None
    Panel = None
    box = None

from .metrics import MetricsCalculator
from ..models.positions import Position, ExitReason
from ..models.signals import BreakoutType

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate reports from completed backtest runs."""
    
    def __init__(self, base_path: str, run_id: str):
        """Initialize report generator.
        
        Args:
            base_path: Base path for results (e.g., "results/")
            run_id: Run ID to generate report for
        """
        self.base_path = Path(base_path)
        self.run_id = run_id
        self.run_dir = self.base_path / run_id
        
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")
    
    def load_period_data(self, period: str) -> Optional[Dict[str, Any]]:
        """Load data for a specific period (train/validation/holdout).
        
        Args:
            period: Period name ("train", "validation", "holdout")
        
        Returns:
            Dictionary with loaded data, or None if period doesn't exist
        """
        period_dir = self.run_dir / period
        
        if not period_dir.exists():
            logger.warning(f"Period directory not found: {period_dir}")
            return None
        
        data = {
            'period': period,
            'directory': period_dir
        }
        
        # Load equity curve
        equity_file = period_dir / "equity_curve.csv"
        if equity_file.exists():
            data['equity_curve_df'] = pd.read_csv(equity_file, parse_dates=['date'])
        else:
            logger.warning(f"Equity curve file not found: {equity_file}")
            return None
        
        # Load trade log
        trade_log_file = period_dir / "trade_log.csv"
        if trade_log_file.exists():
            try:
                # Check if file is empty (has no content or only headers)
                if trade_log_file.stat().st_size == 0:
                    data['trade_log_df'] = pd.DataFrame()
                else:
                    data['trade_log_df'] = pd.read_csv(trade_log_file, parse_dates=['entry_date', 'exit_date'])
            except (pd.errors.EmptyDataError, ValueError):
                # Handle empty CSV files gracefully
                data['trade_log_df'] = pd.DataFrame()
        else:
            data['trade_log_df'] = pd.DataFrame()
        
        # Load weekly summary
        weekly_file = period_dir / "weekly_summary.csv"
        if weekly_file.exists():
            try:
                # Check if file is empty
                if weekly_file.stat().st_size == 0:
                    data['weekly_summary_df'] = pd.DataFrame()
                else:
                    data['weekly_summary_df'] = pd.read_csv(weekly_file, parse_dates=['week_start', 'week_end'])
            except (pd.errors.EmptyDataError, ValueError):
                # Handle empty CSV files gracefully
                data['weekly_summary_df'] = pd.DataFrame()
        else:
            data['weekly_summary_df'] = pd.DataFrame()
        
        # Load monthly report JSON
        monthly_file = period_dir / "monthly_report.json"
        if monthly_file.exists():
            with open(monthly_file, 'r') as f:
                data['monthly_report'] = json.load(f)
        else:
            data['monthly_report'] = None
        
        # Load scenario comparison if exists (for validation)
        scenario_file = period_dir / "scenario_comparison.json"
        if scenario_file.exists():
            with open(scenario_file, 'r') as f:
                data['scenario_comparison'] = json.load(f)
        else:
            data['scenario_comparison'] = None
        
        return data
    
    def compute_metrics_from_data(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Compute metrics from loaded data.
        
        Args:
            data: Dictionary with loaded period data
        
        Returns:
            Dictionary of computed metrics
        """
        equity_df = data['equity_curve_df']
        trade_df = data['trade_log_df']
        
        # Extract equity curve and dates
        equity_curve = equity_df['equity'].tolist()
        dates = equity_df['date'].tolist()
        
        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i-1] > 0:
                daily_return = (equity_curve[i] / equity_curve[i-1]) - 1
            else:
                daily_return = 0.0
            daily_returns.append(daily_return)
        
        # Convert trade log to Position objects for metrics calculation
        closed_trades = []
        if not trade_df.empty:
            for _, row in trade_df.iterrows():
                # Create a Position object for metrics calculation
                # Provide defaults for required fields that may be missing
                entry_date = pd.Timestamp(row['entry_date']) if pd.notna(row.get('entry_date')) else pd.Timestamp.now()
                entry_price = float(row['entry_price']) if pd.notna(row.get('entry_price')) else 0.0
                quantity = int(row['quantity']) if pd.notna(row.get('quantity')) else 1
                initial_stop_price = float(row['initial_stop_price']) if pd.notna(row.get('initial_stop_price')) else entry_price * 0.95
                stop_price = float(row.get('stop_price', initial_stop_price)) if pd.notna(row.get('stop_price')) else initial_stop_price
                
                # Ensure stop_price is valid (below entry_price)
                if stop_price >= entry_price:
                    stop_price = entry_price * 0.95
                if initial_stop_price >= entry_price:
                    initial_stop_price = entry_price * 0.95
                
                trade = Position(
                    symbol=str(row.get('symbol', 'UNKNOWN')),
                    asset_class=str(row.get('asset_class', 'equity')),
                    entry_date=entry_date,
                    entry_price=entry_price,
                    entry_fill_id=str(row.get('entry_fill_id', '')),
                    quantity=quantity,
                    stop_price=stop_price,
                    initial_stop_price=initial_stop_price,
                    hard_stop_atr_mult=float(row.get('hard_stop_atr_mult', 2.5)),
                    entry_slippage_bps=float(row.get('entry_slippage_bps', 0.0)),
                    entry_fee_bps=float(row.get('entry_fee_bps', 0.0)),
                    entry_total_cost=float(row.get('entry_total_cost', 0.0)),
                    exit_date=pd.Timestamp(row['exit_date']) if pd.notna(row.get('exit_date')) else None,
                    exit_price=float(row['exit_price']) if pd.notna(row.get('exit_price')) else None,
                    exit_fill_id=str(row.get('exit_fill_id', '')) if pd.notna(row.get('exit_fill_id')) else None,
                    exit_reason=ExitReason(row['exit_reason']) if pd.notna(row.get('exit_reason')) and row.get('exit_reason') else None,
                    exit_slippage_bps=float(row['exit_slippage_bps']) if pd.notna(row.get('exit_slippage_bps')) else None,
                    exit_fee_bps=float(row['exit_fee_bps']) if pd.notna(row.get('exit_fee_bps')) else None,
                    exit_total_cost=float(row['exit_total_cost']) if pd.notna(row.get('exit_total_cost')) else None,
                    realized_pnl=float(row['realized_pnl']) if pd.notna(row.get('realized_pnl')) else 0.0,
                    triggered_on=self._parse_breakout_type(row.get('triggered_on')),
                    adv20_at_entry=float(row.get('adv20_at_entry', 1000000.0))
                )
                closed_trades.append(trade)
        
        # Create metrics calculator
        metrics_calc = MetricsCalculator(
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            closed_trades=closed_trades,
            dates=dates
        )
        
        return metrics_calc.compute_all_metrics()
    
    def generate_summary_report(self, output_path: Optional[Path] = None) -> Path:
        """Generate summary report for all available periods.
        
        Args:
            output_path: Optional path to save report. Defaults to run_dir/summary_report.json
        
        Returns:
            Path to generated report
        """
        periods = ['train', 'validation', 'holdout']
        period_data = {}
        period_metrics = {}
        
        # Load data for each period
        for period in periods:
            data = self.load_period_data(period)
            if data is not None:
                period_data[period] = data
                period_metrics[period] = self.compute_metrics_from_data(data)
        
        if len(period_data) == 0:
            raise ValueError(f"No period data found in {self.run_dir}")
        
        # Generate summary report
        report = {
            'generated_at': datetime.now().isoformat(),
            'run_id': self.run_id,
            'run_directory': str(self.run_dir),
            'periods': {}
        }
        
        for period, data in period_data.items():
            equity_df = data['equity_curve_df']
            trade_df = data['trade_log_df']
            metrics = period_metrics[period]
            
            # Calculate additional summary stats
            total_return = 0.0
            if len(equity_df) > 0:
                start_equity = equity_df['equity'].iloc[0]
                end_equity = equity_df['equity'].iloc[-1]
                if start_equity > 0:
                    total_return = (end_equity / start_equity) - 1
            
            # Calculate annualized return
            num_days = len(equity_df)
            years = num_days / 252.0 if num_days > 0 else 0
            annualized_return = ((1 + total_return) ** (1 / years) - 1) if years > 0 else 0.0
            
            period_summary = {
                'period': period,
                'date_range': {
                    'start': equity_df['date'].min().isoformat() if len(equity_df) > 0 else None,
                    'end': equity_df['date'].max().isoformat() if len(equity_df) > 0 else None,
                    'trading_days': len(equity_df)
                },
                'equity': {
                    'start': float(start_equity) if len(equity_df) > 0 else 0.0,
                    'end': float(end_equity) if len(equity_df) > 0 else 0.0,
                    'total_return': float(total_return),
                    'total_return_pct': float(total_return * 100),
                    'annualized_return': float(annualized_return),
                    'annualized_return_pct': float(annualized_return * 100)
                },
                'trades': {
                    'total': len(trade_df),
                    'winning': int(len(trade_df[trade_df['realized_pnl'] > 0])) if not trade_df.empty else 0,
                    'losing': int(len(trade_df[trade_df['realized_pnl'] < 0])) if not trade_df.empty else 0,
                    'total_pnl': float(trade_df['realized_pnl'].sum()) if not trade_df.empty else 0.0
                },
                'metrics': metrics
            }
            
            report['periods'][period] = period_summary
        
        # Save report
        if output_path is None:
            output_path = self.run_dir / "summary_report.json"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Summary report saved to {output_path}")
        return output_path
    
    def generate_comparison_report(self, output_path: Optional[Path] = None) -> Path:
        """Generate comparison report between train, validation, and holdout periods.
        
        Args:
            output_path: Optional path to save report. Defaults to run_dir/comparison_report.json
        
        Returns:
            Path to generated report
        """
        periods = ['train', 'validation', 'holdout']
        period_metrics = {}
        
        # Load metrics for each period
        for period in periods:
            data = self.load_period_data(period)
            if data is not None:
                period_metrics[period] = self.compute_metrics_from_data(data)
        
        if len(period_metrics) < 2:
            raise ValueError(f"Need at least 2 periods for comparison, found {len(period_metrics)}")
        
        # Key metrics to compare
        key_metrics = [
            'sharpe_ratio',
            'max_drawdown',
            'calmar_ratio',
            'total_trades',
            'expectancy',
            'profit_factor',
            'win_rate',
            'recovery_factor',
            'turnover',
            'average_holding_period',
            'max_consecutive_losses'
        ]
        
        comparison = {
            'generated_at': datetime.now().isoformat(),
            'run_id': self.run_id,
            'run_directory': str(self.run_dir),
            'period_metrics': period_metrics,
            'comparison': {}
        }
        
        # Compare each metric across periods
        for metric in key_metrics:
            values = {}
            for period, metrics in period_metrics.items():
                if metric in metrics and metrics[metric] is not None:
                    values[period] = metrics[metric]
            
            if len(values) == 0:
                continue
            
            # Determine best and worst
            # For drawdown, lower is better; for others, higher is better
            if metric == 'max_drawdown':
                best_period = min(values.items(), key=lambda x: x[1])
                worst_period = max(values.items(), key=lambda x: x[1])
            else:
                best_period = max(values.items(), key=lambda x: x[1])
                worst_period = min(values.items(), key=lambda x: x[1])
            
            comparison['comparison'][metric] = {
                'best': {
                    'period': best_period[0],
                    'value': float(best_period[1])
                },
                'worst': {
                    'period': worst_period[0],
                    'value': float(worst_period[1])
                },
                'range': float(best_period[1] - worst_period[1]),
                'values': {k: float(v) for k, v in values.items()}
            }
        
        # Calculate degradation metrics (validation vs train, holdout vs train)
        if 'train' in period_metrics and 'validation' in period_metrics:
            train_metrics = period_metrics['train']
            val_metrics = period_metrics['validation']
            
            degradation = {}
            for metric in key_metrics:
                if metric in train_metrics and metric in val_metrics:
                    train_val = train_metrics[metric]
                    val_val = val_metrics[metric]
                    
                    if train_val != 0 and not np.isnan(train_val) and not np.isnan(val_val):
                        if metric == 'max_drawdown':
                            # For drawdown, increase is bad
                            pct_change = ((val_val - train_val) / train_val) * 100
                        else:
                            # For other metrics, decrease is bad
                            pct_change = ((val_val - train_val) / train_val) * 100
                        
                        degradation[metric] = {
                            'train': float(train_val),
                            'validation': float(val_val),
                            'change_pct': float(pct_change)
                        }
            
            comparison['degradation_train_to_validation'] = degradation
        
        if 'train' in period_metrics and 'holdout' in period_metrics:
            train_metrics = period_metrics['train']
            holdout_metrics = period_metrics['holdout']
            
            degradation = {}
            for metric in key_metrics:
                if metric in train_metrics and metric in holdout_metrics:
                    train_val = train_metrics[metric]
                    holdout_val = holdout_metrics[metric]
                    
                    if train_val != 0 and not np.isnan(train_val) and not np.isnan(holdout_val):
                        if metric == 'max_drawdown':
                            pct_change = ((holdout_val - train_val) / train_val) * 100
                        else:
                            pct_change = ((holdout_val - train_val) / train_val) * 100
                        
                        degradation[metric] = {
                            'train': float(train_val),
                            'holdout': float(holdout_val),
                            'change_pct': float(pct_change)
                        }
            
            comparison['degradation_train_to_holdout'] = degradation
        
        # Save report
        if output_path is None:
            output_path = self.run_dir / "comparison_report.json"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        logger.info(f"Comparison report saved to {output_path}")
        return output_path
    
    def _parse_breakout_type(self, value: Any) -> BreakoutType:
        """Parse breakout type from CSV value.
        
        Args:
            value: Value from CSV (could be string, enum, or None)
        
        Returns:
            BreakoutType enum value
        """
        if pd.isna(value) or value is None or value == '':
            return BreakoutType.FAST_20D
        
        # Handle string values
        if isinstance(value, str):
            value_upper = value.upper()
            if value_upper in ['20D', 'FAST_20D']:
                return BreakoutType.FAST_20D
            elif value_upper in ['55D', 'SLOW_55D']:
                return BreakoutType.SLOW_55D
            else:
                return BreakoutType.FAST_20D
        
        # Handle enum values
        if isinstance(value, BreakoutType):
            return value
        
        # Default
        return BreakoutType.FAST_20D
    
    def print_summary(self) -> None:
        """Print a human-readable summary to console."""
        periods = ['train', 'validation', 'holdout']
        period_metrics = {}
        
        # Load metrics for each period
        for period in periods:
            data = self.load_period_data(period)
            if data is not None:
                period_metrics[period] = self.compute_metrics_from_data(data)
        
        if len(period_metrics) == 0:
            if RICH_AVAILABLE and Console:
                console = Console()
                console.print(f"[yellow]No period data found in {self.run_dir}[/yellow]")
            else:
                print(f"No period data found in {self.run_dir}")
            return
        
        # Use rich if available
        if RICH_AVAILABLE and Console and Table and Panel:
            console = Console()
            
            # Title panel
            console.print(Panel(
                f"[bold cyan]Backtest Summary Report[/bold cyan]\n[dim]Run ID: {self.run_id}[/dim]",
                title="Report",
                border_style="cyan",
                box=box.ROUNDED
            ))
            
            # Create metrics table for each period
            for period, metrics in period_metrics.items():
                table = Table(
                    title=f"{period.upper()} Period",
                    box=box.ROUNDED,
                    show_header=True,
                    header_style="bold magenta"
                )
                table.add_column("Metric", style="cyan", no_wrap=True)
                table.add_column("Value", style="green", justify="right")
                
                table.add_row("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.3f}")
                table.add_row("Max Drawdown", f"{metrics.get('max_drawdown', 0)*100:.2f}%")
                table.add_row("Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.3f}")
                table.add_row("Total Trades", f"{int(metrics.get('total_trades', 0))}")
                table.add_row("Win Rate", f"{metrics.get('win_rate', 0)*100:.2f}%")
                table.add_row("Expectancy (R)", f"{metrics.get('expectancy', 0):.3f}")
                table.add_row("Profit Factor", f"{metrics.get('profit_factor', 0):.3f}")
                table.add_row("Recovery Factor", f"{metrics.get('recovery_factor', 0):.3f}")
                
                console.print(table)
            
            # Comparison table if multiple periods
            if len(period_metrics) >= 2:
                comparison_table = Table(
                    title="Period Comparison",
                    box=box.ROUNDED,
                    show_header=True,
                    header_style="bold magenta"
                )
                comparison_table.add_column("Metric", style="cyan", no_wrap=True)
                for period in period_metrics.keys():
                    comparison_table.add_column(period.upper(), style="green", justify="right")
                
                key_metrics = ['sharpe_ratio', 'max_drawdown', 'calmar_ratio', 'win_rate']
                for metric in key_metrics:
                    row = [metric.replace('_', ' ').title()]
                    for period in period_metrics.keys():
                        value = period_metrics[period].get(metric, 0)
                        if metric == 'max_drawdown':
                            row.append(f"{value*100:.2f}%")
                        elif metric == 'win_rate':
                            row.append(f"{value*100:.2f}%")
                        else:
                            row.append(f"{value:.3f}")
                    comparison_table.add_row(*row)
                
                console.print("\n")
                console.print(comparison_table)
        else:
            # Fallback to plain text
            print(f"\n{'='*80}")
            print(f"Backtest Summary Report: {self.run_id}")
            print(f"{'='*80}\n")
            
            for period, metrics in period_metrics.items():
                print(f"\n{period.upper()} Period:")
                print(f"  Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):.3f}")
                print(f"  Max Drawdown:         {metrics.get('max_drawdown', 0)*100:.2f}%")
                print(f"  Calmar Ratio:         {metrics.get('calmar_ratio', 0):.3f}")
                print(f"  Total Trades:         {int(metrics.get('total_trades', 0))}")
                print(f"  Win Rate:            {metrics.get('win_rate', 0)*100:.2f}%")
                print(f"  Expectancy (R):      {metrics.get('expectancy', 0):.3f}")
                print(f"  Profit Factor:       {metrics.get('profit_factor', 0):.3f}")
                print(f"  Recovery Factor:     {metrics.get('recovery_factor', 0):.3f}")
            
            if len(period_metrics) >= 2:
                print(f"\n{'='*80}")
                print("Comparison:")
                print(f"{'='*80}\n")
                
                # Compare key metrics
                key_metrics = ['sharpe_ratio', 'max_drawdown', 'calmar_ratio', 'win_rate']
                for metric in key_metrics:
                    values = {p: m.get(metric, 0) for p, m in period_metrics.items() if metric in m}
                    if len(values) > 0:
                        if metric == 'max_drawdown':
                            best = min(values.items(), key=lambda x: x[1])
                            print(f"  {metric.replace('_', ' ').title()}:")
                            print(f"    Best: {best[0]} ({best[1]*100:.2f}%)")
                        else:
                            best = max(values.items(), key=lambda x: x[1])
                            print(f"  {metric.replace('_', ' ').title()}:")
                            print(f"    Best: {best[0]} ({best[1]:.3f})")
            
            print(f"\n{'='*80}\n")


"""JSON output generation for backtest results."""

from typing import List, Dict, Optional
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from ..models.positions import Position
from .metrics import MetricsCalculator


class JSONWriter:
    """Write backtest results to JSON files."""
    
    def __init__(self, output_dir: str):
        """Initialize JSON writer.
        
        Args:
            output_dir: Directory to write JSON files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def write_monthly_report(
        self,
        equity_curve: List[float],
        dates: List[pd.Timestamp],
        daily_returns: List[float],
        closed_trades: List[Position],
        benchmark_returns: Optional[List[float]] = None
    ) -> str:
        """Write monthly report JSON with aggregated metrics.
        
        Args:
            equity_curve: List of daily equity values
            dates: List of dates corresponding to equity_curve
            daily_returns: List of daily portfolio returns
            closed_trades: List of closed Position objects
            benchmark_returns: Optional benchmark returns for correlation
        
        Returns:
            Path to written file
        """
        if len(equity_curve) != len(dates):
            raise ValueError(
                f"equity_curve length ({len(equity_curve)}) must match "
                f"dates length ({len(dates)})"
            )
        
        # Create DataFrame with daily data
        df = pd.DataFrame({
            "date": dates,
            "equity": equity_curve,
            "daily_return": [0.0] + daily_returns  # First day has no return
        })
        
        # Add month identifier
        df["month"] = df["date"].dt.to_period("M").astype(str)
        
        # Calculate overall metrics
        metrics_calc = MetricsCalculator(
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            closed_trades=closed_trades,
            dates=dates,
            benchmark_returns=benchmark_returns
        )
        overall_metrics = metrics_calc.compute_all_metrics()
        
        # Group by month and aggregate
        monthly_data = []
        for month, group in df.groupby("month"):
            month_start = group["date"].min()
            month_end = group["date"].max()
            
            # Calculate monthly metrics
            month_start_equity = group["equity"].iloc[0]
            month_end_equity = group["equity"].iloc[-1]
            month_return = (month_end_equity / month_start_equity) - 1 if month_start_equity > 0 else 0.0
            
            # Count trades in this month
            month_trades = []
            month_pnl = 0.0
            winning_trades = 0
            losing_trades = 0
            
            for trade in closed_trades:
                if trade.exit_date is not None:
                    if month_start <= trade.exit_date <= month_end:
                        month_trades.append({
                            "symbol": trade.symbol,
                            "entry_date": trade.entry_date.isoformat() if trade.entry_date else None,
                            "exit_date": trade.exit_date.isoformat() if trade.exit_date else None,
                            "realized_pnl": trade.realized_pnl,
                            "r_multiple": self._calculate_r_multiple(trade)
                        })
                        month_pnl += trade.realized_pnl
                        if trade.realized_pnl > 0:
                            winning_trades += 1
                        else:
                            losing_trades += 1
            
            # Calculate monthly volatility (annualized)
            month_returns = group["daily_return"].values
            month_vol = np.std(month_returns) * np.sqrt(252) if len(month_returns) > 1 else 0.0
            
            # Calculate max drawdown for the month
            month_equity = group["equity"].values
            if len(month_equity) > 1:
                month_running_max = np.maximum.accumulate(month_equity)
                month_drawdowns = (month_equity - month_running_max) / month_running_max
                month_max_dd = abs(np.min(month_drawdowns)) if len(month_drawdowns) > 0 else 0.0
            else:
                month_max_dd = 0.0
            
            # Calculate Sharpe for the month
            month_sharpe = 0.0
            if month_vol > 0:
                month_mean_return = np.mean(month_returns)
                month_sharpe = (month_mean_return * 252) / (month_vol)
            
            monthly_data.append({
                "month": month,
                "month_start": month_start.isoformat(),
                "month_end": month_end.isoformat(),
                "start_equity": float(month_start_equity),
                "end_equity": float(month_end_equity),
                "monthly_return": float(month_return),
                "monthly_return_pct": float(month_return * 100),
                "trades_count": len(month_trades),
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": winning_trades / len(month_trades) if len(month_trades) > 0 else 0.0,
                "realized_pnl": float(month_pnl),
                "volatility_annualized": float(month_vol),
                "sharpe_ratio": float(month_sharpe),
                "max_drawdown": float(month_max_dd),
                "max_drawdown_pct": float(month_max_dd * 100),
                "trades": month_trades
            })
        
        # Create report structure
        report = {
            "generated_at": datetime.now().isoformat(),
            "period": {
                "start_date": dates[0].isoformat() if len(dates) > 0 else None,
                "end_date": dates[-1].isoformat() if len(dates) > 0 else None,
                "total_days": len(dates),
                "trading_days": len(daily_returns)
            },
            "overall_metrics": {
                k: float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in overall_metrics.items()
            },
            "monthly_summary": monthly_data
        }
        
        # Write to JSON
        output_path = self.output_dir / "monthly_report.json"
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return str(output_path)
    
    def write_scenario_comparison(
        self,
        scenarios: Dict[str, Dict]
    ) -> str:
        """Write scenario comparison JSON for stress test results.
        
        Args:
            scenarios: Dictionary mapping scenario name to metrics dict
                Example:
                {
                    "baseline": {"sharpe_ratio": 1.2, "max_drawdown": 0.12, ...},
                    "2x_slippage": {"sharpe_ratio": 0.8, "max_drawdown": 0.18, ...},
                    ...
                }
        
        Returns:
            Path to written file
        """
        # Convert numpy types to native Python types
        def convert_value(v):
            if isinstance(v, (np.integer, np.floating)):
                return float(v)
            elif isinstance(v, (np.ndarray, list)):
                return [convert_value(item) for item in v]
            elif isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            elif isinstance(v, pd.Timestamp):
                return v.isoformat()
            else:
                return v
        
        converted_scenarios = {
            name: {k: convert_value(v) for k, v in metrics.items()}
            for name, metrics in scenarios.items()
        }
        
        # Create comparison structure
        comparison = {
            "generated_at": datetime.now().isoformat(),
            "scenarios": converted_scenarios,
            "comparison": self._compare_scenarios(converted_scenarios)
        }
        
        # Write to JSON
        output_path = self.output_dir / "scenario_comparison.json"
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        return str(output_path)
    
    def _calculate_r_multiple(self, trade: Position) -> Optional[float]:
        """Calculate R-multiple for a trade.
        
        Args:
            trade: Position object
        
        Returns:
            R-multiple or None if cannot be calculated
        """
        if trade.exit_price is None or trade.entry_price is None:
            return None
        
        price_change = trade.exit_price - trade.entry_price
        risk = trade.entry_price - trade.initial_stop_price
        
        if risk > 0:
            return price_change / risk
        
        return None
    
    def _compare_scenarios(self, scenarios: Dict[str, Dict]) -> Dict:
        """Compare scenarios and identify best/worst performers.
        
        Args:
            scenarios: Dictionary of scenario metrics
        
        Returns:
            Comparison summary
        """
        if len(scenarios) == 0:
            return {}
        
        # Key metrics to compare
        key_metrics = [
            "sharpe_ratio",
            "max_drawdown",
            "calmar_ratio",
            "total_trades",
            "expectancy",
            "profit_factor",
            "win_rate"
        ]
        
        comparison = {}
        
        for metric in key_metrics:
            if metric not in scenarios[list(scenarios.keys())[0]]:
                continue
            
            values = {
                name: metrics.get(metric)
                for name, metrics in scenarios.items()
                if metrics.get(metric) is not None
            }
            
            if len(values) == 0:
                continue
            
            # For drawdown, lower is better; for others, higher is better
            if metric == "max_drawdown":
                best_scenario = min(values.items(), key=lambda x: x[1])
                worst_scenario = max(values.items(), key=lambda x: x[1])
            else:
                best_scenario = max(values.items(), key=lambda x: x[1])
                worst_scenario = min(values.items(), key=lambda x: x[1])
            
            comparison[metric] = {
                "best": {
                    "scenario": best_scenario[0],
                    "value": float(best_scenario[1])
                },
                "worst": {
                    "scenario": worst_scenario[0],
                    "value": float(worst_scenario[1])
                },
                "range": float(best_scenario[1] - worst_scenario[1])
            }
        
        return comparison


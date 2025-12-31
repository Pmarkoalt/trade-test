"""CSV output generation for backtest results."""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from ..models.positions import Position


class CSVWriter:
    """Write backtest results to CSV files."""
    
    def __init__(self, output_dir: str):
        """Initialize CSV writer.
        
        Args:
            output_dir: Directory to write CSV files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def write_equity_curve(
        self,
        equity_curve: List[float],
        dates: List[pd.Timestamp],
        cash_history: List[float],
        positions_count_history: List[int],
        exposure_history: List[float]
    ) -> str:
        """Write equity curve CSV.
        
        Args:
            equity_curve: List of daily equity values
            dates: List of dates corresponding to equity_curve
            cash_history: List of daily cash values
            positions_count_history: List of daily position counts
            exposure_history: List of daily gross exposure values
        
        Returns:
            Path to written file
        """
        if len(equity_curve) != len(dates):
            raise ValueError(
                f"equity_curve length ({len(equity_curve)}) must match "
                f"dates length ({len(dates)})"
            )
        
        # Create DataFrame
        df = pd.DataFrame({
            "date": dates,
            "equity": equity_curve,
            "cash": cash_history[:len(dates)],
            "positions": positions_count_history[:len(dates)],
            "exposure": exposure_history[:len(dates)]
        })
        
        # Calculate exposure as percentage
        df["exposure_pct"] = (df["exposure"] / df["equity"]) * 100
        
        # Write to CSV
        output_path = self.output_dir / "equity_curve.csv"
        df.to_csv(output_path, index=False)
        
        return str(output_path)
    
    def write_trade_log(self, closed_trades: List[Position]) -> str:
        """Write trade log CSV with all closed trades.
        
        Args:
            closed_trades: List of closed Position objects
        
        Returns:
            Path to written file
        """
        if len(closed_trades) == 0:
            # Create empty DataFrame with correct columns
            df = pd.DataFrame(columns=[
                "trade_id", "symbol", "asset_class", "entry_date", "exit_date",
                "entry_price", "exit_price", "quantity", "entry_fill_id", "exit_fill_id",
                "realized_pnl", "entry_slippage_bps", "entry_fee_bps", "entry_total_cost",
                "exit_slippage_bps", "exit_fee_bps", "exit_total_cost", "total_cost",
                "initial_stop_price", "stop_price", "exit_reason", "triggered_on",
                "holding_days", "r_multiple", "adv20_at_entry"
            ])
        else:
            rows = []
            for i, trade in enumerate(closed_trades):
                # Calculate holding period
                holding_days = None
                if trade.entry_date is not None and trade.exit_date is not None:
                    holding_days = (trade.exit_date - trade.entry_date).days
                
                # Calculate R-multiple
                r_multiple = None
                if trade.exit_price is not None and trade.entry_price is not None:
                    price_change = trade.exit_price - trade.entry_price
                    risk = trade.entry_price - trade.initial_stop_price
                    if risk > 0:
                        r_multiple = price_change / risk
                
                # Calculate total cost
                total_cost = trade.entry_total_cost
                if trade.exit_total_cost is not None:
                    total_cost += trade.exit_total_cost
                
                row = {
                    "trade_id": i + 1,
                    "symbol": trade.symbol,
                    "asset_class": trade.asset_class,
                    "entry_date": trade.entry_date,
                    "exit_date": trade.exit_date,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "quantity": trade.quantity,
                    "entry_fill_id": trade.entry_fill_id,
                    "exit_fill_id": trade.exit_fill_id,
                    "realized_pnl": trade.realized_pnl,
                    "entry_slippage_bps": trade.entry_slippage_bps,
                    "entry_fee_bps": trade.entry_fee_bps,
                    "entry_total_cost": trade.entry_total_cost,
                    "exit_slippage_bps": trade.exit_slippage_bps,
                    "exit_fee_bps": trade.exit_fee_bps,
                    "exit_total_cost": trade.exit_total_cost,
                    "total_cost": total_cost,
                    "initial_stop_price": trade.initial_stop_price,
                    "stop_price": trade.stop_price,
                    "exit_reason": trade.exit_reason.value if trade.exit_reason else None,
                    "triggered_on": trade.triggered_on.value if trade.triggered_on else None,
                    "holding_days": holding_days,
                    "r_multiple": r_multiple,
                    "adv20_at_entry": trade.adv20_at_entry
                }
                rows.append(row)
            
            df = pd.DataFrame(rows)
        
        # Write to CSV
        output_path = self.output_dir / "trade_log.csv"
        df.to_csv(output_path, index=False)
        
        return str(output_path)
    
    def write_weekly_summary(
        self,
        equity_curve: List[float],
        dates: List[pd.Timestamp],
        daily_returns: List[float],
        closed_trades: List[Position]
    ) -> str:
        """Write weekly summary CSV with aggregated metrics.
        
        Args:
            equity_curve: List of daily equity values
            dates: List of dates corresponding to equity_curve
            daily_returns: List of daily portfolio returns
            closed_trades: List of closed Position objects
        
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
        
        # Add week identifier
        df["week"] = df["date"].dt.to_period("W").astype(str)
        
        # Group by week and aggregate
        weekly_data = []
        for week, group in df.groupby("week"):
            week_start = group["date"].min()
            week_end = group["date"].max()
            
            # Calculate weekly metrics
            week_start_equity = group["equity"].iloc[0]
            week_end_equity = group["equity"].iloc[-1]
            week_return = (week_end_equity / week_start_equity) - 1 if week_start_equity > 0 else 0.0
            
            # Count trades in this week
            week_trades = 0
            week_pnl = 0.0
            for trade in closed_trades:
                if trade.exit_date is not None:
                    if week_start <= trade.exit_date <= week_end:
                        week_trades += 1
                        week_pnl += trade.realized_pnl
            
            # Calculate weekly volatility (annualized)
            week_returns = group["daily_return"].values
            week_vol = np.std(week_returns) * np.sqrt(252) if len(week_returns) > 1 else 0.0
            
            # Calculate max drawdown for the week
            week_equity = group["equity"].values
            if len(week_equity) > 1:
                week_running_max = np.maximum.accumulate(week_equity)
                week_drawdowns = (week_equity - week_running_max) / week_running_max
                week_max_dd = abs(np.min(week_drawdowns)) if len(week_drawdowns) > 0 else 0.0
            else:
                week_max_dd = 0.0
            
            weekly_data.append({
                "week": week,
                "week_start": week_start,
                "week_end": week_end,
                "start_equity": week_start_equity,
                "end_equity": week_end_equity,
                "weekly_return": week_return,
                "weekly_return_pct": week_return * 100,
                "trades": week_trades,
                "realized_pnl": week_pnl,
                "volatility_annualized": week_vol,
                "max_drawdown": week_max_dd,
                "max_drawdown_pct": week_max_dd * 100
            })
        
        weekly_df = pd.DataFrame(weekly_data)
        
        # Write to CSV
        output_path = self.output_dir / "weekly_summary.csv"
        weekly_df.to_csv(output_path, index=False)
        
        return str(output_path)


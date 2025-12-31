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
        
        # Ensure all arrays have the same length
        # Validate lengths match
        if len(equity_curve) != len(dates):
            raise ValueError(
                f"equity_curve length ({len(equity_curve)}) must match "
                f"dates length ({len(dates)})"
            )
        
        # Align daily_returns to match dates/equity_curve length
        # If daily_returns is one shorter (typical), prepend 0.0 for first day
        if len(daily_returns) == len(dates) - 1:
            daily_returns_aligned = [0.0] + daily_returns
        elif len(daily_returns) == len(dates):
            daily_returns_aligned = daily_returns
        else:
            # Handle mismatched lengths: pad or truncate
            if len(daily_returns) < len(dates):
                # Pad with zeros at the beginning
                daily_returns_aligned = [0.0] * (len(dates) - len(daily_returns)) + daily_returns
            else:
                # Truncate to match dates length
                daily_returns_aligned = daily_returns[:len(dates)]
        
        # Final validation
        if len(daily_returns_aligned) != len(dates):
            raise ValueError(
                f"Failed to align daily_returns: expected {len(dates)}, got {len(daily_returns_aligned)}"
            )
        
        # Create DataFrame with daily data
        df = pd.DataFrame({
            "date": dates,
            "equity": equity_curve,
            "daily_return": daily_returns_aligned
        })
        
        # Add week identifier
        df["week"] = df["date"].dt.to_period("W").astype(str)
        
        # Group by week and aggregate
        weekly_data = []
        for week, group in df.groupby("week"):
            # Ensure week_start and week_end are scalars
            week_start = pd.Timestamp(group["date"].min())
            week_end = pd.Timestamp(group["date"].max())
            
            # Calculate weekly metrics
            week_start_equity = float(group["equity"].iloc[0])
            week_end_equity = float(group["equity"].iloc[-1])
            week_return = float((week_end_equity / week_start_equity) - 1 if week_start_equity > 0 else 0.0)
            
            # Count trades in this week
            week_trades = 0
            week_pnl = 0.0
            for trade in closed_trades:
                if hasattr(trade, 'exit_date') and trade.exit_date is not None:
                    trade_exit = pd.Timestamp(trade.exit_date)
                    if week_start <= trade_exit <= week_end:
                        week_trades += 1
                        week_pnl += float(getattr(trade, 'realized_pnl', 0.0))
            
            # Calculate weekly volatility (annualized)
            week_returns = group["daily_return"].values
            week_vol = float(np.std(week_returns) * np.sqrt(252) if len(week_returns) > 1 else 0.0)
            
            # Calculate max drawdown for the week
            week_equity = group["equity"].values
            if len(week_equity) > 1:
                week_running_max = np.maximum.accumulate(week_equity)
                week_drawdowns = (week_equity - week_running_max) / week_running_max
                week_max_dd = float(abs(np.min(week_drawdowns)) if len(week_drawdowns) > 0 else 0.0)
            else:
                week_max_dd = 0.0
            
            # Ensure all values are scalars (not arrays)
            weekly_data.append({
                "week": str(week),  # Convert Period to string
                "week_start": week_start.strftime("%Y-%m-%d"),
                "week_end": week_end.strftime("%Y-%m-%d"),
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
        
        # Create DataFrame from list of dicts (each dict is a row)
        if not weekly_data:
            # If no weekly data, create empty DataFrame with correct columns
            weekly_df = pd.DataFrame(columns=[
                "week", "week_start", "week_end", "start_equity", "end_equity",
                "weekly_return", "weekly_return_pct", "trades", "realized_pnl",
                "volatility_annualized", "max_drawdown", "max_drawdown_pct"
            ])
        else:
            # Ensure all values in each dict are scalars (not arrays/lists)
            # Convert any potential arrays to scalars
            cleaned_weekly_data = []
            for row in weekly_data:
                cleaned_row = {}
                for key, value in row.items():
                    # Convert to scalar if it's an array/list
                    if isinstance(value, (list, np.ndarray, pd.Series)):
                        if len(value) > 0:
                            cleaned_row[key] = value[0] if len(value) == 1 else str(value)
                        else:
                            cleaned_row[key] = "" if isinstance(value, (list, pd.Series)) else 0.0
                    else:
                        cleaned_row[key] = value
                cleaned_weekly_data.append(cleaned_row)
            weekly_df = pd.DataFrame(cleaned_weekly_data)
        
        # Write to CSV
        output_path = self.output_dir / "weekly_summary.csv"
        weekly_df.to_csv(output_path, index=False)
        
        return str(output_path)


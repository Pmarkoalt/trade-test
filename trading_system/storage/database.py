"""Database operations for storing and querying backtest results."""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..models.positions import Position
from ..reporting.metrics import MetricsCalculator
from .schema import create_schema

logger = logging.getLogger(__name__)


def get_default_db_path() -> Path:
    """Get default database path.

    Returns:
        Path to default SQLite database file
    """
    # Default to results directory in project root
    default_path = Path("results/backtest_results.db")
    default_path.parent.mkdir(parents=True, exist_ok=True)
    return default_path


class ResultsDatabase:
    """Database interface for storing and querying backtest results."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize results database.

        Args:
            db_path: Path to SQLite database file (default: results/backtest_results.db)
        """
        if db_path is None:
            db_path = get_default_db_path()

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            create_schema(conn)
        finally:
            conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection.

        Returns:
            SQLite connection
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn

    def store_results(
        self,
        results: Dict[str, Any],
        config_path: Optional[str] = None,
        strategy_name: Optional[str] = None,
        period: str = "train",
        split_name: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> int:
        """Store backtest results in database.

        Args:
            results: Results dictionary from engine.run() or similar
            config_path: Path to configuration file used
            strategy_name: Name of strategy used
            period: Period name (train/validation/holdout)
            split_name: Name of walk-forward split
            notes: Optional notes about this run

        Returns:
            run_id of stored results
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Extract run metadata
            start_date = results.get("start_date")
            end_date = results.get("end_date")
            starting_equity = results.get("starting_equity", 100000.0)

            # Convert dates to strings if they're Timestamp objects
            if isinstance(start_date, pd.Timestamp):
                start_date = start_date.isoformat()
            if isinstance(end_date, pd.Timestamp):
                end_date = end_date.isoformat()

            # Insert backtest run
            cursor.execute(
                """
                INSERT OR IGNORE INTO backtest_runs 
                (config_path, strategy_name, split_name, period, start_date, end_date, starting_equity, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (config_path, strategy_name, split_name, period, start_date, end_date, starting_equity, notes),
            )

            # Get run_id
            if cursor.lastrowid == 0:
                # Row already exists, fetch existing run_id
                cursor.execute(
                    """
                    SELECT run_id FROM backtest_runs
                    WHERE config_path = ? AND strategy_name = ? AND split_name = ? 
                    AND period = ? AND start_date = ? AND end_date = ?
                """,
                    (config_path, strategy_name, split_name, period, start_date, end_date),
                )
                row = cursor.fetchone()
                if row:
                    run_id = row[0]
                    # Delete existing data for this run
                    self._delete_run_data(cursor, run_id)
                else:
                    raise ValueError("Failed to insert or find run_id")
            else:
                run_id = cursor.lastrowid

            # Store metrics
            self._store_metrics(cursor, run_id, results)

            # Store trades
            closed_trades = results.get("closed_trades", [])
            if closed_trades:
                self._store_trades(cursor, run_id, closed_trades)

            # Store equity curve
            equity_curve = results.get("equity_curve", [])
            daily_events = results.get("daily_events", [])
            if equity_curve and daily_events:
                self._store_equity_curve(cursor, run_id, equity_curve, daily_events)

            # Store daily returns
            daily_returns = results.get("daily_returns", [])
            dates = [event.get("date") for event in daily_events] if daily_events else []
            if daily_returns and dates:
                self._store_daily_returns(cursor, run_id, daily_returns, dates)

            # Compute and store monthly summary
            if equity_curve and daily_returns and dates and closed_trades:
                self._store_monthly_summary(cursor, run_id, equity_curve, daily_returns, dates, closed_trades)

            conn.commit()
            logger.info(f"Stored results with run_id={run_id}")
            return run_id

        except Exception as e:
            conn.rollback()
            logger.error(f"Error storing results: {e}", exc_info=True)
            raise
        finally:
            conn.close()

    def _delete_run_data(self, cursor: sqlite3.Cursor, run_id: int) -> None:
        """Delete all data for a run (used when updating existing run).

        Args:
            cursor: Database cursor
            run_id: Run ID to delete
        """
        cursor.execute("DELETE FROM monthly_summary WHERE run_id = ?", (run_id,))
        cursor.execute("DELETE FROM daily_returns WHERE run_id = ?", (run_id,))
        cursor.execute("DELETE FROM equity_curve WHERE run_id = ?", (run_id,))
        cursor.execute("DELETE FROM trades WHERE run_id = ?", (run_id,))
        cursor.execute("DELETE FROM run_metrics WHERE run_id = ?", (run_id,))

    def _store_metrics(self, cursor: sqlite3.Cursor, run_id: int, results: Dict[str, Any]) -> None:
        """Store metrics for a run.

        Args:
            cursor: Database cursor
            run_id: Run ID
            results: Results dictionary
        """
        # Extract basic metrics from results
        metrics = {
            "run_id": run_id,
            "sharpe_ratio": results.get("sharpe_ratio"),
            "max_drawdown": results.get("max_drawdown"),
            "total_return": results.get("total_return"),
            "total_trades": results.get("total_trades", 0),
            "winning_trades": results.get("winning_trades", 0),
            "losing_trades": results.get("losing_trades", 0),
            "win_rate": results.get("win_rate"),
            "avg_r_multiple": results.get("avg_r_multiple"),
            "realized_pnl": results.get("realized_pnl"),
            "final_cash": results.get("final_cash"),
            "final_positions": results.get("final_positions", 0),
            "ending_equity": results.get("ending_equity"),
        }

        # Compute additional metrics if we have the data
        equity_curve = results.get("equity_curve", [])
        daily_returns = results.get("daily_returns", [])
        closed_trades = results.get("closed_trades", [])
        dates = results.get("dates", [])
        benchmark_returns = results.get("benchmark_returns")

        if equity_curve and daily_returns and closed_trades:
            try:
                metrics_calc = MetricsCalculator(
                    equity_curve=equity_curve,
                    daily_returns=daily_returns,
                    closed_trades=closed_trades,
                    dates=dates if dates else None,
                    benchmark_returns=benchmark_returns,
                )
                all_metrics = metrics_calc.compute_all_metrics()

                # Add computed metrics
                metrics.update(
                    {
                        "calmar_ratio": all_metrics.get("calmar_ratio"),
                        "expectancy": all_metrics.get("expectancy"),
                        "profit_factor": all_metrics.get("profit_factor"),
                        "correlation_to_benchmark": all_metrics.get("correlation_to_benchmark"),
                        "percentile_99_daily_loss": all_metrics.get("percentile_99_daily_loss"),
                        "recovery_factor": all_metrics.get("recovery_factor"),
                        "drawdown_duration": int(all_metrics.get("drawdown_duration", 0)),
                        "turnover": all_metrics.get("turnover"),
                        "average_holding_period": all_metrics.get("average_holding_period"),
                        "max_consecutive_losses": int(all_metrics.get("max_consecutive_losses", 0)),
                    }
                )
            except Exception as e:
                logger.warning(f"Error computing additional metrics: {e}")

        # Insert metrics
        cursor.execute(
            """
            INSERT INTO run_metrics (
                run_id, sharpe_ratio, max_drawdown, calmar_ratio, total_return,
                total_trades, winning_trades, losing_trades, win_rate, avg_r_multiple,
                realized_pnl, final_cash, final_positions, ending_equity,
                expectancy, profit_factor, correlation_to_benchmark,
                percentile_99_daily_loss, recovery_factor, drawdown_duration,
                turnover, average_holding_period, max_consecutive_losses
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                metrics["run_id"],
                metrics.get("sharpe_ratio"),
                metrics.get("max_drawdown"),
                metrics.get("calmar_ratio"),
                metrics.get("total_return"),
                metrics.get("total_trades", 0),
                metrics.get("winning_trades", 0),
                metrics.get("losing_trades", 0),
                metrics.get("win_rate"),
                metrics.get("avg_r_multiple"),
                metrics.get("realized_pnl"),
                metrics.get("final_cash"),
                metrics.get("final_positions", 0),
                metrics.get("ending_equity"),
                metrics.get("expectancy"),
                metrics.get("profit_factor"),
                metrics.get("correlation_to_benchmark"),
                metrics.get("percentile_99_daily_loss"),
                metrics.get("recovery_factor"),
                metrics.get("drawdown_duration"),
                metrics.get("turnover"),
                metrics.get("average_holding_period"),
                metrics.get("max_consecutive_losses"),
            ),
        )

    def _store_trades(self, cursor: sqlite3.Cursor, run_id: int, trades: List[Position]) -> None:
        """Store trades for a run.

        Args:
            cursor: Database cursor
            run_id: Run ID
            trades: List of Position objects
        """
        for trade in trades:
            # Convert dates to strings
            entry_date = trade.entry_date.isoformat() if isinstance(trade.entry_date, pd.Timestamp) else str(trade.entry_date)
            exit_date = (
                trade.exit_date.isoformat()
                if isinstance(trade.exit_date, pd.Timestamp)
                else (str(trade.exit_date) if trade.exit_date else None)
            )

            # Compute R-multiple if possible
            r_multiple = None
            try:
                if trade.exit_price is not None:
                    r_multiple = trade.compute_r_multiple()
            except Exception:
                pass

            cursor.execute(
                """
                INSERT INTO trades (
                    run_id, symbol, asset_class, entry_date, exit_date,
                    entry_price, exit_price, quantity, realized_pnl, r_multiple,
                    exit_reason, entry_fill_id, exit_fill_id,
                    entry_slippage_bps, exit_slippage_bps,
                    entry_fee_bps, exit_fee_bps,
                    entry_total_cost, exit_total_cost,
                    initial_stop_price, adv20_at_entry
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    run_id,
                    trade.symbol,
                    trade.asset_class,
                    entry_date,
                    exit_date,
                    trade.entry_price,
                    trade.exit_price,
                    trade.quantity,
                    trade.realized_pnl,
                    r_multiple,
                    trade.exit_reason.value if trade.exit_reason else None,
                    trade.entry_fill_id,
                    trade.exit_fill_id,
                    trade.entry_slippage_bps,
                    trade.exit_slippage_bps,
                    trade.entry_fee_bps,
                    trade.exit_fee_bps,
                    trade.entry_total_cost,
                    trade.exit_total_cost,
                    trade.initial_stop_price,
                    trade.adv20_at_entry,
                ),
            )

    def _store_equity_curve(
        self, cursor: sqlite3.Cursor, run_id: int, equity_curve: List[float], daily_events: List[Dict]
    ) -> None:
        """Store equity curve for a run.

        Args:
            cursor: Database cursor
            run_id: Run ID
            equity_curve: List of equity values
            daily_events: List of daily event dictionaries
        """
        for i, event in enumerate(daily_events):
            date = event.get("date")
            if isinstance(date, pd.Timestamp):
                date = date.isoformat()

            portfolio_state = event.get("portfolio_state", {})
            equity = equity_curve[i] if i < len(equity_curve) else None

            cursor.execute(
                """
                INSERT INTO equity_curve (run_id, date, equity, cash, open_positions, gross_exposure)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    run_id,
                    date,
                    equity,
                    portfolio_state.get("cash"),
                    portfolio_state.get("open_positions"),
                    portfolio_state.get("gross_exposure"),
                ),
            )

    def _store_daily_returns(self, cursor: sqlite3.Cursor, run_id: int, daily_returns: List[float], dates: List) -> None:
        """Store daily returns for a run.

        Args:
            cursor: Database cursor
            run_id: Run ID
            daily_returns: List of daily return values
            dates: List of dates
        """
        for i, date in enumerate(dates):
            if i >= len(daily_returns):
                break

            if isinstance(date, pd.Timestamp):
                date = date.isoformat()

            cursor.execute(
                """
                INSERT INTO daily_returns (run_id, date, daily_return)
                VALUES (?, ?, ?)
            """,
                (run_id, date, daily_returns[i]),
            )

    def _store_monthly_summary(
        self,
        cursor: sqlite3.Cursor,
        run_id: int,
        equity_curve: List[float],
        daily_returns: List[float],
        dates: List,
        closed_trades: List[Position],
    ) -> None:
        """Compute and store monthly summary for a run.

        Args:
            cursor: Database cursor
            run_id: Run ID
            equity_curve: List of equity values
            daily_returns: List of daily returns
            dates: List of dates
            closed_trades: List of closed Position objects
        """
        # Create DataFrame
        df = pd.DataFrame(
            {
                "date": [pd.Timestamp(d) if not isinstance(d, pd.Timestamp) else d for d in dates],
                "equity": equity_curve[: len(dates)],
                "daily_return": daily_returns[: len(dates)],
            }
        )

        # Add month identifier
        df["month"] = df["date"].dt.to_period("M").astype(str)

        # Group by month
        for month, group in df.groupby("month"):
            month_start = group["date"].min()
            month_end = group["date"].max()
            month_start_equity = group["equity"].iloc[0]
            month_end_equity = group["equity"].iloc[-1]
            monthly_return = (month_end_equity / month_start_equity) - 1 if month_start_equity > 0 else 0.0

            # Count trades in this month
            month_trades = []
            month_pnl = 0.0
            winning_trades = 0
            losing_trades = 0

            for trade in closed_trades:
                if trade.exit_date is not None:
                    exit_date = trade.exit_date if isinstance(trade.exit_date, pd.Timestamp) else pd.Timestamp(trade.exit_date)
                    if month_start <= exit_date <= month_end:
                        month_trades.append(trade)
                        month_pnl += trade.realized_pnl
                        if trade.realized_pnl > 0:
                            winning_trades += 1
                        else:
                            losing_trades += 1

            # Calculate monthly metrics
            month_returns = group["daily_return"].values
            month_vol = np.std(month_returns) * np.sqrt(252) if len(month_returns) > 1 else 0.0

            month_equity = group["equity"].values
            if len(month_equity) > 1:
                month_running_max = np.maximum.accumulate(month_equity)
                month_drawdowns = (month_equity - month_running_max) / month_running_max
                month_max_dd = abs(np.min(month_drawdowns)) if len(month_drawdowns) > 0 else 0.0
            else:
                month_max_dd = 0.0

            month_sharpe = 0.0
            if month_vol > 0:
                month_mean_return = np.mean(month_returns)
                month_sharpe = (month_mean_return * 252) / month_vol

            win_rate = winning_trades / len(month_trades) if len(month_trades) > 0 else 0.0

            cursor.execute(
                """
                INSERT INTO monthly_summary (
                    run_id, month, month_start, month_end,
                    start_equity, end_equity, monthly_return,
                    trades_count, winning_trades, losing_trades, win_rate,
                    realized_pnl, volatility_annualized, sharpe_ratio, max_drawdown
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    run_id,
                    month,
                    month_start.isoformat(),
                    month_end.isoformat(),
                    float(month_start_equity),
                    float(month_end_equity),
                    float(monthly_return),
                    len(month_trades),
                    winning_trades,
                    losing_trades,
                    float(win_rate),
                    float(month_pnl),
                    float(month_vol),
                    float(month_sharpe),
                    float(month_max_dd),
                ),
            )

    def get_run(self, run_id: int) -> Optional[Dict[str, Any]]:
        """Get backtest run by ID.

        Args:
            run_id: Run ID

        Returns:
            Dictionary with run data, or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Get run metadata
            cursor.execute(
                """
                SELECT * FROM backtest_runs WHERE run_id = ?
            """,
                (run_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            run_data = dict(row)

            # Get metrics
            cursor.execute("SELECT * FROM run_metrics WHERE run_id = ?", (run_id,))
            metrics_row = cursor.fetchone()
            if metrics_row:
                run_data["metrics"] = dict(metrics_row)

            return run_data
        finally:
            conn.close()

    def query_runs(
        self,
        config_path: Optional[str] = None,
        strategy_name: Optional[str] = None,
        period: Optional[str] = None,
        split_name: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query backtest runs with filters.

        Args:
            config_path: Filter by config path
            strategy_name: Filter by strategy name
            period: Filter by period (train/validation/holdout)
            split_name: Filter by split name
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            limit: Maximum number of results

        Returns:
            List of run dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            query = "SELECT * FROM backtest_runs WHERE 1=1"
            params = []

            if config_path:
                query += " AND config_path = ?"
                params.append(config_path)
            if strategy_name:
                query += " AND strategy_name = ?"
                params.append(strategy_name)
            if period:
                query += " AND period = ?"
                params.append(period)
            if split_name:
                query += " AND split_name = ?"
                params.append(split_name)
            if start_date:
                query += " AND start_date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND end_date <= ?"
                params.append(end_date)

            query += " ORDER BY created_at DESC"

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            runs = []
            for row in rows:
                run_data = dict(row)
                run_id = run_data["run_id"]

                # Get metrics
                cursor.execute("SELECT * FROM run_metrics WHERE run_id = ?", (run_id,))
                metrics_row = cursor.fetchone()
                if metrics_row:
                    run_data["metrics"] = dict(metrics_row)

                runs.append(run_data)

            return runs
        finally:
            conn.close()

    def get_equity_curve(self, run_id: int) -> pd.DataFrame:
        """Get equity curve for a run.

        Args:
            run_id: Run ID

        Returns:
            DataFrame with date, equity, cash, open_positions, gross_exposure
        """
        conn = self._get_connection()
        try:
            df = pd.read_sql_query(
                "SELECT date, equity, cash, open_positions, gross_exposure FROM equity_curve WHERE run_id = ? ORDER BY date",
                conn,
                params=(run_id,),
                parse_dates=["date"],
            )
            return df
        finally:
            conn.close()

    def get_trades(self, run_id: int) -> pd.DataFrame:
        """Get trades for a run.

        Args:
            run_id: Run ID

        Returns:
            DataFrame with all trade data
        """
        conn = self._get_connection()
        try:
            df = pd.read_sql_query(
                "SELECT * FROM trades WHERE run_id = ? ORDER BY entry_date",
                conn,
                params=(run_id,),
                parse_dates=["entry_date", "exit_date"],
            )
            return df
        finally:
            conn.close()

    def get_monthly_summary(self, run_id: int) -> pd.DataFrame:
        """Get monthly summary for a run.

        Args:
            run_id: Run ID

        Returns:
            DataFrame with monthly metrics
        """
        conn = self._get_connection()
        try:
            df = pd.read_sql_query(
                "SELECT * FROM monthly_summary WHERE run_id = ? ORDER BY month",
                conn,
                params=(run_id,),
                parse_dates=["month_start", "month_end"],
            )
            return df
        finally:
            conn.close()

    def compare_runs(self, run_ids: List[int], metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """Compare multiple runs by metrics.

        Args:
            run_ids: List of run IDs to compare
            metrics: List of metric names to compare (default: all primary metrics)

        Returns:
            DataFrame with metrics for each run
        """
        if metrics is None:
            metrics = [
                "sharpe_ratio",
                "max_drawdown",
                "calmar_ratio",
                "total_return",
                "total_trades",
                "win_rate",
                "avg_r_multiple",
                "profit_factor",
                "expectancy",
                "turnover",
            ]

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Get run metadata
            placeholders = ",".join(["?" for _ in run_ids])
            cursor.execute(
                f"""
                SELECT run_id, config_path, strategy_name, split_name, period, start_date, end_date
                FROM backtest_runs
                WHERE run_id IN ({placeholders})
            """,
                run_ids,
            )

            runs_data = {row["run_id"]: dict(row) for row in cursor.fetchall()}

            # Get metrics for each run
            comparison_data = []
            for run_id in run_ids:
                cursor.execute("SELECT * FROM run_metrics WHERE run_id = ?", (run_id,))
                metrics_row = cursor.fetchone()
                if not metrics_row:
                    continue

                run_info = runs_data.get(run_id, {})
                row_data = {
                    "run_id": run_id,
                    "config_path": run_info.get("config_path"),
                    "strategy_name": run_info.get("strategy_name"),
                    "split_name": run_info.get("split_name"),
                    "period": run_info.get("period"),
                    "start_date": run_info.get("start_date"),
                    "end_date": run_info.get("end_date"),
                }

                metrics_dict = dict(metrics_row)
                for metric in metrics:
                    row_data[metric] = metrics_dict.get(metric)

                comparison_data.append(row_data)

            return pd.DataFrame(comparison_data)
        finally:
            conn.close()

    def archive_runs(self, run_ids: List[int], archive_db_path: Optional[Path] = None) -> None:
        """Archive runs to a separate database.

        Note: This is a simplified implementation that copies run metadata.
        For full archival including all trades and equity curves, use database
        backup/restore tools or implement a more comprehensive copy mechanism.

        Args:
            run_ids: List of run IDs to archive
            archive_db_path: Path to archive database (default: results/backtest_results_archive.db)
        """
        if archive_db_path is None:
            archive_db_path = self.db_path.parent / f"{self.db_path.stem}_archive{self.db_path.suffix}"

        archive_db_path = Path(archive_db_path)
        archive_db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create archive database
        archive_db = ResultsDatabase(archive_db_path)

        conn = self._get_connection()

        try:
            cursor = conn.cursor()
            archived_count = 0

            # Copy each run's data to archive
            for run_id in run_ids:
                # Get run data
                cursor.execute("SELECT * FROM backtest_runs WHERE run_id = ?", (run_id,))
                run_row = cursor.fetchone()
                if not run_row:
                    logger.warning(f"Run ID {run_id} not found, skipping")
                    continue

                run_dict = dict(run_row)
                archived_count += 1
                logger.info(f"Archived run_id {run_id} metadata to archive database")
                # Note: Full implementation would copy all related tables (trades, equity_curve, etc.)

            # Note: Actual deletion from main database is commented out for safety
            # Uncomment the following lines if you want to delete archived runs:
            # placeholders = ','.join(['?' for _ in run_ids])
            # cursor.execute(f"DELETE FROM backtest_runs WHERE run_id IN ({placeholders})", run_ids)
            # conn.commit()

            logger.info(f"Archived metadata for {archived_count} runs to {archive_db_path}")
            logger.warning(
                "Full archival (trades, equity curves) not implemented - use database backup tools for complete archival"
            )
        finally:
            conn.close()

    def delete_run(self, run_id: int) -> None:
        """Delete a run and all associated data.

        Args:
            run_id: Run ID to delete
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Delete in order (respecting foreign key constraints)
            self._delete_run_data(cursor, run_id)
            cursor.execute("DELETE FROM backtest_runs WHERE run_id = ?", (run_id,))
            conn.commit()
            logger.info(f"Deleted run_id={run_id}")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error deleting run: {e}", exc_info=True)
            raise
        finally:
            conn.close()

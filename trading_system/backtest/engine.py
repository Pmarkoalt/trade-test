"""Main backtest engine for walk-forward testing."""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import logging
from pathlib import Path

from ..models.market_data import MarketData
from ..portfolio.portfolio import Portfolio
from ..models.positions import Position
from ..strategies.base_strategy import BaseStrategy
from ..data.loader import load_all_data
from ..indicators.feature_computer import compute_features
from ..data.calendar import get_trading_days
from .splits import WalkForwardSplit
from .event_loop import DailyEventLoop

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Main backtest engine with walk-forward splits and event-driven loop.
    
    This engine implements:
    - Walk-forward splits (train/validation/holdout)
    - Event-driven daily loop with no lookahead
    - Portfolio state management
    - Trade logging and metrics
    """
    
    def __init__(
        self,
        market_data: MarketData,
        strategies: List[BaseStrategy],
        starting_equity: float = 100000.0,
        seed: Optional[int] = None
    ):
        """Initialize backtest engine.
        
        Args:
            market_data: MarketData container with bars and benchmarks
            strategies: List of strategy objects (equity, crypto)
            starting_equity: Starting equity (default: 100,000)
            seed: Optional random seed for reproducibility
        """
        self.market_data = market_data
        self.strategies = strategies
        self.starting_equity = starting_equity
        self.seed = seed
        
        # Initialize random number generator
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        
        # Results storage
        self.results: Dict[str, Any] = {}
        self.daily_events: List[Dict] = []
        self.closed_trades: List[Position] = []
        
        # Initialize portfolio
        self.portfolio = Portfolio(
            date=pd.Timestamp.now(),  # Will be updated during backtest
            cash=starting_equity,
            starting_equity=starting_equity,
            equity=starting_equity
        )
    
    def run(
        self,
        split: WalkForwardSplit,
        period: str = "train"
    ) -> Dict[str, Any]:
        """Run backtest for a specific split and period.
        
        Args:
            split: WalkForwardSplit configuration
            period: One of "train", "validation", "holdout"
        
        Returns:
            Dictionary with backtest results
        """
        # Get date range for period
        start_date, end_date = split.get_period_dates(period)
        
        logger.info(
            f"Running backtest: {split.name} - {period} "
            f"({start_date.date()} to {end_date.date()})"
        )
        
        # Get all trading days in range
        all_dates = self._get_all_dates()
        trading_dates = [d for d in all_dates if start_date <= d <= end_date]
        
        if not trading_dates:
            logger.warning(f"No trading dates found in range {start_date} to {end_date}")
            return self._empty_results()
        
        logger.info(f"Processing {len(trading_dates)} trading days")
        
        # Initialize event loop
        event_loop = self._create_event_loop()
        
        # Reset portfolio for this run
        self.portfolio = Portfolio(
            date=start_date,
            cash=self.starting_equity,
            starting_equity=self.starting_equity,
            equity=self.starting_equity
        )
        
        # Process each day
        self.daily_events = []
        for date in trading_dates:
            try:
                events = event_loop.process_day(date)
                self.daily_events.append(events)
            except Exception as e:
                logger.error(f"Error processing day {date}: {e}")
                continue
        
        # Collect closed trades
        self.closed_trades = [
            p for p in self.portfolio.positions.values() if not p.is_open()
        ]
        
        # Compute results
        results = self._compute_results(split, period)
        
        return results
    
    def _get_all_dates(self) -> List[pd.Timestamp]:
        """Get all available dates from market data.
        
        Returns:
            Sorted list of all dates present in market data
        """
        all_dates = set()
        
        # Get dates from all symbols
        for symbol, bars_df in self.market_data.bars.items():
            all_dates.update(bars_df.index)
        
        # Get dates from benchmarks
        for benchmark_df in self.market_data.benchmarks.values():
            all_dates.update(benchmark_df.index)
        
        return sorted(list(all_dates))
    
    def _create_event_loop(self) -> DailyEventLoop:
        """Create event loop instance.
        
        Returns:
            DailyEventLoop instance
        """
        def compute_features_fn(df_ohlc, symbol, asset_class, benchmark_roc60=None, benchmark_returns=None):
            """Wrapper for compute_features."""
            return compute_features(
                df_ohlc=df_ohlc,
                symbol=symbol,
                asset_class=asset_class,
                benchmark_roc60=benchmark_roc60,
                benchmark_returns=benchmark_returns
            )
        
        def get_next_trading_day(date: pd.Timestamp) -> pd.Timestamp:
            """Get next trading day.
            
            For equity: skip weekends
            For crypto: next calendar day
            """
            # Simple implementation: next calendar day
            # In production, use trading calendar
            next_day = date + pd.Timedelta(days=1)
            
            # Check if next_day exists in data
            all_dates = self._get_all_dates()
            available_dates = [d for d in all_dates if d > date]
            
            if available_dates:
                return min(available_dates)
            else:
                return next_day
        
        return DailyEventLoop(
            market_data=self.market_data,
            portfolio=self.portfolio,
            strategies=self.strategies,
            compute_features_fn=compute_features_fn,
            get_next_trading_day=get_next_trading_day,
            rng=self.rng
        )
    
    def _compute_results(
        self,
        split: WalkForwardSplit,
        period: str
    ) -> Dict[str, Any]:
        """Compute backtest results.
        
        Args:
            split: WalkForwardSplit used
            period: Period name
        
        Returns:
            Dictionary with results
        """
        if not self.daily_events:
            return self._empty_results()
        
        # Extract equity curve
        equity_curve = self.portfolio.equity_curve
        
        # Compute metrics
        total_return = (equity_curve[-1] / equity_curve[0] - 1) if len(equity_curve) > 1 else 0.0
        
        # Compute daily returns
        daily_returns = self.portfolio.daily_returns
        
        # Compute Sharpe ratio (annualized)
        if len(daily_returns) > 1:
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            sharpe_ratio = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Compute max drawdown
        if len(equity_curve) > 1:
            peak = equity_curve[0]
            max_dd = 0.0
            for equity in equity_curve:
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd
        else:
            max_dd = 0.0
        
        # Trade statistics
        total_trades = len(self.closed_trades)
        winning_trades = len([t for t in self.closed_trades if t.realized_pnl > 0])
        losing_trades = len([t for t in self.closed_trades if t.realized_pnl < 0])
        
        # Average R-multiple
        r_multiples = [t.compute_r_multiple() for t in self.closed_trades if t.exit_price is not None]
        avg_r_multiple = np.mean(r_multiples) if r_multiples else 0.0
        
        results = {
            'split_name': split.name,
            'period': period,
            'start_date': split.get_period_dates(period)[0],
            'end_date': split.get_period_dates(period)[1],
            'starting_equity': self.starting_equity,
            'ending_equity': equity_curve[-1] if equity_curve else self.starting_equity,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0.0,
            'avg_r_multiple': avg_r_multiple,
            'realized_pnl': self.portfolio.realized_pnl,
            'final_cash': self.portfolio.cash,
            'final_positions': len(self.portfolio.positions),
            'equity_curve': equity_curve,
            'daily_returns': daily_returns,
            'closed_trades': self.closed_trades
        }
        
        return results
    
    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results dictionary.
        
        Returns:
            Dictionary with default/empty values
        """
        return {
            'split_name': '',
            'period': '',
            'start_date': None,
            'end_date': None,
            'starting_equity': self.starting_equity,
            'ending_equity': self.starting_equity,
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_r_multiple': 0.0,
            'realized_pnl': 0.0,
            'final_cash': self.starting_equity,
            'final_positions': 0,
            'equity_curve': [self.starting_equity],
            'daily_returns': [],
            'closed_trades': []
        }
    
    def export_results(self, output_path: str) -> None:
        """Export backtest results to files.
        
        Exports:
        - equity_curve.csv
        - trade_log.csv
        - daily_metrics.csv
        
        Args:
            output_path: Directory path to write output files
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export equity curve
        if self.portfolio.equity_curve:
            equity_df = pd.DataFrame({
                'date': range(len(self.portfolio.equity_curve)),
                'equity': self.portfolio.equity_curve
            })
            equity_df.to_csv(output_dir / 'equity_curve.csv', index=False)
        
        # Export trade log
        if self.closed_trades:
            trades_data = []
            for trade in self.closed_trades:
                trades_data.append({
                    'symbol': trade.symbol,
                    'asset_class': trade.asset_class,
                    'entry_date': trade.entry_date,
                    'exit_date': trade.exit_date,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'quantity': trade.quantity,
                    'realized_pnl': trade.realized_pnl,
                    'exit_reason': trade.exit_reason.value if trade.exit_reason else None,
                    'r_multiple': trade.compute_r_multiple() if trade.exit_price else None
                })
            trades_df = pd.DataFrame(trades_data)
            trades_df.to_csv(output_dir / 'trade_log.csv', index=False)
        
        # Export daily metrics
        if self.daily_events:
            metrics_data = []
            for event in self.daily_events:
                metrics_data.append({
                    'date': event['date'],
                    'equity': event['portfolio_state'].get('equity', 0),
                    'cash': event['portfolio_state'].get('cash', 0),
                    'open_positions': event['portfolio_state'].get('open_positions', 0),
                    'realized_pnl': event['portfolio_state'].get('realized_pnl', 0),
                    'unrealized_pnl': event['portfolio_state'].get('unrealized_pnl', 0),
                    'gross_exposure': event['portfolio_state'].get('gross_exposure', 0),
                    'risk_multiplier': event['portfolio_state'].get('risk_multiplier', 1.0)
                })
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_csv(output_dir / 'daily_metrics.csv', index=False)
        
        logger.info(f"Exported results to {output_path}")


def create_backtest_engine_from_config(
    config_path: str,
    data_paths: Dict[str, str],
    seed: Optional[int] = None
) -> BacktestEngine:
    """Create backtest engine from configuration files.
    
    Args:
        config_path: Path to strategy config YAML
        data_paths: Dictionary with keys: 'equity', 'crypto', 'benchmark'
        seed: Optional random seed
    
    Returns:
        BacktestEngine instance
    """
    # This is a placeholder - in production, load config and create strategies
    # For now, return a basic engine
    raise NotImplementedError("Config-based engine creation not yet implemented")


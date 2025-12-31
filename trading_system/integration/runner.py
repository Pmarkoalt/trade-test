"""Integration runner for wiring up all components from configuration."""

from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import pandas as pd
import logging

from ..configs.run_config import RunConfig
from ..configs.strategy_config import StrategyConfig
from ..backtest.engine import BacktestEngine
from ..backtest.splits import WalkForwardSplit
from ..data.loader import load_all_data, load_universe
from ..strategies.strategy_loader import load_strategies_from_run_config
from ..models.market_data import MarketData
from ..reporting.csv_writer import CSVWriter
from ..reporting.json_writer import JSONWriter
from ..reporting.metrics import MetricsCalculator
from ..storage import ResultsDatabase
from ..validation.bootstrap import run_bootstrap_test, check_bootstrap_results
from ..validation.permutation import run_permutation_test, check_permutation_results
from ..validation.stress_tests import (
    run_slippage_stress,
    run_bear_market_test,
    run_range_market_test,
    run_flash_crash_simulation,
    check_stress_results
)
from ..validation.correlation_analysis import (
    run_correlation_stress_analysis,
    check_correlation_warnings
)
from ..validation.sensitivity import (
    ParameterSensitivityGrid,
    generate_parameter_grid_from_config,
    apply_parameters_to_strategy_config,
    apply_parameters_to_run_config,
    save_sensitivity_results,
    generate_all_heatmaps,
    HAS_PLOTLY,
    HAS_MATPLOTLIB
)
import numpy as np

logger = logging.getLogger(__name__)


class BacktestRunner:
    """Runner class for executing backtests from configuration."""
    
    def __init__(self, config: RunConfig):
        """Initialize runner with configuration.
        
        Args:
            config: RunConfig instance loaded from YAML
        """
        self.config = config
        self.market_data: Optional[MarketData] = None
        self.strategies: List = []
        self.engine: Optional[BacktestEngine] = None
    
    def load_data(self) -> MarketData:
        """Load all market data.
        
        Returns:
            MarketData object with bars and benchmarks
        """
        logger.info("Loading market data...")
        
        # Get date range
        start_date = pd.Timestamp(self.config.dataset.start_date)
        end_date = pd.Timestamp(self.config.dataset.end_date)
        
        # Determine equity universe
        equity_universe = None
        if self.config.strategies.equity and self.config.strategies.equity.enabled:
            # Load equity strategy config to get universe
            equity_strategy_config = StrategyConfig.from_yaml(
                self.config.strategies.equity.config_path
            )
            if isinstance(equity_strategy_config.universe, list):
                equity_universe = equity_strategy_config.universe
            else:
                # Load from universe file
                equity_universe = load_universe(equity_strategy_config.universe)
        
        # Determine crypto universe
        crypto_universe = None
        crypto_universe_config = None
        if self.config.strategies.crypto and self.config.strategies.crypto.enabled:
            # Load crypto strategy config to get universe
            crypto_strategy_config = StrategyConfig.from_yaml(
                self.config.strategies.crypto.config_path
            )
            
            # Check if we have universe_config for dynamic selection
            if crypto_strategy_config.universe_config is not None:
                # We'll use dynamic universe selection
                crypto_universe_config = crypto_strategy_config.universe_config.model_dump()
            elif isinstance(crypto_strategy_config.universe, list):
                # Explicit list provided
                crypto_universe = crypto_strategy_config.universe
            else:
                # Use traditional load_universe (backward compatible)
                crypto_universe = load_universe(crypto_strategy_config.universe)
        
        # Load all data
        market_data, benchmarks = load_all_data(
            equity_path=self.config.dataset.equity_path,
            crypto_path=self.config.dataset.crypto_path,
            benchmark_path=self.config.dataset.benchmark_path,
            equity_universe=equity_universe or [],
            crypto_universe=crypto_universe,
            crypto_universe_config=crypto_universe_config,
            start_date=start_date,
            end_date=end_date
        )
        
        self.market_data = market_data
        logger.info(f"Loaded data: {len(market_data.bars)} symbols")
        return market_data
    
    def load_strategies(self) -> List:
        """Load and initialize strategy objects.
        
        Returns:
            List of strategy objects
        """
        logger.info("Loading strategies...")
        
        # Use strategy loader to load strategies from config
        equity_config_path = (
            self.config.strategies.equity.config_path
            if self.config.strategies.equity and self.config.strategies.equity.enabled
            else None
        )
        crypto_config_path = (
            self.config.strategies.crypto.config_path
            if self.config.strategies.crypto and self.config.strategies.crypto.enabled
            else None
        )
        
        strategies = load_strategies_from_run_config(
            equity_config_path=equity_config_path,
            crypto_config_path=crypto_config_path
        )
        
        if not strategies:
            raise ValueError("No strategies enabled in configuration")
        
        self.strategies = strategies
        return strategies
    
    def create_engine(self) -> BacktestEngine:
        """Create backtest engine.
        
        Returns:
            BacktestEngine instance
        """
        if self.market_data is None:
            raise ValueError("Market data not loaded. Call load_data() first.")
        if not self.strategies:
            raise ValueError("Strategies not loaded. Call load_strategies() first.")
        
        logger.info("Creating backtest engine...")
        engine = BacktestEngine(
            market_data=self.market_data,
            strategies=self.strategies,
            starting_equity=self.config.portfolio.starting_equity,
            seed=self.config.random_seed
        )
        
        self.engine = engine
        return engine
    
    def create_split(self) -> WalkForwardSplit:
        """Create walk-forward split from config.
        
        Returns:
            WalkForwardSplit instance
        """
        return WalkForwardSplit(
            name="config_split",
            train_start=pd.Timestamp(self.config.splits.train_start),
            train_end=pd.Timestamp(self.config.splits.train_end),
            validation_start=pd.Timestamp(self.config.splits.validation_start),
            validation_end=pd.Timestamp(self.config.splits.validation_end),
            holdout_start=pd.Timestamp(self.config.splits.holdout_start),
            holdout_end=pd.Timestamp(self.config.splits.holdout_end)
        )
    
    def run_backtest(
        self,
        period: str = "train",
        slippage_multiplier: float = 1.0,
        crash_dates: Optional[Any] = None,
        date_filter: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Run backtest for specified period.
        
        Args:
            period: One of "train", "validation", "holdout", or "train+validation" for combined
            slippage_multiplier: Multiplier for base slippage (for stress tests)
            crash_dates: List of dates or 'auto' to generate one per quarter
            date_filter: 'bear', 'range', or a callable function to filter trading dates
        
        Returns:
            Dictionary with backtest results
        """
        if self.engine is None:
            raise ValueError("Engine not created. Call create_engine() first.")
        
        split = self.create_split()
        
        # Handle crash_dates='auto' - generate one crash per quarter
        actual_crash_dates = None
        if crash_dates == 'auto':
            start_date, end_date = split.get_period_dates("train")
            val_start, val_end = split.get_period_dates("validation")
            # Use combined train+validation period for crash generation
            combined_start = start_date
            combined_end = val_end
            
            # Generate one crash date per quarter
            actual_crash_dates = self._generate_quarterly_crash_dates(combined_start, combined_end)
        elif isinstance(crash_dates, list):
            actual_crash_dates = crash_dates
        
        # Handle date_filter string options
        actual_date_filter = None
        if date_filter == 'bear':
            start_date, end_date = split.get_period_dates("train")
            val_start, val_end = split.get_period_dates("validation")
            combined_start = min(start_date, val_start)
            combined_end = max(end_date, val_end)
            # Try SPY first, fall back to BTC
            benchmark_symbol = "SPY" if "SPY" in (self.market_data.benchmarks or {}) else "BTC"
            actual_date_filter = self._create_bear_market_filter(combined_start, combined_end, benchmark_symbol)
        elif date_filter == 'range':
            start_date, end_date = split.get_period_dates("train")
            val_start, val_end = split.get_period_dates("validation")
            combined_start = min(start_date, val_start)
            combined_end = max(end_date, val_end)
            benchmark_symbol = "SPY" if "SPY" in (self.market_data.benchmarks or {}) else "BTC"
            actual_date_filter = self._create_range_market_filter(combined_start, combined_end, benchmark_symbol)
        elif callable(date_filter):
            actual_date_filter = date_filter
        
        # Handle combined period
        if period == "train+validation":
            # Run train and validation separately, then combine
            train_results = self.run_backtest("train", slippage_multiplier, actual_crash_dates, actual_date_filter)
            val_results = self.run_backtest("validation", slippage_multiplier, actual_crash_dates, actual_date_filter)
            
            # Combine results
            return self._combine_results([train_results, val_results])
        
        # If parameters changed, recreate engine with new parameters
        if (slippage_multiplier != 1.0 or actual_crash_dates is not None or actual_date_filter is not None):
            # Create new engine with stress test parameters
            engine = BacktestEngine(
                market_data=self.market_data,
                strategies=self.strategies,
                starting_equity=self.config.portfolio.starting_equity,
                seed=self.config.random_seed,
                slippage_multiplier=slippage_multiplier,
                crash_dates=actual_crash_dates
            )
            old_engine = self.engine
            self.engine = engine
            
            # If date filter is provided, modify the run to filter dates
            if actual_date_filter is not None:
                results = self._run_backtest_with_date_filter(split, period, actual_date_filter)
            else:
                results = self.engine.run(split=split, period=period)
            
            # Restore original engine
            self.engine = old_engine
        else:
            results = self.engine.run(split=split, period=period)
        
        return results
    
    def _run_backtest_with_date_filter(
        self,
        split: WalkForwardSplit,
        period: str,
        date_filter: Callable[[pd.Timestamp], bool]
    ) -> Dict[str, Any]:
        """Run backtest with date filtering.
        
        This modifies the engine's run method to filter dates.
        We'll override the split's get_period_dates to return filtered dates.
        
        Args:
            split: WalkForwardSplit configuration
            period: Period name
            date_filter: Function to filter dates
        
        Returns:
            Dictionary with backtest results
        """
        # Get original date range
        start_date, end_date = split.get_period_dates(period)
        
        # Get all available dates from market data
        all_dates = self.engine._get_all_dates()
        trading_dates = [d for d in all_dates if start_date <= d <= end_date]
        
        # Filter dates
        filtered_dates = [d for d in trading_dates if date_filter(d)]
        
        if not filtered_dates:
            logger.warning(f"No dates passed filter for {period} period")
            return self.engine._empty_results()
        
        # Create a modified split that only includes filtered dates
        # We'll need to manually run the backtest with filtered dates
        # For now, let's modify the engine to accept a custom date list
        # Actually, we can create a custom split class or modify dates
        
        # Simple approach: modify the split temporarily
        filtered_start = min(filtered_dates)
        filtered_end = max(filtered_dates)
        
        # Create a temporary split with filtered dates
        from ..backtest.splits import WalkForwardSplit
        filtered_split = WalkForwardSplit(
            name=f"{split.name}_filtered",
            train_start=filtered_start if period == "train" else split.train_start,
            train_end=filtered_end if period == "train" else split.train_end,
            validation_start=filtered_start if period == "validation" else split.validation_start,
            validation_end=filtered_end if period == "validation" else split.validation_end,
            holdout_start=filtered_start if period == "holdout" else split.holdout_start,
            holdout_end=filtered_end if period == "holdout" else split.holdout_end
        )
        
        # However, the engine.run() will still get all dates in the range
        # We need to modify the engine to filter dates during processing
        # For now, let's patch the _get_all_dates method temporarily
        
        # Better approach: override the run method's date filtering
        # Actually, we can modify the engine to accept a date filter function
        # But that requires more changes. Let's use a simpler approach:
        # Manually filter dates before calling engine.run()
        
        # Store original method and temporarily override
        original_get_all_dates = self.engine._get_all_dates
        
        def filtered_get_all_dates():
            dates = original_get_all_dates()
            return [d for d in dates if date_filter(d)]
        
        # Temporarily override
        self.engine._get_all_dates = filtered_get_all_dates
        
        try:
            results = self.engine.run(split=filtered_split, period=period)
        finally:
            # Restore original method
            self.engine._get_all_dates = original_get_all_dates
        
        return results
    
    def _get_benchmark_monthly_returns(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        benchmark_symbol: str = "SPY"
    ) -> Dict[str, float]:
        """Get benchmark monthly returns for date filtering.
        
        Args:
            start_date: Start date
            end_date: End date
            benchmark_symbol: Benchmark symbol ("SPY" or "BTC")
        
        Returns:
            Dictionary mapping month strings (YYYY-MM) to monthly returns
        """
        if self.market_data is None:
            return {}
        
        if benchmark_symbol not in self.market_data.benchmarks:
            logger.warning(f"Benchmark {benchmark_symbol} not found in market data")
            return {}
        
        benchmark_df = self.market_data.benchmarks[benchmark_symbol]
        
        # Filter to date range
        date_range = benchmark_df[(benchmark_df.index >= start_date) & (benchmark_df.index <= end_date)]
        
        if len(date_range) == 0:
            return {}
        
        # Resample to monthly and compute returns
        monthly_returns = {}
        
        # Group by year-month
        for year_month, group in date_range.groupby(date_range.index.to_period("M")):
            if len(group) < 2:
                continue
            
            month_start_price = group.iloc[0]['close']
            month_end_price = group.iloc[-1]['close']
            
            if month_start_price > 0:
                monthly_return = (month_end_price / month_start_price) - 1.0
                monthly_returns[str(year_month)] = monthly_return
        
        return monthly_returns
    
    def _create_bear_market_filter(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        benchmark_symbol: str = "SPY"
    ) -> Callable[[pd.Timestamp], bool]:
        """Create date filter for bear market months (benchmark return < -5%).
        
        Args:
            start_date: Start date
            end_date: End date
            benchmark_symbol: Benchmark symbol
        
        Returns:
            Function that returns True for dates in bear market months
        """
        monthly_returns = self._get_benchmark_monthly_returns(start_date, end_date, benchmark_symbol)
        
        # Identify bear market months
        bear_months = set()
        for month_str, return_val in monthly_returns.items():
            if return_val < -0.05:  # < -5%
                bear_months.add(month_str)
        
        def date_filter(date: pd.Timestamp) -> bool:
            month_str = date.to_period("M").strftime("%Y-%m")
            return month_str in bear_months
        
        return date_filter
    
    def _create_range_market_filter(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        benchmark_symbol: str = "SPY"
    ) -> Callable[[pd.Timestamp], bool]:
        """Create date filter for range market months (benchmark return between -2% and +2%).
        
        Args:
            start_date: Start date
            end_date: End date
            benchmark_symbol: Benchmark symbol
        
        Returns:
            Function that returns True for dates in range market months
        """
        monthly_returns = self._get_benchmark_monthly_returns(start_date, end_date, benchmark_symbol)
        
        # Identify range market months
        range_months = set()
        for month_str, return_val in monthly_returns.items():
            if -0.02 <= return_val <= 0.02:  # Between -2% and +2%
                range_months.add(month_str)
        
        def date_filter(date: pd.Timestamp) -> bool:
            month_str = date.to_period("M").strftime("%Y-%m")
            return month_str in range_months
        
        return date_filter
    
    def _generate_quarterly_crash_dates(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> List[pd.Timestamp]:
        """Generate crash dates - one per quarter.
        
        Args:
            start_date: Start date
            end_date: End date
        
        Returns:
            List of crash dates (one random date per quarter)
        """
        if self.market_data is None:
            return []
        
        crash_dates = []
        current = start_date
        
        # Get all available dates from market data
        all_dates = set()
        for symbol, bars_df in self.market_data.bars.items():
            all_dates.update(bars_df.index)
        all_dates = sorted(list(all_dates))
        
        while current < end_date:
            # Get quarter end
            quarter_end = current + pd.offsets.QuarterEnd()
            if quarter_end > end_date:
                quarter_end = end_date
            
            # Get all available dates in this quarter
            quarter_dates = [d for d in all_dates if current <= d <= quarter_end]
            
            if quarter_dates:
                # Pick a random date in the quarter
                # Use numpy random if available, otherwise Python random
                try:
                    import numpy as np
                    rng = np.random.default_rng(self.config.random_seed if hasattr(self.config, 'random_seed') else None)
                    crash_date = rng.choice(quarter_dates)
                except:
                    import random
                    if hasattr(self.config, 'random_seed'):
                        random.seed(self.config.random_seed)
                    crash_date = random.choice(quarter_dates)
                crash_dates.append(crash_date)
            
            # Move to next quarter
            current = quarter_end + pd.Timedelta(days=1)
        
        return crash_dates
    
    def _combine_results(self, results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from multiple periods.
        
        Args:
            results_list: List of result dictionaries
        
        Returns:
            Combined results dictionary
        """
        if not results_list:
            return {}
        
        # Combine closed trades
        all_closed_trades = []
        for results in results_list:
            all_closed_trades.extend(results.get('closed_trades', []))
        
        # Combine equity curves
        all_equity_curves = []
        all_daily_returns = []
        for results in results_list:
            all_equity_curves.extend(results.get('equity_curve', []))
            all_daily_returns.extend(results.get('daily_returns', []))
        
        # Compute combined metrics
        if len(all_equity_curves) > 1:
            total_return = (all_equity_curves[-1] / all_equity_curves[0] - 1) if all_equity_curves[0] > 0 else 0.0
            
            # Compute Sharpe
            if len(all_daily_returns) > 1:
                mean_return = np.mean(all_daily_returns)
                std_return = np.std(all_daily_returns)
                sharpe_ratio = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0.0
            else:
                sharpe_ratio = 0.0
            
            # Compute max drawdown
            peak = all_equity_curves[0]
            max_dd = 0.0
            for equity in all_equity_curves:
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak if peak > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd
        else:
            total_return = 0.0
            sharpe_ratio = 0.0
            max_dd = 0.0
        
        # Combine trade stats
        total_trades = len(all_closed_trades)
        winning_trades = len([t for t in all_closed_trades if t.realized_pnl > 0])
        losing_trades = len([t for t in all_closed_trades if t.realized_pnl < 0])
        
        # Average R-multiple
        r_multiples = [t.compute_r_multiple() for t in all_closed_trades if t.exit_price is not None]
        avg_r_multiple = np.mean(r_multiples) if r_multiples else 0.0
        
        # Compute Calmar ratio
        calmar_ratio = abs(total_return / max_dd) if max_dd > 0 else 0.0
        
        return {
            'period': 'combined',
            'starting_equity': results_list[0].get('starting_equity', 100000.0),
            'ending_equity': all_equity_curves[-1] if all_equity_curves else results_list[0].get('starting_equity', 100000.0),
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_dd,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0.0,
            'avg_r_multiple': avg_r_multiple,
            'equity_curve': all_equity_curves,
            'daily_returns': all_daily_returns,
            'closed_trades': all_closed_trades
        }
    
    def _extract_benchmark_returns(
        self,
        dates: List[pd.Timestamp],
        benchmark_symbol: str = "SPY"
    ) -> Optional[List[float]]:
        """Extract benchmark daily returns for given dates.
        
        Args:
            dates: List of dates to extract returns for
            benchmark_symbol: Benchmark symbol ("SPY" or "BTC")
        
        Returns:
            List of daily returns (as decimals) or None if benchmark not available
        """
        if self.market_data is None:
            return None
        
        if benchmark_symbol not in self.market_data.benchmarks:
            logger.warning(f"Benchmark {benchmark_symbol} not found in market data")
            return None
        
        benchmark_df = self.market_data.benchmarks[benchmark_symbol]
        
        # Extract close prices for the dates
        returns = []
        prev_close = None
        
        for date in dates:
            if date not in benchmark_df.index:
                # If date not available, use previous return or 0
                if returns:
                    returns.append(returns[-1] if returns else 0.0)
                else:
                    returns.append(0.0)
                continue
            
            close = benchmark_df.loc[date, 'close']
            
            if prev_close is not None and prev_close > 0:
                daily_return = (close / prev_close) - 1.0
                returns.append(daily_return)
            else:
                # First day has no return
                returns.append(0.0)
            
            prev_close = close
        
        # Remove first return (which is 0.0) to match daily_returns length
        # daily_returns has one less element than equity_curve
        if len(returns) > 0:
            returns = returns[1:]
        
        return returns if returns else None
    
    def save_results(self, results: Dict[str, Any], period: str = "train") -> Path:
        """Save backtest results to output directory.
        
        Args:
            results: Results dictionary from engine.run()
            period: Period name (train/validation/holdout)
        
        Returns:
            Path to output directory
        """
        output_dir = self.config.get_output_dir() / period
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving results to {output_dir}")
        
        # Create writers
        csv_writer = CSVWriter(str(output_dir))
        json_writer = JSONWriter(str(output_dir))
        
        # Get portfolio data
        portfolio = self.engine.portfolio
        daily_events = self.engine.daily_events
        closed_trades = self.engine.closed_trades
        
        # Extract dates and equity curve
        dates = [event['date'] for event in daily_events]
        # Extract from portfolio_state (new structure) or fallback to top-level (legacy)
        equity_curve = []
        cash_history = []
        positions_count = []
        exposure_history = []
        daily_returns = []
        
        for event in daily_events:
            portfolio_state = event.get('portfolio_state', {})
            # Use portfolio_state if available, otherwise fallback to top-level (for backward compatibility)
            equity_curve.append(
                portfolio_state.get('equity', event.get('equity', self.config.portfolio.starting_equity))
            )
            cash_history.append(portfolio_state.get('cash', event.get('cash', 0)))
            positions_count.append(portfolio_state.get('open_positions', event.get('positions_count', 0)))
            exposure_history.append(portfolio_state.get('gross_exposure', event.get('exposure', 0)))
            # Daily returns are computed from equity curve, not stored directly
            daily_returns.append(event.get('return', 0))
        
        # If daily_returns are all zeros, compute from equity curve
        if all(r == 0 for r in daily_returns) and len(equity_curve) > 1:
            daily_returns = []
            for i in range(1, len(equity_curve)):
                if equity_curve[i-1] > 0:
                    daily_returns.append((equity_curve[i] / equity_curve[i-1]) - 1.0)
                else:
                    daily_returns.append(0.0)
            # Pad first day with 0
            daily_returns.insert(0, 0.0)
        
        # Write CSV files
        csv_writer.write_equity_curve(
            equity_curve=equity_curve,
            dates=dates,
            cash_history=cash_history,
            positions_count_history=positions_count,
            exposure_history=exposure_history
        )
        
        csv_writer.write_trade_log(closed_trades)
        
        # Write weekly summary
        csv_writer.write_weekly_summary(
            equity_curve=equity_curve,
            dates=dates,
            daily_returns=daily_returns,
            closed_trades=closed_trades
        )
        
        # Extract benchmark returns from market data
        # Try SPY first (for equity), fall back to BTC (for crypto)
        benchmark_returns = self._extract_benchmark_returns(dates, "SPY")
        if benchmark_returns is None:
            benchmark_returns = self._extract_benchmark_returns(dates, "BTC")
        
        # Write JSON reports
        json_writer.write_monthly_report(
            equity_curve=equity_curve,
            dates=dates,
            daily_returns=daily_returns,
            closed_trades=closed_trades,
            benchmark_returns=benchmark_returns
        )
        
        # Store results in database
        try:
            db = ResultsDatabase()
            
            # Prepare results dictionary for database storage
            # Get split name from results if available
            split_name = results.get('split_name')
            
            # Determine strategy name from config
            strategy_name = None
            if self.config.strategies.equity and self.config.strategies.equity.enabled:
                strategy_name = "equity"
            elif self.config.strategies.crypto and self.config.strategies.crypto.enabled:
                strategy_name = "crypto"
            
            # Get config path (if available)
            config_path = getattr(self.config, '_config_path', None)
            
            # Prepare results dict with all required data
            db_results = {
                'start_date': dates[0] if dates else None,
                'end_date': dates[-1] if dates else None,
                'starting_equity': self.config.portfolio.starting_equity,
                'ending_equity': equity_curve[-1] if equity_curve else self.config.portfolio.starting_equity,
                'equity_curve': equity_curve,
                'daily_returns': daily_returns,
                'closed_trades': closed_trades,
                'daily_events': daily_events,  # Include daily_events for equity curve storage
                'dates': dates,
                'benchmark_returns': benchmark_returns,
            }
            
            # Add metrics from results if available, otherwise compute basic ones
            db_results['sharpe_ratio'] = results.get('sharpe_ratio')
            db_results['max_drawdown'] = results.get('max_drawdown')
            db_results['total_return'] = results.get('total_return')
            db_results['total_trades'] = results.get('total_trades', len(closed_trades))
            db_results['winning_trades'] = results.get('winning_trades', len([t for t in closed_trades if t.realized_pnl > 0]))
            db_results['losing_trades'] = results.get('losing_trades', len([t for t in closed_trades if t.realized_pnl < 0]))
            db_results['win_rate'] = results.get('win_rate')
            db_results['avg_r_multiple'] = results.get('avg_r_multiple')
            db_results['realized_pnl'] = results.get('realized_pnl', sum(t.realized_pnl for t in closed_trades))
            
            # Get portfolio state for final values
            if daily_events:
                last_event = daily_events[-1]
                portfolio_state = last_event.get('portfolio_state', {})
                db_results['final_cash'] = portfolio_state.get('cash')
                db_results['final_positions'] = portfolio_state.get('open_positions', 0)
            else:
                db_results['final_cash'] = results.get('final_cash')
                db_results['final_positions'] = results.get('final_positions', 0)
            
            run_id = db.store_results(
                results=db_results,
                config_path=config_path,
                strategy_name=strategy_name,
                period=period,
                split_name=split_name
            )
            logger.info(f"Results stored in database with run_id={run_id}")
        except Exception as e:
            # Don't fail if database storage fails - just log a warning
            logger.warning(f"Failed to store results in database: {e}", exc_info=True)
        
        logger.info(f"Results saved to {output_dir}")
        return output_dir
    
    
    def initialize(self):
        """Initialize runner by loading data and creating engine."""
        self.load_data()
        self.load_strategies()
        self.create_engine()


def run_backtest(config_path: str, period: str = "train") -> Dict[str, Any]:
    """Convenience function to run a backtest.
    
    Args:
        config_path: Path to run_config.yaml
        period: Period to run ("train", "validation", "holdout")
    
    Returns:
        Results dictionary
    """
    config = RunConfig.from_yaml(config_path)
    runner = BacktestRunner(config)
    runner.initialize()
    
    results = runner.run_backtest(period=period)
    output_dir = runner.save_results(results, period=period)
    
    logger.info(f"Backtest completed. Results saved to {output_dir}")
    return results


def run_validation(config_path: str) -> Dict[str, Any]:
    """Run validation suite on train+validation data.
    
    Args:
        config_path: Path to run_config.yaml
    
    Returns:
        Validation results dictionary
    """
    logger.info("Starting validation suite...")
    
    config = RunConfig.from_yaml(config_path)
    runner = BacktestRunner(config)
    runner.initialize()
    
    split = runner.create_split()
    
    # Run backtest on train+validation combined
    # We'll run train and validation separately, then combine results
    logger.info("Running train period backtest...")
    train_results = runner.run_backtest(period="train")
    train_daily_events = runner.engine.daily_events.copy() if hasattr(runner.engine, 'daily_events') else []
    
    logger.info("Running validation period backtest...")
    validation_results = runner.run_backtest(period="validation")
    validation_daily_events = runner.engine.daily_events.copy() if hasattr(runner.engine, 'daily_events') else []
    
    # Combine closed trades from both periods
    all_closed_trades = train_results.get('closed_trades', []) + validation_results.get('closed_trades', [])
    all_daily_events = train_daily_events + validation_daily_events
    
    if not all_closed_trades:
        logger.warning("No closed trades found for validation suite")
        return {
            'status': 'failed',
            'reason': 'No closed trades available',
            'results': {}
        }
    
    # Extract R-multiples from closed trades
    r_multiples = []
    trades_for_permutation = []
    
    for trade in all_closed_trades:
        if trade.exit_price is not None:
            r_mult = trade.compute_r_multiple()
            r_multiples.append(r_mult)
            
            trades_for_permutation.append({
                'entry_date': trade.entry_date,
                'exit_date': trade.exit_date,
                'symbol': trade.symbol,
                'r_multiple': r_mult
            })
    
    if not r_multiples:
        logger.warning("No valid R-multiples found for validation suite")
        return {
            'status': 'failed',
            'reason': 'No valid R-multiples available',
            'results': {}
        }
    
    validation_results_dict = {}
    warnings = []
    rejections = []
    
    # Get validation config
    validation_config = config.validation if hasattr(config, 'validation') else None
    statistical_config = validation_config.statistical if validation_config and validation_config.statistical else None
    
    n_bootstrap = statistical_config.bootstrap_iterations if statistical_config else 1000
    n_permutation = statistical_config.permutation_iterations if statistical_config else 1000
    random_seed = config.random_seed
    
    # 1. Bootstrap test
    logger.info(f"Running bootstrap test ({n_bootstrap} iterations)...")
    try:
        bootstrap_results = run_bootstrap_test(
            r_multiples=r_multiples,
            n_iterations=n_bootstrap,
            random_seed=random_seed
        )
        bootstrap_passed, bootstrap_warnings = check_bootstrap_results(bootstrap_results)
        validation_results_dict['bootstrap'] = bootstrap_results
        validation_results_dict['bootstrap']['passed'] = bootstrap_passed
        if not bootstrap_passed:
            rejections.append("Bootstrap test failed")
        warnings.extend(bootstrap_warnings)
    except Exception as e:
        logger.error(f"Bootstrap test failed: {e}", exc_info=True)
        validation_results_dict['bootstrap'] = {'error': str(e)}
        rejections.append(f"Bootstrap test error: {e}")
    
    # 2. Permutation test
    logger.info(f"Running permutation test ({n_permutation} iterations)...")
    try:
        # Helper function to compute Sharpe from trades
        def compute_sharpe_from_trades(trades: List[Dict]) -> float:
            """Compute Sharpe ratio from list of trades with R-multiples."""
            if not trades:
                return 0.0
            
            r_mults = [t.get('r_multiple', 0.0) for t in trades]
            if not r_mults:
                return 0.0
            
            # Use same logic as bootstrap test
            mean_r = np.mean(r_mults)
            std_r = np.std(r_mults)
            
            if std_r == 0:
                return 0.0
            
            # Annualize (assume ~15 trades per year)
            trades_per_year = 15.0
            sharpe = mean_r / std_r * np.sqrt(trades_per_year)
            return sharpe
        
        # Get period dates (train+validation combined)
        train_start, train_end = split.get_period_dates("train")
        val_start, val_end = split.get_period_dates("validation")
        period_start = train_start
        period_end = val_end
        
        permutation_results = run_permutation_test(
            actual_trades=trades_for_permutation,
            period=(period_start, period_end),
            compute_sharpe_func=compute_sharpe_from_trades,
            n_iterations=n_permutation,
            random_seed=random_seed
        )
        permutation_passed, permutation_warnings = check_permutation_results(permutation_results)
        validation_results_dict['permutation'] = permutation_results
        if not permutation_passed:
            rejections.append("Permutation test failed")
        warnings.extend(permutation_warnings)
    except Exception as e:
        logger.error(f"Permutation test failed: {e}", exc_info=True)
        validation_results_dict['permutation'] = {'error': str(e)}
        rejections.append(f"Permutation test error: {e}")
    
    # 3. Stress tests
    logger.info("Running stress tests...")
    try:
        stress_config = validation_config.stress_tests if validation_config and validation_config.stress_tests else None
        
        # Create a wrapper function for stress tests that calls runner.run_backtest
        def run_stress_backtest(
            slippage_multiplier: float = 1.0,
            crash_dates: Optional[Any] = None,
            date_filter: Optional[Any] = None
        ) -> Dict[str, Any]:
            """Wrapper function for stress tests."""
            return runner.run_backtest(
                period="train+validation",
                slippage_multiplier=slippage_multiplier,
                crash_dates=crash_dates,
                date_filter=date_filter
            )
        
        # Initialize stress test suite
        from ..validation.stress_tests import StressTestSuite
        stress_suite = StressTestSuite(
            run_backtest_func=run_stress_backtest,
            random_seed=random_seed
        )
        
        # Run stress tests
        stress_results = {}
        
        # Slippage stress tests (2x, 3x)
        if stress_config is None or stress_config.slippage_multipliers:
            multipliers = stress_config.slippage_multipliers if stress_config else [2.0, 3.0]
            logger.info(f"Running slippage stress tests (multipliers: {multipliers})...")
            for mult in multipliers:
                if mult != 1.0:  # Skip 1x (baseline)
                    test_name = f'slippage_{int(mult)}x'
                    logger.info(f"Running {test_name}...")
                    stress_results[test_name] = stress_suite.run_slippage_stress(multiplier=mult)
        
        # Bear market test
        if stress_config is None or stress_config.bear_market_test:
            logger.info("Running bear market test...")
            stress_results['bear_market'] = stress_suite.run_bear_market_test()
        
        # Range market test
        if stress_config is None or stress_config.range_market_test:
            logger.info("Running range market test...")
            stress_results['range_market'] = stress_suite.run_range_market_test()
        
        # Flash crash simulation
        if stress_config is None or stress_config.flash_crash_test:
            logger.info("Running flash crash simulation...")
            stress_results['flash_crash'] = stress_suite.run_flash_crash_simulation()
        
        # Check stress test results
        stress_passed, stress_warnings = check_stress_results(stress_results)
        validation_results_dict['stress_tests'] = stress_results
        validation_results_dict['stress_tests']['passed'] = stress_passed
        if not stress_passed:
            rejections.append("Stress tests failed")
        warnings.extend(stress_warnings)
        
    except Exception as e:
        logger.error(f"Stress tests failed: {e}", exc_info=True)
        validation_results_dict['stress_tests'] = {'error': str(e)}
        rejections.append(f"Stress tests error: {e}")
    
    # 4. Correlation analysis
    logger.info("Running correlation stress analysis...")
    try:
        # Build portfolio history from daily events
        portfolio_history = {}
        dates_with_positions = 0
        
        for event in all_daily_events:
            date = event.get('date')
            if not date:
                continue
            
            # Ensure date is pd.Timestamp (should already be, but convert if needed)
            if not isinstance(date, pd.Timestamp):
                try:
                    date = pd.Timestamp(date)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid date in event: {date}, skipping")
                    continue
            
            # Extract positions from portfolio state if available
            portfolio_state = event.get('portfolio_state', {})
            positions = portfolio_state.get('positions', {})
            
            # Only include dates with at least one position (need positions for correlation)
            if positions:
                dates_with_positions += 1
            
            portfolio_history[date] = {
                'equity': portfolio_state.get('equity', event.get('equity', runner.config.portfolio.starting_equity)),
                'positions': positions
            }
        
        # Build returns data from market data (as pd.Series with date index)
        returns_data = {}
        if runner.market_data:
            for symbol, bars_df in runner.market_data.bars.items():
                if 'close' in bars_df.columns and len(bars_df) > 1:
                    # Compute returns as pd.Series with date index
                    returns = bars_df['close'].pct_change().dropna()
                    if len(returns) > 0:
                        returns_data[symbol] = returns
        
        # Check if we have sufficient data
        if not portfolio_history:
            logger.warning("No portfolio history available for correlation analysis")
            validation_results_dict['correlation'] = {
                'note': 'No portfolio history available (no daily events recorded)',
                'error': 'insufficient_data'
            }
        elif dates_with_positions < 2:
            logger.warning(f"Only {dates_with_positions} dates with positions found (need at least 2 for correlation)")
            validation_results_dict['correlation'] = {
                'note': f'Insufficient position data: only {dates_with_positions} dates with positions (need at least 2)',
                'error': 'insufficient_positions'
            }
        elif not returns_data:
            logger.warning("No returns data available for correlation analysis")
            validation_results_dict['correlation'] = {
                'note': 'No returns data available (market data not loaded or insufficient)',
                'error': 'insufficient_returns'
            }
        elif len(returns_data) < 2:
            logger.warning(f"Only {len(returns_data)} symbols with returns data (need at least 2 for correlation)")
            validation_results_dict['correlation'] = {
                'note': f'Insufficient returns data: only {len(returns_data)} symbols available (need at least 2)',
                'error': 'insufficient_returns_symbols'
            }
        else:
            # We have sufficient data - run analysis
            correlation_results = run_correlation_stress_analysis(
                portfolio_history=portfolio_history,
                returns_data=returns_data
            )
            correlation_passed, correlation_warnings = check_correlation_warnings(correlation_results)
            validation_results_dict['correlation'] = correlation_results
            if not correlation_passed:
                rejections.append("Correlation analysis failed")
            warnings.extend(correlation_warnings)
    except Exception as e:
        logger.error(f"Correlation analysis failed: {e}", exc_info=True)
        validation_results_dict['correlation'] = {
            'error': str(e),
            'error_type': type(e).__name__
        }
    
    # Summary
    overall_passed = len(rejections) == 0
    
    summary = {
        'status': 'passed' if overall_passed else 'failed',
        'rejections': rejections,
        'warnings': warnings,
        'total_trades': len(all_closed_trades),
        'r_multiples_count': len(r_multiples),
        'avg_r_multiple': np.mean(r_multiples) if r_multiples else 0.0,
        'results': validation_results_dict
    }
    
    logger.info(f"Validation suite completed: {'PASSED' if overall_passed else 'FAILED'}")
    if rejections:
        logger.warning(f"Rejections: {', '.join(rejections)}")
    if warnings:
        logger.warning(f"Warnings: {len(warnings)} warnings generated")
    
    return summary


def run_holdout(config_path: str) -> Dict[str, Any]:
    """Run holdout evaluation.
    
    Args:
        config_path: Path to run_config.yaml
    
    Returns:
        Holdout results dictionary
    """
    config = RunConfig.from_yaml(config_path)
    runner = BacktestRunner(config)
    runner.initialize()
    
    results = runner.run_backtest(period="holdout")
    output_dir = runner.save_results(results, period="holdout")
    
    logger.info(f"Holdout evaluation completed. Results saved to {output_dir}")
    return results


def run_sensitivity_analysis(
    config_path: str,
    period: str = "train",
    metric_name: str = "sharpe_ratio",
    asset_class: Optional[str] = None
) -> Dict[str, Any]:
    """Run parameter sensitivity grid search.
    
    Args:
        config_path: Path to run_config.yaml
        period: Period to run backtests on ("train", "validation", "holdout")
        metric_name: Metric to optimize (default: "sharpe_ratio")
        asset_class: "equity" or "crypto" - if None, runs for all enabled strategies
    
    Returns:
        Sensitivity analysis results dictionary
    """
    logger.info("Starting parameter sensitivity analysis...")
    
    config = RunConfig.from_yaml(config_path)
    
    if not config.validation or not config.validation.sensitivity:
        raise ValueError("Sensitivity analysis not configured in run_config.yaml")
    
    if not config.validation.sensitivity.enabled:
        logger.warning("Sensitivity analysis is disabled in config")
        return {'status': 'disabled'}
    
    sensitivity_config = config.validation.sensitivity
    
    # Determine which strategies to analyze
    strategies_to_analyze = []
    if asset_class is None:
        # Analyze all enabled strategies
        if config.strategies.equity and config.strategies.equity.enabled:
            strategies_to_analyze.append(("equity", config.strategies.equity.config_path))
        if config.strategies.crypto and config.strategies.crypto.enabled:
            strategies_to_analyze.append(("crypto", config.strategies.crypto.config_path))
    else:
        # Analyze specific asset class
        if asset_class == "equity" and config.strategies.equity and config.strategies.equity.enabled:
            strategies_to_analyze.append(("equity", config.strategies.equity.config_path))
        elif asset_class == "crypto" and config.strategies.crypto and config.strategies.crypto.enabled:
            strategies_to_analyze.append(("crypto", config.strategies.crypto.config_path))
    
    if not strategies_to_analyze:
        raise ValueError("No enabled strategies found for sensitivity analysis")
    
    all_results = {}
    
    for asset_class_name, strategy_config_path in strategies_to_analyze:
        logger.info(f"Running sensitivity analysis for {asset_class_name} strategy...")
        
        # Generate parameter grid
        parameter_ranges = generate_parameter_grid_from_config(
            sensitivity_config,
            asset_class=asset_class_name
        )
        
        if not parameter_ranges:
            logger.warning(f"No parameter ranges defined for {asset_class_name}")
            continue
        
        logger.info(f"Parameter grid: {parameter_ranges}")
        logger.info(f"Total combinations: {np.prod([len(v) for v in parameter_ranges.values()])}")
        
        # Create metric function that runs backtest with modified parameters
        def metric_func(params: Dict[str, Any]) -> float:
            """Run backtest with given parameters and return metric."""
            try:
                # Create modified configs by deep copying via dict
                modified_run_config = RunConfig(**config.model_dump())
                modified_strategy_config = StrategyConfig.from_yaml(strategy_config_path)
                
                # Separate strategy params from portfolio params
                strategy_params = {}
                portfolio_params = {}
                
                for param_path, value in params.items():
                    if param_path.startswith('volatility_scaling.'):
                        portfolio_params[param_path] = value
                    else:
                        strategy_params[param_path] = value
                
                # Apply strategy parameters
                if strategy_params:
                    modified_strategy_config = apply_parameters_to_strategy_config(
                        modified_strategy_config,
                        strategy_params
                    )
                
                # Apply portfolio parameters
                if portfolio_params:
                    modified_run_config = apply_parameters_to_run_config(
                        modified_run_config,
                        portfolio_params
                    )
                
                # Update strategy reference in run config
                if asset_class_name == "equity":
                    # Temporarily save modified strategy config and use it
                    temp_config_path = Path(strategy_config_path).parent / f"temp_{asset_class_name}_sensitivity.yaml"
                    modified_strategy_config.to_yaml(str(temp_config_path))
                    modified_run_config.strategies.equity.config_path = str(temp_config_path)
                elif asset_class_name == "crypto":
                    temp_config_path = Path(strategy_config_path).parent / f"temp_{asset_class_name}_sensitivity.yaml"
                    modified_strategy_config.to_yaml(str(temp_config_path))
                    modified_run_config.strategies.crypto.config_path = str(temp_config_path)
                
                # Run backtest
                runner = BacktestRunner(modified_run_config)
                runner.initialize()
                results = runner.run_backtest(period=period)
                
                # Calculate metric
                portfolio = runner.engine.portfolio
                daily_events = runner.engine.daily_events
                closed_trades = runner.engine.closed_trades
                
                # Extract data
                dates = [event['date'] for event in daily_events]
                # Extract equity from portfolio_state if available, otherwise from top-level
                equity_curve = []
                daily_returns = []
                for event in daily_events:
                    portfolio_state = event.get('portfolio_state', {})
                    equity = portfolio_state.get('equity', event.get('equity', config.portfolio.starting_equity))
                    equity_curve.append(equity)
                    daily_returns.append(event.get('return', 0))
                
                # If daily_returns are all zeros, compute from equity curve
                if all(r == 0 for r in daily_returns) and len(equity_curve) > 1:
                    daily_returns = []
                    for i in range(1, len(equity_curve)):
                        if equity_curve[i-1] > 0:
                            daily_returns.append((equity_curve[i] / equity_curve[i-1]) - 1.0)
                        else:
                            daily_returns.append(0.0)
                    # Pad first day with 0
                    daily_returns.insert(0, 0.0)
                
                # Get benchmark returns
                benchmark_returns = runner._extract_benchmark_returns(dates, "SPY" if asset_class_name == "equity" else "BTC")
                
                # Calculate metrics
                metrics_calc = MetricsCalculator(
                    equity_curve=equity_curve,
                    daily_returns=daily_returns,
                    closed_trades=closed_trades,
                    dates=dates,
                    benchmark_returns=benchmark_returns
                )
                
                all_metrics = metrics_calc.compute_all_metrics()
                metric_value = all_metrics.get(metric_name, 0.0)
                
                # Clean up temp config
                if temp_config_path.exists():
                    temp_config_path.unlink()
                
                return float(metric_value)
                
            except Exception as e:
                logger.error(f"Error running backtest with params {params}: {e}", exc_info=True)
                return 0.0
        
        # Run grid search
        grid = ParameterSensitivityGrid(
            parameter_ranges=parameter_ranges,
            metric_func=metric_func,
            random_seed=config.random_seed
        )
        
        analysis = grid.run()
        
        # Save results
        output_dir = config.get_output_dir() / "sensitivity" / asset_class_name
        save_sensitivity_results(analysis, parameter_ranges, output_dir, metric_name)
        
        # Generate heatmaps (try plotly if available, fallback to matplotlib)
        use_plotly = HAS_PLOTLY if HAS_PLOTLY else False
        if not HAS_MATPLOTLIB and not HAS_PLOTLY:
            logger.warning("No visualization library available (matplotlib or plotly). Skipping heatmap generation.")
            heatmap_paths = []
        else:
            heatmap_paths = generate_all_heatmaps(grid, output_dir, metric_name, use_plotly=use_plotly)
        
        all_results[asset_class_name] = {
            'analysis': analysis,
            'parameter_ranges': parameter_ranges,
            'output_dir': str(output_dir),
            'heatmaps': [str(p) for p in heatmap_paths]
        }
        
        logger.info(f"Sensitivity analysis completed for {asset_class_name}")
        logger.info(f"Best parameters: {analysis['best_params']}")
        logger.info(f"Best {metric_name}: {max([r['metric'] for r in analysis['results']])}")
        logger.info(f"Has sharp peaks: {analysis['has_sharp_peaks']}")
        logger.info(f"Stable neighborhoods: {len(analysis['stable_neighborhoods'])}")
    
    return {
        'status': 'completed',
        'metric_name': metric_name,
        'period': period,
        'results': all_results
    }


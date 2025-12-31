"""Integration runner for wiring up all components from configuration."""

from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import logging

from ..configs.run_config import RunConfig
from ..configs.strategy_config import StrategyConfig
from ..backtest.engine import BacktestEngine
from ..backtest.splits import WalkForwardSplit
from ..data.loader import load_all_data, load_universe
from ..strategies.equity_strategy import EquityStrategy
from ..strategies.crypto_strategy import CryptoStrategy
from ..models.market_data import MarketData
from ..reporting.csv_writer import CSVWriter
from ..reporting.json_writer import JSONWriter
from ..reporting.metrics import MetricsCalculator
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
        
        # Crypto universe is fixed
        crypto_universe = None
        if self.config.strategies.crypto and self.config.strategies.crypto.enabled:
            crypto_universe = load_universe("crypto")
        
        # Load all data
        market_data, benchmarks = load_all_data(
            equity_path=self.config.dataset.equity_path,
            crypto_path=self.config.dataset.crypto_path,
            benchmark_path=self.config.dataset.benchmark_path,
            equity_universe=equity_universe or [],
            crypto_universe=crypto_universe,
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
        strategies = []
        
        # Load equity strategy
        if self.config.strategies.equity and self.config.strategies.equity.enabled:
            equity_config = StrategyConfig.from_yaml(
                self.config.strategies.equity.config_path
            )
            equity_strategy = EquityStrategy(equity_config)
            strategies.append(equity_strategy)
            logger.info(f"Loaded equity strategy: {equity_config.name}")
        
        # Load crypto strategy
        if self.config.strategies.crypto and self.config.strategies.crypto.enabled:
            crypto_config = StrategyConfig.from_yaml(
                self.config.strategies.crypto.config_path
            )
            crypto_strategy = CryptoStrategy(crypto_config)
            strategies.append(crypto_strategy)
            logger.info(f"Loaded crypto strategy: {crypto_config.name}")
        
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
    
    def run_backtest(self, period: str = "train") -> Dict[str, Any]:
        """Run backtest for specified period.
        
        Args:
            period: One of "train", "validation", "holdout"
        
        Returns:
            Dictionary with backtest results
        """
        if self.engine is None:
            raise ValueError("Engine not created. Call create_engine() first.")
        
        split = self.create_split()
        results = self.engine.run(split=split, period=period)
        
        return results
    
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
        equity_curve = [event.get('equity', self.config.portfolio.starting_equity) for event in daily_events]
        cash_history = [event.get('cash', 0) for event in daily_events]
        positions_count = [event.get('positions_count', 0) for event in daily_events]
        exposure_history = [event.get('exposure', 0) for event in daily_events]
        daily_returns = [event.get('return', 0) for event in daily_events]
        
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
        # For stress tests, we need a function that can run backtests with modified parameters
        # Since the current architecture doesn't easily support parameter modification,
        # we'll run basic stress tests on the existing results
        
        # Slippage stress tests would require modifying the engine's execution parameters
        # For now, we'll note this limitation
        logger.warning("Stress tests require engine parameter modification - skipping for now")
        validation_results_dict['stress_tests'] = {
            'note': 'Stress tests require engine parameter modification - not yet implemented'
        }
    except Exception as e:
        logger.error(f"Stress tests failed: {e}", exc_info=True)
        validation_results_dict['stress_tests'] = {'error': str(e)}
    
    # 4. Correlation analysis
    logger.info("Running correlation stress analysis...")
    try:
        # Build portfolio history from daily events
        portfolio_history = {}
        for event in all_daily_events:
            date = event.get('date')
            if date:
                # Extract positions from portfolio state if available
                portfolio_state = event.get('portfolio_state', {})
                positions = portfolio_state.get('positions', {})
                
                portfolio_history[date] = {
                    'equity': event.get('equity', runner.config.portfolio.starting_equity),
                    'positions': positions
                }
        
        # Build returns data from market data
        returns_data = {}
        if runner.market_data:
            for symbol, bars_df in runner.market_data.bars.items():
                if 'close' in bars_df.columns:
                    returns = bars_df['close'].pct_change().dropna()
                    returns_data[symbol] = returns
        
        if portfolio_history and returns_data and len(portfolio_history) > 0:
            correlation_results = run_correlation_stress_analysis(
                portfolio_history=portfolio_history,
                returns_data=returns_data
            )
            correlation_passed, correlation_warnings = check_correlation_warnings(correlation_results)
            validation_results_dict['correlation'] = correlation_results
            if not correlation_passed:
                rejections.append("Correlation analysis failed")
            warnings.extend(correlation_warnings)
        else:
            logger.warning("Insufficient data for correlation analysis")
            validation_results_dict['correlation'] = {
                'note': 'Insufficient data for correlation analysis (need portfolio history and returns data)'
            }
    except Exception as e:
        logger.error(f"Correlation analysis failed: {e}", exc_info=True)
        validation_results_dict['correlation'] = {'error': str(e)}
    
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


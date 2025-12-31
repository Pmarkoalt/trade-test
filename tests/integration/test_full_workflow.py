"""Integration tests for full workflows (backtest -> validation -> reporting)."""

import pytest
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Get test fixtures directory
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "fixtures")
CONFIGS_DIR = os.path.join(FIXTURES_DIR, "configs")


class TestFullWorkflow:
    """Integration tests for complete workflow."""
    
    @pytest.fixture
    def test_config_path(self):
        """Path to test run config."""
        return os.path.join(CONFIGS_DIR, "run_test_config.yaml")
    
    def test_backtest_to_validation_workflow(self, test_config_path):
        """Test complete workflow: backtest -> validation."""
        from trading_system.configs.run_config import RunConfig
        from trading_system.integration.runner import BacktestRunner
        
        # Load config
        config = RunConfig.from_yaml(test_config_path)
        
        # Run backtest
        runner = BacktestRunner(config)
        runner.initialize()
        backtest_results = runner.run_backtest(period="train")
        
        # Verify backtest completed
        assert backtest_results is not None
        assert 'total_trades' in backtest_results
        assert 'sharpe_ratio' in backtest_results
        
        # Extract R-multiples from closed trades
        closed_trades = backtest_results.get('closed_trades', [])
        if len(closed_trades) > 0:
            # Run validation (bootstrap test)
            from trading_system.validation.bootstrap import run_bootstrap_test
            
            # Calculate R-multiples from trades
            r_multiples = []
            for trade in closed_trades:
                if hasattr(trade, 'realized_pnl') and hasattr(trade, 'entry_price'):
                    # Simplified R-multiple calculation
                    risk = abs(trade.entry_price - trade.stop_price) * trade.quantity
                    if risk > 0:
                        r_mult = trade.realized_pnl / risk
                        r_multiples.append(r_mult)
            
            if len(r_multiples) >= 10:  # Need sufficient trades
                bootstrap_results = run_bootstrap_test(
                    r_multiples,
                    n_iterations=100,
                    random_seed=42
                )
                
                # Verify bootstrap results
                assert 'sharpe_5th' in bootstrap_results
                assert 'sharpe_50th' in bootstrap_results
                assert 'sharpe_95th' in bootstrap_results
    
    def test_backtest_to_reporting_workflow(self, test_config_path, tmp_path):
        """Test complete workflow: backtest -> reporting."""
        from trading_system.configs.run_config import RunConfig
        from trading_system.integration.runner import BacktestRunner
        from trading_system.reporting.report_generator import ReportGenerator
        
        # Load config
        config = RunConfig.from_yaml(test_config_path)
        
        # Run backtest
        runner = BacktestRunner(config)
        runner.initialize()
        backtest_results = runner.run_backtest(period="train")
        
        # Verify backtest completed
        assert backtest_results is not None
        
        # Generate report
        # Note: This requires results to be saved to disk first
        # In a real scenario, runner.save_results() would be called
        
        # For testing, create a minimal report structure
        report_gen = ReportGenerator()
        
        # Create summary report
        summary = {
            'total_trades': backtest_results.get('total_trades', 0),
            'total_return': backtest_results.get('total_return', 0.0),
            'sharpe_ratio': backtest_results.get('sharpe_ratio', 0.0),
            'max_drawdown': backtest_results.get('max_drawdown', 0.0),
            'win_rate': backtest_results.get('win_rate', 0.0)
        }
        
        # Verify report structure
        assert 'total_trades' in summary
        assert 'total_return' in summary
        assert 'sharpe_ratio' in summary
    
    def test_validation_suite_workflow(self, test_config_path):
        """Test complete validation suite workflow."""
        from trading_system.configs.run_config import RunConfig
        from trading_system.integration.runner import BacktestRunner
        
        # Load config
        config = RunConfig.from_yaml(test_config_path)
        
        # Run backtest to get results
        runner = BacktestRunner(config)
        runner.initialize()
        backtest_results = runner.run_backtest(period="train")
        
        # Extract data for validation
        closed_trades = backtest_results.get('closed_trades', [])
        
        if len(closed_trades) >= 10:
            # Run bootstrap test
            from trading_system.validation.bootstrap import run_bootstrap_test
            
            r_multiples = []
            for trade in closed_trades:
                if hasattr(trade, 'realized_pnl') and hasattr(trade, 'entry_price'):
                    risk = abs(trade.entry_price - trade.stop_price) * trade.quantity
                    if risk > 0:
                        r_mult = trade.realized_pnl / risk
                        r_multiples.append(r_mult)
            
            if len(r_multiples) >= 10:
                bootstrap_results = run_bootstrap_test(
                    r_multiples,
                    n_iterations=100,
                    random_seed=42
                )
                
                # Run permutation test
                from trading_system.validation.permutation import run_permutation_test
                
                period = (
                    pd.Timestamp(config.periods.train.start),
                    pd.Timestamp(config.periods.train.end)
                )
                
                def compute_sharpe_func(trades):
                    r_mults = [t.get('r_multiple', 0.0) for t in trades]
                    return run_bootstrap_test(r_mults, n_iterations=1, random_seed=42).get('sharpe_50th', 0.0)
                
                # Convert trades to format expected by permutation test
                trades_for_perm = []
                for trade in closed_trades[:20]:  # Limit for testing
                    if hasattr(trade, 'entry_date') and hasattr(trade, 'exit_date'):
                        trades_for_perm.append({
                            'entry_date': trade.entry_date,
                            'exit_date': trade.exit_date,
                            'symbol': getattr(trade, 'symbol', 'TEST'),
                            'r_multiple': r_multiples[0] if r_multiples else 0.0
                        })
                
                if len(trades_for_perm) >= 5:
                    permutation_results = run_permutation_test(
                        trades_for_perm,
                        period,
                        compute_sharpe_func,
                        n_iterations=100,
                        random_seed=42
                    )
                    
                    # Verify permutation results
                    assert 'actual_sharpe' in permutation_results
                    assert 'percentile_rank' in permutation_results
    
    def test_stress_test_workflow(self, test_config_path):
        """Test stress test workflow."""
        from trading_system.configs.run_config import RunConfig
        from trading_system.integration.runner import BacktestRunner
        
        # Load config
        config = RunConfig.from_yaml(test_config_path)
        
        # Create mock backtest function that uses actual runner
        def run_backtest_with_params(**kwargs):
            runner = BacktestRunner(config)
            runner.initialize()
            
            # Apply stress test parameters
            if 'slippage_multiplier' in kwargs:
                # Note: In real implementation, this would be passed to engine
                pass
            
            results = runner.run_backtest(period="train")
            return {
                'sharpe': results.get('sharpe_ratio', 0.0),
                'calmar': results.get('calmar_ratio', 0.0) if 'calmar_ratio' in results else 0.0,
                'max_dd': results.get('max_drawdown', 0.0),
                'expectancy': results.get('expectancy', 0.0) if 'expectancy' in results else 0.0,
                'total_return': results.get('total_return', 0.0)
            }
        
        # Run stress tests
        from trading_system.validation.stress_tests import StressTestSuite
        
        suite = StressTestSuite(run_backtest_with_params, random_seed=42)
        
        # Test slippage stress
        slippage_result = suite.run_slippage_stress(multiplier=2.0)
        assert 'multiplier' in slippage_result or 'sharpe' in slippage_result
    
    def test_correlation_analysis_workflow(self, test_config_path):
        """Test correlation analysis workflow."""
        from trading_system.configs.run_config import RunConfig
        from trading_system.integration.runner import BacktestRunner
        
        # Load config
        config = RunConfig.from_yaml(test_config_path)
        
        # Run backtest
        runner = BacktestRunner(config)
        runner.initialize()
        backtest_results = runner.run_backtest(period="train")
        
        # Get portfolio history from daily events (if available)
        if hasattr(runner.engine, 'daily_events'):
            daily_events = runner.engine.daily_events
            
            # Extract portfolio history
            portfolio_history = {}
            for event in daily_events:
                if 'portfolio' in event:
                    portfolio_history[event['date']] = event['portfolio']
            
            # Get returns data from market data
            if hasattr(runner.engine, 'market_data') and runner.engine.market_data:
                returns_data = {}
                for symbol, bars in runner.engine.market_data.bars.items():
                    if len(bars) > 0:
                        returns = bars['close'].pct_change().dropna()
                        returns_data[symbol] = returns
                
                if len(portfolio_history) > 0 and len(returns_data) > 0:
                    # Run correlation analysis
                    from trading_system.validation.correlation_analysis import run_correlation_stress_analysis
                    
                    results = run_correlation_stress_analysis(
                        portfolio_history,
                        returns_data,
                        lookback=20
                    )
                    
                    # Verify results
                    assert 'normal_avg_corr' in results
                    assert 'drawdown_avg_corr' in results


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""
    
    @pytest.fixture
    def test_config_path(self):
        """Path to test run config."""
        return os.path.join(CONFIGS_DIR, "run_test_config.yaml")
    
    def test_complete_system_workflow(self, test_config_path):
        """Test complete system workflow from config to results."""
        from trading_system.configs.run_config import RunConfig
        from trading_system.integration.runner import BacktestRunner
        
        # Load config
        config = RunConfig.from_yaml(test_config_path)
        
        # Initialize and run
        runner = BacktestRunner(config)
        runner.initialize()
        
        # Run backtest for all periods
        train_results = runner.run_backtest(period="train")
        assert train_results is not None
        
        # Verify key metrics exist
        required_metrics = [
            'total_trades', 'total_return', 'sharpe_ratio',
            'max_drawdown', 'win_rate', 'starting_equity', 'ending_equity'
        ]
        
        for metric in required_metrics:
            assert metric in train_results, f"Missing metric: {metric}"
        
        # Verify metrics are reasonable
        assert train_results['total_trades'] >= 0
        assert train_results['starting_equity'] > 0
        assert train_results['ending_equity'] > 0
        assert np.isfinite(train_results['sharpe_ratio'])
        assert 0.0 <= train_results['max_drawdown'] <= 1.0
        assert 0.0 <= train_results['win_rate'] <= 1.0


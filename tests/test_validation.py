"""Unit tests for validation suite."""

import pytest
import numpy as np
import pandas as pd
from typing import List, Dict

from trading_system.validation.bootstrap import (
    BootstrapTest,
    run_bootstrap_test,
    check_bootstrap_results,
    compute_sharpe_from_r_multiples,
    compute_max_drawdown_from_r_multiples
)
from trading_system.validation.sensitivity import (
    ParameterSensitivityGrid,
    run_parameter_sensitivity
)
from trading_system.validation.permutation import (
    PermutationTest,
    run_permutation_test,
    check_permutation_results
)
from trading_system.validation.stress_tests import (
    StressTestSuite,
    check_stress_results
)
from trading_system.validation.correlation_analysis import (
    CorrelationStressAnalysis,
    run_correlation_stress_analysis,
    check_correlation_warnings
)


class TestBootstrap:
    """Test bootstrap resampling."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create sample R-multiples (mix of wins and losses)
        self.r_multiples = [
            2.5, 1.8, -1.0, 3.2, -0.5, 2.0, -1.2, 1.5, 0.8, -0.8,
            2.3, 1.2, -1.5, 2.8, -0.3, 1.8, -1.0, 2.2, 1.0, -0.5
        ]
        self.random_seed = 42
    
    def test_compute_sharpe_from_r_multiples(self):
        """Test Sharpe ratio calculation from R-multiples."""
        sharpe = compute_sharpe_from_r_multiples(self.r_multiples)
        assert isinstance(sharpe, float)
        assert sharpe > 0  # Should be positive for this sample
    
    def test_compute_sharpe_empty(self):
        """Test Sharpe calculation with empty list."""
        sharpe = compute_sharpe_from_r_multiples([])
        assert sharpe == 0.0
    
    def test_compute_max_drawdown_from_r_multiples(self):
        """Test max drawdown calculation from R-multiples."""
        max_dd = compute_max_drawdown_from_r_multiples(self.r_multiples)
        assert isinstance(max_dd, float)
        assert max_dd <= 0.0  # Drawdown should be negative or zero
    
    def test_bootstrap_test_run(self):
        """Test bootstrap test execution."""
        test = BootstrapTest(
            self.r_multiples,
            n_iterations=100,
            random_seed=self.random_seed
        )
        results = test.run()
        
        # Check results structure
        assert 'sharpe_5th' in results
        assert 'sharpe_50th' in results
        assert 'sharpe_95th' in results
        assert 'original_sharpe' in results
        assert 'sharpe_percentile_rank' in results
        
        # Check percentiles are ordered correctly
        assert results['sharpe_5th'] <= results['sharpe_50th']
        assert results['sharpe_50th'] <= results['sharpe_95th']
    
    def test_check_bootstrap_results_pass(self):
        """Test bootstrap results check with passing results."""
        results = {
            'sharpe_5th': 0.5,
            'max_dd_95th': 0.20,
            'sharpe_percentile_rank': 50.0
        }
        passed, warnings = check_bootstrap_results(results)
        assert passed
    
    def test_check_bootstrap_results_fail(self):
        """Test bootstrap results check with failing results."""
        results = {
            'sharpe_5th': 0.3,  # Below threshold
            'max_dd_95th': 0.20,
            'sharpe_percentile_rank': 50.0
        }
        passed, warnings = check_bootstrap_results(results)
        assert not passed
        assert len(warnings) > 0
    
    def test_run_bootstrap_test(self):
        """Test convenience function."""
        results = run_bootstrap_test(
            self.r_multiples,
            n_iterations=100,
            random_seed=self.random_seed
        )
        assert 'sharpe_5th' in results


class TestSensitivity:
    """Test parameter sensitivity grid."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parameter_ranges = {
            'atr_mult': [2.0, 2.5, 3.0],
            'clearance': [0.005, 0.010]
        }
        self.random_seed = 42
    
    def test_parameter_sensitivity_grid(self):
        """Test parameter sensitivity grid search."""
        def metric_func(params):
            # Simple test function
            return params['atr_mult'] * params['clearance'] * 100
        
        grid = ParameterSensitivityGrid(
            self.parameter_ranges,
            metric_func,
            random_seed=self.random_seed
        )
        analysis = grid.run()
        
        # Check results
        assert 'results' in analysis
        assert 'best_params' in analysis
        assert 'metric_mean' in analysis
        assert len(analysis['results']) > 0
    
    def test_find_best_params(self):
        """Test finding best parameters."""
        def metric_func(params):
            return params['atr_mult']
        
        grid = ParameterSensitivityGrid(
            self.parameter_ranges,
            metric_func,
            random_seed=self.random_seed
        )
        analysis = grid.run()
        
        # Best should have highest atr_mult
        best_atr = analysis['best_params']['atr_mult']
        assert best_atr == 3.0
    
    def test_check_sharp_peaks(self):
        """Test sharp peak detection."""
        def metric_func(params):
            # Create a sharp peak
            if params['atr_mult'] == 2.5 and params['clearance'] == 0.010:
                return 100.0
            return 1.0
        
        grid = ParameterSensitivityGrid(
            self.parameter_ranges,
            metric_func,
            random_seed=self.random_seed
        )
        analysis = grid.run()
        
        # Should detect sharp peak
        assert analysis['has_sharp_peaks']


class TestPermutation:
    """Test permutation test."""
    
    def setup_method(self):
        """Set up test fixtures."""
        start_date = pd.Timestamp('2023-01-01')
        end_date = pd.Timestamp('2023-12-31')
        
        # Create sample trades
        self.actual_trades = [
            {
                'entry_date': start_date + pd.Timedelta(days=10),
                'exit_date': start_date + pd.Timedelta(days=25),
                'symbol': 'AAPL',
                'r_multiple': 2.0
            },
            {
                'entry_date': start_date + pd.Timedelta(days=50),
                'exit_date': start_date + pd.Timedelta(days=65),
                'symbol': 'MSFT',
                'r_multiple': 1.5
            }
        ]
        
        self.period = (start_date, end_date)
        self.random_seed = 42
        
        def compute_sharpe_func(trades):
            r_multiples = [t.get('r_multiple', 0.0) for t in trades]
            return compute_sharpe_from_r_multiples(r_multiples)
        
        self.compute_sharpe_func = compute_sharpe_func
    
    def test_permutation_test_run(self):
        """Test permutation test execution."""
        test = PermutationTest(
            self.actual_trades,
            self.period,
            self.compute_sharpe_func,
            n_iterations=100,
            random_seed=self.random_seed
        )
        results = test.run()
        
        # Check results structure
        assert 'actual_sharpe' in results
        assert 'random_sharpe_5th' in results
        assert 'random_sharpe_95th' in results
        assert 'percentile_rank' in results
        assert 'passed' in results
    
    def test_check_permutation_results_pass(self):
        """Test permutation results check with passing results."""
        results = {
            'actual_sharpe': 2.0,
            'random_sharpe_95th': 1.5,
            'percentile_rank': 96.0,
            'passed': True
        }
        passed, warnings = check_permutation_results(results)
        assert passed
    
    def test_check_permutation_results_fail(self):
        """Test permutation results check with failing results."""
        results = {
            'actual_sharpe': 1.0,
            'random_sharpe_95th': 1.5,
            'percentile_rank': 90.0,
            'passed': False
        }
        passed, warnings = check_permutation_results(results)
        assert not passed


class TestStressTests:
    """Test stress tests."""
    
    def setup_method(self):
        """Set up test fixtures."""
        def mock_backtest_func(**kwargs):
            # Mock backtest function
            slippage_mult = kwargs.get('slippage_multiplier', 1.0)
            return {
                'sharpe': 1.5 / slippage_mult,  # Degrades with slippage
                'calmar': 2.0 / slippage_mult,
                'max_dd': -0.10,
                'expectancy': 0.5 / slippage_mult,
                'total_return': 0.20
            }
        
        self.run_backtest_func = mock_backtest_func
        self.random_seed = 42
    
    def test_slippage_stress(self):
        """Test slippage stress test."""
        suite = StressTestSuite(self.run_backtest_func, self.random_seed)
        result = suite.run_slippage_stress(multiplier=2.0)
        
        assert 'multiplier' in result
        assert 'sharpe' in result
        assert result['multiplier'] == 2.0
    
    def test_check_stress_results_pass(self):
        """Test stress results check with passing results."""
        stress_results = {
            'slippage_2x': {'sharpe': 0.80, 'calmar': 1.5},
            'slippage_3x': {'calmar': 1.2},
            'bear_market': {'max_dd': -0.20},
            'flash_crash': {'max_dd': -0.20, 'survived': True}
        }
        passed, warnings = check_stress_results(stress_results)
        assert passed
    
    def test_check_stress_results_fail(self):
        """Test stress results check with failing results."""
        stress_results = {
            'slippage_2x': {'sharpe': 0.50},  # Below threshold
            'slippage_3x': {'calmar': 0.8},  # Below threshold
            'flash_crash': {'max_dd': -0.30, 'survived': False}  # Too much DD
        }
        passed, warnings = check_stress_results(stress_results)
        assert not passed


class TestCorrelationAnalysis:
    """Test correlation stress analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample returns data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Create correlated returns
        np.random.seed(42)
        base_returns = np.random.randn(100) * 0.01
        
        self.returns_data = {
            'AAPL': pd.Series(base_returns * 1.0, index=dates),
            'MSFT': pd.Series(base_returns * 0.8 + np.random.randn(100) * 0.002, index=dates),
            'GOOGL': pd.Series(base_returns * 0.6 + np.random.randn(100) * 0.004, index=dates)
        }
        
        # Create portfolio history with drawdowns
        self.portfolio_history = {}
        for i, date in enumerate(dates):
            # Simulate drawdown in middle period
            if 40 <= i < 60:
                equity = 100000 * (1 - 0.10)  # 10% drawdown
                positions = {'AAPL': {}, 'MSFT': {}}
            else:
                equity = 100000
                positions = {'AAPL': {}, 'GOOGL': {}}
            
            self.portfolio_history[date] = {
                'equity': equity,
                'positions': positions
            }
    
    def test_correlation_stress_analysis(self):
        """Test correlation stress analysis."""
        analysis = CorrelationStressAnalysis(
            self.portfolio_history,
            self.returns_data,
            lookback=20
        )
        results = analysis.run()
        
        # Check results structure
        assert 'normal_avg_corr' in results
        assert 'drawdown_avg_corr' in results
        assert 'warning' in results
    
    def test_check_correlation_warnings(self):
        """Test correlation warnings check."""
        results = {
            'drawdown_avg_corr': 0.75,  # Above threshold
            'correlation_increase': 0.25,
            'warning': True
        }
        passed, warnings = check_correlation_warnings(results)
        assert passed  # Warnings don't fail, just warn
        assert len(warnings) > 0
    
    def test_compute_correlation_matrix(self):
        """Test correlation matrix computation."""
        analysis = CorrelationStressAnalysis(
            self.portfolio_history,
            self.returns_data,
            lookback=20
        )
        
        date = pd.Timestamp('2023-01-30')
        symbols = ['AAPL', 'MSFT']
        corr_matrix = analysis._compute_correlation_matrix(symbols, date, 20)
        
        assert corr_matrix is not None
        assert corr_matrix.shape == (2, 2)
    
    def test_compute_avg_pairwise_corr(self):
        """Test average pairwise correlation computation."""
        analysis = CorrelationStressAnalysis(
            self.portfolio_history,
            self.returns_data,
            lookback=20
        )
        
        # Create test correlation matrix
        corr_matrix = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0]
        ])
        
        avg_corr = analysis._compute_avg_pairwise_corr(corr_matrix)
        expected = (0.5 + 0.3 + 0.4) / 3
        
        assert abs(avg_corr - expected) < 1e-5


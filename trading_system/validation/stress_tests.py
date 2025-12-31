"""Stress tests for adverse market conditions."""

from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
import pandas as pd


class StressTestSuite:
    """Suite of stress tests for strategy robustness."""
    
    def __init__(
        self,
        run_backtest_func: Callable,
        random_seed: Optional[int] = None
    ):
        """Initialize stress test suite.
        
        Args:
            run_backtest_func: Function that runs backtest and returns results dict
            random_seed: Random seed for reproducibility
        """
        self.run_backtest_func = run_backtest_func
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def run_all(self) -> Dict[str, Dict]:
        """Run all stress tests.
        
        Returns:
            Dictionary mapping test names to results
        """
        results = {}
        
        # Slippage stress tests
        results['slippage_2x'] = self.run_slippage_stress(multiplier=2.0)
        results['slippage_3x'] = self.run_slippage_stress(multiplier=3.0)
        
        # Market regime tests
        results['bear_market'] = self.run_bear_market_test()
        results['range_market'] = self.run_range_market_test()
        
        # Flash crash simulation
        results['flash_crash'] = self.run_flash_crash_simulation()
        
        return results
    
    def run_slippage_stress(self, multiplier: float) -> Dict:
        """Run slippage stress test.
        
        Args:
            multiplier: Slippage multiplier (1x, 2x, 3x)
        
        Returns:
            Backtest results with stress slippage
        """
        # Modify slippage in backtest function
        results = self.run_backtest_func(slippage_multiplier=multiplier)
        
        return {
            'multiplier': multiplier,
            'results': results,
            'sharpe': results.get('sharpe_ratio', 0.0),
            'calmar': results.get('calmar_ratio', 0.0),
            'max_dd': results.get('max_drawdown', 0.0),
            'total_return': results.get('total_return', 0.0)
        }
    
    def run_bear_market_test(self) -> Dict:
        """Run backtest on bear market months only.
        
        Bear market: months where benchmark return < -5%
        
        Returns:
            Backtest results for bear market only
        """
        # Filter to bear market months
        results = self.run_backtest_func(date_filter='bear')
        
        return {
            'regime': 'bear',
            'results': results,
            'sharpe': results.get('sharpe_ratio', 0.0),
            'max_dd': results.get('max_drawdown', 0.0),
            'total_return': results.get('total_return', 0.0)
        }
    
    def run_range_market_test(self) -> Dict:
        """Run backtest on range market months only.
        
        Range market: months where benchmark return is between -2% and +2%
        
        Returns:
            Backtest results for range market only
        """
        # Filter to range market months
        results = self.run_backtest_func(date_filter='range')
        
        return {
            'regime': 'range',
            'results': results,
            'sharpe': results.get('sharpe_ratio', 0.0),
            'max_dd': results.get('max_drawdown', 0.0),
            'total_return': results.get('total_return', 0.0)
        }
    
    def run_flash_crash_simulation(self) -> Dict:
        """Run flash crash simulation.
        
        Simulates extreme stress:
        - Slippage × 5 on random days (one per quarter)
        - All stops hit at worst possible price
        
        Returns:
            Backtest results with flash crash simulation
        """
        # Run with flash crash simulation
        # crash_dates will be passed as a list of dates (one per quarter)
        results = self.run_backtest_func(
            slippage_multiplier=1.0,  # Base multiplier, crashes apply 5x
            crash_dates='auto'  # Will be generated automatically (one per quarter)
        )
        
        max_dd = results.get('max_drawdown', 0.0)
        
        return {
            'simulation': 'flash_crash',
            'results': results,
            'max_dd': max_dd,
            'survived': max_dd > -0.25  # DD < 25%
        }


def run_slippage_stress(
    run_backtest_func: Callable,
    multiplier: float,
    random_seed: Optional[int] = None
) -> Dict:
    """Run slippage stress test.
    
    Args:
        run_backtest_func: Function that runs backtest
        multiplier: Slippage multiplier
        random_seed: Random seed
    
    Returns:
        Stress test results
    """
    suite = StressTestSuite(run_backtest_func, random_seed)
    return suite.run_slippage_stress(multiplier)


def run_bear_market_test(
    run_backtest_func: Callable,
    random_seed: Optional[int] = None
) -> Dict:
    """Run bear market test.
    
    Args:
        run_backtest_func: Function that runs backtest
        random_seed: Random seed
    
    Returns:
        Bear market test results
    """
    suite = StressTestSuite(run_backtest_func, random_seed)
    return suite.run_bear_market_test()


def run_range_market_test(
    run_backtest_func: Callable,
    random_seed: Optional[int] = None
) -> Dict:
    """Run range market test.
    
    Args:
        run_backtest_func: Function that runs backtest
        random_seed: Random seed
    
    Returns:
        Range market test results
    """
    suite = StressTestSuite(run_backtest_func, random_seed)
    return suite.run_range_market_test()


def run_flash_crash_simulation(
    run_backtest_func: Callable,
    random_seed: Optional[int] = None
) -> Dict:
    """Run flash crash simulation.
    
    Args:
        run_backtest_func: Function that runs backtest
        random_seed: Random seed
    
    Returns:
        Flash crash simulation results
    """
    suite = StressTestSuite(run_backtest_func, random_seed)
    return suite.run_flash_crash_simulation()


def check_stress_results(stress_results: Dict[str, Dict]) -> Tuple[bool, List[str]]:
    """Check stress test results against acceptance criteria.
    
    Args:
        stress_results: Dictionary of stress test results
    
    Returns:
        (passed, warnings)
    """
    warnings = []
    all_passed = True
    
    # Check slippage stress
    if 'slippage_2x' in stress_results:
        result = stress_results['slippage_2x']
        sharpe = result.get('sharpe', 0.0)
        if sharpe <= 0.75:
            all_passed = False
            warnings.append(
                f"REJECT: 2x slippage Sharpe {sharpe:.2f} <= 0.75"
            )
    
    if 'slippage_3x' in stress_results:
        result = stress_results['slippage_3x']
        calmar = result.get('calmar', 0.0)
        if calmar <= 1.0:
            all_passed = False
            warnings.append(
                f"REJECT: 3x slippage Calmar {calmar:.2f} <= 1.0"
            )
    
    # Check bear market
    if 'bear_market' in stress_results:
        result = stress_results['bear_market']
        max_dd = result.get('max_dd', 0.0)
        if max_dd < -0.25:  # DD > 25%
            all_passed = False
            warnings.append(
                f"REJECT: Bear market Max DD {max_dd:.1%} > 25%"
            )
    
    # Check range market
    if 'range_market' in stress_results:
        result = stress_results['range_market']
        max_dd = result.get('max_dd', 0.0)
        total_return = result.get('total_return', 0.0)
        if max_dd < -0.15:  # DD > 15%
            warnings.append(
                f"WARNING: Range market Max DD {max_dd:.1%} > 15%"
            )
        if total_return < -0.05:  # Return < -5%
            warnings.append(
                f"WARNING: Range market return {total_return:.1%} < -5%"
            )
    
    # Check flash crash
    if 'flash_crash' in stress_results:
        result = stress_results['flash_crash']
        max_dd = result.get('max_dd', 0.0)
        survived = result.get('survived', False)
        if not survived or max_dd < -0.25:  # DD > 25%
            all_passed = False
            warnings.append(
                f"REJECT: Flash crash Max DD {max_dd:.1%} > 25% or portfolio did not survive"
            )
    
    return (all_passed, warnings)


"""Validation and robustness testing suite."""

from .bootstrap import BootstrapTest, check_bootstrap_results, run_bootstrap_test
from .correlation_analysis import CorrelationStressAnalysis, check_correlation_warnings, run_correlation_stress_analysis
from .permutation import PermutationTest, check_permutation_results, run_permutation_test
from .sensitivity import ParameterSensitivityGrid, run_parameter_sensitivity
from .stress_tests import (
    StressTestSuite,
    check_stress_results,
    run_bear_market_test,
    run_flash_crash_simulation,
    run_range_market_test,
    run_slippage_stress,
)

__all__ = [
    "ParameterSensitivityGrid",
    "run_parameter_sensitivity",
    "BootstrapTest",
    "run_bootstrap_test",
    "check_bootstrap_results",
    "PermutationTest",
    "run_permutation_test",
    "check_permutation_results",
    "StressTestSuite",
    "run_slippage_stress",
    "run_bear_market_test",
    "run_range_market_test",
    "run_flash_crash_simulation",
    "check_stress_results",
    "CorrelationStressAnalysis",
    "run_correlation_stress_analysis",
    "check_correlation_warnings",
]

"""Validation and robustness testing suite."""

from .sensitivity import ParameterSensitivityGrid, run_parameter_sensitivity
from .bootstrap import BootstrapTest, run_bootstrap_test, check_bootstrap_results
from .permutation import PermutationTest, run_permutation_test, check_permutation_results
from .stress_tests import (
    StressTestSuite,
    run_slippage_stress,
    run_bear_market_test,
    run_range_market_test,
    run_flash_crash_simulation,
    check_stress_results
)
from .correlation_analysis import (
    CorrelationStressAnalysis,
    run_correlation_stress_analysis,
    check_correlation_warnings
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


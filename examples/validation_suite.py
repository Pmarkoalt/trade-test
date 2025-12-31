"""
Validation Suite Example

This example demonstrates how to run the comprehensive validation suite,
including bootstrap tests, permutation tests, and stress tests.

Usage:
    python examples/validation_suite.py
"""

from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.integration.runner import run_validation
from trading_system.configs.run_config import RunConfig


def example_run_validation_suite():
    """Example: Run the full validation suite."""
    print("=" * 60)
    print("Example 1: Running Full Validation Suite")
    print("=" * 60)
    
    config_path = "EXAMPLE_CONFIGS/run_config.yaml"
    
    print(f"\nLoading config from: {config_path}")
    print("Running validation suite (this may take several minutes)...")
    print("\nThe validation suite includes:")
    print("  1. Bootstrap test - Statistical significance of returns")
    print("  2. Permutation test - Randomization test for strategy edge")
    print("  3. Stress tests - Slippage, bear market, range market, flash crash")
    print("  4. Correlation analysis - Portfolio diversification check")
    
    # Run validation suite
    validation_results = run_validation(config_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Validation Suite Results")
    print("=" * 60)
    
    status = validation_results.get('status', 'unknown')
    print(f"\nOverall Status: {status.upper()}")
    
    # Print rejections
    rejections = validation_results.get('rejections', [])
    if rejections:
        print(f"\nRejections ({len(rejections)}):")
        for rejection in rejections:
            print(f"  ✗ {rejection}")
    else:
        print("\n✓ No rejections")
    
    # Print warnings
    warnings = validation_results.get('warnings', [])
    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for warning in warnings[:5]:  # Show first 5
            print(f"  ⚠ {warning}")
        if len(warnings) > 5:
            print(f"  ... and {len(warnings) - 5} more warnings")
    else:
        print("\n✓ No warnings")
    
    # Print trade statistics
    print(f"\nTrade Statistics:")
    print(f"  Total Trades: {validation_results.get('total_trades', 0)}")
    print(f"  R-Multiples Count: {validation_results.get('r_multiples_count', 0)}")
    print(f"  Average R-Multiple: {validation_results.get('avg_r_multiple', 0):.2f}")
    
    # Print detailed results
    results = validation_results.get('results', {})
    
    # Bootstrap results
    if 'bootstrap' in results:
        bootstrap = results['bootstrap']
        print(f"\nBootstrap Test:")
        print(f"  Status: {'PASSED' if bootstrap.get('passed', False) else 'FAILED'}")
        if 'mean_r_multiple' in bootstrap:
            print(f"  Mean R-Multiple: {bootstrap['mean_r_multiple']:.2f}")
        if 'bootstrap_5th_percentile' in bootstrap:
            print(f"  5th Percentile: {bootstrap['bootstrap_5th_percentile']:.2f}")
    
    # Permutation results
    if 'permutation' in results:
        permutation = results['permutation']
        print(f"\nPermutation Test:")
        if 'p_value' in permutation:
            print(f"  P-Value: {permutation['p_value']:.4f}")
        if 'passed' in permutation:
            print(f"  Status: {'PASSED' if permutation['passed'] else 'FAILED'}")
    
    # Stress test results
    if 'stress_tests' in results:
        stress = results['stress_tests']
        print(f"\nStress Tests:")
        print(f"  Status: {'PASSED' if stress.get('passed', False) else 'FAILED'}")
        
        # Slippage stress
        for key in stress.keys():
            if key.startswith('slippage_'):
                slippage_result = stress[key]
                if 'total_return' in slippage_result:
                    print(f"  {key}: Return = {slippage_result['total_return']:.2%}")
        
        # Bear market
        if 'bear_market' in stress:
            bear = stress['bear_market']
            if 'total_return' in bear:
                print(f"  Bear Market: Return = {bear['total_return']:.2%}")
        
        # Range market
        if 'range_market' in stress:
            range_mkt = stress['range_market']
            if 'total_return' in range_mkt:
                print(f"  Range Market: Return = {range_mkt['total_return']:.2%}")
        
        # Flash crash
        if 'flash_crash' in stress:
            flash = stress['flash_crash']
            if 'total_return' in flash:
                print(f"  Flash Crash: Return = {flash['total_return']:.2%}")
    
    # Correlation analysis
    if 'correlation' in results:
        correlation = results['correlation']
        print(f"\nCorrelation Analysis:")
        if 'max_avg_pairwise_correlation' in correlation:
            max_corr = correlation['max_avg_pairwise_correlation']
            print(f"  Max Avg Pairwise Correlation: {max_corr:.2f}")
        if 'warnings' in correlation:
            print(f"  Warnings: {len(correlation['warnings'])}")
    
    return validation_results


def example_save_validation_results():
    """Example: Save validation results to JSON."""
    print("\n" + "=" * 60)
    print("Example 2: Saving Validation Results")
    print("=" * 60)
    
    config_path = "EXAMPLE_CONFIGS/run_config.yaml"
    validation_results = run_validation(config_path)
    
    # Save to JSON
    output_path = Path("results/validation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable format
    def convert_to_dict(obj):
        """Recursively convert objects to dict."""
        if isinstance(obj, dict):
            return {k: convert_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_dict(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return convert_to_dict(obj.__dict__)
        elif hasattr(obj, 'isoformat'):  # datetime
            return obj.isoformat()
        else:
            return obj
    
    results_dict = convert_to_dict(validation_results)
    
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    
    print(f"\nValidation results saved to: {output_path}")
    
    return output_path


def example_cli_validation():
    """Example: Show CLI usage for validation."""
    print("\n" + "=" * 60)
    print("Example 3: CLI Usage for Validation")
    print("=" * 60)
    
    print("""
To run validation suite from the command line:

    python -m trading_system validate \\
        --config EXAMPLE_CONFIGS/run_config.yaml

The validation suite will:
  1. Run backtests on train+validation periods
  2. Perform statistical tests (bootstrap, permutation)
  3. Run stress tests (slippage, bear market, range market, flash crash)
  4. Perform correlation analysis
  5. Generate warnings and rejections

Results are printed to console and can be saved to JSON.
    """)


def example_validation_config():
    """Example: Show validation configuration options."""
    print("\n" + "=" * 60)
    print("Example 4: Validation Configuration")
    print("=" * 60)
    
    print("""
Configure validation in your run_config.yaml:

validation:
  # Statistical tests
  statistical:
    bootstrap_iterations: 1000  # Number of bootstrap samples
    permutation_iterations: 1000  # Number of permutation samples
    bootstrap_5th_percentile_threshold: 0.4  # Reject if < 0.4
  
  # Stress tests
  stress_tests:
    slippage_multipliers: [1.0, 2.0, 3.0]  # Baseline, 2x, 3x slippage
    bear_market_test: true  # Test during bear market months
    range_market_test: true  # Test during range-bound months
    flash_crash_test: true  # Test with flash crash simulation
  
  # Sensitivity analysis (optional)
  sensitivity:
    enabled: true
    equity_atr_mult: [2.0, 2.5, 3.0, 3.5]
    equity_breakout_clearance: [0.000, 0.005, 0.010, 0.015]
    # ... more parameter ranges

See EXAMPLE_CONFIGS/run_config.yaml for a complete example.
    """)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Validation Suite Examples")
    print("=" * 60)
    
    try:
        # Example 1: Run validation suite
        validation_results = example_run_validation_suite()
        
        # Example 2: Save results
        example_save_validation_results()
        
        # Example 3: CLI usage
        example_cli_validation()
        
        # Example 4: Configuration
        example_validation_config()
        
        print("\n" + "=" * 60)
        print("Validation Suite Examples Completed!")
        print("=" * 60)
        print("\nInterpretation:")
        print("  - Status 'passed': All tests passed, strategy is validated")
        print("  - Status 'failed': One or more tests failed, review rejections")
        print("  - Warnings: Non-critical issues that should be reviewed")
        print("  - Rejections: Critical failures that invalidate the strategy")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


"""
Sensitivity Analysis Example

This example demonstrates how to run parameter sensitivity analysis
to find optimal strategy parameters.

Usage:
    python examples/sensitivity_analysis.py
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.integration.runner import run_sensitivity_analysis  # noqa: E402


def example_run_sensitivity_analysis():
    """Example: Run parameter sensitivity analysis."""
    print("=" * 60)
    print("Example 1: Running Parameter Sensitivity Analysis")
    print("=" * 60)

    config_path = "EXAMPLE_CONFIGS/run_config.yaml"

    print(f"\nLoading config from: {config_path}")
    print("Running sensitivity analysis (this may take a while)...")
    print("\nThe analysis will:")
    print("  1. Generate parameter grid from config")
    print("  2. Run backtests for each parameter combination")
    print("  3. Compute metrics (Sharpe ratio, etc.)")
    print("  4. Identify best parameters")
    print("  5. Generate heatmaps")

    # Run sensitivity analysis
    print("\nRunning sensitivity analysis for equity strategy...")
    results = run_sensitivity_analysis(
        config_path=config_path,
        period="train",  # or "validation", "holdout"
        metric_name="sharpe_ratio",  # or "total_return", "calmar_ratio", etc.
        asset_class="equity",  # or "crypto", or None for all
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Sensitivity Analysis Results")
    print("=" * 60)

    status = results.get("status", "unknown")
    print(f"\nStatus: {status.upper()}")
    print(f"Metric: {results.get('metric_name', 'N/A')}")
    print(f"Period: {results.get('period', 'N/A')}")

    # Print results for each asset class
    all_results = results.get("results", {})

    for asset_class, class_results in all_results.items():
        print(f"\n{asset_class.upper()} Strategy Results:")

        analysis = class_results.get("analysis", {})

        # Best parameters
        best_params = analysis.get("best_params", {})
        if best_params:
            print("\n  Best Parameters:")
            for param, value in best_params.items():
                print(f"    {param}: {value}")

        # Best metric value
        results_list = analysis.get("results", [])
        if results_list:
            best_metric = max(r.get("metric", 0) for r in results_list)
            print(f"\n  Best {results.get('metric_name', 'metric')}: {best_metric:.4f}")

        # Stability analysis
        has_sharp_peaks = analysis.get("has_sharp_peaks", False)
        stable_neighborhoods = analysis.get("stable_neighborhoods", [])

        print("\n  Stability Analysis:")
        print(f"    Has Sharp Peaks: {has_sharp_peaks}")
        print(f"    Stable Neighborhoods: {len(stable_neighborhoods)}")

        if stable_neighborhoods:
            print("\n  Stable Parameter Regions:")
            for i, neighborhood in enumerate(stable_neighborhoods[:3], 1):  # Show first 3
                print(f"    Region {i}: {neighborhood.get('params', {})}")
                print(f"      Metric: {neighborhood.get('metric', 0):.4f}")

        # Output directory
        output_dir = class_results.get("output_dir", "")
        print(f"\n  Results saved to: {output_dir}")

        # Heatmaps
        heatmaps = class_results.get("heatmaps", [])
        if heatmaps:
            print("\n  Generated Heatmaps:")
            for heatmap_path in heatmaps:
                print(f"    {heatmap_path}")

    return results


def example_sensitivity_config():
    """Example: Show sensitivity analysis configuration."""
    print("\n" + "=" * 60)
    print("Example 2: Sensitivity Analysis Configuration")
    print("=" * 60)

    print(
        """
Configure sensitivity analysis in your run_config.yaml:

validation:
  sensitivity:
    enabled: true
    
    # Equity strategy parameters to test
    equity_atr_mult: [2.0, 2.5, 3.0, 3.5]  # Stop loss ATR multiplier
    equity_breakout_clearance: [0.000, 0.005, 0.010, 0.015]  # Breakout clearance %
    equity_exit_ma: [20, 50]  # Exit MA period
    
    # Crypto strategy parameters to test
    crypto_atr_mult: [2.5, 3.0, 3.5, 4.0]
    crypto_breakout_clearance: [0.000, 0.005, 0.010, 0.015]
    crypto_exit_mode: ["MA20", "MA50", "staged"]
    
    # Portfolio-level parameters
    vol_scaling_mode: ["continuous", "regime", "off"]

The analysis will test all combinations of these parameters.
Total combinations = product of all parameter list lengths.

Example:
  equity_atr_mult: 4 values
  equity_breakout_clearance: 4 values
  equity_exit_ma: 2 values
  Total = 4 × 4 × 2 = 32 combinations

See EXAMPLE_CONFIGS/run_config.yaml for a complete example.
    """
    )


def example_interpret_results():
    """Example: How to interpret sensitivity analysis results."""
    print("\n" + "=" * 60)
    print("Example 3: Interpreting Results")
    print("=" * 60)

    print(
        """
Interpreting Sensitivity Analysis Results:

1. Best Parameters:
   - Shows the parameter combination with highest metric value
   - Use these as starting point for further optimization

2. Has Sharp Peaks:
   - True: Small parameter changes cause large metric changes (unstable)
   - False: Metric changes smoothly with parameters (stable)
   - Prefer strategies without sharp peaks (more robust)

3. Stable Neighborhoods:
   - Regions where metric is consistently high
   - Indicates robust parameter ranges
   - Prefer parameters in stable neighborhoods

4. Heatmaps:
   - Visualize metric across 2D parameter space
   - Look for smooth, wide regions of high performance
   - Avoid narrow peaks (overfitting risk)

5. Metric Selection:
   - sharpe_ratio: Risk-adjusted returns
   - total_return: Absolute returns (ignores risk)
   - calmar_ratio: Return / max drawdown
   - Choose metric based on your objectives

Best Practices:
  - Run on train period to find parameters
  - Validate on validation period
  - Test on holdout period (final check)
  - Avoid overfitting to train period
  - Prefer stable, wide parameter ranges
    """
    )


def example_cli_sensitivity():
    """Example: Show CLI usage for sensitivity analysis."""
    print("\n" + "=" * 60)
    print("Example 4: CLI Usage for Sensitivity Analysis")
    print("=" * 60)

    print(
        """
To run sensitivity analysis from the command line:

    python -m trading_system sensitivity \\
        --config EXAMPLE_CONFIGS/run_config.yaml \\
        --period train \\
        --metric sharpe_ratio \\
        --asset-class equity

Options:
  --period: train, validation, or holdout
  --metric: sharpe_ratio, total_return, calmar_ratio, etc.
  --asset-class: equity, crypto, or omit for all

Results will be saved to: results/{run_id}/sensitivity/{asset_class}/

The analysis generates:
  - CSV files with all parameter combinations and metrics
  - Heatmap visualizations (if matplotlib/plotly available)
  - Best parameters summary
  - Stability analysis
    """
    )


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Sensitivity Analysis Examples")
    print("=" * 60)

    try:
        # Example 1: Run sensitivity analysis
        # Note: This may take a long time depending on parameter grid size
        print("\nNote: Sensitivity analysis can take a long time.")
        print("      Uncomment the line below to run it.")
        # results = example_run_sensitivity_analysis()

        # Example 2: Configuration
        example_sensitivity_config()

        # Example 3: Interpretation
        example_interpret_results()

        # Example 4: CLI usage
        example_cli_sensitivity()

        print("\n" + "=" * 60)
        print("Sensitivity Analysis Examples Completed!")
        print("=" * 60)
        print("\nTo actually run sensitivity analysis:")
        print("  1. Ensure validation.sensitivity is configured in run_config.yaml")
        print("  2. Uncomment the run_sensitivity_analysis() call above")
        print("  3. Or use the CLI command shown in Example 4")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

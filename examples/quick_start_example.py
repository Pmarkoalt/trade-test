"""
Quick Start Example - Minimal Data Backtest

This example demonstrates the simplest possible backtest using minimal test data.
Perfect for first-time users who want to get started quickly.

The example uses:
- Minimal test data (3 months, 3 symbols)
- Test configuration files
- Simple programmatic execution

Usage:
    python examples/quick_start_example.py
"""

from pathlib import Path
import sys

# Add parent directory to path to import trading_system
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.integration.runner import run_backtest  # noqa: E402


def quick_start_backtest():
    """Run a quick backtest with minimal test data."""
    print("=" * 70)
    print("Quick Start Example - Minimal Data Backtest")
    print("=" * 70)
    print()
    print("This example uses:")
    print("  - Test fixtures (3 symbols, 3 months of data)")
    print("  - Test configuration files")
    print("  - Minimal setup - just run and see results!")
    print()

    # Use test configuration with minimal data
    config_path = "tests/fixtures/configs/run_test_config.yaml"

    print(f"Loading configuration from: {config_path}")
    print("Running backtest on training period...")
    print()

    try:
        # Run the backtest (this is the simplest way)
        results = run_backtest(config_path, period="train")

        print("=" * 70)
        print("Backtest Results Summary")
        print("=" * 70)
        print()

        # Display key metrics
        print("Performance Metrics:")
        print(f"  Total Return:     {results.get('total_return', 0):>8.2%}")
        print(f"  Sharpe Ratio:     {results.get('sharpe_ratio', 0):>8.2f}")
        print(f"  Max Drawdown:     {results.get('max_drawdown', 0):>8.2%}")
        print(f"  Calmar Ratio:     {results.get('calmar_ratio', 0):>8.2f}")
        print()

        print("Trading Statistics:")
        print(f"  Total Trades:     {results.get('total_trades', 0):>8d}")
        print(f"  Win Rate:         {results.get('win_rate', 0):>8.2%}")
        print(f"  Avg R-Multiple:   {results.get('avg_r_multiple', 0):>8.2f}")
        print(f"  Profit Factor:    {results.get('profit_factor', 0):>8.2f}")
        print()

        print("=" * 70)
        print("Where to find detailed results:")
        print("=" * 70)
        print()
        print("Results are saved in: tests/results/{run_id}/train/")
        print()
        print("Key files:")
        print("  - equity_curve.csv    : Daily portfolio value and positions")
        print("  - trade_log.csv       : All executed trades with details")
        print("  - weekly_summary.csv  : Weekly performance summaries")
        print("  - monthly_report.json : Detailed metrics and statistics")
        print("  - backtest.log        : Execution log file")
        print()

        print("=" * 70)
        print("Next Steps")
        print("=" * 70)
        print()
        print("1. Review the output files to understand the results")
        print("2. Try modifying the config file to change strategy parameters")
        print("3. Check out other examples:")
        print("   - examples/basic_backtest.py       : More detailed backtest example")
        print("   - examples/all_strategies_example.py : All strategy types")
        print("   - examples/validation_suite.py     : Validation and stress tests")
        print("4. Read the documentation:")
        print("   - README.md              : System overview")
        print("   - docs/user_guide/       : User guides and tutorials")
        print("   - EXAMPLE_CONFIGS/README.md : Configuration examples")
        print()

        return results

    except Exception as e:
        print(f"\n❌ Error running backtest: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Make sure you're running from the project root directory")
        print("  2. Verify test fixtures exist: tests/fixtures/")
        print("  3. Check that dependencies are installed: pip install -r requirements.txt")
        print()
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        quick_start_backtest()
        print("✅ Quick start example completed successfully!")
        print()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

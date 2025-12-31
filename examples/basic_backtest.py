"""
Basic Backtest Example

This example demonstrates how to run a simple backtest using the trading system.
It shows both programmatic usage and CLI usage.

Usage:
    python examples/basic_backtest.py
"""

from pathlib import Path
import sys

# Add parent directory to path to import trading_system
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.integration.runner import run_backtest
from trading_system.configs.run_config import RunConfig
from trading_system.integration.runner import BacktestRunner
from trading_system.reporting.metrics import MetricsCalculator


def example_programmatic_backtest():
    """Example: Run backtest programmatically."""
    print("=" * 60)
    print("Example 1: Programmatic Backtest")
    print("=" * 60)
    
    # Path to your run config
    config_path = "EXAMPLE_CONFIGS/run_config.yaml"
    
    # Option 1: Use convenience function
    print("\nRunning backtest using convenience function...")
    results = run_backtest(config_path, period="train")
    
    # Print basic results
    print(f"\nBacktest Results:")
    print(f"  Total Return: {results.get('total_return', 0):.2%}")
    print(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
    print(f"  Max Drawdown: {results.get('max_drawdown', 0):.2%}")
    print(f"  Total Trades: {results.get('total_trades', 0)}")
    print(f"  Win Rate: {results.get('win_rate', 0):.2%}")
    
    return results


def example_runner_backtest():
    """Example: Use BacktestRunner for more control."""
    print("\n" + "=" * 60)
    print("Example 2: Using BacktestRunner for More Control")
    print("=" * 60)
    
    config_path = "EXAMPLE_CONFIGS/run_config.yaml"
    
    # Load config
    config = RunConfig.from_yaml(config_path)
    
    # Create runner
    runner = BacktestRunner(config)
    
    # Initialize (loads data and creates engine)
    print("\nInitializing runner (loading data, creating engine)...")
    runner.initialize()
    
    # Run backtest for different periods
    print("\nRunning train period backtest...")
    train_results = runner.run_backtest(period="train")
    
    print("\nRunning validation period backtest...")
    validation_results = runner.run_backtest(period="validation")
    
    # Save results
    print("\nSaving results...")
    train_output = runner.save_results(train_results, period="train")
    validation_output = runner.save_results(validation_results, period="validation")
    
    print(f"\nTrain results saved to: {train_output}")
    print(f"Validation results saved to: {validation_output}")
    
    # Compute detailed metrics
    print("\nComputing detailed metrics...")
    portfolio = runner.engine.portfolio
    daily_events = runner.engine.daily_events
    closed_trades = runner.engine.closed_trades
    
    # Extract equity curve and returns
    dates = [event['date'] for event in daily_events]
    equity_curve = []
    daily_returns = []
    for event in daily_events:
        portfolio_state = event.get('portfolio_state', {})
        equity = portfolio_state.get('equity', event.get('equity', config.portfolio.starting_equity))
        equity_curve.append(equity)
        daily_returns.append(event.get('return', 0))
    
    # Compute from equity curve if returns are zeros
    if all(r == 0 for r in daily_returns) and len(equity_curve) > 1:
        daily_returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i-1] > 0:
                daily_returns.append((equity_curve[i] / equity_curve[i-1]) - 1.0)
            else:
                daily_returns.append(0.0)
        daily_returns.insert(0, 0.0)
    
    # Get benchmark returns
    benchmark_returns = runner._extract_benchmark_returns(dates, "SPY")
    
    # Calculate metrics
    metrics_calc = MetricsCalculator(
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        closed_trades=closed_trades,
        dates=dates,
        benchmark_returns=benchmark_returns
    )
    
    all_metrics = metrics_calc.compute_all_metrics()
    
    print("\nDetailed Metrics:")
    for metric_name, value in all_metrics.items():
        if isinstance(value, float):
            if 'ratio' in metric_name.lower() or 'factor' in metric_name.lower():
                print(f"  {metric_name}: {value:.2f}")
            elif 'return' in metric_name.lower() or 'drawdown' in metric_name.lower():
                print(f"  {metric_name}: {value:.2%}")
            else:
                print(f"  {metric_name}: {value:.4f}")
    
    return train_results, validation_results


def example_cli_usage():
    """Example: Show CLI usage (commented out - run manually)."""
    print("\n" + "=" * 60)
    print("Example 3: CLI Usage")
    print("=" * 60)
    
    print("""
To run a backtest from the command line:

    # Run train period
    python -m trading_system backtest \\
        --config EXAMPLE_CONFIGS/run_config.yaml \\
        --period train

    # Run validation period
    python -m trading_system backtest \\
        --config EXAMPLE_CONFIGS/run_config.yaml \\
        --period validation

    # Run holdout period
    python -m trading_system backtest \\
        --config EXAMPLE_CONFIGS/run_config.yaml \\
        --period holdout

Results will be saved to: results/{run_id}/{period}/
    """)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Basic Backtest Examples")
    print("=" * 60)
    
    try:
        # Example 1: Simple programmatic backtest
        example_programmatic_backtest()
        
        # Example 2: More control with BacktestRunner
        example_runner_backtest()
        
        # Example 3: CLI usage
        example_cli_usage()
        
        print("\n" + "=" * 60)
        print("Examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


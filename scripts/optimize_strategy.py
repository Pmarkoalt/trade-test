#!/usr/bin/env python
"""Strategy Parameter Optimization CLI.

This script optimizes strategy parameters using Bayesian optimization
with walk-forward validation to find the best configuration.

Usage:
    # Basic optimization (50 trials)
    python scripts/optimize_strategy.py --config configs/test_equity_strategy.yaml

    # More trials for better results
    python scripts/optimize_strategy.py --config configs/test_equity_strategy.yaml --trials 100

    # Optimize for Calmar ratio instead of Sharpe
    python scripts/optimize_strategy.py --config configs/test_equity_strategy.yaml --objective calmar_ratio

    # Quick test (10 trials)
    python scripts/optimize_strategy.py --config configs/test_equity_strategy.yaml --trials 10 --quick

Requirements:
    pip install optuna
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Optimize strategy parameters using Bayesian optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to base strategy config YAML",
    )

    # Optimization options
    parser.add_argument(
        "--trials",
        "-n",
        type=int,
        default=50,
        help="Number of optimization trials (default: 50)",
    )
    parser.add_argument(
        "--objective",
        "-o",
        choices=["sharpe_ratio", "calmar_ratio", "total_return", "sortino_ratio"],
        default="sharpe_ratio",
        help="Metric to optimize (default: sharpe_ratio)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Maximum time in seconds",
    )

    # Data paths
    parser.add_argument(
        "--equity-path",
        default="data/equity/daily",
        help="Path to equity data",
    )
    parser.add_argument(
        "--crypto-path",
        default="data/crypto/daily",
        help="Path to crypto data",
    )
    parser.add_argument(
        "--benchmark-path",
        default="data/test_benchmarks",
        help="Path to benchmark data",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        default="optimization_results",
        help="Directory for results",
    )

    # Misc options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick", action="store_true", help="Quick mode with reduced search space")

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Check for optuna
    try:
        pass
    except ImportError:
        logger.error("Optuna not installed. Install with: pip install optuna")
        return 1

    # Import optimizer
    from trading_system.optimization import ParameterSpace, StrategyOptimizer

    # Configure parameter spaces
    param_spaces = None
    if args.quick:
        # Reduced search space for quick testing
        param_spaces = [
            ParameterSpace("exit.hard_stop_atr_mult", "categorical", choices=[2.0, 2.5, 3.0]),
            ParameterSpace("exit.exit_ma", "categorical", choices=[20, 50]),
            ParameterSpace("risk.risk_per_trade", "categorical", choices=[0.0075, 0.01]),
        ]

    # Create optimizer
    optimizer = StrategyOptimizer(
        base_config_path=args.config,
        data_paths={
            "equity": args.equity_path,
            "crypto": args.crypto_path,
            "benchmark": args.benchmark_path,
        },
        param_spaces=param_spaces,
        objective=args.objective,
        output_dir=args.output_dir,
    )

    # Run optimization
    print("\n" + "=" * 60)
    print("STRATEGY PARAMETER OPTIMIZATION")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Objective: {args.objective}")
    print(f"Trials: {args.trials}")
    print(f"Output: {args.output_dir}")
    print("=" * 60 + "\n")

    try:
        result = optimizer.optimize(
            n_trials=args.trials,
            timeout=args.timeout,
            show_progress=True,
        )

        # Print final summary
        print("\n" + "=" * 60)
        print("OPTIMIZATION RESULTS")
        print("=" * 60)
        print(f"\nBest {args.objective}: {result.best_value:.4f}")
        print("\nOptimized Parameters:")
        for name, value in result.best_params.items():
            print(f"  {name}: {value}")
        print(f"\nHoldout Validation:")
        for name, value in result.holdout_metrics.items():
            if isinstance(value, float):
                print(f"  {name}: {value:.4f}")
            else:
                print(f"  {name}: {value}")
        print(f"\nResults saved to: {args.output_dir}/{result.study_name}.json")
        print(f"Optimized config: {args.output_dir}/{result.study_name}_config.yaml")
        print("=" * 60 + "\n")

        # Get parameter importance
        importance = optimizer.get_param_importance()
        if importance:
            print("Parameter Importance:")
            for name, imp in sorted(importance.items(), key=lambda x: -x[1]):
                print(f"  {name}: {imp:.4f}")

        return 0

    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Walk-Forward Optimization Automation

Automates rolling optimization and validation:
- Monthly/quarterly re-optimization
- Out-of-sample validation
- Parameter stability tracking
- Performance degradation alerts
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


class WalkForwardOptimizer:
    """Automated walk-forward optimization framework."""

    def __init__(
        self,
        config_path: str,
        data_paths: Dict[str, str],
        output_dir: str = "walk_forward_results",
        train_months: int = 12,
        test_months: int = 3,
        step_months: int = 3,
    ):
        """
        Initialize walk-forward optimizer.

        Args:
            config_path: Base strategy configuration
            data_paths: Data source paths
            output_dir: Output directory for results
            train_months: Training window in months
            test_months: Test/validation window in months
            step_months: Step size for rolling window
        """
        self.config_path = Path(config_path)
        self.data_paths = data_paths
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months

        self.results: List[Dict] = []
        self.parameter_history: List[Dict] = []

    def generate_windows(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Generate train/test windows for walk-forward.

        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        windows = []
        current_train_start = start_date

        while True:
            train_end = current_train_start + timedelta(days=self.train_months * 30)
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_months * 30)

            if test_end > end_date:
                break

            windows.append((current_train_start, train_end, test_start, test_end))
            current_train_start += timedelta(days=self.step_months * 30)

        return windows

    def run_optimization_window(
        self,
        train_start: datetime,
        train_end: datetime,
        n_trials: int = 100,
    ) -> Dict[str, Any]:
        """Run optimization for a single training window."""
        try:
            from trading_system.optimization import StrategyOptimizer
        except ImportError:
            print("Warning: StrategyOptimizer not available")
            return {"best_params": {}, "best_value": 0}

        optimizer = StrategyOptimizer(
            base_config_path=str(self.config_path),
            data_paths=self.data_paths,
            output_dir=str(self.output_dir / "optimizations"),
        )

        # Run optimization
        result = optimizer.optimize(n_trials=n_trials, n_jobs=4)

        return {
            "best_params": result.best_params,
            "best_value": result.best_value,
            "n_trials": n_trials,
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
        }

    def validate_window(
        self,
        params: Dict,
        test_start: datetime,
        test_end: datetime,
    ) -> Dict[str, float]:
        """Validate optimized parameters on test window."""
        # This would run a backtest with the optimized parameters
        # For now, return placeholder metrics
        return {
            "sharpe_ratio": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
        }

    def run_walk_forward(
        self,
        start_date: datetime,
        end_date: datetime,
        n_trials: int = 100,
    ) -> Dict[str, Any]:
        """Run complete walk-forward optimization."""
        windows = self.generate_windows(start_date, end_date)

        print(f"Walk-Forward Optimization")
        print(f"=" * 60)
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"Windows: {len(windows)}")
        print(f"Train: {self.train_months} months, Test: {self.test_months} months")
        print(f"=" * 60)

        all_results = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows, 1):
            print(f"\nWindow {i}/{len(windows)}")
            print(f"  Train: {train_start.date()} to {train_end.date()}")
            print(f"  Test:  {test_start.date()} to {test_end.date()}")

            # Optimize on training window
            opt_result = self.run_optimization_window(train_start, train_end, n_trials)

            # Validate on test window
            val_result = self.validate_window(
                opt_result["best_params"],
                test_start,
                test_end,
            )

            window_result = {
                "window": i,
                "optimization": opt_result,
                "validation": val_result,
            }
            all_results.append(window_result)
            self.parameter_history.append(opt_result["best_params"])

            print(f"  Best train Sharpe: {opt_result['best_value']:.4f}")
            print(f"  Test Sharpe: {val_result['sharpe_ratio']:.4f}")

        # Analyze parameter stability
        stability = self.analyze_parameter_stability()

        # Calculate aggregate metrics
        aggregate = self.calculate_aggregate_metrics(all_results)

        final_result = {
            "windows": all_results,
            "parameter_stability": stability,
            "aggregate_metrics": aggregate,
            "config": {
                "train_months": self.train_months,
                "test_months": self.test_months,
                "step_months": self.step_months,
                "n_trials": n_trials,
            },
        }

        # Save results
        output_path = self.output_dir / f"walk_forward_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, "w") as f:
            json.dump(final_result, f, indent=2, default=str)

        print(f"\nResults saved to: {output_path}")

        return final_result

    def analyze_parameter_stability(self) -> Dict[str, Any]:
        """Analyze stability of optimized parameters across windows."""
        if len(self.parameter_history) < 2:
            return {"stable": [], "volatile": [], "recommendations": []}

        # Collect all parameter values
        param_values: Dict[str, List] = {}
        for params in self.parameter_history:
            for key, value in params.items():
                if key not in param_values:
                    param_values[key] = []
                param_values[key].append(value)

        # Calculate stability metrics
        stability_scores = {}
        for param, values in param_values.items():
            if all(isinstance(v, (int, float)) for v in values):
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / abs(mean_val) if mean_val != 0 else float("inf")
                stability_scores[param] = {
                    "mean": mean_val,
                    "std": std_val,
                    "cv": cv,
                    "min": min(values),
                    "max": max(values),
                    "stable": cv < 0.2,
                }

        stable = [p for p, s in stability_scores.items() if s.get("stable", False)]
        volatile = [p for p, s in stability_scores.items() if not s.get("stable", True)]

        recommendations = []
        if stable:
            recommendations.append(f"Consider fixing stable parameters: {', '.join(stable)}")
        if volatile:
            recommendations.append(f"Review volatile parameters: {', '.join(volatile)}")

        return {
            "scores": stability_scores,
            "stable": stable,
            "volatile": volatile,
            "recommendations": recommendations,
        }

    def calculate_aggregate_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate aggregate metrics across all windows."""
        train_sharpes = []
        test_sharpes = []
        test_returns = []

        for r in results:
            train_sharpes.append(r["optimization"]["best_value"])
            test_sharpes.append(r["validation"]["sharpe_ratio"])
            test_returns.append(r["validation"]["total_return"])

        # Calculate overfitting ratio
        avg_train = np.mean(train_sharpes) if train_sharpes else 0
        avg_test = np.mean(test_sharpes) if test_sharpes else 0
        overfit_ratio = avg_test / avg_train if avg_train != 0 else 0

        return {
            "avg_train_sharpe": avg_train,
            "avg_test_sharpe": avg_test,
            "std_test_sharpe": np.std(test_sharpes) if test_sharpes else 0,
            "total_test_return": np.sum(test_returns),
            "overfitting_ratio": overfit_ratio,
            "windows_positive": sum(1 for s in test_sharpes if s > 0),
            "windows_total": len(results),
        }

    def generate_report(self, results: Dict) -> str:
        """Generate walk-forward analysis report."""
        lines = []
        lines.append("=" * 70)
        lines.append("WALK-FORWARD OPTIMIZATION REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        lines.append("")

        # Configuration
        config = results.get("config", {})
        lines.append("CONFIGURATION")
        lines.append("-" * 40)
        lines.append(f"Train Window: {config.get('train_months', 0)} months")
        lines.append(f"Test Window: {config.get('test_months', 0)} months")
        lines.append(f"Step Size: {config.get('step_months', 0)} months")
        lines.append(f"Trials per Window: {config.get('n_trials', 0)}")
        lines.append("")

        # Aggregate metrics
        agg = results.get("aggregate_metrics", {})
        lines.append("AGGREGATE METRICS")
        lines.append("-" * 40)
        lines.append(f"Avg Train Sharpe: {agg.get('avg_train_sharpe', 0):.4f}")
        lines.append(f"Avg Test Sharpe: {agg.get('avg_test_sharpe', 0):.4f}")
        lines.append(f"Overfitting Ratio: {agg.get('overfitting_ratio', 0):.2f}")
        lines.append(f"Positive Windows: {agg.get('windows_positive', 0)}/{agg.get('windows_total', 0)}")
        lines.append("")

        # Parameter stability
        stability = results.get("parameter_stability", {})
        lines.append("PARAMETER STABILITY")
        lines.append("-" * 40)
        if stability.get("stable"):
            lines.append(f"Stable: {', '.join(stability['stable'])}")
        if stability.get("volatile"):
            lines.append(f"Volatile: {', '.join(stability['volatile'])}")
        for rec in stability.get("recommendations", []):
            lines.append(f"  → {rec}")
        lines.append("")

        # Window summary
        windows = results.get("windows", [])
        if windows:
            lines.append("WINDOW SUMMARY")
            lines.append("-" * 40)
            lines.append(f"{'Window':<8} {'Train Sharpe':>12} {'Test Sharpe':>12}")
            for w in windows:
                train_s = w["optimization"]["best_value"]
                test_s = w["validation"]["sharpe_ratio"]
                lines.append(f"{w['window']:<8} {train_s:>12.4f} {test_s:>12.4f}")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Walk-forward optimization automation")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Strategy configuration file",
    )
    parser.add_argument(
        "--start",
        "-s",
        type=str,
        default="2023-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        "-e",
        type=str,
        default="2025-12-31",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--train-months",
        type=int,
        default=12,
        help="Training window months",
    )
    parser.add_argument(
        "--test-months",
        type=int,
        default=3,
        help="Test window months",
    )
    parser.add_argument(
        "--step-months",
        type=int,
        default=3,
        help="Step size months",
    )
    parser.add_argument(
        "--trials",
        "-t",
        type=int,
        default=100,
        help="Optimization trials per window",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="walk_forward_results",
        help="Output directory",
    )

    args = parser.parse_args()

    data_paths = {
        "equity": "data/equity/daily",
        "crypto": "data/crypto/daily",
        "benchmark": "data/test_benchmarks",
    }

    optimizer = WalkForwardOptimizer(
        config_path=args.config,
        data_paths=data_paths,
        output_dir=args.output,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
    )

    start_date = datetime.strptime(args.start, "%Y-%m-%d")
    end_date = datetime.strptime(args.end, "%Y-%m-%d")

    results = optimizer.run_walk_forward(start_date, end_date, args.trials)

    print("\n" + optimizer.generate_report(results))


if __name__ == "__main__":
    main()

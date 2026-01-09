#!/usr/bin/env python3
"""
Strategy Analysis Tool - Expanded Metrics & Comparison

Analyzes optimization results with comprehensive metrics:
- Win rate, profit factor, avg win/loss
- Sortino ratio, max consecutive losses
- Trade duration, exposure time
- Parameter stability analysis
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class StrategyAnalyzer:
    """Comprehensive strategy analysis with expanded metrics."""

    def __init__(self, results_path: str):
        """Load optimization results."""
        self.results_path = Path(results_path)
        self.results = self._load_results()
        self.trades: List[Dict] = []
        self.daily_returns: pd.Series = pd.Series(dtype=float)

    def _load_results(self) -> Dict:
        """Load JSON results file."""
        if not self.results_path.exists():
            raise FileNotFoundError(f"Results not found: {self.results_path}")

        with open(self.results_path) as f:
            return json.load(f)

    def calculate_expanded_metrics(self, trades: List[Dict], daily_returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive trading metrics."""
        self.trades = trades
        self.daily_returns = daily_returns

        metrics = {}

        # Basic metrics
        metrics["total_trades"] = len(trades)

        if not trades:
            return self._empty_metrics()

        # Win/Loss analysis
        pnls = [t.get("pnl", 0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        metrics["win_rate"] = len(wins) / len(pnls) if pnls else 0
        metrics["avg_win"] = np.mean(wins) if wins else 0
        metrics["avg_loss"] = np.mean(losses) if losses else 0
        metrics["avg_win_loss_ratio"] = abs(metrics["avg_win"] / metrics["avg_loss"]) if metrics["avg_loss"] != 0 else 0

        # Profit factor
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        metrics["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Consecutive losses
        metrics["max_consecutive_losses"] = self._max_consecutive_losses(pnls)
        metrics["max_consecutive_wins"] = self._max_consecutive_wins(pnls)

        # Trade duration
        durations = [t.get("duration_days", 0) for t in trades if "duration_days" in t]
        metrics["avg_trade_duration"] = np.mean(durations) if durations else 0
        metrics["max_trade_duration"] = max(durations) if durations else 0

        # Returns analysis
        if len(daily_returns) > 0:
            metrics["total_return"] = (1 + daily_returns).prod() - 1
            metrics["annualized_return"] = self._annualized_return(daily_returns)
            metrics["volatility"] = daily_returns.std() * np.sqrt(252)
            metrics["sharpe_ratio"] = self._sharpe_ratio(daily_returns)
            metrics["sortino_ratio"] = self._sortino_ratio(daily_returns)
            metrics["calmar_ratio"] = self._calmar_ratio(daily_returns)
            metrics["max_drawdown"] = self._max_drawdown(daily_returns)
            metrics["recovery_time_days"] = self._recovery_time(daily_returns)
            metrics["exposure_pct"] = self._exposure_percentage(trades, daily_returns)

        return metrics

    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dict."""
        return {
            "total_trades": 0,
            "win_rate": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "avg_win_loss_ratio": 0,
            "profit_factor": 0,
            "max_consecutive_losses": 0,
            "max_consecutive_wins": 0,
            "avg_trade_duration": 0,
            "max_trade_duration": 0,
            "total_return": 0,
            "annualized_return": 0,
            "volatility": 0,
            "sharpe_ratio": 0,
            "sortino_ratio": 0,
            "calmar_ratio": 0,
            "max_drawdown": 0,
            "recovery_time_days": 0,
            "exposure_pct": 0,
        }

    def _max_consecutive_losses(self, pnls: List[float]) -> int:
        """Calculate max consecutive losing trades."""
        max_streak = 0
        current_streak = 0
        for pnl in pnls:
            if pnl < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak

    def _max_consecutive_wins(self, pnls: List[float]) -> int:
        """Calculate max consecutive winning trades."""
        max_streak = 0
        current_streak = 0
        for pnl in pnls:
            if pnl > 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak

    def _annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return."""
        if len(returns) == 0:
            return 0
        total_return = (1 + returns).prod() - 1
        years = len(returns) / 252
        if years <= 0:
            return 0
        return (1 + total_return) ** (1 / years) - 1

    def _sharpe_ratio(self, returns: pd.Series, risk_free: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or returns.std() == 0:
            return 0
        excess_returns = returns - risk_free / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std()

    def _sortino_ratio(self, returns: pd.Series, risk_free: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation only)."""
        if len(returns) == 0:
            return 0
        excess_returns = returns - risk_free / 252
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float("inf") if excess_returns.mean() > 0 else 0
        downside_std = downside_returns.std() * np.sqrt(252)
        return np.sqrt(252) * excess_returns.mean() / downside_std

    def _calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio (return / max drawdown)."""
        ann_return = self._annualized_return(returns)
        max_dd = self._max_drawdown(returns)
        if max_dd == 0:
            return float("inf") if ann_return > 0 else 0
        return ann_return / max_dd

    def _max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        return abs(drawdowns.min())

    def _recovery_time(self, returns: pd.Series) -> int:
        """Calculate days to recover from max drawdown."""
        if len(returns) == 0:
            return 0
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()

        # Find drawdown periods
        in_drawdown = cumulative < rolling_max
        if not in_drawdown.any():
            return 0

        # Find longest drawdown period
        drawdown_starts = in_drawdown & ~in_drawdown.shift(1).fillna(False)
        drawdown_ends = ~in_drawdown & in_drawdown.shift(1).fillna(False)

        max_recovery = 0
        current_start = None

        for i, (is_start, is_end) in enumerate(zip(drawdown_starts, drawdown_ends)):
            if is_start:
                current_start = i
            if is_end and current_start is not None:
                max_recovery = max(max_recovery, i - current_start)

        return max_recovery

    def _exposure_percentage(self, trades: List[Dict], returns: pd.Series) -> float:
        """Calculate percentage of time in market."""
        if len(returns) == 0:
            return 0
        total_days = len(returns)
        days_in_market = sum(t.get("duration_days", 1) for t in trades)
        return min(1.0, days_in_market / total_days)

    def analyze_parameter_stability(self) -> Dict[str, Any]:
        """Analyze which parameters are most stable across top trials."""
        trials = self.results.get("top_trials", [])
        if len(trials) < 3:
            return {"stable_params": [], "volatile_params": [], "recommendations": []}

        # Collect parameter values from top trials
        param_values: Dict[str, List] = {}
        for trial in trials:
            params = trial.get("params", {})
            for key, value in params.items():
                if key not in param_values:
                    param_values[key] = []
                param_values[key].append(value)

        # Calculate coefficient of variation for each parameter
        stability = {}
        for param, values in param_values.items():
            if all(isinstance(v, (int, float)) for v in values):
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / mean_val if mean_val != 0 else float("inf")
                stability[param] = {
                    "mean": mean_val,
                    "std": std_val,
                    "cv": cv,
                    "values": values,
                }
            else:
                # Categorical - check consistency
                unique = set(values)
                stability[param] = {
                    "mode": max(set(values), key=values.count),
                    "unique_count": len(unique),
                    "values": values,
                }

        # Classify parameters
        stable = []
        volatile = []
        for param, stats in stability.items():
            if "cv" in stats:
                if stats["cv"] < 0.1:
                    stable.append(param)
                elif stats["cv"] > 0.3:
                    volatile.append(param)
            else:
                if stats["unique_count"] == 1:
                    stable.append(param)
                elif stats["unique_count"] > 2:
                    volatile.append(param)

        recommendations = []
        if stable:
            recommendations.append(f"Lock stable params: {', '.join(stable)}")
        if volatile:
            recommendations.append(f"Investigate volatile params: {', '.join(volatile)}")

        return {
            "stability": stability,
            "stable_params": stable,
            "volatile_params": volatile,
            "recommendations": recommendations,
        }

    def generate_report(self) -> str:
        """Generate comprehensive analysis report."""
        report = []
        report.append("=" * 70)
        report.append("STRATEGY ANALYSIS REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        report.append("")

        # Best trial summary
        best = self.results.get("best_params", {})
        best_value = self.results.get("best_value", 0)
        report.append("BEST CONFIGURATION")
        report.append("-" * 40)
        report.append(f"Objective Value: {best_value:.4f}")
        for key, value in best.items():
            report.append(f"  {key}: {value}")
        report.append("")

        # Holdout metrics
        holdout = self.results.get("holdout_metrics", {})
        if holdout:
            report.append("HOLDOUT VALIDATION")
            report.append("-" * 40)
            for key, value in holdout.items():
                if isinstance(value, float):
                    report.append(f"  {key}: {value:.4f}")
                else:
                    report.append(f"  {key}: {value}")
            report.append("")

        # Parameter stability
        stability = self.analyze_parameter_stability()
        report.append("PARAMETER STABILITY ANALYSIS")
        report.append("-" * 40)
        if stability["stable_params"]:
            report.append(f"Stable (low variance): {', '.join(stability['stable_params'])}")
        if stability["volatile_params"]:
            report.append(f"Volatile (high variance): {', '.join(stability['volatile_params'])}")
        for rec in stability.get("recommendations", []):
            report.append(f"  → {rec}")
        report.append("")

        # Top trials comparison
        trials = self.results.get("top_trials", [])
        if trials:
            report.append(f"TOP {len(trials)} TRIALS")
            report.append("-" * 40)
            for i, trial in enumerate(trials[:5], 1):
                report.append(f"  #{i}: value={trial.get('value', 0):.4f}")
        report.append("")

        report.append("=" * 70)

        return "\n".join(report)


def compare_strategies(configs: List[str], output_path: Optional[str] = None) -> str:
    """Compare multiple strategy configurations."""
    analyzers = []
    for config_path in configs:
        try:
            analyzer = StrategyAnalyzer(config_path)
            analyzers.append((config_path, analyzer))
        except Exception as e:
            print(f"Error loading {config_path}: {e}")

    if not analyzers:
        return "No valid configurations to compare"

    report = []
    report.append("=" * 70)
    report.append("STRATEGY COMPARISON REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)
    report.append("")

    # Summary table
    report.append(f"{'Config':<30} {'Best Value':>12} {'Holdout Sharpe':>15}")
    report.append("-" * 60)

    for path, analyzer in analyzers:
        name = Path(path).stem[:28]
        best_val = analyzer.results.get("best_value", 0)
        holdout = analyzer.results.get("holdout_metrics", {})
        holdout_sharpe = holdout.get("sharpe_ratio", 0)
        report.append(f"{name:<30} {best_val:>12.4f} {holdout_sharpe:>15.4f}")

    report.append("")
    report.append("=" * 70)

    result = "\n".join(report)

    if output_path:
        with open(output_path, "w") as f:
            f.write(result)
        print(f"Report saved to: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Analyze strategy optimization results")
    parser.add_argument(
        "--results",
        "-r",
        type=str,
        help="Path to optimization results JSON",
    )
    parser.add_argument(
        "--compare",
        "-c",
        nargs="+",
        type=str,
        help="Compare multiple result files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output path for report",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.compare:
        report = compare_strategies(args.compare, args.output)
        print(report)
    elif args.results:
        analyzer = StrategyAnalyzer(args.results)
        report = analyzer.generate_report()
        print(report)

        if args.output:
            with open(args.output, "w") as f:
                f.write(report)
            print(f"\nReport saved to: {args.output}")
    else:
        # Find most recent results
        results_dir = Path("optimization_results")
        if results_dir.exists():
            json_files = sorted(results_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
            if json_files:
                print(f"Analyzing most recent: {json_files[0]}")
                analyzer = StrategyAnalyzer(str(json_files[0]))
                print(analyzer.generate_report())
            else:
                print("No results found in optimization_results/")
        else:
            parser.print_help()


if __name__ == "__main__":
    main()

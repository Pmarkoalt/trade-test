#!/usr/bin/env python3
"""
Strategy Comparison Tool

Compare multiple strategy configurations side-by-side with:
- Performance metrics comparison
- Parameter difference analysis
- HTML report generation
- Recommendation engine
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


class StrategyComparator:
    """Compare multiple strategy configurations."""

    def __init__(self):
        self.strategies: List[Dict] = []
        self.names: List[str] = []

    def add_strategy(self, name: str, results_path: str) -> None:
        """Add a strategy result for comparison."""
        path = Path(results_path)
        if not path.exists():
            print(f"Warning: {results_path} not found")
            return

        with open(path) as f:
            data = json.load(f)

        self.strategies.append(data)
        self.names.append(name)

    def compare_metrics(self) -> pd.DataFrame:
        """Compare key metrics across strategies."""
        rows = []

        for name, strategy in zip(self.names, self.strategies):
            holdout = strategy.get("holdout_metrics", {})
            strategy.get("best_params", {})

            row = {
                "Strategy": name,
                "Best Sharpe": strategy.get("best_value", 0),
                "Holdout Sharpe": holdout.get("sharpe_ratio", 0),
                "Holdout Return": holdout.get("total_return", 0),
                "Holdout Trades": holdout.get("total_trades", 0),
                "Max Drawdown": holdout.get("max_drawdown", 0),
                "Trials": strategy.get("n_completed", 0),
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def compare_parameters(self) -> pd.DataFrame:
        """Compare best parameters across strategies."""
        all_params = set()
        for strategy in self.strategies:
            all_params.update(strategy.get("best_params", {}).keys())

        rows = []
        for param in sorted(all_params):
            row = {"Parameter": param}
            for name, strategy in zip(self.names, self.strategies):
                value = strategy.get("best_params", {}).get(param, "N/A")
                if isinstance(value, float):
                    row[name] = f"{value:.4f}"
                else:
                    row[name] = str(value)
            rows.append(row)

        return pd.DataFrame(rows)

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on comparison."""
        recommendations = []

        if len(self.strategies) < 2:
            return ["Add more strategies for comparison"]

        # Find best performer
        metrics_df = self.compare_metrics()

        # Best holdout Sharpe
        best_holdout_idx = metrics_df["Holdout Sharpe"].idxmax()
        best_holdout = metrics_df.loc[best_holdout_idx, "Strategy"]
        recommendations.append(f"Best holdout performance: {best_holdout}")

        # Check for overfitting
        for _, row in metrics_df.iterrows():
            train_sharpe = row["Best Sharpe"]
            holdout_sharpe = row["Holdout Sharpe"]
            if train_sharpe > 0 and holdout_sharpe < train_sharpe * 0.3:
                recommendations.append(
                    f"⚠️ {row['Strategy']} shows overfitting (train: {train_sharpe:.2f}, holdout: {holdout_sharpe:.2f})"
                )

        # Check for low trade count
        for _, row in metrics_df.iterrows():
            if row["Holdout Trades"] < 10:
                recommendations.append(
                    f"⚠️ {row['Strategy']} has low trade count ({row['Holdout Trades']}) - may not be statistically significant"
                )

        # Best risk-adjusted
        valid_calmar = metrics_df[metrics_df["Max Drawdown"] > 0]
        if len(valid_calmar) > 0:
            calmar = valid_calmar["Holdout Return"] / valid_calmar["Max Drawdown"]
            best_calmar_idx = calmar.idxmax()
            best_calmar = metrics_df.loc[best_calmar_idx, "Strategy"]
            recommendations.append(f"Best risk-adjusted (Calmar): {best_calmar}")

        return recommendations

    def generate_html_report(self, output_path: str) -> None:
        """Generate HTML comparison report."""
        metrics_df = self.compare_metrics()
        params_df = self.compare_parameters()
        recommendations = self.generate_recommendations()

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Strategy Comparison Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #3498db; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        tr:hover {{ background: #f1f1f1; }}
        .positive {{ color: #27ae60; font-weight: bold; }}
        .negative {{ color: #e74c3c; font-weight: bold; }}
        .warning {{ background: #fff3cd; padding: 10px; border-radius: 4px; margin: 5px 0; }}
        .recommendation {{ background: #d4edda; padding: 10px; border-radius: 4px; margin: 5px 0; }}
        .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Strategy Comparison Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>Performance Metrics</h2>
        {metrics_df.to_html(index=False, classes='metrics-table')}

        <h2>Parameter Comparison</h2>
        {params_df.to_html(index=False, classes='params-table')}

        <h2>Recommendations</h2>
        {''.join(f'<div class="{"warning" if "⚠️" in r else "recommendation"}">{r}</div>' for r in recommendations)}
    </div>
</body>
</html>
"""

        with open(output_path, "w") as f:
            f.write(html)

        print(f"HTML report saved to: {output_path}")

    def generate_text_report(self) -> str:
        """Generate text comparison report."""
        lines = []
        lines.append("=" * 70)
        lines.append("STRATEGY COMPARISON REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        lines.append("")

        # Metrics comparison
        lines.append("PERFORMANCE METRICS")
        lines.append("-" * 70)
        metrics_df = self.compare_metrics()
        lines.append(metrics_df.to_string(index=False))
        lines.append("")

        # Parameters comparison
        lines.append("PARAMETER COMPARISON")
        lines.append("-" * 70)
        params_df = self.compare_parameters()
        lines.append(params_df.to_string(index=False))
        lines.append("")

        # Recommendations
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 70)
        for rec in self.generate_recommendations():
            lines.append(f"  • {rec}")
        lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare strategy configurations")
    parser.add_argument(
        "--strategies",
        "-s",
        nargs="+",
        type=str,
        required=True,
        help="Strategy result files (name:path format or just paths)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output path for report",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML report",
    )

    args = parser.parse_args()

    comparator = StrategyComparator()

    for strategy in args.strategies:
        if ":" in strategy:
            name, path = strategy.split(":", 1)
        else:
            name = Path(strategy).stem
            path = strategy
        comparator.add_strategy(name, path)

    if args.html and args.output:
        comparator.generate_html_report(args.output)
    else:
        report = comparator.generate_text_report()
        print(report)

        if args.output:
            with open(args.output, "w") as f:
                f.write(report)
            print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()

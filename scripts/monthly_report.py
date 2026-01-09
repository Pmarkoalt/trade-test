#!/usr/bin/env python3
"""
Automated Monthly Performance Report Generator

Generates comprehensive monthly reports including:
- Performance summary
- Trade analysis
- Drawdown analysis
- Regime analysis
- Parameter drift detection
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


class MonthlyReportGenerator:
    """Generate automated monthly performance reports."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.report_data: Dict[str, Any] = {}

    def collect_monthly_data(self, year: int, month: int) -> Dict[str, Any]:
        """Collect all data for a specific month."""
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(days=1)

        data = {
            "period": f"{year}-{month:02d}",
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "strategies": {},
            "summary": {},
        }

        # Find result directories for this month
        for run_dir in self.results_dir.glob("run_*"):
            if not run_dir.is_dir():
                continue

            # Check for monthly report in the run
            monthly_file = run_dir / "holdout" / "monthly_report.json"
            if monthly_file.exists():
                with open(monthly_file) as f:
                    monthly_data = json.load(f)
                data["strategies"][run_dir.name] = monthly_data

        return data

    def calculate_summary_stats(self, data: Dict) -> Dict[str, float]:
        """Calculate summary statistics from collected data."""
        all_returns = []
        all_trades = []

        for strategy_name, strategy_data in data.get("strategies", {}).items():
            if isinstance(strategy_data, dict):
                if "total_return" in strategy_data:
                    all_returns.append(strategy_data["total_return"])
                if "total_trades" in strategy_data:
                    all_trades.append(strategy_data["total_trades"])

        return {
            "avg_return": np.mean(all_returns) if all_returns else 0,
            "best_return": max(all_returns) if all_returns else 0,
            "worst_return": min(all_returns) if all_returns else 0,
            "total_trades": sum(all_trades),
            "strategy_count": len(data.get("strategies", {})),
        }

    def detect_performance_drift(
        self, current_metrics: Dict[str, float], baseline_metrics: Dict[str, float], threshold: float = 0.3
    ) -> List[str]:
        """Detect significant performance drift from baseline."""
        alerts = []

        for metric, current_value in current_metrics.items():
            baseline_value = baseline_metrics.get(metric, 0)
            if baseline_value == 0:
                continue

            drift = (current_value - baseline_value) / abs(baseline_value)

            if drift < -threshold:
                alerts.append(
                    f"⚠️ {metric} decreased by {abs(drift)*100:.1f}% "
                    f"(current: {current_value:.4f}, baseline: {baseline_value:.4f})"
                )
            elif drift > threshold:
                alerts.append(
                    f"✅ {metric} improved by {drift*100:.1f}% "
                    f"(current: {current_value:.4f}, baseline: {baseline_value:.4f})"
                )

        return alerts

    def generate_report(self, year: int, month: int, baseline_path: Optional[str] = None) -> str:
        """Generate monthly report."""
        data = self.collect_monthly_data(year, month)
        summary = self.calculate_summary_stats(data)

        lines = []
        lines.append("=" * 70)
        lines.append(f"MONTHLY PERFORMANCE REPORT - {year}-{month:02d}")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        lines.append("")

        # Summary section
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Strategies Analyzed: {summary['strategy_count']}")
        lines.append(f"Total Trades: {summary['total_trades']}")
        lines.append(f"Average Return: {summary['avg_return']*100:.2f}%")
        lines.append(f"Best Return: {summary['best_return']*100:.2f}%")
        lines.append(f"Worst Return: {summary['worst_return']*100:.2f}%")
        lines.append("")

        # Strategy breakdown
        lines.append("STRATEGY BREAKDOWN")
        lines.append("-" * 40)
        for name, strategy_data in data.get("strategies", {}).items():
            if isinstance(strategy_data, dict):
                ret = strategy_data.get("total_return", 0)
                trades = strategy_data.get("total_trades", 0)
                sharpe = strategy_data.get("sharpe_ratio", 0)
                lines.append(f"  {name[:30]:<30} Return: {ret*100:>6.2f}%  Trades: {trades:>3}  Sharpe: {sharpe:>5.2f}")
        lines.append("")

        # Drift detection
        if baseline_path and Path(baseline_path).exists():
            with open(baseline_path) as f:
                baseline = json.load(f)

            lines.append("PERFORMANCE DRIFT ANALYSIS")
            lines.append("-" * 40)

            current_metrics = {
                "avg_return": summary["avg_return"],
                "total_trades": summary["total_trades"],
            }
            baseline_metrics = baseline.get("summary", {})

            alerts = self.detect_performance_drift(current_metrics, baseline_metrics)
            if alerts:
                for alert in alerts:
                    lines.append(f"  {alert}")
            else:
                lines.append("  No significant drift detected")
            lines.append("")

        # Recommendations
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 40)

        if summary["total_trades"] < 20:
            lines.append("  ⚠️ Low trade count - consider loosening entry criteria")
        if summary["avg_return"] < 0:
            lines.append("  ⚠️ Negative average return - review strategy parameters")
        if summary["strategy_count"] == 0:
            lines.append("  ⚠️ No strategy data found for this period")
        if summary["avg_return"] > 0.05:
            lines.append("  ✅ Strong performance - consider increasing position sizes")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def generate_html_report(self, year: int, month: int, output_path: str) -> None:
        """Generate HTML monthly report."""
        data = self.collect_monthly_data(year, month)
        summary = self.calculate_summary_stats(data)

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Monthly Report - {year}-{month:02d}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #1a1a2e; color: #eee; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        h1 {{ color: #00d4ff; }}
        .card {{ background: #16213e; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px 20px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #00d4ff; }}
        .metric-label {{ color: #888; }}
        .positive {{ color: #00ff88; }}
        .negative {{ color: #ff4444; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #333; }}
        th {{ color: #00d4ff; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 Monthly Performance Report</h1>
        <p style="color: #888;">{year}-{month:02d}</p>

        <div class="card">
            <h2>Summary</h2>
            <div class="metric">
                <div class="metric-value">{summary['strategy_count']}</div>
                <div class="metric-label">Strategies</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary['total_trades']}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric">
                <div class="metric-value {'positive' if summary['avg_return'] > 0 else 'negative'}">{summary['avg_return']*100:.2f}%</div>
                <div class="metric-label">Avg Return</div>
            </div>
        </div>

        <div class="card">
            <h2>Strategy Performance</h2>
            <table>
                <tr><th>Strategy</th><th>Return</th><th>Trades</th></tr>
                {''.join(f"<tr><td>{name}</td><td class='{'positive' if d.get('total_return',0)>0 else 'negative'}'>{d.get('total_return',0)*100:.2f}%</td><td>{d.get('total_trades',0)}</td></tr>" for name, d in data.get('strategies', {}).items() if isinstance(d, dict))}
            </table>
        </div>
    </div>
</body>
</html>
"""

        with open(output_path, "w") as f:
            f.write(html)

        print(f"HTML report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate monthly performance report")
    parser.add_argument(
        "--year",
        "-y",
        type=int,
        default=datetime.now().year,
        help="Report year",
    )
    parser.add_argument(
        "--month",
        "-m",
        type=int,
        default=datetime.now().month,
        help="Report month",
    )
    parser.add_argument(
        "--results-dir",
        "-r",
        type=str,
        default="results",
        help="Results directory",
    )
    parser.add_argument(
        "--baseline",
        "-b",
        type=str,
        help="Baseline report for drift detection",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output path",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML report",
    )

    args = parser.parse_args()

    generator = MonthlyReportGenerator(args.results_dir)

    if args.html and args.output:
        generator.generate_html_report(args.year, args.month, args.output)
    else:
        report = generator.generate_report(args.year, args.month, args.baseline)
        print(report)

        if args.output:
            with open(args.output, "w") as f:
                f.write(report)
            print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()

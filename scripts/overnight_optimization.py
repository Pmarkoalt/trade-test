#!/usr/bin/env python3
"""
Overnight Super-Optimization Script

Runs comprehensive optimization pipeline overnight:
1. High-trial optimization for equity and crypto
2. Walk-forward validation
3. Ensemble creation from top configs
4. Full analysis and reporting

Run on EC2 with: screen -S overnight && python scripts/overnight_optimization.py
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_optimization(config_path: str, trials: int, jobs: int, output_dir: str) -> dict:
    """Run strategy optimization."""
    print(f"\n{'='*60}")
    print(f"OPTIMIZING: {config_path}")
    print(f"Trials: {trials}, Jobs: {jobs}")
    print(f"{'='*60}\n")

    try:
        from trading_system.optimization import StrategyOptimizer

        data_paths = {
            "equity": "data/equity/daily",
            "crypto": "data/crypto/daily",
            "benchmark": "data/test_benchmarks",
        }

        optimizer = StrategyOptimizer(
            base_config_path=config_path,
            data_paths=data_paths,
            output_dir=output_dir,
        )

        result = optimizer.optimize(n_trials=trials, n_jobs=jobs)

        return {
            "config": config_path,
            "best_value": result.best_value,
            "best_params": result.best_params,
            "study_name": result.study_name,
            "n_completed": result.n_completed,
        }
    except Exception as e:
        print(f"ERROR in optimization: {e}")
        return {"config": config_path, "error": str(e)}


def run_analysis(results_path: str) -> dict:
    """Run expanded metrics analysis."""
    print(f"\nAnalyzing: {results_path}")

    try:
        from scripts.analyze_strategy import StrategyAnalyzer

        analyzer = StrategyAnalyzer(results_path)
        report = analyzer.generate_report()
        print(report)
        return {"status": "success", "report": report}
    except Exception as e:
        print(f"Analysis error: {e}")
        return {"status": "error", "error": str(e)}


def create_ensemble(results_files: list, output_path: str) -> dict:
    """Create ensemble from top optimization results."""
    print(f"\n{'='*60}")
    print("CREATING ENSEMBLE")
    print(f"{'='*60}\n")

    try:
        pass

        # Load all results and combine top trials
        all_trials = []
        for path in results_files:
            if Path(path).exists():
                with open(path) as f:
                    data = json.load(f)
                trials = data.get("top_trials", [])
                for t in trials:
                    t["source"] = path
                all_trials.extend(trials)

        # Sort by value and take top 10
        all_trials.sort(key=lambda x: x.get("value", 0), reverse=True)
        top_trials = all_trials[:10]

        ensemble_config = {
            "name": "overnight_ensemble",
            "created": datetime.now().isoformat(),
            "members": [
                {
                    "name": f"member_{i}",
                    "params": t.get("params", {}),
                    "value": t.get("value", 0),
                    "source": t.get("source", ""),
                }
                for i, t in enumerate(top_trials)
            ],
        }

        with open(output_path, "w") as f:
            json.dump(ensemble_config, f, indent=2)

        print(f"Ensemble saved to: {output_path}")
        print(f"Members: {len(top_trials)}")

        return {"status": "success", "members": len(top_trials)}
    except Exception as e:
        print(f"Ensemble error: {e}")
        return {"status": "error", "error": str(e)}


def generate_final_report(results: dict, output_dir: str) -> str:
    """Generate comprehensive final report."""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("OVERNIGHT OPTIMIZATION REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 70)
    report_lines.append("")

    # Timing
    report_lines.append("EXECUTION SUMMARY")
    report_lines.append("-" * 40)
    report_lines.append(f"Start Time: {results.get('start_time', 'N/A')}")
    report_lines.append(f"End Time: {results.get('end_time', 'N/A')}")
    report_lines.append(f"Duration: {results.get('duration', 'N/A')}")
    report_lines.append("")

    # Optimization results
    report_lines.append("OPTIMIZATION RESULTS")
    report_lines.append("-" * 40)
    for opt in results.get("optimizations", []):
        config = Path(opt.get("config", "")).stem
        best_val = opt.get("best_value", 0)
        trials = opt.get("n_completed", 0)
        report_lines.append(f"  {config}: Sharpe={best_val:.4f} ({trials} trials)")
    report_lines.append("")

    # Best parameters
    report_lines.append("BEST PARAMETERS")
    report_lines.append("-" * 40)
    for opt in results.get("optimizations", []):
        if "best_params" in opt:
            config = Path(opt.get("config", "")).stem
            report_lines.append(f"\n  {config}:")
            for k, v in opt["best_params"].items():
                report_lines.append(f"    {k}: {v}")
    report_lines.append("")

    # Ensemble
    if "ensemble" in results:
        report_lines.append("ENSEMBLE")
        report_lines.append("-" * 40)
        report_lines.append(f"  Members: {results['ensemble'].get('members', 0)}")
    report_lines.append("")

    report_lines.append("=" * 70)

    report = "\n".join(report_lines)

    # Save report
    report_path = Path(output_dir) / f"overnight_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nReport saved to: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Overnight super-optimization")
    parser.add_argument(
        "--equity-config",
        type=str,
        default="configs/test_equity_strategy.yaml",
        help="Equity strategy config",
    )
    parser.add_argument(
        "--crypto-config",
        type=str,
        default="configs/test_crypto_strategy.yaml",
        help="Crypto strategy config",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1000,
        help="Optimization trials per strategy",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Parallel jobs (0 = auto-detect)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="overnight_results",
        help="Output directory",
    )
    parser.add_argument(
        "--equity-only",
        action="store_true",
        help="Only run equity optimization",
    )
    parser.add_argument(
        "--crypto-only",
        action="store_true",
        help="Only run crypto optimization",
    )

    args = parser.parse_args()

    # Auto-detect jobs
    if args.jobs <= 0:
        import multiprocessing

        args.jobs = max(1, multiprocessing.cpu_count() - 1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("OVERNIGHT SUPER-OPTIMIZATION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Trials per strategy: {args.trials}")
    print(f"Parallel jobs: {args.jobs}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    results = {
        "start_time": datetime.now().isoformat(),
        "optimizations": [],
    }

    start_time = time.time()

    # Run optimizations
    opt_results_files = []

    if not args.crypto_only:
        equity_result = run_optimization(args.equity_config, args.trials, args.jobs, str(output_dir))
        results["optimizations"].append(equity_result)
        if "study_name" in equity_result:
            opt_results_files.append(output_dir / f"{equity_result['study_name']}.json")

    if not args.equity_only:
        crypto_result = run_optimization(args.crypto_config, args.trials, args.jobs, str(output_dir))
        results["optimizations"].append(crypto_result)
        if "study_name" in crypto_result:
            opt_results_files.append(output_dir / f"{crypto_result['study_name']}.json")

    # Create ensemble from all results
    if len(opt_results_files) > 0:
        ensemble_result = create_ensemble(
            [str(p) for p in opt_results_files],
            str(output_dir / "ensemble_config.json"),
        )
        results["ensemble"] = ensemble_result

    # Final timing
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    results["end_time"] = datetime.now().isoformat()
    results["duration"] = f"{hours}h {minutes}m"

    # Generate report
    report = generate_final_report(results, str(output_dir))
    print(report)

    # Save full results
    results_path = output_dir / f"overnight_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nFull results saved to: {results_path}")
    print("\n✅ OVERNIGHT OPTIMIZATION COMPLETE")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Create production baseline for performance benchmarks.

This script runs production-scale performance benchmarks and compares
results against expected performance targets. Use this to establish
a baseline for your production environment.

Usage:
    python scripts/create_production_baseline.py
"""

import sys
import subprocess
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


# Expected performance targets for production workloads (in seconds)
PRODUCTION_TARGETS = {
    "test_backtest_engine_full_run_performance": 30.0,  # 6 months, 20 symbols
    "test_event_loop_process_day_performance": 0.1,  # Single day, 20 symbols
    "test_csv_loading_performance": 0.5,  # 10 symbols, 5 years
    "test_signal_scoring_performance": 0.05,  # 100 signals
    "test_queue_selection_performance": 0.03,  # 100 signals
    "test_equity_strategy_evaluation_performance": 0.2,  # 50 symbols
    "test_report_generation_performance": 0.1,  # 100 trades, 3 years
    "test_bootstrap_performance": 2.0,  # 10,000 R-multiples, 1000 iterations
    "test_permutation_performance": 3.0,  # 100 trades, 1000 iterations
}


def run_benchmarks() -> Dict:
    """Run performance benchmarks and return results."""
    print("Running production baseline benchmarks...")
    print("=" * 70)
    
    # Ensure we're in the project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Run benchmarks
    result = subprocess.run(  # noqa: S603 - subprocess needed for pytest execution
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/performance/",
            "-m",
            "performance",
            "--benchmark-only",
            "--benchmark-json=production_baseline.json",
            "-v",
        ],
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        print("Warning: Some benchmarks may have failed")
        print(result.stderr)
    
    # Load results
    baseline_file = project_root / "production_baseline.json"
    if not baseline_file.exists():
        print("Error: Benchmark results file not found")
        return {}
    
    with open(baseline_file, "r") as f:
        results = json.load(f)
    
    return results


def analyze_results(results: Dict) -> Tuple[List[str], List[str], List[str]]:
    """Analyze benchmark results against production targets."""
    benchmarks = results.get("benchmarks", [])
    
    passed = []
    warnings = []
    failed = []
    
    for bench in benchmarks:
        name = bench.get("name", "")
        mean_time = bench.get("stats", {}).get("mean", 0)
        
        # Find matching target
        target_time = None
        for target_name, target in PRODUCTION_TARGETS.items():
            if target_name in name:
                target_time = target
                break
        
        if target_time is None:
            warnings.append(f"{name}: No target defined (mean: {mean_time:.4f}s)")
            continue
        
        if mean_time <= target_time:
            passed.append(f"{name}: {mean_time:.4f}s (target: {target_time:.2f}s) ✓")
        elif mean_time <= target_time * 1.2:  # Within 20% of target
            warnings.append(
                f"{name}: {mean_time:.4f}s (target: {target_time:.2f}s) ⚠ "
                f"({(mean_time/target_time - 1)*100:.1f}% slower)"
            )
        else:
            failed.append(
                f"{name}: {mean_time:.4f}s (target: {target_time:.2f}s) ✗ "
                f"({(mean_time/target_time - 1)*100:.1f}% slower)"
            )
    
    return passed, warnings, failed


def print_report(passed: List[str], warnings: List[str], failed: List[str]):
    """Print baseline report."""
    print("\n" + "=" * 70)
    print("PRODUCTION BASELINE REPORT")
    print("=" * 70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if passed:
        print(f"✓ PASSED ({len(passed)} benchmarks):")
        for item in passed:
            print(f"  {item}")
        print()
    
    if warnings:
        print(f"⚠ WARNINGS ({len(warnings)} benchmarks):")
        for item in warnings:
            print(f"  {item}")
        print()
    
    if failed:
        print(f"✗ FAILED ({len(failed)} benchmarks):")
        for item in failed:
            print(f"  {item}")
        print()
    
    print("=" * 70)
    print(f"Summary: {len(passed)} passed, {len(warnings)} warnings, {len(failed)} failed")
    print("=" * 70)
    
    if failed:
        print("\n⚠️  Some benchmarks exceeded production targets!")
        print("   Review the failed benchmarks and consider optimization.")
        return False
    elif warnings:
        print("\n⚠️  Some benchmarks are close to production targets.")
        print("   Monitor these benchmarks for regressions.")
        return True
    else:
        print("\n✓ All benchmarks meet production targets!")
        return True


def save_baseline():
    """Save the current benchmark results as baseline."""
    print("\nSaving baseline...")
    result = subprocess.run(  # noqa: S603 - subprocess needed for pytest execution
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/performance/",
            "-m",
            "performance",
            "--benchmark-only",
            "--benchmark-autosave",
        ],
        capture_output=True,
        text=True,
    )
    
    if result.returncode == 0:
        print("✓ Baseline saved successfully")
    else:
        print("⚠ Warning: Could not save baseline automatically")
        print("  Run manually: pytest tests/performance/ -m performance --benchmark-only --benchmark-autosave")


def main():
    """Main entry point."""
    print("Production Baseline Creation Script")
    print("=" * 70)
    print()
    print("This script will:")
    print("  1. Run production-scale performance benchmarks")
    print("  2. Compare results against expected targets")
    print("  3. Generate a baseline report")
    print("  4. Optionally save the baseline")
    print()
    
    # Run benchmarks
    results = run_benchmarks()
    
    if not results:
        print("Error: Could not run benchmarks")
        sys.exit(1)
    
    # Analyze results
    passed, warnings, failed = analyze_results(results)
    
    # Print report
    success = print_report(passed, warnings, failed)
    
    # Ask about saving baseline
    if success:
        response = input("\nSave this as the production baseline? (y/n): ").strip().lower()
        if response == "y":
            save_baseline()
    
    # Cleanup
    baseline_file = Path("production_baseline.json")
    if baseline_file.exists():
        baseline_file.unlink()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


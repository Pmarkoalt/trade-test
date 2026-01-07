#!/usr/bin/env python
"""ML Continuous Learning CLI - Phase 4 Implementation.

This script provides commands for:
1. Scheduled model retraining
2. Drift detection and monitoring
3. Model status and history

Usage:
    # Check status
    python scripts/ml_continuous_learning.py --status
    
    # Check for drift
    python scripts/ml_continuous_learning.py --check-drift
    
    # Run retraining cycle (if needed)
    python scripts/ml_continuous_learning.py --retrain
    
    # Force retraining
    python scripts/ml_continuous_learning.py --retrain --force
    
    # Run full cycle (drift check + retrain if needed)
    python scripts/ml_continuous_learning.py --cycle

For cron scheduling (weekly retraining):
    0 0 * * 0 cd /path/to/trade-test && python scripts/ml_continuous_learning.py --cycle >> logs/retrain.log 2>&1
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.ml_refinement.continuous_learning import (
    ContinuousLearningManager,
    run_scheduled_retrain,
)


# Configuration
DEFAULT_DB_PATH = "ml_features.db"
DEFAULT_MODEL_DIR = "models/signal_quality"


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )


def cmd_status(args):
    """Show current status of continuous learning system."""
    manager = ContinuousLearningManager(
        feature_db_path=args.db_path,
        model_dir=args.model_dir,
    )
    
    status = manager.get_status()
    
    print("\n" + "=" * 60)
    print("ML CONTINUOUS LEARNING STATUS")
    print("=" * 60)
    print(f"Current Model: {status['current_model_id'] or 'None'}")
    print(f"Model Path: {status['current_model']}")
    print(f"Total Labeled Samples: {status['total_labeled_samples']}")
    print(f"Samples Since Last Retrain: {status['samples_since_last_retrain']}")
    print(f"Last Retrain: {status['last_retrain'] or 'Never'}")
    print(f"Total Retrains: {status['retrain_count']}")
    print("-" * 60)
    print(f"Min Samples for Retrain: {status['min_samples_for_retrain']}")
    print(f"Min AUC Improvement: {status['min_auc_improvement']}")
    
    # Check if retrain needed
    should_retrain, reason = manager.should_retrain()
    print("-" * 60)
    print(f"Retrain Needed: {'YES' if should_retrain else 'No'}")
    print(f"Reason: {reason}")
    print("=" * 60 + "\n")


def cmd_check_drift(args):
    """Check for concept drift."""
    manager = ContinuousLearningManager(
        feature_db_path=args.db_path,
        model_dir=args.model_dir,
    )
    
    print("\n" + "=" * 60)
    print("DRIFT DETECTION REPORT")
    print("=" * 60)
    
    report = manager.check_drift(lookback_days=args.lookback_days)
    
    print(f"Timestamp: {report.timestamp}")
    print(f"Drift Detected: {'YES ⚠️' if report.drift_detected else 'No ✓'}")
    print(f"Drift Score: {report.drift_score:.4f}")
    print(f"Prediction Drift: {report.prediction_drift:.4f}")
    
    if report.feature_drifts:
        print("\nFeature Drifts:")
        for name, drift in sorted(report.feature_drifts.items(), key=lambda x: -x[1])[:5]:
            print(f"  {name}: {drift:.4f}")
    
    print(f"\nRecommendation: {report.recommendation}")
    print("=" * 60 + "\n")
    
    return 1 if report.drift_detected else 0


def cmd_retrain(args):
    """Run retraining cycle."""
    manager = ContinuousLearningManager(
        feature_db_path=args.db_path,
        model_dir=args.model_dir,
        min_samples_for_retrain=args.min_samples,
        min_auc_improvement=args.min_auc_improvement,
    )
    
    print("\n" + "=" * 60)
    print("RETRAINING CYCLE")
    print("=" * 60)
    
    if args.force:
        print("Force retrain requested")
        result = manager.retrain(force=True)
    else:
        result = manager.check_and_retrain()
    
    print(f"Timestamp: {result.timestamp}")
    print(f"Triggered By: {result.triggered_by}")
    
    if result.training_result:
        tr = result.training_result
        print(f"\nTraining Results:")
        print(f"  Model ID: {tr.model_id}")
        print(f"  Samples: {tr.train_samples}")
        print(f"  CV AUC: {tr.cv_metrics.get('auc', 'N/A'):.4f}" if tr.cv_metrics else "  CV AUC: N/A")
        
        if tr.top_features:
            print(f"\n  Top Features:")
            for name, imp in tr.top_features[:5]:
                print(f"    {name}: {imp:.4f}")
    
    if result.comparison_result:
        cr = result.comparison_result
        print(f"\nModel Comparison:")
        print(f"  Current: {cr.current_model_id}")
        print(f"  New: {cr.new_model_id}")
        print(f"  AUC Improvement: {cr.improvement.get('auc', 0):.4f}")
        print(f"  Decision: {cr.reason}")
    
    print(f"\nDeployed: {'YES ✓' if result.deployed else 'No'}")
    
    if result.error:
        print(f"Error: {result.error}")
    
    print("=" * 60 + "\n")
    
    return 0 if result.deployed or result.triggered_by == "skipped" else 1


def cmd_cycle(args):
    """Run full cycle: drift check + retrain if needed."""
    print("\n" + "=" * 60)
    print("FULL CONTINUOUS LEARNING CYCLE")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60 + "\n")
    
    # Step 1: Check drift
    print("Step 1: Checking for drift...")
    manager = ContinuousLearningManager(
        feature_db_path=args.db_path,
        model_dir=args.model_dir,
        min_samples_for_retrain=args.min_samples,
        min_auc_improvement=args.min_auc_improvement,
    )
    
    drift_report = manager.check_drift()
    print(f"  Drift Score: {drift_report.drift_score:.4f}")
    print(f"  Drift Detected: {drift_report.drift_detected}")
    
    # Step 2: Check and retrain
    print("\nStep 2: Checking if retraining needed...")
    should_retrain, reason = manager.should_retrain()
    
    # Force retrain if drift detected
    force_retrain = drift_report.drift_detected and args.retrain_on_drift
    
    if should_retrain or force_retrain:
        if force_retrain:
            print("  Drift detected - forcing retrain")
        else:
            print(f"  Retraining triggered: {reason}")
        
        result = manager.retrain(force=force_retrain)
        
        print(f"\n  Training completed: {result.training_result.model_id if result.training_result else 'N/A'}")
        print(f"  Deployed: {result.deployed}")
    else:
        print(f"  Skipping retrain: {reason}")
        result = None
    
    # Summary
    print("\n" + "=" * 60)
    print("CYCLE COMPLETE")
    print(f"Finished: {datetime.now().isoformat()}")
    if result and result.deployed:
        print(f"New Model: {result.training_result.model_id}")
    print("=" * 60 + "\n")
    
    return 0


def cmd_history(args):
    """Show retraining history."""
    manager = ContinuousLearningManager(
        feature_db_path=args.db_path,
        model_dir=args.model_dir,
    )
    
    state = manager._state
    
    print("\n" + "=" * 60)
    print("RETRAINING HISTORY")
    print("=" * 60)
    
    history = state.get("retrain_history", [])
    if not history:
        print("No retraining history available.")
    else:
        print(f"{'Timestamp':<25} {'Model ID':<35} {'AUC':<8} {'Deployed':<10}")
        print("-" * 78)
        for entry in history[-10:]:  # Last 10 entries
            timestamp = entry.get("timestamp", "")[:19]
            model_id = entry.get("model_id", "N/A")[:33]
            auc = entry.get("cv_auc", 0)
            deployed = "Yes" if entry.get("deployed") else "No"
            print(f"{timestamp:<25} {model_id:<35} {auc:<8.4f} {deployed:<10}")
    
    print("\n" + "-" * 60)
    print("DRIFT HISTORY (Last 10)")
    print("-" * 60)
    
    drift_history = state.get("drift_history", [])
    if not drift_history:
        print("No drift history available.")
    else:
        print(f"{'Timestamp':<25} {'Drift Score':<15} {'Detected':<10}")
        print("-" * 50)
        for entry in drift_history[-10:]:
            timestamp = entry.get("timestamp", "")[:19]
            score = entry.get("drift_score", 0)
            detected = "Yes ⚠️" if entry.get("drift_detected") else "No"
            print(f"{timestamp:<25} {score:<15.4f} {detected:<10}")
    
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="ML Continuous Learning Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Global options
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Feature database path")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Model directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    # Commands
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--check-drift", action="store_true", help="Check for concept drift")
    parser.add_argument("--retrain", action="store_true", help="Run retraining cycle")
    parser.add_argument("--cycle", action="store_true", help="Run full cycle (drift + retrain)")
    parser.add_argument("--history", action="store_true", help="Show retraining history")
    
    # Retrain options
    parser.add_argument("--force", action="store_true", help="Force retraining")
    parser.add_argument("--min-samples", type=int, default=20, help="Min samples for retrain")
    parser.add_argument("--min-auc-improvement", type=float, default=0.01, help="Min AUC improvement")
    parser.add_argument("--lookback-days", type=int, default=30, help="Lookback days for drift")
    parser.add_argument("--retrain-on-drift", action="store_true", help="Force retrain on drift")
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    # Execute command
    if args.status:
        cmd_status(args)
    elif args.check_drift:
        return cmd_check_drift(args)
    elif args.retrain:
        return cmd_retrain(args)
    elif args.cycle:
        return cmd_cycle(args)
    elif args.history:
        cmd_history(args)
    else:
        parser.print_help()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

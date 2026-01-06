#!/usr/bin/env python
"""ML Training Pipeline - Phase 2 Implementation.

This script:
1. Runs backtests with ML data collection to accumulate labeled samples
2. Trains signal quality model using walk-forward validation
3. Evaluates model on holdout period
4. Reports feature importance

Usage:
    python scripts/ml_training_pipeline.py --accumulate  # Run backtests to collect data
    python scripts/ml_training_pipeline.py --train       # Train model on accumulated data
    python scripts/ml_training_pipeline.py --evaluate    # Evaluate trained model
    python scripts/ml_training_pipeline.py --full        # Run full pipeline
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.backtest import BacktestEngine, WalkForwardSplit
from trading_system.configs.strategy_config import StrategyConfig
from trading_system.data.loader import load_all_data
from trading_system.ml_refinement.config import MLConfig, ModelType
from trading_system.ml_refinement.storage.feature_db import FeatureDatabase
from trading_system.ml_refinement.training import ModelTrainer
from trading_system.strategies.strategy_loader import load_strategy_from_config


# Configuration
DEFAULT_DB_PATH = "ml_features.db"
DEFAULT_MODEL_DIR = "models/signal_quality"
MIN_SAMPLES_FOR_TRAINING = 50  # Reduced from 100 for testing


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )


def run_backtest_for_data_collection(
    strategy_config_path: str,
    split: WalkForwardSplit,
    period: str,
    db_path: str,
) -> dict:
    """Run a single backtest with ML data collection enabled.
    
    Args:
        strategy_config_path: Path to strategy config YAML
        split: Walk-forward split configuration
        period: One of "train", "validation", "holdout"
        db_path: Path to feature database
        
    Returns:
        Backtest results dictionary
    """
    logger.info(f"Running backtest for {period} period with ML data collection...")
    
    # Load market data - use full available range to ensure enough lookback
    start_date = pd.Timestamp("2024-01-01")  # Start of available data
    end_date = pd.Timestamp("2025-12-31")  # End of available data
    
    # Load strategy to get universe
    strategy = load_strategy_from_config(strategy_config_path)
    universe = strategy.universe
    
    market_data, _ = load_all_data(
        equity_path="data/equity/daily",
        crypto_path="data/crypto/daily",
        benchmark_path="data/test_benchmarks",
        equity_universe=universe,
        start_date=start_date,
        end_date=end_date,
    )
    
    # Create engine with ML data collection
    engine = BacktestEngine(
        market_data=market_data,
        strategies=[strategy],
        starting_equity=100000.0,
        seed=42,
        ml_data_collection=True,
        ml_feature_db_path=db_path,
    )
    
    # Run backtest
    results = engine.run(split, period=period)
    
    # Get ML stats
    ml_stats = results.get("ml_data_collection", {})
    
    logger.info(
        f"  {period}: {results['total_trades']} trades, "
        f"{results['total_return']*100:.2f}% return, "
        f"{ml_stats.get('signals_recorded', 0)} signals recorded, "
        f"{ml_stats.get('outcomes_recorded', 0)} outcomes"
    )
    
    return results


def accumulate_samples(
    db_path: str = DEFAULT_DB_PATH,
    strategy_config: str = "configs/test_equity_strategy.yaml",
    expanded_universe: bool = True,
) -> int:
    """Run multiple backtests to accumulate ML training samples.
    
    Args:
        db_path: Path to feature database
        strategy_config: Path to strategy configuration
        expanded_universe: Use expanded universe for more trades
        
    Returns:
        Total number of labeled samples accumulated
    """
    logger.info("=" * 60)
    logger.info("PHASE 2.1: Accumulating ML Training Samples")
    logger.info("=" * 60)
    
    # Remove existing database to start fresh (optional)
    if os.path.exists(db_path):
        logger.info(f"Using existing database: {db_path}")
    
    # Define splits using available data (2024-01-08 to 2025-12-31)
    # Use later dates to ensure enough lookback period for indicators
    splits = [
        WalkForwardSplit(
            name="2025_H1",
            train_start=pd.Timestamp("2025-01-01"),
            train_end=pd.Timestamp("2025-03-31"),
            validation_start=pd.Timestamp("2025-04-01"),
            validation_end=pd.Timestamp("2025-05-15"),
            holdout_start=pd.Timestamp("2025-05-16"),
            holdout_end=pd.Timestamp("2025-06-30"),
        ),
        WalkForwardSplit(
            name="2025_H2",
            train_start=pd.Timestamp("2025-04-01"),
            train_end=pd.Timestamp("2025-06-30"),
            validation_start=pd.Timestamp("2025-07-01"),
            validation_end=pd.Timestamp("2025-08-31"),
            holdout_start=pd.Timestamp("2025-09-01"),
            holdout_end=pd.Timestamp("2025-10-31"),
        ),
        WalkForwardSplit(
            name="2025_full",
            train_start=pd.Timestamp("2025-01-01"),
            train_end=pd.Timestamp("2025-06-30"),
            validation_start=pd.Timestamp("2025-07-01"),
            validation_end=pd.Timestamp("2025-09-30"),
            holdout_start=pd.Timestamp("2025-10-01"),
            holdout_end=pd.Timestamp("2025-12-31"),
        ),
    ]
    
    # Run backtests for each split and period
    total_signals = 0
    total_outcomes = 0
    
    for split in splits:
        logger.info(f"\n--- Processing split: {split.name} ---")
        for period in ["train", "validation", "holdout"]:
            try:
                results = run_backtest_for_data_collection(
                    strategy_config_path=strategy_config,
                    split=split,
                    period=period,
                    db_path=db_path,
                )
                ml_stats = results.get("ml_data_collection", {})
                total_signals += ml_stats.get("signals_recorded", 0)
                total_outcomes += ml_stats.get("outcomes_recorded", 0)
            except Exception as e:
                logger.warning(f"Error in {split.name}/{period}: {e}")
                continue
    
    # Check database status
    db = FeatureDatabase(db_path)
    db.initialize()
    total_samples = db.count_samples(require_target=False)
    labeled_samples = db.count_samples(require_target=True)
    db.close()
    
    logger.info("\n" + "=" * 60)
    logger.info("Data Collection Summary:")
    logger.info(f"  Total signals recorded: {total_signals}")
    logger.info(f"  Total outcomes recorded: {total_outcomes}")
    logger.info(f"  Database total samples: {total_samples}")
    logger.info(f"  Database labeled samples: {labeled_samples}")
    logger.info("=" * 60)
    
    return labeled_samples


def train_model(
    db_path: str = DEFAULT_DB_PATH,
    model_dir: str = DEFAULT_MODEL_DIR,
    min_samples: int = MIN_SAMPLES_FOR_TRAINING,
) -> dict:
    """Train signal quality model on accumulated data.
    
    Args:
        db_path: Path to feature database
        model_dir: Directory to save trained model
        min_samples: Minimum samples required for training
        
    Returns:
        Training results dictionary
    """
    logger.info("=" * 60)
    logger.info("PHASE 2.2: Training Signal Quality Model")
    logger.info("=" * 60)
    
    # Initialize database
    db = FeatureDatabase(db_path)
    db.initialize()
    
    # Check sample count
    labeled_samples = db.count_samples(require_target=True)
    logger.info(f"Available labeled samples: {labeled_samples}")
    
    if labeled_samples < min_samples:
        logger.warning(
            f"Insufficient samples ({labeled_samples} < {min_samples}). "
            "Run --accumulate first to collect more data."
        )
        db.close()
        return {"success": False, "error": f"Insufficient samples: {labeled_samples}"}
    
    # Create ML config
    ml_config = MLConfig(
        enabled=True,
        model_dir=model_dir,
        feature_db_path=db_path,
    )
    ml_config.training.min_training_samples = min_samples
    ml_config.training.min_validation_samples = 5
    
    # Create trainer
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    trainer = ModelTrainer(
        config=ml_config,
        feature_db=db,
        model_dir=model_dir,
    )
    
    # Train model
    logger.info("Starting walk-forward training...")
    result = trainer.train(
        model_type=ModelType.SIGNAL_QUALITY,
        start_date="2025-01-01",
        end_date="2025-12-31",
    )
    
    db.close()
    
    if result.success:
        logger.info("\n" + "=" * 60)
        logger.info("Training Results:")
        logger.info(f"  Model ID: {result.model_id}")
        logger.info(f"  Training samples: {result.train_samples}")
        logger.info(f"  Validation samples: {result.val_samples}")
        logger.info(f"  CV Folds: {result.n_folds}")
        logger.info(f"  Training time: {result.total_time_seconds:.1f}s")
        logger.info("\nCross-Validation Metrics:")
        for metric, value in result.cv_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        logger.info("\nFinal Model Metrics:")
        for metric, value in result.final_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        logger.info("\nTop 10 Features:")
        for i, (name, importance) in enumerate(result.top_features[:10], 1):
            logger.info(f"  {i}. {name}: {importance:.4f}")
        logger.info("=" * 60)
    else:
        logger.error(f"Training failed: {result.error_message}")
    
    return {
        "success": result.success,
        "model_id": result.model_id,
        "cv_metrics": result.cv_metrics,
        "final_metrics": result.final_metrics,
        "top_features": result.top_features,
        "error": result.error_message,
    }


def evaluate_model(
    db_path: str = DEFAULT_DB_PATH,
    model_dir: str = DEFAULT_MODEL_DIR,
) -> dict:
    """Evaluate trained model on holdout data.
    
    Args:
        db_path: Path to feature database
        model_dir: Directory containing trained model
        
    Returns:
        Evaluation results dictionary
    """
    logger.info("=" * 60)
    logger.info("PHASE 2.3: Evaluating Model on Holdout")
    logger.info("=" * 60)
    
    # Find the latest model
    model_path = Path(model_dir)
    if not model_path.exists():
        logger.error(f"Model directory not found: {model_dir}")
        return {"success": False, "error": "Model not found"}
    
    model_files = list(model_path.glob("*.pkl"))
    if not model_files:
        logger.error("No trained models found")
        return {"success": False, "error": "No models found"}
    
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading model: {latest_model}")
    
    # Load model
    from trading_system.ml_refinement.models.base_model import SignalQualityModel
    
    model = SignalQualityModel()
    if not model.load(str(latest_model)):
        logger.error("Failed to load model")
        return {"success": False, "error": "Model load failed"}
    
    # Get holdout data from database
    db = FeatureDatabase(db_path)
    db.initialize()
    
    # Use last 3 months as holdout
    X_holdout, y_holdout, feature_names = db.get_training_data(
        start_date="2025-10-01",
        end_date="2025-12-31",
        require_target=True,
    )
    db.close()
    
    if len(X_holdout) == 0:
        logger.warning("No holdout samples available")
        return {"success": False, "error": "No holdout data"}
    
    logger.info(f"Holdout samples: {len(X_holdout)}")
    
    # Make predictions
    try:
        y_pred = model.predict(X_holdout)
        y_proba = model.predict_proba(X_holdout)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        y_binary = (y_holdout > 0).astype(int)
        
        metrics = {
            "accuracy": accuracy_score(y_binary, y_pred),
            "precision": precision_score(y_binary, y_pred, zero_division=0),
            "recall": recall_score(y_binary, y_pred, zero_division=0),
            "f1": f1_score(y_binary, y_pred, zero_division=0),
        }
        
        # AUC requires both classes present
        if len(set(y_binary)) > 1:
            metrics["auc"] = roc_auc_score(y_binary, y_proba)
        else:
            metrics["auc"] = 0.5
        
        logger.info("\n" + "=" * 60)
        logger.info("Holdout Evaluation Results:")
        logger.info(f"  Samples: {len(X_holdout)}")
        logger.info(f"  Win rate (actual): {y_binary.mean()*100:.1f}%")
        logger.info("\nMetrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Calibration analysis
        logger.info("\nPrediction Distribution:")
        bins = [0, 0.3, 0.5, 0.7, 1.0]
        for i in range(len(bins)-1):
            mask = (y_proba >= bins[i]) & (y_proba < bins[i+1])
            count = mask.sum()
            if count > 0:
                actual_win_rate = y_binary[mask].mean()
                logger.info(f"  P({bins[i]:.1f}-{bins[i+1]:.1f}): {count} samples, {actual_win_rate*100:.1f}% win rate")
        
        logger.info("=" * 60)
        
        return {
            "success": True,
            "samples": len(X_holdout),
            "metrics": metrics,
        }
        
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return {"success": False, "error": str(e)}


def run_full_pipeline(
    db_path: str = DEFAULT_DB_PATH,
    model_dir: str = DEFAULT_MODEL_DIR,
    strategy_config: str = "configs/test_equity_strategy.yaml",
):
    """Run the complete ML training pipeline.
    
    Args:
        db_path: Path to feature database
        model_dir: Directory to save trained model
        strategy_config: Path to strategy configuration
    """
    logger.info("=" * 60)
    logger.info("ML TRAINING PIPELINE - PHASE 2")
    logger.info("=" * 60)
    start_time = datetime.now()
    
    # Step 1: Accumulate samples
    labeled_samples = accumulate_samples(
        db_path=db_path,
        strategy_config=strategy_config,
    )
    
    if labeled_samples < MIN_SAMPLES_FOR_TRAINING:
        logger.warning(f"Insufficient samples for training: {labeled_samples}")
        # Continue anyway for testing
    
    # Step 2: Train model
    train_results = train_model(
        db_path=db_path,
        model_dir=model_dir,
        min_samples=min(labeled_samples, MIN_SAMPLES_FOR_TRAINING) if labeled_samples > 10 else 10,
    )
    
    if not train_results.get("success"):
        logger.error("Training failed, skipping evaluation")
        return
    
    # Step 3: Evaluate on holdout
    eval_results = evaluate_model(
        db_path=db_path,
        model_dir=model_dir,
    )
    
    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info(f"Labeled samples: {labeled_samples}")
    if train_results.get("success"):
        logger.info(f"Model ID: {train_results.get('model_id')}")
        cv_auc = train_results.get("cv_metrics", {}).get("auc", 0)
        logger.info(f"CV AUC: {cv_auc:.4f}")
    if eval_results.get("success"):
        holdout_auc = eval_results.get("metrics", {}).get("auc", 0)
        logger.info(f"Holdout AUC: {holdout_auc:.4f}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="ML Training Pipeline")
    parser.add_argument("--accumulate", action="store_true", help="Run backtests to collect data")
    parser.add_argument("--train", action="store_true", help="Train model on accumulated data")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate trained model")
    parser.add_argument("--full", action="store_true", help="Run full pipeline")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Feature database path")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Model output directory")
    parser.add_argument("--strategy", default="configs/test_equity_strategy.yaml", help="Strategy config")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    if args.full:
        run_full_pipeline(
            db_path=args.db_path,
            model_dir=args.model_dir,
            strategy_config=args.strategy,
        )
    elif args.accumulate:
        accumulate_samples(
            db_path=args.db_path,
            strategy_config=args.strategy,
        )
    elif args.train:
        train_model(
            db_path=args.db_path,
            model_dir=args.model_dir,
        )
    elif args.evaluate:
        evaluate_model(
            db_path=args.db_path,
            model_dir=args.model_dir,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

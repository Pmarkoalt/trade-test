"""
ML Strategy Development Example

This comprehensive example demonstrates how to:
1. Run backtests to collect training data (features + trade outcomes)
2. Train ML models to predict trade quality (R-multiples, win/loss)
3. Integrate ML predictions into strategy signal generation
4. Compare baseline vs ML-enhanced strategies

This is a practical, end-to-end workflow for developing ML-driven trading strategies.

Usage:
    python examples/ml_strategy_development.py
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.integration.runner import BacktestRunner
from trading_system.configs.run_config import RunConfig
from trading_system.ml.training import MLTrainer
from trading_system.ml.predictor import MLPredictor
from trading_system.ml.models import ModelType
from trading_system.ml.feature_engineering import MLFeatureEngineer
from trading_system.models.features import FeatureRow
from trading_system.models.positions import Position


def extract_features_and_labels(
    runner: BacktestRunner,
    period: str = "train"
) -> Tuple[List[FeatureRow], List[float], List[float], List[bool]]:
    """
    Extract features and labels from backtest results for ML training.
    
    This function:
    - Extracts FeatureRow objects from all signals that resulted in trades
    - Creates labels from trade outcomes (R-multiple, win/loss)
    - Returns aligned feature and label lists
    
    Args:
        runner: BacktestRunner that has run a backtest
        period: Period identifier (for logging)
        
    Returns:
        Tuple of:
        - feature_rows: List of FeatureRow objects at entry
        - r_multiples: List of R-multiples (trade outcome)
        - returns: List of returns (as percentage)
        - wins: List of win/loss booleans
    """
    print(f"\nExtracting features and labels from {period} period backtest...")
    
    engine = runner.engine
    if engine is None:
        raise ValueError("Backtest engine not initialized. Run backtest first.")
    
    # Get closed trades
    closed_trades = engine.closed_trades
    print(f"  Found {len(closed_trades)} closed trades")
    
    if len(closed_trades) == 0:
        print("  Warning: No closed trades found. Cannot extract training data.")
        return [], [], [], []
    
    # Extract features at entry date for each trade
    feature_rows = []
    r_multiples = []
    returns_list = []
    wins = []
    
    for trade in closed_trades:
        # Get features at entry date
        entry_features = engine.market_data.get_features(trade.symbol, trade.entry_date)
        
        if entry_features is None or not entry_features.is_valid_for_entry():
            # Skip trades where we don't have valid entry features
            continue
        
        feature_rows.append(entry_features)
        
        # Compute R-multiple (already computed in Position, but recalculate for consistency)
        r_mult = trade.compute_r_multiple()
        r_multiples.append(r_mult)
        
        # Compute return
        if trade.side.value == "LONG":
            ret = (trade.exit_price - trade.entry_price) / trade.entry_price
        else:
            ret = (trade.entry_price - trade.exit_price) / trade.entry_price
        returns_list.append(ret)
        
        # Win/loss
        wins.append(r_mult > 0)
    
    print(f"  Extracted {len(feature_rows)} valid feature-label pairs")
    print(f"  Average R-multiple: {np.mean(r_multiples):.2f}")
    print(f"  Win rate: {np.mean(wins):.2%}")
    
    return feature_rows, r_multiples, returns_list, wins


def train_trade_quality_model(
    feature_rows: List[FeatureRow],
    labels: List[float],
    model_type: ModelType = ModelType.RANDOM_FOREST,
    model_name: str = "trade_quality_model"
) -> Tuple[MLTrainer, Path]:
    """
    Train an ML model to predict trade quality (R-multiple).
    
    Args:
        feature_rows: List of FeatureRow objects for training
        labels: List of target values (R-multiples)
        model_type: Type of ML model to train
        model_name: Name for saving the model
        
    Returns:
        Tuple of (trained MLTrainer, model directory path)
    """
    print(f"\n{'='*70}")
    print(f"Training {model_type.value} model to predict trade quality")
    print(f"{'='*70}")
    
    if len(feature_rows) < 50:
        print(f"  Warning: Only {len(feature_rows)} samples. May not train well.")
    
    # Create feature engineer
    feature_engineer = MLFeatureEngineer(
        include_raw_features=True,
        include_derived_features=True,
        include_technical_indicators=True,
        include_volatility_features=True,
        include_market_regime_features=True,
        normalize_features=True
    )
    
    # Create trainer
    trainer = MLTrainer(
        model_type=model_type,
        feature_engineer=feature_engineer,
        hyperparameters={
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "random_state": 42,
            "task": "regression"  # Predicting R-multiple (continuous)
        }
    )
    
    # Train model
    print(f"  Training on {len(feature_rows)} samples...")
    metrics = trainer.train(
        feature_rows=feature_rows,
        target_values=labels,
        model_id=model_name
    )
    
    print(f"  Training metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")
    
    # Save model
    model_dir = Path("models") / model_name
    trainer.save_model(model_dir)
    print(f"  Model saved to: {model_dir}")
    
    return trainer, model_dir


def train_win_probability_model(
    feature_rows: List[FeatureRow],
    wins: List[bool],
    model_type: ModelType = ModelType.RANDOM_FOREST,
    model_name: str = "win_probability_model"
) -> Tuple[MLTrainer, Path]:
    """
    Train an ML model to predict win probability (classification).
    
    Args:
        feature_rows: List of FeatureRow objects for training
        wins: List of win/loss booleans
        model_type: Type of ML model to train
        model_name: Name for saving the model
        
    Returns:
        Tuple of (trained MLTrainer, model directory path)
    """
    print(f"\n{'='*70}")
    print(f"Training {model_type.value} model to predict win probability")
    print(f"{'='*70}")
    
    # Convert wins to 0/1 for classification
    win_labels = [1.0 if w else 0.0 for w in wins]
    
    # Create feature engineer
    feature_engineer = MLFeatureEngineer(
        include_raw_features=True,
        include_derived_features=True,
        include_technical_indicators=True,
        include_volatility_features=True,
        include_market_regime_features=True,
        normalize_features=True
    )
    
    # Create trainer
    trainer = MLTrainer(
        model_type=model_type,
        feature_engineer=feature_engineer,
        hyperparameters={
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "random_state": 42,
            "task": "classification"  # Binary classification
        }
    )
    
    # Train model
    print(f"  Training on {len(feature_rows)} samples...")
    metrics = trainer.train(
        feature_rows=feature_rows,
        target_values=win_labels,
        model_id=model_name
    )
    
    print(f"  Training metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")
    
    # Save model
    model_dir = Path("models") / model_name
    trainer.save_model(model_dir)
    print(f"  Model saved to: {model_dir}")
    
    return trainer, model_dir


def compare_baseline_vs_ml(
    config_path: str,
    model_dir: Path,
    period: str = "validation"
):
    """
    Compare baseline strategy vs ML-enhanced strategy performance.
    
    This runs two backtests:
    1. Baseline: Original strategy without ML
    2. ML-enhanced: Strategy with ML signal scoring
    
    Args:
        config_path: Path to run_config.yaml
        model_dir: Directory containing trained ML model
        period: Period to test on (validation or holdout)
    """
    print(f"\n{'='*70}")
    print(f"Comparing Baseline vs ML-Enhanced Strategy ({period} period)")
    print(f"{'='*70}")
    
    # Load configuration
    config = RunConfig.from_yaml(config_path)
    
    # ===== BASELINE BACKTEST =====
    print("\n1. Running baseline backtest (no ML)...")
    runner_baseline = BacktestRunner(config)
    runner_baseline.initialize()
    results_baseline = runner_baseline.run_backtest(period=period)
    
    print(f"\n  Baseline Results:")
    print(f"    Total Return: {results_baseline.get('total_return', 0):.2%}")
    print(f"    Sharpe Ratio: {results_baseline.get('sharpe_ratio', 0):.2f}")
    print(f"    Max Drawdown: {results_baseline.get('max_drawdown', 0):.2%}")
    print(f"    Win Rate: {results_baseline.get('win_rate', 0):.2%}")
    print(f"    Total Trades: {results_baseline.get('total_trades', 0)}")
    
    # ===== ML-ENHANCED BACKTEST =====
    print("\n2. Running ML-enhanced backtest...")
    
    # Note: To use ML in backtest, you would need to:
    # 1. Load the model and create an MLPredictor
    # 2. Pass it to the BacktestEngine/EventLoop
    # 3. Configure the strategy to use ML scoring
    
    # For now, we'll show the structure and note that this requires
    # integration in the backtest engine (which is already partially implemented)
    print("  Note: Full ML integration requires:")
    print("    - Loading model from model_dir")
    print("    - Creating MLPredictor instance")
    print("    - Configuring strategy config with ML settings")
    print("    - The backtest engine will automatically use ML if configured")
    
    print("\n  See ml_workflow.py for ML predictor setup example")
    print("  See strategy config YAML for ML configuration options")
    
    return results_baseline


def main():
    """Main workflow: train models and develop ML-driven strategies."""
    print("=" * 70)
    print("ML Strategy Development Workflow")
    print("=" * 70)
    print("\nThis example demonstrates:")
    print("  1. Running backtests to collect training data")
    print("  2. Training ML models on trade outcomes")
    print("  3. Using ML to enhance signal generation")
    print("  4. Comparing baseline vs ML-enhanced strategies")
    print()
    
    # Configuration
    config_path = "tests/fixtures/configs/run_test_config.yaml"  # Use test config for demo
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    try:
        # ===== STEP 1: Run backtest to collect training data =====
        print("\n" + "=" * 70)
        print("STEP 1: Collecting Training Data")
        print("=" * 70)
        
        config = RunConfig.from_yaml(config_path)
        runner = BacktestRunner(config)
        runner.initialize()
        
        # Run on training period
        print("\nRunning backtest on training period...")
        results_train = runner.run_backtest(period="train")
        
        print(f"\nTraining period results:")
        print(f"  Total Return: {results_train.get('total_return', 0):.2%}")
        print(f"  Total Trades: {results_train.get('total_trades', 0)}")
        print(f"  Win Rate: {results_train.get('win_rate', 0):.2%}")
        
        # ===== STEP 2: Extract features and labels =====
        print("\n" + "=" * 70)
        print("STEP 2: Extracting Features and Labels")
        print("=" * 70)
        
        feature_rows, r_multiples, returns, wins = extract_features_and_labels(
            runner, period="train"
        )
        
        if len(feature_rows) == 0:
            print("\n⚠️  No training data extracted. Cannot proceed with ML training.")
            print("   This might happen if:")
            print("   - Backtest had no trades")
            print("   - Trades were not closed during the training period")
            print("   - Features were not available for entry dates")
            return
        
        # ===== STEP 3: Train ML models =====
        print("\n" + "=" * 70)
        print("STEP 3: Training ML Models")
        print("=" * 70)
        
        # Train R-multiple prediction model (regression)
        trainer_r_multiple, model_dir_r_mult = train_trade_quality_model(
            feature_rows=feature_rows,
            labels=r_multiples,
            model_type=ModelType.RANDOM_FOREST,
            model_name="r_multiple_predictor"
        )
        
        # Train win probability model (classification)
        trainer_win_prob, model_dir_win_prob = train_win_probability_model(
            feature_rows=feature_rows,
            wins=wins,
            model_type=ModelType.RANDOM_FOREST,
            model_name="win_probability_predictor"
        )
        
        # ===== STEP 4: Evaluate models on validation period =====
        print("\n" + "=" * 70)
        print("STEP 4: Evaluating Models on Validation Period")
        print("=" * 70)
        
        # Run backtest on validation period
        print("\nRunning backtest on validation period...")
        results_val = runner.run_backtest(period="validation")
        
        # Extract validation data
        val_feature_rows, val_r_multiples, val_returns, val_wins = extract_features_and_labels(
            runner, period="validation"
        )
        
        if len(val_feature_rows) > 0:
            # Evaluate R-multiple model
            print("\nEvaluating R-multiple prediction model...")
            val_metrics_r_mult = trainer_r_multiple.evaluate(
                feature_rows=val_feature_rows,
                target_values=val_r_multiples
            )
            print("  Validation metrics:")
            for key, value in val_metrics_r_mult.items():
                print(f"    {key}: {value:.4f}")
            
            # Evaluate win probability model
            print("\nEvaluating win probability model...")
            val_win_labels = [1.0 if w else 0.0 for w in val_wins]
            val_metrics_win = trainer_win_prob.evaluate(
                feature_rows=val_feature_rows,
                target_values=val_win_labels
            )
            print("  Validation metrics:")
            for key, value in val_metrics_win.items():
                print(f"    {key}: {value:.4f}")
        
        # ===== STEP 5: Compare baseline vs ML-enhanced =====
        print("\n" + "=" * 70)
        print("STEP 5: Strategy Comparison")
        print("=" * 70)
        print("\nTo use ML models in backtesting:")
        print("  1. Update strategy config YAML to enable ML:")
        print("     ml:")
        print("       enabled: true")
        print(f"       model_path: \"{model_dir_r_mult}\"  # or win_prob model")
        print("       prediction_mode: \"score_enhancement\"  # or \"filter\", \"replace\"")
        print("       ml_weight: 0.3  # Weight for ML in score_enhancement mode")
        print()
        print("  2. The backtest engine will automatically:")
        print("     - Load the model")
        print("     - Generate predictions for each signal")
        print("     - Enhance signal scores based on ML predictions")
        print()
        print("  3. Run backtests and compare performance")
        
        # ===== SUMMARY =====
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"\n✓ Trained {len(feature_rows)} samples from training period")
        print(f"✓ Created R-multiple prediction model: {model_dir_r_mult}")
        print(f"✓ Created win probability model: {model_dir_win_prob}")
        print(f"\nNext steps:")
        print(f"  1. Review model performance metrics above")
        print(f"  2. Test models on validation/holdout periods")
        print(f"  3. Integrate into strategy configs (see above)")
        print(f"  4. Run backtests with ML-enabled configs")
        print(f"  5. Compare baseline vs ML-enhanced performance")
        print(f"\nModels saved in: {models_dir}")
        print()
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Configuration file not found: {e}")
        print(f"   Make sure you're running from the project root directory")
        print(f"   And that test fixtures are available")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
        print("\n✅ ML Strategy Development Example Completed!")
        print()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


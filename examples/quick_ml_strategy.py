"""
Quick ML Strategy Example

A simplified example showing the essential steps to:
1. Train an ML model on backtest results
2. Use it to enhance trading signals
3. Test the ML-enhanced strategy

This is a streamlined version focusing on the core workflow.

Usage:
    python examples/quick_ml_strategy.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.configs.run_config import RunConfig  # noqa: E402
from trading_system.integration.runner import BacktestRunner  # noqa: E402


def quick_ml_workflow():
    """Quick workflow to train and test an ML-enhanced strategy."""
    print("=" * 70)
    print("Quick ML Strategy Workflow")
    print("=" * 70)
    print()

    # Step 1: Run a backtest to get training data
    print("Step 1: Running backtest to collect training data...")
    config_path = "tests/fixtures/configs/run_test_config.yaml"

    config = RunConfig.from_yaml(config_path)
    runner = BacktestRunner(config)
    runner.initialize()

    # Run on training period
    results = runner.run_backtest(period="train")
    print(f"  ✓ Completed backtest: {results.get('total_trades', 0)} trades")

    # Step 2: Extract features and labels
    print("\nStep 2: Extracting features and labels...")
    from examples.ml_strategy_development import extract_features_and_labels

    feature_rows, r_multiples, _, wins = extract_features_and_labels(runner, period="train")

    if len(feature_rows) < 50:
        print(f"  ⚠️  Only {len(feature_rows)} samples. Consider running on more data.")
        print("     For this demo, we'll proceed anyway.")

    # Step 3: Train a simple model
    print("\nStep 3: Training ML model...")
    print("  (This uses the full workflow from ml_strategy_development.py)")
    print("  Run: python examples/ml_strategy_development.py")
    print("  for complete training and evaluation")

    print("\n" + "=" * 70)
    print("Quick Reference: Using ML in Your Strategies")
    print("=" * 70)
    print(
        """
1. Train a model (see ml_strategy_development.py for full example):

   from trading_system.ml.training import MLTrainer
   from trading_system.ml.models import ModelType

   trainer = MLTrainer(
       model_type=ModelType.RANDOM_FOREST,
       hyperparameters={"n_estimators": 100, "max_depth": 10, "task": "regression"}
   )
   trainer.train(feature_rows, r_multiples, model_id="my_model")
   trainer.save_model(Path("models/my_model"))

2. Configure your strategy to use ML (in strategy YAML config):

   ml:
     enabled: true
     model_path: "models/my_model"
     prediction_mode: "score_enhancement"  # Options: score_enhancement, filter, replace
     ml_weight: 0.3  # For score_enhancement mode (0.0-1.0)
     confidence_threshold: 0.5  # For filter mode (0.0-1.0)

3. Run backtest with ML-enabled strategy:

   from trading_system.integration.runner import run_backtest
   results = run_backtest("configs/run_config.yaml", period="validation")

4. Compare results:
   - Baseline strategy (ml.enabled: false)
   - ML-enhanced strategy (ml.enabled: true)

   Look at metrics like:
   - Total return
   - Sharpe ratio
   - Win rate
   - Average R-multiple
   """
    )

    print("\n" + "=" * 70)
    print("Next Steps")
    print("=" * 70)
    print(
        """
1. Run the full ML workflow:
   python examples/ml_strategy_development.py

2. Review the trained models in models/ directory

3. Test different prediction modes:
   - score_enhancement: Blend ML prediction with original signal score
   - filter: Only take signals above confidence threshold
   - replace: Use ML prediction as the signal score

4. Experiment with different ML models:
   - RandomForest (default, good starting point)
   - GradientBoosting (often better performance)
   - XGBoost (requires xgboost package)
   - LightGBM (requires lightgbm package)

5. Tune hyperparameters for better performance

6. Validate on out-of-sample data (holdout period)
    """
    )


if __name__ == "__main__":
    try:
        quick_ml_workflow()
        print("\n✅ Quick ML Strategy Example Completed!")
        print()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

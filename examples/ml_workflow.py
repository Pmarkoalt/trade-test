"""
ML Workflow Example

This example demonstrates how to:
1. Train an ML model on historical backtest data
2. Use the model for predictions during backtesting
3. Integrate ML predictions into signal scoring

Usage:
    python examples/ml_workflow.py
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.integration.runner import BacktestRunner
from trading_system.configs.run_config import RunConfig
from trading_system.ml.training import MLTrainer
from trading_system.ml.predictor import MLPredictor
from trading_system.ml.models import MLModel
from trading_system.ml.feature_engineering import MLFeatureEngineer
from trading_system.ml.versioning import MLModelVersioning


def example_train_ml_model():
    """Example: Train an ML model on historical backtest data."""
    print("=" * 60)
    print("Example 1: Training ML Model")
    print("=" * 60)
    
    # Step 1: Run a backtest to generate training data
    print("\nStep 1: Running backtest to generate training data...")
    config_path = "EXAMPLE_CONFIGS/run_config.yaml"
    config = RunConfig.from_yaml(config_path)
    
    runner = BacktestRunner(config)
    runner.initialize()
    
    # Run backtest on train period
    train_results = runner.run_backtest(period="train")
    
    # Step 2: Extract features and labels from backtest results
    print("\nStep 2: Extracting features and labels from backtest...")
    
    # Get daily events and closed trades
    daily_events = runner.engine.daily_events
    closed_trades = runner.engine.closed_trades
    
    # Build feature matrix and labels
    # In practice, you'd extract FeatureRow objects from daily_events
    # and create labels from closed_trades (e.g., R-multiple, win/loss)
    
    print("Note: Feature extraction from backtest results requires")
    print("      accessing the feature computation pipeline.")
    print("      This is a simplified example.")
    
    # Step 3: Train model
    print("\nStep 3: Training ML model...")
    
    # Initialize trainer
    trainer = MLTrainer(
        model_type="random_forest",  # or "xgboost", "lightgbm", etc.
        target_variable="r_multiple",  # or "win_probability", "expected_return"
        random_seed=42
    )
    
    # In practice, you'd pass actual feature data here
    # For this example, we'll create dummy data
    print("Creating dummy training data for demonstration...")
    n_samples = 1000
    n_features = 20
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randn(n_samples)  # R-multiples
    
    # Train model
    model = trainer.train(X_train, y_train)
    
    print(f"Model trained successfully!")
    print(f"  Model type: {type(model).__name__}")
    print(f"  Training samples: {n_samples}")
    
    # Step 4: Save model
    print("\nStep 4: Saving model...")
    model_dir = Path("models/ml_model_v1")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model.save(str(model_dir))
    print(f"Model saved to: {model_dir}")
    
    return model, model_dir


def example_load_and_use_model():
    """Example: Load a trained model and use it for predictions."""
    print("\n" + "=" * 60)
    print("Example 2: Loading and Using ML Model")
    print("=" * 60)
    
    # Step 1: Load model
    print("\nStep 1: Loading saved model...")
    model_dir = Path("models/ml_model_v1")
    
    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}")
        print("Please run example_train_ml_model() first.")
        return None
    
    model = MLModel.load(str(model_dir))
    print(f"Model loaded from: {model_dir}")
    
    # Step 2: Create feature engineer (must match training setup)
    print("\nStep 2: Creating feature engineer...")
    feature_engineer = MLFeatureEngineer(
        include_technical_indicators=True,
        include_price_features=True,
        include_volume_features=True,
        normalize=True
    )
    
    # In practice, you'd fit this on training data
    # For this example, we'll skip fitting
    print("Feature engineer created (not fitted in this example)")
    
    # Step 3: Create predictor
    print("\nStep 3: Creating ML predictor...")
    predictor = MLPredictor(
        model=model,
        feature_engineer=feature_engineer,
        prediction_mode="score_enhancement",  # or "filter", "replace"
        confidence_threshold=0.5
    )
    
    print(f"Predictor created with mode: {predictor.prediction_mode}")
    
    # Step 4: Use predictor in backtest
    print("\nStep 4: Using predictor in backtest...")
    print("Note: To use ML predictions in backtest, you need to:")
    print("  1. Enable ML in strategy config (ml.enabled: true)")
    print("  2. Set model_path in strategy config")
    print("  3. The backtest engine will automatically load and use the model")
    
    return predictor


def example_ml_integration_config():
    """Example: Show how to configure ML in strategy config."""
    print("\n" + "=" * 60)
    print("Example 3: ML Integration Configuration")
    print("=" * 60)
    
    print("""
To enable ML in your strategy, add this to your strategy config YAML:

ml:
  enabled: true
  model_path: "models/ml_model_v1"  # Path to saved model directory
  prediction_mode: "score_enhancement"  # Options: "score_enhancement", "filter", "replace"
  ml_weight: 0.3  # Weight for ML prediction (0.0-1.0) in score_enhancement mode
  confidence_threshold: 0.5  # Minimum confidence for filtering mode (0.0-1.0)

Prediction modes:
  - "score_enhancement": ML prediction is added as weighted component to signal score
  - "filter": Signals below confidence threshold are filtered out
  - "replace": Signal score is replaced with ML prediction

Example strategy config with ML enabled:
  See EXAMPLE_CONFIGS/equity_config.yaml for a template.
    """)


def example_model_versioning():
    """Example: Use model versioning to track ML models."""
    print("\n" + "=" * 60)
    print("Example 4: ML Model Versioning")
    print("=" * 60)
    
    # Initialize versioning
    versioning = MLModelVersioning(base_dir="models")
    
    # Register a new model version
    print("\nRegistering new model version...")
    model_info = {
        "model_type": "random_forest",
        "training_date": pd.Timestamp.now().isoformat(),
        "training_samples": 1000,
        "performance_metrics": {
            "train_r2": 0.75,
            "validation_r2": 0.70
        }
    }
    
    model_path = "models/ml_model_v1"
    version = versioning.register_model(model_path, model_info)
    print(f"Model registered with version: {version}")
    
    # List all versions
    print("\nListing all model versions...")
    versions = versioning.list_versions()
    for v in versions:
        print(f"  Version {v['version']}: {v['model_path']}")
        print(f"    Training date: {v.get('training_date', 'N/A')}")
    
    # Get latest version
    print("\nGetting latest model version...")
    latest = versioning.get_latest_version()
    if latest:
        print(f"Latest version: {latest['version']}")
        print(f"Model path: {latest['model_path']}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ML Workflow Examples")
    print("=" * 60)
    
    try:
        # Example 1: Train model
        model, model_dir = example_train_ml_model()
        
        # Example 2: Load and use model
        predictor = example_load_and_use_model()
        
        # Example 3: Configuration
        example_ml_integration_config()
        
        # Example 4: Model versioning
        example_model_versioning()
        
        print("\n" + "=" * 60)
        print("ML Workflow Examples Completed!")
        print("=" * 60)
        print("\nNote: This is a simplified example.")
        print("In production, you would:")
        print("  1. Extract features from actual backtest results")
        print("  2. Create labels from trade outcomes (R-multiples, win/loss)")
        print("  3. Train on train period, validate on validation period")
        print("  4. Test on holdout period")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


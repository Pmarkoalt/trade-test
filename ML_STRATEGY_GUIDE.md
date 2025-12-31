# ML Strategy Development Guide

This guide walks you through using machine learning to enhance your trading strategies with the trading system.

## Overview

The trading system includes a complete ML pipeline that allows you to:
1. **Train models** on historical backtest results to learn from past trades
2. **Predict trade quality** (R-multiples, win probability) before entering positions
3. **Enhance signals** by combining ML predictions with rule-based strategy scores
4. **Filter signals** based on ML confidence to only take high-probability trades

## Quick Start

The fastest way to get started is to run the example scripts:

```bash
# Full workflow example (recommended)
python examples/ml_strategy_development.py

# Quick reference example
python examples/quick_ml_strategy.py
```

These examples will:
- Run backtests to collect training data
- Train ML models on trade outcomes
- Show you how to integrate ML into your strategies

## Workflow

### Step 1: Collect Training Data

Run a backtest on your training period to generate trades:

```python
from trading_system.integration.runner import BacktestRunner
from trading_system.configs.run_config import RunConfig

# Load configuration
config = RunConfig.from_yaml("configs/run_config.yaml")
runner = BacktestRunner(config)
runner.initialize()

# Run backtest on training period
results = runner.run_backtest(period="train")
```

### Step 2: Extract Features and Labels

Extract features (technical indicators at entry) and labels (trade outcomes):

```python
from examples.ml_strategy_development import extract_features_and_labels

feature_rows, r_multiples, returns, wins = extract_features_and_labels(
    runner, period="train"
)
```

This function extracts:
- **Features**: All technical indicators (MA, ATR, RSI, MACD, etc.) at the entry date
- **Labels**:
  - `r_multiples`: Risk-adjusted returns (target for regression)
  - `wins`: Win/loss boolean (target for classification)

### Step 3: Train ML Models

Train models to predict trade outcomes:

#### Option A: Predict R-Multiple (Regression)

```python
from trading_system.ml.training import MLTrainer
from trading_system.ml.models import ModelType
from pathlib import Path

# Create trainer
trainer = MLTrainer(
    model_type=ModelType.RANDOM_FOREST,
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 20,
        "task": "regression"  # Predicting continuous R-multiple
    }
)

# Train model
metrics = trainer.train(
    feature_rows=feature_rows,
    target_values=r_multiples,
    model_id="r_multiple_predictor"
)

# Save model
trainer.save_model(Path("models/r_multiple_predictor"))
```

#### Option B: Predict Win Probability (Classification)

```python
# Convert wins to 0/1
win_labels = [1.0 if w else 0.0 for w in wins]

# Train classification model
trainer = MLTrainer(
    model_type=ModelType.RANDOM_FOREST,
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 10,
        "task": "classification"  # Binary classification
    }
)

metrics = trainer.train(
    feature_rows=feature_rows,
    target_values=win_labels,
    model_id="win_probability_predictor"
)

trainer.save_model(Path("models/win_probability_predictor"))
```

### Step 4: Integrate ML into Strategy

Configure your strategy YAML to use the trained model:

```yaml
# In your strategy config (e.g., equity_config.yaml)
ml:
  enabled: true
  model_path: "models/r_multiple_predictor"  # Path to saved model
  prediction_mode: "score_enhancement"  # See modes below
  ml_weight: 0.3  # Weight for ML prediction (0.0-1.0)
  confidence_threshold: 0.5  # For filter mode (0.0-1.0)
```

#### Prediction Modes

1. **score_enhancement** (Recommended)
   - Combines ML prediction with original signal score
   - Formula: `final_score = (1 - ml_weight) * original_score + ml_weight * ml_prediction`
   - Good for gradual enhancement without completely replacing rule-based logic

2. **filter**
   - Filters out signals below confidence threshold
   - Only takes trades where ML confidence ≥ threshold
   - Good for reducing trade count while improving quality

3. **replace**
   - Replaces signal score entirely with ML prediction
   - Use when ML model is highly trusted
   - More aggressive approach

### Step 5: Run ML-Enhanced Backtest

Run backtest with ML-enabled strategy:

```python
# The backtest engine automatically loads and uses ML models
# if they're configured in the strategy config

results = runner.run_backtest(period="validation")
```

### Step 6: Compare Performance

Compare baseline vs ML-enhanced results:

```python
# Baseline (ml.enabled: false)
results_baseline = run_backtest(config_path, period="validation")

# ML-enhanced (ml.enabled: true)
results_ml = run_backtest(config_path, period="validation")

# Compare key metrics
print(f"Baseline Return: {results_baseline['total_return']:.2%}")
print(f"ML Return: {results_ml['total_return']:.2%}")
print(f"Baseline Sharpe: {results_baseline['sharpe_ratio']:.2f}")
print(f"ML Sharpe: {results_ml['sharpe_ratio']:.2f}")
```

## Available ML Models

The system supports several model types:

1. **RandomForest** (Default, good starting point)
   - Robust, handles non-linear relationships well
   - Good for initial experiments

2. **GradientBoosting** (Often better performance)
   - Typically better accuracy than RandomForest
   - Longer training time

3. **XGBoost** (Requires `xgboost` package)
   - State-of-the-art performance
   - Fast and scalable

4. **LightGBM** (Requires `lightgbm` package)
   - Very fast training
   - Good for large datasets

5. **Linear/Logistic Regression** (Baseline)
   - Fast, interpretable
   - Limited to linear relationships

## Feature Engineering

The system automatically creates rich feature sets from technical indicators:

### Raw Features
- Moving averages (MA20, MA50, MA200)
- ATR (volatility)
- ROC60 (momentum)
- Breakout levels (20D, 55D highs)
- Volume (ADV20)

### Derived Features
- Price relative to MAs (close/MA20 - 1)
- Breakout strength (normalized by ATR)
- Relative strength vs benchmark
- MA relationships (MA20/MA50, MA50/MA200)
- Breakout clearance percentages

### Technical Indicators
- RSI (momentum)
- MACD (trend)
- Bollinger Bands (volatility)
- Realized volatility

### Market Regime Features
- Trend detection (bullish/bearish)
- Volatility regime (high/low)

See `trading_system/ml/feature_engineering.py` for complete feature list.

## Best Practices

### 1. Data Requirements
- **Minimum**: 100+ trades for meaningful training
- **Recommended**: 500+ trades for robust models
- Use walk-forward splits (train/validation/holdout)

### 2. Model Selection
- Start with RandomForest (simple, robust)
- Move to GradientBoosting or XGBoost if you need better performance
- Always validate on out-of-sample data

### 3. Hyperparameter Tuning
- Use cross-validation or validation set
- Don't overfit to training data
- Test on holdout period for final evaluation

### 4. ML Weight Selection
- Start with `ml_weight: 0.3` (30% ML, 70% original)
- Gradually increase if ML helps
- Don't completely replace rule-based logic initially

### 5. Validation
- Train on train period
- Tune on validation period
- Final test on holdout period (never seen during training)

## Example: Complete Workflow

```python
from pathlib import Path
from trading_system.integration.runner import BacktestRunner
from trading_system.configs.run_config import RunConfig
from trading_system.ml.training import MLTrainer
from trading_system.ml.models import ModelType
from examples.ml_strategy_development import extract_features_and_labels

# 1. Run backtest to collect data
config = RunConfig.from_yaml("configs/run_config.yaml")
runner = BacktestRunner(config)
runner.initialize()
results_train = runner.run_backtest(period="train")

# 2. Extract features and labels
feature_rows, r_multiples, _, wins = extract_features_and_labels(runner, "train")

# 3. Train model
trainer = MLTrainer(
    model_type=ModelType.RANDOM_FOREST,
    hyperparameters={"n_estimators": 100, "max_depth": 10, "task": "regression"}
)
trainer.train(feature_rows, r_multiples, model_id="my_model")
trainer.save_model(Path("models/my_model"))

# 4. Update strategy config YAML to enable ML (see Step 4 above)

# 5. Run ML-enhanced backtest
results_ml = runner.run_backtest(period="validation")

# 6. Compare results
print(f"Baseline trades: {results_train['total_trades']}")
print(f"ML-enhanced trades: {results_ml['total_trades']}")
print(f"Baseline return: {results_train['total_return']:.2%}")
print(f"ML-enhanced return: {results_ml['total_return']:.2%}")
```

## Troubleshooting

### "No training data extracted"
- Backtest had no trades
- Trades not closed during period
- Features not available at entry dates
- **Solution**: Ensure you have sufficient data and trades

### "Model performance is poor"
- Not enough training samples (need 100+)
- Model overfitting to training data
- Features not predictive
- **Solution**:
  - Collect more data
  - Use simpler model (reduce max_depth)
  - Add regularization
  - Try different feature sets

### "ML predictions not improving strategy"
- ML weight too low/high
- Wrong prediction mode
- Model not properly trained
- **Solution**:
  - Try different ml_weight values
  - Experiment with prediction modes
  - Validate model performance separately
  - Ensure model is actually loaded

## Advanced Topics

### Custom Feature Engineering

You can customize features by modifying `MLFeatureEngineer`:

```python
from trading_system.ml.feature_engineering import MLFeatureEngineer

feature_engineer = MLFeatureEngineer(
    include_raw_features=True,
    include_derived_features=True,
    include_technical_indicators=True,
    include_volatility_features=True,
    include_market_regime_features=True,
    normalize_features=True,
    rsi_period=14,  # Customize RSI period
    macd_fast=12,   # Customize MACD
    # ... other parameters
)
```

### Ensemble Models

Combine multiple models for better predictions:

```python
from trading_system.ml.ensemble import EnsemblePredictor

# Create ensemble from multiple models
ensemble = EnsemblePredictor(
    models=[model1, model2, model3],
    weights=[0.4, 0.3, 0.3]  # Weighted average
)
```

### Online Learning

Update models with new data over time:

```python
from trading_system.ml.online_learning import OnlineLearner

learner = OnlineLearner(base_model=initial_model)
learner.update(new_features, new_labels)  # Incrementally update
```

## Next Steps

1. Run the example scripts to see the workflow in action
2. Experiment with different models and hyperparameters
3. Test on different time periods and market conditions
4. Integrate ML into your production strategies
5. Monitor performance and retrain periodically

## Additional Resources

- `examples/ml_strategy_development.py` - Complete workflow example
- `examples/ml_workflow.py` - ML integration examples
- `trading_system/ml/` - ML module source code
- `docs/` - Additional documentation

For questions or issues, review the code examples and documentation.


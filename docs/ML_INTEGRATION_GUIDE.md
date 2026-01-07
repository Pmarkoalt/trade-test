# ML Integration Guide

## Overview

The ML Integration system enhances the trading system by using machine learning to **predict signal quality** - helping identify which trading signals are more likely to succeed based on historical patterns.

> **Important Distinction**: This ML system enhances *signal filtering/scoring* for an existing strategy. It does NOT automatically discover new strategies or optimize strategy parameters. See [What ML Does vs. Doesn't Do](#what-ml-does-vs-doesnt-do) below.

---

## Quick Start

```bash
# 1. Run backtests to collect training data
python scripts/ml_training_pipeline.py --accumulate

# 2. Train the ML model
python scripts/ml_training_pipeline.py --train

# 3. Enable ML in your strategy config
# Edit configs/equity_strategy_with_ml.yaml

# 4. Run backtest with ML enhancement
python -m trading_system backtest --config configs/equity_strategy_with_ml.yaml

# 5. Set up continuous learning (optional)
python scripts/ml_continuous_learning.py --cycle
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ML Integration Pipeline                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  PHASE 1: Data Accumulation                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Backtest   │───▶│   Extract    │───▶│   Feature    │          │
│  │   Engine     │    │   Features   │    │   Database   │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                                       │                    │
│         ▼                                       ▼                    │
│  ┌──────────────┐                      ┌──────────────┐             │
│  │ Trade Closes │─────────────────────▶│ Label with   │             │
│  │ (R-multiple) │                      │ Outcomes     │             │
│  └──────────────┘                      └──────────────┘             │
│                                                                      │
│  PHASE 2: Training                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Feature    │───▶│ Walk-Forward │───▶│   Trained    │          │
│  │   Database   │    │  Validation  │    │    Model     │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                                      │
│  PHASE 3: Live Integration                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Signal     │───▶│     ML       │───▶│  Enhanced    │          │
│  │  Generated   │    │  Prediction  │    │   Signal     │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                                      │
│  PHASE 4: Continuous Learning                                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Monitor    │───▶│   Retrain    │───▶│   Deploy     │          │
│  │    Drift     │    │  if Needed   │    │  if Better   │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## What ML Does vs. Doesn't Do

### ✅ What This ML System DOES

| Capability | Description |
|------------|-------------|
| **Signal Quality Prediction** | Predicts probability a signal will be profitable |
| **Signal Filtering** | Removes low-confidence signals before execution |
| **Score Enhancement** | Blends ML confidence with technical score |
| **Pattern Recognition** | Learns which feature combinations lead to wins |
| **Adaptive Learning** | Improves as more trades are accumulated |
| **Drift Detection** | Alerts when market conditions change |

### ❌ What This ML System Does NOT Do

| Limitation | What You'd Need Instead |
|------------|------------------------|
| **Discover New Strategies** | Strategy evolution / genetic algorithms |
| **Optimize Parameters** | Hyperparameter optimization (Optuna, grid search) |
| **Find New Entry Rules** | Rule discovery / symbolic regression |
| **Adjust Position Sizing** | Kelly criterion / ML-based sizing |
| **Time Market Regimes** | Regime detection models |
| **Generate Alpha Directly** | Price prediction models |

---

## How the ML Model Improves Over Time

### The Learning Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                     Continuous Improvement                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. Run Backtests ──▶ 2. Accumulate Data ──▶ 3. Retrain Model  │
│         │                      │                     │           │
│         │                      │                     ▼           │
│         │                      │              4. Compare AUC     │
│         │                      │                     │           │
│         │                      │         ┌───────────┴───────┐   │
│         │                      │         ▼                   ▼   │
│         │                      │    Improved?            Same?   │
│         │                      │         │                   │   │
│         │                      │         ▼                   ▼   │
│         │                      │    Deploy New          Keep Old │
│         │                      │                                 │
│         └──────────────────────┴─────────────────────────────────┘
│                              │
│                              ▼
│                    Model Gets Better At:
│                    • Identifying winning patterns
│                    • Filtering false breakouts
│                    • Recognizing regime conditions
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### What Improves

1. **More Data = Better Patterns**: More trade outcomes teach the model which signals succeed
2. **Feature Importance**: Model learns which indicators matter most
3. **Market Adaptation**: Continuous learning adapts to changing conditions
4. **Confidence Calibration**: Predictions become more accurate over time

### What Does NOT Improve Automatically

1. **Strategy Rules**: Entry/exit logic stays the same
2. **Parameters**: ATR multipliers, MA periods, etc. don't change
3. **Universe Selection**: Which stocks/crypto to trade
4. **Risk Management**: Position sizing rules

---

## Configuration Reference

### Strategy Config (equity_strategy_with_ml.yaml)

```yaml
ml:
  enabled: true                    # Enable ML integration
  model_path: "models/signal_quality"  # Path to trained model
  prediction_mode: "score_enhancement" # How to use predictions
  ml_weight: 0.3                  # 30% ML, 70% technical
  confidence_threshold: 0.5       # For filter mode
```

### Prediction Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `score_enhancement` | Blend ML score with technical score | Default - balanced approach |
| `filter` | Remove signals below threshold | Conservative - fewer but better trades |
| `replace` | Use ML score instead of technical | Aggressive - trust ML fully |

### Run Config (backtest_config.yaml)

```yaml
ml_data_collection:
  enabled: true           # Collect training data during backtest
  db_path: "features.db"  # Where to store features
```

---

## CLI Reference

### Training Pipeline

```bash
# Accumulate training data from backtests
python scripts/ml_training_pipeline.py --accumulate

# Train model on accumulated data
python scripts/ml_training_pipeline.py --train

# Evaluate on holdout
python scripts/ml_training_pipeline.py --evaluate

# Run full pipeline
python scripts/ml_training_pipeline.py --full
```

### Continuous Learning

```bash
# Check system status
python scripts/ml_continuous_learning.py --status

# Check for concept drift
python scripts/ml_continuous_learning.py --check-drift

# Run retraining (if needed)
python scripts/ml_continuous_learning.py --retrain

# Force retraining
python scripts/ml_continuous_learning.py --retrain --force

# Full cycle (drift + retrain)
python scripts/ml_continuous_learning.py --cycle

# View history
python scripts/ml_continuous_learning.py --history
```

### Cron Schedule

```bash
# Weekly retraining - every Sunday at midnight
0 0 * * 0 cd /path/to/trade-test && python scripts/ml_continuous_learning.py --cycle >> logs/retrain.log 2>&1
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `trading_system/backtest/ml_data_collector.py` | Extracts features during backtest |
| `trading_system/ml_refinement/continuous_learning.py` | Manages retraining cycle |
| `trading_system/ml_refinement/models/base_model.py` | SignalQualityModel class |
| `trading_system/ml_refinement/storage/feature_db.py` | SQLite feature storage |
| `trading_system/ml/predictor.py` | MLPredictor for live integration |
| `scripts/ml_training_pipeline.py` | Training pipeline CLI |
| `scripts/ml_continuous_learning.py` | Continuous learning CLI |
| `configs/equity_strategy_with_ml.yaml` | Example ML-enabled config |

---

---

## Strategy Parameter Optimization

In addition to ML signal enhancement, the system now includes **automatic parameter optimization** using Bayesian optimization.

### Quick Start

```bash
# Find optimal parameters (50 trials, ~1 hour)
python scripts/optimize_strategy.py --config configs/test_equity_strategy.yaml

# Quick test (5 trials, ~5 min)
python scripts/optimize_strategy.py --config configs/test_equity_strategy.yaml --trials 5 --quick

# Optimize for different metrics
python scripts/optimize_strategy.py --config configs/test_equity_strategy.yaml --objective calmar_ratio
```

### Parameters Optimized

| Parameter | Range | Description |
|-----------|-------|-------------|
| `exit.hard_stop_atr_mult` | 1.5 - 4.0 | Stop loss ATR multiplier |
| `exit.exit_ma` | 20, 50 | Exit moving average period |
| `entry.fast_clearance` | 0.0 - 0.02 | Fast breakout clearance % |
| `entry.slow_clearance` | 0.005 - 0.025 | Slow breakout clearance % |
| `risk.risk_per_trade` | 0.5% - 1.5% | Risk per trade |
| `risk.max_positions` | 4 - 12 | Maximum concurrent positions |
| `eligibility.trend_ma` | 20, 50, 100 | Trend filter MA period |

### How Optimization Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    BAYESIAN OPTIMIZATION                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Trial 1: Sample parameters → Run backtest → Record Sharpe      │
│  Trial 2: Sample parameters → Run backtest → Record Sharpe      │
│  ...                                                             │
│  Trial N: TPE sampler learns which regions work best            │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │    Train     │───▶│  Validation  │───▶│   Holdout    │       │
│  │   Period     │    │   (Score)    │    │   (Verify)   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
│  Walk-forward validation prevents overfitting                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Output Files

After optimization completes:

```
optimization_results/
├── strategy_opt_20260106_002443.json       # Full results
└── strategy_opt_20260106_002443_config.yaml # Optimized config (use this!)
```

### Parameter Importance

The optimizer reports which parameters matter most:

```
Parameter Importance:
  exit.exit_ma: 82.6%           ← Most impactful
  exit.hard_stop_atr_mult: 12.6%
  risk.risk_per_trade: 4.8%
```

---

## Complete Optimization Workflow

For best results, use **both** strategy optimization AND ML signal enhancement:

```
┌─────────────────────────────────────────────────────────────────┐
│                    RECOMMENDED WORKFLOW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  STEP 1: OPTIMIZE PARAMETERS                                    │
│  ────────────────────────────────                               │
│  python scripts/optimize_strategy.py --config strategy.yaml     │
│  → Output: optimized_config.yaml with best parameters           │
│                                                                  │
│  STEP 2: COLLECT ML TRAINING DATA                               │
│  ────────────────────────────────                               │
│  Run backtests with optimized config + ml_data_collection: true │
│  → Output: feature_db with labeled samples                      │
│                                                                  │
│  STEP 3: TRAIN ML MODEL                                         │
│  ────────────────────────────────                               │
│  python scripts/ml_training_pipeline.py --train                 │
│  → Output: Trained signal quality model                         │
│                                                                  │
│  STEP 4: ENABLE ML IN STRATEGY                                  │
│  ────────────────────────────────                               │
│  Add ml.enabled: true to optimized config                       │
│  → Result: Optimized params + ML signal filtering               │
│                                                                  │
│  STEP 5: CONTINUOUS IMPROVEMENT                                 │
│  ────────────────────────────────                               │
│  python scripts/ml_continuous_learning.py --cycle               │
│  → Retrains model as new data accumulates                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Why Both?

| Approach | What It Does | Limitation |
|----------|--------------|------------|
| **Optimization Only** | Finds best parameters | Same parameters for all signals |
| **ML Only** | Filters bad signals | Uses potentially suboptimal parameters |
| **Both** | Best parameters + smart filtering | Optimal combination |

---

## Extending the System

### To Add New ML Models

1. Create new model class in `trading_system/ml_refinement/models/`
2. Register in `ModelType` enum
3. Update trainer to support new model type

---

## Troubleshooting

### Model Not Loading
```
ML model path does not exist
```
**Solution**: Train a model first with `python scripts/ml_training_pipeline.py --train`

### Feature Mismatch
```
X has 44 features, but model expects 40
```
**Solution**: The `_MLModelAdapter` should handle this automatically. If persists, retrain model.

### Insufficient Samples
```
Insufficient samples: X < 50
```
**Solution**: Run more backtests with `ml_data_collection.enabled: true`

### No Improvement on Retrain
```
Kept current model: AUC improvement below threshold
```
**Expected**: System only deploys better models. Collect more data or adjust threshold.

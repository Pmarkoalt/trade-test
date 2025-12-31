# Trading System Examples

This directory contains example scripts demonstrating common workflows with the trading system.

## Available Examples

### 0. Quick Start (`quick_start_example.py`) ⭐ **START HERE**

The simplest possible backtest using minimal test data. Perfect for first-time users.

**What it shows:**
- Minimal setup with test fixtures (3 symbols, 3 months)
- Simple programmatic execution
- Quick results overview

**Run it (Docker Recommended):**
```bash
# Using Docker (recommended)
docker-compose run --rm trading-system python examples/quick_start_example.py

# Or using native installation
python examples/quick_start_example.py
```

**Key concepts:**
- `run_backtest()` - Simplest way to run a backtest
- Test fixtures - Pre-configured minimal data
- Results overview - Key metrics at a glance

**Best for:** Users new to the system who want to see results immediately.

---

### 1. All Strategies (`all_strategies_example.py`)

Demonstrates how to load and configure all available strategy types.

**What it shows:**
- Loading all 6 strategy types (Momentum Equity/Crypto, Mean Reversion, Multi-Timeframe, Factor, Pairs)
- Strategy configuration overview
- How to use strategies in backtests

**Run it (Docker Recommended):**
```bash
# Using Docker (recommended)
docker-compose run --rm trading-system python examples/all_strategies_example.py

# Or using native installation
python examples/all_strategies_example.py
```

**Key concepts:**
- `StrategyConfig` - Loading strategy configurations
- `load_strategy_from_config()` - Creating strategy instances
- Strategy registry - Available strategy types
- Multi-strategy backtests - Combining strategies

**Best for:** Understanding the different strategy types and their use cases.

---

### 2. Basic Backtest (`basic_backtest.py`)

Demonstrates how to run a simple backtest using the trading system.

**What it shows:**
- Programmatic backtest execution
- Using `BacktestRunner` for more control
- Computing detailed metrics
- CLI usage examples

**Run it (Docker Recommended):**
```bash
# Using Docker (recommended)
docker-compose run --rm trading-system python examples/basic_backtest.py

# Or using native installation
python examples/basic_backtest.py
```

**Key concepts:**
- `run_backtest()` - Convenience function for quick backtests
- `BacktestRunner` - More control over the backtest process
- `MetricsCalculator` - Compute performance metrics

---

### 3. ML Workflow (`ml_workflow.py`)

Demonstrates how to train and use ML models for signal enhancement.

**What it shows:**
- Training an ML model on historical backtest data
- Loading and using trained models
- Configuring ML integration in strategy configs
- Model versioning

**Run it (Docker Recommended):**
```bash
# Using Docker (recommended)
docker-compose run --rm trading-system python examples/ml_workflow.py

# Or using native installation
python examples/ml_workflow.py
```

**Key concepts:**
- `MLTrainer` - Train models on backtest data
- `MLPredictor` - Use models for predictions
- `MLModelVersioning` - Track model versions
- Strategy config ML settings

**Note:** This is a simplified example. In production, you would:
1. Extract features from actual backtest results
2. Create labels from trade outcomes (R-multiples, win/loss)
3. Train on train period, validate on validation period
4. Test on holdout period

---

### 4. Custom Strategy (`custom_strategy.py`)

Demonstrates how to create a custom trading strategy.

**What it shows:**
- Extending `StrategyInterface` base class
- Implementing required methods
- Registering custom strategies
- Using custom strategies in backtests

**Run it (Docker Recommended):**
```bash
# Using Docker (recommended)
docker-compose run --rm trading-system python examples/custom_strategy.py

# Or using native installation
python examples/custom_strategy.py
```

**Key concepts:**
- `StrategyInterface` - Base class for all strategies
- `register_strategy()` - Register custom strategies
- `create_strategy()` - Create strategy from config
- Required methods: `check_eligibility()`, `check_entry_triggers()`, `check_exit_signals()`, etc.

**Example strategy:** Simple Moving Average Crossover
- Entry: Price crosses above 20-day MA and 20-day MA > 50-day MA
- Exit: Price crosses below 20-day MA
- Stop: 2x ATR below entry price

---

### 5. Validation Suite (`validation_suite.py`)

Demonstrates how to run the comprehensive validation suite.

**What it shows:**
- Running bootstrap and permutation tests
- Running stress tests (slippage, bear market, range market, flash crash)
- Correlation analysis
- Interpreting validation results

**Run it (Docker Recommended):**
```bash
# Using Docker (recommended)
docker-compose run --rm trading-system python examples/validation_suite.py

# Or using native installation
python examples/validation_suite.py
```

**Key concepts:**
- `run_validation()` - Run full validation suite
- Bootstrap test - Statistical significance
- Permutation test - Randomization test
- Stress tests - Market condition tests
- Correlation analysis - Portfolio diversification

**Validation includes:**
1. **Bootstrap test** - Statistical significance of returns
2. **Permutation test** - Randomization test for strategy edge
3. **Stress tests** - Slippage, bear market, range market, flash crash
4. **Correlation analysis** - Portfolio diversification check

---

### 6. Sensitivity Analysis (`sensitivity_analysis.py`)

Demonstrates how to run parameter sensitivity analysis.

**What it shows:**
- Configuring parameter grids
- Running grid search
- Finding optimal parameters
- Interpreting results and heatmaps

**Run it (Docker Recommended):**
```bash
# Using Docker (recommended)
docker-compose run --rm trading-system python examples/sensitivity_analysis.py

# Or using native installation
python examples/sensitivity_analysis.py
```

**Key concepts:**
- `run_sensitivity_analysis()` - Run parameter grid search
- Parameter ranges in config
- Best parameters identification
- Stability analysis
- Heatmap visualization

**Note:** Sensitivity analysis can take a long time depending on parameter grid size.

---

## Quick Start

**New users should start here:**

**Using Docker (Recommended):**

1. **Build the Docker image (if not already built):**
   ```bash
   make docker-build
   ```

2. **Run the quick start example (minimal data):**
   ```bash
   docker-compose run --rm trading-system python examples/quick_start_example.py
   ```
   This uses test fixtures and runs in seconds - perfect for your first backtest!

3. **See all available strategies:**
   ```bash
   docker-compose run --rm trading-system python examples/all_strategies_example.py
   ```
   Learn about all 6 strategy types and how to use them.

4. **Run a detailed backtest:**
   ```bash
   docker-compose run --rm trading-system python examples/basic_backtest.py
   ```
   More detailed example with advanced features.

5. **Create a custom strategy:**
   ```bash
   docker-compose run --rm trading-system python examples/custom_strategy.py
   ```
   Learn how to build your own strategies.

6. **Run validation suite:**
   ```bash
   docker-compose run --rm trading-system python examples/validation_suite.py
   ```
   Test strategy robustness with statistical tests and stress tests.

**Using Native Installation (Alternative):**

If you prefer not to use Docker, replace `docker-compose run --rm trading-system` with direct Python commands:

```bash
# Run the quick start example
python examples/quick_start_example.py

# See all available strategies
python examples/all_strategies_example.py

# Run a detailed backtest
python examples/basic_backtest.py

# Create a custom strategy
python examples/custom_strategy.py

# Run validation suite
python examples/validation_suite.py
```

## Prerequisites

All examples require:
- **Docker and Docker Compose** (Recommended) - See [DOCKER_SETUP.md](../DOCKER_SETUP.md)
- OR Python 3.9+ with dependencies from `requirements.txt` (Alternative)
- Configuration files in `EXAMPLE_CONFIGS/`
- Test data (for backtests)

## Configuration Files

Examples use configuration files from `EXAMPLE_CONFIGS/`:
- `run_config.yaml` - Main backtest configuration
- `equity_config.yaml` - Equity momentum strategy configuration
- `crypto_config.yaml` - Crypto momentum strategy configuration
- `mean_reversion_config.yaml` - Mean reversion strategy configuration
- `multi_timeframe_config.yaml` - Multi-timeframe strategy configuration
- `factor_config.yaml` - Factor-based strategy configuration
- `pairs_config.yaml` - Pairs trading strategy configuration

The quick start example uses test configurations from `tests/fixtures/configs/` which point to minimal test data.

## Output

Examples generate output in:
- `results/` - Backtest results, validation results, sensitivity analysis
- `models/` - Trained ML models (if using ML workflow)

## Next Steps

After running examples:
1. Review the generated output files
2. Modify configuration files to test different parameters
3. Create your own custom strategies
4. Integrate ML models into your strategies
5. Run validation suite on your strategies

## Additional Resources

- **Main README:** `README.md` - System overview and installation
- **Testing Guide:** `TESTING_GUIDE.md` - Comprehensive testing instructions
- **Architecture Docs:** `agent-files/` - Detailed system architecture
- **API Documentation:** `docs/api/` - API reference (if generated)

## Troubleshooting

**Example fails with "Config not found":**
- Ensure you're running from the project root directory
- Check that `EXAMPLE_CONFIGS/` directory exists

**Example fails with "Data not found":**
- Examples use test data from `tests/fixtures/`
- For production, update paths in `run_config.yaml`

**ML workflow fails:**
- ML workflow is a simplified example
- In production, you need actual feature extraction from backtests
- See `trading_system/ml/` for full ML implementation

**Sensitivity analysis takes too long:**
- Reduce parameter grid size in `run_config.yaml`
- Use fewer parameter combinations
- Run on smaller time periods

## Contributing

To add a new example:
1. Create a new Python file in `examples/`
2. Follow the pattern of existing examples
3. Include docstrings and comments
4. Update this README
5. Test the example works

---

**Last Updated:** 2024-12-19


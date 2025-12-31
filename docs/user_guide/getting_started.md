# Getting Started Guide

This guide will walk you through installing, configuring, and running your first backtest with the trading system.

## Table of Contents

1. [Installation](#installation)
2. [Understanding the System](#understanding-the-system)
3. [Your First Backtest](#your-first-backtest)
4. [Understanding Results](#understanding-results)
5. [Next Steps](#next-steps)

---

## Installation

### Docker Installation (Recommended) ⭐

**Docker is the recommended installation method** as it provides a consistent environment across all systems and avoids environment-specific issues.

#### Prerequisites

- Docker Desktop installed ([Download here](https://www.docker.com/products/docker-desktop/))
- Docker Compose 2.0+ (included with Docker Desktop)

#### Step 1: Build the Docker Image

Navigate to the project directory and build the Docker image:

```bash
cd trade-test
make docker-build
```

#### Step 2: Verify Installation

Run unit tests to verify everything is working:

```bash
make docker-test-unit
```

For more detailed Docker setup instructions, see [DOCKER_SETUP.md](../../DOCKER_SETUP.md).

---

### Native Python Installation (Alternative)

If you prefer not to use Docker, you can install natively. However, this may lead to environment-specific issues, especially on macOS.

#### Prerequisites

- Python 3.9+ (3.11+ recommended)
- pip or conda package manager
- Git (optional, for cloning the repository)

#### Step 1: Install Dependencies

Navigate to the project directory:

```bash
cd trade-test
```

**Option A: Master Setup Script (Recommended)**

```bash
# Run the master setup script (handles everything)
./setup.sh
```

**Option B: Manual Installation**

```bash
# Install required packages
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

#### Step 2: Verify Installation

Run the quick test script to verify everything is working:

```bash
./quick_test.sh
```

Or manually check:

```bash
python -c "import pandas, numpy, pydantic, yaml; print('Dependencies OK')"
```

#### Step 3: Run Unit Tests (Optional)

Verify the system is functioning correctly:

```bash
pytest tests/ -v
```

---

## Understanding the System

The trading system is a config-driven backtesting framework that:

- **Generates signals** at daily close (D)
- **Executes trades** at next day open (D+1)
- **Supports multiple strategies**: Equity momentum, crypto momentum, mean reversion, pairs trading, and more
- **Uses walk-forward validation**: Train → Validation → Holdout splits
- **Includes realistic execution costs**: Slippage, fees, and capacity constraints

### Key Components

1. **Configuration Files**: YAML files define strategy parameters and run settings
2. **Data Pipeline**: Loads OHLCV data from CSV, Parquet, HDF5, or database sources
3. **Backtest Engine**: Event-driven engine that processes daily signals
4. **Portfolio Manager**: Tracks positions, cash, risk, and exposure
5. **Validation Suite**: Statistical tests and stress scenarios

---

## Your First Backtest

### Step 1: Prepare Your Data

Ensure your data files are in the correct format. Each symbol should have a CSV file with columns:

- `date`: Date (YYYY-MM-DD format)
- `symbol`: Symbol name
- `open`, `high`, `low`, `close`: OHLC prices
- `volume`: Trading volume

Example CSV structure:
```csv
date,symbol,open,high,low,close,volume
2023-01-01,AAPL,150.0,151.0,149.0,150.5,1000000
2023-01-02,AAPL,150.5,152.0,150.0,151.5,1200000
```

Place your data files in a directory structure like:
```
data/
├── equity/
│   └── ohlcv/
│       ├── AAPL.csv
│       ├── MSFT.csv
│       └── GOOGL.csv
├── crypto/
│   └── ohlcv/
│       ├── BTC.csv
│       └── ETH.csv
└── benchmarks/
    ├── SPY.csv
    └── BTC.csv
```

### Step 2: Use Example Configurations

The system comes with example configurations in `EXAMPLE_CONFIGS/`. Let's start with the test configuration:

```bash
# Copy example config to your workspace (optional)
cp EXAMPLE_CONFIGS/run_config.yaml my_config.yaml
```

### Step 3: Run Your First Backtest

**Using Docker (Recommended):**

```bash
# Basic backtest on training period
docker-compose run --rm trading-system backtest \
    --config /app/tests/fixtures/configs/run_test_config.yaml \
    --period train

# Or use the shorter alias
docker-compose run --rm trading-system bt \
    --config /app/tests/fixtures/configs/run_test_config.yaml \
    -p train
```

**Using Native Installation:**

```bash
# Basic backtest on training period
python -m trading_system backtest \
    --config tests/fixtures/configs/run_test_config.yaml \
    --period train

# Or use the shorter alias
python -m trading_system bt \
    --config tests/fixtures/configs/run_test_config.yaml \
    -p train
```

**Note:** When using Docker, config paths should be relative to `/app/` (e.g., `/app/tests/fixtures/configs/run_test_config.yaml`).

### Step 4: Check the Results

After running, results are saved to the output directory (default: `results/{run_id}/train/`):

- **`equity_curve.csv`**: Daily portfolio equity, cash, positions, exposure
- **`trade_log.csv`**: All executed trades with entry/exit details
- **`weekly_summary.csv`**: Weekly performance summaries
- **`monthly_report.json`**: Monthly metrics and statistics
- **`backtest.log`**: Execution log file

View your results:

```bash
# Navigate to results directory
cd results/<run_id>/train/

# View equity curve
cat equity_curve.csv | head -20

# View trade log
cat trade_log.csv | head -20

# View monthly report (formatted)
cat monthly_report.json | python -m json.tool
```

---

## Understanding Results

### Equity Curve

The equity curve shows your portfolio value over time:

```csv
date,equity,cash,positions,exposure
2023-10-01,100000.0,100000.0,0,0.0
2023-10-02,100500.0,95000.0,1,5000.0
```

- **equity**: Total portfolio value (cash + positions)
- **cash**: Available cash
- **positions**: Number of open positions
- **exposure**: Gross exposure (sum of position values)

### Trade Log

The trade log contains detailed information about each trade:

```csv
entry_date,exit_date,symbol,entry_price,exit_price,shares,realized_pnl,r_multiple
2023-10-02,2023-10-15,AAPL,150.5,155.0,100,450.0,2.5
```

- **entry_date/exit_date**: Trade dates
- **entry_price/exit_price**: Execution prices
- **shares**: Position size
- **realized_pnl**: Profit/loss in dollars
- **r_multiple**: Risk-adjusted return multiple

### Monthly Report (JSON)

The monthly report contains performance metrics:

```json
{
  "total_return": 0.15,
  "sharpe_ratio": 1.2,
  "max_drawdown": 0.08,
  "calmar_ratio": 1.875,
  "total_trades": 45,
  "win_rate": 0.55,
  "avg_r_multiple": 1.1
}
```

Key metrics:
- **total_return**: Total portfolio return (15% = 0.15)
- **sharpe_ratio**: Risk-adjusted return measure (higher is better)
- **max_drawdown**: Maximum peak-to-trough decline (8% = 0.08)
- **win_rate**: Percentage of winning trades (55% = 0.55)

---

## Customizing Your Configuration

### Creating a Custom Run Config

1. **Copy an example config**:

```bash
cp EXAMPLE_CONFIGS/run_config.yaml my_run_config.yaml
```

2. **Edit the configuration** to match your data paths:

```yaml
dataset:
  equity_path: "data/equity/ohlcv/"  # Your data path
  crypto_path: "data/crypto/ohlcv/"
  benchmark_path: "data/benchmarks/"

  start_date: "2023-01-01"  # Your date range
  end_date: "2024-12-31"
```

3. **Set your walk-forward splits**:

```yaml
splits:
  train_start: "2023-01-01"
  train_end: "2024-03-31"      # 15 months training
  validation_start: "2024-04-01"
  validation_end: "2024-06-30"  # 3 months validation
  holdout_start: "2024-07-01"
  holdout_end: "2024-12-31"     # 6 months holdout (LOCKED)
```

4. **Point to your strategy configs**:

```yaml
strategies:
  equity:
    config_path: "EXAMPLE_CONFIGS/equity_config.yaml"
    enabled: true
```

### Using the Config Wizard

For interactive configuration setup:

```bash
python -m trading_system config wizard --type run
```

This will guide you through creating a configuration file step-by-step.

---

## Running Different Periods

The system supports three backtest periods:

**Using Docker (Recommended):**

```bash
# Training period (parameter optimization)
docker-compose run --rm trading-system backtest -c /app/my_config.yaml -p train

# Validation period (out-of-sample testing)
docker-compose run --rm trading-system backtest -c /app/my_config.yaml -p validation

# Holdout period (final evaluation - LOCKED)
docker-compose run --rm trading-system backtest -c /app/my_config.yaml -p holdout
```

**Using Native Installation:**

```bash
# Training period (parameter optimization)
python -m trading_system backtest -c my_config.yaml -p train

# Validation period (out-of-sample testing)
python -m trading_system backtest -c my_config.yaml -p validation

# Holdout period (final evaluation - LOCKED)
python -m trading_system backtest -c my_config.yaml -p holdout
```

**Important**: The holdout period should be locked before any backtesting begins. Never adjust parameters based on holdout results.

---

## Running Validation Suite

Before deploying a strategy, run the validation suite to check robustness:

**Using Docker (Recommended):**

```bash
docker-compose run --rm trading-system validate --config /app/my_config.yaml
```

Or use the alias:

```bash
docker-compose run --rm trading-system val -c /app/my_config.yaml
```

**Using Native Installation:**

```bash
python -m trading_system validate --config my_config.yaml
```

Or use the alias:

```bash
python -m trading_system val -c my_config.yaml
```

This runs:
- **Bootstrap tests**: Statistical significance of returns
- **Permutation tests**: Random chance hypothesis testing
- **Stress tests**: Bear markets, range markets, flash crashes
- **Correlation analysis**: Portfolio diversification checks

---

## Programmatic Usage

You can also use the system programmatically. For Docker, use an interactive shell:

**Using Docker (Recommended):**

```bash
# Start an interactive shell in Docker
docker-compose run --rm trading-system /bin/bash

# Then run your Python code inside the container
python your_script.py
```

**Using Native Installation:**

```python
from trading_system.integration.runner import run_backtest, run_validation
from trading_system.configs.run_config import RunConfig

# Load configuration
config = RunConfig.from_yaml("my_config.yaml")

# Run backtest
results = run_backtest("my_config.yaml", period="train")

# Access results
print(f"Total return: {results['total_return']:.2%}")
print(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
print(f"Total trades: {results['total_trades']}")
```

For more details, see the [Examples Guide](examples.md).

---

## Next Steps

Now that you've run your first backtest:

1. **Explore Examples**: See [Examples Guide](examples.md) for common use cases
2. **Learn Best Practices**: Read [Best Practices Guide](best_practices.md)
3. **Customize Strategies**: Modify strategy configs in `EXAMPLE_CONFIGS/`
4. **Run Validation**: Validate your strategy with the validation suite
5. **Analyze Results**: Review output files and metrics

---

## Common Issues

### "Config file not found"

- Check the path to your config file
- Use absolute paths if relative paths don't work
- Ensure the file exists: `ls -la my_config.yaml`

### "No data loaded"

- Verify your data paths in the config are correct
- Check that CSV files exist and are readable
- Ensure date columns are in YYYY-MM-DD format
- Check the log file for specific errors

### "Insufficient data for indicators"

- Ensure you have at least 250 days of historical data (for indicator calculations)
- Check that `min_lookback_days` in config matches your data availability
- Verify date ranges in your config cover your data

### "No trades generated"

- Check that your strategy parameters are appropriate for the data
- Review eligibility filters (trend MA, breakout clearance, etc.)
- Verify your universe has symbols that meet the criteria
- Check the log file for strategy-specific warnings

For more troubleshooting help, see the Troubleshooting Guide (coming soon).

---

## Getting Help

- **Documentation**: See `agent-files/` for detailed technical documentation
- **Example Configs**: Check `EXAMPLE_CONFIGS/` for working configurations
- **Test Fixtures**: See `tests/fixtures/` for sample data and configs
- **CLI Help**: Run `python -m trading_system --help` or `python -m trading_system <command> --help`

---

**Congratulations!** You've successfully run your first backtest. Continue to the [Examples Guide](examples.md) to learn more advanced usage patterns.

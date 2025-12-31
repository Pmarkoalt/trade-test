# Testing Guide - Trading System V0.1

This guide explains how to test the trading system locally to verify functionality and success.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Quick Start: Running Tests Independently](#quick-start-running-tests-independently)
4. [Running Unit Tests](#running-unit-tests)
5. [Running Integration Tests](#running-integration-tests)
6. [Running CLI Commands](#running-cli-commands)
7. [Verifying System Functionality](#verifying-system-functionality)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

**Docker (Recommended):**
- Docker Desktop installed ([Download here](https://www.docker.com/products/docker-desktop/))
- Docker Compose 2.0+ (included with Docker Desktop)

**OR Native Python (Alternative):**
- Python 3.9+ (3.11+ recommended)
- pip or conda package manager

## Environment Setup

### Docker Setup (Recommended) ⭐

**Docker is the recommended testing environment** as it provides a consistent environment across all systems and avoids environment-specific issues.

#### 1. Build the Docker Image

```bash
# Build the Docker image (includes all dependencies)
make docker-build
```

#### 2. Verify Installation

```bash
# Run unit tests to verify setup
make docker-test-unit
```

For detailed Docker setup, see [DOCKER_SETUP.md](DOCKER_SETUP.md).

---

### Native Python Setup (Alternative)

If you prefer not to use Docker, you can set up a native Python environment. However, this may lead to environment-specific issues, especially on macOS.

#### 1. Install Dependencies

First, ensure you have the required Python packages installed. The system requires:

- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `pydantic` - Configuration validation
- `pytest` - Testing framework
- `pyyaml` - YAML configuration parsing

Install dependencies:

```bash
# Using pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Or install the package in editable mode with dev dependencies
pip install -e ".[dev]"
```

#### 2. Verify Installation

```bash
# Check Python version
python --version  # Should be 3.9+

# Verify core dependencies
python -c "import pandas, numpy, pydantic, yaml; print('Dependencies OK')"

# Verify pytest
pytest --version
```

## Quick Start: Running Tests Independently

The easiest way to run different test types is using Docker with the Makefile (recommended) or native Python commands.

### Using Docker with Makefile (Recommended) ⭐

```bash
# Run unit tests only
make docker-test-unit

# Run integration tests only
make docker-test-integration

# Run validation suite
make docker-test-validation

# Run unit + integration tests
make docker-test-all

# Run tests with coverage report
docker-compose run --rm --entrypoint pytest trading-system tests/ \
    --cov=trading_system --cov-report=html --cov-report=term-missing

# Run with custom pytest arguments
make docker-test-unit ARGS='-k test_name -v'

# Use custom config for validation
make docker-test-validation TEST_CONFIG=tests/fixtures/configs/run_test_config.yaml

# Show all available targets
make help
```

### Using Native Python with Makefile (Alternative)

```bash
# Run unit tests only
make test-unit

# Run integration tests only
make test-integration

# Run validation suite
make test-validation

# Run unit + integration tests
make test-all

# Run tests with coverage report
make test-coverage

# Run with custom pytest arguments
make test-unit ARGS='-k test_name -v'

# Use custom config for validation
make test-validation TEST_CONFIG=path/to/config.yaml

# Show all available targets
make help
```

### Using Shell Scripts (Native Installation Only)

```bash
# Run unit tests
./scripts/run_unit_tests.sh

# Run integration tests
./scripts/run_integration_tests.sh

# Run validation suite (with default config)
./scripts/run_validation_suite.sh

# Run validation suite (with custom config)
./scripts/run_validation_suite.sh path/to/config.yaml

# Pass additional pytest arguments to unit/integration scripts
./scripts/run_unit_tests.sh -k test_name -v
```

### What Each Test Type Does

- **Unit Tests**: Test individual components in isolation (data loading, indicators, strategies, portfolio, etc.)
- **Integration Tests**: Test end-to-end workflows and component interactions
- **Validation Suite**: Run comprehensive statistical validation (bootstrap, permutation, stress tests)

## Running Unit Tests

Unit tests verify individual components in isolation.

### Run All Unit Tests

**Using Docker (Recommended):**

```bash
# From project root
make docker-test-unit

# Or directly with docker-compose
docker-compose run --rm --entrypoint pytest trading-system tests/ \
    --ignore=tests/integration --ignore=tests/performance --ignore=tests/property -v

# With coverage report
docker-compose run --rm --entrypoint pytest trading-system tests/ \
    --ignore=tests/integration --ignore=tests/performance --ignore=tests/property \
    --cov=trading_system --cov-report=html --cov-report=term-missing

# Coverage reports will be generated in:
# - htmlcov/index.html (interactive HTML report) - accessible on host machine
# - Terminal output (shows missing lines)
```

**Using Native Installation:**

```bash
# From project root
pytest tests/

# With verbose output
pytest tests/ -v

# With coverage report
# Using helper script (recommended)
./scripts/coverage_report.sh

# Or manually
pytest tests/ --cov=trading_system --cov-report=html --cov-report=term-missing

# Coverage reports will be generated in:
# - htmlcov/index.html (interactive HTML report)
# - Terminal output (shows missing lines)
```

### Run Specific Test Files

**Using Docker (Recommended):**

```bash
# Test data loading
docker-compose run --rm --entrypoint pytest trading-system tests/test_data_loading.py -v

# Test indicators
docker-compose run --rm --entrypoint pytest trading-system tests/test_indicators.py -v

# Test strategies
docker-compose run --rm --entrypoint pytest trading-system tests/test_equity_strategy.py -v
docker-compose run --rm --entrypoint pytest trading-system tests/test_crypto_strategy.py -v

# Test portfolio
docker-compose run --rm --entrypoint pytest trading-system tests/test_portfolio.py -v

# Test execution
docker-compose run --rm --entrypoint pytest trading-system tests/test_execution.py -v

# Test backtest engine
docker-compose run --rm --entrypoint pytest trading-system tests/test_backtest_engine.py -v
```

**Using Native Installation:**

```bash
# Test data loading
pytest tests/test_data_loading.py -v

# Test indicators
pytest tests/test_indicators.py -v

# Test strategies
pytest tests/test_equity_strategy.py -v
pytest tests/test_crypto_strategy.py -v

# Test portfolio
pytest tests/test_portfolio.py -v

# Test execution
pytest tests/test_execution.py -v

# Test backtest engine
pytest tests/test_backtest_engine.py -v
```

### Run Tests by Category

```bash
# Run only unit tests (exclude integration)
pytest tests/ -k "not integration" -v

# Run specific test class
pytest tests/test_data_loading.py::TestLoadOHLCVData -v

# Run specific test method
pytest tests/test_data_loading.py::TestLoadOHLCVData::test_load_valid_data -v
```

## Running Integration Tests

Integration tests verify that components work together correctly.

**Using Docker (Recommended):**

```bash
# Run full integration test
make docker-test-integration

# Or directly with docker-compose
docker-compose run --rm --entrypoint pytest trading-system tests/integration/test_end_to_end.py -v

# Run with detailed output
docker-compose run --rm --entrypoint pytest trading-system tests/integration/test_end_to_end.py -v -s
```

**Using Native Installation:**

```bash
# Run full integration test
pytest tests/integration/test_end_to_end.py -v

# Run with detailed output
pytest tests/integration/test_end_to_end.py -v -s
```

### Expected Behavior

The integration test should:
- Load test data from `tests/fixtures/`
- Generate signals from test strategies
- Execute trades through the backtest engine
- Verify no lookahead bias
- Validate portfolio operations

## Running CLI Commands

The CLI provides commands to run full backtests and validation suites.

### 1. Run a Backtest

**Using Docker (Recommended):**

```bash
# Using the test configuration
docker-compose run --rm trading-system backtest \
    --config /app/tests/fixtures/configs/run_test_config.yaml --period train
```

**Using Native Installation:**

```bash
# Using the test configuration
python -m trading_system backtest --config tests/fixtures/configs/run_test_config.yaml --period train

# Or using the module directly
python -m trading_system.cli backtest --config tests/fixtures/configs/run_test_config.yaml
```

### 2. Run Validation Suite

**Using Docker (Recommended):**

```bash
# Run validation suite (bootstrap, permutation, stress tests)
make docker-test-validation

# Or directly with docker-compose
docker-compose run --rm trading-system validate \
    --config /app/tests/fixtures/configs/run_test_config.yaml
```

**Using Native Installation:**

```bash
# Run validation suite (bootstrap, permutation, stress tests)
python -m trading_system validate --config tests/fixtures/configs/run_test_config.yaml
```

### 3. Run Holdout Evaluation

**Using Docker (Recommended):**

```bash
# Run holdout period evaluation
docker-compose run --rm trading-system holdout \
    --config /app/tests/fixtures/configs/run_test_config.yaml
```

**Using Native Installation:**

```bash
# Run holdout period evaluation
python -m trading_system holdout --config tests/fixtures/configs/run_test_config.yaml
```

**Note:** When using Docker, config paths should be relative to `/app/` (e.g., `/app/tests/fixtures/configs/run_test_config.yaml`).

### Output Locations

Results are saved to:
- **Test runs**: `tests/results/{run_id}/{period}/`
- **Logs**: `tests/results/{run_id}/backtest.log`
- **CSV files**: `equity_curve.csv`, `trade_log.csv`, `weekly_summary.csv`
- **JSON reports**: `monthly_report.json`

## Verifying System Functionality

### Step 1: Quick Smoke Test

Run a minimal test to verify basic functionality:

```bash
# Test data loading
python -c "
from trading_system.data import load_ohlcv_data
import os
fixtures_dir = 'tests/fixtures'
data = load_ohlcv_data(fixtures_dir, ['AAPL'])
print(f'Loaded {len(data[\"AAPL\"])} rows for AAPL')
print('✓ Data loading works')
"
```

### Step 2: Run Unit Tests

```bash
# Run all unit tests
pytest tests/ -v --tb=short

# Check for failures
echo $?  # Should be 0 if all tests pass
```

### Step 3: Run Integration Test

```bash
# Run end-to-end integration test
pytest tests/integration/test_end_to_end.py -v
```

### Step 4: Run Full Backtest

```bash
# Run a complete backtest with test data
python -m trading_system backtest \
    --config tests/fixtures/configs/run_test_config.yaml \
    --period train
```

**What to verify:**
- No errors in the log file
- Output files are created in `tests/results/`
- `equity_curve.csv` contains data
- `trade_log.csv` shows executed trades (if any)
- Portfolio equity updates correctly

### Step 5: Check Results

```bash
# View equity curve
cat tests/results/*/train/equity_curve.csv | head -20

# View trade log
cat tests/results/*/train/trade_log.csv | head -20

# Check log file for errors
tail -50 tests/results/*/backtest.log
```

### Step 6: Run Validation Suite

```bash
# Run validation suite
python -m trading_system validate \
    --config tests/fixtures/configs/run_test_config.yaml
```

**What to verify:**
- Bootstrap test completes
- Permutation test completes
- No critical rejections (warnings are OK)
- Validation results saved

## Test Data

The test fixtures include:

- **Equity data**: AAPL, MSFT, GOOGL (3 months: Oct-Dec 2023)
- **Crypto data**: BTC, ETH, SOL (3 months: Oct-Dec 2023)
- **Benchmarks**: SPY, BTC
- **Configs**: Test configurations for equity and crypto strategies

See `tests/fixtures/README.md` for details.

## Expected Test Results

### Unit Tests
- All unit tests should pass
- No errors or warnings (unless expected)

### Integration Tests
- Data loads successfully
- Signals are generated
- Portfolio operations work correctly
- No lookahead bias detected

### Backtest Results
- Equity curve shows progression
- Trades are executed (may be limited with 3-month test data)
- Portfolio equity updates correctly
- No execution errors

### Validation Suite
- Bootstrap test: Should complete (may show warnings if few trades)
- Permutation test: Should complete
- Correlation analysis: Should complete (if sufficient data)

## Troubleshooting

### Issue: Import Errors

```bash
# Ensure you're in the project root
cd /path/to/trade-test

# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in development mode
pip install -e .
```

### Issue: Missing Test Data

```bash
# Verify test fixtures exist
ls tests/fixtures/*.csv

# Check config files
ls tests/fixtures/configs/*.yaml
```

### Issue: Pytest Not Found

```bash
# Install pytest
pip install pytest

# Or with conda
conda install pytest
```

### Issue: Module Not Found Errors

```bash
# Ensure trading_system package is importable
python -c "import trading_system; print('OK')"

# If not, add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: No Trades Generated

This is expected with limited test data (3 months). The system requires:
- 20+ days for breakout signals
- 50+ days for MA50 eligibility
- 200+ days for MA200 (crypto)

To get more trades, use a longer date range or adjust test data.

### Issue: Validation Suite Warnings

Warnings are normal if:
- Few trades (< 10) - statistical tests need more data
- Limited date range - some indicators need more history

These are informational, not failures.

## Next Steps

After verifying basic functionality:

1. **Expand test data**: Add more historical data for better test coverage
2. **Run on real data**: Use actual market data for more realistic testing
3. **Tune parameters**: Adjust strategy parameters based on results
4. **Add custom tests**: Create tests for your specific use cases

## Quick Reference

```bash
# Run all tests
pytest tests/ -v

# Run integration test
pytest tests/integration/ -v

# Run backtest
python -m trading_system backtest --config tests/fixtures/configs/run_test_config.yaml

# Run validation
python -m trading_system validate --config tests/fixtures/configs/run_test_config.yaml

# Generate coverage report
./scripts/coverage_report.sh

# Check results
ls -la tests/results/*/train/
```

## Additional Resources

- Test fixtures documentation: `tests/fixtures/README.md`
- Expected trades: `tests/fixtures/EXPECTED_TRADES.md`
- Example configs: `EXAMPLE_CONFIGS/README.md`
- Architecture docs: `agent-files/01_ARCHITECTURE_OVERVIEW.md`


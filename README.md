# Trading System V0.1

A config-driven daily momentum trading system for equities and cryptocurrency with walk-forward backtesting, realistic execution costs, and comprehensive validation suite.

## Overview

This trading system implements a systematic momentum strategy that:
- Generates signals at daily close (D)
- Executes trades at next day open (D+1)
- Supports both equity and cryptocurrency markets
- Uses walk-forward backtesting with train/validation/holdout splits
- Includes realistic execution costs (fees + slippage)
- Provides robust validation through statistical tests and stress scenarios

## Key Features

### 🎯 Strategy Components
- **Equity Momentum Strategy**: Breakout-based entries with MA trend filters
- **Crypto Momentum Strategy**: Similar logic with crypto-specific parameters
- **Signal Generation**: 20D and 55D breakout triggers with configurable clearances
- **Exit Management**: MA cross exits, ATR-based stop losses, staged exits (crypto)
- **Position Sizing**: Risk-based sizing (0.75% risk per trade)
- **Portfolio Management**: Correlation guards, volatility scaling, capacity constraints

### 📊 Data & Indicators
- **OHLCV Data Loading**: CSV-based with validation
- **Technical Indicators**: MA (20/50/200), ATR14, ROC60, breakout levels, ADV20
- **Feature Computation**: Automated indicator calculation pipeline
- **Data Validation**: OHLCV relationship checks, missing data handling

### 🔄 Backtesting Engine
- **Event-Driven Loop**: Daily event processing with no lookahead bias
- **Walk-Forward Splits**: Train/validation/holdout period management
- **Realistic Execution**: Slippage models, fee calculation, capacity constraints
- **Portfolio Tracking**: Equity curve, positions, cash, exposure monitoring

### ✅ Validation Suite
- **Statistical Tests**: Bootstrap analysis, permutation tests
- **Stress Tests**: Slippage stress, bear market, range market, flash crash scenarios
- **Sensitivity Analysis**: Parameter grid search
- **Correlation Analysis**: Portfolio diversification monitoring

### 📈 Reporting & Metrics
- **CSV Outputs**: Equity curve, trade log, weekly summaries
- **JSON Reports**: Monthly performance reports with metrics
- **Performance Metrics**: Sharpe ratio, Calmar ratio, max drawdown, R-multiples, profit factor
- **Benchmark Comparison**: Relative performance vs SPY/BTC

## Installation

### Prerequisites
- Python 3.9+ (3.11+ recommended)
- pip or conda package manager

### Install Dependencies

```bash
# Clone or navigate to the repository
cd trade-test

# Install required packages
pip install -r requirements.txt
```

### Verify Installation

```bash
# Quick verification
./quick_test.sh

# Or manually check
python -c "import pandas, numpy, pydantic, yaml; print('Dependencies OK')"
```

## Quick Start

### 1. Run Unit Tests

```bash
# Run all unit tests
pytest tests/ -v

# Run specific test file
pytest tests/test_data_loading.py -v
```

### 2. Run Integration Test

```bash
# End-to-end integration test
pytest tests/integration/test_end_to_end.py -v
```

### 3. Run a Backtest

```bash
# Using test configuration
python -m trading_system backtest \
    --config tests/fixtures/configs/run_test_config.yaml \
    --period train
```

Results will be saved to `tests/results/{run_id}/train/`

### 4. Run Validation Suite

```bash
# Run validation suite (bootstrap, permutation, stress tests)
python -m trading_system validate \
    --config tests/fixtures/configs/run_test_config.yaml
```

## Usage

### CLI Commands

The system provides several CLI commands:

```bash
# Run backtest
python -m trading_system backtest --config <config_path> [--period train|validation|holdout]

# Run validation suite
python -m trading_system validate --config <config_path>

# Run holdout evaluation
python -m trading_system holdout --config <config_path>

# Generate report (future)
python -m trading_system report --run-id <run_id>
```

### Configuration Files

The system uses YAML configuration files:

- **Strategy Configs** (`equity_config.yaml`, `crypto_config.yaml`): Define strategy parameters
- **Run Config** (`run_config.yaml`): Define backtest run parameters, data paths, splits

Example configurations are in:
- `EXAMPLE_CONFIGS/` - Production-ready examples
- `tests/fixtures/configs/` - Test configurations

### Programmatic Usage

```python
from trading_system.integration.runner import run_backtest, run_validation
from trading_system.configs.run_config import RunConfig

# Load configuration
config = RunConfig.from_yaml("path/to/run_config.yaml")

# Run backtest
results = run_backtest("path/to/run_config.yaml", period="train")

# Run validation
validation_results = run_validation("path/to/run_config.yaml")
```

## Project Structure

```
trade-test/
├── trading_system/          # Main package
│   ├── backtest/            # Backtest engine and event loop
│   ├── configs/             # Configuration models (Pydantic)
│   ├── data/                # Data loading and validation
│   ├── execution/           # Order execution, fills, slippage, fees
│   ├── indicators/          # Technical indicators (MA, ATR, momentum, etc.)
│   ├── integration/         # Integration runner
│   ├── models/              # Data models (Bar, Signal, Order, Position, etc.)
│   ├── portfolio/           # Portfolio management and risk
│   ├── reporting/           # CSV/JSON output and metrics
│   ├── strategies/          # Strategy implementations (equity, crypto)
│   ├── validation/          # Validation suite (bootstrap, permutation, stress)
│   └── cli.py               # Command-line interface
│
├── tests/                   # Test suite
│   ├── fixtures/           # Test data and configurations
│   ├── integration/        # Integration tests
│   └── utils/              # Test utilities and helpers
│
├── EXAMPLE_CONFIGS/         # Example configuration files
├── agent-files/            # Architecture and design documentation
│
├── requirements.txt        # Python dependencies
├── TESTING_GUIDE.md        # Comprehensive testing guide
├── QUICK_START_TESTING.md  # Quick testing reference
└── README.md               # This file
```

## Strategy Details

### Equity Strategy
- **Trend Filter**: Close > MA50, MA50 slope > 0.5% over 20 days
- **Entry**: 20D breakout (0.5% clearance) OR 55D breakout (1.0% clearance)
- **Exit**: MA20 cross below OR hard stop (2.5x ATR14)
- **Risk**: 0.75% of equity per trade
- **Capacity**: 0.5% of 20D average dollar volume

### Crypto Strategy
- **Trend Filter**: Close > MA200 (strict requirement)
- **Entry**: Same breakout triggers as equity
- **Exit**: Staged exit (MA20 warning → tighten stop → MA50 exit)
- **Stop**: 3.0x ATR14 (tightens to 2.0x after MA20 break)
- **Capacity**: 0.25% of 20D average dollar volume (stricter)

## Testing

### Quick Test
```bash
./quick_test.sh
```

### Unit Tests
```bash
pytest tests/ -v
```

### Integration Tests
```bash
pytest tests/integration/ -v
```

### Test Data
Test fixtures include 3 months of sample data (Oct-Dec 2023) for:
- **Equity**: AAPL, MSFT, GOOGL
- **Crypto**: BTC, ETH, SOL
- **Benchmarks**: SPY, BTC

See `tests/fixtures/README.md` for details.

For comprehensive testing instructions, see [TESTING_GUIDE.md](TESTING_GUIDE.md).

## Output Files

After running a backtest, results are saved to `{output_dir}/{period}/`:

- **equity_curve.csv**: Daily portfolio equity, cash, positions, exposure
- **trade_log.csv**: All executed trades with entry/exit details
- **weekly_summary.csv**: Weekly performance summaries
- **monthly_report.json**: Monthly metrics and statistics
- **backtest.log**: Execution log file

## Key Design Principles

### No Lookahead Bias
- Indicators at date `t` only use data ≤ `t`
- Signals generated at close, executed at next open
- Strict temporal ordering enforced

### Realistic Execution
- Slippage models based on ADV and volatility
- Fee calculation (1 bps per side for equity)
- Capacity constraints (order size vs ADV)
- Stress slippage during market stress

### Deterministic Results
- All randomness uses seeded RNG
- Reproducible backtests
- Configurable random seed

### Config-Driven
- Strategy parameters in YAML files
- No hardcoded values
- Easy parameter tuning and testing

## Documentation

Comprehensive documentation is available in `agent-files/`:

- **Architecture Overview**: System design and module responsibilities
- **Configuration Guide**: Parameter documentation
- **Data Pipeline**: Data loading and validation
- **Indicators Library**: Technical indicator specifications
- **Strategy Details**: Equity and crypto strategy logic
- **Backtest Engine**: Event loop and execution flow
- **Validation Suite**: Statistical and stress testing
- **Portfolio State Machine**: Portfolio update sequence

## Status

**Version**: 0.1.0

This is the initial implementation (V0.1) with:
- ✅ Core backtest engine
- ✅ Equity and crypto strategies
- ✅ Data pipeline and validation
- ✅ Execution simulation
- ✅ Portfolio management
- ✅ Validation suite
- ✅ CLI interface
- ✅ Comprehensive test suite

### Future Enhancements
- Paper trading adapters
- Additional strategy types
- Real-time data integration
- Enhanced reporting and visualization

## Contributing

This is a V0.1 implementation. For questions or issues:
1. Review the documentation in `agent-files/`
2. Check `TESTING_GUIDE.md` for testing procedures
3. Review test fixtures and examples

## License

[Add your license information here]

## Acknowledgments

Built with:
- Python 3.9+
- pandas & numpy for data processing
- pydantic for configuration validation
- pytest for testing

---

For detailed testing instructions, see [TESTING_GUIDE.md](TESTING_GUIDE.md)  
For quick testing reference, see [QUICK_START_TESTING.md](QUICK_START_TESTING.md)


# Frequently Asked Questions (FAQ)

## Table of Contents

1. [Getting Started](#getting-started)
2. [Configuration](#configuration)
3. [Data](#data)
4. [Strategies](#strategies)
5. [Backtesting](#backtesting)
6. [Validation & Testing](#validation--testing)
7. [Performance & Optimization](#performance--optimization)
8. [Troubleshooting](#troubleshooting)
9. [Docker](#docker)
10. [Advanced Topics](#advanced-topics)

---

## Getting Started

### Q: What is this trading system?

**A:** This is a config-driven daily momentum trading system for equities and cryptocurrency with walk-forward backtesting, realistic execution costs, and comprehensive validation suite. It generates signals at daily close (D) and executes trades at next day open (D+1).

### Q: What Python version do I need?

**A:** Python 3.9+ is required, with Python 3.11+ recommended for best performance.

### Q: How do I install the system?

**A:** 
```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
./quick_test.sh
```

See the [README.md](README.md) for detailed installation instructions, including Docker setup.

### Q: Where do I start?

**A:** 
1. Run the quick test: `./quick_test.sh`
2. Review example configs in `EXAMPLE_CONFIGS/`
3. Run a simple backtest: `python -m trading_system backtest --config tests/fixtures/configs/run_test_config.yaml --period train`
4. Read the [README.md](README.md) and documentation in `agent-files/`

### Q: What are the main CLI commands?

**A:** The system provides several commands:
- `backtest` - Run a backtest on a specific period (train/validation/holdout)
- `validate` - Run the validation suite (bootstrap, permutation, stress tests)
- `holdout` - Run holdout evaluation
- `report` - Generate reports from previous runs

See `python -m trading_system --help` for details.

---

## Configuration

### Q: What configuration files do I need?

**A:** You need two types of config files:
1. **Strategy Config** (`equity_config.yaml`, `crypto_config.yaml`, etc.) - Defines strategy parameters
2. **Run Config** (`run_config.yaml`) - Defines backtest run parameters, data paths, and walk-forward splits

Example configs are in `EXAMPLE_CONFIGS/` directory.

### Q: How do I create a new configuration?

**A:** 
1. Copy an example config from `EXAMPLE_CONFIGS/`
2. Modify the parameters for your use case
3. Validate it: `python -m trading_system config validate --path your_config.yaml`
4. Or use the config wizard: `python -m trading_system config wizard`

### Q: What's the difference between equity and crypto configs?

**A:** 
- **Equity**: Uses MA50 trend filter, 0.5% capacity limit, 1 bps fees
- **Crypto**: Uses MA200 trend filter (stricter), 0.25% capacity limit, staged exits, different stop loss logic

See the [README.md](README.md) Strategy Details section for specifics.

### Q: How do I configure walk-forward splits?

**A:** In your `run_config.yaml`, specify:
```yaml
splits:
  train_start: "2020-01-01"
  train_end: "2022-12-31"
  validation_start: "2023-01-01"
  validation_end: "2023-06-30"
  holdout_start: "2023-07-01"
  holdout_end: "2023-12-31"
```

### Q: Can I use custom strategy parameters?

**A:** Yes! All strategy parameters are configurable in the strategy config YAML files. See `EXAMPLE_CONFIGS/` for examples of different strategy types (momentum, mean reversion, pairs, multi-timeframe, factor).

### Q: What happens if my config has errors?

**A:** The system will validate your config and show specific error messages with:
- Which field has the error
- What the expected format/value is
- Where in the config file the error occurred

Use `python -m trading_system config validate --path your_config.yaml` to check before running.

---

## Data

### Q: What data format do I need?

**A:** OHLCV (Open, High, Low, Close, Volume) CSV files with:
- Date column (YYYY-MM-DD format)
- Columns: `date`, `open`, `high`, `low`, `close`, `volume`
- Valid OHLC relationships (High >= Low, High >= Open/Close, etc.)

### Q: Where do I put my data files?

**A:** Specify the data directory in your `run_config.yaml`:
```yaml
data:
  data_dir: "path/to/your/data"
  universe_file: "universe.csv"  # List of symbols
```

### Q: What data sources are supported?

**A:** The system supports multiple data sources:
- CSV files (most common)
- Database (PostgreSQL, MySQL, SQLite)
- API sources (with adapters)
- Parquet files
- HDF5 files

See `trading_system/data/sources/` for available sources.

### Q: How do I handle missing data?

**A:** The system automatically handles missing data:
- Missing days are forward-filled (last known value)
- Missing values within a day are interpolated
- Invalid OHLC relationships are flagged and can be filtered

See `trading_system/data/validator.py` for validation rules.

### Q: Can I use different timeframes?

**A:** Yes! The system supports multi-timeframe strategies. See `EXAMPLE_CONFIGS/multi_timeframe_config.yaml` for an example.

### Q: What if my data has gaps or invalid values?

**A:** The data validator will:
- Check OHLC relationships
- Flag missing days
- Validate date formats
- Check for duplicate dates

Errors will be reported with specific dates and symbols. Fix the data files and re-run.

---

## Strategies

### Q: What strategies are available?

**A:** The system includes:
- **Equity Momentum** - Breakout-based with MA trend filters
- **Crypto Momentum** - Similar with crypto-specific parameters
- **Mean Reversion** - Mean reversion strategies
- **Pairs Trading** - Statistical arbitrage
- **Multi-Timeframe** - Strategies using multiple timeframes
- **Factor-Based** - Factor model strategies

See `trading_system/strategies/` for implementations.

### Q: How do I create a custom strategy?

**A:** 
1. Create a new strategy class inheriting from `BaseStrategy`
2. Implement the required methods (generate_signals, etc.)
3. Register it in `trading_system/strategies/strategy_registry.py`
4. Create a config file for your strategy

See existing strategies in `trading_system/strategies/` for examples.

### Q: How does position sizing work?

**A:** The system uses risk-based position sizing:
- Default: 0.75% of equity risked per trade
- Position size = (Risk Amount) / (Entry Price - Stop Loss Price)
- Capacity constraints limit position size based on average dollar volume

### Q: How are stop losses calculated?

**A:** Stop losses are ATR-based:
- **Equity**: 2.5x ATR14 below entry
- **Crypto**: 3.0x ATR14 (tightens to 2.0x after MA20 break)

These are configurable in strategy configs.

### Q: What's the difference between equity and crypto strategies?

**A:** 
- **Equity**: MA50 trend filter, 0.5% capacity, simpler exit logic
- **Crypto**: MA200 trend filter (stricter), 0.25% capacity, staged exits (MA20 warning → tighten stop → MA50 exit)

Crypto strategies are more conservative due to higher volatility.

---

## Backtesting

### Q: How does the backtest engine work?

**A:** The engine uses an event-driven loop:
1. Processes each day in chronological order
2. Generates signals at daily close (D)
3. Executes trades at next day open (D+1)
4. Applies realistic execution costs (slippage + fees)
5. Updates portfolio state
6. No lookahead bias - indicators only use data ≤ current date

### Q: What execution costs are included?

**A:** 
- **Slippage**: Based on ADV (average dollar volume) and volatility
- **Fees**: 1 bps per side for equity (configurable)
- **Capacity constraints**: Order size limited by ADV percentage
- **Stress slippage**: Higher slippage during market stress

### Q: How do I run a backtest?

**A:** 
```bash
python -m trading_system backtest \
    --config path/to/run_config.yaml \
    --period train
```

Use `--period train|validation|holdout` to specify which period to test.

### Q: Where are results saved?

**A:** Results are saved to `{output_dir}/{run_id}/{period}/`:
- `equity_curve.csv` - Daily portfolio equity, cash, positions
- `trade_log.csv` - All executed trades
- `weekly_summary.csv` - Weekly performance summaries
- `monthly_report.json` - Monthly metrics
- `backtest.log` - Execution log

### Q: How long does a backtest take?

**A:** Depends on:
- Number of symbols
- Date range length
- Strategy complexity
- System resources

Typical backtests (100 symbols, 3 years) take 1-5 minutes.

### Q: Can I run multiple backtests in parallel?

**A:** Yes, you can run multiple backtests with different configs simultaneously. Each run gets a unique `run_id` and writes to separate directories.

### Q: What metrics are calculated?

**A:** The system calculates comprehensive metrics:
- **Returns**: Total return, annualized return, monthly returns
- **Risk**: Sharpe ratio, Calmar ratio, max drawdown, volatility
- **Trade Statistics**: Win rate, profit factor, R-multiples, average trade
- **Benchmark Comparison**: Relative performance vs SPY/BTC

See `trading_system/reporting/metrics.py` for full list.

---

## Validation & Testing

### Q: What is the validation suite?

**A:** The validation suite includes:
- **Bootstrap Analysis**: Statistical significance testing
- **Permutation Tests**: Randomization-based significance
- **Stress Tests**: Slippage stress, bear market, range market, flash crash
- **Sensitivity Analysis**: Parameter grid search

### Q: How do I run validation?

**A:** 
```bash
python -m trading_system validate --config path/to/run_config.yaml
```

This runs all validation tests and reports results.

### Q: What do validation results mean?

**A:** 
- **Bootstrap**: P-value indicates probability results are due to chance
- **Permutation**: Tests if strategy outperforms random trading
- **Stress Tests**: Shows performance under adverse conditions
- **Sensitivity**: Identifies parameter robustness

See `trading_system/validation/` for details.

### Q: How do I run tests?

**A:** 
```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_data_loading.py -v

# Integration tests
pytest tests/integration/ -v
```

### Q: What test data is available?

**A:** Test fixtures include 3 months of sample data (Oct-Dec 2023):
- **Equity**: AAPL, MSFT, GOOGL
- **Crypto**: BTC, ETH, SOL
- **Benchmarks**: SPY, BTC

See `tests/fixtures/README.md` for details.

---

## Performance & Optimization

### Q: How can I speed up backtests?

**A:** 
1. Reduce date range or number of symbols
2. Use faster data sources (Parquet/HDF5 instead of CSV)
3. Disable detailed logging
4. Run on faster hardware
5. Use Docker for consistent performance

### Q: How much memory does the system use?

**A:** Depends on:
- Number of symbols
- Date range length
- Indicator calculations

Typical usage: 500MB - 2GB for 100 symbols over 3 years.

### Q: Can I optimize strategy parameters?

**A:** Yes! Use sensitivity analysis:
```bash
python -m trading_system sensitivity --config path/to/config.yaml
```

This runs a grid search over parameter ranges.

### Q: How do I compare different strategies?

**A:** 
1. Run backtests with different strategy configs
2. Use the report command to compare results:
```bash
python -m trading_system report --run-id <run_id>
```

### Q: What's the best way to tune parameters?

**A:** 
1. Start with example configs
2. Run sensitivity analysis to find promising ranges
3. Use walk-forward validation (train → validation → holdout)
4. Avoid overfitting to training data
5. Test on holdout period for final validation

---

## Troubleshooting

### Q: I get "ConfigurationError" - what does this mean?

**A:** Your config file has validation errors. The error message will show:
- Which field is invalid
- What the expected format is
- Where in the file the error is

Fix the config and validate again: `python -m trading_system config validate --path your_config.yaml`

### Q: I get "DataError" or "DataNotFoundError" - what's wrong?

**A:** 
- Check data file paths in your config
- Verify data files exist and are readable
- Check file format (CSV with correct columns)
- Validate OHLC relationships

### Q: I get "StrategyError" - what should I check?

**A:** 
- Verify strategy type matches asset class (equity vs crypto)
- Check strategy configuration parameters
- Ensure strategy is registered in the system

### Q: My backtest is very slow - why?

**A:** 
- Check date range length
- Reduce number of symbols
- Check if detailed logging is enabled
- Verify data source performance (CSV can be slow for large datasets)

### Q: I get "InsufficientCapitalError" - what does this mean?

**A:** Your portfolio doesn't have enough capital for the requested position size. Check:
- Initial capital in run config
- Position sizing parameters
- Risk per trade settings

### Q: Results look wrong - how do I debug?

**A:** 
1. Check the log file: `{output_dir}/{run_id}/{period}/backtest.log`
2. Review trade log: `trade_log.csv`
3. Check equity curve: `equity_curve.csv`
4. Verify data quality and dates
5. Check for lookahead bias (shouldn't happen, but verify)

### Q: How do I see detailed error information?

**A:** 
- Check log files in the output directory
- Use `--verbose` flag for more output
- Review exception messages (they include context like date, symbol, step)

### Q: My data has missing days - is that OK?

**A:** The system handles missing days by forward-filling. However:
- Too many missing days can affect indicator calculations
- Check the data validator warnings
- Consider filling gaps in your data preprocessing

### Q: How do I check if my backtest has lookahead bias?

**A:** The system is designed to prevent lookahead bias:
- Indicators only use data ≤ current date
- Signals generated at close, executed at next open
- Strict temporal ordering

If you suspect issues, check the event loop logic in `trading_system/backtest/event_loop.py`.

---

## Docker

### Q: How do I use Docker?

**A:** 
```bash
# Build image
docker build -t trading-system:latest .

# Run backtest
docker run --rm -v $(pwd)/EXAMPLE_CONFIGS:/app/configs:ro \
  -v $(pwd)/results:/app/results \
  trading-system:latest backtest --config /app/configs/run_config.yaml

# Or use docker-compose
docker-compose run --rm trading-system backtest --config /app/configs/run_config.yaml
```

### Q: What volumes do I need to mount?

**A:** 
- `./data` → `/app/data` (read-only) - Input data files
- `./EXAMPLE_CONFIGS` → `/app/configs` (read-only) - Configuration files
- `./results` → `/app/results` (read-write) - Output results
- `./tests/fixtures` → `/app/tests/fixtures` (read-only) - Test data

### Q: Can I use Docker for development?

**A:** Yes! Use:
```bash
docker-compose run --rm trading-system /bin/bash
```

This gives you an interactive shell in the container.

---

## Advanced Topics

### Q: How do I integrate ML models?

**A:** ML infrastructure exists (`trading_system/ml/`) but is not yet integrated into the backtest event loop. To integrate:
1. Use `MLPredictor` from `trading_system/ml/predictor.py`
2. Add ML config options to strategy configs
3. Integrate into `DailyEventLoop` for signal enhancement

See `agent-files/REVIEW_SUMMARY.md` for details on ML integration status.

### Q: Can I use this for live trading?

**A:** The system includes paper trading adapters (Alpaca, IB) and real-time trading infrastructure, but:
- **Use at your own risk** - This is a backtesting system
- Test thoroughly before live trading
- Start with paper trading
- Monitor closely

### Q: How do I add a new data source?

**A:** 
1. Create a new source class inheriting from `BaseDataSource`
2. Implement required methods (`load_data`, etc.)
3. Register it in the data loader
4. Update config to use your source

See `trading_system/data/sources/` for examples.

### Q: How do I add custom indicators?

**A:** 
1. Create indicator function in `trading_system/indicators/`
2. Register it in `FeatureComputer`
3. Use it in your strategy config

See existing indicators for examples.

### Q: Can I backtest multiple strategies simultaneously?

**A:** Yes, you can run multiple backtests with different strategy configs. Each gets its own run_id and results directory.

### Q: How do I export results to a database?

**A:** The system includes results storage infrastructure. See `trading_system/storage/` for database integration.

### Q: What's the difference between train, validation, and holdout periods?

**A:** 
- **Train**: Period for strategy development and parameter tuning
- **Validation**: Period for out-of-sample validation (don't tune on this!)
- **Holdout**: Final test period (never tune on this - final validation only)

This prevents overfitting and provides realistic performance estimates.

---

## Still Have Questions?

- Review the [README.md](README.md) for overview
- Check documentation in `agent-files/` for detailed architecture
- Review example configs in `EXAMPLE_CONFIGS/`
- Check test files in `tests/` for usage examples
- Review `agent-files/REVIEW_SUMMARY.md` for implementation status

---

**Last Updated**: 2024-12-19


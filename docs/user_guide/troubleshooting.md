# Troubleshooting Guide

This guide helps you diagnose and resolve common issues when using the trading system.

## Table of Contents

1. [Quick Diagnostic Steps](#quick-diagnostic-steps)
2. [Configuration Errors](#configuration-errors)
3. [Data Errors](#data-errors)
4. [Strategy Errors](#strategy-errors)
5. [Backtest Errors](#backtest-errors)
6. [Data Quality Issues](#data-quality-issues)
7. [Performance Troubleshooting](#performance-troubleshooting)
8. [Debugging Tips](#debugging-tips)
9. [Common Error Messages](#common-error-messages)

---

## Quick Diagnostic Steps

If you encounter an error, follow these steps in order:

### Step 1: Validate Your Configuration

```bash
python -m trading_system config validate --path <your_config.yaml>
```

This will catch most configuration errors early.

### Step 2: Check Data Files

Verify that:
- Data files exist at the paths specified in your config
- Data files are valid CSV files with proper format
- Required columns are present: `date`, `open`, `high`, `low`, `close`, `volume`

### Step 3: Check Log Files

Log files are written to:
- `{output_dir}/{run_id}/{period}/backtest.log` - Main execution log
- `{output_dir}/{run_id}/{period}/equity_curve.csv` - Portfolio state over time

Review the log file for detailed error messages and warnings.

### Step 4: Run Unit Tests

```bash
# Quick test
./quick_test.sh

# Full test suite
pytest tests/ -v
```

This verifies your environment is set up correctly.

---

## Configuration Errors

### Error: `ConfigurationError`

**Symptoms:**
```
Configuration error: Invalid configuration file
```

**Common Causes:**
1. **Invalid YAML syntax** - Missing quotes, incorrect indentation, trailing commas
2. **Missing required fields** - Required config fields not specified
3. **Invalid field values** - Values don't match expected types (e.g., string instead of number)
4. **Invalid date formats** - Dates not in YYYY-MM-DD format
5. **Invalid file paths** - Paths in config don't exist (though these may also show as DataError)

**Solutions:**

1. **Validate your config file:**
   ```bash
   python -m trading_system config validate --path <config.yaml>
   ```

2. **Check YAML syntax:**
   - Use a YAML validator (online or VS Code extension)
   - Ensure proper indentation (2 spaces, no tabs)
   - Quote strings with special characters
   - Check for trailing commas (not allowed in YAML)

3. **Review example configs:**
   ```bash
   # Compare with working examples
   cat EXAMPLE_CONFIGS/run_config.yaml
   cat EXAMPLE_CONFIGS/equity_config.yaml
   ```

4. **Check required fields:**
   - `run_config.yaml` requires: `data_paths`, `start_date`, `end_date`, `output_dir`
   - Strategy configs require: `strategy_type`, `asset_class`, `entry`, `exit`, `risk_management`

5. **Fix date formats:**
   - Use `YYYY-MM-DD` format (e.g., `2023-01-01`)
   - Ensure dates are logical (start_date < end_date)

**Example Error Message:**
```
Configuration error: Invalid value for 'start_date': Expected YYYY-MM-DD format, got '01/01/2023'
```

**Fix:**
```yaml
# Wrong
start_date: 01/01/2023

# Correct
start_date: 2023-01-01
```

---

## Data Errors

### Error: `DataError` or `DataNotFoundError`

**Symptoms:**
```
Data error: Data file not found: /path/to/data/AAPL.csv
Data error: Symbol 'AAPL' not found in data source
```

**Common Causes:**
1. **File not found** - Data file path doesn't exist
2. **Invalid file format** - CSV file is malformed or empty
3. **Missing columns** - Required columns (open, high, low, close, volume) missing
4. **Wrong file path** - Path in config doesn't match actual file location
5. **Symbol not in data source** - Symbol specified in universe but not in data files

**Solutions:**

1. **Verify file paths:**
   ```bash
   # Check if file exists
   ls -la /path/to/data/AAPL.csv

   # Check config path
   cat <config.yaml> | grep -A 5 data_paths
   ```

2. **Check file format:**
   ```bash
   # Quick check
   head -5 /path/to/data/AAPL.csv

   # Should see columns like:
   # date,open,high,low,close,volume
   # 2023-01-01,150.0,152.0,149.0,151.0,1000000
   ```

3. **Verify required columns:**
   ```python
   import pandas as pd
   df = pd.read_csv('/path/to/data/AAPL.csv', index_col='date')
   print(df.columns)  # Should include: open, high, low, close, volume
   ```

4. **Check symbol names:**
   - Ensure symbols in universe match filenames (case-sensitive)
   - For CSV source: filename should be `{SYMBOL}.csv` (e.g., `AAPL.csv`)

5. **Use absolute paths:**
   ```yaml
   # Instead of relative paths
   equity_path: ./data/equity

   # Use absolute paths
   equity_path: /full/path/to/data/equity
   ```

### Error: `DataValidationError`

**Symptoms:**
```
Data validation error: Invalid OHLC data for AAPL at 2023-01-15
```

**Common Causes:**
1. **Invalid OHLC relationships** - low > high, or open/close outside [low, high]
2. **Negative prices or volumes** - Prices or volumes are negative
3. **Non-chronological dates** - Dates are not in ascending order
4. **Duplicate dates** - Multiple rows with the same date
5. **Extreme moves** - Price moves >50% in one day (likely data error)

**Solutions:**

1. **Check OHLC relationships:**
   ```python
   import pandas as pd
   df = pd.read_csv('AAPL.csv', index_col='date')

   # Check for invalid relationships
   invalid = (df['low'] > df['high']) | \
             (df['open'] < df['low']) | (df['open'] > df['high']) | \
             (df['close'] < df['low']) | (df['close'] > df['high'])

   print(df[invalid])  # Shows problematic rows
   ```

2. **Fix data quality issues:**
   - **Negative values**: Remove or fix rows with negative prices/volumes
   - **OHLC violations**: Check if data source has errors, may need to re-download
   - **Date ordering**: Sort by date: `df.sort_index(inplace=True)`
   - **Duplicate dates**: Remove duplicates: `df[~df.index.duplicated(keep='first')]`

3. **Handle extreme moves:**
   - Extreme moves (>50% in one day) are warnings, not errors
   - Review these dates manually to determine if real market events or data errors
   - Consider removing obviously incorrect data points

4. **Use the validator directly:**
   ```python
   from trading_system.data.validator import validate_ohlcv
   import pandas as pd

   df = pd.read_csv('AAPL.csv', index_col='date')
   if not validate_ohlcv(df, 'AAPL'):
       # Check logs for specific issues
       pass
   ```

---

## Strategy Errors

### Error: `StrategyError`

**Symptoms:**
```
Strategy error: Strategy 'EquityMomentumStrategy' not found
Strategy error: Invalid strategy configuration for symbol AAPL
```

**Common Causes:**
1. **Strategy not registered** - Strategy class not imported or registered
2. **Asset class mismatch** - Using equity strategy for crypto symbols or vice versa
3. **Invalid strategy parameters** - Parameter values outside valid ranges
4. **Missing required parameters** - Required strategy config fields missing

**Solutions:**

1. **Check strategy type and asset class:**
   ```yaml
   # In strategy config
   strategy_type: EquityMomentumStrategy  # Must match registered strategy
   asset_class: equity  # Must match data source (equity or crypto)
   ```

2. **Verify strategy registration:**
   ```python
   from trading_system.strategies.strategy_registry import get_strategy_class

   # Check if strategy exists
   strategy_class = get_strategy_class('EquityMomentumStrategy')
   print(strategy_class)
   ```

3. **Review strategy parameters:**
   - Check parameter ranges (e.g., percentages should be 0-1, not 0-100)
   - Ensure required fields are present (entry, exit, risk_management)
   - Compare with example configs in `EXAMPLE_CONFIGS/`

4. **Check asset class matching:**
   - Equity strategies only work with equity data (from `equity_path`)
   - Crypto strategies only work with crypto data (from `crypto_path`)
   - Ensure symbols in universe match the asset class

**Example Error:**
```
Strategy error: Invalid strategy configuration: entry.clearance_20d must be between 0 and 1, got 50
```

**Fix:**
```yaml
# Wrong (using percentage as integer)
entry:
  clearance_20d: 50

# Correct (using decimal)
entry:
  clearance_20d: 0.50  # 50%
```

---

## Backtest Errors

### Error: `BacktestError`

**Symptoms:**
```
Backtest error: Unexpected error during backtest execution
Backtest error at date: 2023-01-15, step: process_signals
```

**Common Causes:**
1. **Missing data for specific date** - Data missing for date in backtest range
2. **Indicator calculation failure** - Indicator can't compute with available data
3. **Portfolio error** - Portfolio state inconsistency or insufficient capital
4. **Execution error** - Order execution or fill simulation fails
5. **Out of memory** - Large universe or long date range exceeds memory

**Solutions:**

1. **Check log file for specific error:**
   ```bash
   tail -100 {output_dir}/{run_id}/train/backtest.log
   ```

2. **Review error context:**
   - Error message includes `date` and `step` attributes
   - Check what happened at that specific date
   - Look for related warnings before the error

3. **Verify data availability:**
   ```python
   from trading_system.data.loader import DataLoader
   from trading_system.configs.run_config import RunConfig

   config = RunConfig.from_yaml('config.yaml')
   loader = DataLoader(config.data_paths)
   data = loader.load_universe(
       config.universe.equity_symbols + config.universe.crypto_symbols,
       config.start_date,
       config.end_date
   )

   # Check date coverage for each symbol
   for symbol, df in data.items():
       print(f"{symbol}: {df.index.min()} to {df.index.max()}")
   ```

4. **Check for insufficient data:**
   - Some indicators require minimum history (e.g., MA200 needs 200 days)
   - Ensure data covers full date range plus required warm-up period
   - Reduce date range or add more historical data

5. **Memory issues:**
   - Reduce universe size (fewer symbols)
   - Reduce date range
   - Use lazy loading (if implemented)
   - Increase system RAM

6. **Common backtest date/step errors:**
   - **`load_data`**: Data loading failed - check file paths and data quality
   - **`compute_features`**: Indicator calculation failed - check data availability
   - **`generate_signals`**: Signal generation failed - check strategy config
   - **`process_signals`**: Signal processing failed - check portfolio state
   - **`update_portfolio`**: Portfolio update failed - check for state inconsistencies

---

## Data Quality Issues

### Missing Data

**Symptoms:**
- Warnings in log: `Warning: Missing data for AAPL on 2023-01-15`
- Gaps in equity curve
- Fewer trades than expected

**Impact:**
- **1 day missing**: System skips signal updates for that symbol on that day
- **2+ consecutive days missing**: Symbol marked as "unhealthy", position exited if held

**Solutions:**

1. **Check for missing dates:**
   ```python
   from trading_system.data.validator import detect_missing_data
   import pandas as pd

   df = pd.read_csv('AAPL.csv', index_col='date', parse_dates=True)
   missing = detect_missing_data(df, 'AAPL', asset_class='equity')
   print(f"Missing dates: {missing['missing_dates']}")
   print(f"Consecutive gaps: {missing['consecutive_gaps']}")
   ```

2. **Fill missing data:**
   - **For equities**: Missing weekdays are normal (holidays, weekends)
   - **For crypto**: Missing days may indicate data source issues
   - Consider forward-filling or using data source that handles holidays

3. **Review data source:**
   - Check if data source properly handles market holidays
   - Verify data source is up-to-date
   - Consider using a different data source

### Extreme Price Moves

**Symptoms:**
- Warnings: `Extreme moves (>50%) at dates: [...]`
- Unrealistic returns in backtest results

**Impact:**
- Extreme moves are warnings, not errors (system continues)
- May indicate data errors that affect backtest accuracy

**Solutions:**

1. **Verify if real market events:**
   - Stock splits, dividends, corporate actions
   - Flash crashes or market anomalies
   - Data errors from source

2. **Handle splits/dividends:**
   - Use adjusted prices if available
   - Adjust historical prices for splits
   - Consider using a data source that provides adjusted prices

3. **Manual review:**
   ```python
   import pandas as pd
   df = pd.read_csv('AAPL.csv', index_col='date', parse_dates=True)

   returns = df['close'].pct_change()
   extreme = abs(returns) > 0.50

   print(df[extreme])  # Review these dates manually
   ```

### Invalid OHLC Relationships

**Symptoms:**
- Error: `Invalid OHLC at dates: [...]`
- Data validation fails

**Common Issues:**
- `low > high` - Impossible price relationship
- `open < low` or `open > high` - Open outside daily range
- `close < low` or `close > high` - Close outside daily range

**Solutions:**

1. **Check data source:**
   - Verify data source quality
   - May need to re-download data
   - Report to data provider if systematic issue

2. **Manual correction (if isolated):**
   ```python
   # Fix individual rows
   df.loc['2023-01-15', 'high'] = max(df.loc['2023-01-15', ['open', 'close']])
   df.loc['2023-01-15', 'low'] = min(df.loc['2023-01-15', ['open', 'close']])
   ```

3. **Prefer high-quality data sources:**
   - Use reputable data providers
   - Verify data quality before backtesting
   - Consider data validation scripts

---

## Performance Troubleshooting

### Slow Backtest Execution

**Symptoms:**
- Backtest takes hours or days to complete
- System uses 100% CPU but progress is slow
- Memory usage continuously increases

**Common Causes:**
1. **Large universe** - Too many symbols to process
2. **Long date range** - Many years of data
3. **Complex indicators** - Expensive indicator calculations
4. **Inefficient data loading** - Loading all data into memory at once
5. **No caching** - Repeated indicator calculations

**Solutions:**

1. **Reduce universe size:**
   - Test with fewer symbols first (5-10 symbols)
   - Scale up gradually
   - Use universe filtering/filtering criteria

2. **Reduce date range:**
   - Start with shorter periods (3-6 months)
   - Use walk-forward splits to test longer periods incrementally

3. **Enable indicator caching:**
   ```python
   # Caching is enabled by default in feature_computer
   # Check if cache is working:
   from trading_system.indicators.cache import IndicatorCache

   cache = IndicatorCache(max_size=1000)
   # Cache will automatically store computed indicators
   ```

4. **Use parallel processing:**
   ```python
   # For multi-symbol computations
   from trading_system.indicators.parallel import compute_features_parallel

   # Parallel computation for multiple symbols
   results = compute_features_parallel(symbols, data, config)
   ```

5. **Profile performance:**
   ```python
   # Use built-in profiling
   from trading_system.indicators.profiling import IndicatorProfiler

   profiler = IndicatorProfiler()
   # Profiling will show which indicators take longest
   ```

6. **Memory optimization:**
   - Use lazy loading (if available)
   - Process symbols in batches
   - Clear unused data structures

**Expected Performance:**
- Small universe (5-10 symbols), 1 year: ~1-5 minutes
- Medium universe (20-50 symbols), 1 year: ~10-30 minutes
- Large universe (100+ symbols), 1 year: ~1-3 hours

### High Memory Usage

**Symptoms:**
- System runs out of memory (OOM errors)
- Memory usage grows continuously
- System becomes unresponsive

**Solutions:**

1. **Reduce universe size:**
   - Process fewer symbols at once
   - Use batch processing

2. **Reduce date range:**
   - Shorter backtest periods
   - Use walk-forward analysis instead of single long period

3. **Check for memory leaks:**
   ```python
   # Use memory profiler
   from trading_system.data.memory_profiler import MemoryProfiler

   profiler = MemoryProfiler()
   # Monitor memory usage during backtest
   ```

4. **Clear intermediate data:**
   - Delete data structures that are no longer needed
   - Use generators instead of lists where possible

5. **Increase system RAM:**
   - For large universes (100+ symbols), 16GB+ RAM recommended
   - Consider using a machine with more memory

### Slow Indicator Calculations

**Symptoms:**
- Individual indicator calculations take long time
- Backtest progress is slow during feature computation phase

**Solutions:**

1. **Use optimized indicator implementations:**
   - System uses vectorized operations where possible
   - Indicators are optimized for performance

2. **Enable caching:**
   - Indicator results are cached automatically
   - Cache persists across multiple backtest runs

3. **Profile specific indicators:**
   ```python
   from trading_system.indicators.profiling import IndicatorProfiler

   profiler = IndicatorProfiler()
   # Run backtest with profiling enabled
   # Check which indicators are slowest
   ```

4. **Consider reducing indicator complexity:**
   - Remove unnecessary indicators
   - Use simpler versions where acceptable

---

## Debugging Tips

### Enable Verbose Logging

**Set log level to DEBUG:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or in config (if supported):
```yaml
logging:
  level: DEBUG
```

### Use Interactive Debugger

**Add breakpoints:**
```python
import pdb; pdb.set_trace()  # Python debugger
# Or use ipdb for better experience
import ipdb; ipdb.set_trace()
```

**In Jupyter/IPython:**
```python
%pdb on  # Automatic debugger on error
```

### Inspect Intermediate States

**Check portfolio state:**
```python
# During backtest, log portfolio state
print(f"Date: {date}, Cash: {portfolio.cash}, Positions: {len(portfolio.positions)}")
```

**Check data at specific date:**
```python
# Inspect data for specific symbol and date
symbol = 'AAPL'
date = pd.Timestamp('2023-01-15')
bar = data[symbol].loc[date]
print(bar)
```

**Check computed features:**
```python
# Verify indicator values
features = feature_computer.compute_features_for_date(data[symbol], symbol, date)
print(features)
```

### Compare with Expected Results

**Use test fixtures:**
```python
# Compare your results with known good results
from tests.utils.assertions import assert_trade_log_matches
assert_trade_log_matches(actual_trades, expected_trades)
```

**Review example outputs:**
- Check `tests/fixtures/EXPECTED_TRADES.md` for expected trade patterns
- Compare equity curves with similar configurations

### Isolate the Problem

**Test components individually:**
```python
# Test data loading
from trading_system.data.loader import DataLoader
loader = DataLoader(config.data_paths)
data = loader.load_universe(['AAPL'], '2023-01-01', '2023-12-31')

# Test indicator computation
from trading_system.indicators.feature_computer import FeatureComputer
computer = FeatureComputer(config.strategy_config)
features = computer.compute_features_for_date(data['AAPL'], 'AAPL', pd.Timestamp('2023-06-15'))

# Test strategy
from trading_system.strategies.strategy_registry import get_strategy_class
Strategy = get_strategy_class('EquityMomentumStrategy')
strategy = Strategy(config.strategy_config)
signal = strategy.generate_signal(features, 'AAPL', pd.Timestamp('2023-06-15'))
```

### Common Debugging Workflows

**1. Debugging signal generation:**
```python
# Check why signal wasn't generated
features = compute_features(...)
eligibility = strategy.check_eligibility(features, symbol, date)
print(eligibility.reasons)  # See why symbol is/isn't eligible

signal = strategy.generate_signal(features, symbol, date)
print(signal)  # Check signal details
```

**2. Debugging portfolio state:**
```python
# Check why trade wasn't executed
portfolio = Portfolio(...)
can_trade = portfolio.can_enter_position(symbol, quantity, price)
print(f"Can trade: {can_trade}")
print(f"Cash: {portfolio.cash}")
print(f"Positions: {len(portfolio.positions)}")
print(f"Exposure: {portfolio.get_total_exposure()}")
```

**3. Debugging execution:**
```python
# Check order execution
order = create_order(...)
fill = fill_simulator.simulate_fill(order, bar, market_state)
print(f"Order: {order}")
print(f"Fill: {fill}")
print(f"Slippage: {fill.slippage_bps}")
```

---

## Common Error Messages

### Configuration Errors

| Error Message | Cause | Solution |
|--------------|-------|----------|
| `Invalid value for 'start_date': Expected YYYY-MM-DD format` | Date format incorrect | Use `YYYY-MM-DD` format (e.g., `2023-01-01`) |
| `Missing required field: 'strategy_type'` | Required config field missing | Add missing field to config file |
| `Invalid strategy_type: 'InvalidStrategy'` | Strategy not registered | Check strategy name spelling, ensure strategy is imported |
| `start_date (2023-12-31) must be before end_date (2023-01-01)` | Date range invalid | Ensure start_date < end_date |

### Data Errors

| Error Message | Cause | Solution |
|--------------|-------|----------|
| `Data file not found: /path/to/AAPL.csv` | File path incorrect | Verify file exists, check path in config |
| `Symbol 'AAPL' not found in data source` | Symbol not in data files | Ensure symbol file exists or remove from universe |
| `Missing columns: ['open', 'high']` | CSV missing required columns | Ensure CSV has: date, open, high, low, close, volume |
| `Invalid OHLC at dates: [...]` | Price relationships violated | Check data quality, fix or remove invalid rows |
| `Non-positive prices found` | Negative or zero prices | Fix data source, remove invalid rows |

### Strategy Errors

| Error Message | Cause | Solution |
|--------------|-------|----------|
| `Strategy 'X' not found` | Strategy not registered | Check strategy name, ensure strategy class exists |
| `Asset class mismatch: equity strategy with crypto symbol` | Wrong strategy for asset class | Use equity strategy for equity symbols, crypto for crypto |
| `Invalid clearance_20d: must be between 0 and 1, got 50` | Parameter value out of range | Use decimal (0.50) not percentage (50) |
| `Insufficient data for indicator: need 200 days, have 100` | Not enough historical data | Reduce date range start or add more historical data |

### Backtest Errors

| Error Message | Cause | Solution |
|--------------|-------|----------|
| `Error at date: 2023-01-15, step: compute_features` | Indicator calculation failed | Check data availability for that date, verify sufficient history |
| `Error at date: 2023-01-15, step: process_signals` | Signal processing failed | Check portfolio state, verify signal format |
| `Out of memory` | Too much data loaded | Reduce universe size or date range, increase system RAM |
| `No trades generated` | No signals met criteria | Check strategy parameters, verify data quality, review eligibility criteria |

### Execution Errors

| Error Message | Cause | Solution |
|--------------|-------|----------|
| `Insufficient capital for trade` | Not enough cash | Reduce position size, check initial capital |
| `Order rejected: exceeds capacity` | Order too large for market | Reduce position size or adjust capacity parameters |
| `Fill simulation failed` | Invalid order parameters | Check order price/quantity, verify market data |

---

## Getting Help

If you've tried the troubleshooting steps above and still can't resolve the issue:

1. **Check the documentation:**
   - Review `README.md` for usage instructions
   - Check `TESTING_GUIDE.md` for testing procedures
   - Review architecture docs in `agent-files/`
   - See `ERROR_CODE_REFERENCE.md` for comprehensive error code reference

2. **Review example configurations:**
   - Compare with working examples in `EXAMPLE_CONFIGS/`
   - Review test fixtures in `tests/fixtures/configs/`

3. **Run tests:**
   - Ensure system works with test data: `pytest tests/ -v`
   - Compare your setup with test configuration

4. **Collect information for bug report:**
   - Full error message and stack trace
   - Configuration file (sanitized if sensitive)
   - Log file output
   - System information (Python version, OS)
   - Steps to reproduce

---

## Quick Reference

### Validate Configuration
```bash
python -m trading_system config validate --path <config.yaml>
```

### Check Data Files
```bash
head -5 /path/to/data/AAPL.csv  # Verify format
ls -la /path/to/data/*.csv      # List all data files
```

### Run Quick Test
```bash
./quick_test.sh
```

### View Logs
```bash
tail -100 {output_dir}/{run_id}/train/backtest.log
```

### Test Data Loading
```python
from trading_system.data.loader import DataLoader
from trading_system.configs.run_config import RunConfig

config = RunConfig.from_yaml('config.yaml')
loader = DataLoader(config.data_paths)
data = loader.load_universe(['AAPL'], config.start_date, config.end_date)
print(data['AAPL'].head())
```

### Test Strategy
```python
from trading_system.strategies.strategy_registry import get_strategy_class
from trading_system.configs.strategy_config import StrategyConfig

config = StrategyConfig.from_yaml('equity_config.yaml')
Strategy = get_strategy_class(config.strategy_type)
strategy = Strategy(config)
# Test with sample features
```

---

**Last Updated:** 2024-12-19

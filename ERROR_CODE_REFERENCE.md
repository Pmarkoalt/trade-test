# Error Code Reference Guide

This document provides a comprehensive reference for all error types in the trading system, including error codes, causes, and troubleshooting steps.

## Table of Contents

1. [Error Categories](#error-categories)
2. [Data Errors](#data-errors)
3. [Configuration Errors](#configuration-errors)
4. [Strategy Errors](#strategy-errors)
5. [Portfolio Errors](#portfolio-errors)
6. [Execution Errors](#execution-errors)
7. [Indicator Errors](#indicator-errors)
8. [Backtest Errors](#backtest-errors)
9. [Validation Errors](#validation-errors)
10. [Quick Reference](#quick-reference)

---

## Error Categories

All errors in the trading system inherit from `TradingSystemError`. The error hierarchy is:

```
TradingSystemError (base)
├── DataError
│   ├── DataValidationError
│   ├── DataNotFoundError
│   └── DataSourceError
├── ConfigurationError
├── StrategyError
│   └── StrategyNotFoundError
├── PortfolioError
│   ├── InsufficientCapitalError
│   └── PositionNotFoundError
├── ExecutionError
│   ├── OrderRejectedError
│   └── FillError
├── IndicatorError
├── BacktestError
└── ValidationError
```

---

## Data Errors

### `DataError`

**Base class** for all data-related errors.

**Common Attributes:**
- `symbol` (Optional[str]): Symbol associated with the error
- `date` (Optional[str]): Date associated with the error
- `data_path` (Optional[str]): Data directory path where error occurred
- `file_path` (Optional[str]): Specific file path that caused the error

**When it occurs:**
- General data loading or processing failures
- Base class for more specific data errors

**Troubleshooting:**
1. Check `symbol` attribute for which symbol caused the issue
2. Check `data_path` and `file_path` attributes for file locations
3. Verify data files exist and are readable
4. Check file format and structure

---

### `DataNotFoundError`

**Error Code:** `DATA_NOT_FOUND`

**When it occurs:**
- Requested data file does not exist
- Symbol not found in data source
- Required data unavailable for specified date range

**Common Causes:**
1. File path incorrect in configuration
2. Symbol file missing from data directory
3. Date range outside available data
4. Symbol not in universe but referenced

**Example Messages:**
```
Data file not found: /path/to/data/AAPL.csv
Symbol 'AAPL' not found in data source
```

**Troubleshooting:**
1. **Verify file path:**
   ```bash
   ls -la /path/to/data/AAPL.csv
   ```

2. **Check configuration:**
   ```bash
   # Check data_paths in config
   cat config.yaml | grep -A 5 data_paths
   ```

3. **List available symbols:**
   ```python
   from trading_system.data.sources import CSVDataSource
   source = CSVDataSource("/path/to/data")
   print(source.get_available_symbols())
   ```

4. **Verify symbol naming:**
   - CSV files should be named `{SYMBOL}.csv` (e.g., `AAPL.csv`)
   - Case-sensitive matching
   - No extra spaces or special characters

5. **Check date range:**
   ```python
   # Check available date range for symbol
   source = CSVDataSource("/path/to/data")
   start, end = source.get_date_range("AAPL")
   print(f"AAPL: {start} to {end}")
   ```

---

### `DataValidationError`

**Error Code:** `DATA_VALIDATION_FAILED`

**When it occurs:**
- Data fails validation checks (OHLC relationships, missing columns, etc.)
- Invalid data format or structure
- Data quality issues detected

**Common Causes:**
1. Invalid OHLC relationships (low > high, prices outside range)
2. Missing required columns (open, high, low, close, volume)
3. Negative prices or volumes
4. Non-chronological dates
5. Duplicate dates
6. Extreme price moves (>50% in one day)

**Example Messages:**
```
Data validation failed for AAPL
Invalid OHLC data at dates: ['2023-01-15']
```

**Troubleshooting:**
1. **Check OHLC relationships:**
   ```python
   import pandas as pd
   df = pd.read_csv('AAPL.csv', index_col='date')

   # Check for invalid relationships
   invalid = (df['low'] > df['high']) | \
             (df['open'] < df['low']) | (df['open'] > df['high']) | \
             (df['close'] < df['low']) | (df['close'] > df['high'])
   print(df[invalid])
   ```

2. **Verify required columns:**
   ```python
   df = pd.read_csv('AAPL.csv')
   required = ['date', 'open', 'high', 'low', 'close', 'volume']
   missing = [col for col in required if col not in df.columns]
   print(f"Missing columns: {missing}")
   ```

3. **Check for negative values:**
   ```python
   numeric_cols = ['open', 'high', 'low', 'close', 'volume']
   negative = (df[numeric_cols] < 0).any()
   print(df[negative])
   ```

4. **Fix date ordering:**
   ```python
   df.sort_index(inplace=True)  # Sort by date
   ```

5. **Remove duplicate dates:**
   ```python
   df = df[~df.index.duplicated(keep='first')]
   ```

6. **Use the validator directly:**
   ```python
   from trading_system.data.validator import validate_ohlcv
   if not validate_ohlcv(df, 'AAPL'):
       # Check logs for specific validation failures
       pass
   ```

---

### `DataSourceError`

**Error Code:** `DATA_SOURCE_ERROR`

**When it occurs:**
- Data source operations fail (network errors, API errors, file I/O errors)
- Connection failures to data APIs
- Data source initialization failures

**Common Attributes:**
- `source_type` (Optional[str]): Type of data source (CSVDataSource, APIDataSource, etc.)
- `data_path` (Optional[str]): Data path configured
- `file_path` (Optional[str]): Specific file path (if applicable)

**Common Causes:**
1. Network connection failures (for API sources)
2. API rate limiting or authentication failures
3. File system errors (permissions, disk full)
4. Invalid data source configuration
5. Data source initialization failures

**Example Messages:**
```
Network error loading AAPL: Connection timeout
Error loading AAPL from /path/to/data/AAPL.csv: Permission denied
```

**Troubleshooting:**
1. **For API sources:**
   - Check API key is valid and not expired
   - Verify network connectivity
   - Check API rate limits (may need to wait or upgrade plan)
   - Review API documentation for error codes

2. **For file sources:**
   ```bash
   # Check file permissions
   ls -la /path/to/data/AAPL.csv

   # Check disk space
   df -h /path/to/data

   # Test file readability
   head -5 /path/to/data/AAPL.csv
   ```

3. **Check data source configuration:**
   ```python
   # Verify data source initialization
   from trading_system.data.sources import CSVDataSource
   source = CSVDataSource("/path/to/data")  # Should not raise error
   ```

4. **Review logs:**
   - Check log files for detailed error messages
   - Look for stack traces indicating root cause

---

## Configuration Errors

### `ConfigurationError`

**Error Code:** `CONFIG_ERROR`

**When it occurs:**
- Configuration file validation fails
- Invalid configuration values
- Missing required configuration fields
- Configuration file parsing errors

**Common Attributes:**
- `config_path` (Optional[str]): Path to configuration file
- `field` (Optional[str]): Specific field that caused the error
- `errors` (Optional[List[Dict]]): List of validation errors

**Common Causes:**
1. Invalid YAML syntax
2. Missing required fields
3. Invalid field values (wrong type, out of range)
4. Invalid date formats (not YYYY-MM-DD)
5. Invalid file paths in configuration
6. Logic errors (e.g., start_date >= end_date)

**Example Messages:**
```
Configuration validation failed: Invalid value for 'start_date'
Configuration file: /path/to/config.yaml
```

**Troubleshooting:**
1. **Validate configuration:**
   ```bash
   python -m trading_system config validate --path config.yaml
   ```

2. **Check YAML syntax:**
   - Use YAML validator (online tool or VS Code extension)
   - Ensure proper indentation (2 spaces, no tabs)
   - Quote strings with special characters
   - Remove trailing commas (not allowed in YAML)

3. **Review example configs:**
   ```bash
   # Compare with working examples
   cat EXAMPLE_CONFIGS/run_config.yaml
   cat EXAMPLE_CONFIGS/equity_config.yaml
   ```

4. **Check required fields:**
   - `run_config.yaml`: `data_paths`, `start_date`, `end_date`, `output_dir`
   - Strategy configs: `strategy_type`, `asset_class`, `entry`, `exit`, `risk_management`

5. **Fix date formats:**
   ```yaml
   # Wrong
   start_date: 01/01/2023

   # Correct
   start_date: 2023-01-01
   ```

6. **Generate template:**
   ```bash
   python -m trading_system config template
   ```

---

## Strategy Errors

### `StrategyError`

**Error Code:** `STRATEGY_ERROR`

**When it occurs:**
- Strategy configuration errors
- Strategy execution failures
- Invalid strategy parameters

**Common Attributes:**
- `strategy_name` (Optional[str]): Name of the strategy
- `symbol` (Optional[str]): Symbol associated with the error

**Common Causes:**
1. Invalid strategy configuration
2. Strategy parameter values out of range
3. Missing required strategy parameters
4. Strategy execution logic errors

**Example Messages:**
```
Strategy error: Invalid strategy configuration for AAPL
Strategy error: Invalid clearance_20d: must be between 0 and 1, got 50
```

**Troubleshooting:**
1. **Check strategy configuration:**
   ```yaml
   # Verify strategy type and asset class match
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

3. **Check parameter ranges:**
   - Percentages should be 0-1 (decimal), not 0-100
   - Ensure required fields are present (entry, exit, risk_management)
   - Compare with example configs in `EXAMPLE_CONFIGS/`

4. **Verify asset class matching:**
   - Equity strategies only work with equity data (from `equity_path`)
   - Crypto strategies only work with crypto data (from `crypto_path`)
   - Ensure symbols in universe match the asset class

---

### `StrategyNotFoundError`

**Error Code:** `STRATEGY_NOT_FOUND`

**When it occurs:**
- Strategy class cannot be found or is not registered
- Strategy name misspelled in configuration
- Strategy module not imported

**Example Messages:**
```
Strategy 'EquityMomentumStrategy' not found
```

**Troubleshooting:**
1. **Check strategy name spelling:**
   - Verify exact spelling in configuration
   - Check available strategies in strategy registry

2. **Verify strategy is registered:**
   ```python
   from trading_system.strategies.strategy_registry import STRATEGY_REGISTRY
   print(list(STRATEGY_REGISTRY.keys()))  # List all registered strategies
   ```

3. **Check strategy imports:**
   - Ensure strategy module is imported (usually automatic)
   - Verify strategy class exists in strategy module

4. **Review strategy configuration:**
   ```yaml
   # Correct format
   strategy_type: EquityMomentumStrategy  # Exact class name
   ```

---

## Portfolio Errors

### `PortfolioError`

**Error Code:** `PORTFOLIO_ERROR`

**Base class** for portfolio-related errors.

**Common Attributes:**
- `symbol` (Optional[str]): Symbol associated with the error
- `position_id` (Optional[str]): Position ID associated with the error

---

### `InsufficientCapitalError`

**Error Code:** `INSUFFICIENT_CAPITAL`

**When it occurs:**
- Not enough cash available to execute a trade
- Position size exceeds available capital

**Example Messages:**
```
Insufficient capital for trade: Requested $10000, available $5000
```

**Troubleshooting:**
1. **Check portfolio cash:**
   ```python
   print(f"Available cash: {portfolio.cash}")
   print(f"Requested trade value: {quantity * price}")
   ```

2. **Reduce position size:**
   - Adjust position sizing parameters in strategy config
   - Reduce initial capital allocation per trade

3. **Check initial capital:**
   ```yaml
   # In run_config.yaml
   portfolio:
     initial_capital: 100000  # Increase if needed
   ```

---

### `PositionNotFoundError`

**Error Code:** `POSITION_NOT_FOUND`

**When it occurs:**
- Attempting to access or modify a position that doesn't exist
- Position ID invalid or position already closed

**Example Messages:**
```
Position not found: position_id=12345
Position not found for symbol: AAPL
```

**Troubleshooting:**
1. **Verify position exists:**
   ```python
   if symbol in portfolio.positions:
       position = portfolio.positions[symbol]
   else:
       print(f"No position found for {symbol}")
   ```

2. **Check position ID:**
   - Ensure position ID is correct
   - Verify position hasn't been closed

---

## Execution Errors

### `ExecutionError`

**Error Code:** `EXECUTION_ERROR`

**Base class** for execution-related errors.

**Common Attributes:**
- `order_id` (Optional[str]): Order ID associated with the error
- `symbol` (Optional[str]): Symbol associated with the error

---

### `OrderRejectedError`

**Error Code:** `ORDER_REJECTED`

**When it occurs:**
- Order rejected by execution engine
- Order exceeds capacity or violates constraints
- Invalid order parameters

**Example Messages:**
```
Order rejected: exceeds capacity for symbol AAPL
Order rejected: invalid order type
```

**Troubleshooting:**
1. **Check order parameters:**
   - Verify order type (market, limit)
   - Check order quantity and price
   - Verify symbol is valid

2. **Review capacity constraints:**
   - Check position size limits
   - Verify market capacity settings

3. **Check order validation:**
   - Ensure order meets all validation criteria
   - Review execution engine logs

---

### `FillError`

**Error Code:** `FILL_ERROR`

**When it occurs:**
- Order fill simulation fails
- Invalid fill parameters
- Fill calculation errors

**Example Messages:**
```
Fill simulation failed for order: order_id=12345
```

**Troubleshooting:**
1. **Check order parameters:**
   - Verify order price and quantity
   - Check market data availability

2. **Review fill simulator settings:**
   - Check slippage and fee configurations
   - Verify market state data

---

## Indicator Errors

### `IndicatorError`

**Error Code:** `INDICATOR_ERROR`

**When it occurs:**
- Indicator calculation fails
- Insufficient data for indicator calculation
- Invalid indicator parameters

**Common Attributes:**
- `indicator_name` (Optional[str]): Name of the indicator
- `symbol` (Optional[str]): Symbol associated with the error

**Common Causes:**
1. Insufficient historical data (e.g., MA200 needs 200 days)
2. Invalid indicator parameters
3. Data quality issues affecting indicator calculation
4. Indicator implementation bugs

**Example Messages:**
```
Indicator error: Insufficient data for MA200 (need 200 days, have 100)
Indicator error: RSI calculation failed for AAPL
```

**Troubleshooting:**
1. **Check data availability:**
   ```python
   # Verify sufficient data for indicator
   df = data[symbol]
   print(f"Available data: {len(df)} days")
   print(f"Date range: {df.index.min()} to {df.index.max()}")
   ```

2. **Reduce indicator period:**
   - Use shorter period (e.g., MA50 instead of MA200)
   - Adjust indicator parameters in strategy config

3. **Add more historical data:**
   - Extend start date to include more history
   - Ensure data covers required warm-up period

4. **Check indicator parameters:**
   ```yaml
   # Verify indicator parameters are valid
   indicators:
     ma_period: 50  # Must be positive integer
   ```

---

## Backtest Errors

### `BacktestError`

**Error Code:** `BACKTEST_ERROR`

**When it occurs:**
- Unexpected errors during backtest execution
- Backtest engine failures
- State inconsistencies

**Common Attributes:**
- `date` (Optional[str]): Date when error occurred
- `step` (Optional[str]): Backtest step where error occurred (load_data, compute_features, generate_signals, process_signals, update_portfolio)

**Common Causes:**
1. Missing data for specific date
2. Indicator calculation failure
3. Portfolio state inconsistency
4. Execution error
5. Memory issues (out of memory)

**Example Messages:**
```
Backtest error at date: 2023-01-15, step: compute_features
Backtest error: Unexpected error during backtest execution
```

**Troubleshooting:**
1. **Check log file:**
   ```bash
   tail -100 {output_dir}/{run_id}/train/backtest.log
   ```

2. **Review error context:**
   - Note the `date` and `step` attributes
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
   - Increase system RAM

6. **Common backtest step errors:**
   - **`load_data`**: Data loading failed - check file paths and data quality
   - **`compute_features`**: Indicator calculation failed - check data availability
   - **`generate_signals`**: Signal generation failed - check strategy config
   - **`process_signals`**: Signal processing failed - check portfolio state
   - **`update_portfolio`**: Portfolio update failed - check for state inconsistencies

---

## Validation Errors

### `ValidationError`

**Error Code:** `VALIDATION_ERROR`

**When it occurs:**
- Validation suite failures
- Statistical validation errors
- Bootstrap or permutation test failures

**Common Attributes:**
- `validation_type` (Optional[str]): Type of validation that failed

**Common Causes:**
1. Validation test failures (bootstrap, permutation, etc.)
2. Statistical significance issues
3. Performance metric validation failures
4. Walk-forward validation failures

**Example Messages:**
```
Validation error: Bootstrap test failed
Validation error: Permutation test p-value too high
```

**Troubleshooting:**
1. **Review validation results:**
   - Check validation report for detailed results
   - Review which tests failed and why

2. **Adjust validation parameters:**
   - Modify significance thresholds if appropriate
   - Adjust bootstrap or permutation test parameters

3. **Check strategy performance:**
   - Review backtest results
   - Verify strategy meets validation criteria

---

## Quick Reference

### Error Code Summary

| Error Code | Exception Class | Common Cause |
|------------|----------------|--------------|
| `DATA_NOT_FOUND` | `DataNotFoundError` | File not found, symbol missing |
| `DATA_VALIDATION_FAILED` | `DataValidationError` | Invalid data format or quality |
| `DATA_SOURCE_ERROR` | `DataSourceError` | Data source operation failed |
| `CONFIG_ERROR` | `ConfigurationError` | Configuration validation failed |
| `STRATEGY_ERROR` | `StrategyError` | Strategy configuration/execution error |
| `STRATEGY_NOT_FOUND` | `StrategyNotFoundError` | Strategy class not found |
| `INSUFFICIENT_CAPITAL` | `InsufficientCapitalError` | Not enough cash for trade |
| `POSITION_NOT_FOUND` | `PositionNotFoundError` | Position doesn't exist |
| `ORDER_REJECTED` | `OrderRejectedError` | Order rejected by execution engine |
| `FILL_ERROR` | `FillError` | Fill simulation failed |
| `INDICATOR_ERROR` | `IndicatorError` | Indicator calculation failed |
| `BACKTEST_ERROR` | `BacktestError` | Backtest execution failed |
| `VALIDATION_ERROR` | `ValidationError` | Validation test failed |

### Quick Troubleshooting Checklist

1. **Data Errors:**
   - [ ] Check file paths in configuration
   - [ ] Verify files exist and are readable
   - [ ] Check file format (CSV with required columns)
   - [ ] Validate data quality (OHLC relationships, no negative values)

2. **Configuration Errors:**
   - [ ] Validate config: `python -m trading_system config validate --path config.yaml`
   - [ ] Check YAML syntax
   - [ ] Verify required fields are present
   - [ ] Check date formats (YYYY-MM-DD)

3. **Strategy Errors:**
   - [ ] Verify strategy name is correct
   - [ ] Check asset class matches data source
   - [ ] Verify parameter ranges (percentages as decimals 0-1)

4. **Backtest Errors:**
   - [ ] Check log files for detailed error
   - [ ] Verify data availability for all dates
   - [ ] Check for sufficient historical data for indicators
   - [ ] Review error date and step attributes

5. **General:**
   - [ ] Review example configs in `EXAMPLE_CONFIGS/`
   - [ ] Check `TROUBLESHOOTING.md` for detailed guidance
   - [ ] Review log files in output directory
   - [ ] Run validation: `python -m trading_system config validate --path config.yaml`

---

## Getting Help

If you've tried the troubleshooting steps above and still can't resolve the error:

1. **Collect error information:**
   - Full error message and stack trace
   - Configuration file (sanitized if sensitive)
   - Log file output
   - System information (Python version, OS)
   - Steps to reproduce

2. **Review documentation:**
   - `README.md` - Usage instructions
   - `TROUBLESHOOTING.md` - Detailed troubleshooting guide
   - `FAQ.md` - Frequently asked questions
   - `EXAMPLE_CONFIGS/` - Example configurations

3. **Check test fixtures:**
   - Review test configurations in `tests/fixtures/configs/`
   - Compare with your configuration

---

**Last Updated:** 2024-12-19


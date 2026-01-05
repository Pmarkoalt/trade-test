# Type Safety and Error Handling Improvements

## Summary

This document outlines the improvements made to enhance type safety and error handling throughout the trading system codebase.

## Improvements Made

### 1. Custom Exception Classes (`trading_system/exceptions.py`)

Created a comprehensive exception hierarchy for better error categorization:

- **Base Exception**: `TradingSystemError`
- **Data Exceptions**: `DataError`, `DataValidationError`, `DataNotFoundError`, `DataSourceError`
- **Configuration Exceptions**: `ConfigurationError`
- **Strategy Exceptions**: `StrategyError`, `StrategyNotFoundError`
- **Portfolio Exceptions**: `PortfolioError`, `InsufficientCapitalError`, `PositionNotFoundError`
- **Execution Exceptions**: `ExecutionError`, `OrderRejectedError`, `FillError`
- **Indicator Exceptions**: `IndicatorError`
- **Backtest Exceptions**: `BacktestError`
- **Validation Exceptions**: `ValidationError`

Each exception class includes relevant context fields (symbol, date, strategy_name, etc.) for better debugging.

### 2. Type Hint Improvements

#### `trading_system/backtest/event_loop.py`
- Fixed `Optional = None` → `Optional[np.random.Generator] = None`
- Fixed `Optional[set]` → `Optional[Set[pd.Timestamp]]`
- Fixed `Callable` → `Callable[[pd.DataFrame, str, str, Optional[pd.Series], Optional[pd.Series], bool, bool], pd.DataFrame]`
- Fixed `Dict` → `Dict[str, Any]` for return types and metadata dictionaries
- Added proper imports: `Any`, `Tuple`, `Set` from typing

#### `trading_system/data/sources/base_source.py`
- Fixed `Dict[str, any]` → `Dict[str, Any]` (lowercase `any` is invalid)
- Added `Any` import from typing

#### `trading_system/configs/doc_generator.py`
- Fixed `dict` → `Dict[str, Any]` for return type
- Added proper imports

### 3. Error Handling Improvements

#### `trading_system/backtest/event_loop.py`
- Replaced generic `except Exception` with specific exception types:
  - `ValueError`, `KeyError`, `IndexError` for expected data errors
  - `BacktestError` for unexpected errors with context (date, step)
- Added proper exception chaining with `from e`

#### `trading_system/data/loader.py`
- Improved error handling in `load_universe()`:
  - Specific handling for `FileNotFoundError`
  - Specific handling for `pd.errors.EmptyDataError`
  - Specific handling for `pd.errors.ParserError`
  - Proper exception chaining

#### `trading_system/data/sources/api_source.py`
- Added specific exception handling:
  - `DataValidationError` for validation failures
  - `ConnectionError`, `TimeoutError` for network issues
  - `ValueError` for data format errors
  - `DataSourceError` for other source-related errors
- All exceptions include context (symbol, source_type)

#### `trading_system/strategies/strategy_registry.py`
- Replaced `ValueError` with `StrategyNotFoundError` for better error categorization
- Includes strategy name and symbol context

#### `trading_system/indicators/feature_computer.py`
- Replaced `ValueError` with `IndicatorError` for indicator-related errors
- Added error handling in `compute_features_for_date()`:
  - `KeyError`, `IndexError` for missing dates
  - `KeyError`, `ValueError`, `TypeError` for FeatureRow conversion errors
- All errors include indicator name and symbol context

#### `trading_system/cli.py`
- Enhanced error handling in `cmd_backtest()`:
  - Specific handling for `FileNotFoundError`
  - Specific handling for `ConfigurationError`
  - Specific handling for `DataError`, `DataNotFoundError`
  - Specific handling for `StrategyError`
  - Specific handling for `BacktestError`
  - Specific handling for `TradingSystemError`
  - Generic `Exception` as final fallback
- Each exception type provides context-specific troubleshooting tips

## Benefits

### Type Safety
1. **Better IDE Support**: More accurate autocomplete and type checking
2. **Catch Errors Early**: Type checkers (mypy, pyright) can catch type mismatches before runtime
3. **Self-Documenting Code**: Type hints serve as inline documentation
4. **Refactoring Safety**: Type hints help ensure refactoring doesn't break contracts

### Error Handling
1. **Better Debugging**: Specific exception types with context make it easier to identify issues
2. **Graceful Degradation**: Specific exception handling allows for targeted recovery strategies
3. **User-Friendly Messages**: Context-rich exceptions enable better error messages in CLI
4. **Error Tracking**: Exception hierarchy makes it easier to categorize and track errors in logs

## Migration Guide

### For Developers

When adding new code:

1. **Use Custom Exceptions**: Import from `trading_system.exceptions` and use appropriate exception types
2. **Add Type Hints**: Always include type hints for function parameters and return types
3. **Be Specific**: Catch specific exceptions before generic `Exception`
4. **Include Context**: Add relevant context fields to exceptions (symbol, date, etc.)
5. **Chain Exceptions**: Use `from e` when re-raising exceptions

### Example

**Before:**
```python
def load_data(symbol: str):
    try:
        data = fetch(symbol)
    except Exception as e:
        raise ValueError(f"Error: {e}")
```

**After:**
```python
from trading_system.exceptions import DataSourceError, DataNotFoundError

def load_data(symbol: str) -> pd.DataFrame:
    try:
        data = fetch(symbol)
        if data is None:
            raise DataNotFoundError(f"Data not found for {symbol}", symbol=symbol)
        return data
    except ConnectionError as e:
        raise DataSourceError(
            f"Connection error loading {symbol}: {e}",
            symbol=symbol,
            source_type="API"
        ) from e
```

## Testing

All changes maintain backward compatibility. Existing code will continue to work, but new code should adopt these patterns.

## Next Steps

1. Gradually migrate remaining generic `except Exception` blocks to specific exceptions
2. Add type hints to remaining functions without them
3. Consider adding mypy or pyright for static type checking in CI/CD
4. Document exception handling patterns in contributing guidelines

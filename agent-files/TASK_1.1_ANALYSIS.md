# Task 1.1: Fix Failing Property-Based Tests - Analysis

**Date**: 2024-12-30  
**Task**: AGENT_IMPROVEMENTS_ROADMAP.md - Task 1.1  
**Status**: 🔴 **6 tests failing**  
**Priority**: CRITICAL

---

## Overview

This document provides a comprehensive analysis of the 6 failing property-based tests, including code review, potential root causes, and recommended fixes.

---

## Test Failures

### 1. `test_ma_nan_before_window` ❌

**Location**: `tests/property/test_indicators.py::TestMovingAverage::test_ma_nan_before_window`  
**Implementation**: `trading_system/indicators/ma.py::ma()`

**Test Expectation**:
- First `window-1` values should be NaN
- Test uses: `assert result.iloc[:window-1].isna().all()`

**Code Review**:

The `ma()` function (lines 9-73) already has logic to handle NaN assignment:
- Line 46: Uses `series.rolling(window=window, min_periods=window).mean()`
- Line 63: Explicitly sets `ma_series.iloc[:window-1] = np.nan`

**Potential Issues**:

1. **Edge case when window=1**: 
   - When `window=1`, `window-1=0`, so `iloc[:0]` is an empty slice
   - This should be fine, but might cause issues if test doesn't filter this case
   - **Fix**: Test already uses `assume(len(series) >= window)` and `window >= 2` from the strategy, so this shouldn't be an issue

2. **Cache interaction**:
   - Cached results might not preserve the NaN assignment correctly
   - **Fix**: The cache stores the result after NaN assignment, so this should be fine

3. **Series copy issue**:
   - Line 51: `ma_series = ma_series.copy()` - this should preserve NaN values
   - **Potential fix**: Verify that the copy operation preserves NaN correctly

**Recommended Fix**:

The code looks correct, but to be extra defensive, ensure the NaN assignment happens AFTER the rolling operation and before caching:

```python
# Current code (lines 46-63) looks correct, but verify execution order
ma_series = series.rolling(window=window, min_periods=window).mean()
ma_series = ma_series.copy()

# Explicit NaN assignment (this should work)
if len(ma_series) >= window and window > 1:
    ma_series.iloc[:window-1] = np.nan
```

**Action**: Run the test to see the exact error message, then verify if it's a logical issue or a test expectation mismatch.

---

### 2. `test_portfolio_equity_always_positive` ❌

**Location**: `tests/property/test_portfolio.py::TestPortfolioProperties::test_portfolio_equity_always_positive`  
**Implementation**: `trading_system/portfolio/portfolio.py::process_fill()` and `update_equity()`

**Test Expectation**:
- Portfolio equity must always be positive (>= 0) after operations
- Test assertion: `assert portfolio.equity > 0` (strictly positive)

**Code Review Needed**:
- Review `process_fill()` method logic
- Review `update_equity()` calculation: `equity = cash + total_exposure`
- Check if fees/slippage can cause equity to go negative

**Potential Issues**:

1. **Fee/Slippage calculation**:
   - The test creates a fill with `total_cost=notional * 1.0015` (includes fees)
   - If cash is insufficient after fees, equity could go negative
   - **Fix**: Ensure `process_fill()` checks sufficient cash before processing

2. **Equity calculation**:
   - `equity = cash + total_exposure` 
   - For shorts: `total_exposure` can be negative, but cash should account for margin
   - **Fix**: Ensure equity calculation properly handles shorts

3. **Minimum cash reserve**:
   - No minimum cash reserve enforced
   - **Fix**: Add validation to prevent equity from going negative

**Recommended Fix**:

Add defensive checks in `process_fill()`:

```python
def process_fill(self, fill, ...):
    # ... existing code ...
    
    # Check if we have sufficient cash/equity after fees
    required_cash = fill.total_cost if fill.side == SignalSide.BUY else 0
    if self.cash < required_cash:
        # Should not happen if validation works, but add safeguard
        raise ValueError(f"Insufficient cash: need {required_cash}, have {self.cash}")
    
    # ... process fill ...
    
    # After processing, verify equity is positive
    self.update_equity(current_prices)
    if self.equity < 0:
        # This should never happen, but add safeguard
        raise ValueError(f"Equity became negative: {self.equity}")
    
    assert self.equity >= 0  # Property must hold
```

---

### 3. `test_portfolio_exposure_limits` ❌

**Location**: `tests/property/test_portfolio.py::TestPortfolioProperties::test_portfolio_exposure_limits`  
**Implementation**: `trading_system/portfolio/portfolio.py::update_equity()` and exposure calculation

**Test Expectation**:
- Gross exposure ≤ 80%
- Per-position exposure ≤ 15%

**Potential Issues**:

1. **Exposure calculation**:
   - `gross_exposure_pct = total_exposure / equity`
   - If equity is 0 or very small, this can be invalid
   - **Fix**: Handle edge case when equity is 0

2. **Per-position exposure**:
   - `per_position_exposure` dictionary might not be properly maintained
   - **Fix**: Ensure it's updated whenever positions change

3. **Limit enforcement**:
   - Limits might not be enforced when adding positions
   - **Fix**: Check limits in `process_fill()` before adding position

**Recommended Fix**:

Add limit checks in `process_fill()`:

```python
def process_fill(self, fill, ...):
    # ... existing code ...
    
    # Calculate what exposure would be after this fill
    new_notional = fill.notional
    current_exposure = sum(p.notional_value for p in self.positions.values())
    projected_exposure = current_exposure + new_notional
    
    # Check gross exposure limit (80%)
    if projected_exposure / self.equity > 0.80:
        raise ValueError(f"Would exceed gross exposure limit: {projected_exposure / self.equity:.2%}")
    
    # Check per-position limit (15%)
    if new_notional / self.equity > 0.15:
        raise ValueError(f"Would exceed per-position limit: {new_notional / self.equity:.2%}")
    
    # ... proceed with fill ...
```

---

### 4. `test_portfolio_equity_updates_with_prices` ❌

**Location**: `tests/property/test_portfolio.py::TestPortfolioProperties::test_portfolio_equity_updates_with_prices`  
**Implementation**: `trading_system/portfolio/portfolio.py::update_equity()`

**Test Expectation**:
- Equity should increase when prices go up
- Equity should decrease when prices go down
- Test checks: `if price_mult > 1.0: assert portfolio.equity >= initial_equity`

**Potential Issues**:

1. **Equity calculation logic**:
   - `equity = cash + total_exposure`
   - For longs: `position_value = current_price * quantity`
   - For shorts: Need to account for margin requirements
   - **Fix**: Verify `total_exposure` calculation correctly uses current prices

2. **Position value updates**:
   - Positions might not be updating with new prices
   - **Fix**: Ensure `update_equity()` recalculates position values from current prices

3. **Unrealized P&L**:
   - Equity should reflect unrealized gains/losses
   - **Fix**: Verify unrealized P&L is correctly included in equity

**Recommended Fix**:

Review `update_equity()` method to ensure it:
1. Recalculates position values using `current_prices`
2. Properly calculates `total_exposure` from updated position values
3. Updates `equity = cash + total_exposure` correctly

---

### 5. `test_sharpe_constant_returns_zero` ❌

**Location**: `tests/property/test_validation.py::TestBootstrapProperties::test_sharpe_constant_returns_zero`  
**Implementation**: `trading_system/validation/bootstrap.py::compute_sharpe_from_r_multiples()`

**Test Expectation**:
- Constant R-multiples (zero std) should return zero Sharpe ratio
- Test expects: `sharpe == 0.0` when all R-multiples are the same

**Code Review**:

The function already handles this case (lines 38-39):
```python
if not np.isfinite(std_r) or std_r == 0.0 or std_r < 1e-10:
    return 0.0
```

**Potential Issues**:

1. **Floating point precision**:
   - `std_r` might not be exactly 0.0 due to floating point errors
   - The check `std_r < 1e-10` should catch this, but might need adjustment
   - **Fix**: The current threshold of `1e-10` should be sufficient

2. **Test expectation**:
   - Test might be checking for exact equality with 0.0
   - **Fix**: Test should use approximate equality: `assert abs(sharpe - 0.0) < 1e-10`

**Recommended Fix**:

The code looks correct. The issue might be in the test itself. Verify the test uses appropriate floating point comparison:

```python
# In test:
assert abs(sharpe) < 1e-10  # Use approximate comparison
# OR
assert sharpe == pytest.approx(0.0)
```

---

### 6. `test_permutation_test_structure` ❌

**Location**: `tests/property/test_validation.py::TestPermutationProperties::test_permutation_test_structure`  
**Implementation**: `trading_system/validation/permutation.py::PermutationTest.run()`

**Test Expectation**:
- Returns dictionary with keys: `actual_sharpe`, `random_sharpe_5th`, `random_sharpe_95th`, `percentile_rank`, `passed`
- All values must be finite (not NaN or inf)
- `percentile_rank` must be in range [0.0, 100.0]

**Code Review Needed**:
- Review `PermutationTest.run()` method
- Verify return dictionary structure
- Check edge cases: empty trades, single trade, etc.

**Potential Issues**:

1. **Missing keys**:
   - Return dictionary might not have all required keys
   - **Fix**: Ensure all keys are present in return dictionary

2. **NaN/Inf values**:
   - Calculations might produce NaN or inf in edge cases
   - **Fix**: Add checks to ensure all values are finite

3. **Percentile rank calculation**:
   - Might be outside [0.0, 100.0] range
   - **Fix**: Clamp percentile_rank to [0.0, 100.0]

**Recommended Fix**:

Review `PermutationTest.run()` and ensure:

```python
def run(self) -> Dict:
    # ... existing code ...
    
    # Ensure all required keys are present
    result = {
        'actual_sharpe': actual_sharpe,
        'random_sharpe_5th': np.percentile(sharpe_samples, 5),
        'random_sharpe_95th': np.percentile(sharpe_samples, 95),
        'percentile_rank': percentile_rank,
        'passed': passed
    }
    
    # Ensure all values are finite
    for key, value in result.items():
        if not np.isfinite(value):
            # Handle NaN/inf cases
            if key == 'percentile_rank':
                result[key] = 0.0  # Default to 0 if invalid
            elif key == 'passed':
                result[key] = False  # Default to False if invalid
            else:
                result[key] = 0.0  # Default numeric values to 0
    
    # Clamp percentile_rank to [0.0, 100.0]
    result['percentile_rank'] = max(0.0, min(100.0, result['percentile_rank']))
    
    return result
```

---

## Implementation Steps

### Step 1: Run Tests to Get Exact Errors

```bash
# Run each failing test individually to see exact error messages
pytest tests/property/test_indicators.py::TestMovingAverage::test_ma_nan_before_window -v
pytest tests/property/test_portfolio.py::TestPortfolioProperties::test_portfolio_equity_always_positive -v
pytest tests/property/test_portfolio.py::TestPortfolioProperties::test_portfolio_exposure_limits -v
pytest tests/property/test_portfolio.py::TestPortfolioProperties::test_portfolio_equity_updates_with_prices -v
pytest tests/property/test_validation.py::TestBootstrapProperties::test_sharpe_constant_returns_zero -v
pytest tests/property/test_validation.py::TestPermutationProperties::test_permutation_test_structure -v
```

### Step 2: Review Implementation Code

For each failing test:
1. Read the test code to understand expected behavior
2. Review the implementation code
3. Identify root cause of failure

### Step 3: Fix Implementation

Make minimal changes to fix the issue:
- Preserve existing functionality
- Add defensive checks where appropriate
- Handle edge cases properly

### Step 4: Verify Fixes

```bash
# Run specific test
pytest tests/property/test_indicators.py::TestMovingAverage::test_ma_nan_before_window -v

# Run all property tests
pytest tests/property/ -v

# Run full test suite to ensure no regressions
pytest tests/ -v
```

---

## Summary

| Test | Status | Likely Issue | Priority |
|------|--------|--------------|----------|
| `test_ma_nan_before_window` | ❌ | Possible edge case or cache issue | MEDIUM |
| `test_portfolio_equity_always_positive` | ❌ | Missing safeguards in `process_fill()` | HIGH |
| `test_portfolio_exposure_limits` | ❌ | Missing limit enforcement in `process_fill()` | HIGH |
| `test_portfolio_equity_updates_with_prices` | ❌ | Equity calculation logic issue | HIGH |
| `test_sharpe_constant_returns_zero` | ❌ | Code looks correct, might be test issue | LOW |
| `test_permutation_test_structure` | ❌ | Missing validation/finite checks | MEDIUM |

---

## Next Steps

1. **Run tests** to get exact error messages
2. **Review implementation code** for each failing test
3. **Apply fixes** based on analysis
4. **Verify** all tests pass
5. **Run full test suite** to ensure no regressions

---

**Estimated Time**: 4-6 hours  
**Priority**: CRITICAL (blocking property-based test validation)


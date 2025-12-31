# Agent Fixes and Recommendations

**Date**: 2024-12-30
**Status**: Active
**Priority**: CRITICAL - Tests are failing

---

## Executive Summary

This document provides a comprehensive list of fixes and recommendations for agents to address. The codebase has undergone significant improvements (short selling, ML enhancements, test coverage expansion) but several issues need immediate attention.

**Current Test Status**: 624 passed, 110 failed, 46 errors
**Target**: All tests passing

---

## CRITICAL FIXES (Priority 1)

### Fix 1.1: Add Missing `side` Parameter to Test Fixtures

**Status**: CRITICAL - Blocking 40+ tests
**Files Affected**: 15+ test files
**Estimated Effort**: 2-3 hours

**Problem**: The `Position` model was updated to require a `side` parameter (PositionSide.LONG or PositionSide.SHORT) for short selling support, but test fixtures were not updated.

**Error**:
```
TypeError: __init__() missing 1 required positional argument: 'side'
```

**Files to Update**:
```
tests/test_portfolio.py
tests/test_equity_strategy.py
tests/test_crypto_strategy.py
tests/test_mean_reversion.py
tests/test_pairs.py
tests/test_models.py
tests/test_edge_cases.py
tests/test_adapters.py
tests/test_factor_strategy.py
tests/test_multi_timeframe_strategy.py
tests/test_execution.py
tests/test_backtest_engine.py
tests/test_scoring.py
```

**Required Change**:

Add `side=PositionSide.LONG` to all Position instantiations:

```python
# Before
position = Position(
    symbol="AAPL",
    asset_class="equity",
    entry_date=pd.Timestamp("2024-01-02"),
    entry_price=150.0,
    ...
)

# After
from trading_system.models.positions import Position, PositionSide

position = Position(
    symbol="AAPL",
    asset_class="equity",
    entry_date=pd.Timestamp("2024-01-02"),
    entry_price=150.0,
    side=PositionSide.LONG,  # ADD THIS LINE
    ...
)
```

**Search Pattern**:
```bash
grep -r "Position(" tests/*.py | grep -v "side=" | grep -v "PositionSide"
```

**Verification**:
```bash
pytest tests/test_portfolio.py tests/test_models.py -v
```

---

### Fix 1.2: ML Ensemble Tests Require sklearn

**Status**: HIGH - Blocking 4+ tests
**Files Affected**: `tests/test_ml_ensemble.py`, `tests/test_ml_features.py`
**Estimated Effort**: 15 minutes

**Problem**: The ML tests require `sklearn` (scikit-learn) which is an optional dependency.

**Error**:
```
ModuleNotFoundError: No module named 'sklearn'
```

**Required Action**:

Choose one of these approaches:

1. **Install sklearn for testing** (recommended for full coverage):
   ```bash
   pip install scikit-learn
   ```

2. **OR skip ML tests when sklearn not available**:
   Add to test files:
   ```python
   pytest.importorskip("sklearn")
   ```

3. **OR mark tests as optional**:
   ```python
   @pytest.mark.skipif(
       not importlib.util.find_spec("sklearn"),
       reason="sklearn not installed"
   )
   ```

4. **For Docker**: Ensure sklearn is in requirements.txt or requirements-dev.txt

**Verification**:
```bash
pip install scikit-learn && pytest tests/test_ml_ensemble.py -v
```

---

## HIGH PRIORITY FIXES (Priority 2)

### Fix 2.1: Review and Update Strategy Exit Signal Tests

**Status**: HIGH - Multiple strategy tests failing
**Files Affected**: Strategy test files
**Estimated Effort**: 1-2 hours

**Problem**: Exit signal tests are failing across multiple strategy test files. The test structures may not align with updated strategy implementations.

**Affected Tests**:
- `tests/test_equity_strategy.py::TestExitSignals`
- `tests/test_mean_reversion.py::TestExitSignals`
- `tests/test_pairs.py::TestPairExitSignals`

**Required Action**:
1. Review the current `check_exit_signals()` method signatures
2. Update test fixtures to match current method signatures
3. Add missing required parameters

**Verification**:
```bash
pytest tests/test_equity_strategy.py tests/test_mean_reversion.py tests/test_pairs.py -v
```

---

### Fix 2.2: Update Multi-Timeframe Strategy Entry Trigger Tests

**Status**: HIGH - 5+ tests failing
**File**: `tests/test_multi_timeframe_strategy.py`
**Estimated Effort**: 1 hour

**Problem**: The `check_entry_triggers()` method signature or behavior may have changed.

**Affected Tests**:
- `TestEntryTriggers::test_entry_trigger_valid`
- `TestEntryTriggers::test_entry_trigger_below_ma50`
- `TestEntryTriggers::test_entry_trigger_below_breakout`
- `TestEntryTriggers::test_entry_trigger_missing_ma50`
- `TestStopPriceUpdates::test_update_stop_price_fixed_stop`

**Required Action**:
1. Review `EquityMultiTimeframeStrategy.check_entry_triggers()` signature
2. Update test fixtures accordingly
3. Ensure feature data contains all required fields

---

### Fix 2.3: Update Factor Strategy Stop Price Tests

**Status**: HIGH
**File**: `tests/test_factor_strategy.py`
**Estimated Effort**: 30 minutes

**Problem**: Stop price update test is failing.

**Affected Test**:
- `TestStopPriceUpdates::test_update_stop_price_fixed_stop`

**Required Action**:
1. Review `EquityFactorStrategy.update_stop_price()` method
2. Update test to match current implementation

---

## MEDIUM PRIORITY FIXES (Priority 3)

### Fix 3.1: Rebuild Docker Image with New Test Files

**Status**: MEDIUM - Needed for coverage verification
**Estimated Effort**: 15 minutes

**Problem**: 15 new test files were created but Docker image needs rebuilding to include them.

**Required Action**:
```bash
# Rebuild Docker image
docker-compose build trading-system

# Verify new tests are included
docker-compose run --rm --entrypoint pytest trading-system tests/ --collect-only | grep test_storage
```

---

### Fix 3.2: Verify Property-Based Tests Still Pass

**Status**: MEDIUM
**Files**: `tests/property/*.py`
**Estimated Effort**: 30 minutes

**Problem**: 6 property-based tests were previously failing. Fixes were documented as applied but need verification.

**Required Action**:
```bash
pytest tests/property/ -v
```

**Expected Result**: All 6 tests should pass:
- `test_ma_nan_before_window`
- `test_portfolio_equity_always_positive`
- `test_portfolio_exposure_limits`
- `test_portfolio_equity_updates_with_prices`
- `test_sharpe_constant_returns_zero`
- `test_permutation_test_structure`

---

## DOCUMENTATION UPDATES (Priority 4)

### Fix 4.1: Update Test Coverage Documentation

**Status**: LOW
**Files**: `TEST_COVERAGE_STATUS.md`
**Estimated Effort**: 15 minutes

**Required Action**:
1. After fixes are applied, regenerate coverage report
2. Update `TEST_COVERAGE_STATUS.md` with new statistics
3. Document any remaining coverage gaps

---

### Fix 4.2: Commit Modified Files

**Status**: LOW
**Estimated Effort**: 10 minutes

**Currently Modified Files**:
```
M docker-compose.yml
M tests/test_factor_strategy.py
M tests/test_multi_timeframe_strategy.py
M tests/test_storage_database.py
```

**Untracked Files**:
```
?? TEST_COVERAGE_STATUS.md
```

**Required Action**:
1. Review changes in modified files
2. Commit meaningful changes
3. Either commit or .gitignore untracked files

---

## IMPLEMENTATION ORDER

Execute fixes in this order for optimal workflow:

1. **Fix 1.1** - Add `side` parameter to all Position instances in tests (unblocks most tests)
2. **Fix 1.2** - Create/fix ML ensemble module (unblocks ML tests)
3. **Fix 2.1-2.3** - Update strategy test fixtures (unblocks strategy tests)
4. **Verify** - Run full test suite
5. **Fix 3.2** - Verify property-based tests
6. **Fix 3.1** - Rebuild Docker for coverage
7. **Fix 4.1-4.2** - Documentation updates

---

## QUICK FIX SCRIPT

Here's a script to help fix the Position side parameter issue:

```python
#!/usr/bin/env python3
"""Script to add side parameter to Position instantiations in test files."""

import re
import glob

# Find all test files
test_files = glob.glob('tests/*.py') + glob.glob('tests/**/*.py', recursive=True)

for filepath in test_files:
    with open(filepath, 'r') as f:
        content = f.read()

    # Check if file has Position without side
    if 'Position(' in content and 'side=' not in content:
        # Add import if needed
        if 'PositionSide' not in content and 'from trading_system.models.positions import' in content:
            content = content.replace(
                'from trading_system.models.positions import Position',
                'from trading_system.models.positions import Position, PositionSide'
            )

        # Add side=PositionSide.LONG after quantity= line
        # This is a simplified approach - manual review recommended
        print(f"NEEDS UPDATE: {filepath}")
```

---

## VERIFICATION COMMANDS

After applying fixes, run these commands to verify:

```bash
# Quick smoke test
pytest tests/test_portfolio.py tests/test_models.py -v

# Full test suite
pytest tests/ --ignore=tests/integration --ignore=tests/performance -v

# Property-based tests only
pytest tests/property/ -v

# Integration tests (requires Docker)
docker-compose run --rm --entrypoint pytest trading-system tests/integration/ -v

# Coverage report
pytest tests/ --cov=trading_system --cov-report=html --cov-report=term-missing
```

---

## SUCCESS CRITERIA

- [ ] All unit tests pass (0 failures, 0 errors)
- [ ] All property-based tests pass
- [ ] Integration tests pass in Docker
- [ ] Test coverage > 80% (target: > 90%)
- [ ] No uncommitted changes
- [ ] Documentation up to date

---

## NOTES FOR AGENTS

1. **Start with Fix 1.1** - This is the root cause of most failures
2. **Use `PositionSide.LONG`** for existing long-only tests
3. **Add `PositionSide.SHORT`** tests for short selling scenarios
4. **Check method signatures** - Many strategy methods may have changed
5. **Run tests incrementally** - Fix one file, verify, move to next
6. **Keep backwards compatibility** - Don't break working tests

---

**Last Updated**: 2024-12-30
**Author**: Audit Agent

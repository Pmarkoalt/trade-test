# Agent Improvements Roadmap

This document provides a structured roadmap for improving the Trading System based on the comprehensive audit report. Use this as a guide for implementing fixes and enhancements.

## Table of Contents

1. [Immediate Fixes (Priority 1)](#immediate-fixes-priority-1)
2. [Short-Term Improvements (Priority 2-4)](#short-term-improvements-priority-2-4)
3. [Medium-Term Enhancements (Priority 5-6)](#medium-term-enhancements-priority-5-6)
4. [Test Coverage Improvements](#test-coverage-improvements)
5. [Code Quality Tasks](#code-quality-tasks)

---

## Immediate Fixes (Priority 1)

### 🔴 Task 1.1: Fix Failing Property-Based Tests

**Status**: ❌ **6 tests failing**  
**Priority**: CRITICAL  
**Estimated Effort**: 4-6 hours

#### Test Failures

1. **`tests/property/test_indicators.py::test_ma_nan_before_window`**
   - **Issue**: Moving average NaN handling before window period
   - **Expected**: First `window-1` values should be NaN
   - **Location**: `trading_system/indicators/ma.py`
   - **Action**: 
     - Review the `ma()` function implementation
     - Ensure proper NaN assignment for first `window-1` indices
     - Check that `min_periods` parameter is correctly used in `rolling()`
     - Verify no edge cases with empty or very short series

2. **`tests/property/test_portfolio.py::test_portfolio_equity_always_positive`**
   - **Issue**: Portfolio equity can become non-positive after operations
   - **Expected**: Equity must always be positive (>= 0)
   - **Location**: `trading_system/portfolio/portfolio.py`
   - **Action**:
     - Review `process_fill()` method - ensure it doesn't allow positions that would make equity negative
     - Review `update_equity()` method - check calculation logic: `equity = cash + total_exposure`
     - Add safeguards to prevent negative equity (e.g., minimum cash reserve)
     - Check if slippage/fees can cause equity to go negative
     - Consider adding validation: `assert portfolio.equity > 0` after operations

3. **`tests/property/test_portfolio.py::test_portfolio_exposure_limits`**
   - **Issue**: Portfolio may exceed exposure limits
   - **Expected**: 
     - Gross exposure ≤ 80%
     - Per-position exposure ≤ 15%
   - **Location**: `trading_system/portfolio/portfolio.py`
   - **Action**:
     - Review exposure calculation in `update_equity()`
     - Ensure `gross_exposure_pct` is correctly calculated: `total_exposure / equity`
     - Verify `per_position_exposure` dictionary is properly maintained
     - Check if exposure limits are enforced when adding positions
     - Review `process_fill()` to ensure it respects limits before adding positions

4. **`tests/property/test_portfolio.py::test_portfolio_equity_updates_with_prices`**
   - **Issue**: Portfolio equity doesn't update correctly with price changes
   - **Expected**: Equity should increase when prices go up, decrease when prices go down
   - **Location**: `trading_system/portfolio/portfolio.py::update_equity()`
   - **Action**:
     - Review `update_equity()` method logic
     - Ensure `unrealized_pnl` is correctly calculated from position price changes
     - Verify: `equity = cash + sum(position_values)` where `position_value = current_price * quantity`
     - Check that position updates properly handle price changes
     - Ensure positions are correctly marked as open/closed

5. **`tests/property/test_validation.py::test_sharpe_constant_returns_zero`**
   - **Issue**: Sharpe ratio calculation with constant returns
   - **Expected**: Constant R-multiples (zero std) should return zero Sharpe ratio
   - **Location**: `trading_system/validation/bootstrap.py::compute_sharpe_from_r_multiples()`
   - **Action**:
     - Review Sharpe calculation: `sharpe = mean(returns) / std(returns) * sqrt(252)` (annualized)
     - Handle edge case: when `std(returns) == 0`, return 0.0 instead of NaN or inf
     - Add check: `if std_dev == 0: return 0.0`
     - Verify annualization factor is correct

6. **`tests/property/test_validation.py::test_permutation_test_structure`**
   - **Issue**: Permutation test structure validation
   - **Expected**: Returns valid structure with all required fields
   - **Location**: `trading_system/validation/permutation.py`
   - **Action**:
     - Review `PermutationTest.run()` method
     - Ensure it returns dictionary with keys: `actual_sharpe`, `random_sharpe_5th`, `random_sharpe_95th`, `percentile_rank`, `passed`
     - Verify all values are finite (not NaN or inf)
     - Check `percentile_rank` is in range [0.0, 100.0]
     - Review edge cases: empty trades, single trade, etc.

#### Implementation Steps

1. **Run failing tests** to see exact error messages:
   ```bash
   pytest tests/property/test_indicators.py::test_ma_nan_before_window -v
   pytest tests/property/test_portfolio.py -v
   pytest tests/property/test_validation.py -v
   ```

2. **Analyze each failure**:
   - Read the test code to understand expected behavior
   - Review the implementation code
   - Identify root cause of failure

3. **Fix the implementation**:
   - Make minimal changes to fix the issue
   - Preserve existing functionality
   - Add defensive checks where appropriate

4. **Verify fixes**:
   - Run the specific test
   - Run all property tests: `pytest tests/property/ -v`
   - Run full test suite to ensure no regressions

5. **Update documentation** if behavior changes

---

### 🔴 Task 1.2: Commit Modified Files

**Status**: ⚠️ **Pending**  
**Priority**: HIGH  
**Estimated Effort**: 15 minutes

#### Action Items

1. **Review modified file**: `trading_system/backtest/engine.py`
   - Check what changes were made
   - Ensure changes are intentional and correct
   - Review diff: `git diff trading_system/backtest/engine.py`

2. **Review untracked documentation files**:
   - `ENVIRONMENT_ISSUE.md`
   - `TEST_DEBUG_SUMMARY.md`
   - Decide: commit or add to `.gitignore`

3. **Commit changes**:
   ```bash
   git add trading_system/backtest/engine.py
   git commit -m "Fix: [describe changes]"
   ```

4. **Handle documentation files**:
   - If permanent documentation: `git add ENVIRONMENT_ISSUE.md TEST_DEBUG_SUMMARY.md`
   - If temporary: Add to `.gitignore`

---

## Short-Term Improvements (Priority 2-4)

### 🟡 Task 2.1: Add Short Selling Support

**Status**: ❌ **Missing**  
**Priority**: HIGH  
**Estimated Effort**: 16-24 hours

#### Current State

- System is **long-only** currently
- Mean reversion strategy could benefit from short signals
- Pairs trading is partially implemented (needs shorts)

#### Implementation Requirements

1. **Position Model Updates** (`trading_system/models/positions.py`):
   - Add `side` field to `Position` (LONG/SHORT)
   - Update P&L calculation for short positions:
     - Long: `(current_price - entry_price) * quantity`
     - Short: `(entry_price - current_price) * quantity`
   - Update stop loss logic for shorts (stop above entry, not below)
   - Update exit conditions for shorts

2. **Portfolio Updates** (`trading_system/portfolio/portfolio.py`):
   - Update `process_fill()` to handle short positions
   - Modify `update_equity()` to correctly value short positions
   - Update exposure calculation:
     - Gross exposure = |long positions| + |short positions|
     - Net exposure = long - short
   - Add short position limits (separate from long limits)

3. **Strategy Updates**:
   - **Mean Reversion** (`trading_system/strategies/mean_reversion.py`):
     - Add short signals when price is above mean
     - Update entry/exit logic for shorts
   - **Pairs Trading** (`trading_system/strategies/pairs.py`):
     - Complete long-short pair implementation
     - Add hedge ratio calculation
     - Update exit logic for pair trades

4. **Risk Management Updates**:
   - Update correlation guard for short positions
   - Add separate risk limits for shorts
   - Update volatility scaling for shorts

5. **Execution Updates** (`trading_system/execution/`):
   - Update slippage model for shorts (may have different impact)
   - Consider short sale restrictions (uptick rule, availability)
   - Add short borrow cost modeling

6. **Testing**:
   - Add unit tests for short position handling
   - Add property tests for short positions
   - Add integration tests for short strategies
   - Test edge cases: short squeeze scenarios

#### Design Considerations

- **Entry**: Short when signal indicates overvaluation
- **Stop Loss**: Place stop above entry price (opposite of long)
- **Exit**: Exit when mean reversion occurs or stop hit
- **Exposure Limits**: 
  - Gross exposure limit still applies (80%)
  - Net exposure limit (e.g., ±40%)
  - Separate limits for longs vs shorts

#### Files to Modify

- `trading_system/models/positions.py`
- `trading_system/portfolio/portfolio.py`
- `trading_system/strategies/mean_reversion.py`
- `trading_system/strategies/pairs.py`
- `trading_system/execution/slippage.py` (if needed)
- `trading_system/portfolio/risk_scaling.py` (if needed)
- `tests/test_portfolio.py`
- `tests/test_mean_reversion.py`
- `tests/test_pairs.py`

---

### 🟡 Task 2.2: Improve ML Integration

**Status**: ⚠️ **Basic Implementation**  
**Priority**: MEDIUM  
**Estimated Effort**: 20-30 hours

#### Current State

- ML integration exists in `trading_system/ml/`
- Basic feature engineering
- Limited model support

#### Enhancement Requirements

1. **Feature Engineering** (`trading_system/ml/features.py`):
   - Add technical indicators as features:
     - RSI, MACD, Bollinger Bands
     - Volume indicators (OBV, volume profile)
     - Volatility features (ATR, realized vol)
   - Add market regime features:
     - Trend detection
     - Volatility regime classification
     - Market breadth indicators
   - Add cross-asset features:
     - Sector/industry momentum
     - Correlation features
     - Relative strength vs benchmark

2. **Ensemble Models** (`trading_system/ml/models.py`):
   - Implement ensemble methods:
     - Voting classifier/regressor
     - Stacking
     - Boosting (XGBoost, LightGBM)
   - Add model selection framework
   - Add hyperparameter tuning support

3. **Online Learning**:
   - Implement incremental learning for models
   - Add concept drift detection
   - Implement model retraining pipeline
   - Add model versioning

4. **Feature Store**:
   - Implement feature caching/storage
   - Add feature versioning
   - Add feature validation

5. **Model Evaluation**:
   - Add backtesting-specific metrics
   - Add model performance tracking
   - Add feature importance analysis
   - Add prediction confidence intervals

6. **Integration Points**:
   - Update strategies to use ML predictions
   - Add ML signal scoring
   - Integrate with signal queue

#### Files to Modify/Create

- `trading_system/ml/features.py` (enhance)
- `trading_system/ml/models.py` (add ensemble support)
- `trading_system/ml/online_learning.py` (new)
- `trading_system/ml/ensemble.py` (new)
- `trading_system/ml/feature_store.py` (new)
- `tests/test_ml_features.py` (expand)
- `tests/test_ml_ensemble.py` (new)

---

### 🟡 Task 2.3: Add Live Trading Adapter Tests

**Status**: ⚠️ **Skeleton Exists**  
**Priority**: MEDIUM  
**Estimated Effort**: 12-16 hours

#### Current State

- Adapters exist: `alpaca_adapter.py`, `ib_adapter.py`
- No integration tests with paper trading APIs

#### Implementation Requirements

1. **Paper Trading Test Setup**:
   - Configure paper trading accounts (Alpaca, IB paper)
   - Set up test credentials (use environment variables)
   - Create mock/test adapters for unit testing

2. **Unit Tests** (`tests/test_adapters.py`):
   - Test connection/disconnection
   - Test order submission/cancellation
   - Test position querying
   - Test market data retrieval
   - Test error handling

3. **Integration Tests** (`tests/integration/test_live_trading.py`):
   - Test full order lifecycle (submit -> fill -> cancel)
   - Test position tracking
   - Test account balance updates
   - Test with paper trading accounts
   - Add rate limiting tests
   - Add reconnection logic tests

4. **Mock Adapters** (for testing without API access):
   - Create mock implementations
   - Simulate order fills
   - Simulate market data
   - Simulate errors/timeouts

5. **Error Handling**:
   - Test network failures
   - Test API rate limits
   - Test invalid orders
   - Test insufficient funds
   - Test position limits

#### Files to Create/Modify

- `tests/test_adapters.py` (new or expand)
- `tests/integration/test_live_trading.py` (new)
- `tests/fixtures/mock_adapter.py` (new)
- `trading_system/adapters/base_adapter.py` (if not exists)
- Update `trading_system/adapters/alpaca_adapter.py` (add error handling)
- Update `trading_system/adapters/ib_adapter.py` (add error handling)

#### Testing Approach

1. **Unit Tests**: Use mocks, no API calls
2. **Integration Tests**: Use paper trading APIs (marked as `@pytest.mark.integration`)
3. **CI/CD**: Skip integration tests in CI, run manually

---

## Medium-Term Enhancements (Priority 5-6)

### 🟢 Task 3.1: Performance Optimization

**Status**: ⚠️ **Opportunities Identified**  
**Priority**: MEDIUM  
**Estimated Effort**: 16-24 hours

#### Optimization Opportunities

1. **Indicator Caching**:
   - Current: Basic caching exists
   - Enhance: More aggressive caching
   - Action:
     - Cache indicators across strategies
     - Add cache invalidation logic
     - Profile cache hit rates
     - Optimize cache key generation

2. **Parallel Backtest Runs**:
   - Current: Sequential parameter sensitivity runs
   - Enhance: Parallel execution
   - Action:
     - Use `multiprocessing` or `joblib` for parallel runs
     - Add progress tracking
     - Optimize memory usage for parallel runs
     - Add result aggregation

3. **Memory Optimization**:
   - Current: Load all data into memory
   - Enhance: Chunked/lazy loading for large universes
   - Action:
     - Implement chunked data loading
     - Use lazy evaluation for indicators
     - Optimize DataFrame memory usage (dtypes)
     - Profile memory usage with large datasets

4. **Vectorization**:
   - Review indicator calculations
   - Replace loops with vectorized operations
   - Use NumPy/Pandas optimizations

#### Files to Modify

- `trading_system/indicators/cache.py` (enhance)
- `trading_system/validation/sensitivity.py` (add parallel execution)
- `trading_system/data/loaders.py` (add chunked loading)
- `trading_system/backtest/engine.py` (profile and optimize)

---

### 🟢 Task 3.2: Documentation Improvements

**Status**: ⚠️ **Good but can improve**  
**Priority**: LOW  
**Estimated Effort**: 8-12 hours

#### Documentation Tasks

1. **API Documentation**:
   - Auto-generate from docstrings using Sphinx
   - Add code examples to docstrings
   - Document all public functions/classes

2. **Example Notebooks** (`examples/notebooks/`):
   - Add more example notebooks:
     - Short selling example
     - ML workflow example
     - Custom strategy example
     - Live trading example

3. **README Updates**:
   - Update test coverage badge
   - Add performance benchmarks
   - Add architecture diagrams
   - Update quick start guide

4. **User Guide** (`docs/user_guide/`):
   - Expand strategy customization guide
   - Add troubleshooting section
   - Add best practices guide

#### Tools to Use

- Sphinx for API docs
- Jupyter notebooks for examples
- Mermaid for diagrams
- Coverage badge service

---

## Test Coverage Improvements

### 📊 Task 4.1: Increase Test Coverage to >90%

**Status**: ⚠️ **Current: ~37.71%, Target: >90%**  
**Priority**: HIGH  
**Estimated Effort**: 40-60 hours

#### Low Coverage Areas

1. **Storage/Schema Modules** (15.62%):
   - `trading_system/storage/`
   - `trading_system/schema/`
   - Action: Add comprehensive tests for database operations, schema validation

2. **Factor Strategy** (15.89%):
   - `trading_system/strategies/factor_strategy.py`
   - Action: Add unit tests, integration tests, edge case tests

3. **Multi-Timeframe Strategy** (19.44%):
   - `trading_system/strategies/multi_timeframe.py`
   - Action: Test multiple timeframe coordination, signal generation

4. **Strategy Loader/Registry** (22-37%):
   - `trading_system/strategies/loader.py`
   - `trading_system/strategies/registry.py`
   - Action: Test strategy loading, registration, discovery

5. **Sensitivity Analysis** (32.61%):
   - `trading_system/validation/sensitivity.py`
   - Action: Test grid search, parameter combinations, result aggregation

#### Coverage Strategy

1. **Identify gaps**:
   ```bash
   pytest --cov=trading_system --cov-report=html
   # Review htmlcov/index.html
   ```

2. **Prioritize**:
   - Start with core modules (portfolio, strategies, backtest)
   - Then supporting modules (indicators, execution)
   - Finally utilities (storage, reporting)

3. **Write tests**:
   - Unit tests for each function/method
   - Edge case tests
   - Integration tests for workflows
   - Property-based tests for invariants

4. **Maintain coverage**:
   - Add coverage check to CI/CD
   - Require coverage increase for new code
   - Regular coverage reports

#### Files Needing Tests

- `trading_system/storage/database.py`
- `trading_system/storage/schema.py`
- `trading_system/strategies/factor_strategy.py`
- `trading_system/strategies/multi_timeframe.py`
- `trading_system/strategies/loader.py`
- `trading_system/strategies/registry.py`
- `trading_system/validation/sensitivity.py`

---

## Code Quality Tasks

### 🧹 Task 5.1: Code Cleanup

**Status**: ✅ **Generally Good**  
**Priority**: LOW  
**Estimated Effort**: 4-8 hours

#### Tasks

1. **Remove dead code**:
   - Find unused functions/classes
   - Remove commented-out code
   - Remove duplicate code

2. **Improve type hints**:
   - Add missing type hints
   - Use `typing` module properly
   - Add return type annotations

3. **Code formatting**:
   - Run `black` formatter
   - Run `isort` for imports
   - Fix linting issues

4. **Documentation**:
   - Add missing docstrings
   - Fix docstring format
   - Add type info to docstrings

#### Tools

- `black` for formatting
- `isort` for import sorting
- `mypy` for type checking
- `pylint` or `flake8` for linting

---

## Implementation Guidelines

### General Approach

1. **Start with Priority 1 tasks** (failing tests)
2. **One task at a time** - complete before moving to next
3. **Test thoroughly** - add tests for new code
4. **Document changes** - update docstrings, README if needed
5. **Maintain backward compatibility** - don't break existing functionality

### Testing Requirements

- All new code must have tests
- Property-based tests for critical invariants
- Integration tests for workflows
- Edge case coverage

### Code Review Checklist

- [ ] Tests pass
- [ ] Coverage maintained/increased
- [ ] Documentation updated
- [ ] Type hints added
- [ ] No linting errors
- [ ] Backward compatible (if applicable)

### Git Workflow

1. Create feature branch: `git checkout -b fix/property-tests`
2. Make changes and commit
3. Run tests: `pytest tests/`
4. Check coverage: `pytest --cov`
5. Create pull request
6. Review and merge

---

## Success Criteria

### Immediate (Priority 1)
- ✅ All 6 property-based tests passing
- ✅ Modified files committed
- ✅ Documentation files properly handled

### Short-Term (Priority 2-4)
- ✅ Short selling fully implemented and tested
- ✅ ML integration enhanced with ensemble support
- ✅ Live trading adapters tested with paper trading
- ✅ Test coverage >90%

### Medium-Term (Priority 5-6)
- ✅ Performance optimizations implemented
- ✅ Documentation comprehensive and up-to-date

---

## Notes

- **Estimated efforts** are rough estimates - actual time may vary
- **Priorities** can be adjusted based on business needs
- **Test failures** should be fixed before adding new features
- **Backward compatibility** should be maintained unless breaking changes are explicitly needed

---

## Questions or Issues?

If you encounter issues or need clarification:
1. Check existing documentation
2. Review test cases for examples
3. Check error messages and stack traces
4. Review similar implementations in codebase

---

**Last Updated**: [Date]  
**Version**: 1.0


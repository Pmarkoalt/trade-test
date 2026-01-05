# Parallel Agent Prompts

Prompts for agents working in parallel on different modules. Each prompt is self-contained with all necessary context.

---

## Foundation Layer (Start First)

### Agent 1: Data Structures & Core Models

**Prompt:**

```
You are implementing the core data structures for a momentum trading system.

TASK: Create Python classes/dataclasses for all data models used throughout the system.

REFERENCE DOCUMENTATION:
- DATA_STRUCTURES.md (complete specifications)
- EDGE_CASES.md (validation rules)

REQUIREMENTS:
1. Implement all classes from DATA_STRUCTURES.md:
   - Bar (OHLCV data)
   - FeatureRow (indicators)
   - Signal (entry intent)
   - Order (execution intent)
   - Fill (execution result)
   - Position (open position)
   - Portfolio (portfolio state)
   - MarketData (data container)
   - StrategyConfig (Pydantic model)

2. Use dataclasses for simple models, Pydantic for configs
3. Include all validation logic from EDGE_CASES.md
4. Add type hints for all fields
5. Include helper methods as specified

DELIVERABLES:
- trading_system/models/bar.py
- trading_system/models/features.py
- trading_system/models/signals.py
- trading_system/models/orders.py
- trading_system/models/positions.py
- trading_system/models/portfolio.py
- trading_system/models/market_data.py
- trading_system/configs/strategy_config.py

TESTING:
- Unit tests for each class
- Validation tests for edge cases
- Test file: tests/test_models.py

Start with Bar and FeatureRow, then move to more complex models.
```

---

### Agent 2: Data Loading & Validation

**Prompt:**

```
You are implementing the data loading pipeline for a momentum trading system.

TASK: Create data loading functions with validation and quality checks.

REFERENCE DOCUMENTATION:
- DATA_LOADING_SPEC.md (complete file format specs)
- EDGE_CASES.md (sections 1-4: data quality edge cases)
- 03_DATA_PIPELINE_AND_VALIDATION.md (data requirements)

REQUIREMENTS:
1. Implement all loading functions from DATA_LOADING_SPEC.md:
   - load_ohlcv_data() - Load CSV files for symbols
   - load_universe() - Load universe lists
   - load_benchmark() - Load SPY/BTC benchmarks
   - validate_ohlcv() - Validate OHLC relationships
   - detect_missing_data() - Find gaps in data
   - load_all_data() - Complete loading pipeline

2. Handle all edge cases from EDGE_CASES.md:
   - Missing data (single day, consecutive)
   - Invalid OHLC data
   - Extreme price moves (>50%)
   - Symbol not in universe
   - Benchmark data missing

3. Support both equity (market calendar) and crypto (365 days)

DELIVERABLES:
- trading_system/data/loader.py
- trading_system/data/validator.py
- trading_system/data/calendar.py (trading calendar handling)
- trading_system/data/__init__.py

TESTING:
- Unit tests with sample CSV files
- Test missing data handling
- Test validation edge cases
- Test file: tests/test_data_loading.py

Create sample test data files in tests/fixtures/
```

---

## Core Components Layer (Can Start After Foundation)

### Agent 3: Indicators Library

**Prompt:**

```
You are implementing the technical indicators library for a momentum trading system.

TASK: Create vectorized indicator calculation functions.

REFERENCE DOCUMENTATION:
- 04_INDICATORS_LIBRARY.md (indicator specifications)
- EDGE_CASES.md (section 5-6: indicator edge cases)
- DATA_STRUCTURES.md (FeatureRow structure)

REQUIREMENTS:
1. Implement all indicator functions:
   - ma() - Moving averages (20, 50, 200)
   - atr() - Average True Range (Wilder's smoothing, period 14)
   - roc() - Rate of change (60-day)
   - highest_close() - Highest close over window (exclude today)
   - adv() - Average dollar volume (20-day)
   - rolling_corr() - Rolling correlation (20-day)

2. Handle edge cases:
   - Return NaN until sufficient lookback
   - Never forward-fill NaN values
   - highest_close_20d and highest_close_55d exclude today's close

3. Create compute_features() function that computes all indicators for a symbol

4. Use vectorized operations (pandas/numpy) for performance

DELIVERABLES:
- trading_system/indicators/ma.py
- trading_system/indicators/atr.py
- trading_system/indicators/momentum.py (ROC)
- trading_system/indicators/breakouts.py (highest_close)
- trading_system/indicators/volume.py (ADV)
- trading_system/indicators/correlation.py
- trading_system/indicators/feature_computer.py (main compute_features function)
- trading_system/indicators/__init__.py

TESTING:
- Unit tests with known data (verify calculations)
- Test edge cases (insufficient lookback, NaN handling)
- Test no-lookahead (verify highest_close excludes today)
- Test file: tests/test_indicators.py

Use known indicator values from external sources to verify correctness.
```

---

### Agent 4: Portfolio Management

**Prompt:**

```
You are implementing the portfolio management system for a momentum trading system.

TASK: Create portfolio state management with position sizing and risk controls.

REFERENCE DOCUMENTATION:
- 05_PORTFOLIO_AND_RISK.md (portfolio specifications)
- PORTFOLIO_STATE_MACHINE.md (state update sequence)
- EDGE_CASES.md (sections 7-13: position and portfolio edge cases)
- DATA_STRUCTURES.md (Portfolio and Position classes)

REQUIREMENTS:
1. Implement Portfolio class with:
   - Cash and equity tracking
   - Position management (add/remove/update)
   - Equity curve and daily returns
   - Exposure calculations
   - Volatility scaling (20D portfolio vol vs 252D median)
   - Correlation metrics (avg pairwise correlation)

2. Implement position sizing:
   - Risk-based sizing (0.75% of equity per trade)
   - Apply volatility scaling multiplier
   - Clamp to max position notional (15%)
   - Clamp to max exposure (80%)
   - Handle insufficient cash edge case

3. Implement stop management:
   - Trailing stops (can only move up for longs)
   - Crypto staged exit (tighten stop after MA20 break)

4. Handle all edge cases from EDGE_CASES.md

DELIVERABLES:
- trading_system/portfolio/portfolio.py
- trading_system/portfolio/position_sizing.py
- trading_system/portfolio/risk_scaling.py
- trading_system/portfolio/correlation.py
- trading_system/portfolio/__init__.py

TESTING:
- Unit tests for position sizing
- Unit tests for volatility scaling
- Unit tests for correlation calculations
- Test edge cases (insufficient cash, insufficient history)
- Test file: tests/test_portfolio.py
```

---

### Agent 5: Execution Model

**Prompt:**

```
You are implementing the execution model with realistic slippage and fees.

TASK: Create execution simulation with dynamic slippage model.

REFERENCE DOCUMENTATION:
- 09_EXECUTION_AND_COSTS.md (execution specifications)
- EDGE_CASES.md (section 14: execution edge cases)
- DATA_STRUCTURES.md (Fill class)
- MASTER_FILE.md (sections on slippage model)

REQUIREMENTS:
1. Implement slippage calculation:
   - Base slippage (8 bps equity, 10 bps crypto)
   - Volatility multiplier (ATR14 / mean(ATR14 60D))
   - Size penalty (order_notional / 1% ADV20)
   - Weekend penalty (1.5x for crypto Sat/Sun UTC)
   - Stress multiplier (2x during stress weeks)
   - Normal distribution with variance

2. Implement fee calculation:
   - Equity: 1 bp per side
   - Crypto: 8 bps per side

3. Implement weekly return calculation:
   - Equity: Last 5 trading days
   - Crypto: Last 7 calendar days

4. Handle edge cases:
   - Slippage bounds (0 to 500 bps)
   - Missing data (reject order)
   - Extreme values

DELIVERABLES:
- trading_system/execution/slippage.py
- trading_system/execution/fees.py
- trading_system/execution/fill_simulator.py
- trading_system/execution/capacity.py (capacity constraint checks)
- trading_system/execution/__init__.py

TESTING:
- Unit tests for slippage calculation
- Unit tests for fee calculation
- Test stress multiplier logic
- Test edge cases (bounds, missing data)
- Test file: tests/test_execution.py

Use seeded RNG for reproducibility.
```

---

## Strategy Layer (Needs Indicators)

### Agent 6: Strategy Factory - Equity

**Prompt:**

```
You are implementing the equity momentum strategy.

TASK: Create equity strategy with eligibility filters, entry triggers, and exit logic.

REFERENCE DOCUMENTATION:
- 06_STRATEGY_EQUITY.md (equity strategy specifications)
- EXAMPLE_CONFIGS/equity_config.yaml (config example)
- DATA_STRUCTURES.md (Signal, StrategyConfig)
- EDGE_CASES.md (relevant sections)

REQUIREMENTS:
1. Implement EquityStrategy class:
   - Load config from YAML (see EXAMPLE_CONFIGS/)
   - Eligibility filter:
     * close > MA50
     * MA50 slope > 0.5% over 20 days
     * Optional: relative strength vs SPY (v1.1)
   - Entry triggers (OR logic):
     * Fast: close >= highest_close_20d * 1.005
     * Slow: close >= highest_close_55d * 1.010
   - Exit logic:
     * Trailing: close < MA20 (or MA50, configurable)
     * Hard stop: close < entry - 2.5 * ATR14
   - Capacity check: order_notional <= 0.5% * ADV20

2. Implement signal generation:
   - Check eligibility
   - Check entry triggers
   - Calculate stop price
   - Check capacity
   - Create Signal object

3. Implement stop updates:
   - Update trailing stops daily
   - Check exit signals (priority: hard stop > trailing MA)

DELIVERABLES:
- trading_system/strategies/equity_strategy.py
- trading_system/strategies/base_strategy.py (base class)
- trading_system/strategies/__init__.py

TESTING:
- Unit tests for eligibility filters
- Unit tests for entry triggers
- Unit tests for exit logic
- Test with known data (verify signals)
- Test file: tests/test_equity_strategy.py
```

---

### Agent 7: Strategy Factory - Crypto

**Prompt:**

```
You are implementing the crypto momentum strategy.

TASK: Create crypto strategy with staged exit logic.

REFERENCE DOCUMENTATION:
- 07_STRATEGY_CRYPTO.md (crypto strategy specifications)
- EXAMPLE_CONFIGS/crypto_config.yaml (config example)
- DATA_STRUCTURES.md (Signal, StrategyConfig)
- EDGE_CASES.md (relevant sections)

REQUIREMENTS:
1. Implement CryptoStrategy class:
   - Load config from YAML (see EXAMPLE_CONFIGS/)
   - Eligibility filter:
     * close > MA200 (STRICT, no exceptions)
     * Optional: relative strength vs BTC (v1.1)
   - Entry triggers (same as equity):
     * Fast: close >= highest_close_20d * 1.005
     * Slow: close >= highest_close_55d * 1.010
   - Exit logic (staged):
     * Stage 1: close < MA20 → tighten stop to 2.0 * ATR14
     * Stage 2: close < MA50 OR tightened stop hit → exit
   - Hard stop: entry - 3.0 * ATR14 (wider than equity)
   - Capacity check: order_notional <= 0.25% * ADV20 (stricter)

2. Implement staged exit logic:
   - Track tightened_stop flag
   - Tighten once when MA20 breaks (never reset)
   - Exit on MA50 or tightened stop

3. Same signal generation structure as equity

DELIVERABLES:
- trading_system/strategies/crypto_strategy.py
- Extend base_strategy.py if needed
- Update trading_system/strategies/__init__.py

TESTING:
- Unit tests for eligibility (MA200 strict)
- Unit tests for staged exit logic
- Test stop tightening (verify it happens once)
- Test file: tests/test_crypto_strategy.py
```

---

### Agent 8: Signal Scoring & Queue

**Prompt:**

```
You are implementing the position queue ranking system.

TASK: Create signal scoring and selection when signals exceed available slots.

REFERENCE DOCUMENTATION:
- 08_SIGNAL_SCORING_AND_QUEUE.md (scoring specifications)
- EDGE_CASES.md (section 13: position queue edge cases)
- 05_PORTFOLIO_AND_RISK.md (correlation guard)

REQUIREMENTS:
1. Implement scoring function:
   - Breakout strength: (close - MA) / ATR14
   - Momentum strength: relative ROC60 vs benchmark
   - Diversification bonus: 1 - avg_corr_to_portfolio
   - Rank-normalize each component (0-1 scale)
   - Weighted score: 0.5 * breakout + 0.3 * momentum + 0.2 * diversification

2. Implement selection logic:
   - Sort by score (descending)
   - Apply constraints in order:
     * Max positions
     * Max exposure
     * Capacity constraint
     * Correlation guard (if >= 4 positions)
   - Return selected signals

3. Implement correlation guard:
   - Compute avg pairwise correlation of portfolio
   - If > 0.70 and >= 4 positions:
     * Reject candidates with corr > 0.75 to portfolio

DELIVERABLES:
- trading_system/strategies/scoring.py
- trading_system/strategies/queue.py
- Update trading_system/strategies/__init__.py

TESTING:
- Unit tests for scoring calculation
- Unit tests for rank normalization
- Unit tests for selection logic
- Test correlation guard
- Test file: tests/test_scoring.py
```

---

## Engine Layer (Needs All Above)

### Agent 9: Backtest Engine

**Prompt:**

```
You are implementing the walk-forward backtest engine.

TASK: Create event-driven daily loop with no lookahead.

REFERENCE DOCUMENTATION:
- 10_BACKTEST_ENGINE.md (backtest specifications)
- PORTFOLIO_STATE_MACHINE.md (detailed state update sequence)
- 11_WALK_FORWARD_SPLITS_AND_PREREG.md (split protocol)
- EDGE_CASES.md (all sections)

REQUIREMENTS:
1. Implement daily event loop:
   - Update data through day t close
   - Generate signals at day t close
   - Create orders for day t+1 open
   - Execute orders at day t+1 open
   - Update stops at day t+1 close
   - Check exits at day t+1 close
   - Execute exit orders at day t+2 open
   - Update portfolio metrics
   - Log daily state

2. Implement walk-forward splits:
   - Train/validation/holdout date ranges
   - Load from config
   - Run only on specified period

3. Ensure no lookahead:
   - Indicators use data <= current date
   - Orders created at t execute at t+1
   - Stops updated at t+1 check exits at t+2

4. Handle all edge cases from EDGE_CASES.md

DELIVERABLES:
- trading_system/backtest/engine.py
- trading_system/backtest/event_loop.py
- trading_system/backtest/splits.py
- trading_system/backtest/__init__.py

TESTING:
- Integration test with toy dataset
- Verify no lookahead (test with known data)
- Verify execution timing
- Test file: tests/test_backtest_engine.py

Create a small 3-symbol, 3-month test dataset with known expected trades.
```

---

## Analysis Layer (Needs Backtest Engine)

### Agent 10: Validation Suite

**Prompt:**

```
You are implementing the validation and robustness testing suite.

TASK: Create validation tests for parameter sensitivity, bootstrap, permutation, and stress tests.

REFERENCE DOCUMENTATION:
- 12_VALIDATION_SUITE.md (validation specifications)
- ALGORITHMS_PSEUDOCODE.md (bootstrap, permutation algorithms)
- EDGE_CASES.md (relevant sections)

REQUIREMENTS:
1. Implement parameter sensitivity grid:
   - Grid search over tunable parameters
   - Plot heatmaps (use matplotlib)
   - Check for sharp peaks (overfitting indicator)
   - Select robust parameters

2. Implement bootstrap test:
   - Resample trade R-multiples (1000 iterations)
   - Compute confidence intervals
   - Check 5th percentile Sharpe >= 0.4

3. Implement permutation test:
   - Randomize entry dates (1000 iterations)
   - Preserve exit logic and holding periods
   - Check actual Sharpe >= 95th percentile random

4. Implement stress tests:
   - Slippage multipliers (1x, 2x, 3x)
   - Bear market months only
   - Range market months
   - Flash crash simulation

5. Implement correlation stress analysis:
   - Compare correlation during drawdowns vs normal
   - Warn if correlation > 0.70 during stress

DELIVERABLES:
- trading_system/validation/sensitivity.py
- trading_system/validation/bootstrap.py
- trading_system/validation/permutation.py
- trading_system/validation/stress_tests.py
- trading_system/validation/correlation_analysis.py
- trading_system/validation/__init__.py

TESTING:
- Unit tests for each validation test
- Test with known data (verify algorithms)
- Test file: tests/test_validation.py

Use seeded RNG for reproducibility.
```

---

### Agent 11: Metrics & Reporting

**Prompt:**

```
You are implementing the metrics calculation and reporting system.

TASK: Create metrics computation and output generation (CSV/JSON).

REFERENCE DOCUMENTATION:
- 13_METRICS_AND_GO_NO_GO.md (metrics specifications)
- 10_BACKTEST_ENGINE.md (output formats)
- MASTER_FILE.md (metrics definitions)

REQUIREMENTS:
1. Implement primary metrics:
   - Sharpe ratio (annualized)
   - Max drawdown
   - Calmar ratio
   - Total trades count

2. Implement secondary metrics:
   - Expectancy (R-multiples)
   - Profit factor
   - Correlation to benchmark
   - 99th percentile daily loss

3. Implement tertiary metrics:
   - Recovery factor
   - Drawdown duration
   - Turnover (trades/month)
   - Average holding period
   - Max consecutive losses
   - Win rate

4. Implement output generation:
   - equity_curve.csv (daily equity, cash, positions, exposure)
   - trade_log.csv (all closed trades with details)
   - weekly_summary.csv (weekly aggregated metrics)
   - monthly_report.json (monthly metrics)
   - scenario_comparison.json (stress test results)

DELIVERABLES:
- trading_system/reporting/metrics.py
- trading_system/reporting/csv_writer.py
- trading_system/reporting/json_writer.py
- trading_system/reporting/__init__.py

TESTING:
- Unit tests for each metric calculation
- Test output file formats
- Test file: tests/test_reporting.py

Verify metrics against known calculations.
```

---

## Integration Layer (Needs Everything)

### Agent 12: CLI & Integration

**Prompt:**

```
You are implementing the CLI interface and integrating all modules.

TASK: Create command-line interface and wire up all components.

REFERENCE DOCUMENTATION:
- 15_CLI_COMMANDS.md (CLI specifications)
- EXAMPLE_CONFIGS/run_config.yaml (config structure)
- All previous modules

REQUIREMENTS:
1. Implement CLI commands:
   - `backtest --config <path>` - Run backtest
   - `validate --config <path>` - Run validation suite
   - `holdout --config <path>` - Run holdout evaluation
   - `report --run-id <id>` - Generate reports

2. Implement config loading:
   - Load run_config.yaml
   - Load strategy configs (equity, crypto)
   - Validate config structure

3. Integrate all modules:
   - Data loading → Indicators → Strategies → Portfolio → Execution → Backtest → Reporting

4. Add logging:
   - Use loguru or rich for structured logging
   - Log levels: DEBUG, INFO, WARNING, ERROR
   - Log to file and console

DELIVERABLES:
- trading_system/cli.py
- trading_system/configs/run_config.py (Pydantic model)
- trading_system/integration/runner.py (main integration logic)
- trading_system/__init__.py

TESTING:
- Test CLI commands with sample configs
- Test end-to-end with toy dataset
- Test file: tests/test_cli.py

Create example run configs in tests/fixtures/
```

---

## Testing Agent (Parallel with Implementation)

### Agent 13: Test Suite & Fixtures

**Prompt:**

```
You are creating comprehensive test fixtures and test utilities.

TASK: Create test data, fixtures, and helper functions for all test suites.

REQUIREMENTS:
1. Create sample data files:
   - 3-symbol equity dataset (3 months, known signals)
   - 3-symbol crypto dataset (3 months, known signals)
   - Benchmark data (SPY, BTC)
   - Known expected trades for verification

2. Create test fixtures:
   - Sample configs (equity, crypto, run)
   - Sample portfolios (various states)
   - Sample signals, orders, fills
   - Edge case data (missing data, invalid OHLC, etc.)

3. Create test utilities:
   - Helper functions for creating test data
   - Assertion helpers (check no lookahead, etc.)
   - Mock data generators

4. Create integration test dataset:
   - Small but realistic dataset
   - Known expected trades
   - Verify system produces expected results

DELIVERABLES:
- tests/fixtures/equity_sample.csv
- tests/fixtures/crypto_sample.csv
- tests/fixtures/benchmarks/
- tests/fixtures/configs/
- tests/utils/test_helpers.py
- tests/utils/assertions.py
- tests/integration/test_end_to_end.py

TESTING:
- Verify fixtures are correct
- Verify test utilities work
- Run integration test to verify system

Create fixtures that can be used by all other test suites.
```

---

## Summary

### Parallelization Strategy

**Phase 1 (Foundation - Can Start Immediately):**
- Agent 1: Data Structures
- Agent 2: Data Loading
- Agent 13: Test Fixtures

**Phase 2 (Core Components - After Foundation):**
- Agent 3: Indicators
- Agent 4: Portfolio
- Agent 5: Execution

**Phase 3 (Strategy - After Indicators):**
- Agent 6: Equity Strategy
- Agent 7: Crypto Strategy
- Agent 8: Signal Scoring

**Phase 4 (Engine - After All Above):**
- Agent 9: Backtest Engine

**Phase 5 (Analysis - After Engine):**
- Agent 10: Validation Suite
- Agent 11: Metrics & Reporting

**Phase 6 (Integration - After Everything):**
- Agent 12: CLI & Integration

### Dependencies

```
Data Structures → [Indicators, Portfolio, Execution, Strategies]
Data Loading → [Backtest Engine]
Indicators → [Strategies]
Portfolio → [Backtest Engine]
Execution → [Backtest Engine]
Strategies → [Backtest Engine]
Backtest Engine → [Validation, Reporting]
Validation → [CLI]
Reporting → [CLI]
Test Fixtures → [All Test Suites]
```

### Estimated Timeline

- **Week 1:** Agents 1, 2, 13 (Foundation)
- **Week 2:** Agents 3, 4, 5 (Core Components)
- **Week 3:** Agents 6, 7, 8 (Strategies)
- **Week 4:** Agent 9 (Backtest Engine)
- **Week 5:** Agents 10, 11 (Analysis)
- **Week 6:** Agent 12 (Integration & Testing)

Each agent can work independently with their assigned prompt and reference documentation.

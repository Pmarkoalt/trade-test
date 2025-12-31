# Performance Characteristics

This document describes expected performance characteristics for the trading system. These benchmarks help identify performance regressions and guide optimization efforts.

## Running Performance Benchmarks

Performance benchmarks are run using `pytest-benchmark`:

```bash
# Run all performance benchmarks
pytest tests/performance/ -m performance --benchmark-only

# Compare against previous run (regression detection)
pytest tests/performance/ -m performance --benchmark-only --benchmark-compare

# Compare against specific baseline
pytest tests/performance/ -m performance --benchmark-only --benchmark-compare=0001

# Save baseline
pytest tests/performance/ -m performance --benchmark-only --benchmark-autosave
```

## Expected Performance Characteristics

### Indicators

Performance benchmarks for indicator calculations on large datasets (10,000 data points):

| Indicator | Expected Time | Notes |
|-----------|--------------|-------|
| Moving Average (MA20) | < 5ms | Vectorized pandas operation |
| ATR (ATR14) | < 10ms | Requires OHLC data processing |
| ROC (ROC60) | < 8ms | Rolling calculation |
| Highest Close (20D) | < 6ms | Rolling max operation |
| ADV (20D) | < 8ms | Dollar volume rolling average |
| Compute Features (full set) | < 50ms | All indicators + feature engineering |

**Data**: 10,000 daily bars per symbol  
**Target**: Sub-50ms for full feature computation per symbol

### Portfolio Operations

Performance benchmarks for portfolio operations:

| Operation | Expected Time | Notes |
|-----------|--------------|-------|
| Update Equity (50 positions) | < 5ms | Price update and P&L calculation |
| Exposure Calculation (50 positions) | < 2ms | Gross exposure computation |
| Process Fill | < 1ms | Single position creation |

**Data**: Portfolio with 50 open positions  
**Target**: Portfolio updates should scale linearly with position count

### Validation Suite

Performance benchmarks for statistical validation:

| Test | Expected Time | Notes |
|------|--------------|-------|
| Bootstrap Test (10,000 R-multiples, 1000 iterations) | < 2s | Resampling-based validation |
| Permutation Test (100 trades, 1000 iterations) | < 3s | Date permutation validation |

**Data**: Large datasets as specified  
**Target**: Statistical tests complete in reasonable time for CI/CD

### Backtest Engine

Performance benchmarks for backtest execution:

| Component | Expected Time | Notes |
|-----------|--------------|-------|
| Event Loop (process_day) | < 100ms | Single day processing with 20 symbols |
| Full Backtest (6 months, 20 symbols) | < 30s | Complete backtest run |

**Data**: 
- 20 symbols
- 6 months of daily data (~130 trading days)
- Full feature computation and signal generation

**Target**: 
- Single day processing: < 100ms
- Full 6-month backtest: < 30s
- Should scale roughly linearly with symbol count and time period

### Data Loading

Performance benchmarks for data loading operations:

| Operation | Expected Time | Notes |
|-----------|--------------|-------|
| CSV Loading (10 symbols, 5 years) | < 500ms | Includes validation and dtype optimization |
| Feature Computation Scaling | Linear | Time should scale roughly linearly with symbol count |

**Data**: 
- 10 symbols
- 5 years of daily data (~1,250 trading days per symbol)

**Target**: 
- Bulk loading: < 500ms for 10 symbols
- Linear scaling with symbol count

### Signal Scoring and Queue

Performance benchmarks for signal processing:

| Operation | Expected Time | Notes |
|-----------|--------------|-------|
| Signal Scoring (100 signals) | < 50ms | Includes breakout, momentum, diversification scoring |
| Queue Selection (100 signals, 10 existing positions) | < 30ms | Ranking and constraint checking |

**Data**: 
- 100 candidate signals
- Portfolio with 10 existing positions
- Full correlation and constraint checking

**Target**: 
- Scoring: < 50ms for 100 signals
- Selection: < 30ms for 100 signals

### Strategy Evaluation

Performance benchmarks for strategy evaluation:

| Operation | Expected Time | Notes |
|-----------|--------------|-------|
| Equity Strategy Evaluation (50 symbols, single date) | < 200ms | Eligibility + entry signal generation |
| Multi-Symbol Scaling | Linear | Time should scale linearly with symbol count |

**Data**: 
- 50 symbols
- 3 years of historical data per symbol
- Single date evaluation

**Target**: 
- 50 symbols evaluated: < 200ms
- Linear scaling with symbol count

### Reporting

Performance benchmarks for report generation:

| Operation | Expected Time | Notes |
|-----------|--------------|-------|
| Metrics Calculation (100 trades, 3 years equity curve) | < 100ms | All performance metrics |
| CSV Export (100 trades) | < 10ms | DataFrame to CSV |

**Data**: 
- 100 closed trades
- 3 years of daily equity curve (~750 data points)

**Target**: 
- Metrics: < 100ms
- CSV export: < 10ms

## Performance Regression Detection

The benchmark suite is configured to detect performance regressions. When running with `--benchmark-compare`, pytest-benchmark will:

1. Compare current run against previous baseline
2. Flag any operations that are significantly slower (>20% degradation)
3. Store new baseline if performance is acceptable

### CI/CD Integration

Performance benchmarks can be run in CI/CD to catch regressions:

```yaml
# Example GitHub Actions step
- name: Run Performance Benchmarks
  run: |
    pytest tests/performance/ -m performance \
      --benchmark-only \
      --benchmark-compare \
      --benchmark-fail=20%
```

The `--benchmark-fail=20%` flag will fail the build if any benchmark is more than 20% slower than the baseline.

## Scaling Characteristics

### Symbol Count Scaling

Most operations should scale roughly linearly with the number of symbols:

- **Feature Computation**: O(n) where n = number of symbols
- **Strategy Evaluation**: O(n) where n = number of symbols
- **Data Loading**: O(n) where n = number of symbols

### Time Period Scaling

Backtest operations should scale roughly linearly with time period:

- **Event Loop**: O(d) where d = number of trading days
- **Full Backtest**: O(d × n) where d = days, n = symbols

### Position Count Scaling

Portfolio operations should scale roughly linearly with position count:

- **Equity Update**: O(p) where p = number of positions
- **Exposure Calculation**: O(p) where p = number of positions

## Memory Usage

Expected memory usage patterns:

- **Data Loading**: ~1-2 MB per symbol per year (with dtype optimization)
- **Feature Computation**: ~5-10 MB per symbol (intermediate calculations)
- **Backtest State**: ~50-100 MB for 20 symbols, 1 year of data

## Optimization Guidelines

1. **Use vectorized operations**: Prefer pandas/numpy vectorized operations over loops
2. **Cache intermediate results**: Use caching for repeated indicator calculations
3. **Optimize data types**: Use appropriate dtypes (float32 vs float64) to reduce memory
4. **Batch operations**: Process multiple symbols in batches when possible
5. **Profile regularly**: Use profiling to identify bottlenecks

## Troubleshooting Performance Issues

If benchmarks are slower than expected:

1. **Check system load**: High CPU/memory usage can affect benchmarks
2. **Verify data size**: Ensure test data matches expected size
3. **Check for regressions**: Compare against previous baseline
4. **Profile bottlenecks**: Use `cProfile` or `py-spy` to identify slow operations
5. **Review recent changes**: Check git history for recent changes that might affect performance

## Baseline Updates

Baselines should be updated when:

1. **Legitimate optimizations**: After performance improvements that make operations faster
2. **Hardware changes**: When running on significantly different hardware
3. **Dependency updates**: When upgrading pandas/numpy that changes performance characteristics
4. **Algorithm changes**: When changing algorithms in ways that legitimately affect performance

Update baseline with:
```bash
pytest tests/performance/ -m performance --benchmark-only --benchmark-autosave
```

## Production Workload Baselines

Production workloads typically involve:
- **50-200 symbols** in the universe
- **3-10 years** of historical data per symbol
- **Multiple strategies** running simultaneously
- **Full validation suite** execution
- **Complete reporting** generation

### Production Baseline Targets

These are the expected performance characteristics for production-scale workloads:

| Workload | Configuration | Expected Time | Notes |
|----------|--------------|---------------|-------|
| **Full Backtest** | 100 symbols, 5 years, single strategy | < 5 minutes | Complete backtest with all features |
| **Walk-Forward Analysis** | 100 symbols, 10 years, 5 splits | < 30 minutes | Train/validation/holdout splits |
| **Validation Suite** | 1000 trades, full bootstrap/permutation | < 5 minutes | All statistical validation tests |
| **Multi-Strategy Backtest** | 100 symbols, 3 years, 3 strategies | < 15 minutes | Parallel strategy evaluation |
| **Feature Computation** | 200 symbols, 5 years | < 2 minutes | Full feature set for all symbols |
| **Report Generation** | 500 trades, 5 years equity curve | < 30 seconds | All metrics and visualizations |

### Establishing Production Baselines

To establish a production baseline for your environment:

1. **Run the production baseline script**:
   ```bash
   python scripts/create_production_baseline.py
   ```

2. **Verify baseline meets targets**: The script will report if any operations exceed expected times

3. **Save the baseline**:
   ```bash
   pytest tests/performance/ -m performance --benchmark-only --benchmark-autosave
   ```

4. **Document your environment**: Note hardware specs, Python version, and dependency versions

### Production Baseline Script

The `scripts/create_production_baseline.py` script runs production-scale benchmarks and compares against expected targets. It:

- Runs benchmarks with production-scale data (100+ symbols, 5+ years)
- Compares results against expected performance targets
- Reports any operations that exceed expected times
- Generates a baseline report for documentation

### Monitoring Production Performance

In production, monitor these key metrics:

1. **Backtest Duration**: Should scale linearly with symbol count and time period
2. **Memory Usage**: Should remain stable and not grow unbounded
3. **Feature Computation Time**: Should be consistent across symbols
4. **Validation Suite Duration**: Should complete within expected timeframes

If performance degrades:
1. Check system resources (CPU, memory, disk I/O)
2. Compare against baseline benchmarks
3. Profile slow operations
4. Review recent code changes


# Best Practices Guide

This guide provides recommendations and best practices for using the trading system effectively.

## Table of Contents

1. [Walk-Forward Testing](#walk-forward-testing)
2. [Parameter Selection](#parameter-selection)
3. [Data Quality](#data-quality)
4. [Configuration Management](#configuration-management)
5. [Validation Workflow](#validation-workflow)
6. [Performance Expectations](#performance-expectations)
7. [Risk Management](#risk-management)
8. [Code Organization](#code-organization)
9. [Common Pitfalls](#common-pitfalls)

---

## Walk-Forward Testing

### The Golden Rule: Lock Your Holdout

**Never** adjust parameters based on holdout results. The holdout period is your final, unbiased test. Once you start backtesting, the holdout dates should be **locked** and never changed.

### Recommended Split Structure

For a 24-month dataset:

```
Train:     15 months  (Months 1-15)   - Parameter optimization
Validation: 3 months  (Months 16-18)  - Out-of-sample testing
Holdout:    6 months  (Months 19-24)  - Final evaluation (LOCKED)
```

For an 18-month minimum:

```
Train:     12 months  (Months 1-12)   - Parameter optimization
Validation: 3 months  (Months 13-15)  - Out-of-sample testing
Holdout:    3 months  (Months 16-18)  - Final evaluation (LOCKED)
```

### Workflow Order

1. **Define splits** - Lock holdout dates before any backtesting
2. **Train period** - Optimize parameters, test ideas
3. **Validation period** - Test optimized parameters out-of-sample
4. **Run validation suite** - Statistical tests and stress scenarios
5. **Holdout evaluation** - Final test with locked parameters

**Never**: Optimize → Validate → **Adjust** → Re-test holdout ❌

**Always**: Optimize → Validate → **Lock** → Test holdout ✅

---

## Parameter Selection

### Frozen vs Tunable Parameters

**FROZEN parameters** (marked in configs) should **never** be changed:
- `risk_per_trade`: 0.75% (frozen)
- `max_positions`: 8 (frozen)
- `max_exposure`: 80% (frozen)
- `max_position_notional`: 15% (frozen)
- `capacity.max_order_pct_adv`: 0.5% equity, 0.25% crypto (frozen)

**Tunable parameters** can be adjusted during train/validation:
- Entry clearances (`fast_clearance`, `slow_clearance`)
- Exit MA periods (`exit_ma`)
- Stop loss multipliers (`hard_stop_atr_mult`)
- Eligibility filters (`ma_slope_min`)
- Volatility scaling mode

### Parameter Optimization Best Practices

1. **Start with defaults** - Example configs provide good starting points
2. **Use sensitivity analysis** - Grid search to find robust parameters
3. **Avoid overfitting** - If parameters are too specific, they won't generalize
4. **Test parameter stability** - Use sensitivity analysis to check for sharp peaks
5. **Document changes** - Keep notes on what you tried and why

### Sensitivity Analysis Tips

```yaml
validation:
  sensitivity:
    enabled: true
    # Test reasonable ranges, not extremes
    equity_atr_mult: [2.0, 2.5, 3.0]           # Good: reasonable range
    equity_breakout_clearance: [0.005, 0.010]   # Good: 2-3 values
    # equity_atr_mult: [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]  # Avoid: too many
```

- **Use 2-4 values per parameter** - More values = exponentially more combinations
- **Test meaningful ranges** - Don't test unrealistic extremes
- **Focus on important parameters** - Entry/exit parameters usually matter most
- **Check for stability** - Look for flat regions, not just peaks

---

## Data Quality

### Data Requirements

1. **Minimum lookback**: At least 250 days for indicator calculations
2. **Data quality**: Clean OHLCV data without gaps or errors
3. **Benchmark data**: Ensure benchmark files (SPY, BTC) are available
4. **Date consistency**: All symbols should have overlapping trading dates

### Pre-Backtest Checks

```python
# Verify data quality before backtesting
import pandas as pd

# Check for missing values
df = pd.read_csv('data/equity/ohlcv/AAPL.csv', parse_dates=['date'])
print(df.isnull().sum())

# Check date range
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# Check for duplicates
print(f"Duplicate dates: {df['date'].duplicated().sum()}")

# Verify OHLC relationships (high >= low, etc.)
assert (df['high'] >= df['low']).all(), "Invalid OHLC data"
assert (df['high'] >= df['close']).all(), "Invalid OHLC data"
assert (df['close'] >= df['low']).all(), "Invalid OHLC data"
```

### Handling Missing Data

The system handles missing data, but best practices:

1. **Fill gaps properly** - Use forward fill for missing days (holidays)
2. **Remove symbols with too many gaps** - If >5% missing, consider excluding
3. **Verify after loading** - Check log files for data quality warnings
4. **Consistent dates** - Ensure all symbols have the same trading calendar

---

## Configuration Management

### Version Control

**Always version control your configs:**

```bash
# Track configs in git
git add my_config.yaml
git add EXAMPLE_CONFIGS/
git commit -m "Add backtest configuration"
```

### Naming Conventions

Use descriptive names for config files:

```
# Good
equity_conservative_config.yaml
crypto_aggressive_config.yaml
run_2024_q1_config.yaml

# Avoid
config1.yaml
test.yaml
final_config.yaml
```

### Config Organization

Organize configs in directories:

```
configs/
├── equity/
│   ├── conservative.yaml
│   ├── aggressive.yaml
│   └── balanced.yaml
├── crypto/
│   └── crypto_config.yaml
└── runs/
    ├── run_2024_q1.yaml
    └── run_2024_q2.yaml
```

### Document Your Configs

Add comments to explain parameter choices:

```yaml
# Equity Momentum Strategy - Conservative Variant
# Date: 2024-01-15
# Rationale: Increased clearance for more selective entries
#            Tighter stops for better risk control

entry:
  fast_clearance: 0.010   # 1.0% (vs 0.5% default) - more selective
  slow_clearance: 0.015   # 1.5% (vs 1.0% default) - larger breakouts required

exit:
  hard_stop_atr_mult: 2.0  # 2.0x (vs 2.5x default) - tighter stops
```

---

## Validation Workflow

### Always Run Validation Before Deploying

The validation suite checks:
- **Bootstrap test**: Statistical significance of returns
- **Permutation test**: Random chance hypothesis
- **Stress tests**: Bear markets, range markets, flash crashes
- **Correlation analysis**: Portfolio diversification

### Validation Checklist

Before considering a strategy production-ready:

- [ ] Bootstrap test passes (p-value < 0.05)
- [ ] Permutation test passes (p-value < 0.05)
- [ ] Stress tests show acceptable degradation
- [ ] Correlation analysis shows good diversification
- [ ] Validation period performance confirms training period
- [ ] Holdout period (if available) shows similar performance

### Interpreting Validation Results

**Bootstrap Test:**
- **Pass**: Strategy returns are statistically significant
- **Fail**: Returns could be due to luck/chance

**Permutation Test:**
- **Pass**: Strategy performs better than random
- **Fail**: Strategy is not better than random trading

**Stress Tests:**
- **Slippage stress**: Check degradation at 2x, 3x slippage
- **Bear market**: Verify strategy handles downtrends
- **Range market**: Check performance in sideways markets
- **Flash crash**: Ensure strategy survives extreme events

**Acceptable degradation:**
- 2x slippage: <30% Sharpe degradation
- 3x slippage: <50% Sharpe degradation
- Bear market: Positive returns not required, but limited losses
- Range market: Can underperform, but should still be viable

---

## Performance Expectations

### Realistic Expectations

**Good performance metrics:**
- Sharpe ratio: 1.0-2.0 (excellent if >1.5)
- Max drawdown: <20% (acceptable), <15% (good), <10% (excellent)
- Win rate: 45-60% (depends on strategy type)
- Calmar ratio: >1.0 (good), >2.0 (excellent)

**Red flags:**
- Sharpe ratio <0.5 (may not be viable)
- Max drawdown >30% (too risky)
- Win rate <35% (check strategy logic)
- Validation performance much worse than training (overfitting)

### Comparing to Benchmarks

Always compare to benchmarks (SPY for equity, BTC for crypto):

```python
# From monthly_report.json
{
  "total_return": 0.15,           # 15% portfolio return
  "benchmark_return": 0.12,       # 12% benchmark return
  "excess_return": 0.03,          # 3% excess return
  "correlation_to_benchmark": 0.65  # 65% correlation
}
```

**Good signs:**
- Positive excess return
- Reasonable correlation (0.4-0.8 for momentum strategies)
- Similar or better Sharpe ratio vs benchmark

**Warning signs:**
- Negative excess return
- Very high correlation (>0.9) - may be overfitting to benchmark
- Much worse Sharpe ratio than benchmark

---

## Risk Management

### Portfolio-Level Risk Controls

The system includes built-in risk controls:

1. **Position sizing**: Risk-based sizing (0.75% per trade)
2. **Max positions**: Limits number of concurrent positions (8 default)
3. **Max exposure**: Limits gross exposure (80% default)
4. **Position limits**: Max 15% per position
5. **Capacity constraints**: Order size vs average dollar volume
6. **Correlation guards**: Prevents over-concentration

### Verify Risk Controls

Check that risk controls are working:

```python
# From equity_curve.csv
import pandas as pd

equity_df = pd.read_csv('equity_curve.csv', parse_dates=['date'])

# Check exposure limits
max_exposure = equity_df['exposure'].max()
assert max_exposure <= 80000, f"Exposure {max_exposure} exceeds 80% limit"

# Check position count
max_positions = equity_df['positions'].max()
assert max_positions <= 8, f"Positions {max_positions} exceed limit"

# Check individual position sizes from trade_log.csv
trades_df = pd.read_csv('trade_log.csv')
# Verify position notional doesn't exceed 15% of equity
```

### Additional Risk Considerations

1. **Sector concentration**: Monitor if trading multiple sectors
2. **Market regime**: Strategy may perform differently in different regimes
3. **Liquidity**: Ensure symbols have sufficient volume
4. **Correlation**: Monitor portfolio correlation (correlation guard helps)

---

## Code Organization

### Project Structure

Keep your workspace organized:

```
my_trading_project/
├── configs/              # Your configuration files
│   ├── equity/
│   ├── crypto/
│   └── runs/
├── data/                 # Your data files
│   ├── equity/
│   ├── crypto/
│   └── benchmarks/
├── results/              # Backtest results (gitignored)
│   └── run_*/
├── scripts/              # Custom analysis scripts
│   └── analyze_results.py
└── notebooks/            # Jupyter notebooks for analysis
    └── strategy_analysis.ipynb
```

### Results Management

**Gitignore results directory:**

```bash
# .gitignore
results/
*.log
```

**Organize results by date/project:**

```bash
results/
├── 2024-01-15_equity_conservative/
├── 2024-01-20_equity_aggressive/
└── 2024-01-25_crypto_momentum/
```

### Analysis Scripts

Create reusable analysis scripts:

```python
# scripts/analyze_backtest.py
"""Standard backtest analysis script."""

import sys
from pathlib import Path

def main(results_dir: str):
    # Your analysis code here
    pass

if __name__ == '__main__':
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'results/latest'
    main(results_dir)
```

---

## Common Pitfalls

### 1. Lookahead Bias

**Pitfall**: Using future information in backtest
**Solution**: System prevents this, but verify:
- Indicators only use data up to current date
- Signals generated at close, executed at next open
- No future price data leakage

### 2. Overfitting

**Pitfall**: Optimizing parameters that only work on training data
**Solution**:
- Use validation period to verify generalization
- Check sensitivity analysis for sharp peaks (overfitting sign)
- If validation << training performance, you've overfit

### 3. Survivorship Bias

**Pitfall**: Only testing symbols that survived (didn't delist)
**Solution**:
- Use universe files that include all symbols at period start
- System handles missing data, but be aware of bias

### 4. Data Snooping

**Pitfall**: Testing many strategies/configs and only reporting best
**Solution**:
- Document all tests you run
- Use holdout period for final evaluation (locked)
- Be honest about multiple testing

### 5. Ignoring Transaction Costs

**Pitfall**: Assuming perfect execution
**Solution**:
- System includes slippage and fees by default
- Verify `slippage_model: "full"` in config
- Run stress tests with higher slippage

### 6. Changing Holdout Period

**Pitfall**: Adjusting holdout dates after seeing results
**Solution**:
- **Lock holdout dates before any backtesting**
- Never change holdout based on results
- Treat holdout as final exam (can't study after)

### 7. Not Running Validation Suite

**Pitfall**: Deploying strategy based only on backtest returns
**Solution**:
- Always run validation suite before deployment
- Check statistical significance (bootstrap/permutation)
- Verify stress test results

### 8. Ignoring Correlation

**Pitfall**: Portfolio full of highly correlated positions
**Solution**:
- Enable correlation guard in config
- Monitor correlation analysis in validation results
- Diversify across sectors/assets

---

## Summary Checklist

Before deploying a strategy, verify:

- [ ] Walk-forward splits properly defined (holdout locked)
- [ ] Parameters optimized on training period only
- [ ] Validation period confirms training results
- [ ] Validation suite passes (bootstrap, permutation, stress tests)
- [ ] Holdout period (if available) shows acceptable performance
- [ ] Data quality verified (no gaps, correct dates)
- [ ] Configs version controlled and documented
- [ ] Risk controls verified (exposure, positions, correlation)
- [ ] Performance metrics meet targets (Sharpe, drawdown, etc.)
- [ ] Results compared to benchmarks
- [ ] No overfitting (validation performance similar to training)
- [ ] Transaction costs included (slippage, fees)
- [ ] Documentation complete (parameters, rationale, results)

---

## Next Steps

- Review [Getting Started Guide](getting_started.md) for basics
- See [Examples Guide](examples.md) for practical examples
- Explore example configs in `EXAMPLE_CONFIGS/`
- Review technical documentation in `agent-files/`

**Remember**: Systematic trading is about process, not perfection. Follow the workflow, document your decisions, and let the validation suite guide you.

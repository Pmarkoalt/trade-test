# Examples and Common Use Cases

This guide provides practical examples for common workflows and use cases.

## Table of Contents

1. [Basic Backtest Workflow](#basic-backtest-workflow)
2. [Multi-Period Backtesting](#multi-period-backtesting)
3. [Parameter Sensitivity Analysis](#parameter-sensitivity-analysis)
4. [Strategy Customization](#strategy-customization)
5. [Working with Different Asset Classes](#working-with-different-asset-classes)
6. [Custom Strategy Types](#custom-strategy-types)
7. [Programmatic Usage](#programmatic-usage)
8. [Comparing Multiple Strategies](#comparing-multiple-strategies)
9. [Exporting and Analyzing Results](#exporting-and-analyzing-results)

---

## Basic Backtest Workflow

### Example 1: Simple Equity Backtest

Run a backtest on the training period with default equity strategy:

```bash
python -m trading_system backtest \
    --config EXAMPLE_CONFIGS/run_config.yaml \
    --period train
```

**What this does:**
- Loads equity data from configured path
- Applies equity momentum strategy
- Runs backtest on training period (2023-01-01 to 2024-03-31 by default)
- Generates results in `results/{run_id}/train/`

### Example 2: Crypto-Only Backtest

Modify `run_config.yaml` to enable only crypto:

```yaml
strategies:
  equity:
    enabled: false  # Disable equity
  crypto:
    config_path: "EXAMPLE_CONFIGS/crypto_config.yaml"
    enabled: true   # Enable crypto
```

Then run:

```bash
python -m trading_system backtest -c my_config.yaml -p train
```

---

## Multi-Period Backtesting

### Example 3: Full Walk-Forward Workflow

Run complete walk-forward analysis:

```bash
# Step 1: Training period (parameter optimization)
python -m trading_system backtest -c my_config.yaml -p train

# Step 2: Validation period (out-of-sample test)
python -m trading_system backtest -c my_config.yaml -p validation

# Step 3: Run validation suite (statistical tests)
python -m trading_system validate -c my_config.yaml

# Step 4: Holdout evaluation (final test - parameters LOCKED)
python -m trading_system holdout -c my_config.yaml
```

**Workflow Notes:**
- **Train**: Optimize parameters, test ideas
- **Validation**: Test optimized parameters out-of-sample
- **Validate**: Run statistical tests and stress scenarios
- **Holdout**: Final evaluation with locked parameters (never adjust based on holdout)

### Example 4: Running All Periods in Sequence

Create a simple script to run all periods:

```bash
#!/bin/bash
# run_all_periods.sh

CONFIG="my_config.yaml"

echo "Running training period..."
python -m trading_system backtest -c $CONFIG -p train

echo "Running validation period..."
python -m trading_system backtest -c $CONFIG -p validation

echo "Running validation suite..."
python -m trading_system validate -c $CONFIG

echo "Running holdout evaluation..."
python -m trading_system holdout -c $CONFIG

echo "Done!"
```

Make it executable and run:

```bash
chmod +x run_all_periods.sh
./run_all_periods.sh
```

---

## Parameter Sensitivity Analysis

### Example 5: Finding Optimal Parameters

Run sensitivity analysis to find best parameter values:

```bash
python -m trading_system sensitivity \
    --config my_config.yaml \
    --metric sharpe_ratio \
    --period train
```

**What this does:**
- Tests different parameter combinations
- Computes metric (e.g., Sharpe ratio) for each combination
- Generates heatmaps showing parameter sensitivity
- Saves results to `results/{run_id}/sensitivity/`

### Example 6: Custom Sensitivity Grid

Configure sensitivity parameters in `run_config.yaml`:

```yaml
validation:
  sensitivity:
    enabled: true
    equity_atr_mult: [2.0, 2.5, 3.0]           # Test 3 stop loss values
    equity_breakout_clearance: [0.005, 0.010]   # Test 2 clearance values
    equity_exit_ma: [20, 50]                    # Test 2 exit MA values
    vol_scaling_mode: ["continuous", "off"]     # Test volatility scaling
```

This creates a grid of 3 × 2 × 2 × 2 = 24 combinations to test.

---

## Strategy Customization

### Example 7: Modifying Equity Strategy Parameters

Edit `equity_config.yaml` to adjust strategy behavior:

```yaml
# Make strategy more aggressive
entry:
  fast_clearance: 0.003   # Lower clearance (easier entries)
  slow_clearance: 0.008   # Lower clearance

# Make exits faster
exit:
  exit_ma: 20             # Use MA20 instead of MA50

# Adjust risk
risk:
  risk_per_trade: 0.010   # Increase to 1.0% (default is 0.75%)
  max_positions: 10       # Allow more positions (default is 8)
```

**Important**: Only adjust parameters marked as tunable (not FROZEN) during train/validation phases.

### Example 8: Creating a Conservative Strategy

Modify for more conservative trading:

```yaml
# More stringent eligibility
eligibility:
  ma_slope_min: 0.010      # Require 1.0% MA slope (vs 0.5%)
  require_close_above_trend_ma: true

# Larger breakouts required
entry:
  fast_clearance: 0.010    # 1.0% clearance (vs 0.5%)
  slow_clearance: 0.015    # 1.5% clearance (vs 1.0%)

# Tighter stops
exit:
  hard_stop_atr_mult: 2.0  # 2.0x ATR (vs 2.5x)

# Lower risk per trade
risk:
  risk_per_trade: 0.005    # 0.5% risk per trade (vs 0.75%)
```

---

## Working with Different Asset Classes

### Example 9: Equity-Only Configuration

Disable crypto and focus on equities:

```yaml
strategies:
  equity:
    config_path: "EXAMPLE_CONFIGS/equity_config.yaml"
    enabled: true
  crypto:
    enabled: false  # Disable crypto
```

Use NASDAQ-100 or S&P 500 universe in `equity_config.yaml`:

```yaml
universe: "NASDAQ-100"  # or "SP500"
benchmark: "SPY"
```

### Example 10: Crypto-Only Configuration

Focus on cryptocurrency trading:

```yaml
strategies:
  equity:
    enabled: false
  crypto:
    config_path: "EXAMPLE_CONFIGS/crypto_config.yaml"
    enabled: true
```

Crypto strategy uses a fixed 10-asset universe (BTC, ETH, SOL, etc.).

### Example 11: Mixed Portfolio

Run both equity and crypto strategies simultaneously:

```yaml
strategies:
  equity:
    config_path: "EXAMPLE_CONFIGS/equity_config.yaml"
    enabled: true
  crypto:
    config_path: "EXAMPLE_CONFIGS/crypto_config.yaml"
    enabled: true
```

The portfolio manager handles positions from both strategies with correlation guards.

---

## Strategy Customization

### Creating Custom Strategies

You can create your own trading strategies by implementing the `StrategyInterface` class.

#### Step 1: Generate Strategy Template

Use the CLI to generate a strategy template:

```bash
# Interactive wizard
python -m trading_system strategy create

# Or specify options directly
python -m trading_system strategy create \
    --name my_custom_strategy \
    --type custom \
    --asset-class equity \
    --output trading_system/strategies/custom/my_custom_strategy_equity.py
```

This creates a template with all required methods that you need to implement.

#### Step 2: Implement Required Methods

All strategies must implement these methods:

1. **`check_eligibility(features: FeatureRow) -> Tuple[bool, List[str]]`**
   - Check if symbol is eligible for entry
   - Returns: (is_eligible, list_of_failure_reasons)

2. **`check_entry_triggers(features: FeatureRow) -> Tuple[Optional[BreakoutType], float]`**
   - Check if entry triggers are met
   - Returns: (breakout_type, clearance) or (None, 0.0)

3. **`check_exit_signals(position: Position, features: FeatureRow) -> Optional[ExitReason]`**
   - Check if exit signals are met
   - Returns: ExitReason or None

4. **`compute_stop_price(entry_price, entry_date, features) -> Optional[float]`**
   - Compute stop loss price for new position
   - Returns: stop_price or None

#### Step 3: Register Your Strategy

```python
from trading_system.strategies.strategy_registry import register_strategy
from trading_system.strategies.custom.my_custom_strategy_equity import MyCustomStrategyEquity

register_strategy(
    strategy_type="my_custom_strategy",
    asset_class="equity",
    strategy_class=MyCustomStrategyEquity
)
```

#### Step 4: Create Strategy Config

Create a YAML config file for your strategy:

```yaml
name: "my_custom_strategy_equity"
asset_class: "equity"
universe: ["AAPL", "MSFT", "GOOGL"]
benchmark: "SPY"

indicators:
  ma_periods: [20, 50]
  atr_period: 14
  adv_lookback: 20

entry:
  fast_clearance: 0.005  # Your custom entry parameters

exit:
  mode: "ma_cross"
  exit_ma: 20
  hard_stop_atr_mult: 2.0

risk:
  risk_per_trade: 0.0075
  max_positions: 8
```

#### Step 5: Use in Backtest

Reference your strategy config in `run_config.yaml`:

```yaml
strategies:
  equity:
    config_path: "configs/my_custom_strategy_config.yaml"
    enabled: true
```

Then run the backtest normally.

#### Example: Simple MA Crossover Strategy

See `examples/custom_strategy.py` for a complete example of a simple moving average crossover strategy.

---

## Custom Strategy Types

### Example 12: Mean Reversion Strategy

Use the mean reversion strategy config:

```yaml
strategies:
  equity:
    config_path: "EXAMPLE_CONFIGS/mean_reversion_config.yaml"
    enabled: true
  crypto:
    enabled: false
```

Mean reversion strategy:
- Enters on oversold conditions (Z-score < -2.0)
- Exits when reverted to mean (Z-score >= 0.0)
- Best for liquid ETFs (SPY, QQQ, etc.)

### Example 13: Pairs Trading Strategy

Configure pairs trading:

```yaml
strategies:
  equity:
    config_path: "EXAMPLE_CONFIGS/pairs_config.yaml"
    enabled: true
```

Pairs config defines specific pairs (e.g., XLE/XLK, GLD/TLT) and trades spread divergences.

### Example 14: Multi-Timeframe Strategy

Use multi-timeframe approach:

```yaml
strategies:
  equity:
    config_path: "EXAMPLE_CONFIGS/multi_timeframe_config.yaml"
    enabled: true
```

This strategy uses:
- Higher timeframe (daily) for trend filter (MA50)
- Lower timeframe (weekly) for entry signals (breakout)
- Best for trending equities

### Example 15: Factor-Based Strategy

Run factor strategy:

```yaml
strategies:
  equity:
    config_path: "EXAMPLE_CONFIGS/factor_config.yaml"
    enabled: true
```

Factor strategy combines:
- Momentum (40% weight)
- Value (30% weight)
- Quality (30% weight)
- Rebalances monthly/quarterly

---

## Programmatic Usage

### Example 16: Basic Programmatic Backtest

```python
from trading_system.integration.runner import run_backtest
from trading_system.configs.run_config import RunConfig

# Run backtest
results = run_backtest("my_config.yaml", period="train")

# Access key metrics
print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
print(f"Total Trades: {results['total_trades']}")
print(f"Win Rate: {results['win_rate']:.2%}")
```

### Example 17: Advanced Programmatic Usage

```python
from trading_system.integration.runner import BacktestRunner
from trading_system.configs.run_config import RunConfig
import pandas as pd

# Load configuration
config = RunConfig.from_yaml("my_config.yaml")

# Create runner
runner = BacktestRunner(config)
runner.initialize()

# Run backtest
results = runner.run_backtest(period="train")

# Access detailed data
portfolio = runner.engine.portfolio
closed_trades = runner.engine.closed_trades
daily_events = runner.engine.daily_events

# Extract equity curve
equity_curve = [event['portfolio_state']['equity'] for event in daily_events]
dates = [event['date'] for event in daily_events]

# Create DataFrame
equity_df = pd.DataFrame({
    'date': dates,
    'equity': equity_curve
})
equity_df.set_index('date', inplace=True)

# Plot equity curve (if matplotlib available)
try:
    import matplotlib.pyplot as plt
    equity_df.plot(title="Equity Curve")
    plt.ylabel("Portfolio Value ($)")
    plt.show()
except ImportError:
    print("matplotlib not available for plotting")
```

### Example 18: Running Validation Programmatically

```python
from trading_system.integration.runner import run_validation

# Run validation suite
validation_results = run_validation("my_config.yaml")

# Check results
if validation_results['status'] == 'passed':
    print("✅ Validation PASSED")
else:
    print("❌ Validation FAILED")
    print(f"Rejections: {validation_results['rejections']}")

# Access detailed results
bootstrap = validation_results['results']['bootstrap']
permutation = validation_results['results']['permutation']
stress_tests = validation_results['results']['stress_tests']

print(f"Bootstrap p-value: {bootstrap.get('p_value', 'N/A')}")
print(f"Permutation p-value: {permutation.get('p_value', 'N/A')}")
```

---

## Comparing Multiple Strategies

### Example 19: A/B Testing Strategies

Run backtests with different configs and compare:

```bash
# Run Strategy A
python -m trading_system backtest -c config_strategy_a.yaml -p train
# Results saved to results/{run_id_a}/train/

# Run Strategy B
python -m trading_system backtest -c config_strategy_b.yaml -p train
# Results saved to results/{run_id_b}/train/

# Compare metrics from monthly_report.json files
```

Then compare the `monthly_report.json` files:

```python
import json

# Load results
with open('results/run_a/train/monthly_report.json') as f:
    results_a = json.load(f)

with open('results/run_b/train/monthly_report.json') as f:
    results_b = json.load(f)

# Compare key metrics
print("Strategy A vs Strategy B:")
print(f"Sharpe Ratio: {results_a['sharpe_ratio']:.2f} vs {results_b['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results_a['max_drawdown']:.2%} vs {results_b['max_drawdown']:.2%}")
print(f"Total Return: {results_a['total_return']:.2%} vs {results_b['total_return']:.2%}")
```

### Example 20: Parameter Comparison

Test different parameter sets:

```bash
# Test aggressive parameters
python -m trading_system backtest -c config_aggressive.yaml -p train

# Test conservative parameters
python -m trading_system backtest -c config_conservative.yaml -p train

# Compare results
```

---

## Exporting and Analyzing Results

### Example 21: Loading Results into Pandas

```python
import pandas as pd
import json

# Load equity curve
equity_df = pd.read_csv('results/run_20231219_120000/train/equity_curve.csv',
                        parse_dates=['date'], index_col='date')

# Load trade log
trades_df = pd.read_csv('results/run_20231219_120000/train/trade_log.csv',
                        parse_dates=['entry_date', 'exit_date'])

# Load monthly report
with open('results/run_20231219_120000/train/monthly_report.json') as f:
    metrics = json.load(f)

# Analyze trades
print(f"Total trades: {len(trades_df)}")
print(f"Average R-multiple: {trades_df['r_multiple'].mean():.2f}")
print(f"Win rate: {(trades_df['realized_pnl'] > 0).mean():.2%}")

# Analyze equity curve
print(f"Peak equity: ${equity_df['equity'].max():,.2f}")
print(f"Final equity: ${equity_df['equity'].iloc[-1]:,.2f}")
print(f"Max drawdown: {metrics['max_drawdown']:.2%}")
```

### Example 22: Creating Custom Analysis Script

```python
import pandas as pd
import json
from pathlib import Path

def analyze_backtest_results(results_dir: str):
    """Analyze backtest results and print summary."""
    results_path = Path(results_dir)

    # Load data
    equity_df = pd.read_csv(results_path / 'equity_curve.csv',
                           parse_dates=['date'], index_col='date')
    trades_df = pd.read_csv(results_path / 'trade_log.csv',
                           parse_dates=['entry_date', 'exit_date'])

    with open(results_path / 'monthly_report.json') as f:
        metrics = json.load(f)

    # Compute additional metrics
    equity_df['daily_return'] = equity_df['equity'].pct_change()
    equity_df['cumulative_return'] = (equity_df['equity'] / equity_df['equity'].iloc[0]) - 1

    # Drawdown analysis
    equity_df['running_max'] = equity_df['equity'].expanding().max()
    equity_df['drawdown'] = (equity_df['equity'] - equity_df['running_max']) / equity_df['running_max']

    # Print summary
    print("=" * 60)
    print("BACKTEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nPerformance Metrics:")
    print(f"  Total Return: {metrics['total_return']:.2%}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Calmar Ratio: {metrics['calmar_ratio']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")

    print(f"\nTrade Statistics:")
    print(f"  Total Trades: {len(trades_df)}")
    print(f"  Winning Trades: {metrics['winning_trades']}")
    print(f"  Losing Trades: {metrics['losing_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.2%}")
    print(f"  Avg R-Multiple: {metrics['avg_r_multiple']:.2f}")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")

    print(f"\nPortfolio Stats:")
    print(f"  Starting Equity: ${equity_df['equity'].iloc[0]:,.2f}")
    print(f"  Ending Equity: ${equity_df['equity'].iloc[-1]:,.2f}")
    print(f"  Peak Equity: ${equity_df['equity'].max():,.2f}")
    print(f"  Max Drawdown Date: {equity_df['drawdown'].idxmin()}")

    return {
        'equity_df': equity_df,
        'trades_df': trades_df,
        'metrics': metrics
    }

# Usage
results = analyze_backtest_results('results/run_20231219_120000/train')
```

### Example 23: Batch Processing Multiple Runs

```python
from pathlib import Path
import json
import pandas as pd

def compare_multiple_runs(results_base_dir: str):
    """Compare multiple backtest runs."""
    base_path = Path(results_base_dir)

    all_metrics = []

    # Find all run directories
    for run_dir in base_path.iterdir():
        if run_dir.is_dir() and run_dir.name.startswith('run_'):
            train_path = run_dir / 'train'
            if (train_path / 'monthly_report.json').exists():
                with open(train_path / 'monthly_report.json') as f:
                    metrics = json.load(f)
                    metrics['run_id'] = run_dir.name
                    all_metrics.append(metrics)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(all_metrics)

    # Sort by Sharpe ratio
    comparison_df = comparison_df.sort_values('sharpe_ratio', ascending=False)

    print("Strategy Comparison (sorted by Sharpe Ratio):")
    print(comparison_df[['run_id', 'sharpe_ratio', 'total_return',
                         'max_drawdown', 'win_rate', 'total_trades']].to_string())

    return comparison_df

# Usage
comparison = compare_multiple_runs('results')
```

---

## Next Steps

- See [Getting Started Guide](getting_started.md) for installation and basics
- See [Best Practices Guide](best_practices.md) for recommendations
- Explore example configs in `EXAMPLE_CONFIGS/`
- Review technical documentation in `agent-files/`


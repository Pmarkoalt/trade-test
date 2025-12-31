# Example Configuration Files

This directory contains example YAML configuration files for the trading system.

## Files

### `equity_config.yaml`
Complete configuration for the equity momentum strategy (NASDAQ-100 / S&P 500).

**Key parameters:**
- Trend filter: MA50 with 0.5% slope over 20 days
- Entry: 20D (0.5% clearance) OR 55D (1.0% clearance) breakout
- Exit: MA20 cross (or MA50, testable)
- Stop: 2.5x ATR14
- Capacity: 0.5% of ADV20

### `crypto_config.yaml`
Complete configuration for the crypto momentum strategy (fixed 10-asset universe).

**Key parameters:**
- Trend filter: Close > MA200 (strict)
- Entry: Same breakout triggers as equity
- Exit: Staged (MA20 warning → tighten stop → MA50 exit)
- Stop: 3.0x ATR14 (tighten to 2.0x after MA20 break)
- Capacity: 0.25% of ADV20 (stricter than equity)

### `run_config.yaml`
Complete backtest run configuration used by CLI commands.

**Sections:**
- `dataset`: Data paths and date ranges
- `splits`: Train/validation/holdout date ranges
- `strategies`: Which strategy configs to load
- `portfolio`: Starting equity and risk parameters
- `volatility_scaling`: Portfolio-level risk scaling
- `correlation_guard`: Diversification constraints
- `scoring`: Position queue ranking weights
- `execution`: Timing and slippage model
- `output`: Output file paths and logging
- `validation`: Sensitivity grid and stress test configs
- `metrics`: Success/rejection criteria

## Usage

### Load strategy config:
```python
from trading_system.configs import StrategyConfig

equity_config = StrategyConfig.from_yaml("EXAMPLE_CONFIGS/equity_config.yaml")
crypto_config = StrategyConfig.from_yaml("EXAMPLE_CONFIGS/crypto_config.yaml")
```

### Load run config:
```python
from trading_system.configs import RunConfig

run_config = RunConfig.from_yaml("EXAMPLE_CONFIGS/run_config.yaml")
```

### CLI usage:
```bash
# Run backtest with config
python -m trading_system backtest --config EXAMPLE_CONFIGS/run_config.yaml

# Run validation suite
python -m trading_system validate --config EXAMPLE_CONFIGS/run_config.yaml

# Run holdout evaluation
python -m trading_system holdout --config EXAMPLE_CONFIGS/run_config.yaml
```

## Customization

1. **Copy example files** to your project directory
2. **Modify parameters** as needed (respect frozen parameters)
3. **Update data paths** to match your file structure
4. **Adjust date ranges** for your dataset

## Notes

- **Frozen parameters** (marked FROZEN in comments) should NOT be changed
- **Tunable parameters** can be adjusted during train/validation phase only
- **Holdout dates** must be LOCKED before any backtesting begins
- All paths are relative to project root (or use absolute paths)


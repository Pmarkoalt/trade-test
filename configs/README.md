# Production Configuration Files

This directory contains production-ready configuration files for the trading system.

## Files

### `production_run_config.yaml`
Production run configuration for backtesting and validation.

**Key differences from example config:**
- Extended date ranges for production data
- Production output paths (`results/production/`)
- Production-ready logging settings
- References production strategy configs

**Usage:**
```bash
# Validate production config
python -m trading_system config validate --config configs/production_run_config.yaml

# Run backtest with production config
python -m trading_system backtest --config configs/production_run_config.yaml

# Run validation suite
python -m trading_system validate --config configs/production_run_config.yaml

# Run holdout evaluation
python -m trading_system holdout --config configs/production_run_config.yaml
```

### `production_equity_config.yaml`
Production equity momentum strategy configuration.

**Features:**
- NASDAQ-100 or S&P 500 universe
- All frozen parameters properly set
- Production-ready cost and slippage settings
- ML integration ready (disabled by default)

### `production_crypto_config.yaml`
Production crypto momentum strategy configuration.

**Features:**
- Fixed 10-asset universe (BTC, ETH, BNB, XRP, ADA, SOL, DOT, MATIC, LTC, LINK)
- Stricter capacity constraints than equity
- Weekend penalty enabled
- Higher slippage costs for crypto

## Configuration

### Before Using Production Configs

1. **Update date ranges** in `production_run_config.yaml`:
   - `dataset.start_date` and `dataset.end_date` - Match your data availability
   - `splits.train_start` through `splits.holdout_end` - Set appropriate train/validation/holdout periods
   - **IMPORTANT**: Holdout dates must be LOCKED before any backtesting begins

2. **Update data paths** if different from defaults:
   - `dataset.equity_path`
   - `dataset.crypto_path`
   - `dataset.benchmark_path`

3. **Adjust portfolio settings**:
   - `portfolio.starting_equity` - Set to your production capital

4. **Review output paths**:
   - `output.base_path` - Default is `results/production/`

5. **Review logging settings**:
   - `output.log_level` - INFO recommended for production
   - `output.log_json_format` - Set to true for structured logging
   - `output.log_use_rich` - Set to false if running in non-interactive environment

## Validation

Before using production configs, validate them:

```bash
# Validate run config
python -m trading_system config validate --config configs/production_run_config.yaml

# Validate strategy configs (if validating individually)
python -m trading_system config validate --config configs/production_equity_config.yaml --type strategy
python -m trading_system config validate --config configs/production_crypto_config.yaml --type strategy
```

## Notes

- **Frozen parameters** (marked FROZEN in comments) should NOT be changed
- **Tunable parameters** can be adjusted during train/validation phase only
- **Holdout dates** must be LOCKED before any backtesting begins
- All paths are relative to project root (or use absolute paths)
- Production configs are based on validated example configs structure

## Related Documentation

- Example configs: `EXAMPLE_CONFIGS/README.md`
- Configuration guide: `agent-files/02_CONFIGS_AND_PARAMETERS.md`
- Migration guide: `MIGRATION_GUIDE.md`
- FAQ: `FAQ.md` (configuration section)


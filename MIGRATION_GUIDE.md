# Migration Guide

**Last Updated**: 2024-12-19
**Current System Version**: 0.0.2

This guide helps you migrate between different versions of the Trading System and update your configuration files when breaking changes occur.

---

## Table of Contents

1. [Overview](#overview)
2. [Version History](#version-history)
3. [Config Migration Guide](#config-migration-guide)
4. [System Version Migration](#system-version-migration)
5. [Breaking Changes](#breaking-changes)
6. [Migration Utilities](#migration-utilities)
7. [Troubleshooting](#troubleshooting)

---

## Overview

### What This Guide Covers

- **Config Migration**: How to update your YAML configuration files when the config schema changes
- **Version Migration**: How to upgrade the Trading System itself between versions
- **Breaking Changes**: Documentation of changes that require manual intervention
- **Migration Tools**: Scripts and utilities to help automate migrations

### When to Use This Guide

- Upgrading from one system version to another
- Updating configuration files after schema changes
- Troubleshooting issues after an upgrade
- Understanding what changed between versions

---

## Version History

### Version 0.0.2 (Current)
**Release Date**: 2024-12-19
**Status**: Initial Release

**Features**:
- Core backtest engine with event loop
- Equity and crypto momentum strategies
- Data pipeline with multiple sources (CSV, database, Parquet, HDF5)
- Execution simulation (slippage, fees, fills)
- Portfolio management with risk controls
- Validation suite (bootstrap, permutation, stress tests)
- CLI interface
- Reporting (CSV, JSON, visualization)
- Multiple strategy types (momentum, mean reversion, pairs, multi-timeframe, factor)

**Config Schema Version**: 1.0

---

## Config Migration Guide

### Config Schema Versions

The Trading System uses semantic versioning for configuration schemas:
- **Major version** (X.0.0): Breaking changes requiring manual migration
- **Minor version** (0.X.0): New optional fields, backward compatible
- **Patch version** (0.0.X): Bug fixes, no schema changes

### Current Config Schema: 1.0

#### Run Config Structure (`run_config.yaml`)

```yaml
# Version 1.0 structure
dataset:
  equity_path: "data/equity/ohlcv/"
  crypto_path: "data/crypto/ohlcv/"
  benchmark_path: "data/benchmarks/"
  format: "csv"  # Options: "csv", "parquet", "database"
  start_date: "2023-01-01"
  end_date: "2024-12-31"
  min_lookback_days: 250

splits:
  train_start: "2023-01-01"
  train_end: "2024-03-31"
  validation_start: "2024-04-01"
  validation_end: "2024-06-30"
  holdout_start: "2024-07-01"
  holdout_end: "2024-12-31"

strategies:
  equity:
    config_path: "configs/equity_config.yaml"
    enabled: true
  crypto:
    config_path: "configs/crypto_config.yaml"
    enabled: true

portfolio:
  starting_equity: 100000

volatility_scaling:
  enabled: true
  mode: "continuous"  # Options: "continuous", "regime", "off"
  lookback: 20
  baseline_lookback: 252
  min_multiplier: 0.33
  max_multiplier: 1.0

correlation_guard:
  enabled: true
  min_positions: 4
```

#### Strategy Config Structure (`equity_config.yaml`, `crypto_config.yaml`)

```yaml
# Version 1.0 structure
name: "equity_momentum"
asset_class: "equity"  # or "crypto"
universe: "NASDAQ-100"  # or list: ["AAPL", "MSFT", ...]
benchmark: "SPY"  # or "BTC" for crypto

indicators:
  ma_periods: [20, 50, 200]
  atr_period: 14
  roc_period: 60
  breakout_fast: 20
  breakout_slow: 55
  adv_lookback: 20
  corr_lookback: 20

eligibility:
  trend_ma: 50
  ma_slope_lookback: 20
  ma_slope_min: 0.005
  require_close_above_trend_ma: true
  require_close_above_ma200: false
  relative_strength_enabled: false
  relative_strength_min: 0.0

entry:
  fast_clearance: 0.005
  slow_clearance: 0.010

exit:
  mode: "ma_cross"  # or "staged" for crypto
  exit_ma: 20
  hard_stop_atr_mult: 2.5
  tightened_stop_atr_mult: null  # Required if mode="staged"

risk:
  risk_per_trade: 0.0075
  max_positions: 8
  max_exposure: 0.80
  max_position_notional: 0.15

capacity:
  max_order_pct_adv: 0.005

costs:
  fee_bps: 1
  slippage_base_bps: 8
  slippage_std_mult: 0.75
  weekend_penalty: 1.0
  stress_threshold: -0.03
  stress_slippage_mult: 2.0

# Optional: ML configuration (v1.0+)
ml:
  enabled: false
  model_path: null
  prediction_mode: "score_enhancement"  # Options: "score_enhancement", "filter", "replace"
  ml_weight: 0.3
  confidence_threshold: 0.5

# Optional: Crypto universe configuration (v1.0+)
universe_config:
  mode: "fixed"  # Options: "fixed", "custom", "dynamic"
  symbols: ["BTC", "ETH", ...]
  # ... other dynamic universe options
```

### Migration Steps

#### Step 1: Backup Your Configs

Before migrating, always backup your existing configuration files:

```bash
# Create backup directory
mkdir -p config_backups/$(date +%Y%m%d)

# Copy all config files
cp EXAMPLE_CONFIGS/*.yaml config_backups/$(date +%Y%m%d)/
cp tests/fixtures/configs/*.yaml config_backups/$(date +%Y%m%d)/ 2>/dev/null || true
```

#### Step 2: Validate Current Configs

Test that your current configs are valid before migration:

```bash
# Validate run config
python -c "from trading_system.configs.run_config import RunConfig; RunConfig.from_yaml('path/to/run_config.yaml')"

# Validate strategy config
python -c "from trading_system.configs.strategy_config import StrategyConfig; StrategyConfig.from_yaml('path/to/strategy_config.yaml')"
```

#### Step 3: Apply Migration

Follow the specific migration instructions for your target version (see [Breaking Changes](#breaking-changes) section).

#### Step 4: Verify Migration

After migration, verify your configs work:

```bash
# Run a quick validation test
python -m trading_system backtest --config path/to/run_config.yaml --period train --dry-run 2>&1 | head -20
```

---

## System Version Migration

### Upgrading from Previous Versions

#### From Version 0.0.1 to 0.0.2

**Note**: This is the initial release, so there are no previous versions to migrate from. This section is provided as a template for future migrations.

**Steps**:

1. **Check Current Version**:
   ```bash
   python -c "import trading_system; print(trading_system.__version__)"  # If version info exists
   # Or check pyproject.toml
   grep "version" pyproject.toml
   ```

2. **Update Dependencies**:
   ```bash
   # Update to latest version
   pip install --upgrade -r requirements.txt

   # Or if using editable install
   pip install -e . --upgrade
   ```

3. **Update Optional Dependencies** (if needed):
   ```bash
   # For database support
   pip install -e ".[database]"

   # For ML features
   pip install -e ".[ml]"

   # For all optional features
   pip install -e ".[all]"
   ```

4. **Verify Installation**:
   ```bash
   # Run quick test
   ./quick_test.sh

   # Or manually
   pytest tests/test_data_loading.py -v
   ```

5. **Update Configs** (if schema changed):
   - See [Config Migration Guide](#config-migration-guide) above
   - Check [Breaking Changes](#breaking-changes) section

6. **Test Your Workflows**:
   ```bash
   # Run integration test
   pytest tests/integration/test_end_to_end.py -v

   # Run a small backtest
   python -m trading_system backtest \
     --config tests/fixtures/configs/run_test_config.yaml \
     --period train
   ```

### Python Version Requirements

- **Version 0.0.2**: Requires Python 3.9+ (3.11+ recommended)
- Check your Python version:
  ```bash
  python --version
  ```

If upgrading Python, ensure all dependencies are reinstalled:
```bash
pip install --upgrade -r requirements.txt
```

---

## Breaking Changes

### Version 0.0.2

**Status**: Initial release - no breaking changes from previous versions.

**Note**: Future versions will document breaking changes here.

#### Template for Future Breaking Changes

When breaking changes occur in future versions, they will be documented here with:

1. **Change Description**: What changed and why
2. **Impact**: What this affects (configs, code, data, etc.)
3. **Migration Steps**: Step-by-step instructions
4. **Examples**: Before/after examples
5. **Deprecation Timeline**: When old behavior will be removed

**Example Format** (for future reference):

```markdown
### Version X.Y.Z Breaking Changes

#### Change: Config Field Renamed

**Description**: The `risk.risk_per_trade` field has been renamed to `risk.risk_per_trade_pct` for clarity.

**Impact**: All strategy config files need to be updated.

**Migration Steps**:
1. Find all occurrences of `risk_per_trade` in your configs
2. Replace with `risk_per_trade_pct`
3. Verify configs load correctly

**Before**:
```yaml
risk:
  risk_per_trade: 0.0075
```

**After**:
```yaml
risk:
  risk_per_trade_pct: 0.0075
```

**Deprecation**: Old field name will be removed in version X.Y+2.0
```

---

## Migration Utilities

### Config Validator Script

A utility script to validate and migrate configs:

```python
# scripts/validate_config.py (example)
"""
Config validation and migration utility.

Usage:
    python scripts/validate_config.py --config path/to/config.yaml
    python scripts/validate_config.py --migrate --from-version 1.0 --to-version 2.0 --config path/to/config.yaml
"""
```

### Manual Migration Checklist

Use this checklist when migrating:

- [ ] Backup all configuration files
- [ ] Check current system version
- [ ] Review breaking changes for target version
- [ ] Update system dependencies
- [ ] Migrate configuration files
- [ ] Validate migrated configs
- [ ] Run test suite
- [ ] Run integration tests
- [ ] Test with your actual data/configs
- [ ] Update documentation references

### Automated Migration Script

For future versions, an automated migration script may be provided:

```bash
# Example usage (when available)
python scripts/migrate_config.py \
  --from-version 1.0 \
  --to-version 2.0 \
  --config path/to/config.yaml \
  --output path/to/migrated_config.yaml
```

---

## Troubleshooting

### Common Migration Issues

#### Issue: Config Validation Errors After Migration

**Symptoms**:
```
pydantic.ValidationError: ...
```

**Solutions**:
1. Check the error message for the specific field causing issues
2. Review the [Config Migration Guide](#config-migration-guide) for the correct schema
3. Compare your config with examples in `EXAMPLE_CONFIGS/`
4. Use the config validator:
   ```bash
   python -c "from trading_system.configs.run_config import RunConfig; RunConfig.from_yaml('your_config.yaml')"
   ```

#### Issue: Missing Required Fields

**Symptoms**:
```
Field required: ...
```

**Solutions**:
1. Check the breaking changes section for new required fields
2. Review example configs in `EXAMPLE_CONFIGS/`
3. Add missing fields with appropriate default values (if documented)

#### Issue: Deprecated Fields Still in Use

**Symptoms**:
```
DeprecationWarning: Field 'X' is deprecated and will be removed in version Y
```

**Solutions**:
1. Check the breaking changes section for migration instructions
2. Update to the new field name/format
3. Remove deprecated fields before the removal version

#### Issue: Import Errors After Upgrade

**Symptoms**:
```
ModuleNotFoundError: No module named 'trading_system.X'
```

**Solutions**:
1. Reinstall the package:
   ```bash
   pip install -e . --upgrade
   ```
2. Check Python version compatibility
3. Verify all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

#### Issue: Database Schema Mismatch

**Symptoms**:
```
sqlite3.OperationalError: no such column: ...
```

**Solutions**:
1. Check database schema version:
   ```python
   from trading_system.storage.schema import get_schema_version
   # Check version and migrate if needed
   ```
2. Run schema migration (when available):
   ```python
   from trading_system.storage.schema import migrate_schema
   migrate_schema(conn, from_version=1, to_version=2)
   ```
3. For development: Delete old database and recreate:
   ```bash
   rm results/backtest_results.db  # Backup first!
   ```

### Getting Help

If you encounter issues not covered here:

1. **Check Documentation**:
   - Review `agent-files/` for detailed architecture docs
   - Check `README.md` for usage examples
   - Review `TESTING_GUIDE.md` for testing procedures

2. **Validate Your Setup**:
   ```bash
   # Run quick test
   ./quick_test.sh

   # Check configs
   python -m trading_system backtest --config your_config.yaml --dry-run
   ```

3. **Review Examples**:
   - Check `EXAMPLE_CONFIGS/` for working examples
   - Review `tests/fixtures/configs/` for test configurations

4. **Check Logs**:
   - Review backtest logs in `results/{run_id}/`
   - Check for validation errors in console output

---

## Future Migration Notes

### Planned Changes (Not Yet Implemented)

This section will document planned breaking changes in future versions:

- **Version 0.2.0** (Planned):
  - ML integration into backtest event loop
  - Enhanced config validation
  - Additional strategy types

- **Version 1.0.0** (Future):
  - Production-ready API
  - Enhanced error handling
  - Performance optimizations

**Note**: These are planned changes and may not occur. Check release notes for actual changes.

---

## Appendix

### Config Schema Reference

For detailed schema documentation, see:
- `agent-files/02_CONFIGS_AND_PARAMETERS.md` - Complete parameter documentation
- `EXAMPLE_CONFIGS/README.md` - Example configurations
- `trading_system/configs/run_config.py` - RunConfig Pydantic model
- `trading_system/configs/strategy_config.py` - StrategyConfig Pydantic model

### Version Compatibility Matrix

| System Version | Config Schema | Python Version | Status |
|---------------|---------------|----------------|--------|
| 0.0.2         | 1.0           | 3.9+           | Current |

### Migration Script Examples

See `scripts/` directory for migration utilities (when available).

---

**Last Updated**: 2024-12-19
**Maintained By**: Trading System Contributors


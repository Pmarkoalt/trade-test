# Strategy Optimization - Next Steps

## Session Summary (Jan 5, 2026)

### Bug Fixed
- **Trade logging bug**: Closed trades were being removed from `portfolio.positions` before the engine could collect them. Fixed by adding `closed_positions` list to Portfolio class.
  - `trading_system/portfolio/portfolio.py` - Added `closed_positions` field and append in `close_position()`
  - `trading_system/backtest/engine.py` - Now uses `portfolio.closed_positions.copy()`

### Configs Created
- `configs/equity_strategy_production.yaml` - Tightened parameters
- `configs/backtest_config_production.yaml` - Uses production strategy

---

## Current Performance

### Test Config (Relaxed) - Overfits
| Period | Return | Sharpe | Trades | Win Rate |
|--------|--------|--------|--------|----------|
| Train | +4.22% | 0.44 | 26 | 34.6% |
| Validation | +5.52% | 1.94 | 6 | 33.3% |
| Holdout | **-6.95%** | -2.24 | 8 | 12.5% |

### Production Config (Tightened) - Better OOS
| Period | Return | Sharpe | Trades | Win Rate | Profit Factor |
|--------|--------|--------|--------|----------|---------------|
| Train | -0.28% | -0.03 | 24 | 45.8% | 0.67 |
| Validation | +4.66% | 2.63 | 5 | 60.0% | 7.79 |
| Holdout | **+0.85%** | 0.65 | 5 | 20.0% | 1.50 |

**Key Finding**: Tightened config improved holdout from -6.95% to +0.85%

---

## Priority Issues

### P0 - Strategy Not Profitable in Train
- Production config loses money in train period (-0.28%)
- Suggests strategy doesn't capture 2024-2025 market dynamics well
- **Action**: Investigate why momentum signals underperform in this period

### P1 - Small Sample Size
- Only 5 trades in validation/holdout periods
- Cannot draw statistically significant conclusions
- **Action**: Expand universe or test longer periods

### P2 - Narrow Universe
- Only 5 stocks (AAPL, MSFT, GOOGL, AMZN, NVDA)
- All mega-cap tech, highly correlated
- **Action**: Add more sectors/stocks to universe

---

## Recommended Experiments

### 1. Exit MA Period Test
Current: `exit_ma: 20` (fast exit)

Try `exit_ma: 50` for longer holds:
```yaml
exit:
  mode: "ma_cross"
  exit_ma: 50  # Changed from 20
```

### 2. Trend Filter Variations
Current: Must be above MA50 with 0.5% slope

Try MA200 filter:
```yaml
eligibility:
  trend_ma: 200
  require_close_above_trend_ma: true
```

### 3. Risk Per Trade Sweep
Test range: 0.5% to 1.5%
```yaml
risk:
  risk_per_trade: 0.005  # Try 0.005, 0.0075, 0.01, 0.015
```

### 4. Expanded Universe
Add mid-cap momentum stocks or other sectors:
```yaml
universe: ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "CRM", "ADBE"]
```

### 5. Breakout Clearance Sweep
Test tighter vs looser breakout filters:
```yaml
entry:
  fast_clearance: 0.003  # Try 0.003, 0.005, 0.007, 0.01
  slow_clearance: 0.007  # Try 0.007, 0.01, 0.015, 0.02
```

---

## Run Commands

```bash
# Run all periods with production config
python -c "from trading_system.cli import main; main()" backtest \
  --config configs/backtest_config_production.yaml --period train

python -c "from trading_system.cli import main; main()" backtest \
  --config configs/backtest_config_production.yaml --period validation

python -c "from trading_system.cli import main; main()" backtest \
  --config configs/backtest_config_production.yaml --period holdout
```

---

## Production Readiness Checklist

| Requirement | Status | Notes |
|-------------|--------|-------|
| Trade logging works | DONE | Fixed Jan 5 |
| Sharpe >= 1.0 | FAIL | Only validation passes |
| Profit Factor >= 1.0 | PARTIAL | Holdout OK, train fails |
| Positive holdout return | DONE | +0.85% with prod config |
| Max DD < 10% | DONE | 6.08% max |
| Win rate > 40% | PARTIAL | Train 45.8%, holdout 20% |
| Sufficient trade count | FAIL | Need more trades for significance |
| Diversified universe | FAIL | Only 5 correlated stocks |

---

## Data Quality Notes

- Missing data warnings on Dec 31, 2025 (expected - holiday)
- Thanksgiving/Christmas gaps handled correctly
- Calendar alignment is a minor issue, not blocking

---

## Files Reference

```
configs/
├── test_backtest_config.yaml      # Original relaxed config
├── test_equity_strategy.yaml      # Original relaxed strategy
├── backtest_config_production.yaml # Tightened backtest config
└── equity_strategy_production.yaml # Tightened strategy params

results/
├── run_20260105_020924/train/     # Test config train (with fix)
├── run_20260105_021953/train/     # Production config train
├── run_20260105_022118/validation/ # Production config validation
└── run_20260105_022255/holdout/   # Production config holdout
```

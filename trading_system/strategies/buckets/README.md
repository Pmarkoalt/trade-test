# Strategy Buckets

This module implements the strategy buckets for daily signal generation as defined in `PROJECT_NEXT_STEPS.md`.

## Overview

Two production-ready strategy buckets have been implemented:

- **Bucket A: Safe S&P** - Conservative equity strategy for S&P 500 universe
- **Bucket B: Top-Cap Crypto** - Aggressive crypto strategy for top market cap coins

Both buckets generate signals with rationale tags for newsletter consumption and are designed to work with the daily signal generation pipeline.

## Bucket A: Safe S&P Strategy

**File:** `bucket_a_safe_sp.py`
**Config:** `configs/bucket_a_safe_sp.yaml`
**Strategy Type:** `safe_sp`
**Asset Class:** `equity`

### Characteristics

- **Focus:** Low drawdown, stable returns, regime-aware
- **Universe:** S&P 500 core stocks (40+ liquid names)
- **Position Sizing:** Conservative (0.5% risk per trade vs 0.75% standard)
- **Max Positions:** 6 (vs 8 standard)
- **Max Exposure:** 60% (vs 80% standard)

### Eligibility Filters

1. **Long-term trend:** close > MA200
2. **Bullish regime:** MA50 > MA200
3. **Momentum confirmation:** MA50 slope > 0.3% over 20 days
4. **Market regime filter (optional):** SPY > MA50 (risk-off gating)
5. **News sentiment filter (optional):** Sentiment score > threshold

### Entry Triggers

- **Fast:** close >= highest_close_20d * 1.003 (0.3% clearance)
- **Slow:** close >= highest_close_55d * 1.008 (0.8% clearance)

Tighter than standard momentum (0.5% / 1.0%)

### Exit Logic

- **Trailing:** close < MA50 (conservative, slower exit)
- **Hard stop:** entry - 2.0 * ATR14 (tighter than standard 2.5)

### Rationale Tags

Signals include rationale tags for newsletter:
- `technical_20d_breakout` / `technical_55d_breakout`
- `technical_bullish_regime`
- `technical_above_ma200`
- `technical_strong_relative_strength` / `technical_positive_relative_strength`
- `news_positive_sentiment` / `news_neutral_sentiment` (if enabled)

## Bucket B: Top-Cap Crypto Strategy

**File:** `bucket_b_crypto_topcat.py`
**Config:** `configs/bucket_b_topcat_crypto.yaml`
**Strategy Type:** `topcat_crypto`
**Asset Class:** `crypto`

### Characteristics

- **Focus:** Higher turnover, volatility-aware sizing, adaptive exits
- **Universe:** Dynamic top 10 by volume/market cap (monthly rebalance)
- **Position Sizing:** Volatility-adjusted (reduces size for high ATR/close ratio)
- **Max Positions:** 8
- **Max Exposure:** 80%

### Eligibility Filters

1. **Strict trend:** close > MA200 (STRICT, no exceptions)
2. **Relative strength (optional):** Outperformance vs BTC
3. **Volatility filter (optional):** ATR14/close < 15%

### Entry Triggers

- **Fast:** close >= highest_close_20d * 1.005 (0.5% clearance)
- **Slow:** close >= highest_close_55d * 1.010 (1.0% clearance)

### Exit Logic (Staged)

- **Stage 1:** close < MA20 → tighten stop to entry - 2.0 * ATR14
- **Stage 2:** close < MA50 OR tightened stop hit → exit
- **Hard stop:** entry - 3.5 * ATR14 (wider for crypto volatility)

### Volatility-Aware Sizing

Position size is adjusted based on volatility:
```python
adjustment = 1.0 / (1.0 + (ATR14/close) * 10.0)
# Clamped to [0.3, 1.0]
```

Higher volatility → smaller position size

### Rationale Tags

Signals include rationale tags for newsletter:
- `technical_20d_breakout` / `technical_55d_breakout`
- `technical_above_ma200`
- `technical_strong_outperformance_vs_btc` / `technical_positive_relative_strength_vs_btc`
- `technical_underperformance_vs_btc`
- `risk_high_volatility_reduced_size` / `risk_low_volatility_normal_size`

## Universe Selection

### Equity Universe (Bucket A)

**Module:** `trading_system/data/equity_universe.py`

Functions:
- `load_sp500_universe(universe_file=None)` - Load SP500 from file or hardcoded list
- `select_equity_universe(universe_type, universe_file, available_data, min_bars)` - Select and filter universe

Hardcoded SP500 core universe includes 40+ liquid stocks across sectors:
- Mega-cap tech: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
- Financials: JPM, BAC, WFC, GS, MS, C
- Healthcare: UNH, JNJ, LLY, ABBV, MRK, PFE
- Consumer: WMT, HD, MCD, NKE, COST, SBUX
- And more...

### Crypto Universe (Bucket B)

**Module:** `trading_system/data/universe.py`

Functions:
- `select_top_crypto_by_volume(available_data, top_n, lookback_days, reference_date)` - Select top N by volume

Dynamic selection based on:
- Average dollar volume over lookback period
- Optional: market cap, liquidity score
- Monthly rebalancing (configurable)

## Daily Signal Generation

**Script:** `scripts/generate_daily_signals.py`

### Usage

```bash
# Generate signals for latest available date
python scripts/generate_daily_signals.py

# Generate signals for specific date
python scripts/generate_daily_signals.py --date 2024-01-15

# Custom data directories
python scripts/generate_daily_signals.py \
  --equity-data-dir data/equity/daily \
  --crypto-data-dir data/crypto/daily \
  --output-dir results/daily_signals
```

### Output Format

Signals are saved to JSON with structure:
```json
{
  "bucket_a": [
    {
      "symbol": "AAPL",
      "asset_class": "equity",
      "date": "2024-01-15",
      "side": "BUY",
      "trigger_reason": "safe_sp_fast_20d_breakout",
      "entry_price": 185.50,
      "stop_price": 180.25,
      "bucket": "A_SAFE_SP",
      "rationale_tags": [
        "technical_20d_breakout",
        "technical_bullish_regime",
        "technical_above_ma200"
      ],
      ...
    }
  ],
  "bucket_b": [...],
  "metadata": {
    "generated_at": "2024-01-15T10:30:00",
    "reference_date": "2024-01-15",
    "total_signals": 5
  }
}
```

## Integration with Newsletter (Agent 3)

Signals are designed for newsletter consumption:

1. **Rationale tags** provide human-readable reasons for each signal
2. **Bucket metadata** allows grouping in newsletter sections
3. **Urgency scores** help prioritize signals
4. **Entry/stop prices** provide actionable trade plans

Newsletter can consume the JSON output and format as:

```
📊 BUCKET A: Safe S&P Picks

AAPL - BUY at $185.50 (Stop: $180.25)
✓ 20-day breakout
✓ Bullish regime (MA50 > MA200)
✓ Above long-term trend (MA200)

...
```

## Integration with Paper Trading (Agent 4)

Signals include all necessary fields for order execution:

- `entry_price` - Suggested entry price
- `stop_price` - Initial stop loss
- `side` - BUY/SELL
- `symbol` - Ticker symbol
- `asset_class` - equity/crypto (for broker routing)

Paper trading adapter can consume signals and create TradePlan objects.

## Testing

To test the bucket strategies:

```bash
# Run unit tests (when implemented)
pytest tests/test_bucket_strategies.py

# Generate test signals
python scripts/generate_daily_signals.py --date 2024-01-15

# Run backtest with bucket configs
python -m trading_system backtest --config configs/bucket_a_safe_sp.yaml
python -m trading_system backtest --config configs/bucket_b_topcat_crypto.yaml
```

## Configuration Parameters

### Tunable Parameters (Bucket A)

- `ma_slope_min`: MA50 slope threshold (default: 0.003)
- `fast_clearance`: 20D breakout clearance (default: 0.003)
- `slow_clearance`: 55D breakout clearance (default: 0.008)
- `exit_ma`: Exit MA period (default: 50)
- `hard_stop_atr_mult`: Stop ATR multiplier (default: 2.0)
- `risk_per_trade`: Risk per trade (default: 0.005)
- `max_positions`: Max positions (default: 6)

### Tunable Parameters (Bucket B)

- `min_volume_usd`: Min daily volume for universe (default: 5M)
- `max_symbols`: Top N coins (default: 10)
- `rebalance_frequency`: Universe rebalance (default: monthly)
- `fast_clearance`: 20D breakout clearance (default: 0.005)
- `slow_clearance`: 55D breakout clearance (default: 0.010)
- `hard_stop_atr_mult`: Initial stop ATR multiplier (default: 3.5)
- `tightened_stop_atr_mult`: Tightened stop ATR multiplier (default: 2.0)
- `max_volatility_ratio`: Max ATR/close ratio (default: 0.15)

## Next Steps

1. **Backtest validation** - Run backtests on both buckets to validate performance
2. **Optimization** - Use Optuna to optimize parameters for each bucket
3. **News integration** - Add news sentiment data for Bucket A
4. **Sector limits** - Add sector concentration limits for Bucket A
5. **Market cap data** - Add market cap data for better Bucket B universe selection
6. **Walk-forward testing** - Validate robustness with walk-forward analysis

## Files Created

### Strategy Implementations
- `trading_system/strategies/buckets/__init__.py`
- `trading_system/strategies/buckets/bucket_a_safe_sp.py`
- `trading_system/strategies/buckets/bucket_b_crypto_topcat.py`

### Universe Selection
- `trading_system/data/equity_universe.py`
- Enhanced `trading_system/data/universe.py` with `select_top_crypto_by_volume()`

### Configuration
- `configs/bucket_a_safe_sp.yaml`
- `configs/bucket_b_topcat_crypto.yaml`

### Scripts
- `scripts/generate_daily_signals.py`

### Registry Updates
- Updated `trading_system/strategies/strategy_registry.py` to register bucket strategies
- Updated `trading_system/data/__init__.py` to export universe functions

## Agent 2 Deliverables ✓

All deliverables from `PROJECT_NEXT_STEPS.md` Agent 2 section completed:

- ✓ Bucket A (Safe S&P) strategy logic and configuration
- ✓ Bucket B (Top-cap crypto) strategy logic and configuration
- ✓ Universe selection rules (SP500 + dynamic crypto universe)
- ✓ Deterministic daily signal generation for both buckets
- ✓ Clear rationale tags for newsletter use
- ✓ Configs that can be optimized without changing code

# Expected Trades Documentation

This document describes the expected trades that should occur when running the trading system on the test dataset.

## Test Dataset Overview

- **Equity Symbols**: AAPL, MSFT, GOOGL
- **Crypto Symbols**: BTC, ETH, SOL
- **Date Range**: October 1, 2023 - December 31, 2023 (3 months)
- **Benchmark**: SPY (equity), BTC (crypto)

## Expected Trade Characteristics

The test dataset has been designed with specific price patterns that should trigger signals:

### Equity Trades (AAPL, MSFT, GOOGL)

1. **Trending Upward Pattern**
   - All three equity symbols show a consistent upward trend
   - Prices start low and gradually increase over the 3-month period
   - Should trigger MA50 eligibility (close > MA50) after ~50 days
   - Should trigger 20D breakout signals when price exceeds prior 20-day high by 0.5%

2. **Expected Signal Dates** (approximate)
   - First signals expected around late October / early November (after 20-day lookback period)
   - Multiple signals possible as prices continue to break new highs
   - Signals should occur when close > highest_close_20d * 1.005

3. **Exit Conditions**
   - MA20 cross below (equity strategy)
   - Stop loss triggers (if price drops 2.5 * ATR14 below entry)

### Crypto Trades (BTC, ETH, SOL)

1. **Trending Upward Pattern**
   - All three crypto symbols show consistent upward trend
   - Prices increase daily with small volatility
   - Should trigger MA200 eligibility (close > MA200) after ~200 days
   - Note: MA200 requires 200 days of history, so signals may be limited in 3-month dataset

2. **Expected Signal Dates** (approximate)
   - Limited signals expected due to MA200 requirement (needs 200 days)
   - If signals occur, should follow similar pattern to equity (20D/55D breakouts)
   - Staged exit mode: MA50 cross triggers exit

## Verification Checklist

When running integration tests, verify:

- [ ] Signals are generated after sufficient lookback period (20+ days for breakouts)
- [ ] No lookahead bias (signals use only past data)
- [ ] Eligibility filters are applied correctly
- [ ] Capacity checks are performed
- [ ] Orders are created from valid signals
- [ ] Fills simulate realistic slippage and fees
- [ ] Positions are sized correctly (risk-based)
- [ ] Exits occur on stop loss or MA cross
- [ ] Portfolio equity updates correctly
- [ ] No positions exceed max_position_notional (15%)
- [ ] No portfolio exceeds max_exposure (80%)

## Notes

- This is a simplified test dataset for integration testing
- Real backtests require 250+ days of history for full indicator calculations
- Expected trades may vary based on exact implementation details
- The test dataset is designed to trigger at least some signals for validation

## Future Enhancements

- Add specific date/price targets for known signal triggers
- Document exact expected entry/exit prices
- Add test cases for edge conditions (missing data, extreme moves, etc.)


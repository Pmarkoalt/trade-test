Eligibility

close > MA200 (strict MVP)

Optional v1.1:

relative strength vs BTC: roc60_crypto - roc60_btc > 0

Entry (same triggers)

Fast:

close >= high_close_20 * (1 + 0.005)
Slow:

close >= high_close_55 * (1 + 0.010)

Capacity Reject (stricter)

if order_notional > 0.0025 * ADV20: reject

Exits (staged default)

Hard stop base:

stop_price = entry - 3.0 * ATR14

Stage 1:

if close < MA20: tighten stop to entry - 2.0 * ATR14

Stage 2:

if close < MA50 OR close < tightened_stop: exit next open

Alternative mode for validation:

close < MA50 only
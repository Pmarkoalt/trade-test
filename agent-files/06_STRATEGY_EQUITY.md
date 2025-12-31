Eligibility

Must pass all:

close > MA50

MA50[t] / MA50[t-20] - 1 > 0.005

Optional v1.1:

relative strength vs SPY: roc60_stock - roc60_spy > 0

Entry Triggers (OR)

Fast:

close >= high_close_20 * (1 + 0.005)
Slow:

close >= high_close_55 * (1 + 0.010)

Capacity Reject

if order_notional > 0.005 * ADV20: reject

Exits

Trailing:

if close < MA20: exit next open
Hard stop:

stop_price = entry - 2.5 * ATR14

if close < stop_price: exit next open

Exit reason priority

hard_stop

trailing_ma_cross
(so logs are consistent)
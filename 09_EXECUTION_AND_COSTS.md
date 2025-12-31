Fees

equities: 1 bp per side

crypto: 8 bps per side

Slippage Model (dynamic)

Inputs:

base bps (8 eq, 10 crypto)

vol_mult = ATR14 / mean(ATR14 last 60), clip [0.5, 3]

size_penalty = clip(order_notional / (0.01 * ADV20), 0.5, 2.0)

crypto weekend penalty: 1.5 on Sat/Sun UTC

stress multiplier:

equities: if SPY weekly return < -3% => 2.0

crypto: if BTC weekly return < -5% => 2.0

slippage_mean = base * vol_mult * size_penalty * weekend_penalty * stress_mult

slippage_std = slippage_mean * 0.75

draw N(mean, std) and clamp to >= 0

Apply to open price:

BUY: fill = open * (1 + bps/10000)

SELL: fill = open * (1 - bps/10000)

Slippage Clustering

If stress_mult == 2.0:

multiply std by 1.5 (fatter tails)
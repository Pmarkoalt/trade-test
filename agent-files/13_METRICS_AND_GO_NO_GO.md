Primary (holdout must pass all)

Sharpe > 1.0

Max DD < 15%

Calmar > 1.5

trades >= 50 combined

Secondary (pass 3/4)

expectancy > 0.3R after costs

profit factor > 1.4

corr to benchmark < 0.8

99th percentile daily loss < 5%

Hard reject if any

DD > 20%

Sharpe < 0.75

Calmar < 1.0

top-3 trade dependency

2× slippage flips expectancy negative

bootstrap sharpe 5th < 0.4

permutation test percentile < 95

Decision Tree

Pass primary → check secondary → check stress tests → paper trade (12 weeks)

Fail primary or trigger reject → stop, redesign (no holdout tuning)

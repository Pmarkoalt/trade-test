When signals > available slots

Compute candidate features:

breakout strength (ATR-normalized)

relative momentum vs benchmark

diversification bonus = 1 - avg_corr_to_portfolio

Rank-normalize each component across candidates:

map to [0..1] by rank

Score:

0.50 * breakout_rank + 0.30 * momentum_rank + 0.20 * div_rank

Selection loop:

sort desc by score

skip if violates:

max_positions

max_exposure

capacity constraint

correlation guard

Correlation Guard

Apply only if existing positions >= 4:

compute avg pairwise corr of current portfolio

if avg_pairwise_corr > 0.70:

reject/deprioritize candidates whose corr to portfolio > 0.75
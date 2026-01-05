Required Tests (train+val only)
1) Parameter sensitivity grid

ATR mult, clearance, exit mode, vol scaling mode, trend filter choice
Plot heatmaps and check:

no sharp peaks

stable neighborhoods

2) Slippage stress tests

baseline, 2×, 3× slippage multiplier
Acceptance:

2×: Sharpe > 0.75

3×: Calmar > 1.0

3) Bootstrap (1,000)

Resample trade R-multiples; compute CI for Sharpe/Calmar/DD
Reject if:

Sharpe 5th percentile < 0.4

4) Permutation test (1,000)

Randomize entry dates while preserving exit logic/hold distributions
Reject if:

actual Sharpe < 95th percentile of randomized

5) Correlation stress diagnostic

Compare avg pairwise corr during drawdowns vs normal
Warn if drawdown corr > 0.70

6) Adverse scenarios

bear months only

range months

flash crash sim (slippage×5 + forced stop fills)
Acceptance:

portfolio survives, DD < 25% in crash sim

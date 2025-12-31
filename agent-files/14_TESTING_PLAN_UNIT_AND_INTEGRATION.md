Unit tests (pytest)

indicators correctness (MA/ATR/ROC/breakouts)

no-lookahead guard (features at t use only <= t)

sizing clamps work

capacity rejects correct

slippage non-negative, stress increases mean

staged crypto stop tightening logic

score ranking stable & deterministic

Integration tests

run on small 3-symbol toy dataset; verify known expected trades

run 3 months of real data; produce logs; validate schema
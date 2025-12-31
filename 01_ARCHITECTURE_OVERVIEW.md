High-level Modules
trading_system/
  configs/
  data/
  indicators/
  strategies/
  execution/
  portfolio/
  backtest/
  validation/
  reporting/
  cli.py
Core Responsibilities
data/: load + validate OHLCV, compute dollar volume, handle calendars

indicators/: MA, ATR, ROC, breakout highs, correlations

strategies/: generate signals (entries/exits), stop updates, scoring

execution/: fill simulation with fees + slippage, capacity constraints

portfolio/: positions, cash, equity curve, exposure, risk sizing

backtest/: daily event loop, walk-forward split logic

validation/: sensitivity grid, bootstrap, permutation, regime tests

reporting/: CSV/JSON outputs, metrics summaries

Key Invariants
No lookahead: indicators at date t only use data <= t

Execution: fills occur at open[t+1] using slippage/fees

Determinism: all randomness uses seeded RNG (slippage draws, crash sim, permutation)

Data Model (Canonical)
Bar: {date, open, high, low, close, volume, dollar_volume}

FeatureRow: indicators for a symbol at a date

Signal: intent to buy/sell with metadata + score fields

Order: executable action for next open

Fill: realized execution result

Position: entry info, qty, stop_price, unrealized pnl, etc.


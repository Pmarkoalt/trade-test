Goal

Implement a config-driven daily momentum trading system for equities + crypto with:

Daily bars only

Signal at D close

Execute at D+1 open

Walk-forward backtest with train/validate/holdout

Realistic costs (fees + slippage model)

Robustness suite (sensitivity, bootstrap, permutation, stress tests)

Paper-trading adapters later

Deliverables

Core engine (portfolio + orders + fills + logs)

Strategy factory (equity + crypto configs)

Data pipeline (OHLCV, ADV, calendars)

Backtest runner (event loop, no lookahead)

Metrics/reporting

Validation suite

CLI entrypoints

Language & Stack (recommended)

Python 3.11+

pandas/numpy

pydantic (configs)

pytest (unit tests)

polars optional (speed)

rich / loguru (logging)

Non-goals (MVP)

ML, sentiment, news scraping

intraday execution

leverage/derivatives

dynamic crypto universe

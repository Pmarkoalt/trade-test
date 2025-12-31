Data Requirements
Equities

24 months OHLCV for NASDAQ-100 (or SP500)

Benchmark: SPY

Use market calendar (skip weekends/holidays)

Crypto

Daily OHLCV in UTC for fixed list:
BTC, ETH, BNB, XRP, ADA, SOL, DOT, MATIC, LTC, LINK

Benchmark: BTC

Data Validation Rules

For each symbol/day:

OHLC sanity: low <= open/close <= high

Reject day if abs(close/prev_close - 1) > 0.50 (likely bad tick)

Missing days:

1 day missing: skip updates, log warning

2+ consecutive: mark unhealthy; if in position, exit at next available open

Derived Fields

dollar_volume = close * volume

ADV20 = mean(dollar_volume over last 20 bars)

Calendar Handling

Equities: use pandas_market_calendars (NYSE/NASDAQ) or exchange calendar

Crypto: daily UTC dates continuous
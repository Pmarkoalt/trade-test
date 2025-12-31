Portfolio State

cash

positions dict: {symbol: Position}

equity curve series

realized pnl, unrealized pnl

exposure (gross %, per position %)

portfolio returns

Position Object

Fields:

symbol, asset_class

entry_date, entry_price, qty

stop_price

hard_stop_atr_mult

exit_reason (when closed)

cost tracking: entry_fee, entry_slippage_bps, etc.

Position Sizing (risk-based)

Given:

risk_pct = risk_per_trade * risk_multiplier

stop_distance = entry_price - stop_price
Compute:

risk_dollars = equity * risk_pct

qty = floor(risk_dollars / stop_distance)

Clamp:

position notional <= equity * max_position_notional

total exposure <= equity * max_exposure

Volatility Scaling (portfolio-level)

Compute 20D realized vol of portfolio equity returns, compare to median vol over 252D.

vol_ratio = vol_20 / median_vol_252

risk_multiplier = clip(1 / max(vol_ratio, 1), 0.33, 1.0)

Apply to new entries only.
Config-driven Strategy Factory

Two YAML configs:

equity_config.yaml

crypto_config.yaml

Required Config Fields
name: "equity_momentum"
asset_class: "equity"  # or "crypto"
universe: "NASDAQ-100" # or "SP500" or explicit list
benchmark: "SPY"       # equities; crypto uses "BTC"

indicators:
  ma_periods: [20, 50, 200]
  atr_period: 14
  roc_period: 60
  breakout_fast: 20
  breakout_slow: 55
  adv_lookback: 20
  corr_lookback: 20

eligibility:
  # equities
  trend_ma: 50
  ma_slope_lookback: 20
  ma_slope_min: 0.005
  require_close_above_trend_ma: true
  # crypto
  require_close_above_ma200: false

entry:
  fast_clearance: 0.005
  slow_clearance: 0.010

exit:
  mode: "ma_cross"     # equities: "ma_cross"
  exit_ma: 20          # 20 or 50
  hard_stop_atr_mult: 2.5

risk:
  risk_per_trade: 0.0075
  max_positions: 8
  max_exposure: 0.80
  max_position_notional: 0.15

capacity:
  max_order_pct_adv: 0.005 # equities 0.5%
  
costs:
  fee_bps: 1
  slippage_base_bps: 8
  slippage_std_mult: 0.75


Crypto config differences:

eligibility.require_close_above_ma200: true

exit.mode: "staged" default

capacity.max_order_pct_adv: 0.0025

fees/slippage base different

weekend penalty enabled

Frozen vs Tunable

Frozen (structural):

risk_per_trade, max_positions, max_exposure, execution model, universes, ADV caps

Tunable (train/validate only):

ATR mult, clearance, exit mode/MA, vol scaling mode, relative strength filter on/off
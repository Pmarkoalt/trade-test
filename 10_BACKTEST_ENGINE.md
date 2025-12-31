Event-driven daily loop

For each date t:

Update bars up to t close

Compute features using data <= t

Generate signals at close t

Create orders for t+1 open

Execute orders at t+1 open with slippage+fees

Update portfolio

Update stops (based on t+1 close)

Log daily metrics

No Lookahead Rules

orders at t+1 open cannot use any t+1 close info

stop updates can use t+1 close and take effect at t+2 open

Outputs

equity_curve.csv

trade_log.csv

weekly_summary.csv

monthly_report.json

scenario_comparison.json
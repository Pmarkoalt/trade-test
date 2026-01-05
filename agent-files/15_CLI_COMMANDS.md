Suggested CLI

python -m trading_system backtest --config configs/run.yaml

python -m trading_system validate --config configs/run.yaml

python -m trading_system holdout --config configs/run.yaml

python -m trading_system report --run-id <id>

Run config contains:

dataset paths

date ranges

split definitions

strategy configs to load

output folder

RNG seed

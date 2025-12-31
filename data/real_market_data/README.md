# Real Market Data for Testing

This directory contains real historical market data downloaded for testing the trading system with actual market conditions.

## Downloading Data

To download real market data for testing:

```bash
# Install yfinance (if not already installed)
pip install yfinance

# Download default dataset (AAPL, MSFT, GOOGL, AMZN, TSLA, BTC, ETH, SOL)
python scripts/download_real_market_data.py --output data/real_market_data/

# Download specific symbols
python scripts/download_real_market_data.py \
  --equity-symbols AAPL MSFT GOOGL \
  --crypto-symbols BTC ETH \
  --start-date 2020-01-01 \
  --end-date 2024-01-01 \
  --output data/real_market_data/

# Download only equity data
python scripts/download_real_market_data.py --equity-only --output data/real_market_data/

# Download only crypto data
python scripts/download_real_market_data.py --crypto-only --output data/real_market_data/
```

## Directory Structure

```
real_market_data/
├── equity/
│   └── ohlcv/
│       ├── AAPL.csv
│       ├── MSFT.csv
│       └── ...
├── crypto/
│   └── ohlcv/
│       ├── BTC.csv
│       ├── ETH.csv
│       └── ...
└── benchmarks/
    ├── SPY.csv
    └── BTC.csv
```

## Running Tests

Once data is downloaded, run the real market data tests:

```bash
# Run all real market data tests
pytest tests/integration/test_real_market_data.py -v

# Test specific scenarios
pytest tests/integration/test_real_market_data.py::TestRealMarketData::test_real_data_validation -v
pytest tests/integration/test_real_market_data.py::TestRealMarketData::test_bull_market_condition -v
pytest tests/integration/test_real_market_data.py::TestRealMarketData::test_bear_market_condition -v
```

## Market Conditions Tested

The test suite verifies system performance under different market conditions:

1. **Bull Market** (2020-04 to 2021-11): Post-COVID recovery period
2. **Bear Market** (2022): Market downturn period
3. **Range Market** (2019): Sideways/volatile period

## Data Quality Issues Tested

The tests verify that the system handles real-world data quality issues:

- Missing trading days (holidays, weekends)
- Extreme price moves (>50% daily returns)
- Low volume days
- Duplicate dates (if any)
- Data validation and OHLC relationships

## Using Real Data in Backtests

To use this data in a backtest, update your config file:

```yaml
dataset:
  equity_path: "data/real_market_data/equity/ohlcv"
  crypto_path: "data/real_market_data/crypto/ohlcv"
  benchmark_path: "data/real_market_data/benchmarks"
  format: "csv"
  start_date: "2020-01-01"
  end_date: "2024-01-01"
```

Then run your backtest:

```bash
python -m trading_system backtest --config your_config.yaml
```

## Notes

- Data is downloaded from yfinance (free, no API key required)
- Data includes OHLCV columns: date, open, high, low, close, volume, dollar_volume
- Files are saved in CSV format compatible with the trading system
- Data is automatically validated during download
- Missing data or errors are logged but don't stop the download process


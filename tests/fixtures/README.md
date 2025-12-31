# Test Fixtures

This directory contains test data, configurations, and fixtures for the trading system tests.

## Directory Structure

```
fixtures/
├── README.md                    # This file
├── EXPECTED_TRADES.md          # Documentation of expected trades
├── configs/                     # Test configuration files
│   ├── equity_test_config.yaml
│   ├── crypto_test_config.yaml
│   └── run_test_config.yaml
├── benchmarks/                  # Benchmark data (SPY, BTC)
│   ├── SPY.csv
│   └── BTC.csv
├── AAPL_sample.csv             # Equity sample data (3 months)
├── MSFT_sample.csv             # Equity sample data (3 months)
├── GOOGL_sample.csv            # Equity sample data (3 months)
├── BTC_sample.csv              # Crypto sample data (3 months)
├── ETH_sample.csv              # Crypto sample data (3 months)
├── SOL_sample.csv              # Crypto sample data (3 months)
└── [edge case files]           # INVALID_OHLC.csv, MISSING_DAY.csv, etc.
```

## Sample Data Files

### Equity Data (3 symbols, 3 months)

- **AAPL_sample.csv**: Apple Inc. OHLCV data (Oct-Dec 2023)
- **MSFT_sample.csv**: Microsoft Corp. OHLCV data (Oct-Dec 2023)
- **GOOGL_sample.csv**: Google/Alphabet Inc. OHLCV data (Oct-Dec 2023)

Each file contains:
- Date range: 2023-10-01 to 2023-12-31
- Consistent upward trending pattern (designed to trigger signals)
- Valid OHLCV relationships
- Realistic volume data

### Crypto Data (3 symbols, 3 months)

- **BTC_sample.csv**: Bitcoin OHLCV data (Oct-Dec 2023)
- **ETH_sample.csv**: Ethereum OHLCV data (Oct-Dec 2023)
- **SOL_sample.csv**: Solana OHLCV data (Oct-Dec 2023)

Each file contains:
- Date range: 2023-10-01 to 2023-12-31 (daily data, no weekends excluded)
- Consistent upward trending pattern
- Valid OHLCV relationships
- Realistic volume data

**Note**: Crypto data includes weekends (all 7 days), while equity data excludes weekends (trading days only).

## Configuration Files

### equity_test_config.yaml

Test configuration for equity momentum strategy with 3-symbol universe:
- Universe: ["AAPL", "MSFT", "GOOGL"]
- Benchmark: SPY
- Standard equity parameters (MA50 eligibility, MA20 exit, etc.)

### crypto_test_config.yaml

Test configuration for crypto momentum strategy with 3-symbol universe:
- Universe: ["BTC", "ETH", "SOL"]
- Benchmark: BTC
- Standard crypto parameters (MA200 eligibility, staged exit, etc.)

### run_test_config.yaml

Test run configuration for integration testing:
- Points to test fixtures directory
- 3-month date range (Oct-Dec 2023)
- Simplified validation settings
- Test-appropriate metrics thresholds

## Benchmark Data

- **benchmarks/SPY.csv**: SPY ETF data (used as equity benchmark)
- **benchmarks/BTC.csv**: Bitcoin data (used as crypto benchmark)

These files are copied from the main fixtures directory for use in integration tests.

## Edge Case Data

The fixtures directory also contains edge case test data:
- **INVALID_OHLC.csv**: Data with invalid OHLC relationships (for validation testing)
- **MISSING_DAY.csv**: Data with missing trading days (for missing data handling)
- **EXTREME_MOVE.csv**: Data with extreme price moves (for edge case testing)

## Usage

### In Tests

```python
from trading_system.data import load_ohlcv_data

# Load test data (note: files use _sample suffix)
FIXTURES_DIR = "tests/fixtures"
equity_data = load_ohlcv_data(FIXTURES_DIR, ["AAPL_sample", "MSFT_sample", "GOOGL_sample"])
crypto_data = load_ohlcv_data(FIXTURES_DIR, ["BTC_sample", "ETH_sample", "SOL_sample"])

# Or rename files to remove _sample suffix if needed:
# AAPL_sample.csv -> AAPL.csv, etc.
```

### With Test Utilities

```python
from tests.utils import create_sample_bar, create_sample_feature_row

# Create sample data
bar = create_sample_bar(date=pd.Timestamp("2023-11-15"), symbol="AAPL")
features = create_sample_feature_row(date=pd.Timestamp("2023-11-15"), symbol="AAPL")
```

## Data Format

All CSV files follow the standard format:

```csv
date,open,high,low,close,volume
2023-10-01,150.00,152.00,149.50,151.00,50000000
```

- **date**: Trading date (YYYY-MM-DD)
- **open**: Opening price
- **high**: High price
- **low**: Low price
- **close**: Closing price
- **volume**: Trading volume

The data loader automatically computes `dollar_volume = close * volume`.

## Notes

- Sample data files use `_sample` suffix to distinguish from real data files
- Files are designed to trigger known signals for integration testing
- The 3-month dataset provides sufficient data for basic indicator calculations (MA20, MA50)
- MA200 calculations may be limited with only 3 months of data
- See EXPECTED_TRADES.md for expected trade patterns


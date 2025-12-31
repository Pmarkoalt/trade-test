# Data Loading Specification

Complete specification for data file formats, loading procedures, and validation.

---

## File Structure

### Directory Layout

```
data/
├── equity/
│   ├── ohlcv/
│   │   ├── AAPL.csv
│   │   ├── MSFT.csv
│   │   └── ... (one file per symbol)
│   └── universe/
│       └── NASDAQ-100.csv  # List of symbols
├── crypto/
│   ├── ohlcv/
│   │   ├── BTC.csv
│   │   ├── ETH.csv
│   │   └── ... (one file per symbol)
│   └── universe/
│       └── fixed_list.txt  # Fixed 10-asset list
└── benchmarks/
    ├── SPY.csv
    └── BTC.csv
```

---

## CSV File Format

### OHLCV Files (Equity & Crypto)

**File Name:** `{SYMBOL}.csv` (e.g., `AAPL.csv`, `BTC.csv`)

**Required Columns:**
- `date`: Date in format `YYYY-MM-DD` or `YYYY-MM-DD HH:MM:SS` (UTC for crypto)
- `open`: Opening price (float)
- `high`: High price (float)
- `low`: Low price (float)
- `close`: Closing price (float)
- `volume`: Volume (float, shares for equity, units for crypto)

**Optional Columns:**
- `dollar_volume`: Pre-computed dollar volume (if not provided, computed as `close * volume`)

**Example:**
```csv
date,open,high,low,close,volume
2023-01-01,150.25,152.30,149.80,151.50,50000000
2023-01-02,151.50,153.20,150.90,152.80,48000000
2023-01-03,152.80,154.10,152.20,153.50,52000000
```

**Requirements:**
- Dates must be in chronological order
- No duplicate dates per symbol
- No missing dates (except market holidays for equity)
- All prices must be > 0
- Volume must be >= 0

---

### Universe Files

#### Equity Universe (NASDAQ-100 or S&P 500)

**File Name:** `NASDAQ-100.csv` or `SP500.csv`

**Format:** Simple list of symbols, one per line

**Example:**
```csv
symbol
AAPL
MSFT
GOOGL
AMZN
...
```

**Alternative:** Can be specified as list in config: `universe: ["AAPL", "MSFT", ...]`

#### Crypto Universe (Fixed List)

**File Name:** `fixed_list.txt` or specified in config

**Format:** One symbol per line, or comma-separated in config

**Fixed List:**
```
BTC
ETH
BNB
XRP
ADA
SOL
DOT
MATIC
LTC
LINK
```

---

## Data Loading Functions

### Load OHLCV Data

```python
def load_ohlcv_data(
    data_path: str,
    symbols: List[str],
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load OHLCV data for multiple symbols.
    
    Args:
        data_path: Path to directory containing CSV files
        symbols: List of symbols to load
        start_date: Optional start date filter
        end_date: Optional end date filter
    
    Returns:
        Dictionary mapping symbol -> DataFrame with columns:
        - date (index)
        - open, high, low, close, volume
        - dollar_volume (computed if not present)
    """
    data = {}
    
    for symbol in symbols:
        file_path = os.path.join(data_path, f"{symbol}.csv")
        
        if not os.path.exists(file_path):
            log.warning(f"File not found: {file_path}")
            continue
        
        # Load CSV
        df = pd.read_csv(
            file_path,
            parse_dates=['date'],
            index_col='date',
            dtype={
                'open': float,
                'high': float,
                'low': float,
                'close': float,
                'volume': float
            }
        )
        
        # Sort by date
        df = df.sort_index()
        
        # Compute dollar_volume if not present
        if 'dollar_volume' not in df.columns:
            df['dollar_volume'] = df['close'] * df['volume']
        
        # Filter by date range
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        # Validate
        if not validate_ohlcv(df, symbol):
            log.error(f"Validation failed for {symbol}")
            continue
        
        data[symbol] = df
    
    return data
```

---

### Load Universe

```python
def load_universe(
    universe_type: str,
    universe_path: Optional[str] = None
) -> List[str]:
    """
    Load universe list.
    
    Args:
        universe_type: "NASDAQ-100", "SP500", or "crypto"
        universe_path: Optional path to universe file
    
    Returns:
        List of symbols
    """
    if universe_type == "crypto":
        # Fixed list
        return ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOT", "MATIC", "LTC", "LINK"]
    
    # Equity universe
    if universe_path is None:
        if universe_type == "NASDAQ-100":
            universe_path = "data/equity/universe/NASDAQ-100.csv"
        elif universe_type == "SP500":
            universe_path = "data/equity/universe/SP500.csv"
        else:
            raise ValueError(f"Unknown universe type: {universe_type}")
    
    # Load from file
    df = pd.read_csv(universe_path)
    
    if 'symbol' in df.columns:
        symbols = df['symbol'].tolist()
    else:
        # Assume first column is symbols
        symbols = df.iloc[:, 0].tolist()
    
    return [str(s).upper().strip() for s in symbols]
```

---

### Load Benchmark

```python
def load_benchmark(
    benchmark_symbol: str,
    benchmark_path: str,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Load benchmark data (SPY for equity, BTC for crypto).
    
    Args:
        benchmark_symbol: "SPY" or "BTC"
        benchmark_path: Path to benchmark directory
        start_date: Optional start date
        end_date: Optional end date
    
    Returns:
        DataFrame with OHLCV data
    
    Raises:
        ValueError: If benchmark file not found or insufficient data
    """
    file_path = os.path.join(benchmark_path, f"{benchmark_symbol}.csv")
    
    if not os.path.exists(file_path):
        raise ValueError(f"Benchmark file not found: {file_path}")
    
    # Load (same as OHLCV)
    df = load_ohlcv_data(benchmark_path, [benchmark_symbol], start_date, end_date)
    
    if benchmark_symbol not in df:
        raise ValueError(f"Benchmark {benchmark_symbol} not in data")
    
    benchmark_df = df[benchmark_symbol]
    
    # Validate minimum history
    if len(benchmark_df) < 250:
        raise ValueError(f"Benchmark {benchmark_symbol} has insufficient data: {len(benchmark_df)} days")
    
    return benchmark_df
```

---

## Data Validation

### Validate OHLCV DataFrame

```python
def validate_ohlcv(df: pd.DataFrame, symbol: str) -> bool:
    """
    Validate OHLCV data.
    
    Checks:
    1. Required columns present
    2. OHLC relationships valid
    3. No negative prices/volumes
    4. No extreme moves (>50% in one day)
    5. Dates in chronological order
    6. No duplicate dates
    
    Returns:
        True if valid, False otherwise
    """
    # Check required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        log.error(f"{symbol}: Missing columns: {missing_cols}")
        return False
    
    # Check OHLC relationships
    invalid_ohlc = (
        (df['low'] > df['high']) |
        (df['open'] < df['low']) | (df['open'] > df['high']) |
        (df['close'] < df['low']) | (df['close'] > df['high'])
    )
    
    if invalid_ohlc.any():
        invalid_dates = df.index[invalid_ohlc].tolist()
        log.error(f"{symbol}: Invalid OHLC at dates: {invalid_dates[:10]}")
        return False
    
    # Check for negative values
    if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
        log.error(f"{symbol}: Non-positive prices found")
        return False
    
    if (df['volume'] < 0).any():
        log.error(f"{symbol}: Negative volume found")
        return False
    
    # Check for extreme moves
    if len(df) > 1:
        returns = df['close'].pct_change()
        extreme_moves = abs(returns) > 0.50
        if extreme_moves.any():
            extreme_dates = df.index[extreme_moves].tolist()
            log.warning(f"{symbol}: Extreme moves (>50%) at dates: {extreme_dates[:10]}")
            # Mark as data errors (but don't fail validation)
    
    # Check date order
    if not df.index.is_monotonic_increasing:
        log.error(f"{symbol}: Dates not in chronological order")
        return False
    
    # Check for duplicate dates
    if df.index.duplicated().any():
        duplicates = df.index[df.index.duplicated()].tolist()
        log.error(f"{symbol}: Duplicate dates: {duplicates[:10]}")
        return False
    
    return True
```

---

## Calendar Handling

### Equity Calendar (Market Days Only)

```python
def get_trading_days(
    all_dates: pd.DatetimeIndex,
    end_date: pd.Timestamp,
    lookback: int = 5
) -> List[pd.Timestamp]:
    """
    Get last N trading days (excluding weekends/holidays).
    
    For equity stress multiplier calculation.
    """
    # Filter to trading days (exclude weekends)
    trading_days = all_dates[all_dates.weekday < 5]  # Mon-Fri
    
    # Filter to <= end_date
    trading_days = trading_days[trading_days <= end_date]
    
    # Get last N
    return trading_days[-lookback:].tolist()
```

**Alternative:** Use `pandas_market_calendars` for exchange-specific calendars:

```python
import pandas_market_calendars as mcal

def get_trading_calendar(exchange: str = "NASDAQ"):
    """
    Get trading calendar for exchange.
    """
    cal = mcal.get_calendar(exchange)
    return cal
```

### Crypto Calendar (365 Days)

```python
def get_crypto_days(
    end_date: pd.Timestamp,
    lookback: int = 7
) -> List[pd.Timestamp]:
    """
    Get last N calendar days (continuous, no weekends).
    
    For crypto stress multiplier calculation.
    """
    start_date = end_date - pd.Timedelta(days=lookback)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    return dates.tolist()
```

---

## Data Quality Checks

### Missing Data Detection

```python
def detect_missing_data(
    df: pd.DataFrame,
    symbol: str,
    expected_frequency: str = "D"
) -> Dict[str, Any]:
    """
    Detect missing data periods.
    
    Returns:
        Dictionary with:
        - missing_dates: List of missing dates
        - consecutive_gaps: List of (start, end) for consecutive gaps
        - gap_lengths: List of gap lengths
    """
    # Create expected date range
    if expected_frequency == "D":
        expected_dates = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq='D'
        )
    else:
        # Use trading calendar
        expected_dates = get_trading_calendar().schedule(
            start_date=df.index.min(),
            end_date=df.index.max()
        ).index
    
    # Find missing dates
    missing_dates = expected_dates.difference(df.index)
    
    # Find consecutive gaps
    consecutive_gaps = []
    if len(missing_dates) > 0:
        sorted_missing = sorted(missing_dates)
        gap_start = sorted_missing[0]
        
        for i in range(1, len(sorted_missing)):
            if (sorted_missing[i] - sorted_missing[i-1]).days > 1:
                # Gap ended
                consecutive_gaps.append((gap_start, sorted_missing[i-1]))
                gap_start = sorted_missing[i]
        
        # Last gap
        consecutive_gaps.append((gap_start, sorted_missing[-1]))
    
    return {
        'missing_dates': missing_dates.tolist(),
        'consecutive_gaps': consecutive_gaps,
        'gap_lengths': [(end - start).days + 1 for start, end in consecutive_gaps]
    }
```

---

## Data Loading Pipeline

### Complete Loading Procedure

```python
def load_all_data(config: RunConfig) -> Tuple[MarketData, Dict[str, pd.DataFrame]]:
    """
    Load all data for backtest.
    
    Args:
        config: Run configuration
    
    Returns:
        (market_data, benchmarks)
    """
    # Load universes
    equity_universe = load_universe(config.strategies.equity.universe)
    crypto_universe = load_universe("crypto")
    
    # Load equity data
    equity_data = load_ohlcv_data(
        config.dataset.equity_path,
        equity_universe,
        start_date=pd.Timestamp(config.dataset.start_date),
        end_date=pd.Timestamp(config.dataset.end_date)
    )
    
    # Load crypto data
    crypto_data = load_ohlcv_data(
        config.dataset.crypto_path,
        crypto_universe,
        start_date=pd.Timestamp(config.dataset.start_date),
        end_date=pd.Timestamp(config.dataset.end_date)
    )
    
    # Load benchmarks
    spy_benchmark = load_benchmark(
        "SPY",
        config.dataset.benchmark_path,
        start_date=pd.Timestamp(config.dataset.start_date),
        end_date=pd.Timestamp(config.dataset.end_date)
    )
    
    btc_benchmark = load_benchmark(
        "BTC",
        config.dataset.benchmark_path,
        start_date=pd.Timestamp(config.dataset.start_date),
        end_date=pd.Timestamp(config.dataset.end_date)
    )
    
    # Combine into MarketData
    market_data = MarketData()
    market_data.bars = {**equity_data, **crypto_data}
    market_data.benchmarks = {
        "SPY": spy_benchmark,
        "BTC": btc_benchmark
    }
    
    # Validate data quality
    validate_all_data(market_data, config)
    
    return market_data, {"SPY": spy_benchmark, "BTC": btc_benchmark}
```

---

## Summary

Data loading specification includes:
- File structure and naming conventions
- CSV format requirements
- Loading functions with validation
- Calendar handling (equity vs crypto)
- Missing data detection
- Complete loading pipeline

All data loading can be implemented using these specifications.


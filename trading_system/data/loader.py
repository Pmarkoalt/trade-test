"""Data loading functions for OHLCV data, universes, and benchmarks."""

from typing import Dict, List, Optional, Tuple
import os
import logging
import pandas as pd

from ..models.market_data import MarketData
from .validator import validate_ohlcv, detect_missing_data

logger = logging.getLogger(__name__)


# Fixed crypto universe (from spec)
CRYPTO_UNIVERSE = [
    "BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOT", "MATIC", "LTC", "LINK"
]


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
            logger.warning(f"File not found: {file_path}")
            continue
        
        try:
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
            if start_date is not None:
                df = df[df.index >= start_date]
            if end_date is not None:
                df = df[df.index <= end_date]
            
            # Validate
            if not validate_ohlcv(df, symbol):
                logger.error(f"Validation failed for {symbol}, skipping")
                continue
            
            data[symbol] = df
            
        except Exception as e:
            logger.error(f"Error loading {symbol} from {file_path}: {e}")
            continue
    
    return data


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
    if universe_type.lower() == "crypto":
        # Fixed list
        return CRYPTO_UNIVERSE.copy()
    
    # Equity universe
    if universe_path is None:
        # Default paths (can be overridden)
        universe_type_upper = universe_type.upper()
        if universe_type_upper == "NASDAQ-100":
            universe_path = "data/equity/universe/NASDAQ-100.csv"
        elif universe_type_upper in ("SP500", "S&P500", "S&P_500"):
            universe_path = "data/equity/universe/SP500.csv"
        else:
            raise ValueError(f"Unknown universe type: {universe_type}")
    
    if not os.path.exists(universe_path):
        raise FileNotFoundError(
            f"Universe file not found: {universe_path}. "
            f"Universe type: {universe_type}"
        )
    
    try:
        # Load from file
        df = pd.read_csv(universe_path)
        
        if 'symbol' in df.columns:
            symbols = df['symbol'].tolist()
        else:
            # Assume first column is symbols
            symbols = df.iloc[:, 0].tolist()
        
        # Clean and normalize symbols
        symbols = [str(s).upper().strip() for s in symbols if pd.notna(s)]
        
        return symbols
        
    except Exception as e:
        raise ValueError(f"Error loading universe from {universe_path}: {e}")


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
    
    # Load using load_ohlcv_data (reuse logic)
    data = load_ohlcv_data(
        benchmark_path,
        [benchmark_symbol],
        start_date=start_date,
        end_date=end_date
    )
    
    if benchmark_symbol not in data:
        raise ValueError(
            f"Benchmark {benchmark_symbol} not in loaded data. "
            "Check file format and validation."
        )
    
    benchmark_df = data[benchmark_symbol]
    
    # Validate minimum history (250 days as per spec)
    if len(benchmark_df) < 250:
        raise ValueError(
            f"Benchmark {benchmark_symbol} has insufficient data: "
            f"{len(benchmark_df)} days (minimum 250 required)"
        )
    
    return benchmark_df


def load_all_data(
    equity_path: str,
    crypto_path: str,
    benchmark_path: str,
    equity_universe: List[str],
    crypto_universe: Optional[List[str]] = None,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None
) -> Tuple[MarketData, Dict[str, pd.DataFrame]]:
    """
    Load all data for backtest.
    
    Args:
        equity_path: Path to equity OHLCV directory
        crypto_path: Path to crypto OHLCV directory
        benchmark_path: Path to benchmark directory
        equity_universe: List of equity symbols
        crypto_universe: Optional list of crypto symbols (defaults to fixed list)
        start_date: Optional start date filter
        end_date: Optional end date filter
    
    Returns:
        Tuple of (market_data, benchmarks_dict)
        - market_data: MarketData object with all bars
        - benchmarks_dict: Dict with "SPY" and "BTC" DataFrames
    """
    # Default crypto universe if not provided
    if crypto_universe is None:
        crypto_universe = CRYPTO_UNIVERSE
    
    # Load equity data
    logger.info(f"Loading equity data from {equity_path}")
    equity_data = load_ohlcv_data(
        equity_path,
        equity_universe,
        start_date=start_date,
        end_date=end_date
    )
    logger.info(f"Loaded {len(equity_data)} equity symbols")
    
    # Load crypto data
    logger.info(f"Loading crypto data from {crypto_path}")
    crypto_data = load_ohlcv_data(
        crypto_path,
        crypto_universe,
        start_date=start_date,
        end_date=end_date
    )
    logger.info(f"Loaded {len(crypto_data)} crypto symbols")
    
    # Load benchmarks
    logger.info(f"Loading benchmarks from {benchmark_path}")
    spy_benchmark = load_benchmark(
        "SPY",
        benchmark_path,
        start_date=start_date,
        end_date=end_date
    )
    
    btc_benchmark = load_benchmark(
        "BTC",
        benchmark_path,
        start_date=start_date,
        end_date=end_date
    )
    
    # Combine into MarketData
    market_data = MarketData()
    market_data.bars = {**equity_data, **crypto_data}
    market_data.benchmarks = {
        "SPY": spy_benchmark,
        "BTC": btc_benchmark
    }
    
    logger.info(
        f"Data loading complete: {len(market_data.bars)} symbols, "
        f"{len(spy_benchmark)} SPY bars, {len(btc_benchmark)} BTC bars"
    )
    
    return market_data, {"SPY": spy_benchmark, "BTC": btc_benchmark}


"""Data loading functions for OHLCV data, universes, and benchmarks."""

from typing import Dict, List, Optional, Tuple, Union
import os
import logging
import pandas as pd

from ..models.market_data import MarketData
from .validator import validate_ohlcv, detect_missing_data
from .universe import (
    CryptoUniverseManager,
    UniverseConfig,
    create_universe_config_from_dict,
    FIXED_CRYPTO_UNIVERSE,
)
from .sources import (
    BaseDataSource,
    CSVDataSource,
    DataCache,
    CachedDataSource
)
from .memory_profiler import optimize_dataframe_dtypes, MemoryProfiler

logger = logging.getLogger(__name__)


# Fixed crypto universe (from spec) - kept for backward compatibility
CRYPTO_UNIVERSE = FIXED_CRYPTO_UNIVERSE


def load_ohlcv_data(
    data_path: Union[str, BaseDataSource],
    symbols: List[str],
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    use_cache: bool = False,
    cache_dir: str = ".cache",
    optimize_memory: bool = True,
    chunk_size: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load OHLCV data for multiple symbols.
    
    Supports both traditional CSV file paths and new data source objects.
    Includes memory optimization options.
    
    Args:
        data_path: Path to directory containing CSV files, or a BaseDataSource instance
        symbols: List of symbols to load
        start_date: Optional start date filter
        end_date: Optional end date filter
        use_cache: If True, use caching layer (for data sources)
        cache_dir: Cache directory (if use_cache=True)
        optimize_memory: If True, optimize DataFrame dtypes to reduce memory (default: True)
        chunk_size: If provided, load symbols in chunks of this size (for large universes)
    
    Returns:
        Dictionary mapping symbol -> DataFrame with columns:
        - date (index)
        - open, high, low, close, volume
        - dollar_volume (computed if not present)
    """
    # If data_path is a BaseDataSource, use it directly
    if isinstance(data_path, BaseDataSource):
        source = data_path
        
        # Wrap with cache if requested
        if use_cache:
            cache = DataCache(cache_dir=cache_dir)
            source = CachedDataSource(source, cache)
        
        data = source.load_ohlcv(symbols, start_date=start_date, end_date=end_date)
    else:
        # Otherwise, treat as CSV file path (backward compatibility)
        source = CSVDataSource(data_path)
        if use_cache:
            cache = DataCache(cache_dir=cache_dir)
            source = CachedDataSource(source, cache)
        
        # Load in chunks if chunk_size is specified
        if chunk_size and len(symbols) > chunk_size:
            data = {}
            for i in range(0, len(symbols), chunk_size):
                chunk_symbols = symbols[i:i + chunk_size]
                chunk_data = source.load_ohlcv(chunk_symbols, start_date=start_date, end_date=end_date)
                data.update(chunk_data)
                logger.debug(f"Loaded chunk {i//chunk_size + 1}: {len(chunk_data)} symbols")
        else:
            data = source.load_ohlcv(symbols, start_date=start_date, end_date=end_date)
    
    # Optimize memory usage if requested
    if optimize_memory:
        for symbol, df in data.items():
            data[symbol] = optimize_dataframe_dtypes(df)
    
    return data


def load_universe(
    universe_type: Union[str, List[str]],
    universe_path: Optional[str] = None,
    universe_config: Optional[Union[UniverseConfig, dict]] = None,
    available_data: Optional[Dict[str, pd.DataFrame]] = None,
    reference_date: Optional[pd.Timestamp] = None
) -> List[str]:
    """
    Load universe list.
    
    Args:
        universe_type: "NASDAQ-100", "SP500", "crypto", or explicit list of symbols
        universe_path: Optional path to universe file (for equity universes)
        universe_config: Optional UniverseConfig or dict for crypto dynamic universe
        available_data: Optional available OHLCV data (required for dynamic crypto universe)
        reference_date: Optional reference date for dynamic universe selection
    
    Returns:
        List of symbols
    
    Note:
        For crypto with universe_config, uses dynamic selection.
        For crypto without universe_config, returns fixed list (backward compatible).
    """
    # Handle explicit list
    if isinstance(universe_type, list):
        return universe_type
    
    universe_type_lower = universe_type.lower()
    
    if universe_type_lower == "crypto":
        # Check if we have universe config for dynamic selection
        if universe_config is not None:
            # Convert dict to UniverseConfig if needed
            if isinstance(universe_config, dict):
                universe_config = create_universe_config_from_dict(universe_config)
            
            # For dynamic selection, we need available_data
            if available_data is None:
                logger.warning(
                    "universe_config provided but no available_data. "
                    "Falling back to fixed universe."
                )
                return CRYPTO_UNIVERSE.copy()
            
            # Use dynamic universe selection
            manager = CryptoUniverseManager(universe_config)
            selected_universe = manager.select_universe(available_data, reference_date)
            
            # Validate selected universe
            is_valid, warnings = manager.validate_universe(selected_universe, available_data, min_symbols=1)
            if not is_valid:
                logger.warning(f"Universe validation failed: {warnings}")
            elif warnings:
                logger.info(f"Universe validation warnings: {warnings}")
            
            return selected_universe
        else:
            # Fixed list (backward compatible)
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
        
    except FileNotFoundError:
        raise
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"Universe file {universe_path} is empty: {e}") from e
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing universe file {universe_path}: {e}") from e
    except Exception as e:
        raise ValueError(f"Error loading universe from {universe_path}: {e}") from e


def load_benchmark(
    benchmark_symbol: str,
    benchmark_path: Union[str, BaseDataSource],
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    use_cache: bool = False,
    cache_dir: str = ".cache"
) -> pd.DataFrame:
    """
    Load benchmark data (SPY for equity, BTC for crypto).
    
    Supports both traditional CSV file paths and new data source objects.
    
    Args:
        benchmark_symbol: "SPY" or "BTC"
        benchmark_path: Path to benchmark directory, or a BaseDataSource instance
        start_date: Optional start date
        end_date: Optional end date
        use_cache: If True, use caching layer (for data sources)
        cache_dir: Cache directory (if use_cache=True)
    
    Returns:
        DataFrame with OHLCV data
    
    Raises:
        ValueError: If benchmark file not found or insufficient data
    """
    # Load using load_ohlcv_data (reuse logic)
    data = load_ohlcv_data(
        benchmark_path,
        [benchmark_symbol],
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache,
        cache_dir=cache_dir
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
    equity_path: Union[str, BaseDataSource],
    crypto_path: Union[str, BaseDataSource],
    benchmark_path: Union[str, BaseDataSource],
    equity_universe: List[str],
    crypto_universe: Optional[List[str]] = None,
    crypto_universe_config: Optional[Union[UniverseConfig, dict]] = None,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    use_cache: bool = False,
    cache_dir: str = ".cache",
    optimize_memory: bool = True,
    chunk_size: Optional[int] = None,
    profile_memory: bool = False
) -> Tuple[MarketData, Dict[str, pd.DataFrame]]:
    """
    Load all data for backtest.
    
    Supports both traditional CSV file paths and new data source objects.
    Includes memory optimization and profiling options.
    
    Args:
        equity_path: Path to equity OHLCV directory, or a BaseDataSource instance
        crypto_path: Path to crypto OHLCV directory, or a BaseDataSource instance
        benchmark_path: Path to benchmark directory, or a BaseDataSource instance
        equity_universe: List of equity symbols
        crypto_universe: Optional list of crypto symbols (defaults to fixed list or dynamic selection)
        crypto_universe_config: Optional UniverseConfig for dynamic crypto universe selection
        start_date: Optional start date filter
        end_date: Optional end date filter
        use_cache: If True, use caching layer (for data sources)
        cache_dir: Cache directory (if use_cache=True)
        optimize_memory: If True, optimize DataFrame dtypes to reduce memory (default: True)
        chunk_size: If provided, load symbols in chunks of this size (for large universes)
        profile_memory: If True, log memory usage during loading (default: False)
    
    Returns:
        Tuple of (market_data, benchmarks_dict)
        - market_data: MarketData object with all bars
        - benchmarks_dict: Dict with "SPY" and "BTC" DataFrames
    """
    profiler = MemoryProfiler() if profile_memory else None
    if profiler:
        profiler.log_snapshot("before_load_all_data")
    # Determine crypto universe
    if crypto_universe is None:
        if crypto_universe_config is not None:
            # Get available crypto symbols from data source
            if isinstance(crypto_path, BaseDataSource):
                available_crypto_symbols = crypto_path.get_available_symbols()
            else:
                # Fallback to CSV directory scanning
                available_crypto_symbols = []
                if os.path.exists(crypto_path):
                    for filename in os.listdir(crypto_path):
                        if filename.endswith('.csv'):
                            symbol = filename.replace('.csv', '').upper()
                            available_crypto_symbols.append(symbol)
            
            # Load all available crypto data for universe selection
            temp_crypto_data = load_ohlcv_data(
                crypto_path,
                available_crypto_symbols,
                start_date=start_date,
                end_date=end_date,
                use_cache=use_cache,
                cache_dir=cache_dir
            )
            
            # Select universe from available data
            if isinstance(crypto_universe_config, dict):
                universe_config = create_universe_config_from_dict(crypto_universe_config)
            else:
                universe_config = crypto_universe_config
            
            manager = CryptoUniverseManager(universe_config)
            crypto_universe = manager.select_universe(temp_crypto_data, end_date)
            
            # Validate selected universe
            is_valid, warnings = manager.validate_universe(crypto_universe, temp_crypto_data, min_symbols=1)
            if not is_valid:
                logger.warning(f"Universe validation failed: {warnings}")
            elif warnings:
                logger.info(f"Universe validation warnings: {warnings}")
        else:
            # Default to fixed list
            crypto_universe = CRYPTO_UNIVERSE
    
    # Load equity data
    logger.info(f"Loading equity data from {equity_path}")
    if profiler:
        profiler.log_snapshot("before_equity_load")
    equity_data = load_ohlcv_data(
        equity_path,
        equity_universe,
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache,
        cache_dir=cache_dir,
        optimize_memory=optimize_memory,
        chunk_size=chunk_size
    )
    if profiler:
        profiler.log_snapshot("after_equity_load")
        profiler.log_diff("before_equity_load", "after_equity_load")
    logger.info(f"Loaded {len(equity_data)} equity symbols")
    
    # Load crypto data
    logger.info(f"Loading crypto data from {crypto_path}")
    if profiler:
        profiler.log_snapshot("before_crypto_load")
    crypto_data = load_ohlcv_data(
        crypto_path,
        crypto_universe,
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache,
        cache_dir=cache_dir,
        optimize_memory=optimize_memory,
        chunk_size=chunk_size
    )
    if profiler:
        profiler.log_snapshot("after_crypto_load")
        profiler.log_diff("before_crypto_load", "after_crypto_load")
    logger.info(f"Loaded {len(crypto_data)} crypto symbols")
    
    # Load benchmarks
    logger.info(f"Loading benchmarks from {benchmark_path}")
    spy_benchmark = load_benchmark(
        "SPY",
        benchmark_path,
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache,
        cache_dir=cache_dir
    )
    
    btc_benchmark = load_benchmark(
        "BTC",
        benchmark_path,
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache,
        cache_dir=cache_dir
    )
    
    # Combine into MarketData
    market_data = MarketData()
    market_data.bars = {**equity_data, **crypto_data}
    market_data.benchmarks = {
        "SPY": spy_benchmark,
        "BTC": btc_benchmark
    }
    
    if profiler:
        profiler.log_snapshot("after_load_all_data")
        profiler.log_diff("before_load_all_data", "after_load_all_data")
        logger.info(profiler.get_summary())
    
    logger.info(
        f"Data loading complete: {len(market_data.bars)} symbols, "
        f"{len(spy_benchmark)} SPY bars, {len(btc_benchmark)} BTC bars"
    )
    
    return market_data, {"SPY": spy_benchmark, "BTC": btc_benchmark}


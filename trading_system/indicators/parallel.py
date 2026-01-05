"""Parallel processing utilities for multi-symbol indicator calculations."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable, Dict, Optional

import pandas as pd

# Handle multiprocessing import gracefully (may fail in some environments)
try:
    import multiprocessing as mp
except (ImportError, OSError):
    # Fallback if multiprocessing is not available
    import os

    mp = type("mp", (), {"cpu_count": lambda: os.cpu_count() or 1})()


def compute_features_parallel(
    symbols_data: Dict[str, pd.DataFrame],
    compute_fn: Callable,
    asset_classes: Dict[str, str],
    max_workers: Optional[int] = None,
    use_threads: bool = True,
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """Compute features for multiple symbols in parallel.

    Args:
        symbols_data: Dictionary mapping symbol to OHLC DataFrame
        compute_fn: Function to compute features (e.g., compute_features)
        asset_classes: Dictionary mapping symbol to asset class
        max_workers: Maximum number of workers (default: CPU count)
        use_threads: If True, use ThreadPoolExecutor; if False, use ProcessPoolExecutor
        **kwargs: Additional arguments to pass to compute_fn

    Returns:
        Dictionary mapping symbol to features DataFrame

    Example:
        >>> symbols_data = {'AAPL': df_aapl, 'MSFT': df_msft}
        >>> asset_classes = {'AAPL': 'equity', 'MSFT': 'equity'}
        >>> features = compute_features_parallel(
        ...     symbols_data,
        ...     compute_features,
        ...     asset_classes
        ... )
    """
    if max_workers is None:
        max_workers = mp.cpu_count()

    results = {}

    # Prepare arguments for each symbol
    def compute_for_symbol(symbol: str) -> tuple:
        """Compute features for a single symbol."""
        try:
            df_ohlc = symbols_data[symbol]
            asset_class = asset_classes.get(symbol, "equity")
            features_df = compute_fn(df_ohlc=df_ohlc, symbol=symbol, asset_class=asset_class, **kwargs)
            return (symbol, features_df, None)
        except Exception as e:
            return (symbol, None, e)

    # Use appropriate executor
    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    with executor_class(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(compute_for_symbol, symbol): symbol for symbol in symbols_data.keys()}

        # Collect results as they complete
        for future in as_completed(futures):
            symbol, features_df, error = future.result()
            if error is None:
                results[symbol] = features_df
            else:
                # Log error but continue with other symbols
                print(f"Error computing features for {symbol}: {error}")
                results[symbol] = None

    return results


def batch_compute_features(
    symbols_data: Dict[str, pd.DataFrame], compute_fn: Callable, asset_classes: Dict[str, str], batch_size: int = 10, **kwargs
) -> Dict[str, pd.DataFrame]:
    """Compute features for symbols in batches (sequential but batched).

    This is useful when you want to process symbols in groups but don't
    want full parallelization (e.g., for memory management).

    Args:
        symbols_data: Dictionary mapping symbol to OHLC DataFrame
        compute_fn: Function to compute features
        asset_classes: Dictionary mapping symbol to asset class
        batch_size: Number of symbols to process per batch
        **kwargs: Additional arguments to pass to compute_fn

    Returns:
        Dictionary mapping symbol to features DataFrame
    """
    results = {}
    symbols = list(symbols_data.keys())

    for i in range(0, len(symbols), batch_size):
        batch_symbols = symbols[i : i + batch_size]
        batch_data = {sym: symbols_data[sym] for sym in batch_symbols}

        # Process batch
        for symbol in batch_symbols:
            try:
                df_ohlc = batch_data[symbol]
                asset_class = asset_classes.get(symbol, "equity")
                features_df = compute_fn(df_ohlc=df_ohlc, symbol=symbol, asset_class=asset_class, **kwargs)
                results[symbol] = features_df
            except Exception as e:
                print(f"Error computing features for {symbol}: {e}")
                results[symbol] = None

    return results

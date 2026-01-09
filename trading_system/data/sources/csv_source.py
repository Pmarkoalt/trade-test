"""CSV file data source implementation."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..memory_profiler import optimize_dataframe_dtypes
from ..validator import validate_ohlcv
from .base_source import BaseDataSource

logger = logging.getLogger(__name__)

# Default number of parallel workers for data loading
DEFAULT_MAX_WORKERS = 8


class CSVDataSource(BaseDataSource):
    """Data source for loading OHLCV data from CSV files.

    Expected file structure:
    - Directory containing CSV files named {SYMBOL}.csv
    - Each CSV should have: date, open, high, low, close, volume
    """

    def __init__(self, data_path: str):
        """Initialize CSV data source.

        Args:
            data_path: Path to directory containing CSV files
        """
        if not os.path.isdir(data_path):
            raise ValueError(f"Data path is not a directory: {data_path}")
        self.data_path = data_path
        self._available_symbols: Optional[List[str]] = None

    def _load_single_symbol(
        self,
        symbol: str,
        start_date: Optional[pd.Timestamp],
        end_date: Optional[pd.Timestamp],
        optimize_memory: bool,
    ) -> Tuple[str, Optional[pd.DataFrame], Optional[str]]:
        """Load a single symbol's data. Returns (symbol, df, error)."""
        file_path = os.path.join(self.data_path, f"{symbol}.csv")

        if not os.path.exists(file_path):
            return (symbol, None, f"CSV file not found: {file_path}")

        try:
            # Load CSV with float32 dtypes for memory efficiency
            df = pd.read_csv(
                file_path,
                parse_dates=["date"],
                index_col="date",
                dtype={"open": "float32", "high": "float32", "low": "float32", "close": "float32", "volume": "float32"},
            )

            # Sort by date
            df = df.sort_index()

            # Compute dollar_volume if not present
            if "dollar_volume" not in df.columns:
                df["dollar_volume"] = (df["close"] * df["volume"]).astype("float32")

            # Filter by date range
            if start_date is not None:
                df = df[df.index >= start_date]
            if end_date is not None:
                df = df[df.index <= end_date]

            # Validate
            if not validate_ohlcv(df, symbol):
                return (symbol, None, f"Validation failed for {symbol}")

            # Optimize memory if requested
            if optimize_memory:
                df = optimize_dataframe_dtypes(df)

            return (symbol, df, None)

        except Exception as e:
            return (symbol, None, str(e))

    def load_ohlcv(
        self,
        symbols: List[str],
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        optimize_memory: bool = True,
        parallel: bool = True,
        max_workers: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load OHLCV data from CSV files.

        Args:
            symbols: List of symbols to load
            start_date: Optional start date filter
            end_date: Optional end date filter
            optimize_memory: If True, optimize DataFrame dtypes (default: True)
            parallel: If True, load symbols in parallel (default: True)
            max_workers: Number of parallel workers (default: 8)

        Returns:
            Dictionary mapping symbol -> DataFrame
        """
        if max_workers is None:
            max_workers = DEFAULT_MAX_WORKERS

        data = {}

        # Use parallel loading for multiple symbols
        if parallel and len(symbols) > 1:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(symbols))) as executor:
                futures = {
                    executor.submit(self._load_single_symbol, symbol, start_date, end_date, optimize_memory): symbol
                    for symbol in symbols
                }

                for future in as_completed(futures):
                    symbol, df, error = future.result()
                    if error:
                        logger.warning(error)
                    elif df is not None:
                        data[symbol] = df
        else:
            # Sequential loading for single symbol or when parallel=False
            for symbol in symbols:
                symbol, df, error = self._load_single_symbol(symbol, start_date, end_date, optimize_memory)
                if error:
                    logger.warning(error)
                elif df is not None:
                    data[symbol] = df

        return data

    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols from CSV directory.

        Returns:
            List of symbols (filenames without .csv extension)
        """
        if self._available_symbols is None:
            symbols = []
            if os.path.isdir(self.data_path):
                for filename in os.listdir(self.data_path):
                    if filename.endswith(".csv"):
                        symbol = filename[:-4]  # Remove .csv extension
                        symbols.append(symbol)
            self._available_symbols = sorted(symbols)

        return self._available_symbols.copy()

    def get_date_range(self, symbol: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Get available date range for a symbol.

        Args:
            symbol: Symbol to check

        Returns:
            Tuple of (start_date, end_date) or None if symbol not available
        """
        file_path = os.path.join(self.data_path, f"{symbol}.csv")
        if not os.path.exists(file_path):
            return None

        try:
            # Load just to get date range (efficient)
            df = pd.read_csv(file_path, parse_dates=["date"], index_col="date", usecols=["date"])  # Only load date column
            df = df.sort_index()
            if len(df) == 0:
                return None
            return (df.index[0], df.index[-1])
        except Exception as e:
            logger.error(f"Error getting date range for {symbol}: {e}")
            return None

    def supports_incremental(self) -> bool:
        """CSV sources support incremental loading via date filtering."""
        return True

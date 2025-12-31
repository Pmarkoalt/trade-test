"""CSV file data source implementation."""

import logging
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..memory_profiler import optimize_dataframe_dtypes
from ..validator import validate_ohlcv
from .base_source import BaseDataSource

logger = logging.getLogger(__name__)


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

    def load_ohlcv(
        self,
        symbols: List[str],
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        optimize_memory: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Load OHLCV data from CSV files.

        Args:
            symbols: List of symbols to load
            start_date: Optional start date filter
            end_date: Optional end date filter
            optimize_memory: If True, optimize DataFrame dtypes (default: True)

        Returns:
            Dictionary mapping symbol -> DataFrame
        """
        data = {}

        for symbol in symbols:
            file_path = os.path.join(self.data_path, f"{symbol}.csv")

            if not os.path.exists(file_path):
                logger.warning(f"CSV file not found: {file_path}")
                continue

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
                    logger.error(f"Validation failed for {symbol}, skipping")
                    continue

                # Optimize memory if requested
                if optimize_memory:
                    df = optimize_dataframe_dtypes(df)

                data[symbol] = df

            except Exception as e:
                logger.error(f"Error loading {symbol} from {file_path}: {e}")
                continue

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

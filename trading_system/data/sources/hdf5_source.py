"""HDF5 file data source implementation."""

import logging
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..validator import validate_ohlcv
from .base_source import BaseDataSource

logger = logging.getLogger(__name__)


class HDF5DataSource(BaseDataSource):
    """Data source for loading OHLCV data from HDF5 files.

    Supports both single-file format (with symbol column) and multi-file format
    (one file per symbol, or one HDF5 file with multiple keys).
    """

    def __init__(
        self, data_path: str, single_file: bool = False, key_format: str = "/{symbol}", table_name: Optional[str] = None
    ):
        """Initialize HDF5 data source.

        Args:
            data_path: Path to directory containing HDF5 files, or single HDF5 file
            single_file: If True, data_path is a single file with multiple keys
            key_format: Format string for HDF5 keys (e.g., "/{symbol}" or "/data/{symbol}")
            table_name: Table name if using table format (overrides key_format)
        """
        self.data_path = data_path
        self.single_file = single_file
        self.key_format = key_format
        self.table_name = table_name
        self._available_symbols: Optional[List[str]] = None
        self._store: Optional[pd.HDFStore] = None

    def _get_store(self) -> pd.HDFStore:
        """Get HDF5 store (for single file mode)."""
        if self.single_file:
            if self._store is None:
                if not os.path.exists(self.data_path):
                    raise FileNotFoundError(f"HDF5 file not found: {self.data_path}")
                self._store = pd.HDFStore(self.data_path, mode="r")
            return self._store
        return None

    def load_ohlcv(
        self, symbols: List[str], start_date: Optional[pd.Timestamp] = None, end_date: Optional[pd.Timestamp] = None
    ) -> Dict[str, pd.DataFrame]:
        """Load OHLCV data from HDF5 files.

        Args:
            symbols: List of symbols to load
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Dictionary mapping symbol -> DataFrame
        """
        data = {}

        if self.single_file:
            # Load from single HDF5 file
            store = self._get_store()

            for symbol in symbols:
                try:
                    # Determine key
                    if self.table_name:
                        key = self.table_name
                        # Filter by symbol if table has symbol column
                        df = pd.read_hdf(store, key=key)
                        if "symbol" in df.columns:
                            df = df[df["symbol"] == symbol].copy()
                            df.drop("symbol", axis=1, inplace=True)
                    else:
                        key = self.key_format.format(symbol=symbol)
                        df = pd.read_hdf(store, key=key)

                    if df.empty:
                        logger.warning(f"Symbol {symbol} not found in HDF5 file")
                        continue

                    # Ensure date is index
                    if df.index.name != "date" and "date" in df.columns:
                        df.set_index("date", inplace=True)

                    df = self._process_dataframe(df, symbol, start_date, end_date)
                    if df is not None:
                        data[symbol] = df

                except KeyError:
                    logger.warning(f"Key not found for symbol {symbol} in HDF5 file")
                    continue
                except Exception as e:
                    logger.error(f"Error loading {symbol} from HDF5: {e}")
                    continue

        else:
            # Load from multiple files (one per symbol)
            for symbol in symbols:
                file_path = os.path.join(self.data_path, f"{symbol}.h5")

                if not os.path.exists(file_path):
                    logger.warning(f"HDF5 file not found: {file_path}")
                    continue

                try:
                    with pd.HDFStore(file_path, mode="r") as store:
                        # Try default key
                        keys = store.keys()
                        if not keys:
                            logger.warning(f"No keys found in {file_path}")
                            continue

                        # Use first key (or specified key format)
                        key = keys[0]
                        df = pd.read_hdf(store, key=key)

                    # Ensure date is index
                    if df.index.name != "date" and "date" in df.columns:
                        df.set_index("date", inplace=True)

                    df = self._process_dataframe(df, symbol, start_date, end_date)
                    if df is not None:
                        data[symbol] = df

                except Exception as e:
                    logger.error(f"Error loading {symbol} from {file_path}: {e}")
                    continue

        return data

    def _process_dataframe(
        self, df: pd.DataFrame, symbol: str, start_date: Optional[pd.Timestamp], end_date: Optional[pd.Timestamp]
    ) -> Optional[pd.DataFrame]:
        """Process and validate a dataframe.

        Args:
            df: DataFrame to process
            symbol: Symbol name
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Processed DataFrame or None if validation fails
        """
        # Sort by date
        df = df.sort_index()

        # Compute dollar_volume if not present
        if "dollar_volume" not in df.columns:
            df["dollar_volume"] = df["close"] * df["volume"]

        # Filter by date range
        if start_date is not None:
            df = df[df.index >= start_date]
        if end_date is not None:
            df = df[df.index <= end_date]

        # Validate
        if not validate_ohlcv(df, symbol):
            logger.error(f"Validation failed for {symbol}, skipping")
            return None

        return df

    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols.

        Returns:
            List of symbols
        """
        if self._available_symbols is None:
            if self.single_file:
                store = self._get_store()
                keys = store.keys()
                if self.table_name:
                    # Read table and get unique symbols
                    df = pd.read_hdf(store, key=self.table_name)
                    if "symbol" in df.columns:
                        self._available_symbols = sorted(df["symbol"].unique().tolist())
                    else:
                        self._available_symbols = []
                else:
                    # Extract symbols from keys
                    symbols = []
                    for key in keys:
                        # Try to extract symbol from key based on format
                        # This is a simple implementation - may need customization
                        symbol = key.split("/")[-1]  # Last part of key path
                        symbols.append(symbol)
                    self._available_symbols = sorted(symbols)
            else:
                symbols = []
                if os.path.isdir(self.data_path):
                    for filename in os.listdir(self.data_path):
                        if filename.endswith(".h5") or filename.endswith(".hdf5"):
                            symbol = filename.rsplit(".", 1)[0]  # Remove extension
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
        try:
            data = self.load_ohlcv([symbol])
            if symbol in data:
                df = data[symbol]
                if len(df) > 0:
                    return (df.index[0], df.index[-1])
            return None
        except Exception as e:
            logger.error(f"Error getting date range for {symbol}: {e}")
            return None

    def supports_incremental(self) -> bool:
        """HDF5 sources support incremental loading via date filtering."""
        return True

    def __del__(self):
        """Close HDF5 store on cleanup."""
        if self._store is not None:
            try:
                self._store.close()
            except Exception:
                pass

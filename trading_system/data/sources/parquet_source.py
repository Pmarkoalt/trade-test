"""Parquet file data source implementation."""

import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
import logging

from .base_source import BaseDataSource
from ..validator import validate_ohlcv

logger = logging.getLogger(__name__)


class ParquetDataSource(BaseDataSource):
    """Data source for loading OHLCV data from Parquet files.
    
    Supports both single-file format (with symbol column) and multi-file format
    (one file per symbol).
    """
    
    def __init__(self, data_path: str, single_file: bool = False, symbol_column: str = "symbol"):
        """Initialize Parquet data source.
        
        Args:
            data_path: Path to directory containing Parquet files, or single Parquet file
            single_file: If True, data_path is a single file with all symbols
            symbol_column: Column name for symbol (if single_file=True)
        """
        self.data_path = data_path
        self.single_file = single_file
        self.symbol_column = symbol_column
        self._available_symbols: Optional[List[str]] = None
        self._data_cache: Optional[pd.DataFrame] = None
    
    def _load_single_file(self) -> pd.DataFrame:
        """Load data from single Parquet file (with caching)."""
        if self._data_cache is None:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Parquet file not found: {self.data_path}")
            self._data_cache = pd.read_parquet(self.data_path)
            # Ensure date is index
            if 'date' in self._data_cache.columns:
                self._data_cache.set_index('date', inplace=True)
        return self._data_cache
    
    def load_ohlcv(
        self,
        symbols: List[str],
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> Dict[str, pd.DataFrame]:
        """Load OHLCV data from Parquet files.
        
        Args:
            symbols: List of symbols to load
            start_date: Optional start date filter
            end_date: Optional end date filter
        
        Returns:
            Dictionary mapping symbol -> DataFrame
        """
        data = {}
        
        if self.single_file:
            # Load from single file
            df_all = self._load_single_file()
            
            if self.symbol_column not in df_all.columns:
                raise ValueError(f"Symbol column '{self.symbol_column}' not found in Parquet file")
            
            for symbol in symbols:
                symbol_df = df_all[df_all[self.symbol_column] == symbol].copy()
                if symbol_df.empty:
                    logger.warning(f"Symbol {symbol} not found in Parquet file")
                    continue
                
                # Remove symbol column and ensure date is index
                symbol_df.drop(self.symbol_column, axis=1, inplace=True)
                if symbol_df.index.name != 'date' and 'date' in symbol_df.columns:
                    symbol_df.set_index('date', inplace=True)
                
                symbol_df = self._process_dataframe(symbol_df, symbol, start_date, end_date)
                if symbol_df is not None:
                    data[symbol] = symbol_df
        
        else:
            # Load from multiple files (one per symbol)
            for symbol in symbols:
                file_path = os.path.join(self.data_path, f"{symbol}.parquet")
                
                if not os.path.exists(file_path):
                    logger.warning(f"Parquet file not found: {file_path}")
                    continue
                
                try:
                    df = pd.read_parquet(file_path)
                    
                    # Ensure date is index
                    if df.index.name != 'date' and 'date' in df.columns:
                        df.set_index('date', inplace=True)
                    
                    df = self._process_dataframe(df, symbol, start_date, end_date)
                    if df is not None:
                        data[symbol] = df
                
                except Exception as e:
                    logger.error(f"Error loading {symbol} from {file_path}: {e}")
                    continue
        
        return data
    
    def _process_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str,
        start_date: Optional[pd.Timestamp],
        end_date: Optional[pd.Timestamp]
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
            return None
        
        return df
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols.
        
        Returns:
            List of symbols
        """
        if self._available_symbols is None:
            if self.single_file:
                df_all = self._load_single_file()
                if self.symbol_column in df_all.columns:
                    self._available_symbols = sorted(df_all[self.symbol_column].unique().tolist())
                else:
                    self._available_symbols = []
            else:
                symbols = []
                if os.path.isdir(self.data_path):
                    for filename in os.listdir(self.data_path):
                        if filename.endswith('.parquet'):
                            symbol = filename[:-8]  # Remove .parquet extension
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
        """Parquet sources support incremental loading via date filtering."""
        return True


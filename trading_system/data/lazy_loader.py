"""Lazy loading wrapper for market data to reduce memory usage."""

import logging
from typing import Callable, Dict, List, Optional

import pandas as pd

from ..models.bar import Bar
from ..models.features import FeatureRow
from ..models.market_data import MarketData

logger = logging.getLogger(__name__)


class LazyMarketData(MarketData):
    """MarketData wrapper that loads data on-demand.

    This class wraps MarketData and provides lazy loading capabilities,
    loading bars and features only when they are accessed. This is useful
    for large datasets where not all symbols are needed at once.

    Note: Features are computed on-demand, which may be slower but uses less memory.
    """

    def __init__(
        self,
        load_bars_fn: Optional[Callable[[List[str]], Dict[str, pd.DataFrame]]] = None,
        compute_features_fn: Optional[Callable] = None,
        preload_symbols: Optional[List[str]] = None,
    ):
        """Initialize lazy market data loader.

        Args:
            load_bars_fn: Function to load bars for symbols: fn(symbols) -> Dict[symbol, DataFrame]
            compute_features_fn: Function to compute features: fn(df, symbol, asset_class) -> DataFrame
            preload_symbols: Optional list of symbols to preload immediately
        """
        super().__init__()
        self.load_bars_fn = load_bars_fn
        self.compute_features_fn = compute_features_fn
        self._loaded_symbols: set = set()
        self._feature_cache: Dict[str, pd.DataFrame] = {}
        self._asset_classes: Dict[str, str] = {}  # symbol -> asset_class

        # Preload symbols if specified
        if preload_symbols and load_bars_fn:
            self._preload_symbols(preload_symbols)

    def _preload_symbols(self, symbols: List[str]) -> None:
        """Preload bars for specified symbols.

        Args:
            symbols: List of symbols to preload
        """
        if not self.load_bars_fn:
            return

        symbols_to_load = [s for s in symbols if s not in self._loaded_symbols]
        if not symbols_to_load:
            return

        logger.debug(f"Preloading {len(symbols_to_load)} symbols")
        loaded_data = self.load_bars_fn(symbols_to_load)
        self.bars.update(loaded_data)
        self._loaded_symbols.update(loaded_data.keys())

    def _ensure_symbol_loaded(self, symbol: str) -> bool:
        """Ensure symbol bars are loaded.

        Args:
            symbol: Symbol to load

        Returns:
            True if symbol is now loaded, False if loading failed
        """
        if symbol in self._loaded_symbols:
            return True

        if not self.load_bars_fn:
            logger.warning(f"No load function available for symbol {symbol}")
            return False

        logger.debug(f"Lazy loading bars for {symbol}")
        loaded_data = self.load_bars_fn([symbol])
        if symbol in loaded_data:
            self.bars[symbol] = loaded_data[symbol]
            self._loaded_symbols.add(symbol)
            return True
        else:
            logger.warning(f"Failed to load bars for {symbol}")
            return False

    def _ensure_features_computed(self, symbol: str) -> bool:
        """Ensure features are computed for symbol.

        Args:
            symbol: Symbol to compute features for

        Returns:
            True if features are now available, False if computation failed
        """
        if symbol in self.features:
            return True

        if symbol not in self.bars:
            if not self._ensure_symbol_loaded(symbol):
                return False

        if not self.compute_features_fn:
            logger.warning(f"No feature computation function available for {symbol}")
            return False

        # Determine asset class
        asset_class = self._asset_classes.get(symbol, "equity")  # Default to equity

        logger.debug(f"Computing features for {symbol}")
        try:
            features_df = self.compute_features_fn(self.bars[symbol], symbol, asset_class)
            self.features[symbol] = features_df
            return True
        except Exception as e:
            logger.error(f"Error computing features for {symbol}: {e}")
            return False

    def set_asset_class(self, symbol: str, asset_class: str) -> None:
        """Set asset class for a symbol.

        Args:
            symbol: Symbol name
            asset_class: "equity" or "crypto"
        """
        self._asset_classes[symbol] = asset_class

    def get_bar(self, symbol: str, date: pd.Timestamp) -> Optional[Bar]:
        """Get bar for symbol at date (with lazy loading).

        Args:
            symbol: Symbol name
            date: Date timestamp

        Returns:
            Bar object or None if not found
        """
        if not self._ensure_symbol_loaded(symbol):
            return None

        return super().get_bar(symbol, date)

    def get_features(self, symbol: str, date: pd.Timestamp) -> Optional[FeatureRow]:
        """Get features for symbol at date (with lazy loading).

        Args:
            symbol: Symbol name
            date: Date timestamp

        Returns:
            FeatureRow object or None if not found
        """
        if not self._ensure_features_computed(symbol):
            logger.warning(f"FEATURES_NOT_COMPUTED: {symbol} - _ensure_features_computed returned False")
            return None

        result = super().get_features(symbol, date)
        if result is None and symbol in self.features:
            df = self.features[symbol]
            logger.warning(
                f"FEATURES_NOT_IN_INDEX: {symbol} on {date} - Index range: {df.index.min()} to {df.index.max()}, "
                f"len={len(df)}, date_type={type(date).__name__}, index_type={type(df.index[0]).__name__ if len(df) > 0 else 'empty'}"
            )
        return result

    def preload_universe(self, symbols: List[str], asset_classes: Optional[Dict[str, str]] = None) -> None:
        """Preload bars and features for a universe of symbols.

        Args:
            symbols: List of symbols to preload
            asset_classes: Optional dict mapping symbol -> asset_class
        """
        if asset_classes:
            self._asset_classes.update(asset_classes)

        self._preload_symbols(symbols)

        # Compute features for all loaded symbols
        for symbol in symbols:
            if symbol in self.bars:
                self._ensure_features_computed(symbol)

    def unload_symbol(self, symbol: str) -> None:
        """Unload a symbol to free memory.

        Args:
            symbol: Symbol to unload
        """
        if symbol in self.bars:
            del self.bars[symbol]
        if symbol in self.features:
            del self.features[symbol]
        if symbol in self._feature_cache:
            del self._feature_cache[symbol]
        self._loaded_symbols.discard(symbol)

    def get_loaded_symbols(self) -> List[str]:
        """Get list of currently loaded symbols.

        Returns:
            List of loaded symbol names
        """
        return list(self._loaded_symbols)

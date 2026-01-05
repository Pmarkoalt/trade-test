"""Real-time data feed for live trading."""

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set

import pandas as pd

from ..adapters.base_adapter import BaseAdapter
from ..indicators.feature_computer import compute_features
from ..logging.logger import get_logger
from ..models.bar import Bar
from ..models.features import FeatureRow
from ..models.signals import Signal
from ..portfolio.portfolio import Portfolio
from ..strategies.base.strategy_interface import StrategyInterface
from ..strategies.scoring import score_signals

logger = get_logger(__name__)


@dataclass
class RealTimeBar:
    """Real-time bar data from broker."""

    symbol: str
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_bar(self) -> Bar:
        """Convert to Bar model."""
        return Bar(
            date=self.timestamp,
            symbol=self.symbol,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
        )


class LiveDataFeed:
    """Real-time data feed for live trading.

    Maintains rolling windows of OHLCV data for each symbol,
    computes indicators incrementally, and generates signals in real-time.

    Example:
        >>> adapter = AlpacaAdapter(config)
        >>> feed = LiveDataFeed(adapter, strategies, portfolio)
        >>> feed.subscribe(['AAPL', 'GOOGL'])
        >>> feed.start()
        >>> # Feed runs in background, generating signals
        >>> signals = feed.get_pending_signals()
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        strategies: List[StrategyInterface],
        portfolio: Portfolio,
        min_bars_for_indicators: int = 200,
        update_interval_seconds: float = 1.0,
        signal_callback: Optional[Callable[[List[Signal]], None]] = None,
    ):
        """Initialize live data feed.

        Args:
            adapter: Broker adapter for market data
            strategies: List of strategies to generate signals for
            portfolio: Portfolio instance to track state
            min_bars_for_indicators: Minimum bars needed for indicator calculation
            update_interval_seconds: How often to check for new data (seconds)
            signal_callback: Optional callback function called when signals are generated
        """
        self.adapter = adapter
        self.strategies = strategies
        self.portfolio = portfolio
        self.min_bars_for_indicators = min_bars_for_indicators
        self.update_interval = update_interval_seconds
        self.signal_callback = signal_callback

        # Data storage: symbol -> deque of bars (rolling window)
        self._bar_windows: Dict[str, deque] = {}
        self._features_cache: Dict[str, pd.DataFrame] = {}  # symbol -> features DataFrame
        self._latest_prices: Dict[str, float] = {}  # symbol -> latest price

        # Signal queue
        self._pending_signals: List[Signal] = []
        self._signal_lock = threading.Lock()

        # Threading
        self._running = False
        self._feed_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Subscribed symbols
        self._subscribed_symbols: Set[str] = set()

        # Historical data for indicators (loaded once at start)
        self._historical_data: Dict[str, pd.DataFrame] = {}

    def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to market data for symbols.

        Args:
            symbols: List of symbols to subscribe to
        """
        if not self.adapter.is_connected():
            raise RuntimeError("Adapter not connected")

        # Subscribe via adapter
        self.adapter.subscribe_market_data(symbols)

        # Track subscribed symbols
        self._subscribed_symbols.update(symbols)

        # Initialize bar windows
        for symbol in symbols:
            if symbol not in self._bar_windows:
                self._bar_windows[symbol] = deque(maxlen=self.min_bars_for_indicators + 100)
                self._features_cache[symbol] = pd.DataFrame()

        logger.info(f"Subscribed to {len(symbols)} symbols: {symbols}")

    def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from market data.

        Args:
            symbols: List of symbols to unsubscribe from
        """
        if not self.adapter.is_connected():
            return

        self.adapter.unsubscribe_market_data(symbols)
        self._subscribed_symbols.difference_update(symbols)

        # Clean up data
        for symbol in symbols:
            self._bar_windows.pop(symbol, None)
            self._features_cache.pop(symbol, None)
            self._latest_prices.pop(symbol, None)

        logger.info(f"Unsubscribed from {len(symbols)} symbols")

    def load_historical_data(
        self, symbols: List[str], historical_data: Dict[str, pd.DataFrame], lookback_days: int = 252
    ) -> None:
        """Load historical OHLCV data for indicator calculation.

        Args:
            symbols: List of symbols
            historical_data: Dictionary mapping symbol to DataFrame with OHLCV data
            lookback_days: Number of days of historical data to use
        """
        end_date = pd.Timestamp.now().normalize()
        start_date = end_date - pd.Timedelta(days=lookback_days)

        for symbol in symbols:
            if symbol not in historical_data:
                logger.warning(f"No historical data for {symbol}")
                continue

            df = historical_data[symbol]

            # Filter by date range
            if "date" in df.columns:
                df = df.set_index("date")

            df = df.loc[start_date:end_date].copy()

            if len(df) < 60:  # Need at least 60 bars for indicators
                logger.warning(f"Insufficient historical data for {symbol}: {len(df)} bars")
                continue

            # Convert to bars and add to window
            for _, row in df.iterrows():
                bar = Bar(
                    date=row.name if isinstance(row.name, pd.Timestamp) else pd.Timestamp(row.name),
                    symbol=symbol,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
                self._bar_windows[symbol].append(bar)

            # Pre-compute features for historical data
            self._update_features(symbol)

            logger.info(f"Loaded {len(df)} historical bars for {symbol}")

        self._historical_data = historical_data

    def start(self) -> None:
        """Start the data feed thread."""
        if self._running:
            logger.warning("Feed already running")
            return

        if not self.adapter.is_connected():
            raise RuntimeError("Adapter not connected")

        self._running = True
        self._feed_thread = threading.Thread(target=self._feed_loop, daemon=True)
        self._feed_thread.start()
        logger.info("Live data feed started")

    def stop(self) -> None:
        """Stop the data feed thread."""
        if not self._running:
            return

        self._running = False
        if self._feed_thread:
            self._feed_thread.join(timeout=5.0)
        logger.info("Live data feed stopped")

    def _feed_loop(self) -> None:
        """Main feed loop running in background thread."""
        while self._running:
            try:
                self._update_prices()
                self._update_indicators()
                self._generate_signals()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in feed loop: {e}", exc_info=True)
                time.sleep(self.update_interval)

    def _update_prices(self) -> None:
        """Update latest prices from broker."""
        for symbol in self._subscribed_symbols:
            try:
                price = self.adapter.get_current_price(symbol)
                if price is not None:
                    self._latest_prices[symbol] = price

                    # Create a bar from current price (for intraday updates)
                    # Note: In a real system, you'd get actual bar data from websocket
                    # For now, we'll use the latest price as close
                    timestamp = pd.Timestamp.now()

                    # If we have previous bars, create a new bar
                    if symbol in self._bar_windows and len(self._bar_windows[symbol]) > 0:
                        last_bar = self._bar_windows[symbol][-1]

                        # Only update if it's a new bar (e.g., new minute/hour)
                        # For simplicity, we'll update on each price tick
                        # In production, you'd aggregate ticks into bars
                        new_bar = Bar(
                            date=timestamp,
                            symbol=symbol,
                            open=last_bar.close,  # Use previous close as open
                            high=max(last_bar.close, price),
                            low=min(last_bar.close, price),
                            close=price,
                            volume=0.0,  # Volume would come from actual bar data
                        )

                        # Only add if it's a new bar (not same timestamp)
                        if last_bar.date < timestamp:
                            self._bar_windows[symbol].append(new_bar)
            except Exception as e:
                logger.warning(f"Failed to update price for {symbol}: {e}")

    def _update_features(self, symbol: str) -> None:
        """Update features/indicators for a symbol."""
        if symbol not in self._bar_windows:
            return

        bars = list(self._bar_windows[symbol])
        if len(bars) < 60:  # Need at least 60 bars
            return

        # Convert bars to DataFrame
        df = pd.DataFrame(
            {
                "date": [bar.date for bar in bars],
                "open": [bar.open for bar in bars],
                "high": [bar.high for bar in bars],
                "low": [bar.low for bar in bars],
                "close": [bar.close for bar in bars],
                "volume": [bar.volume for bar in bars],
                "dollar_volume": [bar.dollar_volume for bar in bars],
            }
        )
        df = df.set_index("date")

        # Determine asset class from strategies
        asset_class = "equity"  # Default
        for strategy in self.strategies:
            if symbol in strategy.universe:
                asset_class = strategy.asset_class
                break

        # Compute features
        try:
            features_df = compute_features(
                df, symbol=symbol, asset_class=asset_class, use_cache=False, optimize_memory=False  # Don't cache in real-time
            )

            self._features_cache[symbol] = features_df
        except Exception as e:
            logger.warning(f"Failed to compute features for {symbol}: {e}")

    def _update_indicators(self) -> None:
        """Update indicators for all subscribed symbols."""
        for symbol in self._subscribed_symbols:
            self._update_features(symbol)

    def _generate_signals(self) -> None:
        """Generate signals for all strategies and symbols."""
        if not self._subscribed_symbols:
            return

        current_time = pd.Timestamp.now()
        all_signals = []

        # Get candidate returns for scoring
        candidate_returns = self._get_candidate_returns()
        portfolio_returns = self._get_portfolio_returns()

        for strategy in self.strategies:
            for symbol in strategy.universe:
                if symbol not in self._subscribed_symbols:
                    continue

                # Get latest features
                features_df = self._features_cache.get(symbol)
                if features_df is None or len(features_df) == 0:
                    continue

                # Get most recent feature row
                latest_features = features_df.iloc[-1]

                # Convert to FeatureRow
                try:
                    feature_row = FeatureRow(
                        date=latest_features.name,
                        symbol=symbol,
                        asset_class=strategy.asset_class,
                        close=float(latest_features["close"]),
                        open=float(latest_features["open"]),
                        high=float(latest_features["high"]),
                        low=float(latest_features["low"]),
                        ma20=float(latest_features["ma20"]) if pd.notna(latest_features["ma20"]) else None,
                        ma50=float(latest_features["ma50"]) if pd.notna(latest_features["ma50"]) else None,
                        ma200=float(latest_features["ma200"]) if pd.notna(latest_features["ma200"]) else None,
                        atr14=float(latest_features["atr14"]) if pd.notna(latest_features["atr14"]) else None,
                        roc60=float(latest_features["roc60"]) if pd.notna(latest_features["roc60"]) else None,
                        highest_close_20d=(
                            float(latest_features["highest_close_20d"])
                            if pd.notna(latest_features["highest_close_20d"])
                            else None
                        ),
                        highest_close_55d=(
                            float(latest_features["highest_close_55d"])
                            if pd.notna(latest_features["highest_close_55d"])
                            else None
                        ),
                        adv20=float(latest_features["adv20"]) if pd.notna(latest_features["adv20"]) else None,
                        returns_1d=float(latest_features["returns_1d"]) if pd.notna(latest_features["returns_1d"]) else None,
                        benchmark_roc60=(
                            float(latest_features["benchmark_roc60"]) if pd.notna(latest_features["benchmark_roc60"]) else None
                        ),
                        benchmark_returns_1d=(
                            float(latest_features["benchmark_returns_1d"])
                            if pd.notna(latest_features["benchmark_returns_1d"])
                            else None
                        ),
                    )
                except Exception as e:
                    logger.warning(f"Failed to create FeatureRow for {symbol}: {e}")
                    continue

                # Check if valid for entry
                if not feature_row.is_valid_for_entry():
                    continue

                # Estimate order notional for capacity check
                # Use current price for estimation
                current_price = self._latest_prices.get(symbol, feature_row.close)
                estimated_qty = 100  # Placeholder
                order_notional = current_price * estimated_qty

                # Generate signal
                signal = strategy.generate_signal(
                    symbol=symbol,
                    features=feature_row,
                    order_notional=order_notional,
                    diversification_bonus=0.0,  # Will be computed during scoring
                )

                if signal is not None:
                    signal.date = current_time  # Use current time for live signals
                    all_signals.append(signal)

        # Score signals
        if all_signals:

            def get_features_for_signal(sig: Signal) -> Optional[FeatureRow]:
                features_df = self._features_cache.get(sig.symbol)
                if features_df is None or len(features_df) == 0:
                    return None
                latest = features_df.iloc[-1]
                # Convert to FeatureRow (simplified)
                return FeatureRow(
                    date=latest.name,
                    symbol=sig.symbol,
                    asset_class=sig.asset_class,
                    close=float(latest["close"]),
                    open=float(latest["open"]),
                    high=float(latest["high"]),
                    low=float(latest["low"]),
                    ma20=float(latest["ma20"]) if pd.notna(latest["ma20"]) else None,
                    ma50=float(latest["ma50"]) if pd.notna(latest["ma50"]) else None,
                    atr14=float(latest["atr14"]) if pd.notna(latest["atr14"]) else None,
                    roc60=float(latest["roc60"]) if pd.notna(latest["roc60"]) else None,
                    benchmark_roc60=float(latest["benchmark_roc60"]) if pd.notna(latest["benchmark_roc60"]) else None,
                )

            score_signals(
                all_signals, get_features_for_signal, self.portfolio, candidate_returns, portfolio_returns, lookback=20
            )

            # Add to pending signals
            with self._signal_lock:
                self._pending_signals.extend(all_signals)
                # Sort by score (descending)
                self._pending_signals.sort(key=lambda s: s.score, reverse=True)

            # Call callback if provided
            if self.signal_callback:
                try:
                    self.signal_callback(all_signals)
                except Exception as e:
                    logger.error(f"Error in signal callback: {e}")

            logger.info(f"Generated {len(all_signals)} signals")

    def _get_candidate_returns(self) -> Dict[str, List[float]]:
        """Get returns for candidate symbols."""
        returns = {}
        for symbol in self._subscribed_symbols:
            features_df = self._features_cache.get(symbol)
            if features_df is not None and "returns_1d" in features_df.columns:
                returns_list = features_df["returns_1d"].dropna().tolist()
                if len(returns_list) >= 20:
                    returns[symbol] = returns_list[-20:]  # Last 20 returns
        return returns

    def _get_portfolio_returns(self) -> Dict[str, List[float]]:
        """Get returns for existing positions."""
        returns = {}
        for symbol, position in self.portfolio.positions.items():
            if symbol in self._features_cache:
                features_df = self._features_cache[symbol]
                if "returns_1d" in features_df.columns:
                    returns_list = features_df["returns_1d"].dropna().tolist()
                    if len(returns_list) >= 20:
                        returns[symbol] = returns_list[-20:]
        return returns

    def get_pending_signals(self, clear: bool = False) -> List[Signal]:
        """Get pending signals from queue.

        Args:
            clear: If True, clear the queue after returning signals

        Returns:
            List of pending signals (sorted by score, descending)
        """
        with self._signal_lock:
            signals = self._pending_signals.copy()
            if clear:
                self._pending_signals.clear()
            return signals

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol.

        Args:
            symbol: Symbol to get price for

        Returns:
            Latest price, or None if unavailable
        """
        return self._latest_prices.get(symbol)

    def get_latest_features(self, symbol: str) -> Optional[FeatureRow]:
        """Get latest features for a symbol.

        Args:
            symbol: Symbol to get features for

        Returns:
            Latest FeatureRow, or None if unavailable
        """
        features_df = self._features_cache.get(symbol)
        if features_df is None or len(features_df) == 0:
            return None

        latest = features_df.iloc[-1]
        return FeatureRow(
            date=latest.name,
            symbol=symbol,
            asset_class="equity",  # Would need to determine from strategy
            close=float(latest["close"]),
            open=float(latest["open"]),
            high=float(latest["high"]),
            low=float(latest["low"]),
            ma20=float(latest["ma20"]) if pd.notna(latest["ma20"]) else None,
            ma50=float(latest["ma50"]) if pd.notna(latest["ma50"]) else None,
            atr14=float(latest["atr14"]) if pd.notna(latest["atr14"]) else None,
            roc60=float(latest["roc60"]) if pd.notna(latest["roc60"]) else None,
        )

    def is_running(self) -> bool:
        """Check if feed is running."""
        return self._running

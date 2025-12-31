"""Daily event loop for backtesting with no lookahead."""

import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from ..data.validator import detect_missing_data, validate_ohlcv
from ..exceptions import BacktestError, DataError, DataNotFoundError, ExecutionError, PortfolioError, StrategyError
from ..execution.fill_simulator import reject_order_missing_data, simulate_fill
from ..logging import TradeEventType, get_logger, log_portfolio_snapshot, log_signal_generation, log_trade_event
from ..ml.predictor import MLPredictor
from ..models.bar import Bar
from ..models.features import FeatureRow
from ..models.market_data import MarketData
from ..models.orders import Fill, Order, OrderStatus
from ..models.positions import ExitReason, Position
from ..models.signals import Signal, SignalSide
from ..portfolio.portfolio import Portfolio
from ..portfolio.position_sizing import calculate_position_size
from ..strategies.base.strategy_interface import StrategyInterface
from ..strategies.queue import select_signals_from_queue
from ..strategies.scoring import score_signals

logger = get_logger(__name__)


class DailyEventLoop:
    """Event-driven daily loop for backtesting.

    Implements the daily sequence:
    1. Update data through day t close
    2. Generate signals at day t close
    3. Create orders for day t+1 open
    4. Execute orders at day t+1 open
    5. Update stops at day t+1 close
    6. Check exits at day t+1 close
    7. Execute exit orders at day t+2 open
    8. Update portfolio metrics
    9. Log daily state
    """

    def __init__(
        self,
        market_data: MarketData,
        portfolio: Portfolio,
        strategies: List[StrategyInterface],
        compute_features_fn: Callable[
            [pd.DataFrame, str, str, Optional[pd.Series], Optional[pd.Series], bool, bool], pd.DataFrame
        ],
        get_next_trading_day: Callable[[pd.Timestamp], pd.Timestamp],
        rng: Optional[np.random.Generator] = None,
        slippage_multiplier: float = 1.0,
        crash_dates: Optional[Set[pd.Timestamp]] = None,
        ml_predictor: Optional[MLPredictor] = None,
    ):
        """Initialize event loop.

        Args:
            market_data: MarketData container with bars and features
            portfolio: Portfolio state
            strategies: List of strategy objects (equity, crypto)
            compute_features_fn: Function to compute features for a symbol up to a date
            get_next_trading_day: Function to get next trading day from a date
            rng: Optional random number generator for reproducibility
            slippage_multiplier: Multiplier for base slippage (for stress tests)
            crash_dates: Set of dates on which to simulate flash crashes
            ml_predictor: Optional ML predictor for signal enhancement
        """
        self.market_data = market_data
        self.portfolio = portfolio
        self.strategies = strategies
        self.compute_features_fn = compute_features_fn
        self.get_next_trading_day = get_next_trading_day
        self.rng = rng
        self.slippage_multiplier = slippage_multiplier
        self.crash_dates = crash_dates or set()
        self.ml_predictor = ml_predictor

        # Track pending orders and exit orders
        self.pending_orders: List[Order] = []
        self.pending_exit_orders: List[tuple[Order, ExitReason]] = []

        # Track signal metadata for orders (order_id -> signal metadata)
        self.order_signal_metadata: Dict[str, Dict[str, Any]] = {}  # order_id -> {triggered_on, atr_mult, etc}

        # Track returns data for correlation calculations
        self.returns_data: Dict[str, List[float]] = {}

        # Track missing data
        self.missing_data_counts: Dict[str, int] = {}  # symbol -> consecutive missing days

        # Performance optimization: cache last computed date for each symbol
        self._last_computed_date: Dict[str, pd.Timestamp] = {}  # symbol -> last date features were computed
        self._cached_filtered_data: Dict[str, pd.DataFrame] = {}  # symbol -> cached filtered dataframe

    def process_day(self, date: pd.Timestamp) -> Dict[str, Any]:
        """Process a single day in the backtest.

        This implements the full daily sequence:
        - Day t: Update data, generate signals, create orders
        - Day t+1: Execute orders, update stops, check exits
        - Day t+2: Execute exit orders (if any)

        Args:
            date: Current date to process (day t)

        Returns:
            Dictionary with daily metrics and events
        """
        events = {
            "date": date,
            "signals_generated": [],
            "orders_created": [],
            "orders_executed": [],
            "exits_triggered": [],
            "exits_executed": [],
            "portfolio_state": {},
        }

        # Step 1: Update data through day t close
        self._update_data_through_date(date)

        # Step 2: Generate signals at day t close
        signals = self._generate_signals(date)
        events["signals_generated"] = [s.symbol for s in signals]

        # Log signal generation for each symbol checked
        for strategy in self.strategies:
            for symbol in strategy.universe:
                features = self.market_data.get_features(symbol, date)
                if features is None:
                    continue

                # Check if signal was generated for this symbol
                signal = next((s for s in signals if s.symbol == symbol), None)

                if signal:
                    log_signal_generation(
                        logger,
                        symbol=symbol,
                        asset_class=strategy.asset_class,
                        date=date,
                        signal_generated=True,
                        triggered_on=signal.triggered_on.value if signal.triggered_on else None,
                        clearance=signal.breakout_clearance,
                        score=signal.score,
                        breakout_strength=signal.breakout_strength,
                        momentum_strength=signal.momentum_strength,
                        diversification_bonus=signal.diversification_bonus,
                        capacity_passed=signal.capacity_passed,
                    )
                else:
                    # Check why signal wasn't generated
                    is_eligible, failure_reasons = strategy.check_eligibility(features)
                    breakout_type, clearance = strategy.check_entry_triggers(features)

                    log_signal_generation(
                        logger,
                        symbol=symbol,
                        asset_class=strategy.asset_class,
                        date=date,
                        signal_generated=False,
                        is_eligible=is_eligible,
                        failure_reasons=failure_reasons,
                        has_trigger=breakout_type is not None,
                    )

        # Step 3: Create orders for day t+1 open
        orders = self._create_orders(signals, date)
        events["orders_created"] = [o.symbol for o in orders]
        self.pending_orders.extend(orders)

        # Step 4: Execute orders at day t+1 open (if any pending)
        next_day = self.get_next_trading_day(date)
        executed_fills = self._execute_pending_orders(next_day)
        events["orders_executed"] = [f.symbol for f in executed_fills]

        # Step 5: Update stops and check exits at day t+1 close
        exit_signals = self._update_stops_and_check_exits(next_day)
        events["exits_triggered"] = [s[0] for s in exit_signals]

        # Create exit orders for day t+2 open
        exit_orders = self._create_exit_orders(exit_signals, next_day)
        # Convert orders to tuples with exit reasons
        exit_order_tuples = []
        for i, order in enumerate(exit_orders):
            exit_reason = exit_signals[i][1] if i < len(exit_signals) else ExitReason.HARD_STOP
            exit_order_tuples.append((order, exit_reason))
        self.pending_exit_orders.extend(exit_order_tuples)

        # Step 6: Execute exit orders at day t+2 open (if any pending from previous day)
        day_after_next = self.get_next_trading_day(next_day)
        exit_fills = self._execute_pending_exit_orders(day_after_next)
        events["exits_executed"] = [f.symbol for f in exit_fills]

        # Step 7: Update portfolio metrics at day t+1 close
        self._update_portfolio_metrics(next_day)

        # Step 8: Log daily state
        # Get current prices for positions
        current_prices: Dict[str, float] = {}
        for pos_key in self.portfolio.positions.keys():
            symbol_str = pos_key if isinstance(pos_key, str) else pos_key[1]  # Extract symbol from tuple
            bar = self.market_data.get_bar(symbol_str, next_day)
            if bar:
                current_prices[symbol_str] = bar.close

        # Serialize positions for correlation analysis
        positions_dict = {}
        # Create a copy of positions to avoid RuntimeError if positions are modified during iteration
        for pos_key, position in list(self.portfolio.positions.items()):
            if position.is_open():
                positions_dict[symbol] = {
                    "symbol": position.symbol,
                    "asset_class": position.asset_class,
                    "entry_date": position.entry_date,
                    "entry_price": position.entry_price,
                    "quantity": position.quantity,
                    "current_price": current_prices.get(symbol),
                }

        events["portfolio_state"] = {
            "equity": self.portfolio.equity,
            "cash": self.portfolio.cash,
            "open_positions": len(self.portfolio.positions),
            "realized_pnl": self.portfolio.realized_pnl,
            "unrealized_pnl": self.portfolio.unrealized_pnl,
            "gross_exposure": self.portfolio.gross_exposure,
            "risk_multiplier": self.portfolio.risk_multiplier,
            "positions": positions_dict,  # Add serialized positions
        }

        # Log portfolio snapshot
        log_portfolio_snapshot(
            logger,
            date=next_day,
            equity=self.portfolio.equity,
            cash=self.portfolio.cash,
            open_positions=len(self.portfolio.positions),
            realized_pnl=self.portfolio.realized_pnl,
            unrealized_pnl=self.portfolio.unrealized_pnl,
            gross_exposure=self.portfolio.gross_exposure,
            risk_multiplier=self.portfolio.risk_multiplier,
            portfolio_vol_20d=self.portfolio.portfolio_vol_20d,
            median_vol_252d=self.portfolio.median_vol_252d,
            avg_pairwise_corr=self.portfolio.avg_pairwise_corr,
            total_trades=self.portfolio.total_trades,
        )

        return events

    def _update_data_through_date(self, date: pd.Timestamp) -> None:
        """Update all market data up to and including day t close.

        For each symbol:
        1. Load/update bars up to date
        2. Compute indicators using data <= date (no lookahead)
        3. Update features in market_data

        Optimized to:
        - Cache filtered dataframes to avoid repeated filtering
        - Only compute features for new dates
        - Use vectorized pandas operations for updates
        """
        # Update features for all symbols in universe
        all_symbols: Set[str] = set()
        for strategy in self.strategies:
            all_symbols.update(strategy.universe)

        for symbol in all_symbols:
            if symbol not in self.market_data.bars:
                continue

            bars_df = self.market_data.bars[symbol]

            # Optimization: Use cached filtered data if available and date hasn't changed
            # For simplicity, we'll still filter each day but cache the result
            # More advanced: only recompute if date > last_computed_date
            if symbol in self._cached_filtered_data:
                cached_df = self._cached_filtered_data[symbol]
                # Check if we can extend the cache
                if len(cached_df) > 0 and cached_df.index[-1] < date:
                    # Extend cache with new data
                    new_data = bars_df[(bars_df.index > cached_df.index[-1]) & (bars_df.index <= date)]
                    if len(new_data) > 0:
                        available_data = pd.concat([cached_df, new_data])
                        self._cached_filtered_data[symbol] = available_data
                    else:
                        available_data = cached_df
                elif len(cached_df) > 0 and date in cached_df.index:
                    available_data = cached_df
                else:
                    # Need to refilter
                    available_data = bars_df[bars_df.index <= date]
                    self._cached_filtered_data[symbol] = available_data
            else:
                # First time: filter and cache
                available_data = bars_df[bars_df.index <= date]
                self._cached_filtered_data[symbol] = available_data

            if len(available_data) == 0:
                continue

            # Check for missing data - if current date is not in available data
            if date not in available_data.index:
                # Increment consecutive missing count
                self.missing_data_counts[symbol] = self.missing_data_counts.get(symbol, 0) + 1
                consecutive_count = self.missing_data_counts[symbol]

                # Handle based on consecutive count
                if consecutive_count == 1:
                    # Single day missing: log warning, skip signal generation
                    logger.warning(f"MISSING_DATA_1DAY: {symbol} {date}")
                else:
                    # 2+ consecutive days: mark unhealthy, force exit if position exists
                    logger.error(f"DATA_UNHEALTHY: {symbol}, missing {consecutive_count} consecutive days")
                    # Get all missing dates in the gap for proper handling
                    missing_info = detect_missing_data(available_data, symbol, asset_class=self._get_asset_class(symbol))
                    consecutive_dates = self._find_consecutive_missing_dates(date, missing_info, available_data)
                    if consecutive_dates:
                        self._handle_missing_data(symbol, consecutive_dates, date)
                    else:
                        # Fallback: use current date
                        self._handle_missing_data(symbol, [date], date)
                continue
            else:
                # Data is available - reset missing count
                if symbol in self.missing_data_counts:
                    del self.missing_data_counts[symbol]

            # Optimization: Only compute features if we haven't computed for this date yet
            last_computed = self._last_computed_date.get(symbol)
            if last_computed is not None and date <= last_computed:
                # Features already computed for this date, skip
                continue

            # Compute features using data <= date only
            try:
                asset_class = self._get_asset_class(symbol)
                features_df = self.compute_features_fn(available_data, symbol, asset_class, None, None, False, False)

                # Update market_data.features using vectorized operations
                if symbol not in self.market_data.features:
                    self.market_data.features[symbol] = features_df
                else:
                    # Optimization: Use pandas operations instead of iterrows
                    # Only update rows up to current date
                    existing_features = self.market_data.features[symbol]
                    new_rows = features_df[features_df.index <= date]

                    # Use pandas combine_first and update for better performance
                    # This handles both updates and new rows efficiently
                    # First, update existing rows
                    for idx in new_rows.index:
                        if idx in existing_features.index:
                            existing_features.loc[idx] = new_rows.loc[idx]

                    # Then, add new rows that don't exist
                    new_index = new_rows.index[~new_rows.index.isin(existing_features.index)]
                    if len(new_index) > 0:
                        existing_features = pd.concat([existing_features, new_rows.loc[new_index]]).sort_index()

                    self.market_data.features[symbol] = existing_features

                # Update last computed date
                self._last_computed_date[symbol] = date

            except (ValueError, KeyError, IndexError) as e:
                logger.warning(f"Error computing features for {symbol} at {date}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error computing features for {symbol} at {date}: {e}", exc_info=True)
                raise BacktestError(
                    f"Failed to compute features for {symbol} at {date}: {e}", date=str(date), step="feature_computation"
                ) from e

    def _generate_signals(self, date: pd.Timestamp) -> List[Signal]:
        """Generate entry signals for all strategies at day t close.

        Optimized to batch process symbols and reduce redundant feature lookups.

        Args:
            date: Date at which to generate signals

        Returns:
            List of valid signals
        """
        all_signals = []

        # Batch process: collect all symbols first, then process
        for strategy in self.strategies:
            # Pre-filter symbols that might have data (optimization)
            symbols_to_check = [symbol for symbol in strategy.universe if symbol in self.market_data.bars]

            for symbol in symbols_to_check:
                # Get features for this symbol at date
                features = self.market_data.get_features(symbol, date)
                if features is None:
                    continue

                # Check if features are valid for entry
                if not features.is_valid_for_entry():
                    continue

                # Estimate order notional for capacity check
                # We need to estimate position size first
                estimated_qty = self._estimate_position_size(strategy, features, self.portfolio)
                order_notional = features.close * estimated_qty

                # Generate signal
                signal = strategy.generate_signal(
                    symbol=symbol,
                    features=features,
                    order_notional=order_notional,
                    diversification_bonus=0.0,  # Will be computed during scoring
                )

                if signal is not None and signal.is_valid():
                    all_signals.append(signal)
                elif signal is not None:
                    # Signal generated but invalid - log why
                    log_signal_generation(
                        logger,
                        symbol=symbol,
                        asset_class=strategy.asset_class,
                        date=date,
                        signal_generated=False,
                        reason="Signal generated but invalid",
                        eligibility_failures=signal.eligibility_failures if hasattr(signal, "eligibility_failures") else [],
                    )

        # Score signals (batch operation)
        if all_signals:
            self._score_signals(all_signals, date)

        return all_signals

    def _score_signals(self, signals: List[Signal], date: pd.Timestamp) -> None:
        """Score signals using breakout strength, momentum, and diversification.

        Optionally enhances scores with ML predictions if ML predictor is configured.

        Args:
            signals: List of signals to score
            date: Current date
        """

        # Get features function for scoring
        def get_features(signal: Signal) -> Optional[FeatureRow]:
            return self.market_data.get_features(signal.symbol, signal.date)

        # Get returns data for correlation calculations
        candidate_returns = self._get_candidate_returns(signals, date)
        portfolio_returns = self._get_portfolio_returns(date)

        # Score signals
        score_signals(
            signals=signals,
            get_features=get_features,
            portfolio=self.portfolio,
            candidate_returns=candidate_returns,
            portfolio_returns=portfolio_returns,
            lookback=20,
        )

        # Apply ML enhancement if ML predictor is configured
        if self.ml_predictor is not None and signals:
            # Get ML config from first strategy (assume same config for all strategies)
            strategy = self.strategies[0] if self.strategies else None
            if strategy and strategy.config.ml.enabled:
                ml_weight = strategy.config.ml.ml_weight

                # Enhance signal scores with ML predictions
                self.ml_predictor.enhance_signal_scores(signals=signals, get_features=get_features, ml_weight=ml_weight)

                # If in filter mode, remove signals below threshold
                if self.ml_predictor.prediction_mode == "filter":
                    # Filter signals (this modifies the list in-place conceptually)
                    # Since enhance_signal_scores sets score to 0 for filtered signals,
                    # we'll filter them out in signal selection
                    pass  # Filtering happens implicitly via score = 0

    def _create_orders(self, signals: List[Signal], date: pd.Timestamp) -> List[Order]:
        """Create orders from signals for execution at day t+1 open.

        Args:
            signals: List of signals to create orders from
            date: Signal date (day t)

        Returns:
            List of orders created
        """
        if not signals:
            return []

        # Select signals from queue (apply constraints)
        selected_signals = self._select_signals_from_queue(signals)

        orders = []
        next_day = self.get_next_trading_day(date)

        for signal in selected_signals:
            strategy = self._get_strategy_for_signal(signal)
            if strategy is None:
                continue

            # Calculate exact position size
            qty = calculate_position_size(
                equity=self.portfolio.equity,
                risk_pct=strategy.config.risk.risk_per_trade * self.portfolio.risk_multiplier,
                entry_price=signal.entry_price,
                stop_price=signal.stop_price,
                max_position_notional=self.portfolio.equity * strategy.config.risk.max_position_notional,
                available_cash=self.portfolio.cash,
                risk_multiplier=self.portfolio.risk_multiplier,
                max_exposure=self.portfolio.equity * (strategy.config.risk.max_exposure if hasattr(strategy.config.risk, 'max_exposure') and strategy.config.risk.max_exposure is not None else 0.8),
            )

            if qty < 1:
                continue  # Cannot afford position

            # Get expected fill price (next open, estimated)
            next_bar = self.market_data.get_bar(signal.symbol, next_day)
            if next_bar is None:
                # Missing data - skip this order
                continue

            expected_fill_price = next_bar.open

            # Create order
            order = Order(
                order_id=str(uuid.uuid4()),
                symbol=signal.symbol,
                asset_class=signal.asset_class,
                date=date,
                execution_date=next_day,
                side=SignalSide.BUY,
                quantity=qty,
                signal_date=date,
                expected_fill_price=expected_fill_price,
                stop_price=signal.stop_price,
                status=OrderStatus.PENDING,
            )

            # Store signal metadata for this order
            self.order_signal_metadata[order.order_id] = {
                "triggered_on": signal.triggered_on,
                "atr_mult": signal.atr_mult,
                "adv20_at_entry": signal.adv20,
            }

            orders.append(order)

            # Reserve cash (estimate with buffer)
            estimated_cost = expected_fill_price * qty * 1.01  # 1% buffer
            self.portfolio.cash -= estimated_cost

        return orders

    def _select_signals_from_queue(self, signals: List[Signal]) -> List[Signal]:
        """Select signals from queue applying all constraints.

        Args:
            signals: List of scored signals

        Returns:
            List of selected signals
        """
        if not signals:
            return []

        # Get strategy config for constraints
        strategy = self.strategies[0]  # Use first strategy's config (assume same constraints)
        config = strategy.config

        # Get returns data
        date = signals[0].date
        candidate_returns = self._get_candidate_returns(signals, date)
        portfolio_returns = self._get_portfolio_returns(date)

        # Select from queue
        selected = select_signals_from_queue(
            signals=signals,
            portfolio=self.portfolio,
            max_positions=config.risk.max_positions,
            max_exposure=config.risk.max_exposure,
            risk_per_trade=config.risk.risk_per_trade,
            max_position_notional=config.risk.max_position_notional,
            candidate_returns=candidate_returns,
            portfolio_returns=portfolio_returns,
            lookback=20,
        )

        return selected

    def _execute_pending_orders(self, date: pd.Timestamp) -> List[Fill]:
        """Execute pending orders at day t+1 open.

        Args:
            date: Execution date (day t+1)

        Returns:
            List of fills from executed orders
        """
        fills = []
        orders_to_remove = []

        for order in self.pending_orders:
            if order.execution_date != date:
                continue  # Not ready to execute yet

            if order.status != OrderStatus.PENDING:
                orders_to_remove.append(order)
                continue

            # Get open bar
            open_bar = self.market_data.get_bar(order.symbol, date)
            if open_bar is None:
                # Missing data: reject order
                order.status = OrderStatus.REJECTED
                order.rejection_reason = "MISSING_DATA"
                orders_to_remove.append(order)
                continue

            # Get features for slippage calculation
            features = self.market_data.get_features(order.symbol, date)
            if features is None or features.atr14 is None:
                order.status = OrderStatus.REJECTED
                order.rejection_reason = "MISSING_FEATURES"
                orders_to_remove.append(order)
                continue

            # Get ATR14 history for volatility multiplier
            atr14_history = self.market_data.features[order.symbol]["atr14"]

            # Get benchmark bars for stress calculation
            benchmark_symbol = "SPY" if order.asset_class == "equity" else "BTC"
            benchmark_bars = self.market_data.benchmarks.get(benchmark_symbol)
            if benchmark_bars is None:
                benchmark_bars = pd.DataFrame()

            # Base slippage
            base_slippage_bps = 8.0 if order.asset_class == "equity" else 10.0

            # Check if this is a crash date - apply 5x multiplier for flash crashes
            is_crash_date = date in self.crash_dates
            effective_multiplier = 5.0 if is_crash_date else self.slippage_multiplier
            effective_base_slippage = base_slippage_bps * effective_multiplier

            # Simulate fill
            try:
                if features.adv20 is None or features.atr14 is None:
                    continue  # Skip if required features are missing
                fill = simulate_fill(
                    order=order,
                    open_bar=open_bar,
                    atr14=features.atr14,
                    atr14_history=atr14_history,
                    adv20=features.adv20,
                    benchmark_bars=benchmark_bars,
                    base_slippage_bps=effective_base_slippage,
                    rng=self.rng,
                )

                # Update order status
                order.status = OrderStatus.FILLED

                # Adjust cash (actual cost vs reserved)
                actual_cost = fill.fill_price * fill.quantity + fill.total_cost
                reserved_cost = order.expected_fill_price * order.quantity * 1.01
                cash_adjustment = reserved_cost - actual_cost
                self.portfolio.cash += cash_adjustment

                # Create position
                strategy = self._get_strategy_for_signal_symbol(order.symbol)
                if strategy:
                    # Get signal metadata from order
                    signal_metadata = self.order_signal_metadata.get(order.order_id, {})
                    atr_mult = signal_metadata.get("atr_mult", strategy.config.exit.hard_stop_atr_mult)
                    from ..models.signals import BreakoutType
                    triggered_on = signal_metadata.get("triggered_on")
                    if triggered_on is not None and not isinstance(triggered_on, BreakoutType):
                        triggered_on = None
                    # process_fill expects BreakoutType or None, but we need to handle None case
                    adv20_at_entry = signal_metadata.get("adv20_at_entry", features.adv20)
                    if adv20_at_entry is None:
                        adv20_at_entry = 0.0

                    # Provide default BreakoutType if None
                    from ..models.signals import BreakoutType
                    final_triggered_on = triggered_on if triggered_on is not None else BreakoutType.FAST_20D
                    position = self.portfolio.process_fill(
                        fill=fill,
                        stop_price=order.stop_price,
                        atr_mult=atr_mult,
                        triggered_on=final_triggered_on,
                        adv20_at_entry=adv20_at_entry,
                    )

                    # Log trade entry
                    log_trade_event(
                        logger,
                        TradeEventType.ENTRY,
                        symbol=order.symbol,
                        asset_class=order.asset_class,
                        date=fill.date,
                        entry_price=fill.fill_price,
                        quantity=fill.quantity,
                        stop_price=order.stop_price,
                        triggered_on=triggered_on,
                        slippage_bps=fill.slippage_bps,
                        fee_bps=fill.fee_bps,
                        total_cost=fill.total_cost,
                    )

                    # Clean up metadata
                    if order.order_id in self.order_signal_metadata:
                        del self.order_signal_metadata[order.order_id]

                fills.append(fill)
                orders_to_remove.append(order)

            except Exception as e:
                logger.error(f"Error executing order {order.order_id}: {e}")
                order.status = OrderStatus.REJECTED
                order.rejection_reason = f"EXECUTION_ERROR: {str(e)}"

                # Log rejected order
                log_trade_event(
                    logger,
                    TradeEventType.REJECTED,
                    symbol=order.symbol,
                    asset_class=order.asset_class,
                    date=date,
                    reason=order.rejection_reason,
                )

                orders_to_remove.append(order)

        # Remove executed/rejected orders
        for order in orders_to_remove:
            if order in self.pending_orders:
                self.pending_orders.remove(order)

        return fills

    def _update_stops_and_check_exits(self, date: pd.Timestamp) -> List[tuple[str, ExitReason]]:
        """Update trailing stops and check exit signals at day t+1 close.

        Args:
            date: Date at which to check (day t+1 close)

        Returns:
            List of (symbol, exit_reason) tuples for positions that should exit
        """
        # Get current prices and features for all positions
        current_prices: Dict[str, float] = {}
        features_data: Dict[str, Dict[str, Any]] = {}

        # Create a copy of positions to avoid RuntimeError if positions are modified during iteration
        for pos_key, position in list(self.portfolio.positions.items()):
            if not position.is_open():
                continue
            symbol = pos_key if isinstance(pos_key, str) else pos_key[1]  # Extract symbol from tuple

            bar = self.market_data.get_bar(symbol, date)
            if bar is None:
                # Missing data - handle separately
                self._handle_missing_data_for_position(symbol, date)
                continue

            current_prices[symbol] = bar.close

            features = self.market_data.get_features(symbol, date)
            if features:
                features_data[symbol] = {"ma20": features.ma20, "ma50": features.ma50, "atr14": features.atr14}

        # Get strategy config for exit mode
        strategy = self.strategies[0] if self.strategies else None
        exit_mode = strategy.config.exit.mode if strategy else "ma_cross"
        exit_ma = strategy.config.exit.exit_ma if strategy else 20

        # Update stops and check exits
        exit_signals = self.portfolio.update_stops(
            current_prices=current_prices, features_data=features_data, exit_mode=exit_mode, exit_ma=exit_ma
        )

        # On crash dates, force all stops to trigger (flash crash simulation)
        if date in self.crash_dates:
            # Force exit all positions at stops
            forced_exits: List[Tuple[str, ExitReason]] = []
            # Create a copy of positions to avoid RuntimeError if positions are modified during iteration
            for pos_key, position in list(self.portfolio.positions.items()):
                symbol = pos_key if isinstance(pos_key, str) else pos_key[1]  # Extract symbol from tuple
                if position.is_open() and symbol not in [s[0] for s in exit_signals]:
                    # Force stop at worst possible price
                    forced_exits.append((symbol, ExitReason.HARD_STOP))
            exit_signals.extend(forced_exits)

        return exit_signals

    def _create_exit_orders(self, exit_signals: List[tuple[str, ExitReason]], date: pd.Timestamp) -> List[Order]:
        """Create exit orders for positions that should be closed.

        Args:
            exit_signals: List of (symbol, exit_reason) tuples
            date: Date when exit was triggered (day t+1 close)

        Returns:
            List of exit orders
        """
        orders = []
        next_day = self.get_next_trading_day(date)

        for symbol, exit_reason in exit_signals:
            position = self.portfolio.get_position(symbol)
            if position is None or not position.is_open():
                continue

            # Create exit order
            order = Order(
                order_id=str(uuid.uuid4()),
                symbol=symbol,
                asset_class=position.asset_class,
                date=date,
                execution_date=next_day,
                side=SignalSide.SELL,
                quantity=position.quantity,
                signal_date=date,
                expected_fill_price=0.0,  # Will be set at execution
                stop_price=0.0,  # Not used for exits
                status=OrderStatus.PENDING,
            )

            # Store exit reason in order (we'll need to track this)
            # For now, we'll pass it through the execution

            orders.append((order, exit_reason))

        # Store exit orders with their reasons
        self.pending_exit_orders.extend(orders)
        return [o[0] for o in orders]

    def _execute_pending_exit_orders(self, date: pd.Timestamp) -> List[Fill]:
        """Execute pending exit orders at day t+2 open.

        Args:
            date: Execution date (day t+2)

        Returns:
            List of fills from executed exit orders
        """
        fills = []
        orders_to_remove = []

        for order_tuple in self.pending_exit_orders:
            order, exit_reason = order_tuple

            if order.execution_date != date:
                continue  # Not ready to execute yet

            if order.status != OrderStatus.PENDING:
                orders_to_remove.append(order_tuple)
                continue

            # Get open bar
            open_bar = self.market_data.get_bar(order.symbol, date)
            if open_bar is None:
                # Missing data: reject order
                order.status = OrderStatus.REJECTED
                order.rejection_reason = "MISSING_DATA"
                orders_to_remove.append(order_tuple)
                continue

            # Get features for slippage calculation
            features = self.market_data.get_features(order.symbol, date)
            if features is None or features.atr14 is None:
                order.status = OrderStatus.REJECTED
                order.rejection_reason = "MISSING_FEATURES"
                orders_to_remove.append(order_tuple)
                continue

            # Get ATR14 history
            atr14_history = self.market_data.features[order.symbol]["atr14"]

            # Get benchmark bars
            benchmark_symbol = "SPY" if order.asset_class == "equity" else "BTC"
            benchmark_bars = self.market_data.benchmarks.get(benchmark_symbol)
            if benchmark_bars is None:
                benchmark_bars = pd.DataFrame()

            # Base slippage
            base_slippage_bps = 8.0 if order.asset_class == "equity" else 10.0

            # Check if this is a crash date - apply 5x multiplier for flash crashes
            is_crash_date = date in self.crash_dates
            effective_multiplier = 5.0 if is_crash_date else self.slippage_multiplier
            effective_base_slippage = base_slippage_bps * effective_multiplier

            # Simulate fill
            try:
                if features.adv20 is None or features.atr14 is None:
                    continue  # Skip if required features are missing
                fill = simulate_fill(
                    order=order,
                    open_bar=open_bar,
                    atr14=features.atr14,
                    atr14_history=atr14_history,
                    adv20=features.adv20,
                    benchmark_bars=benchmark_bars,
                    base_slippage_bps=effective_base_slippage,
                    rng=self.rng,
                )

                # Update order status
                order.status = OrderStatus.FILLED

                # On crash dates, force stops at worst possible price (for exits only)
                if is_crash_date and exit_reason == ExitReason.HARD_STOP:
                    # Force exit at worst price (slippage already applied above, but make it worse)
                    if order.side == SignalSide.SELL:
                        # SELL at worst (lowest) price - use the low of the day
                        worst_price = min(fill.fill_price, open_bar.low)
                        fill.fill_price = worst_price
                    else:
                        # BUY at worst (highest) price - use the high of the day
                        worst_price = max(fill.fill_price, open_bar.high)
                        fill.fill_price = worst_price

                # Close position
                closed_position = self.portfolio.close_position(symbol=order.symbol, exit_fill=fill, exit_reason=exit_reason)

                if closed_position:
                    # Calculate R-multiple
                    r_multiple = None
                    if closed_position.initial_stop_price and closed_position.initial_stop_price > 0:
                        risk = closed_position.entry_price - closed_position.initial_stop_price
                        if risk > 0:
                            reward = closed_position.realized_pnl / (closed_position.quantity * risk)
                            r_multiple = reward

                    # Log trade exit
                    event_type = TradeEventType.STOP_HIT if exit_reason == ExitReason.HARD_STOP else TradeEventType.EXIT
                    exit_reason_str = exit_reason.value if isinstance(exit_reason, ExitReason) else str(exit_reason)
                    log_trade_event(
                        logger,
                        event_type,
                        symbol=order.symbol,
                        asset_class=order.asset_class,
                        date=fill.date,
                        exit_price=fill.fill_price,
                        exit_reason=exit_reason_str,
                        realized_pnl=closed_position.realized_pnl,
                        r_multiple=r_multiple,
                        slippage_bps=fill.slippage_bps,
                        fee_bps=fill.fee_bps,
                        total_cost=fill.total_cost,
                    )

                    fills.append(fill)

                orders_to_remove.append(order_tuple)

            except Exception as e:
                logger.error(f"Error executing exit order {order.order_id}: {e}")
                order.status = OrderStatus.REJECTED
                order.rejection_reason = f"EXECUTION_ERROR: {str(e)}"
                orders_to_remove.append(order_tuple)

        # Remove executed/rejected orders
        for order_tuple in orders_to_remove:
            if order_tuple in self.pending_exit_orders:
                self.pending_exit_orders.remove(order_tuple)

        return fills

    def _update_portfolio_metrics(self, date: pd.Timestamp) -> None:
        """Update portfolio metrics at day t+1 close.

        Args:
            date: Date at which to update (day t+1 close)
        """
        # Get current prices for all positions
        current_prices: Dict[str, float] = {}
        for pos_key in self.portfolio.positions.keys():
            symbol = pos_key if isinstance(pos_key, str) else pos_key[1]  # Extract symbol from tuple
            bar = self.market_data.get_bar(symbol, date)
            if bar:
                current_prices[symbol] = bar.close

        # Update equity
        self.portfolio.update_equity(current_prices)

        # Update volatility scaling
        self.portfolio.update_volatility_scaling()

        # Update correlation metrics
        self.portfolio.update_correlation_metrics(returns_data=self.returns_data, lookback=20)

        # Append to equity curve
        self.portfolio.append_daily_metrics()

    def _estimate_position_size(self, strategy: StrategyInterface, features: FeatureRow, portfolio: Portfolio) -> int:
        """Estimate position size for capacity check.

        Args:
            strategy: Strategy object
            features: FeatureRow with entry and stop prices
            portfolio: Portfolio state

        Returns:
            Estimated quantity
        """
        from ..portfolio.position_sizing import estimate_position_size

        if features.close is None or features.atr14 is None:
            return 0
        return estimate_position_size(
            equity=portfolio.equity,
            risk_pct=strategy.config.risk.risk_per_trade * portfolio.risk_multiplier,
            entry_price=features.close,
            stop_price=features.close - (strategy.config.exit.hard_stop_atr_mult * features.atr14),
            max_position_notional=portfolio.equity * strategy.config.risk.max_position_notional,
            risk_multiplier=portfolio.risk_multiplier,
        )

    def _get_asset_class(self, symbol: str) -> str:
        """Get asset class for a symbol.

        Args:
            symbol: Symbol name

        Returns:
            "equity" or "crypto"
        """
        for strategy in self.strategies:
            if symbol in strategy.universe:
                return strategy.asset_class
        return "equity"  # Default

    def _get_strategy_for_signal(self, signal: Signal) -> Optional[StrategyInterface]:
        """Get strategy object for a signal.

        Args:
            signal: Signal object

        Returns:
            Strategy object or None
        """
        for strategy in self.strategies:
            if signal.symbol in strategy.universe and signal.asset_class == strategy.asset_class:
                return strategy
        return None

    def _get_strategy_for_signal_symbol(self, symbol: str) -> Optional[StrategyInterface]:
        """Get strategy object for a symbol.

        Args:
            symbol: Symbol name

        Returns:
            Strategy object or None
        """
        for strategy in self.strategies:
            if symbol in strategy.universe:
                return strategy
        return None

    def _get_candidate_returns(self, signals: List[Signal], date: pd.Timestamp) -> Dict[str, List[float]]:
        """Get returns data for candidate symbols.

        Args:
            signals: List of signals
            date: Current date

        Returns:
            Dictionary mapping symbol to list of returns
        """
        candidate_returns = {}

        for signal in signals:
            if signal.symbol in candidate_returns:
                continue

            if signal.symbol in self.returns_data:
                candidate_returns[signal.symbol] = self.returns_data[signal.symbol]
            else:
                # Compute returns from bars
                if signal.symbol in self.market_data.bars:
                    bars_df = self.market_data.bars[signal.symbol]
                    available_data = bars_df[bars_df.index <= date]
                    if len(available_data) > 1:
                        returns = (available_data["close"] / available_data["close"].shift(1) - 1).tolist()
                        candidate_returns[signal.symbol] = returns
                        self.returns_data[signal.symbol] = returns

        return candidate_returns

    def _get_portfolio_returns(self, date: pd.Timestamp) -> Dict[str, List[float]]:
        """Get returns data for existing portfolio positions.

        Args:
            date: Current date

        Returns:
            Dictionary mapping symbol to list of returns
        """
        portfolio_returns: Dict[str, List[float]] = {}

        for pos_key in self.portfolio.positions.keys():
            symbol = pos_key if isinstance(pos_key, str) else pos_key[1]  # Extract symbol from tuple
            if symbol in self.returns_data:
                portfolio_returns[symbol] = self.returns_data[symbol]
            else:
                # Compute returns from bars
                if symbol in self.market_data.bars:
                    bars_df = self.market_data.bars[symbol]
                    available_data = bars_df[bars_df.index <= date]
                    if len(available_data) > 1:
                        returns = (available_data["close"] / available_data["close"].shift(1) - 1).tolist()
                        portfolio_returns[symbol] = [float(r) for r in returns]
                        self.returns_data[symbol] = [float(r) for r in returns]

        return portfolio_returns

    def _handle_missing_data(self, symbol: str, missing_dates: List[pd.Timestamp], date: pd.Timestamp) -> None:
        """Handle missing data for a symbol.

        Args:
            symbol: Symbol with missing data
            missing_dates: List of missing dates
            date: Current date
        """
        consecutive_count = len(missing_dates)

        # Update missing count to reflect the actual consecutive gap length
        # Use the maximum of current count and consecutive_count to handle incremental detection
        # This ensures we don't overwrite a higher count with a lower one
        current_count = self.missing_data_counts.get(symbol, 0)
        final_count = max(current_count, consecutive_count)
        self.missing_data_counts[symbol] = final_count

        # Use final_count to determine if we should close position
        # This handles the case where _find_consecutive_missing_dates doesn't find the full gap
        if final_count >= 2:
            # 2+ consecutive days: mark as unhealthy, close position if exists
            logger.error(f"DATA_UNHEALTHY: {symbol}, missing {final_count} consecutive days")

            # Close position if exists
            position = self.portfolio.get_position(symbol)
            if position and position.is_open():
                # Try to exit at next available data
                next_date = self.get_next_trading_day(date)
                next_bar = self.market_data.get_bar(symbol, next_date)
                if next_bar:
                    # Create exit order for next day
                    # For SELL orders (exits), stop_price validation is different
                    # Use expected_fill_price - 1 to ensure stop < expected_fill_price for validation
                    try:
                        exit_order = Order(
                            order_id=str(uuid.uuid4()),
                            symbol=symbol,
                            asset_class=position.asset_class,
                            date=date,
                            execution_date=next_date,
                            side=SignalSide.SELL,
                            quantity=position.quantity,
                            signal_date=date,
                            expected_fill_price=next_bar.open,
                            stop_price=max(0.01, next_bar.open - 1.0),  # Ensure stop < expected_fill_price
                            status=OrderStatus.PENDING,
                        )
                        self.pending_exit_orders.append((exit_order, ExitReason.DATA_MISSING))
                    except Exception as e:
                        # If order creation fails, force exit at last known close
                        logger.warning(f"Failed to create exit order for {symbol}: {e}, forcing immediate exit")
                        last_bar = self.market_data.bars[symbol].iloc[-1]
                        exit_price = last_bar["close"]
                        from ..models.orders import Fill

                        exit_fill = Fill(
                            fill_id=str(uuid.uuid4()),
                            order_id=str(uuid.uuid4()),
                            symbol=symbol,
                            asset_class=position.asset_class,
                            date=date,
                            side=SignalSide.SELL,
                            quantity=position.quantity,
                            fill_price=exit_price,
                            open_price=exit_price,
                            slippage_bps=0.0,
                            fee_bps=0.0,
                            total_cost=0.0,
                            vol_mult=1.0,
                            size_penalty=1.0,
                            weekend_penalty=1.0,
                            stress_mult=1.0,
                            notional=exit_price * position.quantity,
                        )
                        self.portfolio.close_position(symbol, exit_fill, ExitReason.DATA_MISSING)
                else:
                    # Force exit at last known close
                    last_bar = self.market_data.bars[symbol].iloc[-1]
                    exit_price = last_bar["close"]
                    # Create a fill manually
                    from ..models.orders import Fill

                    exit_fill = Fill(
                        fill_id=str(uuid.uuid4()),
                        order_id=str(uuid.uuid4()),
                        symbol=symbol,
                        asset_class=position.asset_class,
                        date=date,
                        side=SignalSide.SELL,
                        quantity=position.quantity,
                        fill_price=exit_price,
                        open_price=exit_price,
                        slippage_bps=0.0,
                        fee_bps=0.0,
                        total_cost=0.0,
                        vol_mult=1.0,
                        size_penalty=1.0,
                        weekend_penalty=1.0,
                        stress_mult=1.0,
                        notional=exit_price * position.quantity,
                    )
                    self.portfolio.close_position(symbol, exit_fill, ExitReason.DATA_MISSING)
        elif consecutive_count == 1:
            # Single day missing: skip signal generation, log warning
            logger.warning(f"MISSING_DATA_1DAY: {symbol} {date}")

    def _find_consecutive_missing_dates(
        self, date: pd.Timestamp, missing_info: Dict, available_data: pd.DataFrame
    ) -> List[pd.Timestamp]:
        """Find consecutive missing dates that include the given date.

        Args:
            date: Date to check
            missing_info: Result from detect_missing_data
            available_data: Available data DataFrame

        Returns:
            List of consecutive missing dates including the given date, or empty list
        """
        # Check if date is in any consecutive gap
        for gap_start, gap_end in missing_info.get("consecutive_gaps", []):
            if gap_start <= date <= gap_end:
                # Generate all dates in this gap
                gap_dates = pd.date_range(start=gap_start, end=gap_end, freq="D")
                # Filter to only trading days if equity
                if len(available_data) > 0:
                    # Use business days for equity (simplified)
                    gap_dates = [d for d in gap_dates if d.weekday() < 5]
                return [pd.Timestamp(d) for d in gap_dates.tolist()]

        return []

    def _handle_missing_data_for_position(self, symbol: str, date: pd.Timestamp) -> None:
        """Handle missing data for a position.

        Args:
            symbol: Symbol with missing data
            date: Current date
        """
        # Increment missing count
        self.missing_data_counts[symbol] = self.missing_data_counts.get(symbol, 0) + 1

        # If 2+ consecutive days, close position
        if self.missing_data_counts[symbol] >= 2:
            # Get missing info to find consecutive dates
            if symbol in self.market_data.bars:
                bars_df = self.market_data.bars[symbol]
                available_data = bars_df[bars_df.index <= date]
                missing_info = detect_missing_data(available_data, symbol, asset_class=self._get_asset_class(symbol))
                consecutive_dates = self._find_consecutive_missing_dates(date, missing_info, available_data)
                if consecutive_dates:
                    self._handle_missing_data(symbol, consecutive_dates, date)
                else:
                    self._handle_missing_data(symbol, [date], date)

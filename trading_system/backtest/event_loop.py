"""Daily event loop for backtesting with no lookahead."""

from typing import List, Dict, Optional, Callable
import pandas as pd
import logging
import uuid

from ..models.market_data import MarketData
from ..portfolio.portfolio import Portfolio
from ..models.signals import Signal, SignalSide
from ..models.orders import Order, OrderStatus, Fill
from ..models.positions import Position, ExitReason
from ..models.bar import Bar
from ..models.features import FeatureRow
from ..strategies.base_strategy import BaseStrategy
from ..execution.fill_simulator import simulate_fill, reject_order_missing_data
from ..strategies.scoring import score_signals
from ..strategies.queue import select_signals_from_queue
from ..portfolio.position_sizing import calculate_position_size
from ..data.validator import validate_ohlcv, detect_missing_data

logger = logging.getLogger(__name__)


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
        strategies: List[BaseStrategy],
        compute_features_fn: Callable,
        get_next_trading_day: Callable[[pd.Timestamp], pd.Timestamp],
        rng: Optional = None
    ):
        """Initialize event loop.
        
        Args:
            market_data: MarketData container with bars and features
            portfolio: Portfolio state
            strategies: List of strategy objects (equity, crypto)
            compute_features_fn: Function to compute features for a symbol up to a date
            get_next_trading_day: Function to get next trading day from a date
            rng: Optional random number generator for reproducibility
        """
        self.market_data = market_data
        self.portfolio = portfolio
        self.strategies = strategies
        self.compute_features_fn = compute_features_fn
        self.get_next_trading_day = get_next_trading_day
        self.rng = rng
        
        # Track pending orders and exit orders
        self.pending_orders: List[Order] = []
        self.pending_exit_orders: List[tuple[Order, ExitReason]] = []
        
        # Track signal metadata for orders (order_id -> signal metadata)
        self.order_signal_metadata: Dict[str, Dict] = {}  # order_id -> {triggered_on, atr_mult, etc}
        
        # Track returns data for correlation calculations
        self.returns_data: Dict[str, List[float]] = {}
        
        # Track missing data
        self.missing_data_counts: Dict[str, int] = {}  # symbol -> consecutive missing days
    
    def process_day(self, date: pd.Timestamp) -> Dict:
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
            'date': date,
            'signals_generated': [],
            'orders_created': [],
            'orders_executed': [],
            'exits_triggered': [],
            'exits_executed': [],
            'portfolio_state': {}
        }
        
        # Step 1: Update data through day t close
        self._update_data_through_date(date)
        
        # Step 2: Generate signals at day t close
        signals = self._generate_signals(date)
        events['signals_generated'] = [s.symbol for s in signals]
        
        # Step 3: Create orders for day t+1 open
        orders = self._create_orders(signals, date)
        events['orders_created'] = [o.symbol for o in orders]
        self.pending_orders.extend(orders)
        
        # Step 4: Execute orders at day t+1 open (if any pending)
        next_day = self.get_next_trading_day(date)
        executed_fills = self._execute_pending_orders(next_day)
        events['orders_executed'] = [f.symbol for f in executed_fills]
        
        # Step 5: Update stops and check exits at day t+1 close
        exit_signals = self._update_stops_and_check_exits(next_day)
        events['exits_triggered'] = [s[0] for s in exit_signals]
        
        # Create exit orders for day t+2 open
        exit_orders = self._create_exit_orders(exit_signals, next_day)
        self.pending_exit_orders.extend(exit_orders)
        
        # Step 6: Execute exit orders at day t+2 open (if any pending from previous day)
        day_after_next = self.get_next_trading_day(next_day)
        exit_fills = self._execute_pending_exit_orders(day_after_next)
        events['exits_executed'] = [f.symbol for f in exit_fills]
        
        # Step 7: Update portfolio metrics at day t+1 close
        self._update_portfolio_metrics(next_day)
        
        # Step 8: Log daily state
        events['portfolio_state'] = {
            'equity': self.portfolio.equity,
            'cash': self.portfolio.cash,
            'open_positions': len(self.portfolio.positions),
            'realized_pnl': self.portfolio.realized_pnl,
            'unrealized_pnl': self.portfolio.unrealized_pnl,
            'gross_exposure': self.portfolio.gross_exposure,
            'risk_multiplier': self.portfolio.risk_multiplier
        }
        
        return events
    
    def _update_data_through_date(self, date: pd.Timestamp) -> None:
        """Update all market data up to and including day t close.
        
        For each symbol:
        1. Load/update bars up to date
        2. Compute indicators using data <= date (no lookahead)
        3. Update features in market_data
        """
        # Update features for all symbols in universe
        all_symbols = set()
        for strategy in self.strategies:
            all_symbols.update(strategy.universe)
        
        for symbol in all_symbols:
            if symbol not in self.market_data.bars:
                continue
            
            bars_df = self.market_data.bars[symbol]
            
            # Filter to data <= date
            available_data = bars_df[bars_df.index <= date]
            
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
                    missing_info = detect_missing_data(
                        available_data,
                        symbol,
                        asset_class=self._get_asset_class(symbol)
                    )
                    consecutive_dates = self._find_consecutive_missing_dates(
                        date, missing_info, available_data
                    )
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
            
            # Compute features using data <= date only
            try:
                features_df = self.compute_features_fn(
                    available_data,
                    symbol,
                    asset_class=self._get_asset_class(symbol)
                )
                
                # Update market_data.features
                if symbol not in self.market_data.features:
                    self.market_data.features[symbol] = features_df
                else:
                    # Update existing features
                    for idx, row in features_df.iterrows():
                        if idx <= date:
                            self.market_data.features[symbol].loc[idx] = row
                
            except Exception as e:
                logger.warning(f"Error computing features for {symbol} at {date}: {e}")
                continue
    
    def _generate_signals(self, date: pd.Timestamp) -> List[Signal]:
        """Generate entry signals for all strategies at day t close.
        
        Args:
            date: Date at which to generate signals
        
        Returns:
            List of valid signals
        """
        all_signals = []
        
        for strategy in self.strategies:
            for symbol in strategy.universe:
                # Get features for this symbol at date
                features = self.market_data.get_features(symbol, date)
                if features is None:
                    continue
                
                # Check if features are valid for entry
                if not features.is_valid_for_entry():
                    continue
                
                # Estimate order notional for capacity check
                # We need to estimate position size first
                estimated_qty = self._estimate_position_size(
                    strategy, features, self.portfolio
                )
                order_notional = features.close * estimated_qty
                
                # Generate signal
                signal = strategy.generate_signal(
                    symbol=symbol,
                    features=features,
                    order_notional=order_notional,
                    diversification_bonus=0.0  # Will be computed during scoring
                )
                
                if signal is not None and signal.is_valid():
                    all_signals.append(signal)
        
        # Score signals
        if all_signals:
            self._score_signals(all_signals, date)
        
        return all_signals
    
    def _score_signals(self, signals: List[Signal], date: pd.Timestamp) -> None:
        """Score signals using breakout strength, momentum, and diversification.
        
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
            lookback=20
        )
    
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
                risk_multiplier=self.portfolio.risk_multiplier
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
                status=OrderStatus.PENDING
            )
            
            # Store signal metadata for this order
            self.order_signal_metadata[order.order_id] = {
                'triggered_on': signal.triggered_on,
                'atr_mult': signal.atr_mult,
                'adv20_at_entry': signal.adv20
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
            lookback=20
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
            atr14_history = self.market_data.features[order.symbol]['atr14']
            
            # Get benchmark bars for stress calculation
            benchmark_symbol = "SPY" if order.asset_class == "equity" else "BTC"
            benchmark_bars = self.market_data.benchmarks.get(benchmark_symbol)
            if benchmark_bars is None:
                benchmark_bars = pd.DataFrame()
            
            # Base slippage
            base_slippage_bps = 8.0 if order.asset_class == "equity" else 10.0
            
            # Simulate fill
            try:
                fill = simulate_fill(
                    order=order,
                    open_bar=open_bar,
                    atr14=features.atr14,
                    atr14_history=atr14_history,
                    adv20=features.adv20,
                    benchmark_bars=benchmark_bars,
                    base_slippage_bps=base_slippage_bps,
                    rng=self.rng
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
                    atr_mult = signal_metadata.get('atr_mult', strategy.config.exit.hard_stop_atr_mult)
                    triggered_on = signal_metadata.get('triggered_on')
                    adv20_at_entry = signal_metadata.get('adv20_at_entry', features.adv20)
                    
                    position = self.portfolio.process_fill(
                        fill=fill,
                        stop_price=order.stop_price,
                        atr_mult=atr_mult,
                        triggered_on=triggered_on,
                        adv20_at_entry=adv20_at_entry
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
        current_prices = {}
        features_data = {}
        
        for symbol, position in self.portfolio.positions.items():
            if not position.is_open():
                continue
            
            bar = self.market_data.get_bar(symbol, date)
            if bar is None:
                # Missing data - handle separately
                self._handle_missing_data_for_position(symbol, date)
                continue
            
            current_prices[symbol] = bar.close
            
            features = self.market_data.get_features(symbol, date)
            if features:
                features_data[symbol] = {
                    'ma20': features.ma20,
                    'ma50': features.ma50,
                    'atr14': features.atr14
                }
        
        # Get strategy config for exit mode
        strategy = self.strategies[0] if self.strategies else None
        exit_mode = strategy.config.exit.mode if strategy else "ma_cross"
        exit_ma = strategy.config.exit.exit_ma if strategy else 20
        
        # Update stops and check exits
        exit_signals = self.portfolio.update_stops(
            current_prices=current_prices,
            features_data=features_data,
            exit_mode=exit_mode,
            exit_ma=exit_ma
        )
        
        return exit_signals
    
    def _create_exit_orders(
        self,
        exit_signals: List[tuple[str, ExitReason]],
        date: pd.Timestamp
    ) -> List[Order]:
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
                status=OrderStatus.PENDING
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
            atr14_history = self.market_data.features[order.symbol]['atr14']
            
            # Get benchmark bars
            benchmark_symbol = "SPY" if order.asset_class == "equity" else "BTC"
            benchmark_bars = self.market_data.benchmarks.get(benchmark_symbol)
            if benchmark_bars is None:
                benchmark_bars = pd.DataFrame()
            
            # Base slippage
            base_slippage_bps = 8.0 if order.asset_class == "equity" else 10.0
            
            # Simulate fill
            try:
                fill = simulate_fill(
                    order=order,
                    open_bar=open_bar,
                    atr14=features.atr14,
                    atr14_history=atr14_history,
                    adv20=features.adv20,
                    benchmark_bars=benchmark_bars,
                    base_slippage_bps=base_slippage_bps,
                    rng=self.rng
                )
                
                # Update order status
                order.status = OrderStatus.FILLED
                
                # Close position
                closed_position = self.portfolio.close_position(
                    symbol=order.symbol,
                    exit_fill=fill,
                    exit_reason=exit_reason
                )
                
                if closed_position:
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
        current_prices = {}
        for symbol in self.portfolio.positions.keys():
            bar = self.market_data.get_bar(symbol, date)
            if bar:
                current_prices[symbol] = bar.close
        
        # Update equity
        self.portfolio.update_equity(current_prices)
        
        # Update volatility scaling
        self.portfolio.update_volatility_scaling()
        
        # Update correlation metrics
        self.portfolio.update_correlation_metrics(
            returns_data=self.returns_data,
            lookback=20
        )
        
        # Append to equity curve
        self.portfolio.append_daily_metrics()
    
    def _estimate_position_size(
        self,
        strategy: BaseStrategy,
        features: FeatureRow,
        portfolio: Portfolio
    ) -> int:
        """Estimate position size for capacity check.
        
        Args:
            strategy: Strategy object
            features: FeatureRow with entry and stop prices
            portfolio: Portfolio state
        
        Returns:
            Estimated quantity
        """
        from ..portfolio.position_sizing import estimate_position_size
        
        return estimate_position_size(
            equity=portfolio.equity,
            risk_pct=strategy.config.risk.risk_per_trade * portfolio.risk_multiplier,
            entry_price=features.close,
            stop_price=features.close - (strategy.config.exit.hard_stop_atr_mult * features.atr14),
            max_position_notional=portfolio.equity * strategy.config.risk.max_position_notional,
            risk_multiplier=portfolio.risk_multiplier
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
    
    def _get_strategy_for_signal(self, signal: Signal) -> Optional[BaseStrategy]:
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
    
    def _get_strategy_for_signal_symbol(self, symbol: str) -> Optional[BaseStrategy]:
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
    
    def _get_candidate_returns(
        self,
        signals: List[Signal],
        date: pd.Timestamp
    ) -> Dict[str, List[float]]:
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
                        returns = (available_data['close'] / available_data['close'].shift(1) - 1).tolist()
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
        portfolio_returns = {}
        
        for symbol in self.portfolio.positions.keys():
            if symbol in self.returns_data:
                portfolio_returns[symbol] = self.returns_data[symbol]
            else:
                # Compute returns from bars
                if symbol in self.market_data.bars:
                    bars_df = self.market_data.bars[symbol]
                    available_data = bars_df[bars_df.index <= date]
                    if len(available_data) > 1:
                        returns = (available_data['close'] / available_data['close'].shift(1) - 1).tolist()
                        portfolio_returns[symbol] = returns
                        self.returns_data[symbol] = returns
        
        return portfolio_returns
    
    def _handle_missing_data(self, symbol: str, missing_dates: List[pd.Timestamp], date: pd.Timestamp) -> None:
        """Handle missing data for a symbol.
        
        Args:
            symbol: Symbol with missing data
            missing_dates: List of missing dates
            date: Current date
        """
        consecutive_count = len(missing_dates)
        
        if consecutive_count == 1:
            # Single day missing: skip signal generation, log warning
            logger.warning(f"MISSING_DATA_1DAY: {symbol} {date}")
            self.missing_data_counts[symbol] = 1
        else:
            # 2+ consecutive days: mark as unhealthy, close position if exists
            logger.error(f"DATA_UNHEALTHY: {symbol}, missing {consecutive_count} days")
            self.missing_data_counts[symbol] = consecutive_count
            
            # Close position if exists
            position = self.portfolio.get_position(symbol)
            if position and position.is_open():
                # Try to exit at next available data
                next_date = self.get_next_trading_day(date)
                next_bar = self.market_data.get_bar(symbol, next_date)
                if next_bar:
                    # Create exit order for next day
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
                        stop_price=0.0,
                        status=OrderStatus.PENDING
                    )
                    self.pending_exit_orders.append((exit_order, ExitReason.DATA_MISSING))
                else:
                    # Force exit at last known close
                    last_bar = self.market_data.bars[symbol].iloc[-1]
                    exit_price = last_bar['close']
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
                        notional=exit_price * position.quantity
                    )
                    self.portfolio.close_position(symbol, exit_fill, ExitReason.DATA_MISSING)
    
    def _find_consecutive_missing_dates(
        self, 
        date: pd.Timestamp, 
        missing_info: Dict, 
        available_data: pd.DataFrame
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
        for gap_start, gap_end in missing_info.get('consecutive_gaps', []):
            if gap_start <= date <= gap_end:
                # Generate all dates in this gap
                gap_dates = pd.date_range(start=gap_start, end=gap_end, freq='D')
                # Filter to only trading days if equity
                if len(available_data) > 0:
                    # Use business days for equity (simplified)
                    gap_dates = [d for d in gap_dates if d.weekday() < 5]
                return gap_dates.tolist()
        
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
                missing_info = detect_missing_data(
                    available_data,
                    symbol,
                    asset_class=self._get_asset_class(symbol)
                )
                consecutive_dates = self._find_consecutive_missing_dates(
                    date, missing_info, available_data
                )
                if consecutive_dates:
                    self._handle_missing_data(symbol, consecutive_dates, date)
                else:
                    self._handle_missing_data(symbol, [date], date)


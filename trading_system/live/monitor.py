"""Position and risk monitoring for live trading."""

import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Set

import pandas as pd

from ..adapters.base_adapter import BaseAdapter
from ..logging.logger import get_logger
from ..models.features import FeatureRow
from ..models.orders import Fill, Order, OrderStatus, SignalSide
from ..models.positions import ExitReason, Position
from ..models.signals import BreakoutType, Signal
from ..portfolio.portfolio import Portfolio
from ..portfolio.position_sizing import calculate_position_size
from ..strategies.base.strategy_interface import StrategyInterface

logger = get_logger(__name__)


class AlertLevel(str, Enum):
    """Alert severity levels."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class RiskAlert:
    """Risk alert message."""

    level: AlertLevel
    message: str
    timestamp: pd.Timestamp
    symbol: Optional[str] = None
    metric: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None


class LiveMonitor:
    """Position and risk monitoring for live trading.

    Monitors open positions, checks stop losses, updates trailing stops,
    monitors portfolio risk metrics, and generates alerts.

    Example:
        >>> adapter = AlpacaAdapter(config)
        >>> monitor = LiveMonitor(adapter, portfolio, strategies)
        >>> monitor.start()
        >>> # Monitor runs in background
        >>> alerts = monitor.get_alerts()
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        portfolio: Portfolio,
        strategies: List[StrategyInterface],
        update_interval_seconds: float = 5.0,
        alert_callback: Optional[Callable[[RiskAlert], None]] = None,
        get_features: Optional[Callable[[str], Optional[FeatureRow]]] = None,
    ):
        """Initialize live monitor.

        Args:
            adapter: Broker adapter for order execution and position tracking
            portfolio: Portfolio instance to monitor
            strategies: List of strategies (for exit logic)
            update_interval_seconds: How often to check positions/risk (seconds)
            alert_callback: Optional callback function called when alerts are generated
            get_features: Optional function to get latest features for a symbol
        """
        self.adapter = adapter
        self.portfolio = portfolio
        self.strategies = strategies
        self.update_interval = update_interval_seconds
        self.alert_callback = alert_callback
        self.get_features = get_features

        # Alert queue
        self._alerts: List[RiskAlert] = []
        self._alert_lock = threading.Lock()

        # Threading
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None

        # Pending orders
        self._pending_orders: Dict[str, Order] = {}  # order_id -> Order
        self._order_lock = threading.Lock()

        # Strategy lookup
        self._strategy_by_symbol: Dict[str, StrategyInterface] = {}
        for strategy in strategies:
            for symbol in strategy.universe:
                self._strategy_by_symbol[symbol] = strategy

    def start(self) -> None:
        """Start the monitoring thread."""
        if self._running:
            logger.warning("Monitor already running")
            return

        if not self.adapter.is_connected():
            raise RuntimeError("Adapter not connected")

        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Live monitor started")

    def stop(self) -> None:
        """Stop the monitoring thread."""
        if not self._running:
            return

        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Live monitor stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop running in background thread."""
        while self._running:
            try:
                # Sync positions from broker
                self._sync_positions()

                # Update portfolio equity with current prices
                self._update_portfolio_equity()

                # Check stop losses and exit signals
                self._check_exits()

                # Monitor risk metrics
                self._check_risk_limits()

                # Check pending orders
                self._check_pending_orders()

                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}", exc_info=True)
                time.sleep(self.update_interval)

    def _sync_positions(self) -> None:
        """Sync positions from broker to portfolio."""
        try:
            broker_positions = self.adapter.get_positions()

            # Update portfolio positions with broker data
            for symbol, broker_pos in broker_positions.items():
                portfolio_pos = self.portfolio.get_position(symbol)

                if portfolio_pos is None:
                    # New position from broker (not in our portfolio)
                    # This shouldn't happen in normal operation, but handle it
                    logger.warning(f"Found position in broker but not in portfolio: {symbol}")
                    # Add to portfolio (with limited info from broker)
                    self.portfolio.add_position(broker_pos)
                else:
                    # Update existing position
                    # Note: Broker position may not have all our fields
                    # We keep our portfolio position as source of truth
                    pass

            # Check for positions closed in broker but still in portfolio
            portfolio_symbols = set(self.portfolio.positions.keys())
            broker_symbols = set(broker_positions.keys())

            for symbol in portfolio_symbols - broker_symbols:
                # Position closed in broker
                logger.info(f"Position {symbol} closed in broker, removing from portfolio")
                self.portfolio.remove_position(symbol)

        except Exception as e:
            logger.error(f"Failed to sync positions: {e}")

    def _update_portfolio_equity(self) -> None:
        """Update portfolio equity with current market prices."""
        current_prices = {}

        for symbol in self.portfolio.positions.keys():
            try:
                price = self.adapter.get_current_price(symbol)
                if price is not None:
                    current_prices[symbol] = price
            except Exception as e:
                logger.warning(f"Failed to get price for {symbol}: {e}")

        if current_prices:
            self.portfolio.update_equity(current_prices)

    def _check_exits(self) -> None:
        """Check stop losses and exit signals for all positions."""
        if not self.portfolio.positions:
            return

        current_prices = {}
        features_data = {}

        # Get current prices and features
        for symbol, position in self.portfolio.positions.items():
            if not position.is_open():
                continue

            try:
                price = self.adapter.get_current_price(symbol)
                if price is None:
                    continue

                current_prices[symbol] = price

                # Get features if available
                if self.get_features:
                    features = self.get_features(symbol)
                    if features:
                        features_data[symbol] = {"ma20": features.ma20, "ma50": features.ma50, "atr14": features.atr14}
            except Exception as e:
                logger.warning(f"Failed to get data for {symbol}: {e}")

        # Check exit signals
        exit_mode = "ma_cross"  # Default, could be configurable
        exit_ma = 20  # Default

        exit_signals = self.portfolio.update_stops(
            current_prices=current_prices, features_data=features_data, exit_mode=exit_mode, exit_ma=exit_ma
        )

        # Execute exits
        for symbol, exit_reason in exit_signals:
            try:
                self._execute_exit(symbol, exit_reason, current_prices.get(symbol))
            except Exception as e:
                logger.error(f"Failed to execute exit for {symbol}: {e}")

    def _execute_exit(self, symbol: str, exit_reason: ExitReason, exit_price: Optional[float]) -> None:
        """Execute exit for a position.

        Args:
            symbol: Symbol to exit
            exit_reason: Reason for exit
            exit_price: Current market price (if available)
        """
        position = self.portfolio.get_position(symbol)
        if position is None or not position.is_open():
            return

        # Get current price if not provided
        if exit_price is None:
            exit_price = self.adapter.get_current_price(symbol)
            if exit_price is None:
                logger.error(f"Cannot exit {symbol}: no price available")
                return

        # Create exit order
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            asset_class=position.asset_class,
            date=pd.Timestamp.now(),
            execution_date=pd.Timestamp.now(),  # Execute immediately
            side=SignalSide.SELL,
            quantity=position.quantity,
            signal_date=position.entry_date,
            expected_fill_price=exit_price,
            stop_price=position.stop_price,
            status=OrderStatus.PENDING,
        )

        # Submit order
        try:
            fill = self.adapter.submit_order(order)

            # Close position in portfolio
            closed_position = self.portfolio.close_position(symbol=symbol, exit_fill=fill, exit_reason=exit_reason)

            if closed_position:
                logger.info(
                    f"Exited position {symbol}: {exit_reason.value} | "
                    f"Entry: ${position.entry_price:.2f} | Exit: ${fill.fill_price:.2f} | "
                    f"P&L: ${closed_position.realized_pnl:.2f}"
                )

                # Generate alert
                self._add_alert(AlertLevel.INFO, f"Position exited: {symbol} ({exit_reason.value})", symbol=symbol)
        except Exception as e:
            logger.error(f"Failed to execute exit order for {symbol}: {e}")
            self._add_alert(AlertLevel.WARNING, f"Failed to exit position {symbol}: {e}", symbol=symbol)

    def _check_risk_limits(self) -> None:
        """Check portfolio risk limits and generate alerts."""
        # Check exposure limits
        max_exposure_pct = 0.80  # 80% max exposure (could be configurable)
        if self.portfolio.gross_exposure_pct > max_exposure_pct:
            self._add_alert(
                AlertLevel.WARNING,
                f"Exposure limit exceeded: {self.portfolio.gross_exposure_pct:.1%} > {max_exposure_pct:.1%}",
                metric="exposure",
                value=self.portfolio.gross_exposure_pct,
                threshold=max_exposure_pct,
            )

        # Check position count limits
        max_positions = 20  # Could be configurable
        if len(self.portfolio.positions) > max_positions:
            self._add_alert(
                AlertLevel.WARNING,
                f"Position count limit exceeded: {len(self.portfolio.positions)} > {max_positions}",
                metric="position_count",
                value=float(len(self.portfolio.positions)),
                threshold=float(max_positions),
            )

        # Check per-position exposure
        max_position_pct = 0.15  # 15% max per position
        for symbol, exposure_pct in self.portfolio.per_position_exposure.items():
            if exposure_pct > max_position_pct:
                self._add_alert(
                    AlertLevel.WARNING,
                    f"Position exposure limit exceeded: {symbol} at {exposure_pct:.1%} > {max_position_pct:.1%}",
                    symbol=symbol,
                    metric="position_exposure",
                    value=exposure_pct,
                    threshold=max_position_pct,
                )

        # Check unrealized P&L (large drawdown)
        if self.portfolio.unrealized_pnl < -0.10 * self.portfolio.starting_equity:  # 10% drawdown
            self._add_alert(
                AlertLevel.WARNING,
                f"Large unrealized loss: ${self.portfolio.unrealized_pnl:.2f}",
                metric="unrealized_pnl",
                value=self.portfolio.unrealized_pnl,
            )

        # Check cash level
        min_cash_pct = 0.05  # 5% minimum cash
        cash_pct = self.portfolio.cash / self.portfolio.equity if self.portfolio.equity > 0 else 0.0
        if cash_pct < min_cash_pct:
            self._add_alert(
                AlertLevel.WARNING,
                f"Low cash level: {cash_pct:.1%} < {min_cash_pct:.1%}",
                metric="cash_pct",
                value=cash_pct,
                threshold=min_cash_pct,
            )

    def _check_pending_orders(self) -> None:
        """Check status of pending orders."""
        with self._order_lock:
            orders_to_remove = []

            for order_id, order in self._pending_orders.items():
                try:
                    status = self.adapter.get_order_status(order_id)

                    if status == OrderStatus.FILLED:
                        # Order filled - process fill
                        # Note: In a real system, you'd get fill details from broker
                        # For now, we'll assume the adapter handles this
                        logger.info(f"Order {order_id} filled")
                        orders_to_remove.append(order_id)

                    elif status == OrderStatus.REJECTED or status == OrderStatus.CANCELLED:
                        logger.warning(f"Order {order_id} {status.value}")
                        orders_to_remove.append(order_id)

                except Exception as e:
                    logger.warning(f"Failed to check order {order_id}: {e}")

            # Remove processed orders
            for order_id in orders_to_remove:
                self._pending_orders.pop(order_id, None)

    def execute_signal(self, signal: Signal) -> Optional[Fill]:
        """Execute a signal by creating and submitting an order.

        Args:
            signal: Signal to execute

        Returns:
            Fill object if order executed successfully, None otherwise
        """
        if not self.adapter.is_connected():
            logger.error("Adapter not connected")
            return None

        # Get strategy for signal
        strategy = self._strategy_by_symbol.get(signal.symbol)
        if strategy is None:
            logger.error(f"No strategy found for {signal.symbol}")
            return None

        # Calculate position size
        current_price = self.adapter.get_current_price(signal.symbol)
        if current_price is None:
            logger.error(f"Cannot get price for {signal.symbol}")
            return None

        # Use current price for position sizing
        qty = calculate_position_size(
            equity=self.portfolio.equity,
            risk_pct=strategy.config.risk.risk_per_trade * self.portfolio.risk_multiplier,
            entry_price=current_price,
            stop_price=signal.stop_price,
            max_position_notional=self.portfolio.equity * strategy.config.risk.max_position_notional,
            max_exposure=self.portfolio.equity * 0.80,  # 80% max exposure
            available_cash=self.portfolio.cash,
            risk_multiplier=self.portfolio.risk_multiplier,
        )

        if qty < 1:
            logger.warning(f"Cannot afford position for {signal.symbol} (qty={qty})")
            self._add_alert(AlertLevel.WARNING, f"Cannot afford position for {signal.symbol}", symbol=signal.symbol)
            return None

        # Create order
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=signal.symbol,
            asset_class=signal.asset_class,
            date=pd.Timestamp.now(),
            execution_date=pd.Timestamp.now(),  # Execute immediately
            side=SignalSide.BUY,
            quantity=qty,
            signal_date=signal.date,
            expected_fill_price=current_price,
            stop_price=signal.stop_price,
            status=OrderStatus.PENDING,
        )

        # Submit order
        try:
            fill = self.adapter.submit_order(order)

            # Process fill in portfolio
            position = self.portfolio.process_fill(
                fill=fill,
                stop_price=signal.stop_price,
                atr_mult=signal.atr_mult or 2.5,
                triggered_on=signal.triggered_on or signal.metadata.get("breakout_type"),
                adv20_at_entry=signal.adv20 or 0.0,
            )

            logger.info(
                f"Executed signal {signal.symbol}: {qty} shares @ ${fill.fill_price:.2f} | " f"Stop: ${signal.stop_price:.2f}"
            )

            # Track order
            with self._order_lock:
                self._pending_orders[order.order_id] = order

            return fill

        except Exception as e:
            logger.error(f"Failed to execute order for {signal.symbol}: {e}")
            self._add_alert(AlertLevel.WARNING, f"Failed to execute order for {signal.symbol}: {e}", symbol=signal.symbol)
            return None

    def _add_alert(
        self,
        level: AlertLevel,
        message: str,
        symbol: Optional[str] = None,
        metric: Optional[str] = None,
        value: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> None:
        """Add an alert to the queue.

        Args:
            level: Alert severity level
            message: Alert message
            symbol: Optional symbol related to alert
            metric: Optional metric name
            value: Optional metric value
            threshold: Optional threshold value
        """
        alert = RiskAlert(
            level=level,
            message=message,
            timestamp=pd.Timestamp.now(),
            symbol=symbol,
            metric=metric,
            value=value,
            threshold=threshold,
        )

        with self._alert_lock:
            self._alerts.append(alert)
            # Keep only last 1000 alerts
            if len(self._alerts) > 1000:
                self._alerts = self._alerts[-1000:]

        # Log alert
        if level == AlertLevel.CRITICAL:
            logger.critical(f"ALERT [{level.value}]: {message}")
        elif level == AlertLevel.WARNING:
            logger.warning(f"ALERT [{level.value}]: {message}")
        else:
            logger.info(f"ALERT [{level.value}]: {message}")

        # Call callback if provided
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def get_alerts(self, level: Optional[AlertLevel] = None, clear: bool = False) -> List[RiskAlert]:
        """Get alerts from queue.

        Args:
            level: Optional filter by alert level
            clear: If True, clear the queue after returning alerts

        Returns:
            List of alerts (most recent first)
        """
        with self._alert_lock:
            alerts = self._alerts.copy()
            if level:
                alerts = [a for a in alerts if a.level == level]
            alerts.reverse()  # Most recent first

            if clear:
                self._alerts.clear()

            return alerts

    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary.

        Returns:
            Dictionary with portfolio metrics
        """
        return {
            "equity": self.portfolio.equity,
            "cash": self.portfolio.cash,
            "starting_equity": self.portfolio.starting_equity,
            "realized_pnl": self.portfolio.realized_pnl,
            "unrealized_pnl": self.portfolio.unrealized_pnl,
            "gross_exposure": self.portfolio.gross_exposure,
            "gross_exposure_pct": self.portfolio.gross_exposure_pct,
            "open_positions": len(self.portfolio.positions),
            "total_trades": self.portfolio.total_trades,
            "risk_multiplier": self.portfolio.risk_multiplier,
            "portfolio_vol_20d": self.portfolio.portfolio_vol_20d,
            "avg_pairwise_corr": self.portfolio.avg_pairwise_corr,
        }

    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running

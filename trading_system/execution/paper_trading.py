"""Paper trading execution pipeline."""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..adapters.base_adapter import AccountInfo, AdapterConfig, BaseAdapter
from ..models.orders import Fill, Order, OrderStatus
from ..models.positions import Position
from ..storage.database import ResultsDatabase

logger = logging.getLogger(__name__)


@dataclass
class OrderLifecycle:
    """Track order lifecycle from submission to fill."""

    order: Order
    submitted_at: datetime
    status: OrderStatus = OrderStatus.PENDING
    fill: Optional[Fill] = None
    filled_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class PaperTradingConfig:
    """Configuration for paper trading execution."""

    # Broker adapter config
    adapter_config: AdapterConfig

    # Execution settings
    max_orders_per_day: int = 20
    order_timeout_seconds: int = 300  # 5 minutes
    retry_failed_orders: bool = True

    # Logging
    log_dir: Path = field(default_factory=lambda: Path("logs/paper_trading"))
    db_path: Optional[Path] = None  # Database for order/fill tracking


class PaperTradingRunner:
    """Execute orders via paper trading broker adapter.

    This class manages the order lifecycle:
    1. Receive orders (from signal generation)
    2. Submit to broker via adapter
    3. Track order status
    4. Log fills
    5. Reconcile positions with broker
    """

    def __init__(self, config: PaperTradingConfig, adapter: BaseAdapter):
        """Initialize paper trading runner.

        Args:
            config: Paper trading configuration
            adapter: Broker adapter instance (must be connected)
        """
        self.config = config
        self.adapter = adapter
        self.db = ResultsDatabase(db_path=config.db_path)

        # Order tracking
        self.pending_orders: Dict[str, OrderLifecycle] = {}
        self.completed_orders: Dict[str, OrderLifecycle] = {}

        # Setup logging
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup file logging for paper trading."""
        log_file = self.config.log_dir / f"paper_trading_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def submit_orders(self, orders: List[Order]) -> Dict[str, OrderLifecycle]:
        """Submit multiple orders to broker.

        Args:
            orders: List of orders to submit

        Returns:
            Dictionary mapping order_id to OrderLifecycle
        """
        if not self.adapter.is_connected():
            raise ConnectionError("Adapter is not connected to broker")

        # Check daily order limit
        today_orders = sum(
            1 for ol in self.completed_orders.values() if ol.submitted_at.date() == datetime.now().date()
        )
        if today_orders + len(orders) > self.config.max_orders_per_day:
            raise ValueError(
                f"Daily order limit exceeded: {today_orders + len(orders)} > {self.config.max_orders_per_day}"
            )

        results = {}
        for order in orders:
            lifecycle = self._submit_single_order(order)
            results[order.order_id] = lifecycle

        return results

    def _submit_single_order(self, order: Order) -> OrderLifecycle:
        """Submit a single order to broker.

        Args:
            order: Order to submit

        Returns:
            OrderLifecycle tracking object
        """
        lifecycle = OrderLifecycle(order=order, submitted_at=datetime.now())

        try:
            logger.info(f"Submitting order {order.order_id}: {order.side} {order.quantity} {order.symbol}")

            # Submit to broker
            fill = self.adapter.submit_order(order)

            # Update lifecycle
            lifecycle.status = OrderStatus.FILLED
            lifecycle.fill = fill
            lifecycle.filled_at = datetime.now()

            logger.info(
                f"Order {order.order_id} filled: {fill.quantity} @ {fill.fill_price:.2f} "
                f"(slippage: {fill.slippage_bps:.2f} bps, fee: {fill.fee_bps:.2f} bps)"
            )

            # Move to completed
            self.completed_orders[order.order_id] = lifecycle

        except ValueError as e:
            # Invalid order or insufficient funds
            lifecycle.status = OrderStatus.REJECTED
            lifecycle.error_message = str(e)
            logger.error(f"Order {order.order_id} rejected: {e}")
            self.completed_orders[order.order_id] = lifecycle

        except ConnectionError as e:
            # Network error - may retry
            lifecycle.status = OrderStatus.PENDING
            lifecycle.error_message = str(e)
            logger.error(f"Order {order.order_id} failed (network): {e}")
            self.pending_orders[order.order_id] = lifecycle

        except Exception as e:
            # Unexpected error
            lifecycle.status = OrderStatus.REJECTED
            lifecycle.error_message = str(e)
            logger.error(f"Order {order.order_id} failed (unexpected): {e}", exc_info=True)
            self.completed_orders[order.order_id] = lifecycle

        return lifecycle

    def retry_pending_orders(self) -> Dict[str, OrderLifecycle]:
        """Retry pending orders that failed due to transient errors.

        Returns:
            Dictionary of retried orders
        """
        retried = {}
        for order_id, lifecycle in list(self.pending_orders.items()):
            if lifecycle.retry_count >= self.config.max_retries:
                logger.warning(f"Order {order_id} exceeded max retries, marking as rejected")
                lifecycle.status = OrderStatus.REJECTED
                self.completed_orders[order_id] = lifecycle
                del self.pending_orders[order_id]
                continue

            lifecycle.retry_count += 1
            logger.info(f"Retrying order {order_id} (attempt {lifecycle.retry_count}/{self.config.max_retries})")

            new_lifecycle = self._submit_single_order(lifecycle.order)
            retried[order_id] = new_lifecycle

            # Remove from pending if completed
            if new_lifecycle.status in [OrderStatus.FILLED, OrderStatus.REJECTED]:
                if order_id in self.pending_orders:
                    del self.pending_orders[order_id]

        return retried

    def get_account_info(self) -> AccountInfo:
        """Get current account information from broker.

        Returns:
            AccountInfo with equity, cash, buying power
        """
        if not self.adapter.is_connected():
            raise ConnectionError("Adapter is not connected to broker")

        return self.adapter.get_account_info()

    def get_positions(self) -> Dict[str, Position]:
        """Get current positions from broker.

        Returns:
            Dictionary mapping symbol to Position
        """
        if not self.adapter.is_connected():
            raise ConnectionError("Adapter is not connected to broker")

        return self.adapter.get_positions()

    def reconcile_positions(self) -> Dict[str, Position]:
        """Reconcile positions between broker and internal tracking.

        Returns:
            Dictionary of current positions from broker
        """
        broker_positions = self.get_positions()

        logger.info(f"Reconciled {len(broker_positions)} positions from broker")
        for symbol, position in broker_positions.items():
            logger.info(
                f"  {symbol}: {position.side} {position.quantity} @ {position.entry_price:.2f} "
                f"(stop: {position.stop_price:.2f})"
            )

        return broker_positions

    def get_order_summary(self) -> Dict[str, int]:
        """Get summary of order statuses.

        Returns:
            Dictionary with counts by status
        """
        summary = {
            "pending": len(self.pending_orders),
            "filled": sum(1 for ol in self.completed_orders.values() if ol.status == OrderStatus.FILLED),
            "rejected": sum(1 for ol in self.completed_orders.values() if ol.status == OrderStatus.REJECTED),
            "total": len(self.pending_orders) + len(self.completed_orders),
        }
        return summary

    def export_fills_to_csv(self, output_path: Path) -> None:
        """Export all fills to CSV for analysis.

        Args:
            output_path: Path to output CSV file
        """
        fills = []
        for lifecycle in self.completed_orders.values():
            if lifecycle.fill:
                fills.append(
                    {
                        "order_id": lifecycle.order.order_id,
                        "fill_id": lifecycle.fill.fill_id,
                        "symbol": lifecycle.fill.symbol,
                        "asset_class": lifecycle.fill.asset_class,
                        "side": lifecycle.fill.side.value,
                        "quantity": lifecycle.fill.quantity,
                        "fill_price": lifecycle.fill.fill_price,
                        "open_price": lifecycle.fill.open_price,
                        "slippage_bps": lifecycle.fill.slippage_bps,
                        "fee_bps": lifecycle.fill.fee_bps,
                        "total_cost": lifecycle.fill.total_cost,
                        "notional": lifecycle.fill.notional,
                        "submitted_at": lifecycle.submitted_at.isoformat(),
                        "filled_at": lifecycle.filled_at.isoformat() if lifecycle.filled_at else None,
                    }
                )

        if fills:
            df = pd.DataFrame(fills)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(fills)} fills to {output_path}")
        else:
            logger.warning("No fills to export")

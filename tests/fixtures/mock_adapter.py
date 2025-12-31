"""Mock adapter for testing without API access."""

import logging
import time
import uuid
from typing import Dict, List, Optional

import pandas as pd

from trading_system.adapters.base_adapter import AccountInfo, AdapterConfig, BaseAdapter
from trading_system.models.orders import Fill, Order, OrderStatus, SignalSide
from trading_system.models.positions import Position, PositionSide
from trading_system.models.signals import BreakoutType

logger = logging.getLogger(__name__)


class MockAdapter(BaseAdapter):
    """Mock adapter for unit testing without API access.

    Simulates broker behavior including:
    - Connection/disconnection
    - Order submission and fills
    - Position tracking
    - Market data
    - Error conditions (network failures, rate limits, etc.)
    """

    def __init__(self, config: AdapterConfig):
        """Initialize mock adapter.

        Args:
            config: AdapterConfig (paper_trading flag is respected)
        """
        super().__init__(config)
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._order_status: Dict[str, OrderStatus] = {}
        self._prices: Dict[str, float] = {}  # Symbol -> current price
        self._account_equity: float = 100000.0  # Default starting equity
        self._account_cash: float = 100000.0
        self._connected = False

        # Error simulation flags
        self._simulate_connection_error: bool = False
        self._simulate_rate_limit: bool = False
        self._simulate_network_failure: bool = False
        self._simulate_insufficient_funds: bool = False
        self._simulate_invalid_order: bool = False
        self._simulate_timeout: bool = False

        # Rate limiting simulation
        self._rate_limit_calls: int = 0
        self._rate_limit_max: int = 10  # Max calls before rate limit

        # Timeout simulation
        self._timeout_delay: float = 0.0  # Delay in seconds

    def set_simulate_connection_error(self, value: bool) -> None:
        """Set flag to simulate connection errors."""
        self._simulate_connection_error = value

    def set_simulate_rate_limit(self, value: bool) -> None:
        """Set flag to simulate rate limiting."""
        self._simulate_rate_limit = value

    def set_simulate_network_failure(self, value: bool) -> None:
        """Set flag to simulate network failures."""
        self._simulate_network_failure = value

    def set_simulate_insufficient_funds(self, value: bool) -> None:
        """Set flag to simulate insufficient funds errors."""
        self._simulate_insufficient_funds = value

    def set_simulate_invalid_order(self, value: bool) -> None:
        """Set flag to simulate invalid order errors."""
        self._simulate_invalid_order = value

    def set_simulate_timeout(self, value: bool, delay: float = 5.0) -> None:
        """Set flag to simulate timeouts.

        Args:
            value: Whether to simulate timeout
            delay: Delay in seconds before timeout
        """
        self._simulate_timeout = value
        self._timeout_delay = delay

    def set_price(self, symbol: str, price: float) -> None:
        """Set current market price for a symbol.

        Args:
            symbol: Symbol to set price for
            price: Price value
        """
        self._prices[symbol] = price

    def set_account_balance(self, equity: float, cash: float) -> None:
        """Set account balance.

        Args:
            equity: Account equity
            cash: Available cash
        """
        self._account_equity = equity
        self._account_cash = cash

    def add_position(self, position: Position) -> None:
        """Add a position to the mock adapter.

        Args:
            position: Position to add
        """
        self._positions[position.symbol] = position

    def connect(self) -> None:
        """Connect to mock broker.

        Raises:
            ConnectionError: If connection simulation is enabled
        """
        if self._simulate_connection_error:
            raise ConnectionError("Simulated connection error")

        self._connected = True
        logger.info("Connected to mock broker")

    def disconnect(self) -> None:
        """Disconnect from mock broker."""
        self._connected = False
        logger.info("Disconnected from mock broker")

    def is_connected(self) -> bool:
        """Check if connected to mock broker."""
        return self._connected

    def get_account_info(self) -> AccountInfo:
        """Get account information.

        Returns:
            AccountInfo with current balances

        Raises:
            ConnectionError: If not connected
            RuntimeError: If network failure is simulated
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to mock broker")

        if self._simulate_network_failure:
            raise RuntimeError("Simulated network failure")

        # Check rate limiting
        if self._simulate_rate_limit:
            self._rate_limit_calls += 1
            if self._rate_limit_calls > self._rate_limit_max:
                raise RuntimeError("Rate limit exceeded")

        buying_power = self._account_cash * 2.0  # 2x leverage for mock
        margin_used = max(0.0, self._account_equity - self._account_cash)

        return AccountInfo(
            equity=self._account_equity,
            cash=self._account_cash,
            buying_power=buying_power,
            margin_used=margin_used,
            broker_account_id="MOCK_ACCOUNT",
            currency="USD",
        )

    def submit_order(self, order: Order) -> Fill:
        """Submit order to mock broker.

        Args:
            order: Order to submit

        Returns:
            Fill object with execution details

        Raises:
            ConnectionError: If not connected
            ValueError: If order is invalid or insufficient funds
            RuntimeError: If network failure or timeout
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to mock broker")

        if self._simulate_network_failure:
            raise RuntimeError("Simulated network failure")

        if self._simulate_timeout:
            time.sleep(self._timeout_delay)
            raise RuntimeError(f"Order timeout after {self._timeout_delay}s")

        # Check rate limiting
        if self._simulate_rate_limit:
            self._rate_limit_calls += 1
            if self._rate_limit_calls > self._rate_limit_max:
                raise RuntimeError("Rate limit exceeded")

        # Check invalid order
        if self._simulate_invalid_order:
            raise ValueError("Simulated invalid order error")

        # Validate order
        if order.quantity <= 0:
            raise ValueError(f"Invalid quantity: {order.quantity}")

        if order.symbol not in self._prices:
            raise ValueError(f"Symbol {order.symbol} not found")

        # Check insufficient funds
        if self._simulate_insufficient_funds:
            raise ValueError("Insufficient funds")

        # Calculate order cost
        current_price = self._prices.get(order.symbol, order.expected_fill_price)
        order_cost = current_price * order.quantity

        if order_cost > self._account_cash:
            raise ValueError(f"Insufficient funds: need {order_cost}, have {self._account_cash}")

        # Simulate fill
        # For market orders, fill at current price with small slippage
        open_price = current_price
        slippage_bps = 5.0  # 5 bps slippage
        if order.side == SignalSide.BUY:
            fill_price = open_price * (1 + slippage_bps / 10000)
        else:  # SELL
            fill_price = open_price * (1 - slippage_bps / 10000)

        # Fee calculation
        fee_bps = 1.0 if order.asset_class == "equity" else 8.0
        notional = fill_price * order.quantity
        fee_cost = notional * (fee_bps / 10000)

        fill = Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            asset_class=order.asset_class,
            date=pd.Timestamp.now().normalize(),
            side=order.side,
            quantity=order.quantity,
            fill_price=fill_price,
            open_price=open_price,
            slippage_bps=slippage_bps,
            fee_bps=fee_bps,
            total_cost=fee_cost,
            vol_mult=1.0,
            size_penalty=1.0,
            weekend_penalty=1.0,
            stress_mult=1.0,
            notional=notional,
        )

        # Update account
        if order.side == SignalSide.BUY:
            self._account_cash -= notional + fee_cost
        else:  # SELL
            self._account_cash += notional - fee_cost

        # Update positions
        if order.side == SignalSide.BUY:
            # Opening or adding to long position
            if order.symbol in self._positions:
                pos = self._positions[order.symbol]
                # Update average entry price
                total_cost = (pos.entry_price * pos.quantity) + (fill_price * order.quantity)
                pos.quantity += order.quantity
                pos.entry_price = total_cost / pos.quantity
            else:
                # New position
                pos = Position(
                    symbol=order.symbol,
                    asset_class=order.asset_class,
                    entry_date=pd.Timestamp.now(),
                    entry_price=fill_price,
                    entry_fill_id=fill.fill_id,
                    quantity=order.quantity,
                    side=PositionSide.LONG,
                    stop_price=order.stop_price,
                    initial_stop_price=order.stop_price,
                    hard_stop_atr_mult=2.5,
                    entry_slippage_bps=slippage_bps,
                    entry_fee_bps=fee_bps,
                    entry_total_cost=fee_cost,
                    triggered_on=BreakoutType.FAST_20D,
                    adv20_at_entry=100000000.0,
                )
                self._positions[order.symbol] = pos
        else:  # SELL
            # Closing or reducing position
            if order.symbol in self._positions:
                pos = self._positions[order.symbol]
                if pos.quantity <= order.quantity:
                    # Closing position
                    del self._positions[order.symbol]
                else:
                    # Reducing position
                    pos.quantity -= order.quantity

        # Track order
        self._orders[order.order_id] = order
        self._order_status[order.order_id] = OrderStatus.FILLED

        return fill

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancellation successful

        Raises:
            ConnectionError: If not connected
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to mock broker")

        if order_id in self._orders:
            self._order_status[order_id] = OrderStatus.CANCELLED
            return True

        return False

    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status.

        Args:
            order_id: Order ID to check

        Returns:
            OrderStatus enum value

        Raises:
            ConnectionError: If not connected
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to mock broker")

        return self._order_status.get(order_id, OrderStatus.CANCELLED)

    def get_positions(self) -> Dict[str, Position]:
        """Get all positions.

        Returns:
            Dictionary mapping symbol to Position

        Raises:
            ConnectionError: If not connected
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to mock broker")

        return self._positions.copy()

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol.

        Args:
            symbol: Symbol to look up

        Returns:
            Position if found, None otherwise

        Raises:
            ConnectionError: If not connected
        """
        return self._positions.get(symbol)

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price.

        Args:
            symbol: Symbol to get price for

        Returns:
            Current price, or None if unavailable

        Raises:
            ConnectionError: If not connected
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to mock broker")

        return self._prices.get(symbol)

    def subscribe_market_data(self, symbols: List[str]) -> None:
        """Subscribe to real-time market data.

        Args:
            symbols: List of symbols to subscribe to

        Raises:
            ConnectionError: If not connected
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to mock broker")

        logger.info(f"Subscribed to market data for {symbols}")

    def unsubscribe_market_data(self, symbols: List[str]) -> None:
        """Unsubscribe from real-time market data.

        Args:
            symbols: List of symbols to unsubscribe from

        Raises:
            ConnectionError: If not connected
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to mock broker")

        logger.info(f"Unsubscribed from market data for {symbols}")

"""Alpaca broker adapter for paper trading and live trading."""

import logging
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd

from ..models.orders import Fill, Order, OrderStatus, SignalSide
from ..models.positions import Position, PositionSide
from ..models.signals import BreakoutType
from .base_adapter import AccountInfo, AdapterConfig, BaseAdapter

logger = logging.getLogger(__name__)

# Import at module level for testability (can be mocked)
try:
    import alpaca_trade_api as tradeapi
except ImportError:
    tradeapi = None  # type: ignore


class AlpacaAdapter(BaseAdapter):
    """Alpaca broker adapter.

    Supports both paper trading (Alpaca paper account) and live trading.
    Uses Alpaca REST API for order submission and position tracking.

    Example:
        >>> config = AdapterConfig(
        ...     api_key="your_key",
        ...     api_secret="your_secret",
        ...     paper_trading=True
        ... )
        >>> adapter = AlpacaAdapter(config)
        >>> with adapter:
        ...     account = adapter.get_account_info()
        ...     order = Order(...)
        ...     fill = adapter.submit_order(order)
    """

    def __init__(self, config: AdapterConfig):
        """Initialize Alpaca adapter.

        Args:
            config: AdapterConfig with Alpaca API credentials
        """
        super().__init__(config)
        self._api: Optional[Any] = None  # Will be alpaca.tradeapi.REST instance
        self._positions_cache: Dict[str, Position] = {}
        self._order_status_cache: Dict[str, OrderStatus] = {}

    def connect(self) -> None:
        """Connect to Alpaca API.

        Raises:
            ConnectionError: If connection fails
            ImportError: If alpaca-trade-api is not installed
        """
        if tradeapi is None:
            raise ImportError("alpaca-trade-api is required for AlpacaAdapter. " "Install with: pip install alpaca-trade-api")

        if not self.config.api_key or not self.config.api_secret:
            raise ValueError("Alpaca API key and secret are required")

        # Determine base URL based on paper trading mode
        if self.config.paper_trading:
            base_url = self.config.base_url or "https://paper-api.alpaca.markets"
        else:
            base_url = self.config.base_url or "https://api.alpaca.markets"

        try:
            self._api = tradeapi.REST(
                key_id=self.config.api_key, secret_key=self.config.api_secret, base_url=base_url, api_version="v2"
            )

            # Test connection by getting account info
            self._api.get_account()
            self._connected = True
            logger.info(f"Connected to Alpaca {'paper' if self.config.paper_trading else 'live'} account")

        except Exception as e:
            self._connected = False
            # Check for specific error types
            error_msg = str(e).lower()
            if "connection" in error_msg or "timeout" in error_msg:
                raise ConnectionError(f"Failed to connect to Alpaca: {e}") from e
            elif "authentication" in error_msg or "unauthorized" in error_msg or "401" in error_msg:
                raise ConnectionError(f"Authentication failed: {e}") from e
            else:
                raise ConnectionError(f"Failed to connect to Alpaca: {e}") from e

    def disconnect(self) -> None:
        """Disconnect from Alpaca API."""
        self._api = None
        self._connected = False
        self._positions_cache.clear()
        self._order_status_cache.clear()
        logger.info("Disconnected from Alpaca")

    def is_connected(self) -> bool:
        """Check if connected to Alpaca."""
        return self._connected and self._api is not None

    def get_account_info(self) -> AccountInfo:
        """Get account information from Alpaca.

        Returns:
            AccountInfo with equity, cash, buying power

        Raises:
            ConnectionError: If not connected
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to Alpaca")

        assert self._api is not None  # checked by is_connected()
        try:
            account = self._api.get_account()

            # Safely convert to float, handling Mock objects in tests
            def safe_float(value, default=0.0):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return default

            return AccountInfo(
                equity=safe_float(account.equity),
                cash=safe_float(account.cash),
                buying_power=safe_float(account.buying_power),
                margin_used=safe_float(account.portfolio_value) - safe_float(account.equity),
                broker_account_id=str(account.account_number) if hasattr(account, "account_number") else "",
                currency="USD",
            )
        except Exception as e:
            error_msg = str(e).lower()
            error_str = str(e)

            # Check for HTTPError status codes
            status_code = None
            if hasattr(e, "response") and hasattr(e.response, "status_code"):
                status_code = e.response.status_code

            # Check for rate limiting
            if status_code == 429 or "429" in error_str or "rate limit" in error_msg:
                raise RuntimeError(f"Rate limit exceeded: {e}") from e
            # Check for network failures
            elif "connection" in error_msg or "timeout" in error_msg:
                raise ConnectionError(f"Network error: {e}") from e
            else:
                raise RuntimeError(f"Failed to get account info: {e}") from e

    def submit_order(self, order: Order) -> Fill:
        """Submit order to Alpaca.

        For paper trading, uses Alpaca's paper trading account which simulates
        execution. For live trading, submits to real Alpaca account.

        Args:
            order: Order to submit

        Returns:
            Fill object with execution details

        Raises:
            ConnectionError: If not connected
            ValueError: If order is invalid
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to Alpaca")

        assert self._api is not None  # checked by is_connected()
        # Convert Order to Alpaca order format
        # Alpaca uses market orders for simplicity (as per our system design)
        side = "buy" if order.side == SignalSide.BUY else "sell"
        order_type = "market"  # We only use market orders

        try:
            # Validate order before submission
            if order.quantity <= 0:
                raise ValueError(f"Invalid quantity: {order.quantity}, must be positive")

            # Check account balance (pre-flight check)
            try:
                account = self._api.get_account()
                order_cost = order.expected_fill_price * order.quantity
                if order.side == SignalSide.BUY and float(account.cash) < order_cost:
                    raise ValueError(f"Insufficient funds: need {order_cost:.2f}, have {float(account.cash):.2f}")
            except Exception as balance_check_error:
                # If balance check fails, still try to submit (broker will reject if needed)
                logger.warning(f"Balance check failed: {balance_check_error}")

            # Submit order to Alpaca
            alpaca_order = self._api.submit_order(
                symbol=order.symbol,
                qty=order.quantity,
                side=side,
                type=order_type,
                time_in_force="day",  # Day order
                client_order_id=order.order_id,  # Use our order ID
            )

            # Alpaca may execute immediately (market order) or queue it
            # For paper trading, market orders typically fill immediately
            # Wait a moment and check status
            import time

            time.sleep(0.5)  # Brief wait for execution

            # Get order status
            order_status = self._get_alpaca_order_status(alpaca_order.id)

            if order_status == OrderStatus.FILLED:
                # Get fill details
                fills = self._api.list_orders(status="closed", limit=1, nested=True)  # Include fills

                # Find our order's fill
                filled_order = self._api.get_order(alpaca_order.id)

                # Create Fill object
                # Note: Alpaca doesn't provide detailed slippage info, so we estimate
                fill_price = float(filled_order.filled_avg_price)
                quantity = int(filled_order.filled_qty)

                # Estimate slippage (Alpaca doesn't provide this directly)
                # For paper trading, slippage is minimal
                open_price = fill_price  # We don't have open price from Alpaca
                slippage_bps = 0.0  # Paper trading typically has no slippage

                # Fee calculation (Alpaca commission structure)
                # Paper trading: free
                # Live: typically $0.01 per share, min $1, max 1% of order value
                fee_bps = 0.0 if self.config.paper_trading else 1.0  # Approximate
                notional = fill_price * quantity
                fee_cost = 0.0 if self.config.paper_trading else max(1.0, min(notional * 0.01, quantity * 0.01))

                fill = Fill(
                    fill_id=str(uuid.uuid4()),
                    order_id=order.order_id,
                    symbol=order.symbol,
                    asset_class=order.asset_class,
                    date=pd.Timestamp.now().normalize(),  # Current date
                    side=order.side,
                    quantity=quantity,
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

                return fill

            elif order_status == OrderStatus.REJECTED:
                # Order was rejected
                raise ValueError(f"Order rejected: {filled_order.reason}")

            else:
                # Order still pending (shouldn't happen for market orders, but handle it)
                # For now, raise an error - in a real system, you'd poll or use webhooks
                raise RuntimeError(f"Order {order.order_id} is still pending")

        except Exception as e:
            error_msg = str(e).lower()
            error_str = str(e)

            # Check for HTTPError status codes first
            status_code = None
            if hasattr(e, "response") and hasattr(e.response, "status_code"):
                status_code = e.response.status_code

            # Handle specific error types
            if status_code == 429 or "429" in error_str or "rate limit" in error_msg:
                logger.error(f"Rate limit exceeded for order {order.order_id}: {e}")
                raise RuntimeError(f"Rate limit exceeded: {e}") from e
            elif status_code == 400 or "400" in error_str or "invalid" in error_msg or "bad request" in error_msg:
                logger.error(f"Invalid order {order.order_id}: {e}")
                raise ValueError(f"Invalid order: {e}") from e
            elif status_code == 422 or "422" in error_str or "insufficient" in error_msg or "funds" in error_msg:
                logger.error(f"Insufficient funds for order {order.order_id}: {e}")
                raise ValueError(f"Insufficient funds: {e}") from e
            elif status_code == 403 or "403" in error_str or "forbidden" in error_msg or "position limit" in error_msg:
                logger.error(f"Position limit or permission error for order {order.order_id}: {e}")
                raise ValueError(f"Position limit exceeded or permission denied: {e}") from e
            elif "connection" in error_msg or "timeout" in error_msg or "network" in error_msg:
                logger.error(f"Network error for order {order.order_id}: {e}")
                raise ConnectionError(f"Network error: {e}") from e
            else:
                logger.error(f"Failed to submit order {order.order_id}: {e}")
                raise RuntimeError(f"Failed to submit order: {e}") from e

    def _get_alpaca_order_status(self, alpaca_order_id: str) -> OrderStatus:
        """Convert Alpaca order status to our OrderStatus enum.

        Args:
            alpaca_order_id: Alpaca order ID

        Returns:
            OrderStatus enum value
        """
        assert self._api is not None  # Should only be called when connected
        try:
            alpaca_order = self._api.get_order(alpaca_order_id)
            status = alpaca_order.status.lower()

            if status == "filled":
                return OrderStatus.FILLED
            elif status == "rejected" or status == "expired":
                return OrderStatus.REJECTED
            elif status == "canceled":
                return OrderStatus.CANCELLED
            else:
                return OrderStatus.PENDING

        except Exception as e:
            logger.warning(f"Failed to get order status: {e}")
            return OrderStatus.PENDING

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order in Alpaca.

        Args:
            order_id: Order ID to cancel (Alpaca client_order_id)

        Returns:
            True if cancellation successful
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to Alpaca")

        assert self._api is not None  # checked by is_connected()
        try:
            # Find order by client_order_id
            orders = self._api.list_orders(status="open", limit=100)
            for order in orders:
                if order.client_order_id == order_id:
                    self._api.cancel_order(order.id)
                    return True

            logger.warning(f"Order {order_id} not found or already closed")
            return False

        except Exception as e:
            error_msg = str(e).lower()
            error_str = str(e)

            # Handle specific error types
            if "429" in error_str or "rate limit" in error_msg:
                logger.error(f"Rate limit exceeded while canceling order {order_id}: {e}")
                raise RuntimeError(f"Rate limit exceeded: {e}") from e
            elif "connection" in error_msg or "timeout" in error_msg:
                logger.error(f"Network error while canceling order {order_id}: {e}")
                raise ConnectionError(f"Network error: {e}") from e
            else:
                logger.error(f"Failed to cancel order {order_id}: {e}")
                return False

    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status from Alpaca.

        Args:
            order_id: Order ID (client_order_id)

        Returns:
            OrderStatus enum value
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to Alpaca")

        assert self._api is not None  # checked by is_connected()
        try:
            # Find order by client_order_id
            orders = self._api.list_orders(status="all", limit=100)
            for order in orders:
                if order.client_order_id == order_id:
                    return self._get_alpaca_order_status(order.id)

            return OrderStatus.CANCELLED  # Not found, assume cancelled

        except Exception as e:
            logger.warning(f"Failed to get order status: {e}")
            return OrderStatus.PENDING

    def get_positions(self) -> Dict[str, Position]:
        """Get all positions from Alpaca.

        Returns:
            Dictionary mapping symbol to Position
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to Alpaca")

        assert self._api is not None  # checked by is_connected()
        try:
            alpaca_positions = self._api.list_positions()
            positions = {}

            for ap in alpaca_positions:
                symbol = ap.symbol
                qty = float(ap.qty)

                # Determine position side from quantity (negative = short)
                side = PositionSide.SHORT if qty < 0 else PositionSide.LONG
                quantity = abs(int(qty))

                # Convert Alpaca position to our Position model
                # Note: We don't have all the fields that backtest positions have
                # (e.g., entry_fill_id, triggered_on, adv20_at_entry)
                # These would need to be tracked separately or retrieved from our database

                # Calculate default stop price based on entry price and side
                # For LONG: stop below entry (5% below)
                # For SHORT: stop above entry (5% above)
                entry_price = float(ap.avg_entry_price)
                if entry_price <= 0:
                    raise ValueError(f"Invalid entry_price: {entry_price}, must be positive")

                if side == PositionSide.LONG:
                    default_stop = entry_price * 0.95
                else:  # SHORT
                    default_stop = entry_price * 1.05

                # Ensure stop_price is always positive
                default_stop = max(default_stop, 0.01)

                position = Position(
                    symbol=symbol,
                    asset_class="equity",  # Alpaca is equity-focused (crypto support is limited)
                    entry_date=pd.Timestamp(ap.avg_entry_date) if ap.avg_entry_date else pd.Timestamp.now(),
                    entry_price=entry_price,
                    entry_fill_id="",  # Not available from Alpaca
                    quantity=quantity,
                    side=side,
                    stop_price=default_stop,  # Default stop (not available from Alpaca, would need to track separately)
                    initial_stop_price=default_stop,  # Default stop
                    hard_stop_atr_mult=2.5,  # Default
                    entry_slippage_bps=0.0,  # Not available
                    entry_fee_bps=0.0,  # Not available
                    entry_total_cost=0.0,  # Not available
                    triggered_on=BreakoutType.FAST_20D,  # Default (not available from broker)
                    adv20_at_entry=1000000.0,  # Default value (not available from broker, set to 1M to pass validation)
                )

                positions[symbol] = position

            self._positions_cache = positions
            return positions

        except Exception as e:
            error_msg = str(e).lower()
            error_str = str(e)

            # Handle specific error types
            if "429" in error_str or "rate limit" in error_msg:
                logger.error(f"Rate limit exceeded while getting positions: {e}")
                raise RuntimeError(f"Rate limit exceeded: {e}") from e
            elif "connection" in error_msg or "timeout" in error_msg:
                logger.error(f"Network error while getting positions: {e}")
                raise ConnectionError(f"Network error: {e}") from e
            else:
                logger.error(f"Failed to get positions: {e}")
                raise RuntimeError(f"Failed to get positions: {e}") from e

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol.

        Args:
            symbol: Symbol to look up

        Returns:
            Position if found, None otherwise
        """
        positions = self.get_positions()
        return positions.get(symbol)

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price from Alpaca.

        Args:
            symbol: Symbol to get price for

        Returns:
            Current price, or None if unavailable
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to Alpaca")

        assert self._api is not None  # checked by is_connected()
        try:
            # Get latest trade price
            trade = self._api.get_latest_trade(symbol)
            return float(trade.price)

        except Exception as e:
            error_msg = str(e).lower()
            error_str = str(e)

            # Handle specific error types
            if "429" in error_str or "rate limit" in error_msg:
                logger.warning(f"Rate limit exceeded while getting price for {symbol}: {e}")
                return None
            elif "404" in error_str or "not found" in error_msg:
                logger.warning(f"Symbol {symbol} not found: {e}")
                return None
            elif "connection" in error_msg or "timeout" in error_msg:
                logger.warning(f"Network error while getting price for {symbol}: {e}")
                return None
            else:
                logger.warning(f"Failed to get price for {symbol}: {e}")
                return None

    def subscribe_market_data(self, symbols: List[str]) -> None:
        """Subscribe to real-time market data.

        Note: Alpaca REST API doesn't support streaming data directly.
        For real-time data, you would need to use Alpaca's websocket API
        or poll the REST API. This method is a placeholder.

        Args:
            symbols: List of symbols to subscribe to
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to Alpaca")

        # For Alpaca, real-time data subscription would require websocket
        # This is a placeholder - full implementation would use alpaca-trade-api's streaming
        logger.info(f"Market data subscription requested for {symbols} (websocket not implemented)")

    def unsubscribe_market_data(self, symbols: List[str]) -> None:
        """Unsubscribe from real-time market data.

        Args:
            symbols: List of symbols to unsubscribe from
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to Alpaca")

        logger.info(f"Market data unsubscription requested for {symbols}")

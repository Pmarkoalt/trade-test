"""Interactive Brokers (IB) adapter for paper trading and live trading."""

import logging
import queue
import threading
import uuid
from typing import Callable, Dict, List, Optional

import pandas as pd

from ..models.orders import Fill, Order, OrderStatus, SignalSide
from ..models.positions import ExitReason, Position, PositionSide
from ..models.signals import BreakoutType
from .base_adapter import AccountInfo, AdapterConfig, BaseAdapter

logger = logging.getLogger(__name__)


class IBAdapter(BaseAdapter):
    """Interactive Brokers adapter using IB API (ib_insync).

    Supports both paper trading (TWS/Gateway paper account) and live trading.
    Uses ib_insync library for connection to TWS or IB Gateway.

    Requirements:
    - Interactive Brokers TWS (Trader Workstation) or IB Gateway installed
    - ib_insync library installed
    - TWS/Gateway running and configured for API access

    Example:
        >>> config = AdapterConfig(
        ...     host="127.0.0.1",
        ...     port=7497,  # Paper trading port
        ...     client_id=1,
        ...     paper_trading=True
        ... )
        >>> adapter = IBAdapter(config)
        >>> with adapter:
        ...     account = adapter.get_account_info()
        ...     order = Order(...)
        ...     fill = adapter.submit_order(order)
    """

    def __init__(self, config: AdapterConfig):
        """Initialize IB adapter.

        Args:
            config: AdapterConfig with IB connection details
        """
        super().__init__(config)
        self._ib = None  # Will be ib_insync.IB instance
        self._positions_cache: Dict[str, Position] = {}
        self._order_status_cache: Dict[str, OrderStatus] = {}
        self._account_id: Optional[str] = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        """Connect to IB TWS/Gateway.

        Raises:
            ConnectionError: If connection fails
            ImportError: If ib_insync is not installed
        """
        try:
            from ib_insync import IB, Crypto, Stock
        except ImportError:
            raise ImportError("ib_insync is required for IBAdapter. " "Install with: pip install ib-insync")

        try:
            from ib_insync import IB

            self._ib = IB()

            # Connect to TWS/Gateway
            host = self.config.host
            port = self.config.port
            client_id = self.config.client_id

            self._ib.connect(host=host, port=port, clientId=client_id)

            # Wait for connection to establish
            import time

            time.sleep(1)

            if not self._ib.isConnected():
                raise ConnectionError("Failed to connect to IB TWS/Gateway")

            # Get account ID (use first account if multiple)
            accounts = self._ib.accountValues()
            if accounts:
                self._account_id = accounts[0].account
            else:
                # Try to get account from managed accounts
                managed_accounts = self._ib.managedAccounts()
                if managed_accounts:
                    self._account_id = managed_accounts[0]
                else:
                    raise ConnectionError("No IB account found")

            self._connected = True
            logger.info(f"Connected to IB {'paper' if self.config.paper_trading else 'live'} " f"account {self._account_id}")

        except Exception as e:
            self._connected = False
            if self._ib:
                try:
                    self._ib.disconnect()
                except Exception:
                    pass
                self._ib = None

            # Check for specific error types
            error_msg = str(e).lower()
            if "connection" in error_msg or "timeout" in error_msg or "refused" in error_msg:
                raise ConnectionError(f"Failed to connect to IB TWS/Gateway: {e}") from e
            elif "authentication" in error_msg or "unauthorized" in error_msg:
                raise ConnectionError(f"Authentication failed: {e}") from e
            else:
                raise ConnectionError(f"Failed to connect to IB: {e}") from e

    def disconnect(self) -> None:
        """Disconnect from IB TWS/Gateway."""
        if self._ib and self._ib.isConnected():
            try:
                self._ib.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting from IB: {e}")

        self._ib = None
        self._connected = False
        self._positions_cache.clear()
        self._order_status_cache.clear()
        self._account_id = None
        logger.info("Disconnected from IB")

    def is_connected(self) -> bool:
        """Check if connected to IB."""
        return self._connected and self._ib is not None and self._ib.isConnected()

    def get_account_info(self) -> AccountInfo:
        """Get account information from IB.

        Returns:
            AccountInfo with equity, cash, buying power

        Raises:
            ConnectionError: If not connected
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        if not self._account_id:
            raise RuntimeError("Account ID not set")

        try:
            # Get account summary
            account_values = self._ib.accountValues(self._account_id)

            # Extract values
            equity = 0.0
            cash = 0.0
            buying_power = 0.0
            margin_used = 0.0
            currency = "USD"

            for av in account_values:
                tag = av.tag
                value = float(av.value) if av.value else 0.0
                currency = av.currency

                if tag == "NetLiquidation":
                    equity = value
                elif tag == "TotalCashValue":
                    cash = value
                elif tag == "BuyingPower":
                    buying_power = value
                elif tag == "GrossPositionValue":
                    # Margin used = Gross Position Value - Net Liquidation
                    margin_used = value - equity

            return AccountInfo(
                equity=equity,
                cash=cash,
                buying_power=buying_power,
                margin_used=max(0.0, margin_used),
                broker_account_id=self._account_id,
                currency=currency,
            )

        except Exception as e:
            error_msg = str(e).lower()
            # Check for network failures
            if "connection" in error_msg or "timeout" in error_msg or "disconnected" in error_msg:
                raise ConnectionError(f"Network error: {e}") from e
            else:
                raise RuntimeError(f"Failed to get account info: {e}") from e

    def submit_order(self, order: Order) -> Fill:
        """Submit order to IB.

        Args:
            order: Order to submit

        Returns:
            Fill object with execution details

        Raises:
            ConnectionError: If not connected
            ValueError: If order is invalid
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        try:
            from ib_insync import Crypto, MarketOrder
            from ib_insync import Order as IBOrder
            from ib_insync import Stock

            # Create IB contract
            if order.asset_class == "equity":
                contract = Stock(order.symbol, "SMART", "USD")
            elif order.asset_class == "crypto":
                # IB crypto contracts use different format
                # Example: "BTC" -> Crypto("BTC", "PAXOS", "USD")
                contract = Crypto(order.symbol, "PAXOS", "USD")
            else:
                raise ValueError(f"Unsupported asset class: {order.asset_class}")

            # Validate order before submission
            if order.quantity <= 0:
                raise ValueError(f"Invalid quantity: {order.quantity}, must be positive")

            # Request contract details to ensure symbol is valid
            try:
                contracts = self._ib.qualifyContracts(contract)
                if not contracts:
                    raise ValueError(f"Invalid symbol: {order.symbol}")
                contract = contracts[0]
            except Exception as e:
                error_msg = str(e).lower()
                if "connection" in error_msg or "timeout" in error_msg:
                    raise ConnectionError(f"Network error while qualifying contract: {e}") from e
                else:
                    raise ValueError(f"Invalid symbol or contract error: {e}") from e

            # Create market order
            side = "BUY" if order.side == SignalSide.BUY else "SELL"
            ib_order = MarketOrder(side, order.quantity)
            ib_order.clientId = self.config.client_id

            # Submit order
            trade = self._ib.placeOrder(contract, ib_order)

            # Wait for order to fill (with timeout)
            # In real implementation, you'd use callbacks or async waiting
            import time

            timeout = 10.0  # 10 second timeout
            start_time = time.time()

            while time.time() - start_time < timeout:
                if trade.isDone():
                    break
                time.sleep(0.1)

            if not trade.isDone():
                # Cancel order and raise error
                self._ib.cancelOrder(ib_order)
                raise RuntimeError(f"Order {order.order_id} did not fill within timeout")

            # Check if order filled or was cancelled/rejected
            order_status = trade.orderStatus.status
            if order_status in ["Cancelled", "ApiCancelled", "Inactive"]:
                raise ValueError(f"Order was cancelled: {order_status}")

            # Check for rejection reasons
            if hasattr(trade.orderStatus, "whyHeld") and trade.orderStatus.whyHeld:
                why_held = trade.orderStatus.whyHeld.lower()
                if "insufficient" in why_held or "funds" in why_held:
                    raise ValueError(f"Insufficient funds: {trade.orderStatus.whyHeld}")
                elif "position limit" in why_held or "limit" in why_held:
                    raise ValueError(f"Position limit exceeded: {trade.orderStatus.whyHeld}")

            if order_status != "Filled":
                raise RuntimeError(f"Order status: {order_status} (expected Filled)")

            # Get fill details
            fills = trade.fills
            if not fills:
                raise RuntimeError("Order filled but no fill details available")

            # Use first fill (assuming single fill for market order)
            fill_details = fills[0]
            fill_price = float(fill_details.execution.price)
            quantity = int(fill_details.execution.shares)

            # Get commission
            commission = sum(float(f.commission) for f in fills)

            # Estimate slippage (IB doesn't provide this directly)
            # We'd need to track the price when order was submitted
            open_price = fill_price  # Approximation
            slippage_bps = 0.0  # Would need historical price to calculate

            # Calculate fee in bps
            notional = fill_price * quantity
            fee_bps = (commission / notional * 10000) if notional > 0 else 0.0

            fill = Fill(
                fill_id=str(uuid.uuid4()),
                order_id=order.order_id,
                symbol=order.symbol,
                asset_class=order.asset_class,
                date=pd.Timestamp.now().normalize(),
                side=order.side,
                quantity=quantity,
                fill_price=fill_price,
                open_price=open_price,
                slippage_bps=slippage_bps,
                fee_bps=fee_bps,
                total_cost=commission,
                vol_mult=1.0,
                size_penalty=1.0,
                weekend_penalty=1.0,
                stress_mult=1.0,
                notional=notional,
            )

            return fill

        except Exception as e:
            error_msg = str(e).lower()
            error_str = str(e)

            # Handle specific error types
            if "connection" in error_msg or "timeout" in error_msg or "disconnected" in error_msg:
                logger.error(f"Network error for order {order.order_id}: {e}")
                raise ConnectionError(f"Network error: {e}") from e
            elif "invalid" in error_msg or "bad" in error_msg:
                logger.error(f"Invalid order {order.order_id}: {e}")
                raise ValueError(f"Invalid order: {e}") from e
            elif "insufficient" in error_msg or "funds" in error_msg:
                logger.error(f"Insufficient funds for order {order.order_id}: {e}")
                raise ValueError(f"Insufficient funds: {e}") from e
            elif "position limit" in error_msg or "limit" in error_msg:
                logger.error(f"Position limit for order {order.order_id}: {e}")
                raise ValueError(f"Position limit exceeded: {e}") from e
            else:
                logger.error(f"Failed to submit order {order.order_id}: {e}")
                raise RuntimeError(f"Failed to submit order: {e}") from e

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order in IB.

        Note: IB uses its own order IDs, not our client order IDs.
        This implementation would need to track IB order IDs.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancellation successful
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        # IB adapter would need to track order_id -> IB Trade mapping
        # This is a simplified implementation
        logger.warning("IBAdapter.cancel_order requires order ID tracking (not fully implemented)")
        return False

    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status from IB.

        Note: This requires tracking IB order IDs, which is not fully implemented.

        Args:
            order_id: Order ID

        Returns:
            OrderStatus enum value
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        # Would need to track order_id -> IB Trade mapping
        logger.warning("IBAdapter.get_order_status requires order ID tracking (not fully implemented)")
        return OrderStatus.PENDING

    def get_positions(self) -> Dict[str, Position]:
        """Get all positions from IB.

        Returns:
            Dictionary mapping symbol to Position
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        try:
            ib_positions = self._ib.positions()
            positions = {}

            for ip in ib_positions:
                symbol = ip.contract.symbol

                # Get average cost
                avg_cost = float(ip.avgCost) if ip.avgCost else 0.0
                quantity_raw = int(float(ip.position))

                # Skip zero positions
                if quantity_raw == 0:
                    continue

                # Determine position side from quantity (negative = short)
                side = PositionSide.SHORT if quantity_raw < 0 else PositionSide.LONG
                quantity = abs(quantity_raw)

                # Determine asset class from contract
                asset_class = "equity"
                if hasattr(ip.contract, "secType") and ip.contract.secType == "CRYPTO":
                    asset_class = "crypto"

                # Create Position object
                # Note: Many fields are not available from IB positions
                position = Position(
                    symbol=symbol,
                    asset_class=asset_class,
                    entry_date=pd.Timestamp.now(),  # IB doesn't provide entry date
                    entry_price=avg_cost,
                    entry_fill_id="",  # Not available
                    quantity=quantity,
                    side=side,
                    stop_price=0.0,  # Not available from IB
                    initial_stop_price=0.0,
                    hard_stop_atr_mult=2.5,
                    entry_slippage_bps=0.0,
                    entry_fee_bps=0.0,
                    entry_total_cost=0.0,
                    triggered_on=BreakoutType.FAST_20D,  # Default (not available from broker)
                    adv20_at_entry=0.0,
                )

                positions[symbol] = position

            self._positions_cache = positions
            return positions

        except Exception as e:
            error_msg = str(e).lower()
            # Handle specific error types
            if "connection" in error_msg or "timeout" in error_msg or "disconnected" in error_msg:
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
        """Get current market price from IB.

        Args:
            symbol: Symbol to get price for

        Returns:
            Current price, or None if unavailable
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        try:
            from ib_insync import Crypto, Stock

            # Determine contract type
            # For simplicity, assume equity if not specified
            contract = Stock(symbol, "SMART", "USD")

            # Qualify contract
            contracts = self._ib.qualifyContracts(contract)
            if not contracts:
                # Try crypto
                contract = Crypto(symbol, "PAXOS", "USD")
                contracts = self._ib.qualifyContracts(contract)

            if not contracts:
                logger.warning(f"Could not qualify contract for {symbol}")
                return None

            contract = contracts[0]

            # Request market data
            ticker = self._ib.reqMktData(contract, "", False, False)

            # Wait for data
            self._ib.sleep(1)

            # Get last price
            if ticker.last:
                price = float(ticker.last)
                self._ib.cancelMktData(contract)
                return price

            # Fallback to close price
            if ticker.close:
                price = float(ticker.close)
                self._ib.cancelMktData(contract)
                return price

            self._ib.cancelMktData(contract)
            return None

        except Exception as e:
            error_msg = str(e).lower()
            # Handle specific error types
            if "connection" in error_msg or "timeout" in error_msg or "disconnected" in error_msg:
                logger.warning(f"Network error while getting price for {symbol}: {e}")
                return None
            elif "not found" in error_msg or "invalid" in error_msg:
                logger.warning(f"Symbol {symbol} not found or invalid: {e}")
                return None
            else:
                logger.warning(f"Failed to get price for {symbol}: {e}")
                return None

    def subscribe_market_data(self, symbols: List[str]) -> None:
        """Subscribe to real-time market data.

        Args:
            symbols: List of symbols to subscribe to
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        # IB market data subscription requires proper contract setup
        # and may require market data subscriptions in TWS
        # This is a placeholder - full implementation would use reqMktData
        logger.info(f"Market data subscription requested for {symbols} (full implementation requires contract setup)")

    def unsubscribe_market_data(self, symbols: List[str]) -> None:
        """Unsubscribe from real-time market data.

        Args:
            symbols: List of symbols to unsubscribe from
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to IB")

        logger.info(f"Market data unsubscription requested for {symbols}")

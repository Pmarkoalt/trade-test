"""Alpaca broker adapter for paper trading and live trading."""

import logging
import uuid
from typing import Dict, List, Optional
import pandas as pd

from ..models.orders import Order, Fill, OrderStatus, SignalSide
from ..models.positions import Position, ExitReason
from ..models.signals import BreakoutType
from .base_adapter import BaseAdapter, AdapterConfig, AccountInfo

logger = logging.getLogger(__name__)


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
        self._api = None  # Will be alpaca.tradeapi.REST instance
        self._positions_cache: Dict[str, Position] = {}
        self._order_status_cache: Dict[str, OrderStatus] = {}
    
    def connect(self) -> None:
        """Connect to Alpaca API.
        
        Raises:
            ConnectionError: If connection fails
            ImportError: If alpaca-trade-api is not installed
        """
        try:
            import alpaca_trade_api as tradeapi
        except ImportError:
            raise ImportError(
                "alpaca-trade-api is required for AlpacaAdapter. "
                "Install with: pip install alpaca-trade-api"
            )
        
        if not self.config.api_key or not self.config.api_secret:
            raise ValueError("Alpaca API key and secret are required")
        
        # Determine base URL based on paper trading mode
        if self.config.paper_trading:
            base_url = self.config.base_url or "https://paper-api.alpaca.markets"
        else:
            base_url = self.config.base_url or "https://api.alpaca.markets"
        
        try:
            self._api = tradeapi.REST(
                key_id=self.config.api_key,
                secret_key=self.config.api_secret,
                base_url=base_url,
                api_version='v2'
            )
            
            # Test connection by getting account info
            account = self._api.get_account()
            self._connected = True
            logger.info(f"Connected to Alpaca {'paper' if self.config.paper_trading else 'live'} account")
        
        except Exception as e:
            self._connected = False
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
        
        try:
            account = self._api.get_account()
            
            return AccountInfo(
                equity=float(account.equity),
                cash=float(account.cash),
                buying_power=float(account.buying_power),
                margin_used=float(account.portfolio_value) - float(account.equity),
                broker_account_id=account.account_number,
                currency="USD"
            )
        except Exception as e:
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
        
        # Convert Order to Alpaca order format
        # Alpaca uses market orders for simplicity (as per our system design)
        side = "buy" if order.side == SignalSide.BUY else "sell"
        order_type = "market"  # We only use market orders
        
        try:
            # Submit order to Alpaca
            alpaca_order = self._api.submit_order(
                symbol=order.symbol,
                qty=order.quantity,
                side=side,
                type=order_type,
                time_in_force="day",  # Day order
                client_order_id=order.order_id  # Use our order ID
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
                fills = self._api.list_orders(
                    status="closed",
                    limit=1,
                    nested=True  # Include fills
                )
                
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
                    notional=notional
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
            logger.error(f"Failed to submit order {order.order_id}: {e}")
            raise RuntimeError(f"Failed to submit order: {e}") from e
    
    def _get_alpaca_order_status(self, alpaca_order_id: str) -> OrderStatus:
        """Convert Alpaca order status to our OrderStatus enum.
        
        Args:
            alpaca_order_id: Alpaca order ID
        
        Returns:
            OrderStatus enum value
        """
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
        
        try:
            alpaca_positions = self._api.list_positions()
            positions = {}
            
            for ap in alpaca_positions:
                symbol = ap.symbol
                
                # Convert Alpaca position to our Position model
                # Note: We don't have all the fields that backtest positions have
                # (e.g., entry_fill_id, triggered_on, adv20_at_entry)
                # These would need to be tracked separately or retrieved from our database
                
                position = Position(
                    symbol=symbol,
                    asset_class="equity",  # Alpaca is equity-focused (crypto support is limited)
                    entry_date=pd.Timestamp(ap.avg_entry_date) if ap.avg_entry_date else pd.Timestamp.now(),
                    entry_price=float(ap.avg_entry_price),
                    entry_fill_id="",  # Not available from Alpaca
                    quantity=int(float(ap.qty)),
                    stop_price=0.0,  # Not available from Alpaca (would need to track separately)
                    initial_stop_price=0.0,  # Not available
                    hard_stop_atr_mult=2.5,  # Default
                    entry_slippage_bps=0.0,  # Not available
                    entry_fee_bps=0.0,  # Not available
                    entry_total_cost=0.0,  # Not available
                    triggered_on=BreakoutType.FAST_20D,  # Default (not available from broker)
                    adv20_at_entry=0.0  # Not available
                )
                
                positions[symbol] = position
            
            self._positions_cache = positions
            return positions
        
        except Exception as e:
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
        
        try:
            # Get latest trade price
            trade = self._api.get_latest_trade(symbol)
            return float(trade.price)
        
        except Exception as e:
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


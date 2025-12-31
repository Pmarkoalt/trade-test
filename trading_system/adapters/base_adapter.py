"""Base adapter interface for broker APIs."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd

from ..models.orders import Order, Fill, OrderStatus
from ..models.positions import Position


@dataclass
class AdapterConfig:
    """Configuration for broker adapter."""
    
    # Broker connection
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    base_url: Optional[str] = None  # For REST APIs
    
    # Paper trading mode (if True, uses paper trading account/simulation)
    paper_trading: bool = True
    
    # For Interactive Brokers (TWS/Gateway connection)
    host: str = "127.0.0.1"
    port: int = 7497  # Default TWS paper trading port (7496 for live)
    client_id: int = 1
    
    # Additional broker-specific config
    extra_config: Dict = None
    
    def __post_init__(self):
        """Initialize extra_config if not provided."""
        if self.extra_config is None:
            self.extra_config = {}


@dataclass
class AccountInfo:
    """Account information from broker."""
    
    equity: float  # Current account equity
    cash: float  # Available cash
    buying_power: float  # Available buying power
    margin_used: float  # Margin used (if applicable)
    
    # Broker-specific fields
    broker_account_id: str  # Broker account ID
    currency: str = "USD"  # Account currency


class BaseAdapter(ABC):
    """Base interface for broker adapters.
    
    All broker adapters must implement this interface to provide a consistent
    API for order submission, position tracking, and market data access.
    """
    
    def __init__(self, config: AdapterConfig):
        """Initialize adapter with configuration.
        
        Args:
            config: AdapterConfig with broker connection details
        """
        self.config = config
        self._connected = False
    
    @abstractmethod
    def connect(self) -> None:
        """Connect to broker API.
        
        Raises:
            ConnectionError: If connection fails
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from broker API."""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if adapter is connected to broker.
        
        Returns:
            True if connected, False otherwise
        """
        pass
    
    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """Get current account information from broker.
        
        Returns:
            AccountInfo with equity, cash, buying power, etc.
        
        Raises:
            ConnectionError: If not connected
            RuntimeError: If account info cannot be retrieved
        """
        pass
    
    @abstractmethod
    def submit_order(self, order: Order) -> Fill:
        """Submit order to broker and get fill.
        
        For paper trading mode, this may simulate execution or use broker's
        paper trading account. For live trading, submits to real broker.
        
        Args:
            order: Order to submit
        
        Returns:
            Fill object with execution details
        
        Raises:
            ConnectionError: If not connected
            ValueError: If order is invalid
            RuntimeError: If order submission fails
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.
        
        Args:
            order_id: Order ID to cancel
        
        Returns:
            True if cancellation successful, False otherwise
        
        Raises:
            ConnectionError: If not connected
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get current status of an order.
        
        Args:
            order_id: Order ID to check
        
        Returns:
            OrderStatus enum value
        
        Raises:
            ConnectionError: If not connected
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions from broker.
        
        Returns:
            Dictionary mapping symbol to Position object
        
        Raises:
            ConnectionError: If not connected
        """
        pass
    
    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol.
        
        Args:
            symbol: Symbol to look up
        
        Returns:
            Position if found, None otherwise
        
        Raises:
            ConnectionError: If not connected
        """
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol.
        
        Args:
            symbol: Symbol to get price for
        
        Returns:
            Current price, or None if unavailable
        
        Raises:
            ConnectionError: If not connected
        """
        pass
    
    @abstractmethod
    def subscribe_market_data(self, symbols: List[str]) -> None:
        """Subscribe to real-time market data for symbols.
        
        Args:
            symbols: List of symbols to subscribe to
        
        Raises:
            ConnectionError: If not connected
        """
        pass
    
    @abstractmethod
    def unsubscribe_market_data(self, symbols: List[str]) -> None:
        """Unsubscribe from real-time market data.
        
        Args:
            symbols: List of symbols to unsubscribe from
        
        Raises:
            ConnectionError: If not connected
        """
        pass
    
    def __enter__(self):
        """Context manager entry: connect to broker."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: disconnect from broker."""
        self.disconnect()


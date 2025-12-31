"""Custom exception classes for the trading system.

This module provides specific exception types for different error categories,
enabling better error handling and debugging throughout the system.
"""

from typing import Optional, Dict, Any, List


class TradingSystemError(Exception):
    """Base exception for all trading system errors."""
    pass


class DataError(TradingSystemError):
    """Base exception for data-related errors."""
    
    def __init__(
        self,
        message: str,
        symbol: Optional[str] = None,
        date: Optional[str] = None,
        data_path: Optional[str] = None,
        file_path: Optional[str] = None
    ):
        super().__init__(message)
        self.symbol = symbol
        self.date = date
        self.data_path = data_path
        self.file_path = file_path


class DataValidationError(DataError):
    """Raised when data validation fails."""
    pass


class DataNotFoundError(DataError):
    """Raised when requested data is not found."""
    pass


class DataSourceError(DataError):
    """Raised when data source operations fail."""
    
    def __init__(
        self,
        message: str,
        symbol: Optional[str] = None,
        date: Optional[str] = None,
        source_type: Optional[str] = None,
        data_path: Optional[str] = None,
        file_path: Optional[str] = None
    ):
        super().__init__(message, symbol, date, data_path, file_path)
        self.source_type = source_type


class ConfigurationError(TradingSystemError):
    """Base exception for configuration errors."""
    
    def __init__(
        self,
        message: str,
        config_path: Optional[str] = None,
        field: Optional[str] = None,
        errors: Optional[List[Dict[str, Any]]] = None
    ):
        super().__init__(message)
        self.config_path = config_path
        self.field = field
        self.errors = errors or []


class StrategyError(TradingSystemError):
    """Base exception for strategy-related errors."""
    
    def __init__(self, message: str, strategy_name: Optional[str] = None, symbol: Optional[str] = None):
        super().__init__(message)
        self.strategy_name = strategy_name
        self.symbol = symbol


class StrategyNotFoundError(StrategyError):
    """Raised when a strategy class cannot be found."""
    pass


class PortfolioError(TradingSystemError):
    """Base exception for portfolio-related errors."""
    
    def __init__(self, message: str, symbol: Optional[str] = None, position_id: Optional[str] = None):
        super().__init__(message)
        self.symbol = symbol
        self.position_id = position_id


class InsufficientCapitalError(PortfolioError):
    """Raised when there's insufficient capital for a trade."""
    pass


class PositionNotFoundError(PortfolioError):
    """Raised when a position is not found."""
    pass


class ExecutionError(TradingSystemError):
    """Base exception for execution-related errors."""
    
    def __init__(self, message: str, order_id: Optional[str] = None, symbol: Optional[str] = None):
        super().__init__(message)
        self.order_id = order_id
        self.symbol = symbol


class OrderRejectedError(ExecutionError):
    """Raised when an order is rejected."""
    pass


class FillError(ExecutionError):
    """Raised when order fill simulation fails."""
    pass


class IndicatorError(TradingSystemError):
    """Raised when indicator calculation fails."""
    
    def __init__(self, message: str, indicator_name: Optional[str] = None, symbol: Optional[str] = None):
        super().__init__(message)
        self.indicator_name = indicator_name
        self.symbol = symbol


class BacktestError(TradingSystemError):
    """Base exception for backtest-related errors."""
    
    def __init__(self, message: str, date: Optional[str] = None, step: Optional[str] = None):
        super().__init__(message)
        self.date = date
        self.step = step


class ValidationError(TradingSystemError):
    """Base exception for validation errors."""
    
    def __init__(self, message: str, validation_type: Optional[str] = None):
        super().__init__(message)
        self.validation_type = validation_type


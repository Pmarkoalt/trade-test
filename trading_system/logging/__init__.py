"""Enhanced logging module with structured logging, performance metrics, and event tracking."""

from .logger import (
    setup_logging,
    get_logger,
    log_trade_event,
    log_signal_generation,
    log_portfolio_snapshot,
    log_performance_metric,
    TradeEventType,
)

__all__ = [
    'setup_logging',
    'get_logger',
    'log_trade_event',
    'log_signal_generation',
    'log_portfolio_snapshot',
    'log_performance_metric',
    'TradeEventType',
]


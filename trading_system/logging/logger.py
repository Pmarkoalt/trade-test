"""Enhanced logging with structured logging, performance metrics, and event tracking."""

import json
import logging
import logging.handlers
import sys
import time
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import pandas as pd

if TYPE_CHECKING:
    from ..configs.run_config import RunConfig

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

import os

try:
    from loguru import logger as loguru_logger

    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False

try:
    from rich.console import Console
    from rich.logging import RichHandler

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class TradeEventType(str, Enum):
    """Types of trade events to log."""

    ENTRY = "entry"
    EXIT = "exit"
    STOP_HIT = "stop_hit"
    REJECTED = "rejected"


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
            ]:
                log_data[key] = value

        return json.dumps(log_data, default=str)


class RotatingFileHandler(logging.handlers.RotatingFileHandler):
    """File handler with rotation support."""

    pass


class PerformanceContext:
    """Context manager for performance timing."""

    def __init__(self, logger: logging.Logger, operation: str, log_memory: bool = True):
        """Initialize performance context.

        Args:
            logger: Logger instance
            operation: Operation name
            log_memory: Whether to log memory usage
        """
        self.logger = logger
        self.operation = operation
        self.log_memory = log_memory
        from typing import Optional
        self.start_time: Optional[float] = None
        self.start_memory: Optional[float] = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        if self.log_memory and PSUTIL_AVAILABLE:
            try:
                process = psutil.Process(os.getpid())
                self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
            except Exception:
                self.start_memory = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log performance."""
        if self.start_time is None:
            return False
        elapsed = time.perf_counter() - self.start_time

        metrics = {
            "operation": self.operation,
            "duration_seconds": elapsed,
        }

        if self.log_memory and PSUTIL_AVAILABLE:
            try:
                process = psutil.Process(os.getpid())
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
                metrics["memory_mb"] = end_memory
                if self.start_memory is not None:
                    metrics["memory_delta_mb"] = end_memory - self.start_memory
            except Exception:
                pass

        memory_mb_val = metrics.get("memory_mb")
        memory_delta_mb_val = metrics.get("memory_delta_mb")
        memory_mb_float: Optional[float] = None
        memory_delta_mb_float: Optional[float] = None
        if memory_mb_val is not None:
            memory_mb_float = float(memory_mb_val)  # type: ignore[arg-type]
        if memory_delta_mb_val is not None:
            memory_delta_mb_float = float(memory_delta_mb_val)  # type: ignore[arg-type]
        duration_seconds_val = metrics.get("duration_seconds")
        duration_seconds = float(duration_seconds_val) if duration_seconds_val is not None else 0.0  # type: ignore[arg-type]  # metrics dict values are Any
        log_performance_metric(
            self.logger,
            operation=str(metrics["operation"]),
            duration_seconds=duration_seconds,
            memory_mb=memory_mb_float,
            memory_delta_mb=memory_delta_mb_float,
        )

        return False


def setup_logging(config: "RunConfig", use_json: Optional[bool] = None, use_rich: Optional[bool] = None) -> None:
    """Setup enhanced logging configuration.

    Args:
        config: RunConfig instance with output settings
        use_json: Whether to use JSON format for file logs (defaults to config value)
        use_rich: Whether to use rich for console output (defaults to config value)
    """
    # Use config values if not explicitly provided
    if use_json is None:
        use_json = getattr(config.output, "log_json_format", False)
    if use_rich is None:
        use_rich = getattr(config.output, "log_use_rich", True)
    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Get log level
    log_level = getattr(logging, config.output.log_level.upper(), logging.INFO)

    # Setup console handler
    if RICH_AVAILABLE and use_rich:
        console_handler = RichHandler(rich_tracebacks=True, show_path=False, console=Console(stderr=True))
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter("%(message)s", datefmt="[%X]")
        console_handler.setFormatter(console_formatter)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)

    # Setup file handler
    output_dir = config.get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / config.output.log_file

    # Use rotating file handler
    file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)  # 10MB
    file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file

    file_formatter: Union[StructuredFormatter, logging.Formatter]
    if use_json:
        file_formatter = StructuredFormatter()
    else:
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)

    # Configure root logger
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # If loguru is available, configure it for better structured logging
    if LOGURU_AVAILABLE:
        loguru_logger.remove()  # Remove default handler
        loguru_logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=log_level,
            colorize=True,
        )
        loguru_logger.add(
            str(log_file),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB",
            retention=5,
            compression="zip",
        )

    logging.info(f"Logging initialized. Log file: {log_file}, JSON format: {use_json}, Rich: {use_rich}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_trade_event(
    logger: logging.Logger, event_type: TradeEventType, symbol: str, asset_class: str, date: Union[pd.Timestamp, str], **kwargs
) -> None:
    """Log a trade event (entry, exit, stop hit).

    Args:
        logger: Logger instance
        event_type: Type of trade event
        symbol: Symbol name
        asset_class: Asset class (equity/crypto)
        date: Event date
        **kwargs: Additional event-specific fields
    """
    event_data = {"event_type": event_type.value, "symbol": symbol, "asset_class": asset_class, "date": str(date), **kwargs}

    if event_type == TradeEventType.ENTRY:
        logger.info(
            f"TRADE_ENTRY: {symbol} | Price: {kwargs.get('entry_price', 'N/A')} | "
            f"Qty: {kwargs.get('quantity', 'N/A')} | Stop: {kwargs.get('stop_price', 'N/A')} | "
            f"Trigger: {kwargs.get('triggered_on', 'N/A')}",
            extra=event_data,
        )
    elif event_type == TradeEventType.EXIT:
        logger.info(
            f"TRADE_EXIT: {symbol} | Exit Price: {kwargs.get('exit_price', 'N/A')} | "
            f"Reason: {kwargs.get('exit_reason', 'N/A')} | "
            f"P&L: {kwargs.get('realized_pnl', 'N/A'):.2f} | "
            f"R-Multiple: {kwargs.get('r_multiple', 'N/A'):.2f}",
            extra=event_data,
        )
    elif event_type == TradeEventType.STOP_HIT:
        logger.warning(
            f"TRADE_STOP_HIT: {symbol} | Stop Price: {kwargs.get('stop_price', 'N/A')} | "
            f"Exit Price: {kwargs.get('exit_price', 'N/A')} | "
            f"P&L: {kwargs.get('realized_pnl', 'N/A'):.2f}",
            extra=event_data,
        )
    elif event_type == TradeEventType.REJECTED:
        logger.warning(f"TRADE_REJECTED: {symbol} | Reason: {kwargs.get('reason', 'N/A')}", extra=event_data)


def log_signal_generation(
    logger: logging.Logger, symbol: str, asset_class: str, date: Union[pd.Timestamp, str], signal_generated: bool, **kwargs
) -> None:
    """Log signal generation decision and reasoning.

    Args:
        logger: Logger instance
        symbol: Symbol name
        asset_class: Asset class (equity/crypto)
        date: Signal date
        signal_generated: Whether a signal was generated
        **kwargs: Additional fields (eligibility, triggers, failures, etc.)
    """
    signal_data = {
        "symbol": symbol,
        "asset_class": asset_class,
        "date": str(date),
        "signal_generated": signal_generated,
        **kwargs,
    }

    if signal_generated:
        logger.debug(
            f"SIGNAL_GENERATED: {symbol} | Trigger: {kwargs.get('triggered_on', 'N/A')} | "
            f"Clearance: {kwargs.get('clearance', 'N/A'):.4f} | "
            f"Score: {kwargs.get('score', 'N/A'):.2f}",
            extra=signal_data,
        )
    else:
        failure_reasons = kwargs.get("failure_reasons", [])
        if failure_reasons:
            logger.debug(f"SIGNAL_NOT_GENERATED: {symbol} | Reasons: {', '.join(failure_reasons)}", extra=signal_data)
        else:
            logger.debug(f"SIGNAL_NOT_GENERATED: {symbol} | No trigger", extra=signal_data)


def log_portfolio_snapshot(
    logger: logging.Logger,
    date: Union[pd.Timestamp, str],
    equity: float,
    cash: float,
    open_positions: int,
    realized_pnl: float,
    unrealized_pnl: float,
    gross_exposure: float,
    risk_multiplier: float,
    **kwargs,
) -> None:
    """Log daily portfolio state snapshot.

    Args:
        logger: Logger instance
        date: Snapshot date
        equity: Current equity
        cash: Available cash
        open_positions: Number of open positions
        realized_pnl: Cumulative realized P&L
        unrealized_pnl: Current unrealized P&L
        gross_exposure: Gross exposure
        risk_multiplier: Current risk multiplier
        **kwargs: Additional portfolio metrics
    """
    snapshot_data = {
        "date": str(date),
        "equity": equity,
        "cash": cash,
        "open_positions": open_positions,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "gross_exposure": gross_exposure,
        "risk_multiplier": risk_multiplier,
        **kwargs,
    }

    logger.info(
        f"PORTFOLIO_SNAPSHOT: {date} | Equity: ${equity:,.2f} | "
        f"Cash: ${cash:,.2f} | Positions: {open_positions} | "
        f"Realized P&L: ${realized_pnl:,.2f} | Unrealized P&L: ${unrealized_pnl:,.2f} | "
        f"Exposure: ${gross_exposure:,.2f} | Risk Mult: {risk_multiplier:.2f}",
        extra=snapshot_data,
    )


def log_performance_metric(
    logger: logging.Logger,
    operation: str,
    duration_seconds: float,
    memory_mb: Optional[float] = None,
    memory_delta_mb: Optional[float] = None,
    **kwargs,
) -> None:
    """Log performance metrics (timing, memory).

    Args:
        logger: Logger instance
        operation: Operation name
        duration_seconds: Duration in seconds
        memory_mb: Current memory usage in MB
        memory_delta_mb: Memory delta in MB
        **kwargs: Additional performance metrics
    """
    metric_data = {
        "operation": operation,
        "duration_seconds": duration_seconds,
        "memory_mb": memory_mb,
        "memory_delta_mb": memory_delta_mb,
        **kwargs,
    }

    msg = f"PERFORMANCE: {operation} | Duration: {duration_seconds:.4f}s"
    if memory_mb is not None:
        msg += f" | Memory: {memory_mb:.2f} MB"
    if memory_delta_mb is not None:
        msg += f" | Memory Δ: {memory_delta_mb:+.2f} MB"

    logger.debug(msg, extra=metric_data)

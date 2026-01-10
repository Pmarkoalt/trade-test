"""Canonical data contracts for the trading system.

This module defines the shared domain objects used across all system components:
- backtests
- optimization
- daily signal generation
- newsletter
- paper trading
- manual trade tracking

These contracts ensure all layers speak the same language.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd


class AssetClass(str, Enum):
    """Asset class enumeration."""

    EQUITY = "equity"
    CRYPTO = "crypto"


class SignalIntent(str, Enum):
    """Signal execution intent."""

    EXECUTE_NEXT_OPEN = "execute_next_open"
    EXECUTE_MARKET = "execute_market"
    EXECUTE_LIMIT = "execute_limit"
    EXECUTE_MOO = "execute_moo"  # Market-on-open


class OrderMethod(str, Enum):
    """Order entry method."""

    MOO = "MOO"  # Market-on-open
    MKT = "MKT"  # Market order
    LIMIT = "LIMIT"  # Limit order


class PositionSource(str, Enum):
    """Source of position record."""

    SYSTEM = "system"  # System-generated backtest/live trade
    PAPER = "paper"  # Paper trading account
    MANUAL = "manual"  # Manually entered by user


class StopLogicType(str, Enum):
    """Type of stop logic."""

    ATR_TRAILING = "atr_trailing"  # ATR-based trailing stop
    MA_CROSS = "ma_cross"  # Moving average cross
    FIXED_PERCENT = "fixed_percent"  # Fixed percentage stop
    TIGHTENED_ATR = "tightened_atr"  # Tightened ATR stop (crypto)


class ExitLogicType(str, Enum):
    """Type of exit logic."""

    MA_CROSS = "ma_cross"  # Moving average cross
    TIME_STOP = "time_stop"  # Time-based exit
    PROFIT_TARGET = "profit_target"  # Profit target hit
    MANUAL = "manual"  # Manual exit


@dataclass
class Signal:
    """Signal represents a trading opportunity identified by a strategy.

    This is the canonical signal format used across all strategies and buckets.
    It extends the existing Signal model with additional fields for newsletter
    and multi-bucket support.
    """

    symbol: str
    asset_class: AssetClass
    timestamp: pd.Timestamp
    side: str  # "BUY" or "SELL"
    intent: SignalIntent

    confidence: float  # 0.0 to 1.0
    rationale_tags: Dict[str, any] = field(default_factory=dict)

    entry_price: float = 0.0
    stop_price: float = 0.0

    bucket: Optional[str] = None  # "safe_sp500", "aggressive_crypto", "low_float", "unusual_options"
    strategy_name: Optional[str] = None

    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate signal data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")

        if self.entry_price < 0:
            raise ValueError(f"entry_price must be non-negative, got {self.entry_price}")

        if self.stop_price < 0:
            raise ValueError(f"stop_price must be non-negative, got {self.stop_price}")


@dataclass
class Allocation:
    """Allocation recommendation for a signal.

    Defines how much capital to allocate to a trading opportunity,
    considering risk budget, position limits, and liquidity constraints.
    """

    symbol: str
    signal_timestamp: pd.Timestamp

    recommended_position_size_dollars: float
    recommended_position_size_percent: float  # % of total portfolio

    risk_budget_used: float  # Amount of risk budget consumed
    max_positions_constraint_applied: bool
    liquidity_flags: List[str] = field(default_factory=list)
    capacity_flags: List[str] = field(default_factory=list)

    quantity: int = 0  # Calculated number of shares/units
    max_adv_percent: float = 0.0  # Max % of ADV20 used

    notes: str = ""

    def __post_init__(self):
        """Validate allocation data."""
        if self.recommended_position_size_dollars < 0:
            raise ValueError(
                f"recommended_position_size_dollars must be non-negative, "
                f"got {self.recommended_position_size_dollars}"
            )

        if not 0.0 <= self.recommended_position_size_percent <= 100.0:
            raise ValueError(
                f"recommended_position_size_percent must be between 0.0 and 100.0, "
                f"got {self.recommended_position_size_percent}"
            )

        if self.quantity < 0:
            raise ValueError(f"quantity must be non-negative, got {self.quantity}")


@dataclass
class TradePlan:
    """Trade plan defines how to execute and manage a position.

    Contains entry method, stop logic, exit logic, and optional time constraints.
    """

    symbol: str
    signal_timestamp: pd.Timestamp

    entry_method: OrderMethod
    entry_price: float  # Expected entry price

    stop_logic: StopLogicType
    stop_price: float  # Initial stop price
    stop_params: Dict = field(default_factory=dict)  # e.g., {"atr_mult": 2.5}

    exit_logic: ExitLogicType
    exit_params: Dict = field(default_factory=dict)  # e.g., {"ma_period": 20}

    time_stop_days: Optional[int] = None  # Optional max holding period

    allocation: Optional[Allocation] = None  # Link to allocation recommendation

    notes: str = ""

    def __post_init__(self):
        """Validate trade plan data."""
        if self.entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {self.entry_price}")

        if self.stop_price <= 0:
            raise ValueError(f"stop_price must be positive, got {self.stop_price}")

        if self.time_stop_days is not None and self.time_stop_days <= 0:
            raise ValueError(f"time_stop_days must be positive if set, got {self.time_stop_days}")


@dataclass
class PositionRecord:
    """Position record for tracking all positions (system, paper, manual).

    This unified format allows merging positions from different sources
    into a single portfolio view and reporting system.
    """

    symbol: str
    asset_class: AssetClass
    source: PositionSource

    open_timestamp: pd.Timestamp
    close_timestamp: Optional[pd.Timestamp] = None

    entry_price: float = 0.0
    exit_price: Optional[float] = None

    quantity: int = 0
    side: str = "LONG"  # "LONG" or "SHORT"

    fills: List[Dict] = field(default_factory=list)  # List of fill records

    pnl: float = 0.0  # Realized P&L (if closed)
    r_multiple: Optional[float] = None  # R-multiple (if closed)

    stop_price: float = 0.0
    initial_stop_price: float = 0.0

    notes: str = ""
    tags: List[str] = field(default_factory=list)

    bucket: Optional[str] = None  # Which strategy bucket
    strategy_name: Optional[str] = None

    trade_plan: Optional[TradePlan] = None  # Original trade plan (if available)

    metadata: Dict = field(default_factory=dict)

    def is_open(self) -> bool:
        """Check if position is still open."""
        return self.close_timestamp is None

    def compute_unrealized_pnl(self, current_price: float) -> float:
        """Compute unrealized P&L for open position."""
        if not self.is_open():
            return self.pnl

        if self.side == "LONG":
            return (current_price - self.entry_price) * self.quantity
        else:  # SHORT
            return (self.entry_price - current_price) * self.quantity

    def __post_init__(self):
        """Validate position record data."""
        if self.entry_price < 0:
            raise ValueError(f"entry_price must be non-negative, got {self.entry_price}")

        if self.exit_price is not None and self.exit_price < 0:
            raise ValueError(f"exit_price must be non-negative, got {self.exit_price}")

        if self.quantity < 0:
            raise ValueError(f"quantity must be non-negative, got {self.quantity}")

        if self.stop_price < 0:
            raise ValueError(f"stop_price must be non-negative, got {self.stop_price}")


@dataclass
class DailySignalBatch:
    """Batch of signals generated for a single day.

    Used by the daily signal generation entrypoint to package
    all signals, allocations, and trade plans for downstream consumers.
    """

    generation_date: pd.Timestamp
    signals: List[Signal] = field(default_factory=list)
    allocations: List[Allocation] = field(default_factory=list)
    trade_plans: List[TradePlan] = field(default_factory=list)

    bucket_summaries: Dict[str, Dict] = field(default_factory=dict)  # Per-bucket stats

    metadata: Dict = field(default_factory=dict)

    def get_signals_by_bucket(self, bucket: str) -> List[Signal]:
        """Get all signals for a specific bucket."""
        return [s for s in self.signals if s.bucket == bucket]

    def get_top_signals(self, n: int = 10) -> List[Signal]:
        """Get top N signals by confidence."""
        return sorted(self.signals, key=lambda s: s.confidence, reverse=True)[:n]

    def __post_init__(self):
        """Validate batch data."""
        if len(self.signals) != len(self.allocations):
            # Allow mismatches for now (some signals may not have allocations)
            pass

        if len(self.signals) != len(self.trade_plans):
            # Allow mismatches for now (some signals may not have trade plans)
            pass

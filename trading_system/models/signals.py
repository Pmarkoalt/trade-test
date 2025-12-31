"""Signal (entry intent) data model."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

import pandas as pd


class SignalSide(str, Enum):
    """Signal side (buy or sell)."""

    BUY = "BUY"
    SELL = "SELL"  # For exits, though system is long-only


class SignalType(str, Enum):
    """Generic signal type for all strategies."""

    ENTRY_LONG = "ENTRY_LONG"
    ENTRY_SHORT = "ENTRY_SHORT"
    EXIT = "EXIT"
    EXIT_SHORT = "EXIT_SHORT"


class BreakoutType(str, Enum):
    """Breakout type that triggered the signal (momentum-specific)."""

    FAST_20D = "20D"
    SLOW_55D = "55D"


@dataclass
class Signal:
    """Generic signal model for all strategy types.

    This model supports both momentum-specific fields (for backward compatibility)
    and generic fields (signal_type, trigger_reason, metadata, urgency) for all strategies.
    """

    symbol: str
    asset_class: str  # "equity" | "crypto"
    date: pd.Timestamp  # Date when signal was generated (at close)
    side: SignalSide  # BUY or SELL

    # Generic signal fields (for all strategies)
    signal_type: SignalType  # ENTRY_LONG, EXIT, etc.
    trigger_reason: str  # Human-readable reason (e.g., "momentum_breakout_20D", "mean_reversion_oversold")

    # Entry/exit details (required fields)
    entry_price: float  # Close price at signal time (will execute at next open)
    stop_price: float  # Calculated stop price

    # Optional fields with defaults
    metadata: Dict = field(default_factory=dict)  # Strategy-specific data
    urgency: float = 0.5  # 0-1, for position queue prioritization
    suggested_entry_price: Optional[float] = None  # Alias for entry_price (for compatibility)
    suggested_stop_price: Optional[float] = None  # Alias for stop_price (for compatibility)

    # Momentum-specific fields (for backward compatibility)
    atr_mult: Optional[float] = None  # ATR multiplier used (2.5 for equity, 3.0 for crypto)
    triggered_on: Optional[BreakoutType] = None  # Which breakout triggered: "20D" or "55D"
    breakout_clearance: Optional[float] = None  # Actual clearance above prior high (for logging)

    # Scoring components (for queue ranking)
    breakout_strength: float = 0.0  # (close - MA) / ATR14, normalized
    momentum_strength: float = 0.0  # Relative strength vs benchmark
    diversification_bonus: float = 0.0  # 1 - avg_corr_to_portfolio
    score: float = 0.0  # Final weighted score (0-1 after rank normalization)

    # Eligibility status
    passed_eligibility: bool = True  # True if all filters passed
    eligibility_failures: list[str] = field(default_factory=list)  # Reasons if failed

    # Capacity check
    order_notional: float = 0.0  # Estimated order size (for capacity check)
    adv20: float = 0.0  # ADV20 at signal time
    capacity_passed: bool = True  # True if order_notional <= max_pct * ADV20

    def __post_init__(self):
        """Validate signal data and set aliases."""
        # Set suggested_price aliases if not provided
        if self.suggested_entry_price is None:
            self.suggested_entry_price = self.entry_price
        if self.suggested_stop_price is None:
            self.suggested_stop_price = self.stop_price

        # Validate asset class
        if self.asset_class not in ["equity", "crypto"]:
            raise ValueError(f"Invalid asset_class: {self.asset_class}, must be 'equity' or 'crypto'")

        # Validate entry price
        if self.entry_price <= 0:
            raise ValueError(f"Invalid entry_price: {self.entry_price}, must be positive")

        # Validate stop price based on signal type
        if self.signal_type == SignalType.ENTRY_LONG:
            # Long: stop must be below entry
            if self.stop_price >= self.entry_price:
                raise ValueError(
                    f"Invalid stop_price: {self.stop_price} >= entry_price {self.entry_price} "
                    "(stop must be below entry for long positions)"
                )
        elif self.signal_type == SignalType.ENTRY_SHORT:
            # Short: stop must be above entry
            if self.stop_price <= self.entry_price:
                raise ValueError(
                    f"Invalid stop_price: {self.stop_price} <= entry_price {self.entry_price} "
                    "(stop must be above entry for short positions)"
                )
        elif self.signal_type == SignalType.EXIT:
            # Exit signals: stop validation depends on position side (not validated here)
            pass

        # Validate stop price is positive
        if self.stop_price <= 0:
            raise ValueError(f"Invalid stop_price: {self.stop_price}, must be positive")

        # Validate urgency
        if not 0.0 <= self.urgency <= 1.0:
            raise ValueError(f"Invalid urgency: {self.urgency}, must be between 0.0 and 1.0")

        # Validate metadata is a dict
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")

        # Validate eligibility_failures is a list
        if not isinstance(self.eligibility_failures, list):
            raise ValueError("eligibility_failures must be a list")

        # For entry signals, validate adv20 if capacity check is enabled
        if self.signal_type in [SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT]:
            if self.adv20 <= 0:
                raise ValueError(f"Invalid adv20: {self.adv20}, must be positive for entry signals")

    def is_valid(self) -> bool:
        """Check if signal is valid for execution.

        Returns:
            True if signal passed eligibility, capacity check, and is a valid entry signal
        """
        # Entry signals must pass eligibility and capacity
        if self.signal_type == SignalType.ENTRY_LONG:
            return self.passed_eligibility and self.capacity_passed and self.side == SignalSide.BUY
        elif self.signal_type == SignalType.ENTRY_SHORT:
            return self.passed_eligibility and self.capacity_passed and self.side == SignalSide.SELL

        # Exit signals are always valid if they have a symbol
        return self.symbol is not None and len(self.symbol) > 0

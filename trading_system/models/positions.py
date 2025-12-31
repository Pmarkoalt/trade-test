"""Position data model."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

from .signals import BreakoutType


class PositionSide(str, Enum):
    """Position side (long or short)."""

    LONG = "LONG"
    SHORT = "SHORT"


class ExitReason(str, Enum):
    """Exit reason for position closure."""

    TRAILING_MA_CROSS = "trailing_ma_cross"  # MA20 or MA50 cross
    HARD_STOP = "hard_stop"  # ATR-based stop hit
    DATA_MISSING = "data_missing"  # 2+ consecutive missing days
    CAPACITY_REJECT = "capacity_reject"  # Should not happen for open positions
    MANUAL = "manual"  # Manual exit (not used in MVP)


@dataclass
class Position:
    """Open position in portfolio."""

    # Required fields (no defaults) - must come first
    symbol: str
    asset_class: str
    entry_date: pd.Timestamp  # Date when position was opened
    entry_price: float  # Fill price at entry
    entry_fill_id: str  # Reference to entry fill
    quantity: int  # Number of shares/units
    side: PositionSide  # LONG or SHORT
    stop_price: float  # Current stop price
    initial_stop_price: float  # Original stop (for R-multiple calculation)
    hard_stop_atr_mult: float  # ATR multiplier (2.5 equity, 3.0 crypto)
    entry_slippage_bps: float
    entry_fee_bps: float
    entry_total_cost: float
    triggered_on: BreakoutType  # Which breakout triggered entry
    adv20_at_entry: float  # ADV20 at entry (for diagnostics)

    # Optional fields (with defaults)
    strategy_name: Optional[str] = None  # Strategy that owns this position (for multi-strategy)
    tightened_stop: bool = False  # True if stop was tightened (crypto staged exit)
    tightened_stop_atr_mult: Optional[float] = None  # 2.0 for crypto after MA20 break

    # Exit tracking
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_fill_id: Optional[str] = None
    exit_reason: Optional[ExitReason] = None

    # Exit cost tracking
    exit_slippage_bps: Optional[float] = None
    exit_fee_bps: Optional[float] = None
    exit_total_cost: Optional[float] = None

    # P&L
    realized_pnl: float = 0.0  # Only set when position is closed
    unrealized_pnl: float = 0.0  # Updated daily: (current_price - entry_price) * quantity - costs

    def is_open(self) -> bool:
        """Check if position is still open.

        Returns:
            True if position has no exit_date
        """
        return self.exit_date is None

    def update_unrealized_pnl(self, current_price: float) -> None:
        """Update unrealized P&L based on current price.

        Args:
            current_price: Current market price for the symbol
        """
        if not self.is_open():
            return

        # Long: (current_price - entry_price) * quantity
        # Short: (entry_price - current_price) * quantity
        if self.side == PositionSide.LONG:
            price_pnl = (current_price - self.entry_price) * self.quantity
        else:  # SHORT
            price_pnl = (self.entry_price - current_price) * self.quantity

        self.unrealized_pnl = price_pnl - self.entry_total_cost

    def compute_r_multiple(self, exit_price: Optional[float] = None) -> float:
        """Compute R-multiple for closed position.

        Long: R = (exit_price - entry_price) / (entry_price - initial_stop_price)
        Short: R = (entry_price - exit_price) / (initial_stop_price - entry_price)

        Args:
            exit_price: Exit price to use (if None, uses self.exit_price)

        Returns:
            R-multiple value (0.0 if risk <= 0)
        """
        if exit_price is None:
            if self.exit_price is None:
                raise ValueError("exit_price must be provided if position is not closed")
            exit_price = self.exit_price

        if self.side == PositionSide.LONG:
            price_change = exit_price - self.entry_price
            risk = self.entry_price - self.initial_stop_price
        else:  # SHORT
            price_change = self.entry_price - exit_price
            risk = self.initial_stop_price - self.entry_price

        if risk <= 0:
            return 0.0

        return price_change / risk

    def update_stop(self, new_stop_price: float, reason: str = "") -> None:
        """Update stop price (for trailing stops or tightening).

        Long positions: Stop can only move up (never down)
        Short positions: Stop can only move down (never up)

        Args:
            new_stop_price: New stop price to set
            reason: Reason for update (e.g., "trailing", "tighten")
        """
        if new_stop_price <= 0:
            raise ValueError(f"Invalid stop_price: {new_stop_price}, must be positive")

        # Long: stop can only move up (trailing stop logic)
        # Short: stop can only move down (trailing stop logic)
        if self.side == PositionSide.LONG:
            if new_stop_price > self.stop_price:
                self.stop_price = new_stop_price
                if "tighten" in reason.lower():
                    self.tightened_stop = True
                    self.tightened_stop_atr_mult = 2.0  # For crypto
        else:  # SHORT
            if new_stop_price < self.stop_price:
                self.stop_price = new_stop_price
                if "tighten" in reason.lower():
                    self.tightened_stop = True
                    self.tightened_stop_atr_mult = 2.0  # For crypto

    def __post_init__(self):
        """Validate position data."""
        if self.asset_class not in ["equity", "crypto"]:
            raise ValueError(f"Invalid asset_class: {self.asset_class}, must be 'equity' or 'crypto'")

        if self.quantity <= 0:
            raise ValueError(f"Invalid quantity: {self.quantity}, must be positive")

        if self.entry_price <= 0:
            raise ValueError(f"Invalid entry_price: {self.entry_price}, must be positive")

        if self.stop_price <= 0:
            raise ValueError(f"Invalid stop_price: {self.stop_price}, must be positive")

        if self.initial_stop_price <= 0:
            raise ValueError(f"Invalid initial_stop_price: {self.initial_stop_price}, must be positive")

        # Validate stop price relative to entry based on position side
        if self.side == PositionSide.LONG:
            # Long: stop must be below entry
            if self.stop_price >= self.entry_price:
                raise ValueError(
                    f"Invalid stop_price: {self.stop_price} >= entry_price {self.entry_price} "
                    f"(stop must be below entry for long positions)"
                )
            if self.initial_stop_price >= self.entry_price:
                raise ValueError(
                    f"Invalid initial_stop_price: {self.initial_stop_price} >= entry_price "
                    f"{self.entry_price} (stop must be below entry for long positions)"
                )
        else:  # SHORT
            # Short: stop must be above entry
            if self.stop_price <= self.entry_price:
                raise ValueError(
                    f"Invalid stop_price: {self.stop_price} <= entry_price {self.entry_price} "
                    f"(stop must be above entry for short positions)"
                )
            if self.initial_stop_price <= self.entry_price:
                raise ValueError(
                    f"Invalid initial_stop_price: {self.initial_stop_price} <= entry_price "
                    f"{self.entry_price} (stop must be above entry for short positions)"
                )

        if self.adv20_at_entry <= 0:
            raise ValueError(f"Invalid adv20_at_entry: {self.adv20_at_entry}, must be positive")

        if self.entry_total_cost < 0:
            raise ValueError(f"Invalid entry_total_cost: {self.entry_total_cost}, must be >= 0")

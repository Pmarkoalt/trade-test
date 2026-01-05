"""Order and Fill data models."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

from .signals import SignalSide


class OrderStatus(str, Enum):
    """Order status."""

    PENDING = "PENDING"  # Created, awaiting execution
    FILLED = "FILLED"  # Executed
    REJECTED = "REJECTED"  # Rejected due to constraint
    CANCELLED = "CANCELLED"  # Cancelled before execution


@dataclass
class Order:
    """Order to execute at next session open."""

    # Required fields (no defaults)
    order_id: str  # Unique identifier
    symbol: str
    asset_class: str
    date: pd.Timestamp  # Date when order was created (signal date)
    execution_date: pd.Timestamp  # Date when order should execute (next open)
    side: SignalSide  # BUY or SELL
    quantity: int  # Number of shares/units (calculated from risk sizing)
    signal_date: pd.Timestamp  # Original signal date
    expected_fill_price: float  # Next open price (estimated)
    stop_price: float  # Stop price for position

    # Optional fields (with defaults)
    limit_price: Optional[float] = None  # Not used (market orders only)
    status: OrderStatus = OrderStatus.PENDING
    rejection_reason: Optional[str] = None  # If REJECTED
    is_exit: bool = False  # True for exit orders (closing positions)

    # Constraints checked
    capacity_checked: bool = False
    correlation_checked: bool = False
    max_positions_checked: bool = False
    max_exposure_checked: bool = False

    def __post_init__(self):
        """Validate order data."""
        if self.asset_class not in ["equity", "crypto"]:
            raise ValueError(f"Invalid asset_class: {self.asset_class}, must be 'equity' or 'crypto'")

        if self.quantity <= 0:
            raise ValueError(f"Invalid quantity: {self.quantity}, must be positive")

        if self.expected_fill_price <= 0:
            raise ValueError(f"Invalid expected_fill_price: {self.expected_fill_price}, must be positive")

        # Skip stop_price validation for exit orders (closing positions)
        if not self.is_exit:
            if self.stop_price <= 0:
                raise ValueError(f"Invalid stop_price: {self.stop_price}, must be positive")

            # Validate stop price relative to entry based on order side
            if self.side == SignalSide.BUY:
                # Long: stop must be below entry
                if self.stop_price >= self.expected_fill_price:
                    raise ValueError(
                        f"Invalid stop_price: {self.stop_price} >= expected_fill_price "
                        f"{self.expected_fill_price} (stop must be below entry for long positions)"
                    )
            else:  # SELL (short entry)
                # Short: stop must be above entry
                if self.stop_price <= self.expected_fill_price:
                    raise ValueError(
                        f"Invalid stop_price: {self.stop_price} <= expected_fill_price "
                        f"{self.expected_fill_price} (stop must be above entry for short positions)"
                    )


@dataclass
class Fill:
    """Realized execution result."""

    fill_id: str  # Unique identifier
    order_id: str  # Parent order ID
    symbol: str
    asset_class: str
    date: pd.Timestamp  # Execution date (open of day t+1)

    side: SignalSide
    quantity: int  # Actual filled quantity (may differ from order if partial)
    fill_price: float  # Actual execution price (open + slippage)
    open_price: float  # Market open price (before slippage)

    # Costs
    slippage_bps: float  # Actual slippage in basis points
    fee_bps: float  # Fee in basis points (1 for equity, 8 for crypto)
    total_cost: float  # (slippage + fee) * notional

    # Slippage model components (for diagnostics)
    vol_mult: float  # Volatility multiplier
    size_penalty: float  # Size penalty
    weekend_penalty: float  # Weekend penalty (crypto only)
    stress_mult: float  # Stress multiplier

    # Notional
    notional: float  # fill_price * quantity

    def compute_total_cost(self) -> float:
        """Compute total execution cost.

        Returns:
            Total cost = slippage_cost + fee_cost
        """
        notional = self.fill_price * self.quantity
        slippage_cost = notional * (self.slippage_bps / 10000)
        fee_cost = notional * (self.fee_bps / 10000)
        return slippage_cost + fee_cost

    def __post_init__(self):
        """Validate fill data and compute notional if needed."""
        if self.asset_class not in ["equity", "crypto"]:
            raise ValueError(f"Invalid asset_class: {self.asset_class}, must be 'equity' or 'crypto'")

        # Allow quantity=0 for rejected fills (when fill_price=0)
        if self.quantity < 0:
            raise ValueError(f"Invalid quantity: {self.quantity}, must be non-negative")

        if self.quantity == 0 and self.fill_price > 0:
            raise ValueError(f"Invalid quantity: {self.quantity}, must be positive when fill_price > 0")

        if self.fill_price < 0:
            raise ValueError(f"Invalid fill_price: {self.fill_price}, must be non-negative")

        # Allow fill_price=0 for rejected fills, but require positive for actual fills
        if self.fill_price > 0 and self.quantity == 0:
            raise ValueError("Invalid: fill_price > 0 but quantity = 0")

        if self.fill_price > 0 and self.open_price <= 0:
            raise ValueError(f"Invalid open_price: {self.open_price}, must be positive when fill_price > 0")

        # For rejected fills (fill_price=0), open_price can be 0

        if self.slippage_bps < 0:
            raise ValueError(f"Invalid slippage_bps: {self.slippage_bps}, must be >= 0")

        if self.fee_bps < 0:
            raise ValueError(f"Invalid fee_bps: {self.fee_bps}, must be >= 0")

        # Compute notional if not provided
        if not hasattr(self, "notional") or self.notional == 0:
            self.notional = self.fill_price * self.quantity

        # Validate slippage direction matches side
        if self.side == SignalSide.BUY:
            # BUY: fill_price should be >= open_price (paying more)
            if self.fill_price < self.open_price:
                raise ValueError(f"Invalid fill_price for BUY: {self.fill_price} < open_price {self.open_price}")
        else:  # SELL
            # SELL: fill_price should be <= open_price (receiving less)
            if self.fill_price > self.open_price:
                raise ValueError(f"Invalid fill_price for SELL: {self.fill_price} > open_price {self.open_price}")

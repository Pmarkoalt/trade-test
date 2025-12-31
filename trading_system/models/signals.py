"""Signal (entry intent) data model."""

from dataclasses import dataclass
from enum import Enum
import pandas as pd


class SignalSide(str, Enum):
    """Signal side (buy or sell)."""
    BUY = "BUY"
    SELL = "SELL"  # For exits, though system is long-only


class BreakoutType(str, Enum):
    """Breakout type that triggered the signal."""
    FAST_20D = "20D"
    SLOW_55D = "55D"


@dataclass
class Signal:
    """Entry signal generated at day close."""
    
    symbol: str
    asset_class: str  # "equity" | "crypto"
    date: pd.Timestamp  # Date when signal was generated (at close)
    side: SignalSide  # Always BUY for this system
    
    # Entry details
    entry_price: float  # Close price at signal time (will execute at next open)
    stop_price: float  # Calculated stop: entry_price - (ATR_mult * ATR14)
    atr_mult: float  # ATR multiplier used (2.5 for equity, 3.0 for crypto)
    
    # Trigger information
    triggered_on: BreakoutType  # Which breakout triggered: "20D" or "55D"
    breakout_clearance: float  # Actual clearance above prior high (for logging)
    
    # Scoring components (for queue ranking)
    breakout_strength: float  # (close - MA) / ATR14, normalized
    momentum_strength: float  # Relative strength vs benchmark
    diversification_bonus: float  # 1 - avg_corr_to_portfolio
    score: float  # Final weighted score (0-1 after rank normalization)
    
    # Eligibility status
    passed_eligibility: bool  # True if all filters passed
    eligibility_failures: list[str]  # Reasons if failed (e.g., ["below_MA50", "insufficient_slope"])
    
    # Capacity check
    order_notional: float  # Estimated order size (for capacity check)
    adv20: float  # ADV20 at signal time
    capacity_passed: bool  # True if order_notional <= max_pct * ADV20
    
    def is_valid(self) -> bool:
        """Check if signal is valid for execution.
        
        Returns:
            True if signal passed eligibility, capacity check, and is a BUY signal
        """
        return (
            self.passed_eligibility and
            self.capacity_passed and
            self.side == SignalSide.BUY
        )
    
    def __post_init__(self):
        """Validate signal data."""
        if self.asset_class not in ["equity", "crypto"]:
            raise ValueError(
                f"Invalid asset_class: {self.asset_class}, must be 'equity' or 'crypto'"
            )
        
        if self.entry_price <= 0:
            raise ValueError(f"Invalid entry_price: {self.entry_price}, must be positive")
        
        if self.stop_price >= self.entry_price:
            raise ValueError(
                f"Invalid stop_price: {self.stop_price} >= entry_price {self.entry_price} "
                f"(stop must be below entry for long positions)"
            )
        
        if self.stop_price <= 0:
            raise ValueError(f"Invalid stop_price: {self.stop_price}, must be positive")
        
        if self.adv20 <= 0:
            raise ValueError(f"Invalid adv20: {self.adv20}, must be positive")
        
        if not isinstance(self.eligibility_failures, list):
            raise ValueError("eligibility_failures must be a list")


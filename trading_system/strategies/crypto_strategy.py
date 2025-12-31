"""Crypto momentum strategy with staged exit logic."""

from typing import List, Optional
import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy
from ..configs.strategy_config import StrategyConfig
from ..models.signals import BreakoutType
from ..models.features import FeatureRow
from ..models.positions import Position, ExitReason


class CryptoStrategy(BaseStrategy):
    """Crypto momentum strategy with staged exit logic.
    
    Eligibility:
    - close > MA200 (STRICT, no exceptions)
    - Optional: relative strength vs BTC (v1.1)
    
    Entry triggers (same as equity):
    - Fast: close >= highest_close_20d * 1.005
    - Slow: close >= highest_close_55d * 1.010
    
    Exit logic (staged):
    - Stage 1: close < MA20 → tighten stop to 2.0 * ATR14
    - Stage 2: close < MA50 OR tightened stop hit → exit
    
    Hard stop: entry - 3.0 * ATR14 (wider than equity)
    Capacity check: order_notional <= 0.25% * ADV20 (stricter)
    """
    
    def __init__(self, config: StrategyConfig):
        """Initialize crypto strategy.
        
        Args:
            config: Strategy configuration (must have asset_class="crypto")
        """
        if config.asset_class != "crypto":
            raise ValueError(
                f"CryptoStrategy requires asset_class='crypto', got '{config.asset_class}'"
            )
        
        super().__init__(config)
        
        # Validate exit mode
        if config.exit.mode != "staged":
            raise ValueError(
                f"CryptoStrategy requires exit.mode='staged', got '{config.exit.mode}'"
            )
    
    def check_eligibility(self, features: FeatureRow) -> tuple[bool, List[str]]:
        """Check if symbol is eligible for entry.
        
        Requirements:
        1. close > MA200 (STRICT, no exceptions)
        2. Optional: relative strength vs BTC (if enabled)
        
        Args:
            features: FeatureRow with indicators for the symbol
            
        Returns:
            Tuple of (is_eligible, failure_reasons)
        """
        failures = []
        
        # Check MA200 (STRICT requirement)
        if features.ma200 is None or np.isnan(features.ma200):
            failures.append("ma200_missing")
            return False, failures
        
        if features.close <= features.ma200:
            failures.append("below_MA200")
            return False, failures
        
        # Optional: relative strength vs BTC
        if self.config.eligibility.relative_strength_enabled:
            if features.roc60 is None or np.isnan(features.roc60):
                failures.append("roc60_missing")
                return False, failures
            
            if features.benchmark_roc60 is None or np.isnan(features.benchmark_roc60):
                failures.append("benchmark_roc60_missing")
                return False, failures
            
            relative_strength = features.roc60 - features.benchmark_roc60
            if relative_strength <= self.config.eligibility.relative_strength_min:
                failures.append("insufficient_relative_strength")
                return False, failures
        
        return True, []
    
    def check_entry_triggers(
        self, features: FeatureRow
    ) -> tuple[Optional[BreakoutType], float]:
        """Check if entry triggers are met.
        
        Entry triggers (OR logic):
        - Fast: close >= highest_close_20d * (1 + fast_clearance)
        - Slow: close >= highest_close_55d * (1 + slow_clearance)
        
        Args:
            features: FeatureRow with indicators for the symbol
            
        Returns:
            Tuple of (breakout_type, clearance) or (None, 0.0) if no trigger
        """
        fast_clearance = self.config.entry.fast_clearance
        slow_clearance = self.config.entry.slow_clearance
        
        # Check fast trigger (20D)
        if (
            features.highest_close_20d is not None
            and not np.isnan(features.highest_close_20d)
        ):
            fast_threshold = features.highest_close_20d * (1 + fast_clearance)
            if features.close >= fast_threshold:
                clearance = (features.close / features.highest_close_20d) - 1
                return BreakoutType.FAST_20D, clearance
        
        # Check slow trigger (55D)
        if (
            features.highest_close_55d is not None
            and not np.isnan(features.highest_close_55d)
        ):
            slow_threshold = features.highest_close_55d * (1 + slow_clearance)
            if features.close >= slow_threshold:
                clearance = (features.close / features.highest_close_55d) - 1
                return BreakoutType.SLOW_55D, clearance
        
        return None, 0.0
    
    def check_exit_signals(
        self, position: Position, features: FeatureRow
    ) -> Optional[ExitReason]:
        """Check if position should be exited.
        
        Exit logic (staged):
        - Stage 1: close < MA20 → tighten stop (handled in update_stop_price)
        - Stage 2: close < MA50 OR tightened stop hit → exit
        
        Priority:
        1. Hard stop (highest priority)
        2. MA50 cross (if in stage 2)
        
        Args:
            position: Open position to check
            features: FeatureRow with current indicators
            
        Returns:
            ExitReason if exit triggered, None otherwise
        """
        # Priority 1: Hard stop
        if features.close <= position.stop_price:
            return ExitReason.HARD_STOP
        
        # Priority 2: MA50 cross (stage 2 exit)
        if features.ma50 is not None and not np.isnan(features.ma50):
            if features.close < features.ma50:
                return ExitReason.TRAILING_MA_CROSS
        
        return None
    
    def update_stop_price(
        self, position: Position, features: FeatureRow
    ) -> Optional[float]:
        """Update stop price for position (staged exit logic).
        
        Staged exit logic:
        - Stage 1: If close < MA20 and stop not yet tightened:
          → Tighten stop to entry - 2.0 * ATR14
          → Set tightened_stop flag (never reset)
        - Stop can only move up (trailing) or be tightened once
        
        Args:
            position: Open position
            features: FeatureRow with current indicators
            
        Returns:
            New stop price if updated, None if unchanged
        """
        # Stage 1: Check if MA20 break triggers stop tightening
        if (
            not position.tightened_stop
            and features.ma20 is not None
            and not np.isnan(features.ma20)
        ):
            if features.close < features.ma20:
                # Tighten stop to 2.0 * ATR14
                if features.atr14 is not None and not np.isnan(features.atr14):
                    tightened_stop = position.entry_price - (
                        self.config.exit.tightened_stop_atr_mult * features.atr14
                    )
                    
                    # Only tighten if new stop is higher than current (can't move down)
                    if tightened_stop > position.stop_price:
                        position.update_stop(tightened_stop, reason="tighten")
                        return tightened_stop
        
        # For trailing stops, stop can only move up
        # (This is handled by the Position.update_stop method which only allows increases)
        
        return None
    
    def should_tighten_stop(
        self, position: Position, features: FeatureRow
    ) -> bool:
        """Check if stop should be tightened (MA20 break).
        
        Args:
            position: Open position
            features: FeatureRow with current indicators
            
        Returns:
            True if stop should be tightened
        """
        if position.tightened_stop:
            return False  # Already tightened, never reset
        
        if features.ma20 is None or np.isnan(features.ma20):
            return False
        
        return features.close < features.ma20


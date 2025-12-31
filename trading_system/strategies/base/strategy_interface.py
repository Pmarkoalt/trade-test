"""Abstract base class for all trading strategies."""

from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd
import numpy as np

from ...configs.strategy_config import StrategyConfig
from ...models.signals import Signal, SignalSide, SignalType, BreakoutType
from ...models.features import FeatureRow
from ...models.positions import Position, ExitReason


class StrategyInterface(ABC):
    """Abstract base class for all trading strategies.
    
    This interface defines the contract that all strategies must implement,
    regardless of their type (momentum, mean reversion, pairs, etc.).
    """
    
    def __init__(self, config: StrategyConfig):
        """Initialize strategy with configuration.
        
        Args:
            config: Strategy configuration loaded from YAML
        """
        self.config = config
        self.name = config.name  # Strategy name for multi-strategy tracking
        self.asset_class = config.asset_class
        self.universe = config.universe
        self.benchmark = config.benchmark
    
    @abstractmethod
    def check_eligibility(self, features: FeatureRow) -> tuple[bool, List[str]]:
        """Check if symbol is eligible for entry.
        
        Args:
            features: FeatureRow with indicators for the symbol
            
        Returns:
            Tuple of (is_eligible, failure_reasons)
        """
        pass
    
    @abstractmethod
    def check_entry_triggers(
        self, features: FeatureRow
    ) -> tuple[Optional[BreakoutType], float]:
        """Check if entry triggers are met.
        
        Args:
            features: FeatureRow with indicators for the symbol
            
        Returns:
            Tuple of (breakout_type, clearance) or (None, 0.0) if no trigger.
            For non-breakout strategies, breakout_type may be a custom enum value.
        """
        pass
    
    @abstractmethod
    def check_exit_signals(
        self, position: Position, features: FeatureRow
    ) -> Optional[ExitReason]:
        """Check if position should be exited.
        
        Args:
            position: Open position to check
            features: FeatureRow with current indicators
            
        Returns:
            ExitReason if exit triggered, None otherwise
        """
        pass
    
    @abstractmethod
    def update_stop_price(
        self, position: Position, features: FeatureRow
    ) -> Optional[float]:
        """Update stop price for position (trailing stops, tightening).
        
        Args:
            position: Open position
            features: FeatureRow with current indicators
            
        Returns:
            New stop price if updated, None if unchanged
        """
        pass
    
    def calculate_stop_price(
        self, entry_price: float, atr14: float, atr_mult: float
    ) -> float:
        """Calculate initial stop price from entry price and ATR.
        
        Args:
            entry_price: Entry price for the position
            atr14: ATR14 value
            atr_mult: ATR multiplier (e.g., 2.5 for equity, 3.0 for crypto)
            
        Returns:
            Stop price = entry_price - (atr_mult * atr14)
        """
        stop_price = entry_price - (atr_mult * atr14)
        return max(0.01, stop_price)  # Ensure positive
    
    def check_capacity(
        self, order_notional: float, adv20: float
    ) -> bool:
        """Check if order size passes capacity constraint.
        
        Args:
            order_notional: Estimated order notional value
            adv20: 20-day average dollar volume
            
        Returns:
            True if order_notional <= max_pct * adv20
        """
        max_pct = self.config.capacity.max_order_pct_adv
        max_notional = max_pct * adv20
        return order_notional <= max_notional
    
    def calculate_breakout_strength(
        self, close: float, ma: float, atr14: float
    ) -> float:
        """Calculate breakout strength score.
        
        Formula: (close - MA) / ATR14
        
        Args:
            close: Current close price
            ma: Moving average (MA20 or MA50)
            atr14: ATR14 value
            
        Returns:
            Breakout strength (normalized by ATR)
        """
        if atr14 <= 0:
            return 0.0
        return (close - ma) / atr14
    
    def calculate_momentum_strength(
        self, roc60: Optional[float], benchmark_roc60: Optional[float]
    ) -> float:
        """Calculate momentum strength (relative strength vs benchmark).
        
        Args:
            roc60: Asset ROC60
            benchmark_roc60: Benchmark ROC60
            
        Returns:
            Relative strength (roc60 - benchmark_roc60) or 0.0 if missing
        """
        if roc60 is None or np.isnan(roc60):
            return 0.0
        if benchmark_roc60 is None or np.isnan(benchmark_roc60):
            return 0.0
        return roc60 - benchmark_roc60
    
    def generate_signal(
        self,
        symbol: str,
        features: FeatureRow,
        order_notional: float,
        diversification_bonus: float = 0.0,
    ) -> Optional[Signal]:
        """Generate entry signal for a symbol.
        
        Args:
            symbol: Symbol to generate signal for
            features: FeatureRow with indicators
            order_notional: Estimated order notional value
            diversification_bonus: Correlation bonus (1 - avg_corr)
            
        Returns:
            Signal if eligible and triggered, None otherwise
        """
        # Check if features are valid
        if not features.is_valid_for_entry():
            return None
        
        # Check eligibility
        is_eligible, failure_reasons = self.check_eligibility(features)
        
        # Check entry triggers
        breakout_type, clearance = self.check_entry_triggers(features)
        
        # If not eligible or no trigger, return None
        if not is_eligible or breakout_type is None:
            return None
        
        # Calculate stop price
        atr_mult = self.config.exit.hard_stop_atr_mult
        stop_price = self.calculate_stop_price(
            features.close, features.atr14, atr_mult
        )
        
        # Check capacity
        capacity_passed = self.check_capacity(order_notional, features.adv20)
        
        # Calculate scoring components
        ma = features.ma50 if self.config.eligibility.trend_ma == 50 else features.ma20
        breakout_strength = self.calculate_breakout_strength(
            features.close, ma, features.atr14
        )
        momentum_strength = self.calculate_momentum_strength(
            features.roc60, features.benchmark_roc60
        )
        
        # Score will be normalized later in signal queue
        score = 0.0  # Placeholder, will be normalized
        
        # Create signal with generic fields
        trigger_reason = f"momentum_{breakout_type.value.lower()}_breakout"
        if breakout_type == BreakoutType.FAST_20D:
            trigger_reason = f"momentum_breakout_20d"
        elif breakout_type == BreakoutType.SLOW_55D:
            trigger_reason = f"momentum_breakout_55d"
        
        signal = Signal(
            symbol=symbol,
            asset_class=self.asset_class,
            date=features.date,
            side=SignalSide.BUY,
            signal_type=SignalType.ENTRY_LONG,
            trigger_reason=trigger_reason,
            metadata={
                'breakout_type': breakout_type.value,
                'breakout_clearance': clearance,
                'breakout_strength': breakout_strength,
                'momentum_strength': momentum_strength,
                'roc60': features.roc60,
                'benchmark_roc60': features.benchmark_roc60,
            },
            urgency=0.6,  # Default momentum urgency
            entry_price=features.close,
            stop_price=stop_price,
            suggested_entry_price=features.close,
            suggested_stop_price=stop_price,
            # Momentum-specific fields (for backward compatibility)
            atr_mult=atr_mult,
            triggered_on=breakout_type,
            breakout_clearance=clearance,
            breakout_strength=breakout_strength,
            momentum_strength=momentum_strength,
            diversification_bonus=diversification_bonus,
            score=score,
            passed_eligibility=is_eligible,
            eligibility_failures=failure_reasons,
            order_notional=order_notional,
            adv20=features.adv20,
            capacity_passed=capacity_passed,
        )
        
        return signal


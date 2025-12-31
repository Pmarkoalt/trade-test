"""Equity momentum strategy implementation."""

from typing import List, Optional
import numpy as np

from .momentum_base import MomentumBaseStrategy
from ...configs.strategy_config import StrategyConfig
from ...models.features import FeatureRow
from ...models.positions import Position, ExitReason


class EquityMomentumStrategy(MomentumBaseStrategy):
    """Equity momentum strategy with eligibility filters, entry triggers, and exit logic.
    
    Eligibility:
    - close > MA50
    - MA50 slope > 0.5% over 20 days
    - Optional: relative strength vs SPY (v1.1)
    
    Entry triggers (OR logic):
    - Fast: close >= highest_close_20d * 1.005
    - Slow: close >= highest_close_55d * 1.010
    
    Exit logic:
    - Trailing: close < MA20 (or MA50, configurable)
    - Hard stop: close < entry - 2.5 * ATR14
    
    Capacity check: order_notional <= 0.5% * ADV20
    """
    
    def __init__(self, config: StrategyConfig):
        """Initialize equity momentum strategy.
        
        Args:
            config: Strategy configuration (must have asset_class="equity")
        """
        if config.asset_class != "equity":
            raise ValueError(
                f"EquityMomentumStrategy requires asset_class='equity', got '{config.asset_class}'"
            )
        
        super().__init__(config)
    
    def check_eligibility(self, features: FeatureRow) -> tuple[bool, List[str]]:
        """Check if symbol is eligible for entry.
        
        Eligibility requirements:
        1. close > MA50
        2. MA50 slope > 0.5% over 20 days
        3. Optional: relative strength vs SPY (if enabled)
        
        Args:
            features: FeatureRow with indicators for the symbol
        
        Returns:
            Tuple of (is_eligible, failure_reasons)
        """
        failures = []
        
        # Check if features are valid
        if not features.is_valid_for_entry():
            failures.append("insufficient_data")
            return False, failures
        
        # 1. Check close > MA50/MA200
        if self.config.eligibility.require_close_above_trend_ma:
            trend_ma_period = self.config.eligibility.trend_ma
            if trend_ma_period == 50:
                if features.ma50 is None or np.isnan(features.ma50):
                    failures.append("ma50_missing")
                    return False, failures
                
                if features.close <= features.ma50:
                    failures.append("below_MA50")
                    return False, failures
                
                # 2. Check MA50 slope > 0.5% over 20 days
                if features.ma50_slope is None or np.isnan(features.ma50_slope):
                    failures.append("ma50_slope_missing")
                    return False, failures
                
                ma_slope_min = self.config.eligibility.ma_slope_min
                if features.ma50_slope <= ma_slope_min:
                    failures.append(f"insufficient_ma50_slope_{features.ma50_slope:.6f}")
                    return False, failures
            elif trend_ma_period == 200:
                if features.ma200 is None or np.isnan(features.ma200):
                    failures.append("ma200_missing")
                    return False, failures
                
                if features.close <= features.ma200:
                    failures.append("below_MA200")
                    return False, failures
            else:
                failures.append(f"unsupported_trend_ma_{trend_ma_period}")
                return False, failures
        
        # 3. Optional: Relative strength vs benchmark
        if self.config.eligibility.relative_strength_enabled:
            if features.roc60 is None or np.isnan(features.roc60):
                failures.append("roc60_missing")
                return False, failures
            
            if features.benchmark_roc60 is None or np.isnan(features.benchmark_roc60):
                failures.append("benchmark_roc60_missing")
                return False, failures
            
            relative_strength = features.roc60 - features.benchmark_roc60
            if relative_strength < self.config.eligibility.relative_strength_min:
                failures.append(f"insufficient_relative_strength_{relative_strength:.4f}")
                return False, failures
        
        return True, []
    
    def check_exit_signals(
        self, position: Position, features: FeatureRow
    ) -> Optional[ExitReason]:
        """Check if position should be exited.
        
        Exit priority:
        1. Hard stop: close < stop_price
        2. Trailing MA: close < exit_ma (MA20 or MA50)
        
        Args:
            position: Open position to check
            features: FeatureRow with current indicators
        
        Returns:
            ExitReason if exit triggered, None otherwise
        """
        # Priority 1: Hard stop
        if features.close <= position.stop_price:
            return ExitReason.HARD_STOP
        
        # Priority 2: Trailing MA cross
        exit_ma_period = self.config.exit.exit_ma
        if exit_ma_period == 20:
            if features.ma20 is None or np.isnan(features.ma20):
                return None
            ma_level = features.ma20
        elif exit_ma_period == 50:
            if features.ma50 is None or np.isnan(features.ma50):
                return None
            ma_level = features.ma50
        else:
            # Invalid config, skip
            return None
        
        if features.close < ma_level:
            return ExitReason.TRAILING_MA_CROSS
        
        return None
    
    def update_stop_price(
        self, position: Position, features: FeatureRow
    ) -> Optional[float]:
        """Update stop price for position (trailing stops).
        
        For equity strategy, stops are updated by the exit logic (MA cross).
        This method is called to potentially update trailing stops, but equity
        strategy doesn't use trailing stops that move up - it uses MA cross exits.
        
        Args:
            position: Open position
            features: FeatureRow with current indicators
        
        Returns:
            New stop price if updated, None if unchanged
        """
        # Equity strategy doesn't update stops (uses MA cross exits instead)
        # Stops remain at initial hard stop level
        return None


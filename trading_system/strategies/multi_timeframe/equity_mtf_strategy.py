"""Equity multi-timeframe strategy implementation."""

from typing import List, Optional
import numpy as np

from .mtf_strategy_base import MultiTimeframeBaseStrategy
from ...configs.strategy_config import StrategyConfig
from ...models.signals import Signal, SignalSide, SignalType, BreakoutType
from ...models.features import FeatureRow
from ...models.positions import Position, ExitReason


class EquityMultiTimeframeStrategy(MultiTimeframeBaseStrategy):
    """Equity multi-timeframe strategy.
    
    Entry logic:
    - Price above MA50 (higher timeframe trend filter)
    - Price >= weekly breakout level (lower timeframe entry)
    - Liquid assets only (ADV20 > threshold)
    
    Exit logic:
    - Price breaks below MA50 (higher timeframe trend break)
    - Hard stop (ATR-based)
    - Time stop (max_hold_days)
    
    Best for: Trending equities with clear higher timeframe trends
    """
    
    def __init__(self, config: StrategyConfig):
        """Initialize equity multi-timeframe strategy.
        
        Args:
            config: Strategy configuration (must have asset_class="equity")
        """
        if config.asset_class != "equity":
            raise ValueError(
                f"EquityMultiTimeframeStrategy requires asset_class='equity', got '{config.asset_class}'"
            )
        
        super().__init__(config)
        
        # Equity-specific params
        self.min_adv20 = config.parameters.get('min_adv20', 10_000_000)  # $10M minimum ADV
    
    def check_eligibility(self, features: FeatureRow) -> tuple[bool, List[str]]:
        """Check if symbol is eligible for multi-timeframe entry.
        
        Eligibility requirements:
        1. MA50 available (higher timeframe trend)
        2. Highest close 55d available (weekly breakout proxy)
        3. ATR14 available for stop calculation
        4. ADV20 above minimum threshold
        
        Args:
            features: FeatureRow with indicators for the symbol
        
        Returns:
            Tuple of (is_eligible, failure_reasons)
        """
        failures = []
        
        # Check basic price data
        if features.close <= 0 or features.atr14 is None:
            failures.append("insufficient_data")
            return False, failures
        
        # Check MA50 (higher timeframe trend)
        if features.ma50 is None or np.isnan(features.ma50):
            failures.append("ma50_missing")
            return False, failures
        
        # Check highest_close_55d (weekly breakout proxy)
        if features.highest_close_55d is None or np.isnan(features.highest_close_55d):
            failures.append("weekly_breakout_missing")
            return False, failures
        
        # Check ATR14 for stop calculation
        if features.atr14 is None or np.isnan(features.atr14) or features.atr14 <= 0:
            failures.append("atr14_missing")
            return False, failures
        
        # Check liquidity (ADV20)
        if features.adv20 is None or np.isnan(features.adv20):
            failures.append("adv20_missing")
            return False, failures
        
        if features.adv20 < self.min_adv20:
            failures.append(f"insufficient_liquidity_adv20_{features.adv20:.0f}")
            return False, failures
        
        return True, []
    
    def check_entry_triggers(
        self, features: FeatureRow
    ) -> tuple[Optional[BreakoutType], float]:
        """Check if multi-timeframe entry trigger is met.
        
        Entry trigger:
        1. Price above MA50 (higher timeframe trend filter)
        2. Price >= weekly breakout level (lower timeframe entry)
        
        Args:
            features: FeatureRow with indicators for the symbol
        
        Returns:
            Tuple of (breakout_type, clearance) or (None, 0.0) if no trigger.
            Uses BreakoutType.SLOW_55D as placeholder for MTF breakout.
        """
        # Check if required indicators are available
        if features.ma50 is None or np.isnan(features.ma50):
            return None, 0.0
        
        # Use highest_close_55d as proxy for weekly breakout
        # (In a true MTF system with intraday data, this would be computed on 4H bars)
        weekly_breakout = features.highest_close_55d
        if weekly_breakout is None or np.isnan(weekly_breakout):
            return None, 0.0
        
        # Higher timeframe trend filter: price must be above MA50
        if features.close <= features.ma50:
            return None, 0.0
        
        # Lower timeframe breakout: price must be >= weekly breakout level
        if features.close >= weekly_breakout:
            # Calculate clearance (how much above breakout level)
            clearance = (features.close - weekly_breakout) / weekly_breakout if weekly_breakout > 0 else 0.0
            return BreakoutType.SLOW_55D, clearance
        
        return None, 0.0
    
    def generate_signal(
        self,
        symbol: str,
        features: FeatureRow,
        order_notional: float,
        diversification_bonus: float = 0.0,
    ) -> Optional[Signal]:
        """Generate multi-timeframe entry signal for a symbol.
        
        Args:
            symbol: Symbol to generate signal for
            features: FeatureRow with indicators
            order_notional: Estimated order notional value
            diversification_bonus: Correlation bonus (1 - avg_corr)
            
        Returns:
            Signal if eligible and triggered, None otherwise
        """
        # Check basic price data
        if features.close <= 0:
            return None
        
        # Check eligibility
        is_eligible, failure_reasons = self.check_eligibility(features)
        
        # Check entry triggers
        breakout_type, clearance = self.check_entry_triggers(features)
        
        # If not eligible or no trigger, return None
        if not is_eligible or breakout_type is None:
            return None
        
        # Calculate stop price (ATR-based)
        stop_price = self.calculate_stop_price(
            features.close, features.atr14, self.stop_atr_mult
        )
        
        # Check capacity
        capacity_passed = self.check_capacity(order_notional, features.adv20)
        
        # Calculate score (based on clearance and trend strength)
        # Higher clearance = stronger breakout
        # Price further above MA50 = stronger trend
        ma50_distance = (features.close - features.ma50) / features.ma50 if features.ma50 > 0 else 0.0
        score = clearance + ma50_distance  # Combined breakout and trend strength
        
        # Calculate breakout strength (normalized by ATR)
        breakout_strength = self.calculate_breakout_strength(
            features.close, features.ma50, features.atr14
        )
        
        # Create signal
        signal = Signal(
            symbol=symbol,
            asset_class=self.asset_class,
            date=features.date,
            side=SignalSide.BUY,
            signal_type=SignalType.ENTRY_LONG,
            trigger_reason=f"mtf_weekly_breakout_clearance_{clearance:.3f}",
            metadata={
                'higher_tf_ma': self.higher_tf_ma,
                'weekly_lookback': self.weekly_lookback,
                'ma50': features.ma50,
                'weekly_breakout': features.highest_close_55d,
                'clearance': clearance,
                'ma50_distance': ma50_distance,
                'breakout_strength': breakout_strength,
            },
            urgency=0.7,  # Multi-timeframe signals have medium-high urgency
            entry_price=features.close,
            stop_price=stop_price,
            suggested_entry_price=features.close,
            suggested_stop_price=stop_price,
            breakout_strength=breakout_strength,
            momentum_strength=ma50_distance,  # Use trend distance as momentum proxy
            diversification_bonus=diversification_bonus,
            score=score,
            passed_eligibility=is_eligible,
            eligibility_failures=failure_reasons,
            order_notional=order_notional,
            adv20=features.adv20,
            capacity_passed=capacity_passed,
        )
        
        return signal
    
    def check_exit_signals(
        self, position: Position, features: FeatureRow
    ) -> Optional[ExitReason]:
        """Check if position should be exited.
        
        Exit priority:
        1. Hard stop: close < stop_price
        2. Higher timeframe trend break: close < MA50
        3. Time stop: hold_days >= max_hold_days
        
        Args:
            position: Open position to check
            features: FeatureRow with current indicators
        
        Returns:
            ExitReason if exit triggered, None otherwise
        """
        # Priority 1: Hard stop
        if features.close <= position.stop_price:
            return ExitReason.HARD_STOP
        
        # Priority 2: Higher timeframe trend break (price breaks below MA50)
        if features.ma50 is not None and not np.isnan(features.ma50):
            if features.close < features.ma50:
                return ExitReason.TRAILING_MA_CROSS  # Reuse existing exit reason
        
        # Priority 3: Time stop
        hold_days = (features.date - position.entry_date).days
        if hold_days >= self.max_hold_days:
            return ExitReason.MANUAL  # Use MANUAL for time-based exits
        
        return None
    
    def update_stop_price(
        self, position: Position, features: FeatureRow
    ) -> Optional[float]:
        """Update stop price for position (trailing stops).
        
        Multi-timeframe strategy can use trailing stops that move up
        as price moves in favor, but never below initial stop.
        
        Args:
            position: Open position
            features: FeatureRow with current indicators
        
        Returns:
            New stop price if updated, None if unchanged
        """
        # For now, use fixed ATR stop (can be enhanced with trailing logic later)
        # Multi-timeframe strategies typically use fixed stops or MA50-based stops
        return None


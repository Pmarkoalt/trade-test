"""Equity mean reversion strategy implementation."""

from typing import List, Optional

import numpy as np

from ...configs.strategy_config import StrategyConfig
from ...models.features import FeatureRow
from ...models.positions import ExitReason, Position, PositionSide
from ...models.signals import BreakoutType, Signal, SignalSide, SignalType
from .mean_reversion_base import MeanReversionBaseStrategy


class EquityMeanReversionStrategy(MeanReversionBaseStrategy):
    """Equity mean reversion strategy.

    Entry logic:
    - Price below lower Bollinger Band (z-score < -entry_std)
    - Liquid assets only (ADV20 > threshold)

    Exit logic:
    - Price reverts to mean (z-score >= -exit_std)
    - Time stop (max_hold_days)
    - Hard stop (ATR-based)

    Best for: Liquid ETFs (SPY, QQQ, etc.) with mean-reverting behavior
    """

    def __init__(self, config: StrategyConfig):
        """Initialize equity mean reversion strategy.

        Args:
            config: Strategy configuration (must have asset_class="equity")
        """
        if config.asset_class != "equity":
            raise ValueError(f"EquityMeanReversionStrategy requires asset_class='equity', got '{config.asset_class}'")

        super().__init__(config)

        # Equity-specific params
        if config.parameters is None:
            raise ValueError("config.parameters is required")
        self.min_adv20 = config.parameters.get("min_adv20", 10_000_000)  # $10M minimum ADV

    def check_eligibility(self, features: FeatureRow) -> tuple[bool, List[str]]:
        """Check if symbol is eligible for mean reversion entry.

        Eligibility requirements:
        1. Z-score available and valid
        2. ATR14 available for stop calculation
        3. ADV20 above minimum threshold

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

        # Check zscore is available
        if features.zscore is None or np.isnan(features.zscore):
            failures.append("zscore_missing")
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

    def check_entry_triggers(self, features: FeatureRow) -> tuple[Optional[BreakoutType], float]:
        """Check if mean reversion entry trigger is met.

        Entry triggers:
        - Long: z-score < -entry_std (oversold condition)
        - Short: z-score > entry_std (overbought condition)

        Args:
            features: FeatureRow with indicators for the symbol

        Returns:
            Tuple of (breakout_type, zscore) or (None, 0.0) if no trigger.
            Uses BreakoutType.FAST_20D as placeholder for mean reversion trigger.
        """
        # Check if zscore is available
        if not hasattr(features, "zscore") or features.zscore is None:
            return None, 0.0

        if np.isnan(features.zscore):
            return None, 0.0

        # Entry trigger: z-score below entry threshold (oversold) for long
        # or z-score above entry threshold (overbought) for short
        if features.zscore < -self.entry_std:
            # Long entry: oversold condition
            return BreakoutType.FAST_20D, features.zscore
        elif features.zscore > self.entry_std:
            # Short entry: overbought condition
            return BreakoutType.FAST_20D, features.zscore

        # Return zscore even when no trigger (for informational purposes)
        return None, features.zscore

    def generate_signal(
        self,
        symbol: str,
        features: FeatureRow,
        order_notional: float,
        diversification_bonus: float = 0.0,
    ) -> Optional[Signal]:
        """Generate mean reversion entry signal for a symbol.

        Generates long signals when oversold (z-score < -entry_std)
        and short signals when overbought (z-score > entry_std).

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
        breakout_type, zscore = self.check_entry_triggers(features)

        # If not eligible or no trigger, return None
        if not is_eligible or breakout_type is None:
            return None

        # Determine signal type and side based on zscore
        if zscore < -self.entry_std:
            # Long signal: oversold
            signal_type = SignalType.ENTRY_LONG
            side = SignalSide.BUY
            trigger_reason = f"mean_reversion_oversold_z{zscore:.2f}"
            # Stop below entry for long
            if features.close is None or features.atr14 is None:
                return None
            stop_price = self.calculate_stop_price(features.close, features.atr14, self.stop_atr_mult)
        elif zscore > self.entry_std:
            # Short signal: overbought
            signal_type = SignalType.ENTRY_SHORT
            side = SignalSide.SELL
            trigger_reason = f"mean_reversion_overbought_z{zscore:.2f}"
            # Stop above entry for short
            if features.close is None or features.atr14 is None:
                return None
            stop_price = features.close + (self.stop_atr_mult * features.atr14)
        else:
            return None

        # Check capacity
        if features.adv20 is None:
            return None
        capacity_passed = self.check_capacity(order_notional, features.adv20)

        # Calculate score (higher zscore magnitude = stronger signal)
        score = abs(zscore) / self.entry_std  # Normalize by entry threshold

        # Create signal with generic fields
        signal = Signal(
            symbol=symbol,
            asset_class=self.asset_class,
            date=features.date,
            side=side,
            signal_type=signal_type,
            trigger_reason=trigger_reason,
            metadata={
                "zscore": zscore,
                "entry_std": self.entry_std,
                "exit_std": self.exit_std,
                "lookback": self.lookback,
                "max_hold_days": self.max_hold_days,
                "ma_lookback": features.ma_lookback,
                "std_lookback": features.std_lookback,
            },
            urgency=0.7,  # Mean reversion signals have medium urgency
            entry_price=features.close,
            stop_price=stop_price,
            suggested_entry_price=features.close,
            suggested_stop_price=stop_price,
            breakout_strength=abs(zscore),  # Use zscore magnitude as strength
            momentum_strength=0.0,  # Not applicable for mean reversion
            diversification_bonus=diversification_bonus,
            score=score,
            passed_eligibility=is_eligible,
            eligibility_failures=failure_reasons,
            order_notional=order_notional,
            adv20=features.adv20,
            capacity_passed=capacity_passed,
        )

        return signal

    def check_exit_signals(self, position: Position, features: FeatureRow) -> Optional[ExitReason]:
        """Check if position should be exited.

        Exit priority:
        1. Hard stop: close < stop_price (long) or close > stop_price (short)
        2. Mean reversion target: z-score reverted to mean
           - Long: z-score >= -exit_std
           - Short: z-score <= exit_std
        3. Time stop: hold_days >= max_hold_days

        Args:
            position: Open position to check
            features: FeatureRow with current indicators

        Returns:
            ExitReason if exit triggered, None otherwise
        """
        # Priority 1: Hard stop (check based on position side)
        if position.side == PositionSide.LONG:
            if features.close <= position.stop_price:
                return ExitReason.HARD_STOP
        else:  # SHORT
            if features.close >= position.stop_price:
                return ExitReason.HARD_STOP

        # Priority 2: Mean reversion target (price reverted to mean)
        if features.zscore is not None and not np.isnan(features.zscore):
            if position.side == PositionSide.LONG:
                # Long: exit when z-score >= -exit_std (reverted to mean)
                if features.zscore >= -self.exit_std:
                    return ExitReason.TRAILING_MA_CROSS
            else:  # SHORT
                # Short: exit when z-score <= exit_std (reverted to mean)
                if features.zscore <= self.exit_std:
                    return ExitReason.TRAILING_MA_CROSS

        # Priority 3: Time stop
        hold_days = (features.date - position.entry_date).days
        if hold_days >= self.max_hold_days:
            return ExitReason.MANUAL  # Use MANUAL for time-based exits

        return None

    def update_stop_price(self, position: Position, features: FeatureRow) -> Optional[float]:
        """Update stop price for position (trailing stops).

        Mean reversion strategy doesn't use trailing stops that move up.
        Stops remain at initial ATR-based level.

        Args:
            position: Open position
            features: FeatureRow with current indicators

        Returns:
            New stop price if updated, None if unchanged
        """
        # Mean reversion doesn't update stops (uses fixed ATR stop)
        return None

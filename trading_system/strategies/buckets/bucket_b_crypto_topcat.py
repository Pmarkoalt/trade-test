"""Bucket B: Top-Cap Crypto Strategy

Aggressive crypto strategy for top market cap coins with:
- Dynamic universe selection (top N by market cap)
- Volatility-aware position sizing
- Adaptive stop/exit management
- Higher turnover tolerance
"""

from typing import List, Optional

import numpy as np

from ...configs.strategy_config import StrategyConfig
from ...models.features import FeatureRow
from ...models.positions import ExitReason, Position
from ...models.signals import BreakoutType, Signal, SignalSide, SignalType
from ..base.strategy_interface import StrategyInterface


class TopCapCryptoStrategy(StrategyInterface):
    """Top-cap crypto strategy with volatility-aware sizing.

    Eligibility:
    - close > MA200 (STRICT, no exceptions)
    - Optional: relative strength vs BTC
    - Optional: volatility filter (ATR14/close < max threshold)

    Entry triggers:
    - Fast: close >= highest_close_20d * 1.005
    - Slow: close >= highest_close_55d * 1.010

    Exit logic (staged):
    - Stage 1: close < MA20 → tighten stop to 2.0 * ATR14
    - Stage 2: close < MA50 OR tightened stop hit → exit
    - Hard stop: entry - 3.5 * ATR14 (wider for crypto volatility)

    Risk management:
    - Volatility-aware sizing: reduce size for high ATR/close ratio
    - Stricter capacity constraints (0.25% ADV vs 0.5% equity)
    - Dynamic position limits based on market conditions
    """

    def __init__(self, config: StrategyConfig):
        """Initialize Top-Cap Crypto strategy.

        Args:
            config: Strategy configuration (must have asset_class="crypto")
        """
        if config.asset_class != "crypto":
            raise ValueError(f"TopCapCryptoStrategy requires asset_class='crypto', got '{config.asset_class}'")

        super().__init__(config)

        if config.exit.mode != "staged":
            raise ValueError(f"TopCapCryptoStrategy requires exit.mode='staged', got '{config.exit.mode}'")

        self.volatility_filter_enabled = getattr(config.eligibility, "volatility_filter_enabled", False)
        self.max_volatility_ratio = getattr(config.eligibility, "max_volatility_ratio", 0.15)

    def check_eligibility(self, features: FeatureRow) -> tuple[bool, List[str]]:
        """Check if symbol is eligible for entry.

        Requirements:
        1. close > MA200 (STRICT, no exceptions)
        2. Optional: relative strength vs BTC
        3. Optional: volatility filter (ATR14/close < threshold)

        Args:
            features: FeatureRow with indicators for the symbol

        Returns:
            Tuple of (is_eligible, failure_reasons)
        """
        failures = []

        if not features.is_valid_for_entry():
            failures.append("insufficient_data")
            return False, failures

        # 1. Check MA200 (STRICT requirement)
        if features.ma200 is None or np.isnan(features.ma200):
            failures.append("ma200_missing")
            return False, failures

        if features.close <= features.ma200:
            failures.append("below_MA200")
            return False, failures

        # 2. Optional: relative strength vs BTC
        if self.config.eligibility.relative_strength_enabled:
            if features.roc60 is None or np.isnan(features.roc60):
                failures.append("roc60_missing")
                return False, failures

            if features.benchmark_roc60 is None or np.isnan(features.benchmark_roc60):
                failures.append("benchmark_roc60_missing")
                return False, failures

            relative_strength = features.roc60 - features.benchmark_roc60
            if relative_strength <= self.config.eligibility.relative_strength_min:
                failures.append(f"insufficient_relative_strength_{relative_strength:.4f}")
                return False, failures

        # 3. Optional: volatility filter
        if self.volatility_filter_enabled:
            if features.atr14 is None or np.isnan(features.atr14):
                failures.append("atr14_missing")
                return False, failures

            volatility_ratio = features.atr14 / features.close
            if volatility_ratio > self.max_volatility_ratio:
                failures.append(f"excessive_volatility_{volatility_ratio:.4f}")
                return False, failures

        return True, []

    def check_entry_triggers(self, features: FeatureRow) -> tuple[Optional[BreakoutType], float]:
        """Check if entry triggers are met.

        Entry triggers (OR logic):
        - Fast: close >= highest_close_20d * (1 + fast_clearance)
        - Slow: close >= highest_close_55d * (1 + slow_clearance)

        Args:
            features: FeatureRow with indicators for the symbol

        Returns:
            Tuple of (breakout_type, clearance) or (None, 0.0) if no trigger
        """
        if features.close is None or np.isnan(features.close):
            return None, 0.0

        fast_clearance = self.config.entry.fast_clearance
        slow_clearance = self.config.entry.slow_clearance

        # Fast breakout (20D)
        if features.highest_close_20d is not None and not np.isnan(features.highest_close_20d):
            fast_threshold = features.highest_close_20d * (1.0 + fast_clearance)
            if features.close >= fast_threshold:
                clearance = (features.close - features.highest_close_20d) / features.highest_close_20d
                return BreakoutType.FAST_20D, clearance

        # Slow breakout (55D)
        if features.highest_close_55d is not None and not np.isnan(features.highest_close_55d):
            slow_threshold = features.highest_close_55d * (1.0 + slow_clearance)
            if features.close >= slow_threshold:
                clearance = (features.close - features.highest_close_55d) / features.highest_close_55d
                return BreakoutType.SLOW_55D, clearance

        return None, 0.0

    def check_exit_signals(self, position: Position, features: FeatureRow) -> Optional[ExitReason]:
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

    def update_stop_price(self, position: Position, features: FeatureRow) -> Optional[float]:
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
        if not position.tightened_stop and features.ma20 is not None and not np.isnan(features.ma20):
            if features.close < features.ma20:
                # Tighten stop to 2.0 * ATR14
                if features.atr14 is not None and not np.isnan(features.atr14):
                    atr14_val = float(features.atr14)
                    tightened_stop = position.entry_price - (self.config.exit.tightened_stop_atr_mult * atr14_val)

                    # Only tighten if new stop is higher than current (can't move down)
                    if tightened_stop > position.stop_price:
                        position.update_stop(tightened_stop, reason="tighten")
                        return tightened_stop

        return None

    def calculate_volatility_adjustment(self, features: FeatureRow) -> float:
        """Calculate position size adjustment based on volatility.

        Higher volatility → smaller position size
        Formula: adjustment = 1.0 / (1.0 + volatility_ratio * scale_factor)

        Args:
            features: FeatureRow with indicators

        Returns:
            Adjustment factor (0.0-1.0)
        """
        if features.atr14 is None or np.isnan(features.atr14):
            return 1.0

        volatility_ratio = features.atr14 / features.close
        scale_factor = 10.0

        adjustment = 1.0 / (1.0 + volatility_ratio * scale_factor)

        return max(0.3, min(1.0, adjustment))

    def generate_signal(
        self,
        symbol: str,
        features: FeatureRow,
        order_notional: float,
        diversification_bonus: float = 0.0,
    ) -> Optional[Signal]:
        """Generate entry signal with volatility-aware sizing and rationale tags.

        Args:
            symbol: Symbol to generate signal for
            features: FeatureRow with indicators
            order_notional: Estimated order notional value
            diversification_bonus: Correlation bonus (1 - avg_corr)

        Returns:
            Signal with rationale tags if eligible and triggered, None otherwise
        """
        if not features.is_valid_for_entry():
            return None

        is_eligible, failure_reasons = self.check_eligibility(features)
        breakout_type, clearance = self.check_entry_triggers(features)

        if not is_eligible or breakout_type is None:
            return None

        # Calculate stop price
        atr_mult = self.config.exit.hard_stop_atr_mult
        if features.close is None or features.atr14 is None:
            return None
        stop_price = self.calculate_stop_price(features.close, features.atr14, atr_mult)

        # Check capacity
        if features.adv20 is None:
            return None
        capacity_passed = self.check_capacity(order_notional, features.adv20)

        # Calculate scoring components
        if features.ma50 is None:
            return None
        breakout_strength = self.calculate_breakout_strength(features.close, features.ma50, features.atr14)
        momentum_strength = self.calculate_momentum_strength(features.roc60, features.benchmark_roc60)

        # Calculate volatility adjustment
        volatility_adjustment = self.calculate_volatility_adjustment(features)

        # Build rationale tags for newsletter
        rationale_tags = self._build_rationale_tags(features, breakout_type, clearance, volatility_adjustment)

        # Create signal
        trigger_reason = f"topcat_crypto_{breakout_type.value.lower()}_breakout"

        signal = Signal(
            symbol=symbol,
            asset_class=self.asset_class,
            date=features.date,
            side=SignalSide.BUY,
            signal_type=SignalType.ENTRY_LONG,
            trigger_reason=trigger_reason,
            metadata={
                "breakout_type": breakout_type.value,
                "breakout_clearance": clearance,
                "breakout_strength": breakout_strength,
                "momentum_strength": momentum_strength,
                "roc60": features.roc60,
                "benchmark_roc60": features.benchmark_roc60,
                "volatility_adjustment": volatility_adjustment,
                "volatility_ratio": features.atr14 / features.close,
                "rationale_tags": rationale_tags,
                "bucket": "B_TOPCAT_CRYPTO",
            },
            urgency=0.7,
            entry_price=features.close,
            stop_price=stop_price,
            suggested_entry_price=features.close,
            suggested_stop_price=stop_price,
            atr_mult=atr_mult,
            triggered_on=breakout_type,
            breakout_clearance=clearance,
            breakout_strength=breakout_strength,
            momentum_strength=momentum_strength,
            diversification_bonus=diversification_bonus,
            score=0.0,
            passed_eligibility=is_eligible,
            eligibility_failures=failure_reasons,
            order_notional=order_notional,
            adv20=features.adv20 or 0.0,
            capacity_passed=capacity_passed,
        )

        return signal

    def _build_rationale_tags(
        self, features: FeatureRow, breakout_type: BreakoutType, clearance: float, volatility_adjustment: float
    ) -> List[str]:
        """Build rationale tags for newsletter.

        Args:
            features: FeatureRow with indicators
            breakout_type: Type of breakout triggered
            clearance: Breakout clearance percentage
            volatility_adjustment: Volatility-based size adjustment

        Returns:
            List of rationale tags
        """
        tags = []

        # Technical reasons
        if breakout_type == BreakoutType.FAST_20D:
            tags.append("technical_20d_breakout")
        elif breakout_type == BreakoutType.SLOW_55D:
            tags.append("technical_55d_breakout")

        tags.append("technical_above_ma200")

        # Momentum vs BTC
        if features.roc60 is not None and features.benchmark_roc60 is not None:
            relative_strength = features.roc60 - features.benchmark_roc60
            if relative_strength > 0.10:
                tags.append("technical_strong_outperformance_vs_btc")
            elif relative_strength > 0:
                tags.append("technical_positive_relative_strength_vs_btc")
            elif relative_strength < -0.10:
                tags.append("technical_underperformance_vs_btc")

        # Volatility context
        if volatility_adjustment < 0.7:
            tags.append("risk_high_volatility_reduced_size")
        elif volatility_adjustment > 0.9:
            tags.append("risk_low_volatility_normal_size")

        return tags

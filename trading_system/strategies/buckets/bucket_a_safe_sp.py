"""Bucket A: Safe S&P Strategy

Conservative equity strategy for S&P 500 universe with:
- Regime filters (market trend, risk-off gating)
- News sentiment integration
- Strict correlation and concentration limits
- Low drawdown focus
"""

from typing import List, Optional

import numpy as np

from ...configs.strategy_config import StrategyConfig
from ...models.features import FeatureRow
from ...models.positions import ExitReason, Position
from ...models.signals import BreakoutType, Signal, SignalSide, SignalType
from ..base.strategy_interface import StrategyInterface


class SafeSPStrategy(StrategyInterface):
    """Safe S&P strategy with regime filters and news sentiment.

    Eligibility:
    - close > MA200 (long-term trend)
    - MA50 > MA200 (bullish regime)
    - MA50 slope > 0.3% over 20 days (momentum confirmation)
    - Optional: SPY > MA50 (market regime filter)
    - Optional: news sentiment score > threshold (if available)

    Entry triggers:
    - Fast: close >= highest_close_20d * 1.003 (tighter than standard)
    - Slow: close >= highest_close_55d * 1.008 (tighter than standard)

    Exit logic:
    - Trailing: close < MA50 (conservative, slower exit)
    - Hard stop: close < entry - 2.0 * ATR14 (tighter than standard equity)

    Risk management:
    - Lower position sizing (0.5% risk per trade vs 0.75% standard)
    - Stricter correlation limits
    - Concentration limits per sector (if sector data available)
    """

    def __init__(self, config: StrategyConfig):
        """Initialize Safe S&P strategy.

        Args:
            config: Strategy configuration (must have asset_class="equity")
        """
        if config.asset_class != "equity":
            raise ValueError(f"SafeSPStrategy requires asset_class='equity', got '{config.asset_class}'")

        super().__init__(config)

        self.market_regime_enabled = getattr(config.eligibility, "market_regime_enabled", True)
        self.news_sentiment_enabled = getattr(config.eligibility, "news_sentiment_enabled", False)
        self.news_sentiment_min = getattr(config.eligibility, "news_sentiment_min", 0.0)

    def check_eligibility(self, features: FeatureRow) -> tuple[bool, List[str]]:
        """Check if symbol is eligible for entry.

        Eligibility requirements:
        1. close > MA200 (long-term trend)
        2. MA50 > MA200 (bullish regime)
        3. MA50 slope > 0.3% over 20 days
        4. Optional: SPY > MA50 (market regime filter)
        5. Optional: news sentiment > threshold

        Args:
            features: FeatureRow with indicators for the symbol

        Returns:
            Tuple of (is_eligible, failure_reasons)
        """
        failures = []

        if not features.is_valid_for_entry():
            failures.append("insufficient_data")
            return False, failures

        # 1. Check close > MA200
        if features.ma200 is None or np.isnan(features.ma200):
            failures.append("ma200_missing")
            return False, failures

        if features.close <= features.ma200:
            failures.append("below_MA200")
            return False, failures

        # 2. Check MA50 > MA200 (bullish regime)
        if features.ma50 is None or np.isnan(features.ma50):
            failures.append("ma50_missing")
            return False, failures

        if features.ma50 <= features.ma200:
            failures.append("bearish_regime_ma50_below_ma200")
            return False, failures

        # 3. Check MA50 slope > 0.3% over 20 days (tighter than standard 0.5%)
        if features.ma50_slope is None or np.isnan(features.ma50_slope):
            failures.append("ma50_slope_missing")
            return False, failures

        ma_slope_min = getattr(self.config.eligibility, "ma_slope_min", 0.003)
        if features.ma50_slope <= ma_slope_min:
            failures.append(f"insufficient_ma50_slope_{features.ma50_slope:.6f}")
            return False, failures

        # 4. Optional: Market regime filter (SPY > MA50)
        if self.market_regime_enabled:
            if features.benchmark_ma50 is not None and not np.isnan(features.benchmark_ma50):
                benchmark_close = getattr(features, "benchmark_close", None)
                if benchmark_close is not None and not np.isnan(benchmark_close):
                    if benchmark_close <= features.benchmark_ma50:
                        failures.append("market_regime_bearish_spy_below_ma50")
                        return False, failures

        # 5. Optional: News sentiment filter
        if self.news_sentiment_enabled:
            news_sentiment = getattr(features, "news_sentiment", None)
            if news_sentiment is not None and not np.isnan(news_sentiment):
                if news_sentiment < self.news_sentiment_min:
                    failures.append(f"negative_news_sentiment_{news_sentiment:.4f}")
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

        Exit priority:
        1. Hard stop: close < stop_price
        2. Trailing MA: close < MA50 (conservative exit)

        Args:
            position: Open position to check
            features: FeatureRow with current indicators

        Returns:
            ExitReason if exit triggered, None otherwise
        """
        # Priority 1: Hard stop
        if features.close <= position.stop_price:
            return ExitReason.HARD_STOP

        # Priority 2: MA50 cross (conservative exit)
        if features.ma50 is None or np.isnan(features.ma50):
            return None

        if features.close < features.ma50:
            return ExitReason.TRAILING_MA_CROSS

        return None

    def update_stop_price(self, position: Position, features: FeatureRow) -> Optional[float]:
        """Update stop price for position.

        Safe S&P strategy uses MA50 cross for exits, not trailing stops.
        Stops remain at initial hard stop level.

        Args:
            position: Open position
            features: FeatureRow with current indicators

        Returns:
            New stop price if updated, None if unchanged
        """
        return None

    def generate_signal(
        self,
        symbol: str,
        features: FeatureRow,
        order_notional: float,
        diversification_bonus: float = 0.0,
    ) -> Optional[Signal]:
        """Generate entry signal with rationale tags for newsletter.

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

        # Build rationale tags for newsletter
        rationale_tags = self._build_rationale_tags(features, breakout_type, clearance)

        # Create signal
        trigger_reason = f"safe_sp_{breakout_type.value.lower()}_breakout"

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
                "rationale_tags": rationale_tags,
                "bucket": "A_SAFE_SP",
            },
            urgency=0.5,
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

    def _build_rationale_tags(self, features: FeatureRow, breakout_type: BreakoutType, clearance: float) -> List[str]:
        """Build rationale tags for newsletter.

        Args:
            features: FeatureRow with indicators
            breakout_type: Type of breakout triggered
            clearance: Breakout clearance percentage

        Returns:
            List of rationale tags
        """
        tags = []

        # Technical reasons
        if breakout_type == BreakoutType.FAST_20D:
            tags.append("technical_20d_breakout")
        elif breakout_type == BreakoutType.SLOW_55D:
            tags.append("technical_55d_breakout")

        tags.append("technical_bullish_regime")
        tags.append("technical_above_ma200")

        # Momentum
        if features.roc60 is not None and features.benchmark_roc60 is not None:
            relative_strength = features.roc60 - features.benchmark_roc60
            if relative_strength > 0.05:
                tags.append("technical_strong_relative_strength")
            elif relative_strength > 0:
                tags.append("technical_positive_relative_strength")

        # News sentiment (if available)
        if self.news_sentiment_enabled:
            news_sentiment = getattr(features, "news_sentiment", None)
            if news_sentiment is not None and not np.isnan(news_sentiment):
                if news_sentiment > 0.5:
                    tags.append("news_positive_sentiment")
                elif news_sentiment > 0.2:
                    tags.append("news_neutral_sentiment")

        return tags

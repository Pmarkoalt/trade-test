"""Equity factor-based strategy implementation."""

from datetime import datetime
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from ...configs.strategy_config import StrategyConfig
from ...models.features import FeatureRow
from ...models.positions import ExitReason, Position
from ...models.signals import BreakoutType, Signal, SignalSide, SignalType
from .factor_base import FactorBaseStrategy


class EquityFactorStrategy(FactorBaseStrategy):
    """Equity factor-based strategy.

    Multi-factor strategy that ranks assets by:
    - Momentum: 12-month return (ROC252)
    - Value: Distance from 52W high (proxy for value)
    - Quality: Inverse volatility (lower vol = higher quality)

    Entry logic:
    - On rebalance days, rank all assets by composite factor score
    - Enter long positions in top decile

    Exit logic:
    - Exit positions not in top decile on rebalance days
    - Hard stop (ATR-based)
    - Time-based exit (optional)

    Best for: Large-cap equities with sufficient history
    """

    def __init__(self, config: StrategyConfig):
        """Initialize equity factor strategy.

        Args:
            config: Strategy configuration (must have asset_class="equity")
        """
        if config.asset_class != "equity":
            raise ValueError(f"EquityFactorStrategy requires asset_class='equity', got '{config.asset_class}'")

        super().__init__(config)

        # Equity-specific params
        self.min_adv20 = config.parameters.get("min_adv20", 10_000_000)  # $10M minimum ADV
        self.max_hold_days = config.parameters.get("max_hold_days", 90)  # Hold until next rebalance

    def compute_factor_score(self, features: FeatureRow) -> Optional[float]:
        """Compute composite factor score for a symbol.

        Factors computed:
        - Momentum: 12-month return (requires ROC252, approximated from available data)
        - Value: Distance from 52W high (1 - close/high_52w)
        - Quality: Inverse volatility (1 / (volatility + epsilon))

        Note: For MVP, we use proxies since fundamental data isn't available.
        In production, these would use actual fundamental metrics.

        Args:
            features: FeatureRow with indicators for the symbol

        Returns:
            Composite factor score (z-score normalized), or None if insufficient data
        """
        # Check if we have minimum required data
        if features.close <= 0 or features.atr14 is None:
            return None

        # Factor 1: Momentum (12-month return)
        # Use ROC60 as proxy, or compute from available data
        # For now, use roc60 scaled to approximate 12-month return
        momentum_raw = features.roc60
        if momentum_raw is None or np.isnan(momentum_raw):
            # Try to approximate from ma50 slope if available
            if hasattr(features, "ma50_slope") and features.ma50_slope is not None:
                momentum_raw = features.ma50_slope * 12.6  # Scale to approximate annual
            else:
                momentum_raw = 0.0

        # Factor 2: Value (distance from 52W high)
        # Proxy: use highest_close_55d as approximation for 52W high
        # Higher distance from high = better value
        if features.highest_close_55d is not None and not np.isnan(features.highest_close_55d):
            if features.highest_close_55d > 0:
                value_raw = 1.0 - (features.close / features.highest_close_55d)
            else:
                value_raw = 0.0
        else:
            value_raw = 0.0

        # Factor 3: Quality (inverse volatility)
        # Use ATR14 normalized by price as volatility proxy
        if features.atr14 is not None and not np.isnan(features.atr14) and features.close > 0:
            volatility = (features.atr14 / features.close) * np.sqrt(252)  # Annualized
            quality_raw = 1.0 / (volatility + 0.01)  # Inverse vol, add epsilon to avoid division by zero
        else:
            quality_raw = 0.0

        # Store raw factors for metadata
        raw_factors = {"momentum": momentum_raw, "value": value_raw, "quality": quality_raw}

        # Composite score (weighted sum of raw factors)
        # Note: In production, factors would be z-score normalized across universe
        # For now, use raw weighted sum (normalization happens in ranking)
        composite_score = (
            self.factors["momentum"] * momentum_raw + self.factors["value"] * value_raw + self.factors["quality"] * quality_raw
        )

        # Store raw factors in cache for later use
        if not hasattr(self, "_raw_factors_cache"):
            self._raw_factors_cache: Dict[str, Dict[str, float]] = {}
        self._raw_factors_cache[features.symbol] = raw_factors

        return composite_score

    def check_eligibility(self, features: FeatureRow) -> tuple[bool, List[str]]:
        """Check if symbol is eligible for factor-based entry.

        Eligibility requirements:
        1. Sufficient price data
        2. ATR14 available for stop calculation
        3. ADV20 above minimum threshold
        4. Factor score can be computed

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

        # Check if factor score can be computed
        factor_score = self.compute_factor_score(features)
        if factor_score is None or np.isnan(factor_score):
            failures.append("factor_score_unavailable")
            return False, failures

        return True, []

    def _update_top_decile(self, current_date: datetime) -> None:
        """Update top decile symbols based on cached factor scores.

        This should be called after all symbols have been processed for a rebalance day.
        For now, we'll determine top decile based on cached scores.

        Args:
            current_date: Current rebalance date
        """
        if not self._factor_scores_cache:
            self._top_decile_symbols = set()
            return

        # Sort symbols by factor score (descending)
        sorted_symbols = sorted(self._factor_scores_cache.items(), key=lambda x: x[1], reverse=True)

        # Select top decile
        n_select = max(1, int(len(sorted_symbols) * self.top_decile_pct))
        self._top_decile_symbols = {symbol for symbol, _ in sorted_symbols[:n_select]}

        # Update rebalance date
        self._current_rebalance_date = current_date

    def generate_signal(
        self,
        symbol: str,
        features: FeatureRow,
        order_notional: float,
        diversification_bonus: float = 0.0,
    ) -> Optional[Signal]:
        """Generate factor-based entry signal for a symbol.

        Args:
            symbol: Symbol to generate signal for
            features: FeatureRow with indicators
            order_notional: Estimated order notional value
            diversification_bonus: Correlation bonus (1 - avg_corr)

        Returns:
            Signal if eligible and in top decile on rebalance day, None otherwise
        """
        # Check basic price data
        if features.close <= 0:
            return None

        # Check if it's a rebalance day
        current_date = features.date.to_pydatetime()
        is_rebalance = self._is_rebalance_day(current_date)

        # If new rebalance day, reset cache and update top decile
        if is_rebalance:
            if self._current_rebalance_date != current_date:
                # New rebalance day - update top decile after processing all symbols
                # For now, we'll use a simple approach: generate signal if score is high
                # In production, this would be called after all symbols processed
                self._factor_scores_cache = {}
                self._current_rebalance_date = current_date

        # Check eligibility
        is_eligible, failure_reasons = self.check_eligibility(features)

        # Check entry triggers (computes and caches factor score)
        breakout_type, factor_score = self.check_entry_triggers(features)

        # If not eligible or no trigger, return None
        if not is_eligible or breakout_type is None:
            return None

        # On rebalance days, update top decile after collecting all scores
        # For now, we'll generate signal if score is in top half (will be refined)
        # In production, this would be determined after all symbols processed
        if is_rebalance:
            # Update top decile (this is a simplification - ideally done after all symbols)
            self._update_top_decile(current_date)

            # Only generate signal if symbol is in top decile
            if symbol not in self._top_decile_symbols:
                return None

        # Calculate stop price (ATR-based)
        stop_price = self.calculate_stop_price(features.close, features.atr14, self.stop_atr_mult)

        # Check capacity
        capacity_passed = self.check_capacity(order_notional, features.adv20)

        # Get raw factors for metadata
        raw_factors = self._raw_factors_cache.get(symbol, {})

        # Calculate score (factor score normalized)
        score = factor_score  # Will be normalized in signal queue

        # Create signal
        signal = Signal(
            symbol=symbol,
            asset_class=self.asset_class,
            date=features.date,
            side=SignalSide.BUY,
            signal_type=SignalType.ENTRY_LONG,
            trigger_reason=f"factor_rebalance_score_{factor_score:.3f}",
            metadata={
                "factor_score": factor_score,
                "momentum": raw_factors.get("momentum", 0.0),
                "value": raw_factors.get("value", 0.0),
                "quality": raw_factors.get("quality", 0.0),
                "factors": self.factors,
                "rebalance_frequency": self.rebalance_frequency,
                "top_decile_pct": self.top_decile_pct,
            },
            urgency=0.5,  # Low urgency, rebalance trades
            entry_price=features.close,
            stop_price=stop_price,
            suggested_entry_price=features.close,
            suggested_stop_price=stop_price,
            breakout_strength=factor_score,  # Use factor score as strength
            momentum_strength=raw_factors.get("momentum", 0.0),
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
        1. Hard stop: close < stop_price
        2. Rebalance exit: position not in top decile on rebalance day
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

        # Priority 2: Rebalance exit (not in top decile)
        current_date = features.date.to_pydatetime()
        if self._is_rebalance_day(current_date):
            # Update top decile if needed
            if self._current_rebalance_date != current_date:
                # Compute factor score for this symbol
                factor_score = self.compute_factor_score(features)
                if factor_score is not None:
                    self._factor_scores_cache[position.symbol] = factor_score
                self._update_top_decile(current_date)

            # Exit if not in top decile
            if position.symbol not in self._top_decile_symbols:
                return ExitReason.MANUAL  # Use MANUAL for rebalance exits

        # Priority 3: Time stop
        hold_days = (features.date - position.entry_date).days
        if hold_days >= self.max_hold_days:
            return ExitReason.MANUAL  # Use MANUAL for time-based exits

        return None

    def update_stop_price(self, position: Position, features: FeatureRow) -> Optional[float]:
        """Update stop price for position (trailing stops).

        Factor strategy doesn't use trailing stops that move up.
        Stops remain at initial ATR-based level.

        Args:
            position: Open position
            features: FeatureRow with current indicators

        Returns:
            New stop price if updated, None if unchanged
        """
        # Factor strategy doesn't update stops (uses fixed ATR stop)
        return None

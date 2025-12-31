"""Pairs trading strategy implementation.

Pairs trading looks for divergence in correlated pairs and trades the spread:
- When spread widens (z-score > entry_zscore): Short outperformer, long underperformer
- When spread converges (z-score < exit_zscore): Exit both legs

Note: This strategy requires the portfolio manager to support short positions.
For MVP, signals are generated but execution may need portfolio manager updates.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...configs.strategy_config import StrategyConfig
from ...models.features import FeatureRow
from ...models.positions import ExitReason, Position, PositionSide
from ...models.signals import BreakoutType, Signal, SignalSide, SignalType
from ..base.strategy_interface import StrategyInterface


class PairsTradingStrategy(StrategyInterface):
    """Pairs trading strategy for correlated asset pairs.

    This strategy works on PAIRS of assets, not individual assets.
    It generates two signals simultaneously (one long, one short) when
    the spread between the pair diverges from its historical mean.

    Entry logic:
    - Compute spread = log(price1 / price2)
    - Compute z-score of spread over lookback period
    - When z-score > entry_zscore: Short symbol1, long symbol2 (spread too wide)
    - When z-score < -entry_zscore: Long symbol1, short symbol2 (spread too narrow)

    Exit logic:
    - When |z-score| < exit_zscore: Exit both legs (spread converged)
    - Time stop: max_hold_days
    - Hard stop: Individual leg stops (ATR-based)

    Best for: Highly correlated pairs (sector ETFs, index ETFs, etc.)
    """

    def __init__(self, config: StrategyConfig):
        """Initialize pairs trading strategy.

        Args:
            config: Strategy configuration
        """
        super().__init__(config)

        # Extract pairs trading parameters
        if config.parameters is None:
            raise ValueError("config.parameters is required")
        params = config.parameters
        self.pairs = params.get("pairs", [])  # List of (symbol1, symbol2) tuples
        if not isinstance(self.pairs, list):
            raise ValueError("pairs must be a list of [symbol1, symbol2] tuples")

        # Validate pairs format
        for pair in self.pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"Each pair must be [symbol1, symbol2], got: {pair}")

        self.lookback = params.get("lookback", 60)  # Spread lookback period
        self.entry_zscore = params.get("entry_zscore", 2.0)  # Enter when |zscore| > this
        self.exit_zscore = params.get("exit_zscore", 0.5)  # Exit when |zscore| < this
        self.max_hold_days = params.get("max_hold_days", 10)
        self.atr_period = params.get("atr_period", 14)
        self.stop_atr_mult = params.get("stop_atr_mult", 2.0)

        # Validate we have pairs
        if not self.pairs:
            raise ValueError("pairs trading strategy requires at least one pair in parameters.pairs")

    def check_eligibility(self, features: FeatureRow) -> tuple[bool, List[str]]:
        """Check if symbol is eligible for pairs trading.

        Note: For pairs trading, eligibility is checked at the PAIR level,
        not individual symbol level. This method is called per symbol but
        should return True if the symbol is part of a valid pair.

        Args:
            features: FeatureRow with indicators for the symbol

        Returns:
            Tuple of (is_eligible, failure_reasons)
        """
        failures = []

        # Basic checks: need price data and ATR
        if features.close <= 0:
            failures.append("invalid_price")
            return False, failures

        if features.atr14 is None or np.isnan(features.atr14) or features.atr14 <= 0:
            failures.append("atr14_missing")
            return False, failures

        # Check if symbol is part of any pair
        is_in_pair = any(features.symbol in pair for pair in self.pairs)

        if not is_in_pair:
            failures.append("not_in_pair")
            return False, failures

        return True, []

    def check_entry_triggers(self, features: FeatureRow) -> tuple[Optional[BreakoutType], float]:
        """Check if entry triggers are met.

        Note: For pairs trading, entry triggers are checked at the PAIR level.
        This method returns a placeholder - actual trigger logic is in generate_pair_signals.

        Args:
            features: FeatureRow with indicators for the symbol

        Returns:
            Tuple of (breakout_type, value) or (None, 0.0)
        """
        # Pairs trading doesn't use individual symbol triggers
        # Triggers are checked at pair level in generate_pair_signals
        return None, 0.0

    def compute_spread_zscore(
        self, price1: float, price2: float, spread_history: Optional[pd.Series] = None
    ) -> Optional[float]:
        """Compute z-score of spread between two symbols.

        Args:
            price1: Current price of symbol1
            price2: Current price of symbol2
            spread_history: Historical spread series (log(price1/price2))

        Returns:
            Z-score of current spread, or None if insufficient history
        """
        if price1 <= 0 or price2 <= 0:
            return None

        if spread_history is None or len(spread_history) < self.lookback:
            return None

        # Current spread
        current_spread = np.log(price1 / price2)

        # Rolling mean and std of spread
        spread_ma = spread_history.rolling(window=self.lookback, min_periods=self.lookback).mean()
        spread_std = spread_history.rolling(window=self.lookback, min_periods=self.lookback).std()

        if len(spread_ma) == 0 or len(spread_std) == 0:
            return None

        # Get latest values
        latest_ma = spread_ma.iloc[-1]
        latest_std = spread_std.iloc[-1]

        if pd.isna(latest_ma) or pd.isna(latest_std) or latest_std == 0:
            return None

        # Z-score
        zscore = (current_spread - latest_ma) / latest_std

        return float(zscore)

    def compute_hedge_ratio(self, returns1: List[float], returns2: List[float], method: str = "beta") -> Optional[float]:
        """Compute hedge ratio for pairs trading.

        The hedge ratio determines the relative size of the two legs to make
        the pair market-neutral. Common methods:
        - "beta": Linear regression beta (returns1 ~ returns2)
        - "correlation": Correlation-adjusted ratio
        - "equal": 1:1 ratio (default if insufficient data)

        Args:
            returns1: List of daily returns for symbol1
            returns2: List of daily returns for symbol2
            method: Method to use ("beta", "correlation", or "equal")

        Returns:
            Hedge ratio (quantity1 / quantity2), or None if insufficient data
            For equal dollar amounts, use price2 / price1

        Example:
            >>> ratio = compute_hedge_ratio(returns1, returns2, method="beta")
            >>> # Returns ~1.2 means symbol1 moves 1.2x for every 1x move in symbol2
        """
        if len(returns1) < 20 or len(returns2) < 20:
            # Insufficient data: use equal dollar amounts
            return None

        # Align lengths
        min_len = min(len(returns1), len(returns2))
        returns1_aligned = returns1[-min_len:]
        returns2_aligned = returns2[-min_len:]

        if method == "beta":
            # Linear regression: returns1 = alpha + beta * returns2
            # Hedge ratio = beta (how much symbol1 moves per unit move in symbol2)
            try:
                # Compute covariance and variance
                cov = np.cov(returns1_aligned, returns2_aligned)[0, 1]
                var2 = np.var(returns2_aligned)

                if var2 == 0:
                    return None

                beta = cov / var2
                return float(beta)
            except Exception:
                return None

        elif method == "correlation":
            # Correlation-adjusted ratio
            try:
                corr = np.corrcoef(returns1_aligned, returns2_aligned)[0, 1]
                if np.isnan(corr) or corr == 0:
                    return None

                # Use correlation as a scaling factor
                # Higher correlation = closer to 1:1 ratio
                return float(corr)
            except Exception:
                return None

        elif method == "equal":
            # Equal dollar amounts (1:1 ratio)
            return 1.0

        else:
            raise ValueError(f"Unknown hedge ratio method: {method}")

    def generate_pair_signals(
        self,
        symbol1: str,
        symbol2: str,
        features1: FeatureRow,
        features2: FeatureRow,
        spread_zscore: float,
        order_notional: float,
        date: pd.Timestamp,
        returns1: Optional[List[float]] = None,
        returns2: Optional[List[float]] = None,
    ) -> List[Signal]:
        """Generate signals for a pair when spread diverges.

        Args:
            symbol1: First symbol in pair
            symbol2: Second symbol in pair
            features1: FeatureRow for symbol1
            features2: FeatureRow for symbol2
            spread_zscore: Z-score of spread
            order_notional: Estimated order notional value
            date: Current date
            returns1: Optional historical returns for symbol1 (for hedge ratio)
            returns2: Optional historical returns for symbol2 (for hedge ratio)

        Returns:
            List of 2 signals (one long, one short) or empty list
        """
        signals: List[Signal] = []

        # Check if both symbols are eligible
        eligible1, _ = self.check_eligibility(features1)
        eligible2, _ = self.check_eligibility(features2)

        if not eligible1 or not eligible2:
            return signals

        # Compute hedge ratio if returns data available
        hedge_ratio = None
        if returns1 is not None and returns2 is not None:
            hedge_ratio = self.compute_hedge_ratio(returns1, returns2, method="beta")

        # If hedge ratio available, adjust order notional for each leg
        # For equal dollar amounts: notional1 = notional2 = order_notional
        # For hedge ratio: notional1 = order_notional * hedge_ratio, notional2 = order_notional
        # For simplicity, we use equal dollar amounts (can be enhanced later)
        notional1 = order_notional
        notional2 = order_notional

        # Entry trigger: spread diverged significantly
        if spread_zscore > self.entry_zscore:
            # Spread too wide: short symbol1 (outperformer), long symbol2 (underperformer)
            # Signal 1: Short symbol1
            stop1 = features1.close + (self.stop_atr_mult * features1.atr14)  # Stop above for short
            signal1 = Signal(
                symbol=symbol1,
                asset_class=self.asset_class,
                date=date,
                side=SignalSide.SELL,
                signal_type=SignalType.ENTRY_SHORT,
                trigger_reason=f"pairs_divergence_short_{symbol1}_{symbol2}",
                metadata={
                    "pair": f"{symbol1}_{symbol2}",
                    "leg": "short",
                    "spread_zscore": spread_zscore,
                    "entry_zscore": self.entry_zscore,
                    "exit_zscore": self.exit_zscore,
                    "paired_with": symbol2,
                    "hedge_ratio": hedge_ratio if hedge_ratio is not None else 1.0,
                },
                urgency=0.8,
                entry_price=features1.close,
                stop_price=stop1,
                suggested_entry_price=features1.close,
                suggested_stop_price=stop1,
                atr_mult=self.stop_atr_mult,
                breakout_strength=abs(spread_zscore),
                momentum_strength=0.0,
                diversification_bonus=0.0,
                score=abs(spread_zscore) / self.entry_zscore,
                passed_eligibility=eligible1,
                eligibility_failures=[],
                order_notional=order_notional,
                adv20=features1.adv20 if features1.adv20 else 0.0,
                capacity_passed=True,  # Capacity checked at pair level
            )
            signals.append(signal1)

            # Signal 2: Long symbol2
            stop2 = features2.close - (self.stop_atr_mult * features2.atr14)  # Stop below for long
            signal2 = Signal(
                symbol=symbol2,
                asset_class=self.asset_class,
                date=date,
                side=SignalSide.BUY,
                signal_type=SignalType.ENTRY_LONG,
                trigger_reason=f"pairs_divergence_long_{symbol1}_{symbol2}",
                metadata={
                    "pair": f"{symbol1}_{symbol2}",
                    "leg": "long",
                    "spread_zscore": spread_zscore,
                    "entry_zscore": self.entry_zscore,
                    "exit_zscore": self.exit_zscore,
                    "paired_with": symbol1,
                },
                urgency=0.8,
                entry_price=features2.close,
                stop_price=stop2,
                suggested_entry_price=features2.close,
                suggested_stop_price=stop2,
                atr_mult=self.stop_atr_mult,
                breakout_strength=abs(spread_zscore),
                momentum_strength=0.0,
                diversification_bonus=0.0,
                score=abs(spread_zscore) / self.entry_zscore,
                passed_eligibility=eligible2,
                eligibility_failures=[],
                order_notional=order_notional,
                adv20=features2.adv20 if features2.adv20 else 0.0,
                capacity_passed=True,
            )
            signals.append(signal2)

        elif spread_zscore < -self.entry_zscore:
            # Spread too narrow: long symbol1 (underperformer), short symbol2 (outperformer)
            # Signal 1: Long symbol1
            stop1 = features1.close - (self.stop_atr_mult * features1.atr14)  # Stop below for long
            signal1 = Signal(
                symbol=symbol1,
                asset_class=self.asset_class,
                date=date,
                side=SignalSide.BUY,
                signal_type=SignalType.ENTRY_LONG,
                trigger_reason=f"pairs_divergence_long_{symbol1}_{symbol2}",
                metadata={
                    "pair": f"{symbol1}_{symbol2}",
                    "leg": "long",
                    "spread_zscore": spread_zscore,
                    "entry_zscore": self.entry_zscore,
                    "exit_zscore": self.exit_zscore,
                    "paired_with": symbol2,
                },
                urgency=0.8,
                entry_price=features1.close,
                stop_price=stop1,
                suggested_entry_price=features1.close,
                suggested_stop_price=stop1,
                atr_mult=self.stop_atr_mult,
                breakout_strength=abs(spread_zscore),
                momentum_strength=0.0,
                diversification_bonus=0.0,
                score=abs(spread_zscore) / self.entry_zscore,
                passed_eligibility=eligible1,
                eligibility_failures=[],
                order_notional=order_notional,
                adv20=features1.adv20 if features1.adv20 else 0.0,
                capacity_passed=True,
            )
            signals.append(signal1)

            # Signal 2: Short symbol2
            stop2 = features2.close + (self.stop_atr_mult * features2.atr14)  # Stop above for short
            signal2 = Signal(
                symbol=symbol2,
                asset_class=self.asset_class,
                date=date,
                side=SignalSide.SELL,
                signal_type=SignalType.ENTRY_SHORT,
                trigger_reason=f"pairs_divergence_short_{symbol1}_{symbol2}",
                metadata={
                    "pair": f"{symbol1}_{symbol2}",
                    "leg": "short",
                    "spread_zscore": spread_zscore,
                    "entry_zscore": self.entry_zscore,
                    "exit_zscore": self.exit_zscore,
                    "paired_with": symbol1,
                },
                urgency=0.8,
                entry_price=features2.close,
                stop_price=stop2,
                suggested_entry_price=features2.close,
                suggested_stop_price=stop2,
                atr_mult=self.stop_atr_mult,
                breakout_strength=abs(spread_zscore),
                momentum_strength=0.0,
                diversification_bonus=0.0,
                score=abs(spread_zscore) / self.entry_zscore,
                passed_eligibility=eligible2,
                eligibility_failures=[],
                order_notional=order_notional,
                adv20=features2.adv20 if features2.adv20 else 0.0,
                capacity_passed=True,
            )
            signals.append(signal2)

        return signals

    def generate_signal(
        self,
        symbol: str,
        features: FeatureRow,
        order_notional: float,
        diversification_bonus: float = 0.0,
    ) -> Optional[Signal]:
        """Generate signal for a symbol.

        Note: For pairs trading, this method is not used directly.
        Use generate_pair_signals() instead, which generates signals for both legs.

        Args:
            symbol: Symbol to generate signal for
            features: FeatureRow with indicators
            order_notional: Estimated order notional value
            diversification_bonus: Correlation bonus (not used for pairs)

        Returns:
            None (pairs trading uses generate_pair_signals instead)
        """
        # Pairs trading doesn't generate individual signals
        # Signals are generated at pair level
        return None

    def check_exit_signals(self, position: Position, features: FeatureRow) -> Optional[ExitReason]:
        """Check if position should be exited.

        For pairs trading, exits are determined by:
        1. Spread convergence (checked at pair level via check_pair_exit)
        2. Hard stop (individual leg)
        3. Time stop (max_hold_days)

        Args:
            position: Open position to check
            features: FeatureRow with current indicators

        Returns:
            ExitReason if exit triggered, None otherwise
        """
        # Priority 1: Hard stop (check based on position side)
        if position.side == PositionSide.LONG:
            # Long: exit if price drops to stop
            if features.close <= position.stop_price:
                return ExitReason.HARD_STOP
        else:  # SHORT
            # Short: exit if price rises to stop
            if features.close >= position.stop_price:
                return ExitReason.HARD_STOP

        # Priority 2: Time stop
        hold_days = (features.date - position.entry_date).days
        if hold_days >= self.max_hold_days:
            return ExitReason.MANUAL  # Time stop

        # Spread convergence is checked at pair level (not here)
        return None

    def check_pair_exit(
        self,
        symbol1: str,
        symbol2: str,
        spread_zscore: float,
        position1: Optional[Position],
        position2: Optional[Position],
        date: pd.Timestamp,
    ) -> List[Signal]:
        """Check if pair should be exited due to spread convergence.

        Args:
            symbol1: First symbol in pair
            symbol2: Second symbol in pair
            spread_zscore: Current z-score of spread
            position1: Position for symbol1 (if exists)
            position2: Position for symbol2 (if exists)
            date: Current date

        Returns:
            List of exit signals (0, 1, or 2 signals)
        """
        signals: List[Signal] = []

        # Exit when spread converged (|z-score| < exit_zscore)
        if abs(spread_zscore) < self.exit_zscore:
            # Exit both legs
            if position1 and position1.is_open():
                # Determine exit side based on position direction
                if position1.side == PositionSide.LONG:
                    exit_type = SignalType.EXIT
                    exit_side = SignalSide.SELL  # Sell to close long
                else:  # SHORT
                    exit_type = SignalType.EXIT_SHORT
                    exit_side = SignalSide.BUY  # Buy to cover short

                signal1 = Signal(
                    symbol=symbol1,
                    asset_class=self.asset_class,
                    date=date,
                    side=exit_side,
                    signal_type=exit_type,
                    trigger_reason=f"pairs_convergence_{symbol1}_{symbol2}",
                    metadata={
                        "pair": f"{symbol1}_{symbol2}",
                        "spread_zscore": spread_zscore,
                        "exit_zscore": self.exit_zscore,
                    },
                    urgency=0.9,
                    entry_price=position1.entry_price,  # Use entry price for reference
                    stop_price=position1.stop_price,
                    passed_eligibility=True,
                    eligibility_failures=[],
                    order_notional=0.0,
                    adv20=0.0,
                    capacity_passed=True,
                )
                signals.append(signal1)

            if position2 and position2.is_open():
                # Determine exit side based on position direction
                if position2.side == PositionSide.LONG:
                    exit_type = SignalType.EXIT
                    exit_side = SignalSide.SELL  # Sell to close long
                else:  # SHORT
                    exit_type = SignalType.EXIT_SHORT
                    exit_side = SignalSide.BUY  # Buy to cover short

                signal2 = Signal(
                    symbol=symbol2,
                    asset_class=self.asset_class,
                    date=date,
                    side=exit_side,
                    signal_type=exit_type,
                    trigger_reason=f"pairs_convergence_{symbol1}_{symbol2}",
                    metadata={
                        "pair": f"{symbol1}_{symbol2}",
                        "spread_zscore": spread_zscore,
                        "exit_zscore": self.exit_zscore,
                    },
                    urgency=0.9,
                    entry_price=position2.entry_price,
                    stop_price=position2.stop_price,
                    passed_eligibility=True,
                    eligibility_failures=[],
                    order_notional=0.0,
                    adv20=0.0,
                    capacity_passed=True,
                )
                signals.append(signal2)

        return signals

    def update_stop_price(self, position: Position, features: FeatureRow) -> Optional[float]:
        """Update stop price for position.

        Pairs trading uses fixed ATR-based stops (no trailing).

        Args:
            position: Open position
            features: FeatureRow with current indicators

        Returns:
            New stop price if updated, None if unchanged
        """
        # Pairs trading doesn't update stops (uses fixed ATR stops)
        return None

    def get_required_history_days(self) -> int:
        """Get minimum lookback period needed for indicators.

        Returns:
            Number of days (lookback + buffer)
        """
        return int(self.lookback + 20)  # Buffer for stability

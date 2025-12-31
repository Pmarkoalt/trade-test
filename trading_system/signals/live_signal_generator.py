"""Main orchestrator for live signal generation."""

from datetime import date
from typing import Dict, List, Optional
from uuid import uuid4

import numpy as np
import pandas as pd

from ...configs.strategy_config import StrategyConfig
from ...models.features import FeatureRow
from ...models.signals import Signal
from ...portfolio.portfolio import Portfolio
from ...strategies.base.strategy_interface import StrategyInterface
from .config import SignalConfig
from .generators.technical_signals import TechnicalSignalGenerator
from .rankers.signal_scorer import SignalScorer
from .recommendation import Recommendation


class LiveSignalGenerator:
    """Main orchestrator for live signal generation and recommendation creation."""

    def __init__(
        self,
        strategies: List[StrategyInterface],
        signal_config: SignalConfig,
    ):
        """Initialize live signal generator.

        Args:
            strategies: List of strategy instances
            signal_config: Signal configuration
        """
        self.strategies = strategies
        self.config = signal_config
        self.technical_generator = TechnicalSignalGenerator(strategies)
        self.scorer = SignalScorer(signal_config)

    def generate_recommendations(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],
        current_date: date,
        portfolio_state: Optional[Portfolio] = None,
    ) -> List[Recommendation]:
        """Generate trading recommendations for current date.

        Args:
            ohlcv_data: OHLCV data keyed by symbol
            current_date: The date to generate signals for
            portfolio_state: Current portfolio state (optional)

        Returns:
            List of Recommendation objects, sorted by score descending
        """
        # Step 1: Generate signals
        signals = self.technical_generator.generate_signals(
            ohlcv_data=ohlcv_data,
            current_date=current_date,
            portfolio_state=portfolio_state,
        )

        if not signals:
            return []

        # Step 2: Collect features for scoring
        features_dict = self._collect_features(ohlcv_data, current_date, signals)

        # Step 3: Score signals
        scored_signals = self.scorer.score_signals(signals, features_dict)

        # Step 4: Convert to recommendations
        recommendations = []
        strategy_map = {s.name: s for s in self.strategies}

        for signal, score in scored_signals:
            # Skip if below minimum conviction threshold
            conviction = self._score_to_conviction(score)
            if not self._meets_min_conviction(conviction):
                continue

            feature = features_dict.get(signal.symbol)
            if not feature:
                continue

            # Get strategy config (try to find strategy that generated this signal)
            strategy_config = None
            for strategy in self.strategies:
                if signal.symbol in strategy.universe and strategy.asset_class == signal.asset_class:
                    strategy_config = strategy.config
                    break

            if strategy_config is None:
                # Fallback: use first matching strategy config
                strategy_config = self.strategies[0].config if self.strategies else None

            if strategy_config is None:
                continue

            recommendation = self._create_recommendation(signal, score, feature, strategy_config)
            recommendations.append(recommendation)

            # Limit to max_recommendations
            if len(recommendations) >= self.config.max_recommendations:
                break

        return recommendations

    def _collect_features(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],
        current_date: date,
        signals: List[Signal],
    ) -> Dict[str, FeatureRow]:
        """Collect FeatureRow objects for all signals.

        Args:
            ohlcv_data: OHLCV data keyed by symbol
            current_date: Current date
            signals: List of signals

        Returns:
            Dictionary mapping symbol to FeatureRow
        """
        from ...indicators.feature_computer import compute_features, compute_features_for_date

        features_dict = {}

        # Convert date to pd.Timestamp if needed
        if isinstance(current_date, date):
            current_date = pd.Timestamp(current_date)

        for signal in signals:
            if signal.symbol in features_dict:
                continue  # Already collected

            if signal.symbol not in ohlcv_data:
                continue

            data = ohlcv_data[signal.symbol]
            asset_class = signal.asset_class

            try:
                features_df = compute_features(
                    data,
                    symbol=signal.symbol,
                    asset_class=asset_class,
                    use_cache=False,
                    optimize_memory=False,
                )
                feature_row = compute_features_for_date(features_df, current_date)
                if feature_row:
                    features_dict[signal.symbol] = feature_row
            except Exception:
                # Skip symbols with computation errors
                continue

        return features_dict

    def _create_recommendation(
        self,
        signal: Signal,
        score: float,
        feature: FeatureRow,
        config: StrategyConfig,
    ) -> Recommendation:
        """Convert signal to recommendation.

        Args:
            signal: Signal object
            score: Signal score (0-10 scale)
            feature: FeatureRow with indicators
            config: Strategy configuration

        Returns:
            Recommendation object
        """
        # Calculate prices
        entry_price = feature.close  # Will execute at next open
        atr_mult = config.exit.hard_stop_atr_mult
        stop_price = entry_price - (atr_mult * feature.atr14) if feature.atr14 else entry_price * 0.95
        # Target price: 2:1 risk/reward ratio
        stop_distance = entry_price - stop_price
        target_price = entry_price + (2 * stop_distance)

        # Calculate position size
        risk_amount = config.risk.risk_per_trade  # Risk per trade from config
        if stop_distance > 0:
            position_size_pct = risk_amount / (stop_distance / entry_price)
        else:
            position_size_pct = 0.0
        # Cap at max position size
        position_size_pct = min(position_size_pct, config.risk.max_position_notional)

        # Determine conviction
        conviction = self._score_to_conviction(score)

        # Get signal type string
        if signal.triggered_on:
            signal_type = f"breakout_{signal.triggered_on.value.lower()}"
        else:
            signal_type = signal.trigger_reason.replace(" ", "_").lower()

        # Get strategy name (try from strategy that generated signal)
        strategy_name = None
        for strategy in self.strategies:
            if signal.symbol in strategy.universe and strategy.asset_class == signal.asset_class:
                strategy_name = strategy.name
                break

        return Recommendation(
            id=str(uuid4()),
            symbol=signal.symbol,
            asset_class=signal.asset_class,
            direction="BUY",  # System is long-only
            conviction=conviction,
            current_price=feature.close,
            entry_price=entry_price,
            target_price=target_price,
            stop_price=stop_price,
            position_size_pct=position_size_pct,
            risk_pct=risk_amount,
            technical_score=score,
            combined_score=score,  # Phase 1: technical only
            signal_type=signal_type,
            reasoning=self._generate_reasoning(signal, feature, score),
            strategy_name=strategy_name,
        )

    def _score_to_conviction(self, score: float) -> str:
        """Convert score to conviction level.

        Args:
            score: Signal score (0-10 scale)

        Returns:
            "HIGH", "MEDIUM", or "LOW"
        """
        if score >= 8:
            return "HIGH"
        elif score >= 6:
            return "MEDIUM"
        else:
            return "LOW"

    def _meets_min_conviction(self, conviction: str) -> bool:
        """Check if conviction meets minimum threshold.

        Args:
            conviction: Conviction level ("HIGH", "MEDIUM", "LOW")

        Returns:
            True if meets minimum threshold
        """
        conviction_levels = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
        min_level = conviction_levels.get(self.config.min_conviction, 2)
        current_level = conviction_levels.get(conviction, 0)
        return current_level >= min_level

    def _generate_reasoning(self, signal: Signal, feature: FeatureRow, score: float) -> str:
        """Generate human-readable reasoning for recommendation.

        Args:
            signal: Signal object
            feature: FeatureRow with indicators
            score: Signal score

        Returns:
            Reasoning string
        """
        parts = []

        # Breakout information
        if signal.triggered_on:
            parts.append(f"{signal.triggered_on.value} breakout detected")
            if signal.breakout_clearance:
                parts.append(f"with {signal.breakout_clearance:.2%} clearance")

        # Momentum
        if feature.roc60 is not None and not np.isnan(feature.roc60):
            parts.append(f"ROC60: {feature.roc60:.2%}")

        # Score
        parts.append(f"Technical score: {score:.1f}/10")

        # Trend
        if feature.ma50 is not None:
            if feature.close > feature.ma50:
                parts.append("Above MA50 (uptrend)")
            else:
                parts.append("Below MA50")

        return ". ".join(parts) if parts else "Signal generated based on technical indicators"

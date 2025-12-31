"""Main orchestrator for live signal generation."""

import logging
from datetime import date
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd

from ...configs.strategy_config import StrategyConfig
from ...models.features import FeatureRow
from ...models.signals import Signal
from ...portfolio.portfolio import Portfolio
from ...strategies.base.strategy_interface import StrategyInterface
from ...tracking.signal_tracker import SignalTracker
from ...tracking.storage.sqlite_store import SQLiteTrackingStore
from .config import SignalConfig
from .generators.technical_signals import TechnicalSignalGenerator
from .rankers.signal_scorer import SignalScorer
from .recommendation import Recommendation

logger = logging.getLogger(__name__)


class LiveSignalGenerator:
    """Main orchestrator for live signal generation and recommendation creation."""

    def __init__(
        self,
        strategies: List[StrategyInterface],
        signal_config: SignalConfig,
        news_analyzer: Optional[object] = None,  # NewsAnalyzer type
        tracking_db: Optional[str] = None,
    ):
        """Initialize live signal generator.

        Args:
            strategies: List of strategy instances
            signal_config: Signal configuration
            news_analyzer: Optional NewsAnalyzer instance for news integration
            tracking_db: Optional path to tracking database for performance tracking
        """
        self.strategies = strategies
        self.config = signal_config
        self.technical_generator = TechnicalSignalGenerator(strategies)
        self.scorer = SignalScorer(signal_config)
        self.news_analyzer = news_analyzer

        # Initialize tracking if database provided
        self.tracker = None
        if tracking_db:
            try:
                store = SQLiteTrackingStore(tracking_db)
                store.initialize()
                self.tracker = SignalTracker(store)
                logger.info(f"Tracking initialized with database: {tracking_db}")
            except Exception as e:
                logger.warning(f"Failed to initialize tracking: {e}. Continuing without tracking.")
                self.tracker = None

    async def generate_recommendations(
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

        # Step 3: Fetch news analysis if enabled
        news_analysis = None
        if self.news_analyzer and self.config.news_enabled:
            symbols = list(ohlcv_data.keys())
            try:
                news_analysis = await self.news_analyzer.analyze_symbols(
                    symbols=symbols,
                    lookback_hours=self.config.news_lookback_hours
                )
            except Exception:
                # If news analysis fails, continue without it
                news_analysis = None

        # Step 4: Score signals (with news integration)
        scored_signals = self._score_signals(signals, features_dict, news_analysis)

        # Step 5: Convert to recommendations
        recommendations = []
        strategy_map = {s.name: s for s in self.strategies}

        for signal, combined_score, metadata in scored_signals:
            # Skip if below minimum conviction threshold
            conviction = self._score_to_conviction(combined_score)
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

            recommendation = self._create_recommendation(
                signal, combined_score, feature, strategy_config, metadata
            )
            recommendations.append(recommendation)

            # Limit to max_recommendations
            if len(recommendations) >= self.config.max_recommendations:
                break

        # Track generated signals
        if self.tracker:
            for recommendation in recommendations:
                try:
                    signal_id = self.tracker.record_from_recommendation(recommendation)
                    # Store tracking_id on recommendation for later use
                    if not hasattr(recommendation, 'tracking_id'):
                        recommendation.tracking_id = signal_id
                    else:
                        recommendation.tracking_id = signal_id
                    logger.info(f"Tracked signal {signal_id} for {recommendation.symbol}")
                except Exception as e:
                    logger.warning(f"Failed to track signal for {recommendation.symbol}: {e}")

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

    def _score_signals(
        self,
        signals: List[Signal],
        features_dict: Dict[str, FeatureRow],
        news_analysis: Optional[object] = None,  # NewsAnalysisResult type
    ) -> List[Tuple[Signal, float, Dict]]:
        """Score signals with technical and news components.

        Args:
            signals: List of signals to score
            features_dict: Dictionary mapping symbol to FeatureRow
            news_analysis: Optional news analysis result

        Returns:
            List of (signal, combined_score, metadata) tuples sorted by score descending
        """
        scored = []

        # First get technical scores
        technical_scored = self.scorer.score_signals(signals, features_dict)
        technical_scores = {sig.symbol: score for sig, score in technical_scored}

        for signal in signals:
            # Technical score
            technical_score = technical_scores.get(signal.symbol, 0.0)

            # News score and headlines
            news_score = 5.0  # Default neutral
            news_reasoning = "News analysis disabled" if not self.config.news_enabled else "No news data"
            news_headlines = []
            news_sentiment = None

            if news_analysis and self.news_analyzer:
                try:
                    news_score, news_reasoning = self.news_analyzer.get_news_score_for_signal(
                        signal.symbol,
                        news_analysis
                    )
                    # Extract headlines from news_analysis
                    if hasattr(news_analysis, 'symbol_summaries'):
                        symbol_summary = news_analysis.symbol_summaries.get(signal.symbol)
                        if symbol_summary and hasattr(symbol_summary, 'top_headlines'):
                            news_headlines = symbol_summary.top_headlines or []
                        # Extract sentiment label
                        if symbol_summary and hasattr(symbol_summary, 'sentiment_label'):
                            sentiment_label = symbol_summary.sentiment_label
                            if hasattr(sentiment_label, 'value'):
                                label_value = sentiment_label.value.lower()
                                if 'positive' in label_value:
                                    news_sentiment = "positive"
                                elif 'negative' in label_value:
                                    news_sentiment = "negative"
                                else:
                                    news_sentiment = "neutral"
                except Exception:
                    news_score = 5.0  # Neutral
                    news_reasoning = "News analysis error"

            # Combined score
            combined = (
                technical_score * self.config.technical_weight +
                news_score * self.config.news_weight
            )

            metadata = {
                "technical_score": technical_score,
                "news_score": news_score,
                "news_reasoning": news_reasoning,
                "news_headlines": news_headlines,
                "news_sentiment": news_sentiment,
                "combined_score": combined
            }

            scored.append((signal, combined, metadata))

        # Sort by combined score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _create_recommendation(
        self,
        signal: Signal,
        combined_score: float,
        feature: FeatureRow,
        config: StrategyConfig,
        metadata: Optional[Dict] = None,
    ) -> Recommendation:
        """Convert signal to recommendation.

        Args:
            signal: Signal object
            combined_score: Combined technical + news score (0-10 scale)
            feature: FeatureRow with indicators
            config: Strategy configuration
            metadata: Optional metadata dict with technical_score, news_score, news_reasoning

        Returns:
            Recommendation object
        """
        metadata = metadata or {}
        technical_score = metadata.get("technical_score", combined_score)
        news_score = metadata.get("news_score")
        news_reasoning = metadata.get("news_reasoning")

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

        # Determine conviction (can be boosted/penalized by news)
        base_conviction = self._score_to_conviction(combined_score)
        conviction = self._adjust_conviction_for_news(base_conviction, news_score)

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

        # Extract news sentiment and headlines from metadata
        news_sentiment = metadata.get("news_sentiment")
        news_headlines = metadata.get("news_headlines", [])
        
        # Fallback: derive sentiment from score if not in metadata
        if news_sentiment is None and news_score is not None:
            if news_score >= 7.0:
                news_sentiment = "positive"
            elif news_score <= 3.0:
                news_sentiment = "negative"
            else:
                news_sentiment = "neutral"

        # Generate reasoning with news included
        reasoning = self._generate_reasoning(signal, feature, technical_score, news_score, news_reasoning)

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
            technical_score=technical_score,
            combined_score=combined_score,
            news_score=news_score,
            news_reasoning=news_reasoning,
            news_sentiment=news_sentiment,
            news_headlines=news_headlines,
            signal_type=signal_type,
            reasoning=reasoning,
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

    def _adjust_conviction_for_news(self, base_conviction: str, news_score: Optional[float]) -> str:
        """Adjust conviction based on news score.

        Args:
            base_conviction: Base conviction from technical score
            news_score: News score (0-10) or None

        Returns:
            Adjusted conviction level
        """
        if news_score is None:
            return base_conviction

        conviction_levels = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
        current_level = conviction_levels.get(base_conviction, 2)

        # Boost if news is very positive
        if news_score >= self.config.min_news_score_for_boost:
            current_level = min(current_level + 1, 3)
        # Penalize if news is very negative
        elif news_score <= self.config.max_news_score_for_penalty:
            current_level = max(current_level - 1, 1)

        # Convert back to string
        for level_name, level_value in conviction_levels.items():
            if level_value == current_level:
                return level_name

        return base_conviction

    def _generate_reasoning(
        self,
        signal: Signal,
        feature: FeatureRow,
        technical_score: float,
        news_score: Optional[float] = None,
        news_reasoning: Optional[str] = None,
    ) -> str:
        """Generate human-readable reasoning for recommendation.

        Args:
            signal: Signal object
            feature: FeatureRow with indicators
            technical_score: Technical score
            news_score: Optional news score
            news_reasoning: Optional news reasoning

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

        # Technical score
        parts.append(f"Technical score: {technical_score:.1f}/10")

        # News information
        if news_score is not None and news_reasoning:
            parts.append(f"News score: {news_score:.1f}/10 ({news_reasoning})")

        # Trend
        if feature.ma50 is not None:
            if feature.close > feature.ma50:
                parts.append("Above MA50 (uptrend)")
            else:
                parts.append("Below MA50")

        return ". ".join(parts) if parts else "Signal generated based on technical indicators"

    def mark_signals_delivered(
        self,
        recommendations: List[Recommendation],
        method: str = "email",
    ):
        """Mark signals as delivered after sending.

        Args:
            recommendations: List of recommendations that were delivered
            method: Delivery method ("email", "sms", "push")
        """
        if not self.tracker:
            return

        for rec in recommendations:
            if hasattr(rec, "tracking_id") and rec.tracking_id:
                try:
                    self.tracker.mark_delivered(rec.tracking_id, method)
                    logger.debug(f"Marked signal {rec.tracking_id} as delivered via {method}")
                except Exception as e:
                    logger.warning(f"Failed to mark signal {rec.tracking_id} as delivered: {e}")

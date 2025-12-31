"""ML-enhanced signal scoring."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from loguru import logger

from trading_system.ml_refinement.config import MLConfig
from trading_system.ml_refinement.integration.prediction_service import PredictionService


@dataclass
class EnhancedSignalScore:
    """Signal score with ML enhancement."""

    signal_id: str
    symbol: str

    # Original scores
    technical_score: float = 0.0
    news_score: float = 0.0

    # ML scores
    ml_quality_score: float = 0.5

    # Combined
    combined_score: float = 0.0

    # Metadata
    ml_enabled: bool = False
    ml_confidence: str = ""  # "high", "medium", "low"


class MLSignalScorer:
    """
    Enhance signal scores with ML predictions.

    Example:
        scorer = MLSignalScorer(config, prediction_service)

        enhanced = scorer.score_signal(
            signal_id="sig-123",
            technical_score=7.5,
            news_score=6.0,
            ohlcv_data=ohlcv_df,
            signal_metadata=signal_dict,
        )

        print(f"Combined score: {enhanced.combined_score:.1f}")
    """

    def __init__(
        self,
        config: MLConfig,
        prediction_service: PredictionService,
    ):
        """
        Initialize scorer.

        Args:
            config: ML configuration.
            prediction_service: Prediction service.
        """
        self.config = config
        self.prediction_service = prediction_service

    def score_signal(
        self,
        signal_id: str,
        technical_score: float,
        news_score: Optional[float],
        ohlcv_data,
        signal_metadata: Dict,
        benchmark_data=None,
    ) -> EnhancedSignalScore:
        """
        Score a signal with ML enhancement.

        Args:
            signal_id: Signal identifier.
            technical_score: Technical analysis score (0-10).
            news_score: News sentiment score (0-10), optional.
            ohlcv_data: OHLCV DataFrame.
            signal_metadata: Signal metadata.
            benchmark_data: Optional benchmark data.

        Returns:
            EnhancedSignalScore with combined score.
        """
        result = EnhancedSignalScore(
            signal_id=signal_id,
            symbol=signal_metadata.get("symbol", ""),
            technical_score=technical_score,
            news_score=news_score or 0.0,
        )

        # Get ML prediction if enabled
        if self.config.enabled and self.config.use_ml_scores:
            ml_quality = self.prediction_service.predict_signal_quality(
                signal_id=signal_id,
                ohlcv_data=ohlcv_data,
                signal_metadata=signal_metadata,
                benchmark_data=benchmark_data,
            )
            result.ml_quality_score = ml_quality
            result.ml_enabled = True

            # Determine confidence level
            if ml_quality >= self.config.quality_threshold_high:
                result.ml_confidence = "high"
            elif ml_quality <= self.config.quality_threshold_low:
                result.ml_confidence = "low"
            else:
                result.ml_confidence = "medium"

        # Calculate combined score
        result.combined_score = self._calculate_combined_score(
            technical_score=technical_score,
            news_score=news_score,
            ml_quality=result.ml_quality_score if result.ml_enabled else None,
        )

        return result

    def score_signals_batch(
        self,
        signals: List[Dict],
        ohlcv_dict: Dict,
        benchmark_data=None,
    ) -> List[EnhancedSignalScore]:
        """Score multiple signals."""
        results = []

        for signal in signals:
            signal_id = signal.get("signal_id", signal.get("id", ""))
            symbol = signal.get("symbol", "")

            if symbol not in ohlcv_dict:
                continue

            enhanced = self.score_signal(
                signal_id=signal_id,
                technical_score=signal.get("technical_score", 5.0),
                news_score=signal.get("news_score"),
                ohlcv_data=ohlcv_dict[symbol],
                signal_metadata=signal,
                benchmark_data=benchmark_data,
            )
            results.append(enhanced)

        return results

    def _calculate_combined_score(
        self,
        technical_score: float,
        news_score: Optional[float],
        ml_quality: Optional[float],
    ) -> float:
        """
        Calculate combined score from components.

        Uses configurable weights to combine scores.
        """
        # Normalize to 0-10 scale
        normalized_technical = technical_score  # Already 0-10
        normalized_news = news_score if news_score else 5.0  # Default neutral

        # ML quality is 0-1, convert to 0-10
        normalized_ml = (ml_quality * 10) if ml_quality else 5.0

        # Get weights based on config and availability
        if ml_quality is not None and self.config.use_ml_scores:
            # Use ML-weighted combination
            tech_weight = 0.4
            news_weight = 0.2 if news_score else 0.0
            ml_weight = self.config.ml_score_weight  # Default 0.3

            # Redistribute news weight if not available
            if news_score is None:
                tech_weight += 0.1
                ml_weight += 0.1

            total_weight = tech_weight + news_weight + ml_weight

            combined = (
                normalized_technical * tech_weight + normalized_news * news_weight + normalized_ml * ml_weight
            ) / total_weight

        else:
            # Without ML
            if news_score is not None:
                combined = normalized_technical * 0.6 + normalized_news * 0.4
            else:
                combined = normalized_technical

        return combined

    def filter_by_ml_quality(
        self,
        scored_signals: List[EnhancedSignalScore],
        min_quality: Optional[float] = None,
    ) -> List[EnhancedSignalScore]:
        """
        Filter signals by ML quality score.

        Args:
            scored_signals: List of scored signals.
            min_quality: Minimum ML quality (default: use config).

        Returns:
            Filtered list of signals.
        """
        min_quality = min_quality or self.config.quality_threshold_low

        filtered = []
        for signal in scored_signals:
            if not signal.ml_enabled:
                # Include signals without ML (no filtering)
                filtered.append(signal)
            elif signal.ml_quality_score >= min_quality:
                filtered.append(signal)
            else:
                logger.debug(f"Filtered out {signal.symbol}: ML quality " f"{signal.ml_quality_score:.2f} < {min_quality}")

        return filtered

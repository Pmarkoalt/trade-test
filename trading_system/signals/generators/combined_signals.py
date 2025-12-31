"""Combined signal generator that merges technical, news, and sentiment signals."""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from ...models.signals import Signal
from ...portfolio.portfolio import Portfolio
from .technical_signals import TechnicalSignalGenerator


@dataclass
class CombinedSignalConfig:
    """Configuration for combined signal generation."""

    # Weight for technical signals (0-1)
    technical_weight: float = 0.6

    # Weight for sentiment signals (0-1)
    sentiment_weight: float = 0.25

    # Weight for news relevance (0-1)
    news_weight: float = 0.15

    # Minimum combined score to generate a signal
    min_combined_score: float = 0.5

    # Whether to require positive sentiment for buy signals
    require_aligned_sentiment: bool = False

    # Sentiment threshold for alignment check
    sentiment_alignment_threshold: float = 0.1

    # Whether to boost signals with strong news relevance
    boost_high_relevance: bool = True

    # Relevance threshold for boosting
    relevance_boost_threshold: float = 0.7

    # Boost multiplier for high-relevance signals
    relevance_boost_multiplier: float = 1.1

    def __post_init__(self):
        """Validate weights sum to 1."""
        total = self.technical_weight + self.sentiment_weight + self.news_weight
        if not (0.99 <= total <= 1.01):
            # Normalize weights
            self.technical_weight /= total
            self.sentiment_weight /= total
            self.news_weight /= total


@dataclass
class SignalContext:
    """Context information for a signal including sentiment and news data."""

    symbol: str
    signal: Signal
    sentiment_score: float = 0.0
    sentiment_confidence: float = 0.0
    news_relevance: float = 0.0
    news_count: int = 0
    combined_score: float = 0.0


class CombinedSignalGenerator:
    """Generate signals by combining technical, news, and sentiment inputs."""

    def __init__(
        self,
        technical_generator: TechnicalSignalGenerator,
        config: Optional[CombinedSignalConfig] = None,
    ):
        """Initialize combined signal generator.

        Args:
            technical_generator: Technical signal generator instance
            config: Configuration for signal combination
        """
        self.technical_generator = technical_generator
        self.config = config or CombinedSignalConfig()

    def generate_signals(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],
        current_date: date,
        sentiment_data: Optional[Dict[str, Tuple[float, float]]] = None,
        news_relevance: Optional[Dict[str, Tuple[float, int]]] = None,
        portfolio_state: Optional[Portfolio] = None,
    ) -> List[SignalContext]:
        """Generate combined signals for current date.

        Args:
            ohlcv_data: OHLCV data keyed by symbol
            current_date: The date to generate signals for
            sentiment_data: Optional dict of {symbol: (sentiment_score, confidence)}
            news_relevance: Optional dict of {symbol: (relevance_score, news_count)}
            portfolio_state: Current portfolio state

        Returns:
            List of SignalContext objects with combined scores
        """
        # Generate technical signals
        technical_signals = self.technical_generator.generate_signals(
            ohlcv_data=ohlcv_data,
            current_date=current_date,
            portfolio_state=portfolio_state,
        )

        # Create contexts for each signal
        contexts = []
        for signal in technical_signals:
            context = SignalContext(
                symbol=signal.symbol,
                signal=signal,
            )

            # Add sentiment data if available
            if sentiment_data and signal.symbol in sentiment_data:
                context.sentiment_score, context.sentiment_confidence = sentiment_data[signal.symbol]

            # Add news relevance if available
            if news_relevance and signal.symbol in news_relevance:
                context.news_relevance, context.news_count = news_relevance[signal.symbol]

            # Calculate combined score
            context.combined_score = self._calculate_combined_score(context)

            contexts.append(context)

        # Filter and sort by combined score
        filtered_contexts = self._filter_signals(contexts)
        filtered_contexts.sort(key=lambda x: x.combined_score, reverse=True)

        return filtered_contexts

    def _calculate_combined_score(self, context: SignalContext) -> float:
        """Calculate combined score for a signal.

        Args:
            context: Signal context with all data

        Returns:
            Combined score between 0 and 1
        """
        config = self.config

        # Technical score (normalized from signal's score)
        # Assuming signal score is 0-10, normalize to 0-1
        signal_score = context.signal.score if context.signal.score is not None else 0.0
        technical_score = min(signal_score / 10.0, 1.0)

        # Sentiment score (already -1 to 1, shift to 0-1)
        # For buy signals, positive sentiment is good
        # For sell signals, negative sentiment is good
        signal_direction = context.signal.side.value if context.signal.side else "BUY"
        if signal_direction == "BUY":
            sentiment_normalized = (context.sentiment_score + 1) / 2
        else:
            sentiment_normalized = (1 - context.sentiment_score) / 2

        # Weight by confidence
        sentiment_score = sentiment_normalized * context.sentiment_confidence

        # News relevance score (already 0-1)
        news_score = context.news_relevance

        # Calculate weighted average
        combined = (
            config.technical_weight * technical_score
            + config.sentiment_weight * sentiment_score
            + config.news_weight * news_score
        )

        # Apply relevance boost if configured
        if config.boost_high_relevance and context.news_relevance >= config.relevance_boost_threshold:
            combined *= config.relevance_boost_multiplier

        return float(min(combined, 1.0))

    def _filter_signals(self, contexts: List[SignalContext]) -> List[SignalContext]:
        """Filter signals based on configuration.

        Args:
            contexts: List of signal contexts

        Returns:
            Filtered list of signal contexts
        """
        filtered = []
        config = self.config

        for context in contexts:
            # Check minimum score
            if context.combined_score < config.min_combined_score:
                continue

            # Check sentiment alignment if required
            if config.require_aligned_sentiment:
                signal_direction = context.signal.side.value if context.signal.side else "BUY"
                if signal_direction == "BUY":
                    if context.sentiment_score < config.sentiment_alignment_threshold:
                        continue
                else:
                    if context.sentiment_score > -config.sentiment_alignment_threshold:
                        continue

            filtered.append(context)

        return filtered

    def rank_signals(
        self,
        contexts: List[SignalContext],
        max_signals: int = 10,
    ) -> List[SignalContext]:
        """Rank and limit signals.

        Args:
            contexts: List of signal contexts
            max_signals: Maximum number of signals to return

        Returns:
            Top ranked signals
        """
        # Sort by combined score descending
        sorted_contexts = sorted(contexts, key=lambda x: x.combined_score, reverse=True)
        return sorted_contexts[:max_signals]

    def get_signal_summary(self, context: SignalContext) -> Dict:
        """Get a summary of a signal context.

        Args:
            context: Signal context

        Returns:
            Dictionary with signal summary
        """
        signal_direction = context.signal.side.value if context.signal.side else "BUY"
        signal_score = context.signal.score if context.signal.score is not None else 0.0
        target_price = context.signal.metadata.get("target_price") if context.signal.metadata else None
        return {
            "symbol": context.symbol,
            "direction": signal_direction,
            "combined_score": round(context.combined_score, 3),
            "technical_score": round(signal_score, 2),
            "sentiment_score": round(context.sentiment_score, 3),
            "sentiment_confidence": round(context.sentiment_confidence, 3),
            "news_relevance": round(context.news_relevance, 3),
            "news_count": context.news_count,
            "entry_price": context.signal.entry_price,
            "target_price": target_price,
            "stop_price": context.signal.stop_price,
        }

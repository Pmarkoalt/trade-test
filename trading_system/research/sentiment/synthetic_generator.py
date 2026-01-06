"""Synthetic sentiment generator for backtesting.

Generates realistic sentiment data for historical backtesting by simulating
news sentiment based on price movements, market events, and regime detection.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from trading_system.data_pipeline.sources.news.models import SentimentLabel

logger = logging.getLogger(__name__)


class SentimentMode(str, Enum):
    """Sentiment simulation modes."""

    RANDOM_WALK = "random_walk"
    PRICE_CORRELATED = "price_correlated"
    EVENT_BASED = "event_based"
    REGIME_BASED = "regime_based"
    COMBINED = "combined"  # Mix of all modes


@dataclass
class SentimentConfig:
    """Configuration for synthetic sentiment generation."""

    mode: SentimentMode = SentimentMode.PRICE_CORRELATED
    correlation_lag: int = 1  # Days sentiment lags price
    noise_std: float = 0.2  # Noise standard deviation
    event_probability: float = 0.05  # Daily probability of sentiment shock
    event_magnitude: float = 0.4  # Magnitude of event shocks
    regime_bias: float = 0.2  # Sentiment bias in bull/bear regimes
    regime_ma_short: int = 50  # Short MA for regime detection
    regime_ma_long: int = 200  # Long MA for regime detection
    mean_reversion_strength: float = 0.1  # Pull towards neutral
    event_calendar_path: Optional[str] = None  # Path to event calendar CSV
    seed: int = 42


@dataclass
class SyntheticSentimentRecord:
    """A single synthetic sentiment record."""

    date: pd.Timestamp
    symbol: str
    sentiment_score: float  # -1.0 to 1.0
    sentiment_label: SentimentLabel
    confidence: float  # 0.0 to 1.0
    article_count: int  # Simulated article count
    event_type: Optional[str] = None  # 'earnings', 'upgrade', 'downgrade', etc.


class SyntheticSentimentGenerator:
    """Generate synthetic sentiment data for backtesting.

    Supports multiple simulation modes:
    - RANDOM_WALK: Brownian motion around neutral
    - PRICE_CORRELATED: Sentiment follows price movements (lagged)
    - EVENT_BASED: Major events inject sentiment shocks
    - REGIME_BASED: Bull/bear market sentiment regimes
    - COMBINED: Mix of all modes
    """

    def __init__(self, config: Optional[SentimentConfig] = None):
        """Initialize generator.

        Args:
            config: Sentiment generation configuration
        """
        self.config = config or SentimentConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self._event_calendar: Optional[pd.DataFrame] = None

        # Load event calendar if provided
        if self.config.event_calendar_path:
            self._load_event_calendar(self.config.event_calendar_path)

    def _load_event_calendar(self, path: str) -> None:
        """Load event calendar from CSV.

        Expected columns: date, event_type, symbols, sentiment_impact, description
        """
        try:
            calendar_path = Path(path)
            if calendar_path.exists():
                self._event_calendar = pd.read_csv(calendar_path, parse_dates=["date"])
                logger.info(f"Loaded event calendar with {len(self._event_calendar)} events")
            else:
                logger.warning(f"Event calendar not found at {path}")
        except Exception as e:
            logger.error(f"Failed to load event calendar: {e}")

    def generate_for_symbol(
        self,
        symbol: str,
        dates: List[pd.Timestamp],
        price_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate synthetic sentiment for a single symbol.

        Args:
            symbol: Stock symbol
            dates: List of dates to generate sentiment for
            price_data: DataFrame with OHLCV data (must have 'close' column)

        Returns:
            DataFrame with columns:
            - date
            - symbol
            - sentiment_score (-1.0 to 1.0)
            - sentiment_label
            - confidence (0.0 to 1.0)
            - article_count
            - event_type
        """
        if len(dates) == 0:
            return self._empty_dataframe()

        # Ensure dates are sorted
        dates = sorted(dates)

        # Generate base sentiment based on mode
        if self.config.mode == SentimentMode.RANDOM_WALK:
            sentiment_scores = self._generate_random_walk(len(dates))
        elif self.config.mode == SentimentMode.PRICE_CORRELATED:
            sentiment_scores = self._generate_price_correlated(dates, price_data)
        elif self.config.mode == SentimentMode.EVENT_BASED:
            sentiment_scores = self._generate_event_based(dates, symbol)
        elif self.config.mode == SentimentMode.REGIME_BASED:
            sentiment_scores = self._generate_regime_based(dates, price_data)
        elif self.config.mode == SentimentMode.COMBINED:
            sentiment_scores = self._generate_combined(dates, symbol, price_data)
        else:
            sentiment_scores = self._generate_random_walk(len(dates))

        # Add noise
        noise = self.rng.normal(0, self.config.noise_std, len(dates))
        sentiment_scores = sentiment_scores + noise

        # Clip to valid range
        sentiment_scores = np.clip(sentiment_scores, -1.0, 1.0)

        # Generate confidence scores (higher for extreme sentiment)
        confidence_scores = self._generate_confidence(sentiment_scores)

        # Generate article counts (more articles for extreme sentiment)
        article_counts = self._generate_article_counts(sentiment_scores)

        # Get event types from calendar
        event_types = self._get_event_types(dates, symbol)

        # Build records
        records = []
        for i, date in enumerate(dates):
            label = self._score_to_label(sentiment_scores[i])
            records.append(
                SyntheticSentimentRecord(
                    date=date,
                    symbol=symbol,
                    sentiment_score=float(sentiment_scores[i]),
                    sentiment_label=label,
                    confidence=float(confidence_scores[i]),
                    article_count=int(article_counts[i]),
                    event_type=event_types[i],
                )
            )

        # Convert to DataFrame
        df = pd.DataFrame(
            [
                {
                    "date": r.date,
                    "symbol": r.symbol,
                    "sentiment_score": r.sentiment_score,
                    "sentiment_label": r.sentiment_label.value,
                    "confidence": r.confidence,
                    "article_count": r.article_count,
                    "event_type": r.event_type,
                }
                for r in records
            ]
        )

        return df

    def generate_for_symbols(
        self,
        symbols: List[str],
        dates: List[pd.Timestamp],
        price_data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Generate synthetic sentiment for multiple symbols.

        Args:
            symbols: List of stock symbols
            dates: List of dates to generate sentiment for
            price_data: Dictionary mapping symbol to OHLCV DataFrame

        Returns:
            Combined DataFrame with sentiment for all symbols
        """
        all_dfs = []
        for symbol in symbols:
            if symbol in price_data:
                df = self.generate_for_symbol(symbol, dates, price_data[symbol])
                all_dfs.append(df)
            else:
                logger.warning(f"No price data for {symbol}, skipping")

        if not all_dfs:
            return self._empty_dataframe()

        return pd.concat(all_dfs, ignore_index=True)

    def _generate_random_walk(self, n: int) -> np.ndarray:
        """Generate random walk sentiment.

        Args:
            n: Number of data points

        Returns:
            Array of sentiment scores
        """
        # Start at neutral
        scores = np.zeros(n)
        for i in range(1, n):
            # Random step with mean reversion
            step = self.rng.normal(0, 0.1)
            reversion = -self.config.mean_reversion_strength * scores[i - 1]
            scores[i] = scores[i - 1] + step + reversion

        return np.clip(scores, -1.0, 1.0)

    def _generate_price_correlated(
        self,
        dates: List[pd.Timestamp],
        price_data: pd.DataFrame,
    ) -> np.ndarray:
        """Generate price-correlated sentiment.

        Sentiment follows price changes with a lag.

        Args:
            dates: Dates to generate for
            price_data: OHLCV DataFrame

        Returns:
            Array of sentiment scores
        """
        n = len(dates)
        scores = np.zeros(n)

        # Get close prices aligned with dates
        if "close" not in price_data.columns:
            logger.warning("No 'close' column in price data, using random walk")
            return self._generate_random_walk(n)

        # Calculate returns
        close_prices = price_data["close"]
        returns = close_prices.pct_change()

        for i, date in enumerate(dates):
            # Look back by correlation_lag days
            lag = self.config.correlation_lag
            lookback_date = date - pd.Timedelta(days=lag)

            # Get return for lookback date (or closest available)
            try:
                if lookback_date in returns.index:
                    ret = returns.loc[lookback_date]
                else:
                    # Find closest date
                    closest_idx = returns.index.get_indexer([lookback_date], method="nearest")[0]
                    if closest_idx >= 0 and closest_idx < len(returns):
                        ret = returns.iloc[closest_idx]
                    else:
                        ret = 0.0
            except Exception:
                ret = 0.0

            # Convert return to sentiment (-5% to +5% maps to -1 to +1)
            if pd.notna(ret):
                sentiment = np.clip(ret * 20, -1.0, 1.0)  # 5% return = full sentiment
            else:
                sentiment = 0.0

            # Add persistence
            if i > 0:
                sentiment = 0.7 * sentiment + 0.3 * scores[i - 1]

            scores[i] = sentiment

        return scores

    def _generate_event_based(
        self,
        dates: List[pd.Timestamp],
        symbol: str,
    ) -> np.ndarray:
        """Generate event-based sentiment with random shocks.

        Args:
            dates: Dates to generate for
            symbol: Stock symbol

        Returns:
            Array of sentiment scores
        """
        n = len(dates)
        scores = np.zeros(n)

        for i, date in enumerate(dates):
            # Check event calendar first
            event_sentiment = self._get_calendar_event_sentiment(date, symbol)

            if event_sentiment is not None:
                scores[i] = event_sentiment
            elif self.rng.random() < self.config.event_probability:
                # Random event shock
                direction = self.rng.choice([-1, 1])
                magnitude = self.rng.uniform(0.2, self.config.event_magnitude)
                scores[i] = direction * magnitude
            else:
                # Decay towards neutral
                if i > 0:
                    scores[i] = scores[i - 1] * (1 - self.config.mean_reversion_strength)

        return scores

    def _generate_regime_based(
        self,
        dates: List[pd.Timestamp],
        price_data: pd.DataFrame,
    ) -> np.ndarray:
        """Generate regime-based sentiment.

        Detects bull/bear regimes and applies sentiment bias.

        Args:
            dates: Dates to generate for
            price_data: OHLCV DataFrame

        Returns:
            Array of sentiment scores
        """
        n = len(dates)
        scores = np.zeros(n)

        if "close" not in price_data.columns:
            logger.warning("No 'close' column in price data, using random walk")
            return self._generate_random_walk(n)

        close_prices = price_data["close"]

        # Calculate MAs for regime detection
        ma_short = close_prices.rolling(self.config.regime_ma_short, min_periods=1).mean()
        ma_long = close_prices.rolling(self.config.regime_ma_long, min_periods=1).mean()

        for i, date in enumerate(dates):
            # Determine regime
            try:
                if date in ma_short.index and date in ma_long.index:
                    short_val = ma_short.loc[date]
                    long_val = ma_long.loc[date]
                else:
                    # Find closest date
                    closest_idx = ma_short.index.get_indexer([date], method="nearest")[0]
                    if closest_idx >= 0:
                        short_val = ma_short.iloc[closest_idx]
                        long_val = ma_long.iloc[closest_idx]
                    else:
                        short_val = long_val = 0

                if pd.notna(short_val) and pd.notna(long_val) and long_val > 0:
                    if short_val > long_val:
                        # Bull regime
                        regime_bias = self.config.regime_bias
                    else:
                        # Bear regime
                        regime_bias = -self.config.regime_bias
                else:
                    regime_bias = 0.0

            except Exception:
                regime_bias = 0.0

            # Add random variation around regime bias
            scores[i] = regime_bias + self.rng.normal(0, 0.15)

        return np.clip(scores, -1.0, 1.0)

    def _generate_combined(
        self,
        dates: List[pd.Timestamp],
        symbol: str,
        price_data: pd.DataFrame,
    ) -> np.ndarray:
        """Generate combined sentiment using all modes.

        Args:
            dates: Dates to generate for
            symbol: Stock symbol
            price_data: OHLCV DataFrame

        Returns:
            Array of sentiment scores
        """
        # Generate all components
        price_sentiment = self._generate_price_correlated(dates, price_data)
        event_sentiment = self._generate_event_based(dates, symbol)
        regime_sentiment = self._generate_regime_based(dates, price_data)

        # Combine with weights
        # Events have highest priority, then price, then regime
        combined = np.zeros(len(dates))
        for i in range(len(dates)):
            if abs(event_sentiment[i]) > 0.3:
                # Strong event dominates
                combined[i] = 0.6 * event_sentiment[i] + 0.3 * price_sentiment[i] + 0.1 * regime_sentiment[i]
            else:
                # Normal weighting
                combined[i] = 0.5 * price_sentiment[i] + 0.3 * regime_sentiment[i] + 0.2 * event_sentiment[i]

        return np.clip(combined, -1.0, 1.0)

    def _get_calendar_event_sentiment(
        self,
        date: pd.Timestamp,
        symbol: str,
    ) -> Optional[float]:
        """Get sentiment from event calendar for a specific date/symbol.

        Args:
            date: Date to check
            symbol: Stock symbol

        Returns:
            Sentiment impact if event found, None otherwise
        """
        if self._event_calendar is None:
            return None

        # Normalize date for comparison
        date_normalized = pd.Timestamp(date.date())

        # Filter events for this date and symbol
        mask = self._event_calendar["date"].dt.normalize() == date_normalized

        if "symbols" in self._event_calendar.columns:
            # Check if symbol matches (symbols can be comma-separated or "ALL")
            def symbol_matches(symbols_str):
                if pd.isna(symbols_str):
                    return False
                symbols_str = str(symbols_str).upper()
                return symbol.upper() in symbols_str or symbols_str == "ALL"

            mask = mask & self._event_calendar["symbols"].apply(symbol_matches)

        events = self._event_calendar[mask]

        if len(events) > 0:
            # Return the sentiment impact (average if multiple events)
            return float(events["sentiment_impact"].mean())

        return None

    def _get_event_types(
        self,
        dates: List[pd.Timestamp],
        symbol: str,
    ) -> List[Optional[str]]:
        """Get event types for all dates.

        Args:
            dates: List of dates
            symbol: Stock symbol

        Returns:
            List of event types (None if no event)
        """
        event_types = []

        for date in dates:
            event_type = None

            if self._event_calendar is not None:
                date_normalized = pd.Timestamp(date.date())
                mask = self._event_calendar["date"].dt.normalize() == date_normalized

                if "symbols" in self._event_calendar.columns:

                    def symbol_matches(symbols_str):
                        if pd.isna(symbols_str):
                            return False
                        symbols_str = str(symbols_str).upper()
                        return symbol.upper() in symbols_str or symbols_str == "ALL"

                    mask = mask & self._event_calendar["symbols"].apply(symbol_matches)

                events = self._event_calendar[mask]
                if len(events) > 0 and "event_type" in events.columns:
                    event_type = events.iloc[0]["event_type"]

            event_types.append(event_type)

        return event_types

    def _generate_confidence(self, sentiment_scores: np.ndarray) -> np.ndarray:
        """Generate confidence scores.

        Higher confidence for extreme sentiment (more articles, more agreement).

        Args:
            sentiment_scores: Array of sentiment scores

        Returns:
            Array of confidence scores
        """
        # Base confidence
        base_confidence = 0.5

        # Higher confidence for extreme sentiment
        extremity = np.abs(sentiment_scores)
        confidence = base_confidence + 0.4 * extremity

        # Add some noise
        noise = self.rng.normal(0, 0.1, len(sentiment_scores))
        confidence = confidence + noise

        return np.clip(confidence, 0.2, 0.95)

    def _generate_article_counts(self, sentiment_scores: np.ndarray) -> np.ndarray:
        """Generate simulated article counts.

        More articles during high-sentiment periods.

        Args:
            sentiment_scores: Array of sentiment scores

        Returns:
            Array of article counts
        """
        # Base count
        base_count = 3

        # More articles for extreme sentiment
        extremity = np.abs(sentiment_scores)
        extra_articles = (extremity * 10).astype(int)

        # Random variation
        variation = self.rng.integers(-2, 5, len(sentiment_scores))

        counts = base_count + extra_articles + variation
        return np.clip(counts, 1, 20)

    def _score_to_label(self, score: float) -> SentimentLabel:
        """Convert numeric score to sentiment label.

        Args:
            score: Sentiment score (-1 to 1)

        Returns:
            SentimentLabel
        """
        if score >= 0.5:
            return SentimentLabel.VERY_POSITIVE
        elif score >= 0.1:
            return SentimentLabel.POSITIVE
        elif score <= -0.5:
            return SentimentLabel.VERY_NEGATIVE
        elif score <= -0.1:
            return SentimentLabel.NEGATIVE
        else:
            return SentimentLabel.NEUTRAL

    def _empty_dataframe(self) -> pd.DataFrame:
        """Return empty DataFrame with correct schema."""
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "sentiment_score",
                "sentiment_label",
                "confidence",
                "article_count",
                "event_type",
            ]
        )


def generate_sentiment_for_backtest(
    symbols: List[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    price_data: Dict[str, pd.DataFrame],
    config: Optional[SentimentConfig] = None,
) -> pd.DataFrame:
    """Convenience function to generate sentiment for backtesting.

    Args:
        symbols: List of stock symbols
        start_date: Start date
        end_date: End date
        price_data: Dictionary mapping symbol to OHLCV DataFrame
        config: Optional sentiment configuration

    Returns:
        DataFrame with synthetic sentiment for all symbols
    """
    generator = SyntheticSentimentGenerator(config)

    # Generate date range (business days)
    dates = pd.date_range(start=start_date, end=end_date, freq="B").tolist()

    return generator.generate_for_symbols(symbols, dates, price_data)

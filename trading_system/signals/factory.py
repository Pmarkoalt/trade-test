"""Factory functions for creating signal generators with various configurations."""

import logging
import os
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ..strategies.base.strategy_interface import StrategyInterface

from .config import SignalConfig
from .live_signal_generator import LiveSignalGenerator

logger = logging.getLogger(__name__)


def create_live_signal_generator(
    strategies: List["StrategyInterface"],
    news_enabled: bool = True,
    newsapi_key: Optional[str] = None,
    alpha_vantage_key: Optional[str] = None,
    massive_api_key: Optional[str] = None,
    news_lookback_hours: int = 48,
    technical_weight: float = 0.6,
    news_weight: float = 0.4,
    tracking_db: Optional[str] = None,
    max_recommendations: int = 5,
    min_conviction: str = "MEDIUM",
) -> LiveSignalGenerator:
    """Create a LiveSignalGenerator with optional news integration.

    This is a convenience factory that handles:
    - API key loading from environment variables
    - NewsAnalyzer initialization
    - SignalConfig creation
    - Proper error handling for missing dependencies

    Args:
        strategies: List of strategy instances
        news_enabled: Whether to enable news sentiment integration
        newsapi_key: NewsAPI.org key (falls back to NEWSAPI_KEY env var)
        alpha_vantage_key: Alpha Vantage key (falls back to ALPHA_VANTAGE_API_KEY env var)
        news_lookback_hours: How far back to search for news
        technical_weight: Weight for technical signals (0-1)
        news_weight: Weight for news signals (0-1)
        tracking_db: Optional path to SQLite database for signal tracking
        max_recommendations: Maximum recommendations to return
        min_conviction: Minimum conviction level ("LOW", "MEDIUM", "HIGH")

    Returns:
        Configured LiveSignalGenerator instance

    Example:
        ```python
        from trading_system.signals.factory import create_live_signal_generator

        generator = create_live_signal_generator(
            strategies=[equity_strategy],
            news_enabled=True,
            tracking_db="signals.db",
        )

        recommendations = await generator.generate_recommendations(
            ohlcv_data=data,
            current_date=date.today(),
        )
        ```
    """
    # Get API keys from environment if not provided
    if newsapi_key is None:
        newsapi_key = os.getenv("NEWSAPI_KEY")
    if alpha_vantage_key is None:
        alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if massive_api_key is None:
        massive_api_key = os.getenv("MASSIVE_API_KEY")

    # Create signal config
    config = SignalConfig(
        max_recommendations=max_recommendations,
        min_conviction=min_conviction,
        technical_weight=technical_weight,
        news_weight=news_weight,
        news_enabled=news_enabled
        and (newsapi_key is not None or alpha_vantage_key is not None or massive_api_key is not None),
        news_lookback_hours=news_lookback_hours,
    )

    # Initialize news analyzer if enabled and keys available
    news_analyzer = None
    if news_enabled:
        if not newsapi_key and not alpha_vantage_key and not massive_api_key:
            logger.warning(
                "News enabled but no API keys found. "
                "Set NEWSAPI_KEY, ALPHA_VANTAGE_API_KEY, or MASSIVE_API_KEY environment variables. "
                "Continuing without news integration."
            )
        else:
            try:
                from ..research.config import ResearchConfig
                from ..research.news_analyzer import NewsAnalyzer

                research_config = ResearchConfig(
                    newsapi_key=newsapi_key,
                    alpha_vantage_key=alpha_vantage_key,
                    massive_api_key=massive_api_key,
                    lookback_hours=news_lookback_hours,
                )

                news_analyzer = NewsAnalyzer(research_config)

                sources = []
                if newsapi_key:
                    sources.append("NewsAPI")
                if alpha_vantage_key:
                    sources.append("AlphaVantage")
                if massive_api_key:
                    sources.append("Polygon")
                logger.info(f"News integration enabled with sources: {', '.join(sources)}")

            except ImportError as e:
                logger.warning(f"Failed to initialize news analyzer: {e}. Continuing without news.")
            except Exception as e:
                logger.error(f"Error initializing news analyzer: {e}. Continuing without news.")

    # Create and return the generator
    return LiveSignalGenerator(
        strategies=strategies,
        signal_config=config,
        news_analyzer=news_analyzer,
        tracking_db=tracking_db,
    )


def create_backtest_signal_generator(
    strategies: List["StrategyInterface"],
    tracking_db: Optional[str] = None,
) -> LiveSignalGenerator:
    """Create a signal generator for backtesting (no news integration).

    For backtesting, use synthetic sentiment in the backtest engine instead.
    This generator uses only technical signals.

    Args:
        strategies: List of strategy instances
        tracking_db: Optional path to SQLite database for signal tracking

    Returns:
        LiveSignalGenerator configured for backtesting
    """
    config = SignalConfig(
        news_enabled=False,
        technical_weight=1.0,
        news_weight=0.0,
    )

    return LiveSignalGenerator(
        strategies=strategies,
        signal_config=config,
        news_analyzer=None,
        tracking_db=tracking_db,
    )

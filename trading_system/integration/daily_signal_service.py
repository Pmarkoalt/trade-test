"""Daily signal generation service using canonical contracts.

This module provides a high-level interface for generating daily signals
using the canonical Signal, Allocation, TradePlan, and DailySignalBatch contracts.
"""

import os
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..configs.run_config import RunConfig
from ..data.loader import load_universe
from ..data_pipeline.config import DataPipelineConfig
from ..data_pipeline.live_data_fetcher import LiveDataFetcher
from ..logging.logger import get_logger
from ..models.contracts import (
    Allocation,
    AssetClass,
    DailySignalBatch,
    OrderMethod,
    Signal,
    SignalIntent,
    StopLogicType,
    TradePlan,
)
from ..research.config import ResearchConfig
from ..signals.config import SignalConfig
from ..signals.live_signal_generator import LiveSignalGenerator
from ..strategies.strategy_loader import load_strategies_from_run_config

logger = get_logger(__name__)


class DailySignalService:
    """Service for generating daily signals using canonical contracts.

    This service wraps the existing signal generation logic and converts
    outputs to the canonical contract format for use by downstream consumers
    (newsletter, paper trading, reporting).
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        data_pipeline_config: Optional[DataPipelineConfig] = None,
        signal_config: Optional[SignalConfig] = None,
        research_config: Optional[ResearchConfig] = None,
    ):
        """Initialize the daily signal service.

        Args:
            config_path: Path to run config file (optional)
            data_pipeline_config: Data pipeline configuration (optional)
            signal_config: Signal generation configuration (optional)
            research_config: Research/news configuration (optional)
        """
        self.config_path = config_path
        self.data_pipeline_config = data_pipeline_config or self._load_data_pipeline_config()
        self.signal_config = signal_config or self._load_signal_config()
        self.research_config = research_config or self._load_research_config()

        self.data_fetcher = LiveDataFetcher(self.data_pipeline_config)
        self.news_analyzer = None

        if self.research_config.enabled:
            try:
                from ..research.news_analyzer import NewsAnalyzer

                self.news_analyzer = NewsAnalyzer(
                    config=self.research_config,
                    newsapi_key=self.research_config.newsapi_key,
                    alpha_vantage_key=self.research_config.alpha_vantage_key,
                )
                logger.info("News analyzer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize news analyzer: {e}")

    def _load_data_pipeline_config(self) -> DataPipelineConfig:
        """Load data pipeline config from environment."""
        return DataPipelineConfig(
            massive_api_key=os.getenv("MASSIVE_API_KEY"),
            alpha_vantage_api_key=os.getenv("ALPHA_VANTAGE_API_KEY"),
            cache_path=Path(os.getenv("DATA_CACHE_PATH", "data/cache")),
            cache_ttl_hours=int(os.getenv("CACHE_TTL_HOURS", "24")),
        )

    def _load_signal_config(self) -> SignalConfig:
        """Load signal config from environment."""
        return SignalConfig(
            max_recommendations=int(os.getenv("MAX_RECOMMENDATIONS", "5")),
            min_conviction=os.getenv("MIN_CONVICTION", "MEDIUM"),
            technical_weight=float(os.getenv("TECHNICAL_WEIGHT", "0.6")),
            news_weight=float(os.getenv("NEWS_WEIGHT", "0.4")),
            news_enabled=os.getenv("NEWS_ENABLED", "true").lower() == "true",
            news_lookback_hours=int(os.getenv("NEWS_LOOKBACK_HOURS", "48")),
            min_news_score_for_boost=float(os.getenv("MIN_NEWS_SCORE_FOR_BOOST", "7.0")),
            max_news_score_for_penalty=float(os.getenv("MAX_NEWS_SCORE_FOR_PENALTY", "3.0")),
        )

    def _load_research_config(self) -> ResearchConfig:
        """Load research config from environment."""
        return ResearchConfig(
            enabled=os.getenv("RESEARCH_ENABLED", "true").lower() == "true",
            newsapi_key=os.getenv("NEWSAPI_KEY"),
            alpha_vantage_key=os.getenv("ALPHA_VANTAGE_API_KEY"),
            massive_api_key=os.getenv("MASSIVE_API_KEY"),
            lookback_hours=int(os.getenv("RESEARCH_LOOKBACK_HOURS", "48")),
            max_articles_per_symbol=int(os.getenv("MAX_ARTICLES_PER_SYMBOL", "10")),
        )

    async def generate_daily_signals(
        self,
        asset_class: str,
        bucket: Optional[str] = None,
        current_date: Optional[date] = None,
    ) -> DailySignalBatch:
        """Generate daily signals for a specific asset class and bucket.

        Args:
            asset_class: Asset class to generate signals for ("equity" or "crypto")
            bucket: Strategy bucket (e.g., "safe_sp500", "aggressive_crypto")
            current_date: Date to generate signals for (defaults to today)

        Returns:
            DailySignalBatch containing signals, allocations, and trade plans
        """
        if current_date is None:
            current_date = date.today()

        logger.info(f"Generating daily signals for {asset_class} (bucket: {bucket or 'default'})")

        # Load strategies
        strategies = await self._load_strategies(asset_class)
        if not strategies:
            logger.error(f"No strategies loaded for {asset_class}")
            return DailySignalBatch(generation_date=pd.Timestamp(current_date))

        # Initialize signal generator
        tracking_db = os.getenv("TRACKING_DB_PATH", "tracking.db")
        signal_generator = LiveSignalGenerator(
            strategies=strategies,
            signal_config=self.signal_config,
            news_analyzer=self.news_analyzer,
            tracking_db=tracking_db,
        )

        # Get universe symbols
        universe_type = await self._get_universe_type(asset_class)
        symbols = load_universe(universe_type)

        if not symbols:
            logger.warning(f"No symbols found for {asset_class} universe: {universe_type}")
            return DailySignalBatch(generation_date=pd.Timestamp(current_date))

        print(f"📊 Loaded {len(symbols)} symbols from {universe_type} universe")
        logger.info(f"Fetching data for {len(symbols)} symbols")

        # Fetch data
        print(f"🔄 Fetching market data for {len(symbols)} symbols (this may take 2-5 minutes)...")
        ohlcv_data = await self.data_fetcher.fetch_daily_data(symbols=symbols, asset_class=asset_class, lookback_days=252)

        if not ohlcv_data:
            logger.warning(f"No data fetched for {asset_class}")
            return DailySignalBatch(generation_date=pd.Timestamp(current_date))

        # CRITICAL FIX: Use the most recent date available in the data, not today's date
        # This handles cases where today's data isn't available yet (before market close, weekends, holidays)
        most_recent_dates = [df["date"].max() for df in ohlcv_data.values() if not df.empty and "date" in df.columns]
        if most_recent_dates:
            most_recent_date = max(most_recent_dates)
            if isinstance(most_recent_date, str):
                most_recent_date = pd.to_datetime(most_recent_date).date()
            elif isinstance(most_recent_date, pd.Timestamp):
                most_recent_date = most_recent_date.date()

            if most_recent_date < current_date:
                print(f"⚠️  Today's data not available yet. Using most recent date: {most_recent_date}")
                logger.info(f"Adjusting signal date from {current_date} to {most_recent_date} (most recent available data)")
                current_date = most_recent_date

        print(f"✅ Fetched data for {len(ohlcv_data)} symbols")
        logger.info(f"Fetched data for {len(ohlcv_data)} symbols")

        # Generate recommendations using existing signal generator
        print(f"🧮 Generating trading signals...")
        recommendations = await signal_generator.generate_recommendations(
            ohlcv_data=ohlcv_data,
            current_date=current_date,
            portfolio_state=None,
        )

        print(f"✅ Generated {len(recommendations)} recommendations")
        logger.info(f"Generated {len(recommendations)} recommendations")

        # Convert to canonical contracts
        signals = []
        allocations = []
        trade_plans = []

        for rec in recommendations:
            # Convert to canonical Signal
            signal = self._convert_to_canonical_signal(rec, asset_class, bucket)
            signals.append(signal)

            # Create Allocation
            allocation = self._create_allocation(rec, signal)
            allocations.append(allocation)

            # Create TradePlan
            trade_plan = self._create_trade_plan(rec, signal, allocation)
            trade_plans.append(trade_plan)

        # Create batch
        batch = DailySignalBatch(
            generation_date=pd.Timestamp(current_date),
            signals=signals,
            allocations=allocations,
            trade_plans=trade_plans,
            bucket_summaries={
                bucket
                or "default": {
                    "total_signals": len(signals),
                    "asset_class": asset_class,
                    "avg_confidence": sum(s.confidence for s in signals) / len(signals) if signals else 0.0,
                }
            },
            metadata={
                "asset_class": asset_class,
                "bucket": bucket,
                "universe_type": universe_type,
                "symbols_analyzed": len(symbols),
                "data_symbols": len(ohlcv_data),
            },
        )

        return batch

    def _convert_to_canonical_signal(self, recommendation: Dict, asset_class: str, bucket: Optional[str]) -> Signal:
        """Convert recommendation to canonical Signal contract."""
        return Signal(
            symbol=recommendation.get("symbol", ""),
            asset_class=AssetClass.EQUITY if asset_class == "equity" else AssetClass.CRYPTO,
            timestamp=pd.Timestamp(recommendation.get("date", date.today())),
            side=recommendation.get("side", "BUY"),
            intent=SignalIntent.EXECUTE_NEXT_OPEN,
            confidence=recommendation.get("conviction_score", 0.5),
            rationale_tags={
                "technical": recommendation.get("technical_reason", ""),
                "news": recommendation.get("news_summary", ""),
                "conviction": recommendation.get("conviction", "MEDIUM"),
                "strategy": recommendation.get("strategy", ""),
            },
            entry_price=recommendation.get("entry_price", 0.0),
            stop_price=recommendation.get("stop_price", 0.0),
            bucket=bucket,
            strategy_name=recommendation.get("strategy", ""),
            metadata=recommendation,
        )

    def _create_allocation(self, recommendation: Dict, signal: Signal) -> Allocation:
        """Create Allocation from recommendation."""
        position_size = recommendation.get("position_size_dollars", 0.0)
        portfolio_pct = recommendation.get("position_size_pct", 0.0)

        return Allocation(
            symbol=signal.symbol,
            signal_timestamp=signal.timestamp,
            recommended_position_size_dollars=position_size,
            recommended_position_size_percent=portfolio_pct,
            risk_budget_used=recommendation.get("risk_budget_used", 0.0),
            max_positions_constraint_applied=recommendation.get("max_positions_applied", False),
            liquidity_flags=recommendation.get("liquidity_flags", []),
            capacity_flags=recommendation.get("capacity_flags", []),
            quantity=recommendation.get("quantity", 0),
            max_adv_percent=recommendation.get("max_adv_pct", 0.0),
            notes=recommendation.get("notes", ""),
        )

    def _create_trade_plan(self, recommendation: Dict, signal: Signal, allocation: Allocation) -> TradePlan:
        """Create TradePlan from recommendation."""
        return TradePlan(
            symbol=signal.symbol,
            signal_timestamp=signal.timestamp,
            entry_method=OrderMethod.MOO,
            entry_price=signal.entry_price,
            stop_logic=StopLogicType.ATR_TRAILING,
            stop_price=signal.stop_price,
            stop_params={
                "atr_mult": recommendation.get("atr_mult", 2.5),
            },
            exit_logic=recommendation.get("exit_logic", "ma_cross"),
            exit_params={
                "ma_period": recommendation.get("ma_period", 20),
            },
            time_stop_days=recommendation.get("time_stop_days"),
            allocation=allocation,
            notes=recommendation.get("notes", ""),
        )

    async def _load_strategies(self, asset_class: str) -> List:
        """Load strategies for the given asset class."""
        try:
            config_path = self.config_path or os.getenv("RUN_CONFIG_PATH", "configs/production_run_config.yaml")
            if not Path(config_path).exists():
                logger.warning(f"Config path not found: {config_path}")
                return []

            run_config = RunConfig.from_yaml(config_path)
            equity_config_path = (
                run_config.strategies.equity.config_path
                if run_config.strategies.equity and run_config.strategies.equity.enabled
                else None
            )
            crypto_config_path = (
                run_config.strategies.crypto.config_path
                if run_config.strategies.crypto and run_config.strategies.crypto.enabled
                else None
            )

            strategies = load_strategies_from_run_config(
                equity_config_path=equity_config_path if asset_class == "equity" else None,
                crypto_config_path=crypto_config_path if asset_class == "crypto" else None,
            )
            return strategies
        except Exception as e:
            logger.error(f"Failed to load strategies: {e}")
            return []

    async def _get_universe_type(self, asset_class: str) -> str:
        """Get universe type for the given asset class."""
        try:
            config_path = self.config_path or os.getenv("RUN_CONFIG_PATH", "configs/production_run_config.yaml")
            if not Path(config_path).exists():
                return "NASDAQ-100" if asset_class == "equity" else "crypto"

            run_config = RunConfig.from_yaml(config_path)
            if asset_class == "equity" and run_config.strategies.equity:
                strategy_config_path = run_config.strategies.equity.config_path
                if strategy_config_path and Path(strategy_config_path).exists():
                    from ..configs.strategy_config import StrategyConfig

                    strategy = StrategyConfig.from_yaml(strategy_config_path)
                    return strategy.universe
            elif asset_class == "crypto" and run_config.strategies.crypto:
                strategy_config_path = run_config.strategies.crypto.config_path
                if strategy_config_path and Path(strategy_config_path).exists():
                    from ..configs.strategy_config import StrategyConfig

                    strategy = StrategyConfig.from_yaml(strategy_config_path)
                    return strategy.universe
        except Exception as e:
            logger.warning(f"Failed to load universe from config: {e}")

        return "NASDAQ-100" if asset_class == "equity" else "crypto"

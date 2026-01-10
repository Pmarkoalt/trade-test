"""Newsletter job for scheduled daily newsletter generation and delivery.

This job generates and sends daily newsletters with signals organized by strategy buckets.
"""

import os
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ...configs.run_config import RunConfig
from ...data.loader import load_universe
from ...data_pipeline.config import DataPipelineConfig
from ...data_pipeline.live_data_fetcher import LiveDataFetcher
from ...logging.logger import get_logger
from ...models.signals import Signal
from ...output.email.config import EmailConfig
from ...output.email.newsletter_service import NewsletterService
from ...research.config import ResearchConfig
from ...signals.config import SignalConfig
from ...signals.live_signal_generator import LiveSignalGenerator
from ...strategies.strategy_loader import load_strategies_from_run_config

logger = get_logger(__name__)


def load_newsletter_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration for newsletter job.

    Args:
        config_path: Optional path to config file. If None, uses environment variables
                     or default production configs.

    Returns:
        Dictionary with config sections: data_pipeline, signals, email, universe, research
    """
    if config_path is None:
        prod_config_path = Path("configs/production_run_config.yaml")
        if prod_config_path.exists():
            config_path = str(prod_config_path)

    config_dict = {}

    # Load data pipeline config
    data_pipeline_config = DataPipelineConfig(
        massive_api_key=os.getenv("MASSIVE_API_KEY"),
        alpha_vantage_api_key=os.getenv("ALPHA_VANTAGE_API_KEY"),
        cache_path=Path(os.getenv("DATA_CACHE_PATH", "data/cache")),
        cache_ttl_hours=int(os.getenv("CACHE_TTL_HOURS", "24")),
    )
    config_dict["data_pipeline"] = data_pipeline_config

    # Load signals config
    signal_config = SignalConfig(
        max_recommendations=int(os.getenv("MAX_RECOMMENDATIONS", "10")),
        min_conviction=os.getenv("MIN_CONVICTION", "MEDIUM"),
        technical_weight=float(os.getenv("TECHNICAL_WEIGHT", "0.6")),
        news_weight=float(os.getenv("NEWS_WEIGHT", "0.4")),
        news_enabled=os.getenv("NEWS_ENABLED", "true").lower() == "true",
        news_lookback_hours=int(os.getenv("NEWS_LOOKBACK_HOURS", "48")),
        min_news_score_for_boost=float(os.getenv("MIN_NEWS_SCORE_FOR_BOOST", "7.0")),
        max_news_score_for_penalty=float(os.getenv("MAX_NEWS_SCORE_FOR_PENALTY", "3.0")),
    )
    config_dict["signals"] = signal_config

    # Load research config
    research_config = ResearchConfig(
        enabled=os.getenv("RESEARCH_ENABLED", "true").lower() == "true",
        newsapi_key=os.getenv("NEWSAPI_KEY"),
        alpha_vantage_key=os.getenv("ALPHA_VANTAGE_API_KEY"),
        massive_api_key=os.getenv("MASSIVE_API_KEY"),
        lookback_hours=int(os.getenv("RESEARCH_LOOKBACK_HOURS", "48")),
        max_articles_per_symbol=int(os.getenv("MAX_ARTICLES_PER_SYMBOL", "10")),
    )
    config_dict["research"] = research_config

    # Load email config
    email_config = EmailConfig(
        smtp_host=os.getenv("SMTP_HOST", "smtp.sendgrid.net"),
        smtp_port=int(os.getenv("SMTP_PORT", "587")),
        smtp_user=os.getenv("SMTP_USER", "apikey"),
        smtp_password=os.getenv("SMTP_PASSWORD", os.getenv("SENDGRID_API_KEY", "")),
        from_email=os.getenv("FROM_EMAIL", "signals@yourdomain.com"),
        from_name=os.getenv("FROM_NAME", "Trading Assistant"),
        recipients=os.getenv("EMAIL_RECIPIENTS", "").split(",") if os.getenv("EMAIL_RECIPIENTS") else [],
    )
    config_dict["email"] = email_config

    # Load universe config
    universe_config = {}
    if config_path and Path(config_path).exists():
        try:
            run_config = RunConfig.from_yaml(config_path)
            if run_config.strategies.equity and run_config.strategies.equity.enabled:
                equity_strategy_config_path = run_config.strategies.equity.config_path
                if equity_strategy_config_path and Path(equity_strategy_config_path).exists():
                    from ...configs.strategy_config import StrategyConfig

                    equity_strategy = StrategyConfig.from_yaml(equity_strategy_config_path)
                    universe_config["equity"] = equity_strategy.universe
            if run_config.strategies.crypto and run_config.strategies.crypto.enabled:
                crypto_strategy_config_path = run_config.strategies.crypto.config_path
                if crypto_strategy_config_path and Path(crypto_strategy_config_path).exists():
                    from ...configs.strategy_config import StrategyConfig

                    crypto_strategy = StrategyConfig.from_yaml(crypto_strategy_config_path)
                    universe_config["crypto"] = crypto_strategy.universe
        except Exception as e:
            logger.warning(f"Failed to load universe from config: {e}")

    # Fallback to default universes
    if "equity" not in universe_config:
        universe_config["equity"] = "SP500"
    if "crypto" not in universe_config:
        universe_config["crypto"] = "crypto"

    config_dict["universe"] = universe_config

    return config_dict


def get_market_summary(ohlcv_data: Dict[str, pd.DataFrame]) -> Dict:
    """Generate market summary from OHLCV data.

    Args:
        ohlcv_data: Dictionary mapping symbol to OHLCV DataFrame

    Returns:
        Dictionary with market summary statistics
    """
    summary = {
        "spy_price": None,
        "spy_pct": None,
        "btc_price": None,
        "btc_pct": None,
        "regime": "Unknown",
    }

    # Get SPY data if available
    if "SPY" in ohlcv_data:
        spy_df = ohlcv_data["SPY"]
        if len(spy_df) >= 2:
            summary["spy_price"] = float(spy_df["close"].iloc[-1])
            prev_close = float(spy_df["close"].iloc[-2])
            summary["spy_pct"] = ((summary["spy_price"] - prev_close) / prev_close) * 100

    # Get BTC data if available
    if "BTC" in ohlcv_data or "BTCUSD" in ohlcv_data:
        btc_symbol = "BTC" if "BTC" in ohlcv_data else "BTCUSD"
        btc_df = ohlcv_data[btc_symbol]
        if len(btc_df) >= 2:
            summary["btc_price"] = float(btc_df["close"].iloc[-1])
            prev_close = float(btc_df["close"].iloc[-2])
            summary["btc_pct"] = ((summary["btc_price"] - prev_close) / prev_close) * 100

    # Determine market regime based on SPY trend
    if summary["spy_price"] and "SPY" in ohlcv_data:
        spy_df = ohlcv_data["SPY"]
        if len(spy_df) >= 50:
            ma_50 = spy_df["close"].rolling(50).mean().iloc[-1]
            if summary["spy_price"] > ma_50:
                summary["regime"] = "Bullish"
            else:
                summary["regime"] = "Bearish"

    return summary


async def newsletter_job() -> None:
    """Execute daily newsletter generation and delivery.

    This job generates signals for all configured buckets and sends a combined newsletter.
    """
    logger.info("Starting newsletter job")

    try:
        # 1. Load configuration
        config = load_newsletter_config()

        if not config["email"].recipients:
            logger.warning("No email recipients configured. Skipping newsletter job.")
            return

        # 2. Initialize components
        data_fetcher = LiveDataFetcher(config["data_pipeline"])

        # Initialize news analyzer if research is enabled
        news_analyzer = None
        if config["research"].enabled:
            try:
                from ...research.news_analyzer import NewsAnalyzer

                news_analyzer = NewsAnalyzer(
                    config=config["research"],
                    newsapi_key=config["research"].newsapi_key,
                    alpha_vantage_key=config["research"].alpha_vantage_key,
                )
                logger.info("News analyzer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize news analyzer: {e}. Continuing without news.")
                news_analyzer = None

        # 3. Generate signals for each bucket
        signals_by_bucket: Dict[str, List[Signal]] = {}
        all_ohlcv_data: Dict[str, pd.DataFrame] = {}

        # Bucket A: Safe S&P (Equity)
        try:
            logger.info("Generating signals for Bucket A: Safe S&P")
            equity_signals, equity_ohlcv = await generate_bucket_signals(
                bucket_name="safe_sp",
                asset_class="equity",
                universe_type=config["universe"].get("equity", "SP500"),
                data_fetcher=data_fetcher,
                signal_config=config["signals"],
                news_analyzer=news_analyzer,
            )
            signals_by_bucket["safe_sp"] = equity_signals
            all_ohlcv_data.update(equity_ohlcv)
            logger.info(f"Generated {len(equity_signals)} signals for Safe S&P bucket")
        except Exception as e:
            logger.error(f"Failed to generate Safe S&P signals: {e}", exc_info=True)
            signals_by_bucket["safe_sp"] = []

        # Bucket B: Aggressive top-cap crypto
        try:
            logger.info("Generating signals for Bucket B: Crypto Top-Cap")
            crypto_signals, crypto_ohlcv = await generate_bucket_signals(
                bucket_name="crypto_topCap",
                asset_class="crypto",
                universe_type=config["universe"].get("crypto", "crypto"),
                data_fetcher=data_fetcher,
                signal_config=config["signals"],
                news_analyzer=news_analyzer,
            )
            signals_by_bucket["crypto_topCap"] = crypto_signals
            all_ohlcv_data.update(crypto_ohlcv)
            logger.info(f"Generated {len(crypto_signals)} signals for Crypto Top-Cap bucket")
        except Exception as e:
            logger.error(f"Failed to generate Crypto Top-Cap signals: {e}", exc_info=True)
            signals_by_bucket["crypto_topCap"] = []

        # 4. Get market summary
        market_summary = get_market_summary(all_ohlcv_data)

        # 5. Get news analysis
        news_analysis = None
        if news_analyzer and config["research"].enabled:
            try:
                all_symbols = list(all_ohlcv_data.keys())
                news_analysis = await news_analyzer.analyze_symbols(
                    symbols=all_symbols,
                    lookback_hours=config["research"].lookback_hours,
                )
                logger.info(f"Analyzed {news_analysis.total_articles if news_analysis else 0} news articles")
            except Exception as e:
                logger.warning(f"Failed to fetch news analysis: {e}. Continuing without news.")
                news_analysis = None

        # 6. Send newsletter
        newsletter_service = NewsletterService(config["email"])
        success = await newsletter_service.send_daily_newsletter(
            signals_by_bucket=signals_by_bucket,
            market_summary=market_summary,
            news_analysis=news_analysis,
            portfolio_summary=None,
            date_obj=date.today(),
        )

        if success:
            total_signals = sum(len(signals) for signals in signals_by_bucket.values())
            logger.info(f"Newsletter job completed: {total_signals} total signals across {len(signals_by_bucket)} buckets")
        else:
            logger.error("Failed to send newsletter")

    except Exception as e:
        logger.error(f"Newsletter job failed: {e}", exc_info=True)
        raise


async def generate_bucket_signals(
    bucket_name: str,
    asset_class: str,
    universe_type: str,
    data_fetcher: LiveDataFetcher,
    signal_config: SignalConfig,
    news_analyzer: Optional[Any] = None,
) -> tuple[List[Signal], Dict[str, pd.DataFrame]]:
    """Generate signals for a specific strategy bucket.

    Args:
        bucket_name: Name of the bucket
        asset_class: Asset class ("equity" or "crypto")
        universe_type: Universe type to load
        data_fetcher: Data fetcher instance
        signal_config: Signal configuration
        news_analyzer: Optional news analyzer

    Returns:
        Tuple of (list of signals, OHLCV data dictionary)
    """
    # Load strategies for this bucket
    config_path = os.getenv("RUN_CONFIG_PATH", "configs/production_run_config.yaml")
    strategies = []

    if Path(config_path).exists():
        try:
            run_config = RunConfig.from_yaml(config_path)
            if asset_class == "equity" and run_config.strategies.equity and run_config.strategies.equity.enabled:
                equity_config_path = run_config.strategies.equity.config_path
                strategies = load_strategies_from_run_config(equity_config_path=equity_config_path)
            elif asset_class == "crypto" and run_config.strategies.crypto and run_config.strategies.crypto.enabled:
                crypto_config_path = run_config.strategies.crypto.config_path
                strategies = load_strategies_from_run_config(crypto_config_path=crypto_config_path)
        except Exception as e:
            logger.warning(f"Failed to load strategies for {bucket_name}: {e}")

    if not strategies:
        logger.warning(f"No strategies loaded for {bucket_name}")
        return [], {}

    # Get universe symbols
    symbols = load_universe(universe_type)
    if not symbols:
        logger.warning(f"No symbols found for {bucket_name} universe: {universe_type}")
        return [], {}

    # Fetch data
    ohlcv_data = await data_fetcher.fetch_daily_data(symbols=symbols, asset_class=asset_class, lookback_days=252)

    if not ohlcv_data:
        logger.warning(f"No data fetched for {bucket_name}")
        return [], {}

    # Generate signals
    tracking_db = os.getenv("TRACKING_DB_PATH", "tracking.db")
    signal_generator = LiveSignalGenerator(
        strategies=strategies,
        signal_config=signal_config,
        news_analyzer=news_analyzer,
        tracking_db=tracking_db,
    )

    recommendations = await signal_generator.generate_recommendations(
        ohlcv_data=ohlcv_data,
        current_date=date.today(),
        portfolio_state=None,
    )

    # Convert recommendations to signals
    signals = []
    for rec in recommendations:
        # Recommendations are already Signal-like objects in the current implementation
        # If they're not Signal objects, we'd need to convert them here
        if hasattr(rec, "symbol") and hasattr(rec, "side"):
            signals.append(rec)

    return signals, ohlcv_data

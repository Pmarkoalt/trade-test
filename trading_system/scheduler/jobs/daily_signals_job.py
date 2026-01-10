"""Daily signals job for scheduled execution."""

import os
from datetime import date
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ...configs.run_config import RunConfig
from ...data.loader import load_universe
from ...data_pipeline.config import DataPipelineConfig
from ...data_pipeline.live_data_fetcher import LiveDataFetcher
from ...logging.logger import get_logger
from ...output.email.config import EmailConfig
from ...output.email.email_service import EmailService
from ...research.config import ResearchConfig
from ...signals.config import SignalConfig
from ...signals.live_signal_generator import LiveSignalGenerator
from ...strategies.strategy_loader import load_strategies_from_run_config
from ...tracking.performance_calculator import PerformanceCalculator

logger = get_logger(__name__)


def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration for daily signals job.

    Args:
        config_path: Optional path to config file. If None, uses environment variables
                     or default production configs.

    Returns:
        Dictionary with config sections: data_pipeline, signals, email, universe
    """
    # Try to load from production configs if no path provided
    if config_path is None:
        # Try production configs
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
        max_recommendations=int(os.getenv("MAX_RECOMMENDATIONS", "5")),
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
        smtp_host=os.getenv("SMTP_HOST", "email-smtp.us-east-1.amazonaws.com"),
        smtp_port=int(os.getenv("SMTP_PORT", "587")),
        smtp_user=os.getenv("SMTP_USER", ""),
        smtp_password=os.getenv("SMTP_PASSWORD", ""),
        from_email=os.getenv("EMAIL_FROM", ""),
        from_name=os.getenv("EMAIL_FROM_NAME", "Trading Assistant"),
        recipients=os.getenv("EMAIL_RECIPIENTS", "").split(",") if os.getenv("EMAIL_RECIPIENTS") else [],
    )
    config_dict["email"] = email_config

    # Load universe config (from run config if available)
    universe_config = {}
    if config_path and Path(config_path).exists():
        try:
            run_config = RunConfig.from_yaml(config_path)
            # Extract universe info from strategies
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
        universe_config["equity"] = "NASDAQ-100"  # Default equity universe
    if "crypto" not in universe_config:
        universe_config["crypto"] = "crypto"  # Default crypto universe

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
        "total_symbols": len(ohlcv_data),
        "symbols": list(ohlcv_data.keys()),
        "date_range": None,
        "avg_volume": 0.0,
    }

    if not ohlcv_data:
        return summary

    # Get date range from first symbol
    first_symbol = list(ohlcv_data.keys())[0]
    first_df = ohlcv_data[first_symbol]
    if len(first_df) > 0:
        summary["date_range"] = {
            "start": str(first_df.index[0]),
            "end": str(first_df.index[-1]),
        }

    # Calculate average volume
    total_volume = 0.0
    count = 0
    for symbol, df in ohlcv_data.items():
        if "volume" in df.columns and len(df) > 0:
            total_volume += df["volume"].mean()
            count += 1

    if count > 0:
        summary["avg_volume"] = total_volume / count

    return summary


async def send_error_alert(error: Exception) -> None:
    """Send error alert email.

    Args:
        error: Exception that occurred
    """
    try:
        # Try to send error alert via email if configured
        email_recipients = os.getenv("EMAIL_RECIPIENTS", "").split(",")
        if not email_recipients or not email_recipients[0]:
            logger.warning("No email recipients configured, skipping error alert")
            return

        email_config = EmailConfig(
            smtp_host=os.getenv("SMTP_HOST", "email-smtp.us-east-1.amazonaws.com"),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            smtp_user=os.getenv("SMTP_USER", ""),
            smtp_password=os.getenv("SMTP_PASSWORD", ""),
            from_email=os.getenv("EMAIL_FROM", ""),
            from_name=os.getenv("EMAIL_FROM_NAME", "Trading Assistant"),
            recipients=email_recipients,
        )

        email_service = EmailService(email_config)
        error_subject = f"Trading System Error - {date.today()}"
        error_body = """
        <html>
        <body>
            <h2>Trading System Error Alert</h2>
            <p><strong>Date:</strong> {date.today()}</p>
            <p><strong>Error:</strong> {type(error).__name__}</p>
            <p><strong>Message:</strong> {str(error)}</p>
        </body>
        </html>
        """

        # Note: EmailService.send_daily_report is not async, but we'll call it anyway
        # In a real implementation, we might want to make it async or use a thread pool
        import asyncio

        await asyncio.to_thread(
            email_service._send_email,
            to=email_recipients,
            subject=error_subject,
            html=error_body,
        )
        logger.info("Error alert email sent")
    except Exception as e:
        logger.error(f"Failed to send error alert: {e}")


async def daily_signals_job(asset_class: str) -> None:
    """Execute daily signal generation with news analysis.

    Args:
        asset_class: Asset class to generate signals for ("equity" or "crypto")
    """
    logger.info(f"Starting daily signals job for {asset_class}")

    try:
        # 1. Load configuration
        config = load_config()

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

        # Load strategies
        strategies = []
        try:
            # Try to load from run config if available
            config_path = os.getenv("RUN_CONFIG_PATH", "configs/production_run_config.yaml")
            if Path(config_path).exists():
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
        except Exception as e:
            logger.warning(f"Failed to load strategies from config: {e}. Job may not work correctly.")

        if not strategies:
            logger.error(f"No strategies loaded for {asset_class}")
            return

        # Get tracking database path from environment or use default
        tracking_db = os.getenv("TRACKING_DB_PATH", "tracking.db")

        # Initialize signal generator with news analyzer and tracking
        signal_generator = LiveSignalGenerator(
            strategies=strategies,
            signal_config=config["signals"],
            news_analyzer=news_analyzer,
            tracking_db=tracking_db,
        )

        email_service = EmailService(config["email"])

        # 3. Get universe symbols
        universe_type = config["universe"].get(asset_class, "NASDAQ-100" if asset_class == "equity" else "crypto")
        symbols = load_universe(universe_type)

        if not symbols:
            logger.warning(f"No symbols found for {asset_class} universe: {universe_type}")
            return

        logger.info(f"Fetching data for {len(symbols)} symbols")

        # 4. Fetch data
        ohlcv_data = await data_fetcher.fetch_daily_data(symbols=symbols, asset_class=asset_class, lookback_days=252)

        if not ohlcv_data:
            logger.warning(f"No data fetched for {asset_class}")
            return

        logger.info(f"Fetched data for {len(ohlcv_data)} symbols")

        # 5. Generate recommendations (now includes news analysis)
        recommendations = await signal_generator.generate_recommendations(
            ohlcv_data=ohlcv_data,
            current_date=date.today(),
            portfolio_state=None,
        )

        logger.info(f"Generated {len(recommendations)} recommendations")

        # 6. Get news analysis for email (if not already fetched during signal generation)
        news_analysis = None
        if news_analyzer and config["research"].enabled:
            try:
                news_analysis = await news_analyzer.analyze_symbols(
                    symbols=symbols,
                    lookback_hours=config["research"].lookback_hours,
                )
                logger.info(f"Analyzed {news_analysis.total_articles if news_analysis else 0} news articles")
            except Exception as e:
                logger.warning(f"Failed to fetch news analysis for email: {e}. Continuing without news in email.")
                news_analysis = None

        # 7. Get market summary
        market_summary = get_market_summary(ohlcv_data)

        # 8. Send email with news analysis
        success = await email_service.send_daily_report(
            recommendations=recommendations,
            market_summary=market_summary,
            portfolio_summary=None,  # Could be added later
            news_digest=None,  # Legacy format, not used if news_analysis provided
            news_analysis=news_analysis,  # NEW: Pass news analysis
            date_obj=date.today(),
            tracking_store=signal_generator.tracker.store if signal_generator.tracker else None,
        )

        # Mark signals as delivered after successful email send
        if success and signal_generator.tracker:
            signal_generator.mark_signals_delivered(recommendations, method="email")

        # Get performance metrics for logging
        if signal_generator.tracker:
            try:
                calculator = PerformanceCalculator(signal_generator.tracker.store)
                rolling_metrics = calculator.calculate_rolling_metrics(window_days=30)

                logger.info(
                    "Rolling 30-day performance: "
                    f"Win Rate={rolling_metrics.get('win_rate', 0):.0%}, "
                    f"Expectancy={rolling_metrics.get('expectancy_r', 0):.2f}R"
                )
            except Exception as e:
                logger.warning(f"Failed to calculate rolling metrics: {e}")

        if success:
            articles_count = news_analysis.total_articles if news_analysis else 0
            logger.info(
                "Daily signals job completed: "
                f"{len(recommendations)} recommendations, "
                f"{articles_count} articles analyzed"
            )
        else:
            logger.error("Failed to send daily signals email")

    except Exception as e:
        logger.error(f"Daily signals job failed: {e}", exc_info=True)
        # Send error alert
        await send_error_alert(e)
        raise

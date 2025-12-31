"""Daily signals job for scheduled execution."""

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
from ...output.email.config import EmailConfig
from ...output.email.email_service import EmailService
from ...signals.config import SignalConfig
from ...signals.live_signal_generator import LiveSignalGenerator

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
        polygon_api_key=os.getenv("POLYGON_API_KEY"),
        alpha_vantage_api_key=os.getenv("ALPHA_VANTAGE_API_KEY"),
        cache_path=Path(os.getenv("DATA_CACHE_PATH", "data/cache")),
        cache_ttl_hours=int(os.getenv("CACHE_TTL_HOURS", "24")),
    )
    config_dict["data_pipeline"] = data_pipeline_config

    # Load signals config
    signal_config = SignalConfig(
        max_recommendations=int(os.getenv("MAX_RECOMMENDATIONS", "5")),
        min_conviction=os.getenv("MIN_CONVICTION", "MEDIUM"),
        technical_weight=float(os.getenv("TECHNICAL_WEIGHT", "1.0")),
        news_weight=float(os.getenv("NEWS_WEIGHT", "0.0")),
    )
    config_dict["signals"] = signal_config

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
            smtp_host=os.getenv("SMTP_HOST", "smtp.sendgrid.net"),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            smtp_user=os.getenv("SMTP_USER", "apikey"),
            smtp_password=os.getenv("SMTP_PASSWORD", os.getenv("SENDGRID_API_KEY", "")),
            from_email=os.getenv("FROM_EMAIL", "signals@yourdomain.com"),
            from_name=os.getenv("FROM_NAME", "Trading Assistant"),
            recipients=email_recipients,
        )

        email_service = EmailService(email_config)
        error_subject = f"Trading System Error - {date.today()}"
        error_body = f"""
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
    """Execute daily signal generation and email.

    Args:
        asset_class: Asset class to generate signals for ("equity" or "crypto")
    """
    logger.info(f"Starting daily signals job for {asset_class}")

    try:
        # 1. Load configuration
        config = load_config()

        # 2. Initialize components
        data_fetcher = LiveDataFetcher(config["data_pipeline"])
        signal_generator = LiveSignalGenerator(config["signals"])
        email_service = EmailService(config["email"])

        # 3. Get universe symbols
        universe_type = config["universe"].get(asset_class, "NASDAQ-100" if asset_class == "equity" else "crypto")
        symbols = load_universe(universe_type)

        if not symbols:
            logger.warning(f"No symbols found for {asset_class} universe: {universe_type}")
            return

        logger.info(f"Fetching data for {len(symbols)} symbols")

        # 4. Fetch data
        ohlcv_data = await data_fetcher.fetch_daily_data(
            symbols=symbols,
            asset_class=asset_class,
            lookback_days=252
        )

        if not ohlcv_data:
            logger.warning(f"No data fetched for {asset_class}")
            return

        logger.info(f"Fetched data for {len(ohlcv_data)} symbols")

        # 5. Generate signals
        recommendations = await signal_generator.generate(
            ohlcv_data=ohlcv_data,
            asset_class=asset_class,
            current_date=date.today()
        )

        logger.info(f"Generated {len(recommendations)} recommendations")

        # 6. Get market summary
        market_summary = get_market_summary(ohlcv_data)

        # 7. Send email
        # Note: send_daily_report is not async, so we'll use asyncio.to_thread
        import asyncio
        success = await asyncio.to_thread(
            email_service.send_daily_report,
            recommendations=recommendations,
            portfolio_summary=None,  # Could be added later
            news_digest=None,  # Could be added later
        )

        if success:
            logger.info(f"Daily signals job completed: {len(recommendations)} recommendations sent via email")
        else:
            logger.error("Failed to send daily signals email")

    except Exception as e:
        logger.error(f"Daily signals job failed: {e}", exc_info=True)
        # Send error alert
        await send_error_alert(e)
        raise


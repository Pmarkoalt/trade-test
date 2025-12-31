"""Email sending service."""

import asyncio
import smtplib
from datetime import date, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from jinja2 import Environment, FileSystemLoader

    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    # Create dummy objects for runtime when jinja2 is not available
    # These are only created if import fails, so no redefinition occurs at runtime
    # Mypy sees both code paths and flags redefinition, but runtime only has one path
    Environment: Any = type("Environment", (), {})  # type: ignore[no-redef]
    FileSystemLoader: Any = type("FileSystemLoader", (), {})  # type: ignore[no-redef]

from ...logging.logger import get_logger
from ..formatters.recommendation_formatter import RecommendationFormatter
from .config import EmailConfig

logger = get_logger(__name__)


class EmailService:
    """Service for sending email notifications."""

    jinja_env: Optional[Environment]

    def __init__(self, config: EmailConfig):
        """Initialize email service.

        Args:
            config: Email configuration
        """
        self.config = config
        self.formatter = RecommendationFormatter()

        # Setup Jinja2 environment
        if JINJA2_AVAILABLE:
            templates_dir = Path(__file__).parent / "templates"
            self.jinja_env: Optional[Environment] = Environment(
                loader=FileSystemLoader(str(templates_dir)),
                autoescape=True,
            )
            # Add formatter to global template context
            self.jinja_env.globals["formatter"] = self.formatter
        else:
            logger.warning("Jinja2 not available. Email templates will use simple string formatting.")
            self.jinja_env = None

    async def send_daily_report(
        self,
        recommendations: List,
        market_summary: Optional[Dict[str, Any]] = None,
        portfolio_summary: Optional[Dict[str, Any]] = None,
        news_digest: Optional[Dict[str, Any]] = None,
        news_analysis: Optional[Any] = None,  # NewsAnalysisResult type
        date_obj: Optional[date] = None,
        tracking_store: Optional[Any] = None,  # BaseTrackingStore type
    ) -> bool:
        """Send daily signal report email.

        Args:
            recommendations: List of Recommendation objects
            market_summary: Optional market summary data
            portfolio_summary: Optional portfolio summary data
            news_digest: Optional news digest data (legacy format)
            news_analysis: Optional NewsAnalysisResult object (new format)
            date_obj: Optional date (defaults to today)

        Returns:
            True if email sent successfully, False otherwise
        """
        try:
            date_obj = date_obj or date.today()

            # Render template
            if self.jinja_env:
                html_content = await self._render_template(
                    recommendations=recommendations,
                    market_summary=market_summary or {},
                    portfolio_summary=portfolio_summary,
                    news_digest=news_digest,
                    news_analysis=news_analysis,
                    date_obj=date_obj,
                    tracking_store=tracking_store,
                )
            else:
                # Fallback to simple HTML
                html_content = self._generate_simple_html(recommendations, date_obj)

            # Create message
            msg = MIMEMultipart("alternative")
            subject = f"Trading Signals for {date_obj.strftime('%b %d')} - {len(recommendations)} Recommendation{'s' if len(recommendations) != 1 else ''}"
            msg["Subject"] = subject
            msg["From"] = f"{self.config.from_name} <{self.config.from_email}>"
            msg["To"] = ", ".join(self.config.recipients)

            msg.attach(MIMEText(html_content, "html"))

            # Send via SMTP (run in thread pool since SMTP is blocking)
            return await self._send_smtp(msg)
        except Exception as e:
            logger.error(f"Failed to send daily report email: {e}", exc_info=True)
            return False

    async def _render_template(
        self,
        recommendations: List,
        market_summary: Dict[str, Any],
        portfolio_summary: Optional[Dict[str, Any]],
        news_digest: Optional[Dict[str, Any]],
        news_analysis: Optional[Any] = None,  # NewsAnalysisResult type
        date_obj: Optional[date] = None,
        tracking_store: Optional[Any] = None,  # BaseTrackingStore type
    ) -> str:
        """Render email template using Jinja2.

        Args:
            recommendations: List of Recommendation objects
            market_summary: Market summary data
            portfolio_summary: Optional portfolio summary
            news_digest: Optional news digest (legacy format)
            news_analysis: Optional NewsAnalysisResult object (new format)
            date_obj: Date for the report

        Returns:
            Rendered HTML string
        """
        if not self.jinja_env:
            raise RuntimeError("Jinja2 not available")

        date_obj = date_obj or date.today()

        # Get template
        template = self.jinja_env.get_template("daily_signals.html")

        # Prepare template context
        buy_signals = [r for r in recommendations if r.direction == "BUY"]
        sell_signals = [r for r in recommendations if r.direction == "SELL"]

        # Separate news by sentiment if news_analysis is provided
        positive_news = []
        negative_news = []

        if news_analysis and hasattr(news_analysis, "articles"):
            try:
                from ...data_pipeline.sources.news.models import SentimentLabel

                for article in news_analysis.articles:
                    if hasattr(article, "sentiment_label") and article.sentiment_label:
                        if article.sentiment_label in [SentimentLabel.POSITIVE, SentimentLabel.VERY_POSITIVE]:
                            positive_news.append(article)
                        elif article.sentiment_label in [SentimentLabel.NEGATIVE, SentimentLabel.VERY_NEGATIVE]:
                            negative_news.append(article)
            except (ImportError, AttributeError):
                # Fallback if SentimentLabel not available or different structure
                for article in news_analysis.articles:
                    if hasattr(article, "sentiment_label"):
                        label = str(article.sentiment_label).upper()
                        if "POSITIVE" in label:
                            positive_news.append(article)
                        elif "NEGATIVE" in label:
                            negative_news.append(article)

        # Add performance section if tracking available
        performance_context = None
        if tracking_store:
            try:
                from ...tracking.analytics.signal_analytics import SignalAnalyzer
                from ...tracking.performance_calculator import PerformanceCalculator

                calculator = PerformanceCalculator(tracking_store)
                metrics = calculator.calculate_rolling_metrics(window_days=30)

                # Get recent closed trades
                analyzer = SignalAnalyzer(tracking_store)
                analytics = analyzer.analyze()

                performance_context = {
                    "days": 30,
                    "metrics": metrics,
                    "recent_closed": analytics.last_10_trades[:5] if analytics.last_10_trades else [],
                    "streak": (
                        {
                            "count": analytics.current_streak,
                            "type": analytics.current_streak_type,
                        }
                        if analytics.current_streak >= 3
                        else None
                    ),
                }
            except Exception as e:
                logger.warning(f"Failed to generate performance context: {e}")

        context = {
            "recommendations": recommendations,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "market": market_summary,
            "portfolio": portfolio_summary,
            "news": news_digest,  # Legacy format
            "news_analysis": news_analysis,  # New format
            "positive_news": positive_news[:5],
            "negative_news": negative_news[:5],
            "date": date_obj.strftime("%B %d, %Y"),
            "date_short": date_obj.strftime("%Y-%m-%d"),
            "num_recommendations": len(recommendations),
            "generated_at": datetime.now().strftime("%I:%M %p ET"),
            "performance": performance_context,
        }

        # Render template (run in thread pool to avoid blocking)
        return await asyncio.to_thread(template.render, **context)

    async def send_test_email(self) -> bool:
        """Send a test email to verify configuration.

        Returns:
            True if test email sent successfully, False otherwise
        """
        try:
            logger.info("Sending test email...")

            # Create simple test message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = "Trading System - Test Email"
            msg["From"] = f"{self.config.from_name} <{self.config.from_email}>"
            msg["To"] = ", ".join(self.config.recipients)

            test_html = """
            <html>
            <body>
                <h1>Test Email</h1>
                <p>This is a test email from the Trading System.</p>
                <p>If you received this email, your email configuration is working correctly.</p>
                <hr>
                <p><small>Generated at: {}</small></p>
            </body>
            </html>
            """.format(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

            msg.attach(MIMEText(test_html, "html"))

            success = await self._send_smtp(msg)
            if success:
                logger.info(f"Test email sent successfully to {', '.join(self.config.recipients)}")
            else:
                logger.error("Failed to send test email")
            return success
        except Exception as e:
            logger.error(f"Failed to send test email: {e}", exc_info=True)
            return False

    async def _send_smtp(self, msg: MIMEMultipart) -> bool:
        """Send email via SMTP (runs in thread pool).

        Args:
            msg: MIMEMultipart message to send

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Run SMTP operations in thread pool since they're blocking
            return await asyncio.to_thread(self._send_smtp_sync, msg)
        except Exception as e:
            logger.error(f"SMTP error: {e}", exc_info=True)
            return False

    def _send_smtp_sync(self, msg: MIMEMultipart) -> bool:
        """Synchronous SMTP send (called from thread pool).

        Args:
            msg: MIMEMultipart message to send

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_user, self.config.smtp_password)
                server.send_message(msg)
            return True
        except Exception as e:
            logger.error(f"SMTP send error: {e}", exc_info=True)
            raise

    def _generate_simple_html(self, recommendations: List, date_obj: date) -> str:
        """Generate simple HTML fallback if Jinja2 is not available.

        Args:
            recommendations: List of Recommendation objects
            date_obj: Date for the report

        Returns:
            Simple HTML string
        """
        html = """
        <html>
        <head><title>Trading Signals for {date_obj}</title></head>
        <body>
            <h1>Daily Trading Signals - {date_obj}</h1>
            <h2>Recommendations ({len(recommendations)})</h2>
            <ul>
        """
        for rec in recommendations:
            html += f"<li>{rec.symbol} - {rec.direction} - {rec.conviction} conviction</li>"
        html += """
            </ul>
        </body>
        </html>
        """
        return html

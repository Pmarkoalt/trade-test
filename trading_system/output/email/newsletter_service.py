"""Newsletter service for multi-bucket daily trading signals.

This service handles the generation and delivery of daily newsletters
with signals organized by strategy buckets.
"""

import asyncio
from datetime import date
from typing import Any, Dict, List, Optional

from ...logging.logger import get_logger
from ...models.signals import Signal
from .config import EmailConfig
from .email_service import EmailService
from .newsletter_generator import NewsletterGenerator

logger = get_logger(__name__)


class NewsletterService:
    """Service for generating and sending daily newsletters."""

    def __init__(self, email_config: EmailConfig):
        """Initialize newsletter service.

        Args:
            email_config: Email configuration
        """
        self.email_config = email_config
        self.email_service = EmailService(email_config)
        self.newsletter_generator = NewsletterGenerator()

    async def send_daily_newsletter(
        self,
        signals_by_bucket: Dict[str, List[Signal]],
        market_summary: Optional[Dict[str, Any]] = None,
        news_analysis: Optional[Any] = None,
        portfolio_summary: Optional[Dict[str, Any]] = None,
        date_obj: Optional[date] = None,
    ) -> bool:
        """Send daily newsletter with multi-bucket signals.

        Args:
            signals_by_bucket: Dictionary mapping bucket name to list of Signal objects
            market_summary: Optional market summary data
            news_analysis: Optional news analysis result
            portfolio_summary: Optional portfolio summary
            date_obj: Optional date (defaults to today)

        Returns:
            True if newsletter sent successfully, False otherwise
        """
        try:
            date_obj = date_obj or date.today()

            # Generate newsletter context
            context = self.newsletter_generator.generate_newsletter_context(
                signals_by_bucket=signals_by_bucket,
                market_summary=market_summary,
                news_analysis=news_analysis,
                portfolio_summary=portfolio_summary,
                date_obj=date_obj,
            )

            # Render HTML using newsletter template
            if self.email_service.jinja_env:
                template = self.email_service.jinja_env.get_template("newsletter_daily.html")
                html_content = await asyncio.to_thread(template.render, **context)
            else:
                # Fallback to simple HTML
                html_content = self._generate_simple_html(signals_by_bucket, date_obj)

            # Generate plain text summary
            text_content = self.newsletter_generator.generate_plain_text_summary(
                signals_by_bucket=signals_by_bucket,
                date_obj=date_obj,
            )

            # Create and send email
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText

            msg = MIMEMultipart("alternative")
            subject = f"Daily Trading Newsletter - {date_obj.strftime('%b %d')} - {context['total_signals']} Signals"
            msg["Subject"] = subject
            msg["From"] = f"{self.email_config.from_name} <{self.email_config.from_email}>"
            msg["To"] = ", ".join(self.email_config.recipients)

            # Attach both plain text and HTML
            msg.attach(MIMEText(text_content, "plain"))
            msg.attach(MIMEText(html_content, "html"))

            # Send via SMTP
            success = await self.email_service._send_smtp(msg)

            if success:
                logger.info(
                    f"Newsletter sent successfully: {context['total_signals']} signals "
                    f"across {len(context['bucket_sections'])} buckets"
                )
            else:
                logger.error("Failed to send newsletter")

            return success

        except Exception as e:
            logger.error(f"Failed to send daily newsletter: {e}", exc_info=True)
            return False

    def _generate_simple_html(self, signals_by_bucket: Dict[str, List[Signal]], date_obj: date) -> str:
        """Generate simple HTML fallback if Jinja2 is not available.

        Args:
            signals_by_bucket: Dictionary mapping bucket name to list of Signal objects
            date_obj: Date for the newsletter

        Returns:
            Simple HTML string
        """
        html_parts = [
            "<html>",
            "<head><title>Daily Trading Newsletter</title></head>",
            "<body>",
            f"<h1>Daily Trading Newsletter - {date_obj.strftime('%B %d, %Y')}</h1>",
        ]

        total_signals = 0
        for bucket_name, signals in signals_by_bucket.items():
            valid_signals = [s for s in signals if s.is_valid()]
            if not valid_signals:
                continue

            total_signals += len(valid_signals)
            html_parts.append(f"<h2>{bucket_name.upper()}</h2>")
            html_parts.append(f"<p>{len(valid_signals)} signals</p>")
            html_parts.append("<ul>")

            for signal in valid_signals:
                html_parts.append(
                    f"<li>{signal.symbol} - {signal.side.value} @ ${signal.entry_price:.2f} "
                    f"(Stop: ${signal.stop_price:.2f})</li>"
                )

            html_parts.append("</ul>")

        html_parts.append(f"<p><strong>Total: {total_signals} signals</strong></p>")
        html_parts.append("</body>")
        html_parts.append("</html>")

        return "\n".join(html_parts)

    async def send_test_newsletter(self) -> bool:
        """Send a test newsletter to verify configuration.

        Returns:
            True if test newsletter sent successfully, False otherwise
        """
        try:
            logger.info("Sending test newsletter...")

            # Create mock signals for testing
            from ...models.signals import Signal, SignalSide, SignalType

            test_signals = {
                "safe_sp": [
                    Signal(
                        symbol="AAPL",
                        asset_class="equity",
                        date=date.today(),
                        side=SignalSide.BUY,
                        signal_type=SignalType.ENTRY_LONG,
                        trigger_reason="Test signal - momentum breakout",
                        entry_price=150.0,
                        stop_price=145.0,
                        score=0.85,
                        urgency=0.7,
                        adv20=1000000.0,
                    ),
                ],
                "crypto_topCap": [
                    Signal(
                        symbol="BTC",
                        asset_class="crypto",
                        date=date.today(),
                        side=SignalSide.BUY,
                        signal_type=SignalType.ENTRY_LONG,
                        trigger_reason="Test signal - crypto momentum",
                        entry_price=45000.0,
                        stop_price=43500.0,
                        score=0.78,
                        urgency=0.6,
                        adv20=5000000.0,
                    ),
                ],
            }

            test_market = {
                "spy_price": 450.0,
                "spy_pct": 0.5,
                "btc_price": 45000.0,
                "btc_pct": 1.2,
                "regime": "Bullish",
            }

            success = await self.send_daily_newsletter(
                signals_by_bucket=test_signals,
                market_summary=test_market,
                date_obj=date.today(),
            )

            if success:
                logger.info(f"Test newsletter sent successfully to {', '.join(self.email_config.recipients)}")
            else:
                logger.error("Failed to send test newsletter")

            return success

        except Exception as e:
            logger.error(f"Failed to send test newsletter: {e}", exc_info=True)
            return False

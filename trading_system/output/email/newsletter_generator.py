"""Newsletter generator for multi-bucket daily trading signals.

This module generates daily newsletters with signals organized by strategy buckets:
- Bucket A: Safe S&P bets (Equities)
- Bucket B: Aggressive top-cap crypto
- Bucket C: Low-float stock "gamble" (future)
- Bucket D: Unusual options movement (future)
"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from ...logging.logger import get_logger
from ...models.signals import Signal

logger = get_logger(__name__)


class NewsletterGenerator:
    """Generate newsletter content for daily trading signals."""

    def __init__(self):
        """Initialize newsletter generator."""
        pass

    def generate_newsletter_context(
        self,
        signals_by_bucket: Dict[str, List[Signal]],
        market_summary: Optional[Dict[str, Any]] = None,
        news_analysis: Optional[Any] = None,
        portfolio_summary: Optional[Dict[str, Any]] = None,
        date_obj: Optional[date] = None,
    ) -> Dict[str, Any]:
        """Generate newsletter context for template rendering.

        Args:
            signals_by_bucket: Dictionary mapping bucket name to list of Signal objects
            market_summary: Optional market summary data
            news_analysis: Optional news analysis result
            portfolio_summary: Optional portfolio summary
            date_obj: Optional date (defaults to today)

        Returns:
            Dictionary with newsletter context for template rendering
        """
        date_obj = date_obj or date.today()

        # Organize signals by bucket
        bucket_sections = []
        total_signals = 0

        for bucket_name, signals in signals_by_bucket.items():
            if not signals:
                continue

            # Filter valid signals
            valid_signals = [s for s in signals if s.is_valid()]
            total_signals += len(valid_signals)

            # Separate by side
            buy_signals = [s for s in valid_signals if s.side.value == "BUY"]
            sell_signals = [s for s in valid_signals if s.side.value == "SELL"]

            bucket_sections.append(
                {
                    "name": bucket_name,
                    "description": self._get_bucket_description(bucket_name),
                    "total_signals": len(valid_signals),
                    "buy_signals": buy_signals,
                    "sell_signals": sell_signals,
                    "asset_class": signals[0].asset_class if signals else "unknown",
                }
            )

        # Extract news by sentiment if available
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
                # Fallback if SentimentLabel not available
                for article in news_analysis.articles:
                    if hasattr(article, "sentiment_label"):
                        label = str(article.sentiment_label).upper()
                        if "POSITIVE" in label:
                            positive_news.append(article)
                        elif "NEGATIVE" in label:
                            negative_news.append(article)

        context = {
            "bucket_sections": bucket_sections,
            "total_signals": total_signals,
            "market": market_summary or {},
            "portfolio": portfolio_summary,
            "news_analysis": news_analysis,
            "positive_news": positive_news[:5],
            "negative_news": negative_news[:5],
            "date": date_obj.strftime("%B %d, %Y"),
            "date_short": date_obj.strftime("%Y-%m-%d"),
            "generated_at": datetime.now().strftime("%I:%M %p ET"),
        }

        return context

    def _get_bucket_description(self, bucket_name: str) -> str:
        """Get description for a strategy bucket.

        Args:
            bucket_name: Name of the bucket

        Returns:
            Human-readable description
        """
        descriptions = {
            "safe_sp": "Safe S&P 500 bets - Low drawdown, realistic capacity",
            "crypto_topCap": "Aggressive top market cap crypto - Higher turnover",
            "low_float": "Low-float stock gambles - Data-driven, strict risk controls",
            "unusual_options": "Unusual options movement - Follow/fade signals",
        }
        return descriptions.get(bucket_name, bucket_name)

    def format_signal_for_email(self, signal: Signal, index: int) -> Dict[str, Any]:
        """Format a signal for email display.

        Args:
            signal: Signal object
            index: Signal index (for numbering)

        Returns:
            Dictionary with formatted signal data
        """
        # Calculate position sizing info
        risk_amount = abs(signal.entry_price - signal.stop_price)
        risk_pct = (risk_amount / signal.entry_price) * 100 if signal.entry_price > 0 else 0

        # Extract rationale from metadata
        rationale = signal.trigger_reason
        if signal.metadata:
            technical_reason = signal.metadata.get("technical_reason", "")
            news_reason = signal.metadata.get("news_reason", "")
            if technical_reason:
                rationale = f"{rationale} - {technical_reason}"
            if news_reason:
                rationale = f"{rationale} | News: {news_reason}"

        return {
            "index": index,
            "symbol": signal.symbol,
            "side": signal.side.value,
            "signal_type": signal.signal_type.value,
            "entry_price": signal.entry_price,
            "stop_price": signal.stop_price,
            "risk_pct": risk_pct,
            "score": signal.score,
            "urgency": signal.urgency,
            "rationale": rationale,
            "asset_class": signal.asset_class,
            "breakout_strength": signal.breakout_strength,
            "momentum_strength": signal.momentum_strength,
            "diversification_bonus": signal.diversification_bonus,
        }

    def generate_plain_text_summary(
        self,
        signals_by_bucket: Dict[str, List[Signal]],
        date_obj: Optional[date] = None,
    ) -> str:
        """Generate plain text summary for email preview.

        Args:
            signals_by_bucket: Dictionary mapping bucket name to list of Signal objects
            date_obj: Optional date (defaults to today)

        Returns:
            Plain text summary string
        """
        date_obj = date_obj or date.today()
        lines = [
            f"Daily Trading Signals - {date_obj.strftime('%B %d, %Y')}",
            "=" * 60,
            "",
        ]

        total_signals = 0
        for bucket_name, signals in signals_by_bucket.items():
            valid_signals = [s for s in signals if s.is_valid()]
            if not valid_signals:
                continue

            total_signals += len(valid_signals)
            lines.append(f"{bucket_name.upper()}: {len(valid_signals)} signals")

            for i, signal in enumerate(valid_signals[:3], 1):
                lines.append(
                    f"  {i}. {signal.symbol} - {signal.side.value} @ ${signal.entry_price:.2f} "
                    f"(Stop: ${signal.stop_price:.2f})"
                )

            if len(valid_signals) > 3:
                lines.append(f"  ... and {len(valid_signals) - 3} more")
            lines.append("")

        lines.append(f"Total: {total_signals} signals")
        lines.append("")
        lines.append("View full details in the HTML email.")

        return "\n".join(lines)

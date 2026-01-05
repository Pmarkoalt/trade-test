"""Report content generator for emails."""

from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..formatters.recommendation_formatter import RecommendationFormatter

# Type hints for news analysis (avoid circular imports)
try:
    from ...data_pipeline.sources.news.models import SentimentLabel
except ImportError:
    # Fallback if news models not available
    SentimentLabel = Any  # type: ignore[assignment, misc]


class ReportGenerator:
    """Generate email report content."""

    def __init__(self):
        """Initialize report generator."""
        self.formatter = RecommendationFormatter()
        self.templates_dir = Path(__file__).parent / "templates"

    def generate_daily_signals_html(
        self,
        recommendations: List,
        portfolio_summary: Optional[dict] = None,
        news_digest: Optional[dict] = None,
    ) -> str:
        """Generate HTML content for daily signals report.

        Args:
            recommendations: List of Recommendation objects
            portfolio_summary: Optional portfolio summary data
            news_digest: Optional news digest data

        Returns:
            HTML string for email
        """
        # Load base template
        base_template_path = self.templates_dir / "base.html"
        daily_template_path = self.templates_dir / "daily_signals.html"
        styles_path = self.templates_dir / "styles.css"

        if not base_template_path.exists() or not daily_template_path.exists():
            # Fallback to simple HTML if templates don't exist
            return self._generate_simple_html(recommendations, portfolio_summary, news_digest)

        # Read templates and styles
        with open(base_template_path, "r") as f:
            base_template = f.read()

        with open(daily_template_path, "r") as f:
            daily_template = f.read()

        styles = ""
        if styles_path.exists():
            with open(styles_path, "r") as f:
                styles = f.read()

        # Format recommendations
        buy_signals = [r for r in recommendations if r.direction == "BUY"]
        sell_signals = [r for r in recommendations if r.direction == "SELL"]

        # Generate content sections
        buy_signals_html = ""
        if buy_signals:
            buy_signals_html = """
            <div class="section buy-signals">
                <h2>🎯 BUY SIGNALS</h2>
                {self._format_signals_section(buy_signals)}
            </div>
            """

        sell_signals_html = ""
        if sell_signals:
            sell_signals_html = """
            <div class="section sell-signals">
                <h2>📉 SELL SIGNALS</h2>
                {self._format_signals_section(sell_signals)}
            </div>
            """

        portfolio_html = ""
        if portfolio_summary:
            portfolio_html = """
            <div class="section portfolio">
                <h2>📈 CURRENT POSITIONS</h2>
                {self._format_portfolio_section(portfolio_summary)}
            </div>
            """

        news_html = ""
        if news_digest:
            news_html = """
            <div class="section news">
                <h2>📰 NEWS DIGEST</h2>
                {self._format_news_section(news_digest)}
            </div>
            """

        performance_html = ""
        if portfolio_summary:
            performance_html = """
            <div class="section performance">
                <h2>📊 PERFORMANCE (MTD)</h2>
                {self._format_performance_section(portfolio_summary)}
            </div>
            """

        # Replace template variables
        plural = "s" if len(recommendations) != 1 else ""
        content = daily_template.format(
            date=date.today().strftime("%Y-%m-%d"),
            num_recommendations=len(recommendations),
            plural=plural,
            buy_signals=buy_signals_html,
            sell_signals=sell_signals_html,
            portfolio=portfolio_html,
            news=news_html,
            performance=performance_html,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        # Insert into base template
        html = base_template.format(styles=styles, content=content)

        return html

    async def generate_daily_report(
        self,
        recommendations: List,
        market_summary: Dict[str, Any],
        news_analysis: Optional[Any] = None,  # NewsAnalysisResult type
        date_obj: Optional[date] = None,
    ) -> Dict[str, Any]:
        """Generate email report data with news analysis.

        Prepares data for template rendering. The actual HTML rendering
        is handled by email_service.py using Jinja2 templates.

        Args:
            recommendations: List of Recommendation objects
            market_summary: Market summary dictionary
            news_analysis: Optional NewsAnalysisResult object
            date_obj: Optional date (defaults to today)

        Returns:
            Dictionary with prepared data for template context
        """
        date_obj = date_obj or date.today()

        # Separate news by sentiment
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

        return {
            "recommendations": recommendations,
            "market": market_summary,
            "news_analysis": news_analysis,
            "positive_news": positive_news[:5],
            "negative_news": negative_news[:5],
            "date": date_obj.strftime("%B %d, %Y"),
            "generated_at": datetime.now().strftime("%I:%M %p ET"),
        }

    def _format_signals_section(self, signals: List) -> str:
        """Format signals into HTML.

        Args:
            signals: List of Recommendation objects

        Returns:
            HTML string for signals section
        """
        if not signals:
            return ""

        html_parts = []
        for i, rec in enumerate(signals, 1):
            html_parts.append(self.formatter.format_recommendation_html(rec, index=i))

        return "\n".join(html_parts)

    def _format_portfolio_section(self, portfolio_summary: dict) -> str:
        """Format portfolio summary into HTML.

        Args:
            portfolio_summary: Portfolio summary dictionary

        Returns:
            HTML string for portfolio section
        """
        positions = portfolio_summary.get("positions", [])
        if not positions:
            return ""

        rows = []
        for pos in positions:
            pos.get("symbol", "")
            pos.get("entry_price", 0)
            pos.get("current_price", 0)
            pnl_pct = pos.get("pnl_pct", 0)
            pos.get("days_held", 0)
            pos.get("status", "")

            pnl_class = "positive" if pnl_pct >= 0 else "negative"
            rows.append(
                """
                <tr>
                    <td>{symbol}</td>
                    <td>${entry:,.2f}</td>
                    <td>${current:,.2f}</td>
                    <td class="{pnl_class}">{pnl_pct:+.2f}%</td>
                    <td>{days}</td>
                    <td>{status}</td>
                </tr>
                """
            )

        return """
        <table class="portfolio-table">
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Entry</th>
                    <th>Current</th>
                    <th>P&L</th>
                    <th>Days</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """

    def _format_news_section(self, news_digest: dict) -> str:
        """Format news digest into HTML.

        Args:
            news_digest: News digest dictionary

        Returns:
            HTML string for news section
        """
        positive = news_digest.get("positive", [])
        negative = news_digest.get("negative", [])
        watch_list = news_digest.get("watch_list", [])

        html_parts = []

        if positive:
            html_parts.append("<div class='news-section positive'>")
            html_parts.append("<h4>🟢 POSITIVE SENTIMENT</h4>")
            html_parts.append("<ul>")
            for item in positive:
                html_parts.append(f"<li>{item}</li>")
            html_parts.append("</ul>")
            html_parts.append("</div>")

        if negative:
            html_parts.append("<div class='news-section negative'>")
            html_parts.append("<h4>🔴 NEGATIVE SENTIMENT</h4>")
            html_parts.append("<ul>")
            for item in negative:
                html_parts.append(f"<li>{item}</li>")
            html_parts.append("</ul>")
            html_parts.append("</div>")

        if watch_list:
            html_parts.append("<div class='news-section watch'>")
            html_parts.append("<h4>🟡 WATCH LIST</h4>")
            html_parts.append("<ul>")
            for item in watch_list:
                html_parts.append(f"<li>{item}</li>")
            html_parts.append("</ul>")
            html_parts.append("</div>")

        return "\n".join(html_parts) if html_parts else ""

    def _format_performance_section(self, portfolio_summary: dict) -> str:
        """Format performance metrics into HTML.

        Args:
            portfolio_summary: Portfolio summary dictionary

        Returns:
            HTML string for performance section
        """
        metrics = portfolio_summary.get("performance", {})
        if not metrics:
            return ""

        return """
        <div class="performance-section">
            <p><strong>Strategy Return:</strong> {metrics.get('strategy_return', 0):+.2f}%</p>
            <p><strong>Benchmark (SPY):</strong> {metrics.get('benchmark_return', 0):+.2f}%</p>
            <p><strong>Alpha:</strong> {metrics.get('alpha', 0):+.2f}%</p>
            <p><strong>Win Rate:</strong> {metrics.get('win_rate', 0):.0f}%</p>
            <p><strong>Avg Winner:</strong> {metrics.get('avg_winner', 0):+.2f}%</p>
            <p><strong>Avg Loser:</strong> {metrics.get('avg_loser', 0):+.2f}%</p>
        </div>
        """

    def _generate_simple_html(
        self,
        recommendations: List,
        portfolio_summary: Optional[dict],
        news_digest: Optional[dict],
    ) -> str:
        """Generate simple HTML fallback if templates are missing.

        Args:
            recommendations: List of Recommendation objects
            portfolio_summary: Optional portfolio summary
            news_digest: Optional news digest

        Returns:
            Simple HTML string
        """
        html = """
        <html>
        <head><title>Trading Signals for {date.today()}</title></head>
        <body>
            <h1>Daily Trading Signals - {date.today()}</h1>
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

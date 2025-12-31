"""Formatter for Recommendation objects."""


class RecommendationFormatter:
    """Format Recommendation objects for display."""

    def format_recommendation_html(self, recommendation, index: int = 1) -> str:
        """Format a recommendation as HTML.

        Args:
            recommendation: Recommendation object
            index: Index number for the recommendation

        Returns:
            HTML string for the recommendation
        """
        # Calculate percentages
        if recommendation.direction == "BUY":
            target_pct = ((recommendation.target_price - recommendation.entry_price) / recommendation.entry_price) * 100
            stop_pct = ((recommendation.stop_price - recommendation.entry_price) / recommendation.entry_price) * 100
        else:
            target_pct = ((recommendation.entry_price - recommendation.target_price) / recommendation.entry_price) * 100
            stop_pct = ((recommendation.entry_price - recommendation.stop_price) / recommendation.entry_price) * 100

        # Format conviction badge
        conviction_class = recommendation.conviction.lower().replace(" ", "-")
        conviction_badge = f'<span class="conviction-badge {conviction_class}">{recommendation.conviction} CONVICTION</span>'

        # Format news context and headlines
        news_context_html = ""
        if recommendation.news_reasoning:
            news_context_html = f'<div class="news-context"><span class="icon">📰</span> {recommendation.news_reasoning}</div>'

        news_headlines_html = ""
        if recommendation.news_headlines:
            news_headlines_html = '<div class="headlines"><strong>Recent News:</strong><ul>'
            for headline in recommendation.news_headlines[:2]:  # Limit to 2 headlines
                news_headlines_html += f"<li>{headline}</li>"
            news_headlines_html += "</ul></div>"

        # Format scores with news integration
        news_score_class = ""
        if recommendation.news_score is not None:
            if recommendation.news_score >= 7.0:
                news_score_class = "positive"
            elif recommendation.news_score <= 3.0:
                news_score_class = "negative"

        scores_html = """
        <div class="scores">
            <div class="score-item">
                <span class="label">Technical:</span>
                <span class="value">{recommendation.technical_score:.1f}/10</span>
            </div>
        """
        if recommendation.news_score is not None:
            scores_html += """
            <div class="score-item">
                <span class="label">News:</span>
                <span class="value {news_score_class}">{recommendation.news_score:.1f}/10</span>
            </div>
            """
        if recommendation.combined_score > 0:
            scores_html += """
            <div class="score-item combined">
                <span class="label">Combined:</span>
                <span class="value">{recommendation.combined_score:.1f}/10</span>
            </div>
            """
        scores_html += "</div>"

        # Get symbol name (simplified - in production, use a symbol-to-name mapping)
        symbol_name = recommendation.symbol

        html = """
        <div class="recommendation-card">
            <div class="recommendation-header">
                <h3>{index}. {recommendation.symbol} - {symbol_name} {conviction_badge}</h3>
            </div>
            <div class="recommendation-body">
                <p><strong>Signal Type:</strong> {recommendation.signal_type}</p>
                <div class="prices">
                    <p><strong>Entry:</strong> ${recommendation.entry_price:,.2f} (at open)</p>
                    <p><strong>Target:</strong> ${recommendation.target_price:,.2f} ({target_pct:+.1f}%)</p>
                    <p><strong>Stop:</strong> ${recommendation.stop_price:,.2f} ({stop_pct:+.1f}%)</p>
                    <p><strong>Size:</strong> {recommendation.position_size_pct:.1f}% of portfolio</p>
                </div>
                {scores_html}
                {news_context_html}
                {news_headlines_html}
                <div class="reasoning">
                    <p><strong>💡 Reasoning:</strong> {recommendation.reasoning}</p>
                </div>
            </div>
        </div>
        """

        return html

    def format_recommendation_text(self, recommendation, index: int = 1) -> str:
        """Format a recommendation as plain text.

        Args:
            recommendation: Recommendation object
            index: Index number for the recommendation

        Returns:
            Plain text string for the recommendation
        """
        # Calculate percentages
        if recommendation.direction == "BUY":
            target_pct = ((recommendation.target_price - recommendation.entry_price) / recommendation.entry_price) * 100
            stop_pct = ((recommendation.stop_price - recommendation.entry_price) / recommendation.entry_price) * 100
        else:
            target_pct = ((recommendation.entry_price - recommendation.target_price) / recommendation.entry_price) * 100
            stop_pct = ((recommendation.entry_price - recommendation.stop_price) / recommendation.entry_price) * 100

        text = """
{index}. {recommendation.symbol} - {recommendation.direction} [{recommendation.conviction} CONVICTION]
Signal Type: {recommendation.signal_type}

Entry:  ${recommendation.entry_price:,.2f} (at open)
Target: ${recommendation.target_price:,.2f} ({target_pct:+.1f}%)
Stop:   ${recommendation.stop_price:,.2f} ({stop_pct:+.1f}%)
Size:   {recommendation.position_size_pct:.1f}% of portfolio

Technical Score: {recommendation.technical_score:.1f}/10
"""
        if recommendation.news_score is not None:
            text += f"News Score: {recommendation.news_score:.1f}/10\n"
        if recommendation.combined_score > 0:
            text += f"Combined: {recommendation.combined_score:.1f}/10\n"

        if recommendation.news_headlines:
            text += "\n📰 Recent News:\n"
            for headline in recommendation.news_headlines[:3]:
                text += f'• "{headline}"\n'

        text += f"\n💡 Reasoning: {recommendation.reasoning}\n"

        return text

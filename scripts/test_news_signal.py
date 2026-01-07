#!/usr/bin/env python3
"""Test news sentiment signal for a single equity.

This script validates the live news sentiment pipeline by:
1. Fetching recent news articles for a symbol
2. Analyzing sentiment using VADER + financial lexicon
3. Computing a trading signal score (0-10)
4. Displaying results in a readable format

Usage:
    python scripts/test_news_signal.py AAPL
    python scripts/test_news_signal.py AAPL --lookback 72
    python scripts/test_news_signal.py AAPL --max-articles 20

Environment Variables Required:
    NEWSAPI_KEY: NewsAPI.org API key (free tier available)
    ALPHA_VANTAGE_API_KEY: Alpha Vantage API key (free tier available)

At least one API key is required. Both are recommended for better coverage.
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def _load_dotenv_if_present(path: str = ".env") -> None:
    """Load a local .env file into os.environ if present.

    This intentionally avoids adding a dependency on python-dotenv.
    Only sets variables that are not already present in os.environ.
    """
    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        # Best-effort: script can still rely on exported env vars
        return


def get_api_keys():
    """Get API keys from environment variables."""
    _load_dotenv_if_present()
    newsapi_key = os.getenv("NEWSAPI_KEY")
    alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    massive_key = os.getenv("MASSIVE_API_KEY")

    if not newsapi_key and not alpha_vantage_key and not massive_key:
        console.print(
            "[red]Error: No API keys found![/red]\n"
            "Please set at least one of:\n"
            "  - NEWSAPI_KEY (from newsapi.org)\n"
            "  - ALPHA_VANTAGE_API_KEY (from alphavantage.co)\n"
            "  - MASSIVE_API_KEY (Polygon key used by Massive)\n\n"
            "Example:\n"
            "  export NEWSAPI_KEY='your_key_here'\n"
            "  export ALPHA_VANTAGE_API_KEY='your_key_here'\n"
            "  export MASSIVE_API_KEY='your_key_here'"
        )
        sys.exit(1)

    return newsapi_key, alpha_vantage_key, massive_key


async def test_single_equity(
    symbol: str,
    lookback_hours: int = 48,
    max_articles: int = 10,
    verbose: bool = False,
):
    """Fetch and analyze news for a single symbol.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
        lookback_hours: How far back to search for news
        max_articles: Maximum articles to fetch per symbol
        verbose: Show detailed article information
    """
    from trading_system.research.config import ResearchConfig
    from trading_system.research.news_analyzer import NewsAnalyzer

    # Get API keys
    newsapi_key, alpha_vantage_key, massive_key = get_api_keys()

    # Show which sources are active
    sources = []
    if newsapi_key:
        sources.append("NewsAPI.org")
    if alpha_vantage_key:
        sources.append("Alpha Vantage")
    if massive_key:
        sources.append("Polygon (Massive)")

    console.print(f"\n[cyan]News Sources:[/cyan] {', '.join(sources)}")
    console.print(f"[cyan]Lookback:[/cyan] {lookback_hours} hours")
    console.print(f"[cyan]Max Articles:[/cyan] {max_articles}")

    # Create config
    config = ResearchConfig(
        newsapi_key=newsapi_key,
        alpha_vantage_key=alpha_vantage_key,
        massive_api_key=massive_key,
        lookback_hours=lookback_hours,
        max_articles_per_symbol=max_articles,
    )

    # Initialize analyzer
    try:
        analyzer = NewsAnalyzer(config)
    except ImportError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Install vaderSentiment: pip install vaderSentiment[/yellow]")
        sys.exit(1)

    # Fetch and analyze
    console.print(f"\n[yellow]Fetching news for {symbol}...[/yellow]")

    try:
        result = await analyzer.analyze_symbols(
            symbols=[symbol],
            lookback_hours=lookback_hours,
        )
    except Exception as e:
        console.print(f"[red]Error fetching news: {e}[/red]")
        sys.exit(1)

    # Display results
    display_results(symbol, result, analyzer, verbose)


def display_results(symbol: str, result, analyzer, verbose: bool):
    """Display analysis results in a formatted output."""

    console.print(f"\n{'='*60}")
    console.print(f"[bold cyan]NEWS SENTIMENT ANALYSIS: {symbol}[/bold cyan]")
    console.print(f"{'='*60}")

    summary = result.symbol_summaries.get(symbol)

    if not summary or summary.article_count == 0:
        console.print("[yellow]No recent news articles found.[/yellow]")
        console.print("\nPossible reasons:")
        console.print("  - Symbol may not have recent news coverage")
        console.print("  - API rate limits may be in effect")
        console.print("  - Try increasing --lookback hours")
        return

    # Main metrics table
    metrics_table = Table(show_header=False, box=None)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="white")

    # Color code sentiment
    sent_score = summary.avg_sentiment
    if sent_score >= 0.2:
        sent_color = "green"
    elif sent_score <= -0.2:
        sent_color = "red"
    else:
        sent_color = "yellow"

    metrics_table.add_row("Sentiment Score", f"[{sent_color}]{sent_score:.3f}[/{sent_color}]")
    metrics_table.add_row("Sentiment Label", f"[{sent_color}]{summary.sentiment_label.value}[/{sent_color}]")
    metrics_table.add_row("Articles Analyzed", str(summary.article_count))
    metrics_table.add_row("Positive Articles", f"[green]{summary.positive_count}[/green]")
    metrics_table.add_row("Negative Articles", f"[red]{summary.negative_count}[/red]")
    metrics_table.add_row("Neutral Articles", str(summary.neutral_count))
    metrics_table.add_row("Sentiment Trend", summary.sentiment_trend)

    if summary.most_recent_article:
        now = datetime.now(tz=summary.most_recent_article.tzinfo) if summary.most_recent_article.tzinfo else datetime.now()
        age = now - summary.most_recent_article
        hours_ago = age.total_seconds() / 3600
        metrics_table.add_row("Most Recent", f"{hours_ago:.1f} hours ago")

    console.print(Panel(metrics_table, title="Sentiment Metrics"))

    # Top headlines
    if summary.top_headlines:
        console.print("\n[bold]Top Headlines:[/bold]")
        for i, headline in enumerate(summary.top_headlines[:5], 1):
            # Truncate long headlines
            if len(headline) > 80:
                headline = headline[:77] + "..."
            console.print(f"  {i}. {headline}")

    # Trading signal score
    score, reasoning = analyzer.get_news_score_for_signal(symbol, result)

    # Color code score
    if score >= 7:
        score_color = "green"
        score_label = "BULLISH"
    elif score <= 3:
        score_color = "red"
        score_label = "BEARISH"
    else:
        score_color = "yellow"
        score_label = "NEUTRAL"

    console.print(f"\n[bold]Trading Signal Score:[/bold] [{score_color}]{score:.1f}/10 ({score_label})[/{score_color}]")
    console.print(f"[dim]Reasoning: {reasoning}[/dim]")

    # Show individual articles if verbose
    if verbose and result.articles:
        console.print(f"\n[bold]All Articles ({len(result.articles)}):[/bold]")

        articles_table = Table()
        articles_table.add_column("Source", style="cyan", width=15)
        articles_table.add_column("Title", width=50)
        articles_table.add_column("Sentiment", justify="right", width=10)
        articles_table.add_column("Published", width=12)

        symbol_articles = [a for a in result.articles if symbol in a.symbols]
        for article in symbol_articles[:15]:  # Limit to 15
            sent = article.sentiment_score or 0
            if sent >= 0.1:
                sent_str = f"[green]{sent:+.2f}[/green]"
            elif sent <= -0.1:
                sent_str = f"[red]{sent:+.2f}[/red]"
            else:
                sent_str = f"{sent:+.2f}"

            title = article.title[:47] + "..." if len(article.title) > 50 else article.title
            pub_time = article.published_at.strftime("%m/%d %H:%M") if article.published_at else "N/A"

            articles_table.add_row(
                article.source[:15] if article.source else "Unknown",
                title,
                sent_str,
                pub_time,
            )

        console.print(articles_table)

    # Market context
    console.print(f"\n[bold]Market Context:[/bold]")
    console.print(f"  Overall Market Sentiment: {result.market_sentiment:.3f} ({result.market_sentiment_label.value})")
    console.print(f"  Total Articles Processed: {result.total_articles}")


async def test_multiple_symbols(symbols: list, lookback_hours: int = 48):
    """Test multiple symbols and compare sentiment."""
    from trading_system.research.config import ResearchConfig
    from trading_system.research.news_analyzer import NewsAnalyzer

    newsapi_key, alpha_vantage_key, massive_key = get_api_keys()

    config = ResearchConfig(
        newsapi_key=newsapi_key,
        alpha_vantage_key=alpha_vantage_key,
        massive_api_key=massive_key,
        lookback_hours=lookback_hours,
    )

    try:
        analyzer = NewsAnalyzer(config)
    except ImportError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    console.print(f"\n[yellow]Analyzing {len(symbols)} symbols...[/yellow]")

    result = await analyzer.analyze_symbols(symbols=symbols, lookback_hours=lookback_hours)

    # Create comparison table
    table = Table(title="Multi-Symbol Sentiment Comparison")
    table.add_column("Symbol", style="cyan")
    table.add_column("Sentiment", justify="right")
    table.add_column("Label")
    table.add_column("Articles", justify="right")
    table.add_column("Trend")
    table.add_column("Signal Score", justify="right")

    for symbol in symbols:
        summary = result.symbol_summaries.get(symbol)
        if not summary:
            table.add_row(symbol, "N/A", "No data", "0", "-", "-")
            continue

        score, _ = analyzer.get_news_score_for_signal(symbol, result)

        # Color code
        sent = summary.avg_sentiment
        if sent >= 0.1:
            sent_str = f"[green]{sent:+.3f}[/green]"
        elif sent <= -0.1:
            sent_str = f"[red]{sent:+.3f}[/red]"
        else:
            sent_str = f"{sent:+.3f}"

        if score >= 7:
            score_str = f"[green]{score:.1f}[/green]"
        elif score <= 3:
            score_str = f"[red]{score:.1f}[/red]"
        else:
            score_str = f"{score:.1f}"

        table.add_row(
            symbol,
            sent_str,
            summary.sentiment_label.value,
            str(summary.article_count),
            summary.sentiment_trend,
            score_str,
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="Test news sentiment signal for equities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_news_signal.py AAPL
  python scripts/test_news_signal.py AAPL --lookback 72 --verbose
  python scripts/test_news_signal.py AAPL MSFT GOOGL --compare

Environment Variables:
  NEWSAPI_KEY           NewsAPI.org API key
  ALPHA_VANTAGE_API_KEY Alpha Vantage API key
        """,
    )
    parser.add_argument("symbols", nargs="+", help="Stock symbol(s) to analyze")
    parser.add_argument("--lookback", "-l", type=int, default=48, help="Hours to look back (default: 48)")
    parser.add_argument("--max-articles", "-m", type=int, default=10, help="Max articles per symbol (default: 10)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed article information")
    parser.add_argument("--compare", "-c", action="store_true", help="Compare multiple symbols in a table")

    args = parser.parse_args()

    # Normalize symbols to uppercase
    symbols = [s.upper() for s in args.symbols]

    if args.compare or len(symbols) > 1:
        asyncio.run(test_multiple_symbols(symbols, args.lookback))
    else:
        asyncio.run(
            test_single_equity(
                symbol=symbols[0],
                lookback_hours=args.lookback,
                max_articles=args.max_articles,
                verbose=args.verbose,
            )
        )


if __name__ == "__main__":
    main()

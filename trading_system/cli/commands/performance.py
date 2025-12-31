"""CLI commands for performance tracking."""

from datetime import date, timedelta

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from trading_system.tracking.analytics.signal_analytics import SignalAnalyzer
from trading_system.tracking.performance_calculator import PerformanceCalculator
from trading_system.tracking.reports.leaderboard import LeaderboardGenerator
from trading_system.tracking.storage.sqlite_store import SQLiteTrackingStore

console = Console()


def setup_parser(subparsers):
    """Set up performance CLI commands."""
    perf_parser = subparsers.add_parser(
        "performance",
        help="View performance metrics",
        aliases=["perf"],
    )

    perf_subparsers = perf_parser.add_subparsers(dest="perf_command")

    # Summary command
    summary_parser = perf_subparsers.add_parser(
        "summary",
        help="Show performance summary",
    )
    summary_parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to analyze (default: 30)",
    )
    summary_parser.add_argument(
        "--db",
        type=str,
        default="tracking.db",
        help="Path to tracking database",
    )

    # Leaderboard command
    lb_parser = perf_subparsers.add_parser(
        "leaderboard",
        help="Show strategy leaderboard",
        aliases=["lb"],
    )
    lb_parser.add_argument(
        "--period",
        choices=["weekly", "monthly", "all"],
        default="monthly",
        help="Time period (default: monthly)",
    )
    lb_parser.add_argument(
        "--db",
        type=str,
        default="tracking.db",
        help="Path to tracking database",
    )

    # Analytics command
    analytics_parser = perf_subparsers.add_parser(
        "analytics",
        help="Show detailed analytics",
    )
    analytics_parser.add_argument(
        "--db",
        type=str,
        default="tracking.db",
        help="Path to tracking database",
    )

    # Recent trades command
    recent_parser = perf_subparsers.add_parser(
        "recent",
        help="Show recent trades",
    )
    recent_parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of trades to show (default: 10)",
    )
    recent_parser.add_argument(
        "--db",
        type=str,
        default="tracking.db",
        help="Path to tracking database",
    )


def handle_command(args):
    """Handle performance commands."""
    if args.perf_command == "summary":
        show_summary(args.days, args.db)
    elif args.perf_command in ("leaderboard", "lb"):
        show_leaderboard(args.period, args.db)
    elif args.perf_command == "analytics":
        show_analytics(args.db)
    elif args.perf_command == "recent":
        show_recent(args.count, args.db)
    else:
        console.print("[yellow]Use --help to see available commands[/yellow]")


def show_summary(days: int, db_path: str):
    """Show performance summary."""
    store = SQLiteTrackingStore(db_path)
    calculator = PerformanceCalculator(store)

    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    metrics = calculator.calculate_metrics(
        start_date=start_date,
        end_date=end_date,
    )

    # Create summary table
    table = Table(
        title=f"Performance Summary ({days} days)",
        box=box.ROUNDED,
        show_header=False,
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    # Add rows
    table.add_row("Total Signals", str(metrics.total_signals))
    table.add_row("Trades Taken", str(metrics.signals_followed))
    table.add_row("", "")

    # Win/Loss
    win_style = "green" if metrics.win_rate >= 0.5 else "red"
    table.add_row("Win Rate", f"[{win_style}]{metrics.win_rate:.1%}[/{win_style}]")
    table.add_row("Winners", f"[green]{metrics.signals_won}[/green]")
    table.add_row("Losers", f"[red]{metrics.signals_lost}[/red]")
    table.add_row("", "")

    # Returns
    r_style = "green" if metrics.total_r > 0 else "red"
    table.add_row("Total Return", f"[{r_style}]{metrics.total_r:+.2f}R[/{r_style}]")
    table.add_row("Avg Return", f"{metrics.avg_r:+.2f}R")
    table.add_row("Expectancy", f"{metrics.expectancy_r:+.2f}R")
    table.add_row("", "")

    # Risk metrics
    table.add_row("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
    table.add_row("Max Drawdown", f"[red]{metrics.max_drawdown_pct:.1%}[/red]")

    console.print(table)
    store.close()


def show_leaderboard(period: str, db_path: str):
    """Show strategy leaderboard."""
    store = SQLiteTrackingStore(db_path)
    generator = LeaderboardGenerator(store)

    if period == "weekly":
        leaderboard = generator.generate_weekly()
    elif period == "monthly":
        leaderboard = generator.generate_monthly()
    else:
        leaderboard = generator.generate_all_time()

    # Create table
    table = Table(
        title=f"Strategy Leaderboard ({period.upper()})",
        box=box.ROUNDED,
    )
    table.add_column("Rank", justify="center", style="bold")
    table.add_column("Strategy", style="cyan")
    table.add_column("Total R", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Trend", justify="center")

    for entry in leaderboard.entries:
        # Trend symbol
        if entry.trend == "up":
            trend = f"[green]^ +{entry.rank_change}[/green]"
        elif entry.trend == "down":
            trend = f"[red]v {entry.rank_change}[/red]"
        else:
            trend = "[dim]-[/dim]"

        # R color
        r_style = "green" if entry.total_r > 0 else "red"

        table.add_row(
            f"#{entry.rank}",
            entry.display_name,
            f"[{r_style}]{entry.total_r:+.1f}R[/{r_style}]",
            f"{entry.win_rate:.0%}",
            str(entry.trade_count),
            trend,
        )

    console.print(table)

    # Summary panel
    summary_text = f"Profitable: {leaderboard.profitable_strategies}/{leaderboard.total_strategies}"
    if leaderboard.top_performer:
        summary_text += f"\nTop: {leaderboard.top_performer}"
    if leaderboard.needs_attention:
        summary_text += f"\n[yellow]Watch: {leaderboard.needs_attention}[/yellow]"

    console.print(Panel(summary_text, title="Summary", box=box.ROUNDED))
    store.close()


def show_analytics(db_path: str):
    """Show detailed analytics."""
    store = SQLiteTrackingStore(db_path)
    analyzer = SignalAnalyzer(store)

    analytics = analyzer.analyze()

    # Day of week table
    if analytics.performance_by_day_of_week:
        table = Table(title="Performance by Day", box=box.ROUNDED)
        table.add_column("Day", style="cyan")
        table.add_column("Trades", justify="right")
        table.add_column("Win Rate", justify="right")
        table.add_column("Avg R", justify="right")

        for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
            if day in analytics.performance_by_day_of_week:
                data = analytics.performance_by_day_of_week[day]
                r_style = "green" if data["avg_r"] > 0 else "red"
                table.add_row(
                    day,
                    str(data["total"]),
                    f"{data['win_rate']:.0%}",
                    f"[{r_style}]{data['avg_r']:+.2f}[/{r_style}]",
                )

        console.print(table)

    # Insights
    if analytics.insights:
        console.print("\n[bold]Insights:[/bold]")
        for insight in analytics.insights:
            if "WARNING" in insight:
                console.print(f"  [yellow]! {insight}[/yellow]")
            elif "Strong" in insight or "Best" in insight:
                console.print(f"  [green]+ {insight}[/green]")
            else:
                console.print(f"  - {insight}")

    store.close()


def show_recent(count: int, db_path: str):
    """Show recent trades."""
    store = SQLiteTrackingStore(db_path)
    analyzer = SignalAnalyzer(store)

    analytics = analyzer.analyze()

    if not analytics.last_10_trades:
        console.print("[yellow]No recent trades found[/yellow]")
        store.close()
        return

    table = Table(title=f"Recent Trades (Last {count})", box=box.ROUNDED)
    table.add_column("Symbol", style="bold")
    table.add_column("Direction")
    table.add_column("Result", justify="right")
    table.add_column("Exit Reason")
    table.add_column("Date")

    for trade in analytics.last_10_trades[:count]:
        r_style = "green" if trade["r_multiple"] > 0 else "red"

        table.add_row(
            trade["symbol"],
            trade["direction"],
            f"[{r_style}]{trade['r_multiple']:+.2f}R[/{r_style}]",
            trade.get("exit_reason", "Manual"),
            trade.get("exit_date", ""),
        )

    console.print(table)

    # Streak info
    if analytics.current_streak >= 3:
        streak_type = "winning" if analytics.current_streak_type == "win" else "losing"
        streak_color = "green" if analytics.current_streak_type == "win" else "red"
        console.print(
            f"\n[{streak_color}]Currently on a {analytics.current_streak}-trade " f"{streak_type} streak[/{streak_color}]"
        )

    store.close()

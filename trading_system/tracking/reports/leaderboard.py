"""Strategy leaderboard for ranking performance."""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List, Optional

from trading_system.tracking.analytics.strategy_analytics import StrategyAnalyzer
from trading_system.tracking.storage.base_store import BaseTrackingStore


@dataclass
class LeaderboardEntry:
    """Single entry in the leaderboard."""

    rank: int = 0
    previous_rank: int = 0
    rank_change: int = 0  # Positive = improved, negative = declined

    strategy_name: str = ""
    display_name: str = ""

    # Current period metrics
    total_r: float = 0.0
    win_rate: float = 0.0
    expectancy_r: float = 0.0
    trade_count: int = 0

    # Trend
    trend: str = ""  # "up", "down", "stable"
    trend_description: str = ""


@dataclass
class Leaderboard:
    """Strategy leaderboard."""

    period: str = ""  # "weekly", "monthly", "all_time"
    period_start: date = field(default_factory=date.today)
    period_end: date = field(default_factory=date.today)

    entries: List[LeaderboardEntry] = field(default_factory=list)

    # Summary
    total_strategies: int = 0
    profitable_strategies: int = 0

    # Top/Bottom
    top_performer: str = ""
    most_improved: str = ""
    needs_attention: str = ""


class LeaderboardGenerator:
    """
    Generate strategy leaderboards.

    Example:
        generator = LeaderboardGenerator(store)

        weekly = generator.generate_weekly()
        for entry in weekly.entries:
            print(f"#{entry.rank} {entry.strategy_name}: {entry.total_r:.1f}R")

        monthly = generator.generate_monthly()
        all_time = generator.generate_all_time()
    """

    # Display names for strategies
    STRATEGY_DISPLAY_NAMES = {
        "breakout_20d": "20-Day Breakout",
        "breakout_55d": "55-Day Breakout",
        "ma_crossover": "MA Crossover",
        "momentum": "Momentum",
        "mean_reversion": "Mean Reversion",
        "news_sentiment": "News Sentiment",
    }

    def __init__(self, store: BaseTrackingStore):
        """Initialize with storage backend."""
        self.store = store
        self.strategy_analyzer = StrategyAnalyzer(store)

    def generate_weekly(self) -> Leaderboard:
        """Generate weekly leaderboard."""
        end_date = date.today()
        start_date = end_date - timedelta(days=7)
        prev_start = start_date - timedelta(days=7)
        prev_end = start_date - timedelta(days=1)

        return self._generate_leaderboard(
            period="weekly",
            start_date=start_date,
            end_date=end_date,
            prev_start=prev_start,
            prev_end=prev_end,
        )

    def generate_monthly(self) -> Leaderboard:
        """Generate monthly leaderboard."""
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        prev_start = start_date - timedelta(days=30)
        prev_end = start_date - timedelta(days=1)

        return self._generate_leaderboard(
            period="monthly",
            start_date=start_date,
            end_date=end_date,
            prev_start=prev_start,
            prev_end=prev_end,
        )

    def generate_all_time(self) -> Leaderboard:
        """Generate all-time leaderboard."""
        return self._generate_leaderboard(
            period="all_time",
            start_date=date(2020, 1, 1),
            end_date=date.today(),
            prev_start=None,
            prev_end=None,
        )

    def _generate_leaderboard(
        self,
        period: str,
        start_date: date,
        end_date: date,
        prev_start: Optional[date],
        prev_end: Optional[date],
    ) -> Leaderboard:
        """Generate leaderboard for a period."""
        # Get current period metrics
        current = self.strategy_analyzer.compare_strategies(
            start_date=start_date,
            end_date=end_date,
        )

        # Get previous period metrics for comparison
        previous_ranks = {}
        if prev_start and prev_end:
            previous = self.strategy_analyzer.compare_strategies(
                start_date=prev_start,
                end_date=prev_end,
            )
            previous_ranks = {s.name: s.rank for s in previous.strategies}

        # Build entries
        entries = []
        for strategy in current.strategies:
            entry = LeaderboardEntry(
                rank=strategy.rank,
                previous_rank=previous_ranks.get(strategy.name, strategy.rank),
                strategy_name=strategy.name,
                display_name=self.STRATEGY_DISPLAY_NAMES.get(strategy.name, strategy.name.replace("_", " ").title()),
                total_r=strategy.total_r,
                win_rate=strategy.win_rate,
                expectancy_r=strategy.expectancy_r,
                trade_count=strategy.followed_signals,
            )

            # Calculate rank change
            entry.rank_change = entry.previous_rank - entry.rank

            # Determine trend
            if entry.rank_change > 0:
                entry.trend = "up"
                entry.trend_description = f"Up {entry.rank_change} spot(s)"
            elif entry.rank_change < 0:
                entry.trend = "down"
                entry.trend_description = f"Down {abs(entry.rank_change)} spot(s)"
            else:
                entry.trend = "stable"
                entry.trend_description = "No change"

            entries.append(entry)

        # Build leaderboard
        leaderboard = Leaderboard(
            period=period,
            period_start=start_date,
            period_end=end_date,
            entries=entries,
            total_strategies=len(entries),
            profitable_strategies=sum(1 for e in entries if e.total_r > 0),
        )

        # Identify notable entries
        if entries:
            leaderboard.top_performer = entries[0].strategy_name

            # Most improved (biggest positive rank change)
            improved = [e for e in entries if e.rank_change > 0]
            if improved:
                most_improved = max(improved, key=lambda e: e.rank_change)
                leaderboard.most_improved = most_improved.strategy_name

            # Needs attention (negative R with declining rank)
            declining = [e for e in entries if e.total_r < 0 and e.rank_change < 0]
            if declining:
                leaderboard.needs_attention = declining[0].strategy_name

        return leaderboard

    def format_leaderboard_text(self, leaderboard: Leaderboard) -> str:
        """Format leaderboard as text for CLI or logs."""
        lines = [
            f"=== Strategy Leaderboard ({leaderboard.period.upper()}) ===",
            f"Period: {leaderboard.period_start} to {leaderboard.period_end}",
            "",
            f"{'Rank':<6}{'Strategy':<25}{'Total R':<12}{'Win Rate':<12}{'Trades':<8}{'Trend':<10}",
            "-" * 75,
        ]

        for entry in leaderboard.entries:
            trend_symbol = {
                "up": "^",
                "down": "v",
                "stable": "-",
            }.get(entry.trend, "-")

            r_str = f"{entry.total_r:+.1f}R"
            wr_str = f"{entry.win_rate:.0%}"

            lines.append(
                f"#{entry.rank:<5}{entry.display_name:<25}{r_str:<12}{wr_str:<12}{entry.trade_count:<8}{trend_symbol}"
            )

        lines.extend(
            [
                "-" * 75,
                f"Profitable: {leaderboard.profitable_strategies}/{leaderboard.total_strategies}",
            ]
        )

        if leaderboard.top_performer:
            lines.append(f"Top Performer: {leaderboard.top_performer}")
        if leaderboard.most_improved:
            lines.append(f"Most Improved: {leaderboard.most_improved}")
        if leaderboard.needs_attention:
            lines.append(f"Needs Attention: {leaderboard.needs_attention}")

        return "\n".join(lines)

"""Signal-level analytics for performance insights."""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np

from trading_system.tracking.models import SignalOutcome, TrackedSignal
from trading_system.tracking.storage.base_store import BaseTrackingStore


@dataclass
class SignalAnalytics:
    """Analytics results for signals."""

    # Time-based analysis
    performance_by_day_of_week: Dict[str, Dict] = field(default_factory=dict)
    performance_by_month: Dict[str, Dict] = field(default_factory=dict)
    performance_by_holding_period: Dict[str, Dict] = field(default_factory=dict)

    # Score-based analysis
    performance_by_score_bucket: Dict[str, Dict] = field(default_factory=dict)
    score_vs_outcome_correlation: float = 0.0

    # Conviction analysis
    performance_by_conviction: Dict[str, Dict] = field(default_factory=dict)
    conviction_accuracy: Dict[str, float] = field(default_factory=dict)

    # Streaks
    current_streak: int = 0
    current_streak_type: str = ""  # "win" or "loss"
    max_win_streak: int = 0
    max_loss_streak: int = 0

    # Recent performance
    last_10_trades: List[Dict] = field(default_factory=list)
    recent_win_rate: float = 0.0

    # Insights
    insights: List[str] = field(default_factory=list)


class SignalAnalyzer:
    """
    Analyze signal performance patterns.

    Example:
        analyzer = SignalAnalyzer(store)
        analytics = analyzer.analyze()

        print(f"Best day: {analytics.performance_by_day_of_week}")
        for insight in analytics.insights:
            print(f"- {insight}")
    """

    DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def __init__(self, store: BaseTrackingStore):
        """Initialize analyzer with storage backend."""
        self.store = store

    def analyze(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> SignalAnalytics:
        """
        Run comprehensive signal analysis.

        Args:
            start_date: Analysis start date.
            end_date: Analysis end date.

        Returns:
            SignalAnalytics with all computed metrics.
        """
        # Get data
        signals = self.store.get_signals_by_date_range(
            start_date=start_date or date(2020, 1, 1),
            end_date=end_date or date.today(),
        )

        outcomes = []
        for signal in signals:
            outcome = self.store.get_outcome(signal.id)
            if outcome and outcome.was_followed:
                outcomes.append((signal, outcome))

        if not outcomes:
            return SignalAnalytics()

        analytics = SignalAnalytics()

        # Run analyses
        analytics.performance_by_day_of_week = self._analyze_by_day_of_week(outcomes)
        analytics.performance_by_month = self._analyze_by_month(outcomes)
        analytics.performance_by_holding_period = self._analyze_by_holding_period(outcomes)
        analytics.performance_by_score_bucket = self._analyze_by_score(outcomes)
        analytics.score_vs_outcome_correlation = self._calculate_score_correlation(outcomes)
        analytics.performance_by_conviction = self._analyze_by_conviction(outcomes)
        analytics.conviction_accuracy = self._calculate_conviction_accuracy(outcomes)

        streaks = self._calculate_streaks(outcomes)
        analytics.current_streak = streaks["current"]
        analytics.current_streak_type = streaks["current_type"]
        analytics.max_win_streak = streaks["max_win"]
        analytics.max_loss_streak = streaks["max_loss"]

        analytics.last_10_trades = self._get_recent_trades(outcomes, 10)
        analytics.recent_win_rate = self._calculate_recent_win_rate(outcomes, 20)

        analytics.insights = self._generate_insights(analytics)

        return analytics

    def _analyze_by_day_of_week(
        self,
        outcomes: List[Tuple[TrackedSignal, SignalOutcome]],
    ) -> Dict[str, Dict]:
        """Analyze performance by day signal was generated."""
        by_day: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"returns": [], "r_values": []})

        for signal, outcome in outcomes:
            day_idx = signal.created_at.weekday()
            day_name = self.DAY_NAMES[day_idx]
            by_day[day_name]["returns"].append(outcome.return_pct)
            by_day[day_name]["r_values"].append(outcome.r_multiple)

        result = {}
        for day, data in by_day.items():
            wins = sum(1 for r in data["returns"] if r > 0)
            total = len(data["returns"])
            result[day] = {
                "total": total,
                "win_rate": wins / total if total > 0 else 0,
                "avg_return": np.mean(data["returns"]) if data["returns"] else 0,
                "avg_r": np.mean(data["r_values"]) if data["r_values"] else 0,
            }

        return result

    def _analyze_by_month(
        self,
        outcomes: List[Tuple[TrackedSignal, SignalOutcome]],
    ) -> Dict[str, Dict]:
        """Analyze performance by month."""
        by_month: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"returns": [], "r_values": []})

        for signal, outcome in outcomes:
            month_idx = signal.created_at.month - 1
            month_name = self.MONTH_NAMES[month_idx]
            by_month[month_name]["returns"].append(outcome.return_pct)
            by_month[month_name]["r_values"].append(outcome.r_multiple)

        result = {}
        for month, data in by_month.items():
            wins = sum(1 for r in data["returns"] if r > 0)
            total = len(data["returns"])
            result[month] = {
                "total": total,
                "win_rate": wins / total if total > 0 else 0,
                "avg_return": np.mean(data["returns"]) if data["returns"] else 0,
                "avg_r": np.mean(data["r_values"]) if data["r_values"] else 0,
            }

        return result

    def _analyze_by_holding_period(
        self,
        outcomes: List[Tuple[TrackedSignal, SignalOutcome]],
    ) -> Dict[str, Dict]:
        """Analyze performance by holding period buckets."""
        buckets = {
            "1-3 days": (1, 3),
            "4-7 days": (4, 7),
            "8-14 days": (8, 14),
            "15-30 days": (15, 30),
            "30+ days": (31, 9999),
        }

        by_period: Dict[str, Dict[str, List[float]]] = {name: {"returns": [], "r_values": []} for name in buckets}

        for signal, outcome in outcomes:
            days = outcome.holding_days or 0
            for bucket_name, (min_days, max_days) in buckets.items():
                if min_days <= days <= max_days:
                    by_period[bucket_name]["returns"].append(outcome.return_pct)
                    by_period[bucket_name]["r_values"].append(outcome.r_multiple)
                    break

        result = {}
        for period, data in by_period.items():
            if not data["returns"]:
                continue
            wins = sum(1 for r in data["returns"] if r > 0)
            total = len(data["returns"])
            result[period] = {
                "total": total,
                "win_rate": wins / total if total > 0 else 0,
                "avg_return": np.mean(data["returns"]),
                "avg_r": np.mean(data["r_values"]),
            }

        return result

    def _analyze_by_score(
        self,
        outcomes: List[Tuple[TrackedSignal, SignalOutcome]],
    ) -> Dict[str, Dict]:
        """Analyze performance by combined score buckets."""
        buckets = {
            "0-5": (0, 5),
            "5-6": (5, 6),
            "6-7": (6, 7),
            "7-8": (7, 8),
            "8-9": (8, 9),
            "9-10": (9, 10),
        }

        by_score: Dict[str, Dict[str, List[float]]] = {name: {"returns": [], "r_values": []} for name in buckets}

        for signal, outcome in outcomes:
            score = signal.combined_score or 0
            for bucket_name, (min_score, max_score) in buckets.items():
                if min_score <= score < max_score:
                    by_score[bucket_name]["returns"].append(outcome.return_pct)
                    by_score[bucket_name]["r_values"].append(outcome.r_multiple)
                    break

        result = {}
        for score_range, data in by_score.items():
            if not data["returns"]:
                continue
            wins = sum(1 for r in data["returns"] if r > 0)
            total = len(data["returns"])
            result[score_range] = {
                "total": total,
                "win_rate": wins / total if total > 0 else 0,
                "avg_return": np.mean(data["returns"]),
                "avg_r": np.mean(data["r_values"]),
            }

        return result

    def _calculate_score_correlation(
        self,
        outcomes: List[Tuple[TrackedSignal, SignalOutcome]],
    ) -> float:
        """Calculate correlation between score and outcome."""
        if len(outcomes) < 5:
            return 0.0

        scores = [s.combined_score for s, _ in outcomes]
        returns = [o.return_pct for _, o in outcomes]

        if np.std(scores) == 0 or np.std(returns) == 0:
            return 0.0

        correlation = np.corrcoef(scores, returns)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0

    def _analyze_by_conviction(
        self,
        outcomes: List[Tuple[TrackedSignal, SignalOutcome]],
    ) -> Dict[str, Dict]:
        """Analyze performance by conviction level."""
        by_conviction: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"returns": [], "r_values": []})

        for signal, outcome in outcomes:
            conv = signal.conviction.value
            by_conviction[conv]["returns"].append(outcome.return_pct)
            by_conviction[conv]["r_values"].append(outcome.r_multiple)

        result = {}
        for conv, data in by_conviction.items():
            wins = sum(1 for r in data["returns"] if r > 0)
            total = len(data["returns"])
            result[conv] = {
                "total": total,
                "win_rate": wins / total if total > 0 else 0,
                "avg_return": np.mean(data["returns"]) if data["returns"] else 0,
                "avg_r": np.mean(data["r_values"]) if data["r_values"] else 0,
            }

        return result

    def _calculate_conviction_accuracy(
        self,
        outcomes: List[Tuple[TrackedSignal, SignalOutcome]],
    ) -> Dict[str, float]:
        """Calculate how accurate each conviction level is."""
        # HIGH conviction should have highest win rate
        by_conviction = self._analyze_by_conviction(outcomes)

        result = {}
        for conv, metrics in by_conviction.items():
            result[conv] = metrics["win_rate"]

        return result

    def _calculate_streaks(
        self,
        outcomes: List[Tuple[TrackedSignal, SignalOutcome]],
    ) -> Dict:
        """Calculate win/loss streaks."""
        # Sort by exit date
        sorted_outcomes = sorted(outcomes, key=lambda x: x[1].actual_exit_date or date.today())

        if not sorted_outcomes:
            return {"current": 0, "current_type": "", "max_win": 0, "max_loss": 0}

        max_win = 0
        max_loss = 0
        current = 0
        current_type = ""

        win_streak = 0
        loss_streak = 0

        for _, outcome in sorted_outcomes:
            is_win = outcome.return_pct > 0

            if is_win:
                win_streak += 1
                loss_streak = 0
                max_win = max(max_win, win_streak)
            else:
                loss_streak += 1
                win_streak = 0
                max_loss = max(max_loss, loss_streak)

        # Current streak
        if win_streak > 0:
            current = win_streak
            current_type = "win"
        elif loss_streak > 0:
            current = loss_streak
            current_type = "loss"

        return {
            "current": current,
            "current_type": current_type,
            "max_win": max_win,
            "max_loss": max_loss,
        }

    def _get_recent_trades(
        self,
        outcomes: List[Tuple[TrackedSignal, SignalOutcome]],
        n: int = 10,
    ) -> List[Dict]:
        """Get N most recent trades."""
        sorted_outcomes = sorted(
            outcomes,
            key=lambda x: x[1].actual_exit_date or date.today(),
            reverse=True,
        )

        result = []
        for signal, outcome in sorted_outcomes[:n]:
            result.append(
                {
                    "symbol": signal.symbol,
                    "direction": signal.direction.value,
                    "return_pct": outcome.return_pct,
                    "r_multiple": outcome.r_multiple,
                    "exit_date": outcome.actual_exit_date.isoformat() if outcome.actual_exit_date else None,
                    "exit_reason": outcome.exit_reason.value if outcome.exit_reason else None,
                }
            )

        return result

    def _calculate_recent_win_rate(
        self,
        outcomes: List[Tuple[TrackedSignal, SignalOutcome]],
        n: int = 20,
    ) -> float:
        """Calculate win rate for last N trades."""
        sorted_outcomes = sorted(
            outcomes,
            key=lambda x: x[1].actual_exit_date or date.today(),
            reverse=True,
        )

        recent = sorted_outcomes[:n]
        if not recent:
            return 0.0

        wins = sum(1 for _, o in recent if o.return_pct > 0)
        return wins / len(recent)

    def _generate_insights(self, analytics: SignalAnalytics) -> List[str]:
        """Generate actionable insights from analytics."""
        insights = []

        # Day of week insight
        if analytics.performance_by_day_of_week:
            best_day = max(analytics.performance_by_day_of_week.items(), key=lambda x: x[1].get("avg_r", 0))
            worst_day = min(analytics.performance_by_day_of_week.items(), key=lambda x: x[1].get("avg_r", 0))
            if best_day[1]["total"] >= 5:
                insights.append(
                    f"Best performance on {best_day[0]} "
                    f"(avg {best_day[1]['avg_r']:.2f}R, {best_day[1]['win_rate']:.0%} win rate)"
                )
            if worst_day[1]["total"] >= 5 and worst_day[1]["avg_r"] < 0:
                insights.append(f"Consider avoiding {worst_day[0]} signals " f"(avg {worst_day[1]['avg_r']:.2f}R)")

        # Conviction accuracy
        if analytics.conviction_accuracy:
            high_acc = analytics.conviction_accuracy.get("HIGH", 0)
            low_acc = analytics.conviction_accuracy.get("LOW", 0)
            if high_acc > low_acc + 0.1:
                insights.append("HIGH conviction signals outperform: " f"{high_acc:.0%} vs {low_acc:.0%} win rate")
            elif low_acc > high_acc:
                insights.append("WARNING: LOW conviction signals outperforming HIGH - " "review scoring methodology")

        # Score correlation
        if analytics.score_vs_outcome_correlation > 0.3:
            insights.append(f"Combined score is predictive (r={analytics.score_vs_outcome_correlation:.2f})")
        elif analytics.score_vs_outcome_correlation < 0:
            insights.append("WARNING: Negative score-outcome correlation - scoring needs review")

        # Streaks
        if analytics.current_streak >= 5:
            streak_type = "winning" if analytics.current_streak_type == "win" else "losing"
            insights.append(f"Currently on a {analytics.current_streak}-trade {streak_type} streak")

        # Recent performance
        if analytics.recent_win_rate < 0.4:
            insights.append(f"Recent performance declining: {analytics.recent_win_rate:.0%} win rate (last 20)")
        elif analytics.recent_win_rate > 0.7:
            insights.append(f"Strong recent performance: {analytics.recent_win_rate:.0%} win rate (last 20)")

        return insights

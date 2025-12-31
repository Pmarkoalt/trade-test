# Agent Tasks: Phase 3 - Performance Tracking (Part 2: Analytics & Reporting)

**Phase Goal**: Analytics, reporting, and integration for performance tracking
**Duration**: 1 week (Part 2)
**Prerequisites**: Phase 3 Part 1 complete (core tracking infrastructure)

---

## Phase 3 Part 2 Overview

### What We're Building
1. **Performance Analytics** - Detailed breakdowns and insights
2. **Email Templates** - Weekly summary and daily performance sections
3. **CLI Commands** - View performance from command line
4. **Strategy Leaderboard** - Rank strategies by performance
5. **Integration** - Connect tracking to signal generation pipeline

### Architecture Addition

```
trading_system/
├── tracking/
│   ├── analytics/                   # NEW: Analytics modules
│   │   ├── __init__.py
│   │   ├── signal_analytics.py      # Signal-level analysis
│   │   ├── strategy_analytics.py    # Strategy comparison
│   │   └── attribution.py           # Performance attribution
│   └── reports/                     # NEW: Report generation
│       ├── __init__.py
│       ├── performance_report.py    # Report generator
│       └── leaderboard.py           # Strategy rankings
│
├── output/
│   └── email/
│       └── templates/
│           ├── weekly_summary.html  # NEW: Weekly report
│           └── partials/
│               └── performance_section.html  # NEW: Daily section
│
└── cli/
    └── commands/
        └── performance.py           # NEW: CLI commands
```

---

## Task 3.2.1: Implement Signal Analytics

**Context**:
We need detailed analytics to understand which signals perform best under different conditions.

**Objective**:
Create signal analytics module for deep performance analysis.

**Files to Create**:
```
trading_system/tracking/analytics/
├── __init__.py
├── signal_analytics.py
```

**Requirements**:

1. Create `signal_analytics.py`:
```python
"""Signal-level analytics for performance insights."""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from trading_system.tracking.models import (
    ConvictionLevel,
    SignalOutcome,
    TrackedSignal,
)
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
        by_day = defaultdict(lambda: {"returns": [], "r_values": []})

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
        by_month = defaultdict(lambda: {"returns": [], "r_values": []})

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

        by_period = {name: {"returns": [], "r_values": []} for name in buckets}

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

        by_score = {name: {"returns": [], "r_values": []} for name in buckets}

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
        by_conviction = defaultdict(lambda: {"returns": [], "r_values": []})

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
        sorted_outcomes = sorted(
            outcomes,
            key=lambda x: x[1].actual_exit_date or date.today()
        )

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
            result.append({
                "symbol": signal.symbol,
                "direction": signal.direction.value,
                "return_pct": outcome.return_pct,
                "r_multiple": outcome.r_multiple,
                "exit_date": outcome.actual_exit_date.isoformat() if outcome.actual_exit_date else None,
                "exit_reason": outcome.exit_reason.value if outcome.exit_reason else None,
            })

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
            best_day = max(
                analytics.performance_by_day_of_week.items(),
                key=lambda x: x[1].get("avg_r", 0)
            )
            worst_day = min(
                analytics.performance_by_day_of_week.items(),
                key=lambda x: x[1].get("avg_r", 0)
            )
            if best_day[1]["total"] >= 5:
                insights.append(
                    f"Best performance on {best_day[0]} "
                    f"(avg {best_day[1]['avg_r']:.2f}R, {best_day[1]['win_rate']:.0%} win rate)"
                )
            if worst_day[1]["total"] >= 5 and worst_day[1]["avg_r"] < 0:
                insights.append(
                    f"Consider avoiding {worst_day[0]} signals "
                    f"(avg {worst_day[1]['avg_r']:.2f}R)"
                )

        # Conviction accuracy
        if analytics.conviction_accuracy:
            high_acc = analytics.conviction_accuracy.get("HIGH", 0)
            low_acc = analytics.conviction_accuracy.get("LOW", 0)
            if high_acc > low_acc + 0.1:
                insights.append(
                    f"HIGH conviction signals outperform: "
                    f"{high_acc:.0%} vs {low_acc:.0%} win rate"
                )
            elif low_acc > high_acc:
                insights.append(
                    "WARNING: LOW conviction signals outperforming HIGH - "
                    "review scoring methodology"
                )

        # Score correlation
        if analytics.score_vs_outcome_correlation > 0.3:
            insights.append(
                f"Combined score is predictive (r={analytics.score_vs_outcome_correlation:.2f})"
            )
        elif analytics.score_vs_outcome_correlation < 0:
            insights.append(
                "WARNING: Negative score-outcome correlation - scoring needs review"
            )

        # Streaks
        if analytics.current_streak >= 5:
            streak_type = "winning" if analytics.current_streak_type == "win" else "losing"
            insights.append(f"Currently on a {analytics.current_streak}-trade {streak_type} streak")

        # Recent performance
        if analytics.recent_win_rate < 0.4:
            insights.append(
                f"Recent performance declining: {analytics.recent_win_rate:.0%} win rate (last 20)"
            )
        elif analytics.recent_win_rate > 0.7:
            insights.append(
                f"Strong recent performance: {analytics.recent_win_rate:.0%} win rate (last 20)"
            )

        return insights
```

2. Create `analytics/__init__.py`:
```python
"""Analytics modules for performance tracking."""

from trading_system.tracking.analytics.signal_analytics import (
    SignalAnalytics,
    SignalAnalyzer,
)

__all__ = [
    "SignalAnalytics",
    "SignalAnalyzer",
]
```

**Acceptance Criteria**:
- [ ] Day of week analysis shows patterns
- [ ] Score correlation calculated correctly
- [ ] Streak tracking works
- [ ] Insights are actionable and clear
- [ ] All analytics handle empty data gracefully

**Tests to Write**:
```python
class TestSignalAnalyzer:
    def test_day_of_week_analysis(self):
        """Test grouping by day of week."""
        pass

    def test_streak_calculation(self):
        """Test win/loss streak calculation."""
        # Sequence: W, W, W, L, L, W
        # Max win streak = 3, current = 1 win
        pass

    def test_conviction_accuracy(self):
        """Test conviction level accuracy."""
        pass
```

---

## Task 3.2.2: Implement Strategy Analytics

**Context**:
Compare different signal types/strategies to understand which approaches work best.

**Objective**:
Create strategy-level analytics and comparison tools.

**Files to Create**:
```
trading_system/tracking/analytics/strategy_analytics.py
```

**Requirements**:

```python
"""Strategy-level analytics for comparing signal types."""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from trading_system.tracking.models import SignalOutcome, TrackedSignal
from trading_system.tracking.storage.base_store import BaseTrackingStore


@dataclass
class StrategyMetrics:
    """Metrics for a single strategy."""

    name: str = ""
    total_signals: int = 0
    followed_signals: int = 0
    wins: int = 0
    losses: int = 0

    win_rate: float = 0.0
    avg_return_pct: float = 0.0
    total_return_pct: float = 0.0
    avg_r: float = 0.0
    total_r: float = 0.0
    expectancy_r: float = 0.0

    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0

    avg_holding_days: float = 0.0
    avg_winner_r: float = 0.0
    avg_loser_r: float = 0.0

    # Ranking
    rank: int = 0
    score: float = 0.0


@dataclass
class StrategyComparison:
    """Comparison results for multiple strategies."""

    strategies: List[StrategyMetrics] = field(default_factory=list)
    best_strategy: str = ""
    worst_strategy: str = ""

    # Correlations
    strategy_correlations: Dict[Tuple[str, str], float] = field(default_factory=dict)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)


class StrategyAnalyzer:
    """
    Analyze and compare strategy performance.

    Example:
        analyzer = StrategyAnalyzer(store)
        comparison = analyzer.compare_strategies()

        for strategy in comparison.strategies:
            print(f"{strategy.name}: {strategy.win_rate:.0%} win rate, {strategy.expectancy_r:.2f}R expectancy")
    """

    # Minimum trades for valid comparison
    MIN_TRADES = 10

    def __init__(self, store: BaseTrackingStore):
        """Initialize with storage backend."""
        self.store = store

    def compare_strategies(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> StrategyComparison:
        """
        Compare all strategies by performance.

        Args:
            start_date: Analysis start date.
            end_date: Analysis end date.

        Returns:
            StrategyComparison with ranked strategies.
        """
        # Get all signals and outcomes
        signals = self.store.get_signals_by_date_range(
            start_date=start_date or date(2020, 1, 1),
            end_date=end_date or date.today(),
        )

        # Group by signal type
        by_strategy = {}
        for signal in signals:
            strategy = signal.signal_type
            if strategy not in by_strategy:
                by_strategy[strategy] = []

            outcome = self.store.get_outcome(signal.id)
            by_strategy[strategy].append((signal, outcome))

        # Calculate metrics for each strategy
        strategy_metrics = []
        for strategy_name, data in by_strategy.items():
            metrics = self._calculate_strategy_metrics(strategy_name, data)
            if metrics.followed_signals >= self.MIN_TRADES:
                strategy_metrics.append(metrics)

        # Rank strategies
        strategy_metrics = self._rank_strategies(strategy_metrics)

        # Build comparison
        comparison = StrategyComparison(strategies=strategy_metrics)

        if strategy_metrics:
            comparison.best_strategy = strategy_metrics[0].name
            comparison.worst_strategy = strategy_metrics[-1].name

        comparison.strategy_correlations = self._calculate_correlations(by_strategy)
        comparison.recommendations = self._generate_recommendations(comparison)

        return comparison

    def get_strategy_metrics(
        self,
        strategy_name: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Optional[StrategyMetrics]:
        """Get detailed metrics for a specific strategy."""
        signals = self.store.get_signals_by_date_range(
            start_date=start_date or date(2020, 1, 1),
            end_date=end_date or date.today(),
        )

        # Filter by strategy
        strategy_signals = [s for s in signals if s.signal_type == strategy_name]

        if not strategy_signals:
            return None

        data = []
        for signal in strategy_signals:
            outcome = self.store.get_outcome(signal.id)
            data.append((signal, outcome))

        return self._calculate_strategy_metrics(strategy_name, data)

    def _calculate_strategy_metrics(
        self,
        name: str,
        data: List[Tuple[TrackedSignal, Optional[SignalOutcome]]],
    ) -> StrategyMetrics:
        """Calculate comprehensive metrics for a strategy."""
        metrics = StrategyMetrics(name=name)
        metrics.total_signals = len(data)

        # Filter to followed trades with outcomes
        followed = [(s, o) for s, o in data if o and o.was_followed]
        metrics.followed_signals = len(followed)

        if not followed:
            return metrics

        # Basic counts
        returns = [o.return_pct for _, o in followed]
        r_values = [o.r_multiple for _, o in followed]

        winners = [(s, o) for s, o in followed if o.return_pct > 0]
        losers = [(s, o) for s, o in followed if o.return_pct <= 0]

        metrics.wins = len(winners)
        metrics.losses = len(losers)

        # Rates
        metrics.win_rate = metrics.wins / len(followed) if followed else 0

        # Returns
        metrics.avg_return_pct = np.mean(returns)
        metrics.total_return_pct = sum(returns)
        metrics.avg_r = np.mean(r_values)
        metrics.total_r = sum(r_values)

        # Winner/loser stats
        if winners:
            metrics.avg_winner_r = np.mean([o.r_multiple for _, o in winners])
        if losers:
            metrics.avg_loser_r = np.mean([o.r_multiple for _, o in losers])

        # Expectancy
        loss_rate = 1 - metrics.win_rate
        metrics.expectancy_r = (
            (metrics.win_rate * metrics.avg_winner_r) -
            (loss_rate * abs(metrics.avg_loser_r))
        )

        # Sharpe
        if len(returns) > 1:
            std = np.std(returns, ddof=1)
            if std > 0:
                metrics.sharpe_ratio = (np.mean(returns) / std) * np.sqrt(252)

        # Max drawdown
        metrics.max_drawdown_pct = self._calculate_max_drawdown(returns)

        # Profit factor
        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r < 0))
        if gross_loss > 0:
            metrics.profit_factor = gross_profit / gross_loss

        # Holding period
        holding_days = [o.holding_days for _, o in followed if o.holding_days]
        if holding_days:
            metrics.avg_holding_days = np.mean(holding_days)

        return metrics

    def _rank_strategies(
        self,
        strategies: List[StrategyMetrics],
    ) -> List[StrategyMetrics]:
        """Rank strategies by composite score."""
        for strategy in strategies:
            # Composite score: weighted combination of metrics
            # Higher is better
            strategy.score = (
                strategy.expectancy_r * 2.0 +      # Expectancy most important
                strategy.win_rate * 1.0 +          # Win rate matters
                strategy.sharpe_ratio * 0.5 +      # Risk-adjusted return
                (1 + strategy.max_drawdown_pct) * 0.5  # Drawdown penalty
            )

        # Sort by score descending
        strategies.sort(key=lambda x: x.score, reverse=True)

        # Assign ranks
        for i, strategy in enumerate(strategies):
            strategy.rank = i + 1

        return strategies

    def _calculate_correlations(
        self,
        by_strategy: Dict[str, List],
    ) -> Dict[Tuple[str, str], float]:
        """Calculate return correlations between strategies."""
        correlations = {}

        strategy_returns = {}
        for name, data in by_strategy.items():
            returns = [o.return_pct for _, o in data if o and o.was_followed]
            if len(returns) >= self.MIN_TRADES:
                strategy_returns[name] = returns

        names = list(strategy_returns.keys())
        for i, name1 in enumerate(names):
            for name2 in names[i + 1:]:
                # Align returns by index (simplified)
                r1 = strategy_returns[name1]
                r2 = strategy_returns[name2]
                min_len = min(len(r1), len(r2))

                if min_len >= 5:
                    corr = np.corrcoef(r1[:min_len], r2[:min_len])[0, 1]
                    if not np.isnan(corr):
                        correlations[(name1, name2)] = corr

        return correlations

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate max drawdown from returns."""
        if not returns:
            return 0.0

        equity = [1.0]
        for r in returns:
            equity.append(equity[-1] * (1 + r))

        peak = equity[0]
        max_dd = 0.0

        for e in equity:
            peak = max(peak, e)
            dd = (e - peak) / peak
            max_dd = min(max_dd, dd)

        return max_dd

    def _generate_recommendations(
        self,
        comparison: StrategyComparison,
    ) -> List[str]:
        """Generate strategic recommendations."""
        recommendations = []

        if not comparison.strategies:
            return ["Insufficient data for recommendations"]

        best = comparison.strategies[0]
        worst = comparison.strategies[-1]

        # Best strategy recommendation
        recommendations.append(
            f"Focus on {best.name}: {best.expectancy_r:.2f}R expectancy, "
            f"{best.win_rate:.0%} win rate"
        )

        # Worst strategy warning
        if worst.expectancy_r < 0:
            recommendations.append(
                f"Consider disabling {worst.name}: negative expectancy "
                f"({worst.expectancy_r:.2f}R)"
            )

        # Correlation warning
        for (s1, s2), corr in comparison.strategy_correlations.items():
            if corr > 0.7:
                recommendations.append(
                    f"{s1} and {s2} are highly correlated ({corr:.2f}) - "
                    "consider using only one"
                )

        # Diversification suggestion
        low_corr_pairs = [
            (s1, s2) for (s1, s2), corr in comparison.strategy_correlations.items()
            if corr < 0.3
        ]
        if low_corr_pairs:
            recommendations.append(
                f"Diversify with uncorrelated strategies: {low_corr_pairs[0]}"
            )

        return recommendations
```

**Acceptance Criteria**:
- [ ] Strategy comparison ranks by composite score
- [ ] Correlation between strategies calculated
- [ ] Recommendations are actionable
- [ ] Handles strategies with insufficient data

---

## Task 3.2.3: Create Weekly Performance Email Template

**Context**:
Users need a weekly summary email showing overall performance.

**Objective**:
Create Jinja2 HTML template for weekly performance report.

**Files to Create**:
```
trading_system/output/email/templates/weekly_summary.html
```

**Requirements**:

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Weekly Performance Summary</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 24px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 16px;
            margin-bottom: 24px;
        }
        .header h1 {
            margin: 0;
            color: #1a1a1a;
            font-size: 24px;
        }
        .header .date-range {
            color: #666;
            font-size: 14px;
            margin-top: 8px;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }
        .metric-card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }
        .metric-card.highlight {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .metric-value {
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 4px;
        }
        .metric-label {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            opacity: 0.8;
        }
        .positive { color: #22c55e; }
        .negative { color: #ef4444; }
        .neutral { color: #666; }
        .section {
            margin-bottom: 24px;
        }
        .section-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid #e0e0e0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }
        th, td {
            padding: 10px 8px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }
        th {
            background: #f8f9fa;
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
        }
        .trade-row.winner { background: rgba(34, 197, 94, 0.05); }
        .trade-row.loser { background: rgba(239, 68, 68, 0.05); }
        .insight-box {
            background: #fffbeb;
            border-left: 4px solid #f59e0b;
            padding: 12px 16px;
            margin-bottom: 12px;
            border-radius: 0 4px 4px 0;
        }
        .insight-box.success {
            background: #f0fdf4;
            border-left-color: #22c55e;
        }
        .insight-box.warning {
            background: #fef2f2;
            border-left-color: #ef4444;
        }
        .footer {
            text-align: center;
            padding-top: 24px;
            border-top: 1px solid #e0e0e0;
            color: #666;
            font-size: 12px;
        }
        .strategy-bar {
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 4px;
        }
        .strategy-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Weekly Performance Summary</h1>
            <div class="date-range">{{ period_start }} - {{ period_end }}</div>
        </div>

        <!-- Key Metrics Grid -->
        <div class="metric-grid">
            <div class="metric-card highlight">
                <div class="metric-value">{{ "%.1f"|format(metrics.total_r) }}R</div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "%.0f"|format(metrics.win_rate * 100) }}%</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ metrics.signals_followed }}</div>
                <div class="metric-label">Trades Closed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ "%.2f"|format(metrics.expectancy_r) }}R</div>
                <div class="metric-label">Expectancy</div>
            </div>
        </div>

        <!-- Performance Summary -->
        <div class="section">
            <div class="section-title">Performance Breakdown</div>
            <table>
                <tr>
                    <td>Winners</td>
                    <td class="positive">{{ metrics.signals_won }} (avg +{{ "%.2f"|format(metrics.avg_winner_r) }}R)</td>
                </tr>
                <tr>
                    <td>Losers</td>
                    <td class="negative">{{ metrics.signals_lost }} (avg {{ "%.2f"|format(metrics.avg_loser_r) }}R)</td>
                </tr>
                <tr>
                    <td>Sharpe Ratio</td>
                    <td>{{ "%.2f"|format(metrics.sharpe_ratio) }}</td>
                </tr>
                <tr>
                    <td>Max Drawdown</td>
                    <td class="{% if metrics.max_drawdown_pct < -0.1 %}negative{% else %}neutral{% endif %}">
                        {{ "%.1f"|format(metrics.max_drawdown_pct * 100) }}%
                    </td>
                </tr>
                <tr>
                    <td>vs Benchmark</td>
                    <td class="{% if metrics.alpha > 0 %}positive{% else %}negative{% endif %}">
                        {% if metrics.alpha > 0 %}+{% endif %}{{ "%.2f"|format(metrics.alpha * 100) }}% alpha
                    </td>
                </tr>
            </table>
        </div>

        <!-- Strategy Performance -->
        {% if strategies %}
        <div class="section">
            <div class="section-title">Strategy Performance</div>
            <table>
                <thead>
                    <tr>
                        <th>Strategy</th>
                        <th>Trades</th>
                        <th>Win Rate</th>
                        <th>Total R</th>
                    </tr>
                </thead>
                <tbody>
                    {% for strategy in strategies %}
                    <tr>
                        <td>{{ strategy.name }}</td>
                        <td>{{ strategy.followed_signals }}</td>
                        <td>{{ "%.0f"|format(strategy.win_rate * 100) }}%</td>
                        <td class="{% if strategy.total_r > 0 %}positive{% else %}negative{% endif %}">
                            {% if strategy.total_r > 0 %}+{% endif %}{{ "%.1f"|format(strategy.total_r) }}R
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}

        <!-- Recent Trades -->
        {% if recent_trades %}
        <div class="section">
            <div class="section-title">Recent Trades</div>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Direction</th>
                        <th>Result</th>
                        <th>Exit</th>
                    </tr>
                </thead>
                <tbody>
                    {% for trade in recent_trades %}
                    <tr class="trade-row {% if trade.r_multiple > 0 %}winner{% else %}loser{% endif %}">
                        <td><strong>{{ trade.symbol }}</strong></td>
                        <td>{{ trade.direction }}</td>
                        <td class="{% if trade.r_multiple > 0 %}positive{% else %}negative{% endif %}">
                            {% if trade.r_multiple > 0 %}+{% endif %}{{ "%.2f"|format(trade.r_multiple) }}R
                        </td>
                        <td>{{ trade.exit_reason or 'Manual' }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}

        <!-- Insights -->
        {% if insights %}
        <div class="section">
            <div class="section-title">Insights & Recommendations</div>
            {% for insight in insights %}
            <div class="insight-box {% if 'WARNING' in insight %}warning{% elif 'Strong' in insight or 'Best' in insight %}success{% endif %}">
                {{ insight }}
            </div>
            {% endfor %}
        </div>
        {% endif %}

        <!-- Cumulative Performance -->
        <div class="section">
            <div class="section-title">Cumulative Performance (All Time)</div>
            <table>
                <tr>
                    <td>Total Signals Generated</td>
                    <td>{{ cumulative.total_signals }}</td>
                </tr>
                <tr>
                    <td>Total Trades Taken</td>
                    <td>{{ cumulative.total_followed }}</td>
                </tr>
                <tr>
                    <td>Overall Win Rate</td>
                    <td>{{ "%.1f"|format(cumulative.win_rate * 100) }}%</td>
                </tr>
                <tr>
                    <td>Total Return</td>
                    <td class="{% if cumulative.total_r > 0 %}positive{% else %}negative{% endif %}">
                        {% if cumulative.total_r > 0 %}+{% endif %}{{ "%.1f"|format(cumulative.total_r) }}R
                    </td>
                </tr>
            </table>
        </div>

        <div class="footer">
            <p>Generated by Trading Assistant</p>
            <p>{{ generated_at }}</p>
        </div>
    </div>
</body>
</html>
```

**Acceptance Criteria**:
- [ ] Template renders with all metrics
- [ ] Color coding for positive/negative values
- [ ] Responsive design for mobile
- [ ] Strategy comparison table works
- [ ] Insights section displays properly

---

## Task 3.2.4: Create Performance Section for Daily Email

**Context**:
The daily signal email needs a performance summary section.

**Objective**:
Create a partial template for daily email performance section.

**Files to Create**:
```
trading_system/output/email/templates/partials/performance_section.html
```

**Requirements**:

```html
<!-- Performance Section for Daily Email -->
<div class="performance-section" style="margin-top: 24px; padding-top: 24px; border-top: 2px solid #e0e0e0;">
    <h2 style="font-size: 18px; margin-bottom: 16px; color: #1a1a1a;">
        Performance (Last {{ days }} Days)
    </h2>

    <!-- Quick Stats Row -->
    <table style="width: 100%; margin-bottom: 16px;">
        <tr>
            <td style="text-align: center; padding: 12px; background: #f8f9fa; border-radius: 8px;">
                <div style="font-size: 24px; font-weight: bold; color: {% if metrics.total_r > 0 %}#22c55e{% else %}#ef4444{% endif %};">
                    {% if metrics.total_r > 0 %}+{% endif %}{{ "%.1f"|format(metrics.total_r) }}R
                </div>
                <div style="font-size: 11px; color: #666; text-transform: uppercase;">Return</div>
            </td>
            <td style="width: 8px;"></td>
            <td style="text-align: center; padding: 12px; background: #f8f9fa; border-radius: 8px;">
                <div style="font-size: 24px; font-weight: bold;">
                    {{ "%.0f"|format(metrics.win_rate * 100) }}%
                </div>
                <div style="font-size: 11px; color: #666; text-transform: uppercase;">Win Rate</div>
            </td>
            <td style="width: 8px;"></td>
            <td style="text-align: center; padding: 12px; background: #f8f9fa; border-radius: 8px;">
                <div style="font-size: 24px; font-weight: bold;">
                    {{ metrics.signals_followed }}
                </div>
                <div style="font-size: 11px; color: #666; text-transform: uppercase;">Trades</div>
            </td>
        </tr>
    </table>

    <!-- Recent Closed Trades -->
    {% if recent_closed and recent_closed|length > 0 %}
    <div style="margin-bottom: 16px;">
        <h3 style="font-size: 14px; margin-bottom: 8px; color: #666;">Recently Closed</h3>
        <table style="width: 100%; font-size: 13px; border-collapse: collapse;">
            {% for trade in recent_closed[:5] %}
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 8px 4px;">
                    <strong>{{ trade.symbol }}</strong>
                </td>
                <td style="padding: 8px 4px; color: #666;">
                    {{ trade.direction }}
                </td>
                <td style="padding: 8px 4px; text-align: right; color: {% if trade.r_multiple > 0 %}#22c55e{% else %}#ef4444{% endif %};">
                    {% if trade.r_multiple > 0 %}+{% endif %}{{ "%.2f"|format(trade.r_multiple) }}R
                </td>
            </tr>
            {% endfor %}
        </table>
    </div>
    {% endif %}

    <!-- Active Positions -->
    {% if active_positions and active_positions|length > 0 %}
    <div style="margin-bottom: 16px;">
        <h3 style="font-size: 14px; margin-bottom: 8px; color: #666;">Active Positions ({{ active_positions|length }})</h3>
        <table style="width: 100%; font-size: 13px; border-collapse: collapse;">
            <tr style="background: #f8f9fa;">
                <th style="padding: 8px 4px; text-align: left; font-weight: 600;">Symbol</th>
                <th style="padding: 8px 4px; text-align: left; font-weight: 600;">Entry</th>
                <th style="padding: 8px 4px; text-align: right; font-weight: 600;">Current</th>
                <th style="padding: 8px 4px; text-align: right; font-weight: 600;">P&L</th>
            </tr>
            {% for pos in active_positions %}
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 8px 4px;"><strong>{{ pos.symbol }}</strong></td>
                <td style="padding: 8px 4px;">${{ "%.2f"|format(pos.entry_price) }}</td>
                <td style="padding: 8px 4px; text-align: right;">${{ "%.2f"|format(pos.current_price) }}</td>
                <td style="padding: 8px 4px; text-align: right; color: {% if pos.unrealized_pnl > 0 %}#22c55e{% else %}#ef4444{% endif %};">
                    {% if pos.unrealized_pnl > 0 %}+{% endif %}{{ "%.1f"|format(pos.unrealized_pnl_pct * 100) }}%
                </td>
            </tr>
            {% endfor %}
        </table>
    </div>
    {% endif %}

    <!-- Streak Indicator -->
    {% if streak and streak.count >= 3 %}
    <div style="padding: 12px; background: {% if streak.type == 'win' %}#f0fdf4{% else %}#fef2f2{% endif %}; border-radius: 8px; text-align: center;">
        <span style="font-size: 20px;">{% if streak.type == 'win' %}🔥{% else %}⚠️{% endif %}</span>
        <span style="font-weight: 600;">
            {{ streak.count }}-trade {{ streak.type }}ing streak
        </span>
    </div>
    {% endif %}
</div>
```

**Acceptance Criteria**:
- [ ] Section integrates with existing daily email template
- [ ] Shows rolling performance metrics
- [ ] Displays recent closed trades
- [ ] Shows active positions if available
- [ ] Streak indicator appears when relevant

---

## Task 3.2.5: Implement Strategy Leaderboard

**Context**:
A leaderboard helps quickly identify best and worst performing strategies.

**Objective**:
Create leaderboard generation with ranking and trends.

**Files to Create**:
```
trading_system/tracking/reports/leaderboard.py
```

**Requirements**:

```python
"""Strategy leaderboard for ranking performance."""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional

from loguru import logger

from trading_system.tracking.analytics.strategy_analytics import (
    StrategyAnalyzer,
    StrategyMetrics,
)
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
                display_name=self.STRATEGY_DISPLAY_NAMES.get(
                    strategy.name, strategy.name.replace("_", " ").title()
                ),
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

        lines.extend([
            "-" * 75,
            f"Profitable: {leaderboard.profitable_strategies}/{leaderboard.total_strategies}",
        ])

        if leaderboard.top_performer:
            lines.append(f"Top Performer: {leaderboard.top_performer}")
        if leaderboard.most_improved:
            lines.append(f"Most Improved: {leaderboard.most_improved}")
        if leaderboard.needs_attention:
            lines.append(f"Needs Attention: {leaderboard.needs_attention}")

        return "\n".join(lines)
```

**Acceptance Criteria**:
- [ ] Weekly/monthly/all-time leaderboards generate correctly
- [ ] Rank changes calculated from previous period
- [ ] Display names map correctly
- [ ] Text formatting is readable
- [ ] Handles empty/insufficient data

---

## Task 3.2.6: Create Performance CLI Commands

**Context**:
Users need command-line access to view performance metrics.

**Objective**:
Add CLI commands for viewing performance from terminal.

**Files to Create**:
```
trading_system/cli/commands/performance.py
```

**Requirements**:

```python
"""CLI commands for performance tracking."""

import argparse
from datetime import date, timedelta
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from trading_system.tracking.storage.sqlite_store import SQLiteTrackingStore
from trading_system.tracking.performance_calculator import PerformanceCalculator
from trading_system.tracking.analytics.signal_analytics import SignalAnalyzer
from trading_system.tracking.reports.leaderboard import LeaderboardGenerator


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
            f"\n[{streak_color}]Currently on a {analytics.current_streak}-trade "
            f"{streak_type} streak[/{streak_color}]"
        )

    store.close()
```

**Acceptance Criteria**:
- [ ] `trading-system performance summary` shows metrics
- [ ] `trading-system performance leaderboard` shows rankings
- [ ] `trading-system performance analytics` shows insights
- [ ] `trading-system performance recent` shows recent trades
- [ ] Rich formatting displays correctly in terminal

---

## Task 3.2.7: Integration with Signal Generation

**Context**:
The tracking system needs to automatically record signals when generated.

**Objective**:
Add hooks to integrate tracking with signal generation pipeline.

**Files to Modify**:
```
trading_system/signals/live_signal_generator.py  # From Phase 1
trading_system/scheduler/jobs/daily_signals_job.py  # From Phase 1
```

**Requirements**:

1. Add tracking to signal generator:
```python
# In live_signal_generator.py, add tracking support

from trading_system.tracking.signal_tracker import SignalTracker
from trading_system.tracking.storage.sqlite_store import SQLiteTrackingStore


class LiveSignalGenerator:
    """Generate live trading signals with tracking."""

    def __init__(
        self,
        strategies: List[StrategyInterface],
        portfolio_config: PortfolioConfig,
        data_fetcher: LiveDataFetcher,
        tracking_db: Optional[str] = None,
    ):
        self.strategies = strategies
        self.portfolio_config = portfolio_config
        self.data_fetcher = data_fetcher

        # Initialize tracking if database provided
        self.tracker = None
        if tracking_db:
            store = SQLiteTrackingStore(tracking_db)
            store.initialize()
            self.tracker = SignalTracker(store)

    async def generate_daily_signals(
        self,
        current_date: date,
    ) -> List[Recommendation]:
        """Generate signals and track them."""
        # ... existing signal generation code ...

        recommendations = self._select_recommendations(scored_signals)

        # Track generated signals
        if self.tracker:
            for recommendation in recommendations:
                signal_id = self.tracker.record_from_recommendation(recommendation)
                recommendation.tracking_id = signal_id
                logger.info(f"Tracked signal {signal_id} for {recommendation.symbol}")

        return recommendations

    def mark_signals_delivered(
        self,
        recommendations: List[Recommendation],
        method: str = "email",
    ):
        """Mark signals as delivered after sending."""
        if not self.tracker:
            return

        for rec in recommendations:
            if hasattr(rec, "tracking_id") and rec.tracking_id:
                self.tracker.mark_delivered(rec.tracking_id, method)
```

2. Update daily signals job:
```python
# In daily_signals_job.py

async def run_daily_signals_job(config: Config):
    """Run daily signal generation with tracking."""

    # ... existing job setup ...

    # Generate signals (tracking happens automatically)
    recommendations = await signal_generator.generate_daily_signals(date.today())

    # Send email
    success = await email_service.send_daily_report(
        recommendations=recommendations,
        portfolio_summary=portfolio_summary,
        news_digest=news_digest,
        recipients=config.email.recipients,
    )

    # Mark as delivered
    if success:
        signal_generator.mark_signals_delivered(recommendations, method="email")

    # Get performance metrics for logging
    if signal_generator.tracker:
        store = signal_generator.tracker.store
        calculator = PerformanceCalculator(store)
        rolling_metrics = calculator.calculate_rolling_metrics(window_days=30)

        logger.info(
            f"Rolling 30-day performance: "
            f"Win Rate={rolling_metrics['win_rate']:.0%}, "
            f"Expectancy={rolling_metrics['expectancy_r']:.2f}R"
        )
```

3. Add performance to daily email context:
```python
# In report_generator.py

def generate_daily_report_context(
    recommendations: List[Recommendation],
    portfolio_summary: PortfolioSummary,
    news_digest: NewsDigest,
    tracking_store: Optional[BaseTrackingStore] = None,
) -> Dict:
    """Generate context for daily email template."""

    context = {
        "recommendations": recommendations,
        "portfolio": portfolio_summary,
        "news": news_digest,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    # Add performance section if tracking available
    if tracking_store:
        calculator = PerformanceCalculator(tracking_store)
        metrics = calculator.calculate_rolling_metrics(window_days=30)

        # Get recent closed trades
        analyzer = SignalAnalyzer(tracking_store)
        analytics = analyzer.analyze()

        context["performance"] = {
            "days": 30,
            "metrics": metrics,
            "recent_closed": analytics.last_10_trades[:5],
            "streak": {
                "count": analytics.current_streak,
                "type": analytics.current_streak_type,
            } if analytics.current_streak >= 3 else None,
        }

    return context
```

**Acceptance Criteria**:
- [ ] Signals automatically tracked when generated
- [ ] Delivery marked after email sent
- [ ] Performance metrics included in daily email
- [ ] Rolling metrics logged during job execution

---

## Task 3.2.8: Integration Tests

**Context**:
End-to-end tests ensure the tracking system works correctly.

**Objective**:
Write comprehensive integration tests for the tracking pipeline.

**Files to Create**:
```
tests/test_tracking_integration.py
```

**Requirements**:

```python
"""Integration tests for performance tracking."""

import pytest
from datetime import date, datetime, timedelta
from pathlib import Path

from trading_system.tracking.models import (
    ConvictionLevel,
    ExitReason,
    SignalDirection,
    SignalStatus,
)
from trading_system.tracking.storage.sqlite_store import SQLiteTrackingStore
from trading_system.tracking.signal_tracker import SignalTracker
from trading_system.tracking.outcome_recorder import OutcomeRecorder
from trading_system.tracking.performance_calculator import PerformanceCalculator
from trading_system.tracking.analytics.signal_analytics import SignalAnalyzer
from trading_system.tracking.reports.leaderboard import LeaderboardGenerator


@pytest.fixture
def tracking_store(tmp_path):
    """Create a temporary tracking store."""
    db_path = tmp_path / "test_tracking.db"
    store = SQLiteTrackingStore(str(db_path))
    store.initialize()
    yield store
    store.close()


@pytest.fixture
def populated_store(tracking_store):
    """Create store with sample data."""
    tracker = SignalTracker(tracking_store)
    recorder = OutcomeRecorder(tracking_store)

    # Create sample signals and outcomes
    test_data = [
        # (symbol, direction, conviction, entry, exit, r_multiple, exit_reason)
        ("AAPL", "BUY", "HIGH", 150.0, 165.0, 2.0, "target_hit"),
        ("MSFT", "BUY", "MEDIUM", 300.0, 310.0, 1.5, "target_hit"),
        ("GOOGL", "BUY", "HIGH", 140.0, 135.0, -1.0, "stop_hit"),
        ("NVDA", "BUY", "LOW", 450.0, 440.0, -1.0, "stop_hit"),
        ("AMZN", "BUY", "HIGH", 180.0, 195.0, 2.5, "target_hit"),
        ("META", "BUY", "MEDIUM", 350.0, 365.0, 1.8, "target_hit"),
        ("TSLA", "BUY", "LOW", 250.0, 240.0, -1.2, "stop_hit"),
        ("BTC", "BUY", "HIGH", 45000.0, 48000.0, 1.5, "target_hit"),
        ("ETH", "BUY", "MEDIUM", 2500.0, 2400.0, -0.8, "stop_hit"),
        ("SOL", "BUY", "LOW", 100.0, 115.0, 2.0, "target_hit"),
    ]

    for symbol, direction, conviction, entry, exit_price, r_mult, reason in test_data:
        # Create signal
        signal_id = tracker.record_signal(
            symbol=symbol,
            asset_class="equity" if not symbol in ["BTC", "ETH", "SOL"] else "crypto",
            direction=SignalDirection(direction),
            signal_type="breakout_20d",
            conviction=ConvictionLevel(conviction),
            signal_price=entry,
            entry_price=entry,
            target_price=entry * 1.10,
            stop_price=entry * 0.95,
            technical_score=7.0,
            combined_score=7.0,
        )

        # Mark delivered
        tracker.mark_delivered(signal_id, method="email")

        # Record outcome
        recorder.record_outcome(
            signal_id=signal_id,
            entry_price=entry,
            exit_price=exit_price,
            exit_reason=ExitReason(reason),
            was_followed=True,
        )

    return tracking_store


class TestTrackingIntegration:
    """Integration tests for tracking pipeline."""

    def test_full_signal_lifecycle(self, tracking_store):
        """Test complete signal lifecycle: create -> deliver -> close."""
        tracker = SignalTracker(tracking_store)
        recorder = OutcomeRecorder(tracking_store)

        # 1. Record signal
        signal_id = tracker.record_signal(
            symbol="TEST",
            asset_class="equity",
            direction=SignalDirection.BUY,
            signal_type="breakout_20d",
            conviction=ConvictionLevel.HIGH,
            signal_price=100.0,
            entry_price=100.0,
            target_price=110.0,
            stop_price=95.0,
        )

        # Verify pending
        signal = tracker.get_signal(signal_id)
        assert signal.status == SignalStatus.PENDING

        # 2. Mark delivered
        tracker.mark_delivered(signal_id, method="email")
        signal = tracker.get_signal(signal_id)
        assert signal.was_delivered
        assert signal.delivery_method == "email"

        # 3. Mark entry filled
        tracker.mark_entry_filled(signal_id)
        signal = tracker.get_signal(signal_id)
        assert signal.status == SignalStatus.ACTIVE

        # 4. Record outcome
        recorder.record_outcome(
            signal_id=signal_id,
            entry_price=100.0,
            exit_price=108.0,
            exit_reason=ExitReason.TARGET_HIT,
        )

        # Verify closed
        signal = tracker.get_signal(signal_id)
        assert signal.status == SignalStatus.CLOSED

        outcome = recorder.get_outcome(signal_id)
        assert outcome.return_pct == pytest.approx(0.08, rel=0.01)
        assert outcome.r_multiple == pytest.approx(1.6, rel=0.01)  # 8/5

    def test_performance_metrics(self, populated_store):
        """Test performance calculation from populated data."""
        calculator = PerformanceCalculator(populated_store)

        metrics = calculator.calculate_metrics()

        # Should have 10 signals
        assert metrics.total_signals == 10
        assert metrics.signals_followed == 10

        # 6 winners, 4 losers = 60% win rate
        assert metrics.signals_won == 6
        assert metrics.signals_lost == 4
        assert metrics.win_rate == pytest.approx(0.6, rel=0.01)

    def test_signal_analytics(self, populated_store):
        """Test signal analytics generation."""
        analyzer = SignalAnalyzer(populated_store)

        analytics = analyzer.analyze()

        # Should have conviction breakdown
        assert "HIGH" in analytics.performance_by_conviction
        assert "MEDIUM" in analytics.performance_by_conviction
        assert "LOW" in analytics.performance_by_conviction

        # HIGH conviction should have best win rate
        high_wr = analytics.performance_by_conviction["HIGH"]["win_rate"]
        low_wr = analytics.performance_by_conviction["LOW"]["win_rate"]
        assert high_wr >= low_wr

        # Should generate insights
        assert len(analytics.insights) > 0

    def test_leaderboard_generation(self, populated_store):
        """Test strategy leaderboard."""
        generator = LeaderboardGenerator(populated_store)

        leaderboard = generator.generate_all_time()

        # Should have entries
        assert leaderboard.total_strategies > 0

        # First entry should be top performer
        if leaderboard.entries:
            top = leaderboard.entries[0]
            assert top.rank == 1

    def test_rolling_metrics(self, populated_store):
        """Test rolling metrics calculation."""
        calculator = PerformanceCalculator(populated_store)

        rolling = calculator.calculate_rolling_metrics(window_days=30)

        assert "win_rate" in rolling
        assert "avg_r" in rolling
        assert "expectancy_r" in rolling

    def test_equity_curve(self, populated_store):
        """Test equity curve generation."""
        calculator = PerformanceCalculator(populated_store)

        curve = calculator.get_equity_curve(starting_equity=100000.0)

        # Should have entries
        assert len(curve) > 0

        # Each entry should have required fields
        for point in curve:
            assert "equity" in point
            assert "drawdown_pct" in point


class TestTrackingEdgeCases:
    """Edge case tests for tracking."""

    def test_empty_database(self, tracking_store):
        """Test with no data."""
        calculator = PerformanceCalculator(tracking_store)

        metrics = calculator.calculate_metrics()
        assert metrics.total_signals == 0
        assert metrics.win_rate == 0.0

    def test_signal_not_followed(self, tracking_store):
        """Test signal that wasn't followed."""
        tracker = SignalTracker(tracking_store)
        recorder = OutcomeRecorder(tracking_store)

        signal_id = tracker.record_signal(
            symbol="TEST",
            asset_class="equity",
            direction=SignalDirection.BUY,
            signal_type="test",
            conviction=ConvictionLevel.LOW,
            signal_price=100.0,
            entry_price=100.0,
            target_price=110.0,
            stop_price=95.0,
        )

        # Mark as not followed
        recorder.record_missed_signal(signal_id, "Too risky")

        signal = tracker.get_signal(signal_id)
        assert signal.status == SignalStatus.EXPIRED

    def test_duplicate_outcome(self, tracking_store):
        """Test recording duplicate outcome."""
        tracker = SignalTracker(tracking_store)
        recorder = OutcomeRecorder(tracking_store)

        signal_id = tracker.record_signal(
            symbol="TEST",
            asset_class="equity",
            direction=SignalDirection.BUY,
            signal_type="test",
            conviction=ConvictionLevel.HIGH,
            signal_price=100.0,
            entry_price=100.0,
            target_price=110.0,
            stop_price=95.0,
        )

        # First outcome
        recorder.record_outcome(
            signal_id=signal_id,
            entry_price=100.0,
            exit_price=105.0,
            exit_reason=ExitReason.MANUAL,
        )

        # Second outcome should fail or update
        # (depending on implementation choice)
        with pytest.raises(Exception):
            recorder.record_outcome(
                signal_id=signal_id,
                entry_price=100.0,
                exit_price=110.0,
                exit_reason=ExitReason.TARGET_HIT,
            )
```

**Acceptance Criteria**:
- [ ] Full lifecycle test passes
- [ ] Performance metrics calculate correctly
- [ ] Analytics generate insights
- [ ] Leaderboard ranks strategies
- [ ] Edge cases handled gracefully
- [ ] All tests pass with `pytest tests/test_tracking_integration.py`

---

## Summary: Part 2 Tasks

| Task | Description | Key Deliverable |
|------|-------------|-----------------|
| 3.2.1 | Signal Analytics | Day/conviction/score analysis, insights |
| 3.2.2 | Strategy Analytics | Strategy comparison, correlations |
| 3.2.3 | Weekly Email Template | HTML template with metrics |
| 3.2.4 | Daily Email Section | Performance partial template |
| 3.2.5 | Strategy Leaderboard | Ranking with trends |
| 3.2.6 | CLI Commands | `performance summary/leaderboard/analytics` |
| 3.2.7 | Signal Gen Integration | Auto-track signals, mark delivered |
| 3.2.8 | Integration Tests | End-to-end tracking tests |

---

## Phase 3 Complete Checklist

After completing both Part 1 and Part 2:

- [ ] All tracking models defined (TrackedSignal, SignalOutcome, PerformanceMetrics)
- [ ] SQLite storage working with migrations
- [ ] SignalTracker recording all signals
- [ ] OutcomeRecorder calculating returns and R-multiples
- [ ] PerformanceCalculator computing all metrics
- [ ] Signal analytics providing insights
- [ ] Strategy analytics comparing approaches
- [ ] Leaderboard ranking strategies
- [ ] Weekly email template rendering
- [ ] Daily email includes performance section
- [ ] CLI commands working for all views
- [ ] Integration with signal generation automatic
- [ ] All integration tests passing

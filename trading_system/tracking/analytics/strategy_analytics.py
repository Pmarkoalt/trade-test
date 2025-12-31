"""Strategy-level analytics for comparing signal types."""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple
from trading_system.tracking.models import SignalOutcome

import numpy as np

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
        by_strategy: Dict[str, List[Tuple[TrackedSignal, Optional[SignalOutcome]]]] = {}
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
        metrics.avg_return_pct = float(np.mean(returns))
        metrics.total_return_pct = sum(returns)
        metrics.avg_r = float(np.mean(r_values))
        metrics.total_r = sum(r_values)

        # Winner/loser stats
        if winners:
            metrics.avg_winner_r = float(np.mean([o.r_multiple for _, o in winners]))
        if losers:
            metrics.avg_loser_r = float(np.mean([o.r_multiple for _, o in losers]))

        # Expectancy
        loss_rate = 1 - metrics.win_rate
        metrics.expectancy_r = (metrics.win_rate * metrics.avg_winner_r) - (loss_rate * abs(metrics.avg_loser_r))

        # Sharpe
        if len(returns) > 1:
            std = np.std(returns, ddof=1)
            if std > 0:
                metrics.sharpe_ratio = float((np.mean(returns) / std) * np.sqrt(252))

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
            metrics.avg_holding_days = float(np.mean(holding_days))

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
                strategy.expectancy_r * 2.0  # Expectancy most important
                + strategy.win_rate * 1.0  # Win rate matters
                + strategy.sharpe_ratio * 0.5  # Risk-adjusted return
                + (1 + strategy.max_drawdown_pct) * 0.5  # Drawdown penalty
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
            for name2 in names[i + 1 :]:
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
        recommendations.append(f"Focus on {best.name}: {best.expectancy_r:.2f}R expectancy, " f"{best.win_rate:.0%} win rate")

        # Worst strategy warning
        if worst.expectancy_r < 0:
            recommendations.append(f"Consider disabling {worst.name}: negative expectancy " f"({worst.expectancy_r:.2f}R)")

        # Correlation warning
        for (s1, s2), corr in comparison.strategy_correlations.items():
            if corr > 0.7:
                recommendations.append(f"{s1} and {s2} are highly correlated ({corr:.2f}) - " "consider using only one")

        # Diversification suggestion
        low_corr_pairs = [(s1, s2) for (s1, s2), corr in comparison.strategy_correlations.items() if corr < 0.3]
        if low_corr_pairs:
            recommendations.append(f"Diversify with uncorrelated strategies: {low_corr_pairs[0]}")

        return recommendations

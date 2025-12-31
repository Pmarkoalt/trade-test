"""Performance calculator for computing trading metrics."""

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np

from trading_system.tracking.models import PerformanceMetrics, SignalOutcome, TrackedSignal
from trading_system.tracking.storage.base_store import BaseTrackingStore

logger = logging.getLogger(__name__)


class PerformanceCalculator:
    """
    Calculate performance metrics from tracked signals.

    Example:
        calculator = PerformanceCalculator(store)

        # Get overall metrics
        metrics = calculator.calculate_metrics()
        print(f"Win Rate: {metrics.win_rate:.1%}")
        print(f"Expectancy: {metrics.expectancy_r:.2f}R")

        # Get metrics for specific period
        mtd_metrics = calculator.calculate_metrics(
            start_date=date.today().replace(day=1),
            end_date=date.today(),
        )
    """

    # Risk-free rate for Sharpe calculation (annualized)
    RISK_FREE_RATE = 0.05  # 5% (T-bills)

    # Trading days per year for annualization
    TRADING_DAYS_PER_YEAR = 252

    def __init__(self, store: BaseTrackingStore):
        """
        Initialize calculator.

        Args:
            store: Storage backend with signal/outcome data.
        """
        self.store = store

    def calculate_metrics(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        symbol: Optional[str] = None,
        asset_class: Optional[str] = None,
        signal_type: Optional[str] = None,
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.

        Args:
            start_date: Start of period (default: all time).
            end_date: End of period (default: today).
            symbol: Filter by symbol.
            asset_class: Filter by asset class.
            signal_type: Filter by signal type.

        Returns:
            PerformanceMetrics with all calculated values.
        """
        # Get signals and outcomes
        signals = self._get_filtered_signals(start_date, end_date, symbol, asset_class, signal_type)
        outcomes = self._get_outcomes_for_signals(signals)

        if not signals:
            return PerformanceMetrics(
                period_start=start_date or date.today(),
                period_end=end_date or date.today(),
            )

        # Basic counts
        total_signals = len(signals)
        followed_outcomes = [o for o in outcomes if o.was_followed]

        winners = [o for o in followed_outcomes if o.return_pct > 0]
        losers = [o for o in followed_outcomes if o.return_pct < 0]

        # Win rate
        total_closed = len(followed_outcomes)
        win_rate = len(winners) / total_closed if total_closed > 0 else 0.0

        # Follow rate
        delivered_signals = [s for s in signals if s.was_delivered]
        follow_rate = len(followed_outcomes) / len(delivered_signals) if delivered_signals else 0.0

        # Returns
        returns_pct = [o.return_pct for o in followed_outcomes]
        total_return = sum(returns_pct) if returns_pct else 0.0
        avg_return = float(np.mean(returns_pct)) if returns_pct else 0.0

        winner_returns = [o.return_pct for o in winners]
        loser_returns = [o.return_pct for o in losers]
        avg_winner = float(np.mean(winner_returns)) if winner_returns else 0.0
        avg_loser = float(np.mean(loser_returns)) if loser_returns else 0.0

        # R-multiples
        r_values = [o.r_multiple for o in followed_outcomes]
        total_r = sum(r_values) if r_values else 0.0
        avg_r = float(np.mean(r_values)) if r_values else 0.0

        winner_r = [o.r_multiple for o in winners]
        loser_r = [o.r_multiple for o in losers]
        avg_winner_r = float(np.mean(winner_r)) if winner_r else 0.0
        avg_loser_r = float(np.mean(loser_r)) if loser_r else 0.0

        # Expectancy = (Win% * AvgWin) - (Loss% * AvgLoss)
        loss_rate = 1 - win_rate
        expectancy_r = float((win_rate * avg_winner_r) - (loss_rate * abs(avg_loser_r)))

        # Risk metrics
        sharpe = self._calculate_sharpe(returns_pct)
        sortino = self._calculate_sortino(returns_pct)
        max_dd = self._calculate_max_drawdown(returns_pct)
        calmar = abs(total_return / max_dd) if max_dd != 0 else 0.0

        # Benchmark comparison
        alphas = [o.alpha for o in followed_outcomes if o.alpha is not None]
        benchmark_returns = [o.benchmark_return_pct for o in followed_outcomes]
        total_benchmark = sum(benchmark_returns) if benchmark_returns else 0.0
        total_alpha = sum(alphas) if alphas else 0.0

        # Build metrics by category
        def get_asset_class(outcome: SignalOutcome) -> Optional[str]:
            signal = self._get_signal(outcome.signal_id)
            return signal.asset_class if signal else None

        def get_signal_type(outcome: SignalOutcome) -> Optional[str]:
            signal = self._get_signal(outcome.signal_id)
            return signal.signal_type if signal else None

        def get_conviction(outcome: SignalOutcome) -> Optional[str]:
            signal = self._get_signal(outcome.signal_id)
            return signal.conviction.value if signal and signal.conviction else None

        metrics_by_asset = self._metrics_by_category(outcomes, get_asset_class)
        metrics_by_type = self._metrics_by_category(outcomes, get_signal_type)
        metrics_by_conviction = self._metrics_by_category(outcomes, get_conviction)

        return PerformanceMetrics(
            period_start=start_date or min(s.created_at.date() for s in signals),
            period_end=end_date or max(s.created_at.date() for s in signals),
            total_signals=total_signals,
            signals_followed=len(followed_outcomes),
            signals_won=len(winners),
            signals_lost=len(losers),
            win_rate=win_rate,
            follow_rate=follow_rate,
            total_return_pct=total_return,
            avg_return_pct=avg_return,
            avg_winner_pct=avg_winner,
            avg_loser_pct=avg_loser,
            total_r=total_r,
            avg_r=avg_r,
            avg_winner_r=avg_winner_r,
            avg_loser_r=avg_loser_r,
            expectancy_r=expectancy_r,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            benchmark_return_pct=total_benchmark,
            alpha=total_alpha,
            metrics_by_asset_class=metrics_by_asset,
            metrics_by_signal_type=metrics_by_type,
            metrics_by_conviction=metrics_by_conviction,
        )

    def calculate_rolling_metrics(
        self,
        window_days: int = 30,
    ) -> Dict:
        """
        Calculate rolling performance metrics.

        Args:
            window_days: Rolling window size in days.

        Returns:
            Dict with rolling metrics.
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=window_days)

        metrics = self.calculate_metrics(
            start_date=start_date,
            end_date=end_date,
        )

        return {
            "window_days": window_days,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "win_rate": metrics.win_rate,
            "avg_r": metrics.avg_r,
            "expectancy_r": metrics.expectancy_r,
            "sharpe": metrics.sharpe_ratio,
            "total_signals": metrics.total_signals,
        }

    def get_equity_curve(
        self,
        starting_equity: float = 100000.0,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Dict]:
        """
        Generate equity curve from outcomes.

        Args:
            starting_equity: Starting portfolio value.
            start_date: Start date.
            end_date: End date.

        Returns:
            List of dicts with date and equity value.
        """
        outcomes = self.store.get_outcomes_by_date_range(
            start_date=start_date or date(2020, 1, 1),
            end_date=end_date or date.today(),
        )

        if not outcomes:
            return []

        # Sort by exit date
        outcomes.sort(key=lambda o: o.actual_exit_date or date.today())

        equity = starting_equity
        curve = []
        high_water_mark = equity

        for outcome in outcomes:
            if not outcome.was_followed:
                continue

            # Calculate position return
            position_size = starting_equity * 0.0075  # Default 0.75% risk
            position_return = position_size * outcome.r_multiple
            equity += position_return

            high_water_mark = max(high_water_mark, equity)
            drawdown = (equity - high_water_mark) / high_water_mark if high_water_mark > 0 else 0

            curve.append(
                {
                    "date": outcome.actual_exit_date.isoformat() if outcome.actual_exit_date else None,
                    "equity": equity,
                    "high_water_mark": high_water_mark,
                    "drawdown_pct": drawdown,
                }
            )

        return curve

    def _get_filtered_signals(
        self,
        start_date: Optional[date],
        end_date: Optional[date],
        symbol: Optional[str],
        asset_class: Optional[str],
        signal_type: Optional[str],
    ) -> List[TrackedSignal]:
        """Get signals with filters applied."""
        signals = self.store.get_signals_by_date_range(
            start_date=start_date or date(2020, 1, 1),
            end_date=end_date or date.today(),
            symbol=symbol,
            asset_class=asset_class,
        )

        if signal_type:
            signals = [s for s in signals if s.signal_type == signal_type]

        return signals

    def _get_outcomes_for_signals(
        self,
        signals: List[TrackedSignal],
    ) -> List[SignalOutcome]:
        """Get outcomes for given signals."""
        outcomes = []
        for signal in signals:
            outcome = self.store.get_outcome(signal.id)
            if outcome:
                outcomes.append(outcome)
        return outcomes

    def _get_signal(self, signal_id: str) -> Optional[TrackedSignal]:
        """Get signal by ID (with caching potential)."""
        return self.store.get_signal(signal_id)

    def _calculate_sharpe(
        self,
        returns: List[float],
        annualize: bool = True,
    ) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0

        avg_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        if std_return == 0:
            return 0.0

        # Daily risk-free rate
        daily_rf = self.RISK_FREE_RATE / self.TRADING_DAYS_PER_YEAR

        sharpe = (avg_return - daily_rf) / std_return

        if annualize:
            # Annualize assuming daily returns
            sharpe *= np.sqrt(self.TRADING_DAYS_PER_YEAR)

        return float(sharpe)

    def _calculate_sortino(
        self,
        returns: List[float],
        annualize: bool = True,
    ) -> float:
        """Calculate Sortino ratio (downside deviation only)."""
        if len(returns) < 2:
            return 0.0

        avg_return = np.mean(returns)

        # Downside returns only
        downside = [r for r in returns if r < 0]
        if not downside:
            return float("inf") if avg_return > 0 else 0.0

        downside_std = np.std(downside, ddof=1)

        if downside_std == 0:
            return 0.0

        daily_rf = self.RISK_FREE_RATE / self.TRADING_DAYS_PER_YEAR
        sortino = (avg_return - daily_rf) / downside_std

        if annualize:
            sortino *= np.sqrt(self.TRADING_DAYS_PER_YEAR)

        return float(sortino)

    def _calculate_max_drawdown(
        self,
        returns: List[float],
    ) -> float:
        """Calculate maximum drawdown from returns."""
        if not returns:
            return 0.0

        # Build equity curve
        equity = [1.0]  # Start at 1.0
        for r in returns:
            equity.append(equity[-1] * (1 + r))

        # Calculate running max and drawdown
        running_max = equity[0]
        max_dd = 0.0

        for e in equity:
            running_max = max(running_max, e)
            dd = (e - running_max) / running_max
            max_dd = min(max_dd, dd)

        return max_dd

    def _metrics_by_category(
        self,
        outcomes: List[SignalOutcome],
        category_func,
    ) -> Dict[str, Dict]:
        """Calculate metrics grouped by category."""
        # Intermediate storage with lists
        categories: Dict[str, Dict[str, List[float]]] = {}

        for outcome in outcomes:
            if not outcome.was_followed:
                continue

            try:
                category = category_func(outcome)
                if category is None:
                    continue
            except Exception:  # nosec B112 - exception handling for category computation, skip invalid outcomes
                continue

            if category not in categories:
                categories[category] = {
                    "returns": [],
                    "r_values": [],
                }

            categories[category]["returns"].append(outcome.return_pct)
            categories[category]["r_values"].append(outcome.r_multiple)

        # Calculate metrics per category
        result = {}
        for cat, data in categories.items():
            wins = sum(1 for r in data["returns"] if r > 0)
            total = len(data["returns"])

            result[cat] = {
                "total": total,
                "win_rate": wins / total if total > 0 else 0,
                "avg_return": np.mean(data["returns"]) if data["returns"] else 0,
                "avg_r": np.mean(data["r_values"]) if data["r_values"] else 0,
            }

        return result

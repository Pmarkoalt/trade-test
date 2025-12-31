"""Data service for dashboard."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from trading_system.tracking.storage.sqlite_store import SQLiteTrackingStore
from trading_system.tracking.performance_calculator import PerformanceCalculator
from trading_system.tracking.analytics.signal_analytics import SignalAnalyzer
from trading_system.tracking.reports.leaderboard import LeaderboardGenerator
from trading_system.tracking.models import SignalStatus, TrackedSignal


@dataclass
class DashboardData:
    """Container for dashboard data."""

    # Signals
    recent_signals: List[TrackedSignal]
    pending_signals: List[TrackedSignal]
    active_signals: List[TrackedSignal]

    # Performance
    metrics_30d: dict
    metrics_90d: dict
    metrics_all: dict

    # Analytics
    analytics: dict
    leaderboard: dict

    # Counts
    signal_counts: dict

    # Timestamp
    fetched_at: datetime


class DashboardDataService:
    """
    Service for fetching dashboard data.

    Example:
        service = DashboardDataService(tracking_db_path="tracking.db")
        data = service.get_dashboard_data()

        print(f"Recent signals: {len(data.recent_signals)}")
        print(f"Win rate: {data.metrics_30d['win_rate']:.0%}")
    """

    def __init__(
        self,
        tracking_db_path: str = "tracking.db",
        feature_db_path: str = "features.db",
    ):
        """
        Initialize data service.

        Args:
            tracking_db_path: Path to tracking database.
            feature_db_path: Path to feature database.
        """
        self.tracking_db_path = tracking_db_path
        self.feature_db_path = feature_db_path

    def get_dashboard_data(self) -> DashboardData:
        """Get all dashboard data."""
        store = SQLiteTrackingStore(self.tracking_db_path)

        try:
            # Signals
            recent_signals = store.get_recent_signals(days=30)
            pending_signals = store.get_signals_by_status(SignalStatus.PENDING, limit=50)
            active_signals = store.get_signals_by_status(SignalStatus.ACTIVE, limit=50)

            # Performance calculator
            calculator = PerformanceCalculator(store)

            metrics_30d = calculator.calculate_rolling_metrics(window_days=30)
            metrics_90d = calculator.calculate_rolling_metrics(window_days=90)
            metrics_all = calculator.calculate_metrics()

            # Analytics
            analyzer = SignalAnalyzer(store)
            analytics = analyzer.analyze()

            # Leaderboard
            lb_generator = LeaderboardGenerator(store)
            leaderboard = lb_generator.generate_monthly()

            # Counts
            signal_counts = store.count_signals_by_status()

            return DashboardData(
                recent_signals=recent_signals,
                pending_signals=pending_signals,
                active_signals=active_signals,
                metrics_30d=metrics_30d,
                metrics_90d=metrics_90d,
                metrics_all=metrics_all.__dict__,
                analytics=analytics.__dict__,
                leaderboard=leaderboard.__dict__,
                signal_counts=signal_counts,
                fetched_at=datetime.now(),
            )

        finally:
            store.close()

    def get_signals_dataframe(
        self,
        days: int = 30,
        status: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get signals as a DataFrame."""
        store = SQLiteTrackingStore(self.tracking_db_path)

        try:
            if status:
                signals = store.get_signals_by_status(
                    SignalStatus(status),
                    limit=500,
                )
            else:
                signals = store.get_recent_signals(days=days)

            if not signals:
                return pd.DataFrame()

            # Convert to DataFrame
            data = []
            for signal in signals:
                data.append({
                    "id": signal.id,
                    "symbol": signal.symbol,
                    "direction": signal.direction.value,
                    "signal_type": signal.signal_type,
                    "conviction": signal.conviction.value,
                    "status": signal.status.value,
                    "entry_price": signal.entry_price,
                    "target_price": signal.target_price,
                    "stop_price": signal.stop_price,
                    "combined_score": signal.combined_score,
                    "technical_score": signal.technical_score,
                    "news_score": signal.news_score,
                    "created_at": signal.created_at,
                    "was_delivered": signal.was_delivered,
                })

            df = pd.DataFrame(data)
            return df

        finally:
            store.close()

    def get_performance_timeseries(
        self,
        days: int = 90,
    ) -> pd.DataFrame:
        """Get performance timeseries for charting."""
        store = SQLiteTrackingStore(self.tracking_db_path)

        try:
            calculator = PerformanceCalculator(store)

            # Get equity curve
            curve = calculator.get_equity_curve(
                starting_equity=100000,
                start_date=date.today() - timedelta(days=days),
                end_date=date.today(),
            )

            if not curve:
                return pd.DataFrame()

            df = pd.DataFrame(curve)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            return df

        finally:
            store.close()

    def get_signal_details(self, signal_id: str) -> Optional[dict]:
        """Get detailed information for a single signal."""
        store = SQLiteTrackingStore(self.tracking_db_path)

        try:
            signal = store.get_signal(signal_id)
            if not signal:
                return None

            outcome = store.get_outcome(signal_id)

            return {
                "signal": signal.__dict__,
                "outcome": outcome.__dict__ if outcome else None,
            }

        finally:
            store.close()

    def get_daily_summary(self, target_date: date = None) -> dict:
        """Get summary for a specific date."""
        target_date = target_date or date.today()
        store = SQLiteTrackingStore(self.tracking_db_path)

        try:
            signals = store.get_signals_by_date_range(
                start_date=target_date,
                end_date=target_date,
            )

            return {
                "date": target_date.isoformat(),
                "total_signals": len(signals),
                "by_conviction": {
                    "HIGH": len([s for s in signals if s.conviction.value == "HIGH"]),
                    "MEDIUM": len([s for s in signals if s.conviction.value == "MEDIUM"]),
                    "LOW": len([s for s in signals if s.conviction.value == "LOW"]),
                },
                "by_direction": {
                    "BUY": len([s for s in signals if s.direction.value == "BUY"]),
                    "SELL": len([s for s in signals if s.direction.value == "SELL"]),
                },
                "signals": [s.__dict__ for s in signals],
            }

        finally:
            store.close()

    def get_strategy_comparison(self) -> pd.DataFrame:
        """Get strategy performance comparison."""
        store = SQLiteTrackingStore(self.tracking_db_path)

        try:
            lb_gen = LeaderboardGenerator(store)
            leaderboard = lb_gen.generate_all_time()

            if not leaderboard.entries:
                return pd.DataFrame()

            data = []
            for entry in leaderboard.entries:
                data.append({
                    "rank": entry.rank,
                    "strategy": entry.display_name,
                    "total_r": entry.total_r,
                    "win_rate": entry.win_rate,
                    "expectancy": entry.expectancy_r,
                    "trades": entry.trade_count,
                    "trend": entry.trend,
                })

            return pd.DataFrame(data)

        finally:
            store.close()


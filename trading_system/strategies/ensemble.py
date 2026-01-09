"""
Ensemble Strategy Module

Combines multiple strategy configurations for robust signal generation:
- Voting ensemble: Majority vote from top N configs
- Weighted ensemble: Weight by historical performance
- Stacking ensemble: ML meta-learner on strategy outputs
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EnsembleMember:
    """Individual strategy in the ensemble."""

    name: str
    config: Dict[str, Any]
    weight: float = 1.0
    historical_sharpe: float = 0.0
    historical_win_rate: float = 0.0


@dataclass
class EnsembleSignal:
    """Aggregated signal from ensemble."""

    symbol: str
    direction: str  # "long", "short", "neutral"
    confidence: float  # 0-1 aggregated confidence
    agreement_ratio: float  # % of members agreeing
    member_signals: Dict[str, str]  # Individual member signals


class EnsembleStrategy:
    """Combine multiple strategy configurations."""

    def __init__(
        self,
        members: List[EnsembleMember],
        voting_method: str = "weighted",
        min_agreement: float = 0.5,
        rebalance_weights: bool = True,
    ):
        """
        Initialize ensemble strategy.

        Args:
            members: List of ensemble member strategies
            voting_method: "majority", "weighted", or "unanimous"
            min_agreement: Minimum agreement ratio to generate signal
            rebalance_weights: Auto-rebalance weights based on performance
        """
        self.members = members
        self.voting_method = voting_method
        self.min_agreement = min_agreement
        self.rebalance_weights = rebalance_weights

        # Normalize weights
        self._normalize_weights()

        # Performance tracking
        self.member_performance: Dict[str, List[float]] = {m.name: [] for m in members}

    def _normalize_weights(self) -> None:
        """Normalize member weights to sum to 1."""
        total = sum(m.weight for m in self.members)
        if total > 0:
            for m in self.members:
                m.weight = m.weight / total

    def add_member(
        self,
        name: str,
        config: Dict[str, Any],
        weight: float = 1.0,
        historical_sharpe: float = 0.0,
    ) -> None:
        """Add a new member to the ensemble."""
        member = EnsembleMember(
            name=name,
            config=config,
            weight=weight,
            historical_sharpe=historical_sharpe,
        )
        self.members.append(member)
        self.member_performance[name] = []
        self._normalize_weights()

    def remove_member(self, name: str) -> None:
        """Remove a member from the ensemble."""
        self.members = [m for m in self.members if m.name != name]
        if name in self.member_performance:
            del self.member_performance[name]
        self._normalize_weights()

    def aggregate_signals(
        self,
        member_signals: Dict[str, Dict[str, Any]],
    ) -> List[EnsembleSignal]:
        """
        Aggregate signals from all members.

        Args:
            member_signals: Dict mapping member_name -> {symbol -> signal_info}

        Returns:
            List of aggregated ensemble signals
        """
        # Collect all symbols
        all_symbols = set()
        for signals in member_signals.values():
            all_symbols.update(signals.keys())

        ensemble_signals = []

        for symbol in all_symbols:
            signal = self._aggregate_symbol_signals(symbol, member_signals)
            if signal is not None:
                ensemble_signals.append(signal)

        return ensemble_signals

    def _aggregate_symbol_signals(
        self,
        symbol: str,
        member_signals: Dict[str, Dict[str, Any]],
    ) -> Optional[EnsembleSignal]:
        """Aggregate signals for a single symbol."""
        votes = {"long": 0.0, "short": 0.0, "neutral": 0.0}
        member_votes = {}
        total_weight = 0.0

        for member in self.members:
            if member.name not in member_signals:
                continue

            symbol_signals = member_signals[member.name]
            if symbol not in symbol_signals:
                member_votes[member.name] = "neutral"
                continue

            signal_info = symbol_signals[symbol]
            direction = signal_info.get("direction", "neutral")
            confidence = signal_info.get("confidence", 1.0)

            member_votes[member.name] = direction

            if self.voting_method == "weighted":
                vote_weight = member.weight * confidence
            else:
                vote_weight = 1.0

            votes[direction] += vote_weight
            total_weight += vote_weight

        if total_weight == 0:
            return None

        # Normalize votes
        for direction in votes:
            votes[direction] /= total_weight

        # Determine winning direction
        winning_direction = max(votes, key=votes.get)
        agreement_ratio = votes[winning_direction]

        # Check minimum agreement
        if self.voting_method == "unanimous":
            if agreement_ratio < 1.0:
                return None
        elif agreement_ratio < self.min_agreement:
            return None

        return EnsembleSignal(
            symbol=symbol,
            direction=winning_direction,
            confidence=agreement_ratio,
            agreement_ratio=agreement_ratio,
            member_signals=member_votes,
        )

    def update_member_performance(
        self,
        member_name: str,
        trade_pnl: float,
    ) -> None:
        """Update performance tracking for a member."""
        if member_name in self.member_performance:
            self.member_performance[member_name].append(trade_pnl)

        if self.rebalance_weights:
            self._rebalance_weights()

    def _rebalance_weights(self) -> None:
        """Rebalance weights based on recent performance."""
        min_trades = 10  # Minimum trades before rebalancing

        sharpes = {}
        for member in self.members:
            pnls = self.member_performance.get(member.name, [])
            if len(pnls) >= min_trades:
                returns = np.array(pnls)
                if returns.std() > 0:
                    sharpes[member.name] = returns.mean() / returns.std()
                else:
                    sharpes[member.name] = 0.0

        if not sharpes:
            return

        # Convert Sharpes to weights (higher Sharpe = higher weight)
        min_sharpe = min(sharpes.values())
        adjusted = {k: v - min_sharpe + 0.1 for k, v in sharpes.items()}
        total = sum(adjusted.values())

        for member in self.members:
            if member.name in adjusted:
                member.weight = adjusted[member.name] / total

    def get_member_stats(self) -> pd.DataFrame:
        """Get performance statistics for all members."""
        stats = []
        for member in self.members:
            pnls = self.member_performance.get(member.name, [])

            if pnls:
                wins = [p for p in pnls if p > 0]
                win_rate = len(wins) / len(pnls)
                avg_pnl = np.mean(pnls)
                sharpe = np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0
            else:
                win_rate = 0
                avg_pnl = 0
                sharpe = 0

            stats.append(
                {
                    "name": member.name,
                    "weight": member.weight,
                    "trades": len(pnls),
                    "win_rate": win_rate,
                    "avg_pnl": avg_pnl,
                    "sharpe": sharpe,
                }
            )

        return pd.DataFrame(stats)

    def generate_report(self) -> str:
        """Generate ensemble performance report."""
        lines = []
        lines.append("=" * 60)
        lines.append("ENSEMBLE STRATEGY REPORT")
        lines.append("=" * 60)
        lines.append(f"Members: {len(self.members)}")
        lines.append(f"Voting Method: {self.voting_method}")
        lines.append(f"Min Agreement: {self.min_agreement:.0%}")
        lines.append("")

        lines.append("MEMBER PERFORMANCE")
        lines.append("-" * 40)
        stats = self.get_member_stats()
        lines.append(stats.to_string(index=False))
        lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


class EnsembleBuilder:
    """Build ensemble from optimization results."""

    @staticmethod
    def from_top_trials(
        optimization_results: Dict,
        top_n: int = 5,
        weight_by_sharpe: bool = True,
    ) -> EnsembleStrategy:
        """
        Create ensemble from top N optimization trials.

        Args:
            optimization_results: Optimization results dict
            top_n: Number of top trials to include
            weight_by_sharpe: Weight members by their Sharpe ratio
        """
        trials = optimization_results.get("top_trials", [])[:top_n]

        members = []
        for i, trial in enumerate(trials):
            sharpe = trial.get("value", 0)
            weight = sharpe if weight_by_sharpe and sharpe > 0 else 1.0

            member = EnsembleMember(
                name=f"trial_{trial.get('number', i)}",
                config=trial.get("params", {}),
                weight=weight,
                historical_sharpe=sharpe,
            )
            members.append(member)

        return EnsembleStrategy(members, voting_method="weighted")

    @staticmethod
    def from_config_files(
        config_paths: List[str],
        weights: Optional[List[float]] = None,
    ) -> EnsembleStrategy:
        """Create ensemble from multiple config files."""
        import yaml

        members = []
        for i, path in enumerate(config_paths):
            with open(path) as f:
                config = yaml.safe_load(f)

            weight = weights[i] if weights and i < len(weights) else 1.0
            name = config.get("name", f"config_{i}")

            member = EnsembleMember(
                name=name,
                config=config,
                weight=weight,
            )
            members.append(member)

        return EnsembleStrategy(members)

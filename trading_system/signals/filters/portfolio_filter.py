"""Portfolio filter for signals based on portfolio constraints."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import logging

from ...models.signals import Signal

logger = logging.getLogger(__name__)


@dataclass
class PortfolioFilterConfig:
    """Configuration for portfolio-based filtering."""

    # Maximum number of positions
    max_positions: int = 10

    # Maximum exposure per symbol (as fraction of portfolio)
    max_symbol_exposure: float = 0.10

    # Maximum exposure per sector
    max_sector_exposure: float = 0.30

    # Maximum total long exposure
    max_long_exposure: float = 1.0

    # Maximum total short exposure
    max_short_exposure: float = 0.5

    # Maximum number of correlated positions (correlation > 0.7)
    max_correlated_positions: int = 3

    # Whether to allow adding to existing positions
    allow_position_additions: bool = True

    # Maximum times to add to a position
    max_adds_per_position: int = 2

    # Minimum cash reserve (as fraction of portfolio)
    min_cash_reserve: float = 0.10

    # Whether to diversify across asset classes
    require_diversification: bool = False

    # Correlation threshold for "correlated" positions
    correlation_threshold: float = 0.7


@dataclass
class PortfolioState:
    """Current portfolio state for filtering."""

    # Current positions by symbol
    positions: Dict[str, float] = None  # symbol -> exposure

    # Number of times added to each position
    position_adds: Dict[str, int] = None  # symbol -> add count

    # Sector exposure
    sector_exposure: Dict[str, float] = None  # sector -> exposure

    # Total long/short exposure
    long_exposure: float = 0.0
    short_exposure: float = 0.0

    # Cash available (as fraction)
    cash_available: float = 1.0

    # Correlation matrix (if available)
    correlations: Dict[str, Dict[str, float]] = None

    def __post_init__(self):
        if self.positions is None:
            self.positions = {}
        if self.position_adds is None:
            self.position_adds = {}
        if self.sector_exposure is None:
            self.sector_exposure = {}


class PortfolioFilter:
    """Filter signals based on portfolio constraints."""

    def __init__(
        self,
        config: Optional[PortfolioFilterConfig] = None,
        symbol_sectors: Optional[Dict[str, str]] = None,
    ):
        """Initialize portfolio filter.

        Args:
            config: Filter configuration
            symbol_sectors: Mapping of symbols to sectors
        """
        self.config = config or PortfolioFilterConfig()
        self.symbol_sectors = symbol_sectors or {}

    def filter(
        self,
        signals: List[Signal],
        portfolio_state: PortfolioState,
    ) -> List[Signal]:
        """Filter signals based on portfolio constraints.

        Args:
            signals: List of signals to filter
            portfolio_state: Current portfolio state

        Returns:
            Filtered list of signals
        """
        filtered = []

        for signal in signals:
            reasons = self._check_constraints(signal, portfolio_state)
            if not reasons:
                filtered.append(signal)
                # Update simulated state for subsequent checks
                self._update_simulated_state(signal, portfolio_state)
            else:
                logger.debug(f"Signal {signal.symbol} filtered: {', '.join(reasons)}")

        logger.info(f"Portfolio filter: {len(filtered)}/{len(signals)} signals passed")
        return filtered

    def _check_constraints(
        self,
        signal: Signal,
        state: PortfolioState,
    ) -> List[str]:
        """Check all portfolio constraints.

        Args:
            signal: Signal to check
            state: Current portfolio state

        Returns:
            List of violation reasons (empty if passes)
        """
        violations = []

        # Check position count
        if signal.symbol not in state.positions:
            if len(state.positions) >= self.config.max_positions:
                violations.append(f"Max positions ({self.config.max_positions}) reached")

        # Check if adding to existing position
        if signal.symbol in state.positions:
            if not self.config.allow_position_additions:
                violations.append("Position additions not allowed")
            elif state.position_adds.get(signal.symbol, 0) >= self.config.max_adds_per_position:
                violations.append(f"Max adds ({self.config.max_adds_per_position}) reached for {signal.symbol}")

        # Check symbol exposure
        current_exposure = state.positions.get(signal.symbol, 0.0)
        new_exposure = current_exposure + getattr(signal, 'position_size_pct', 0.05)
        if new_exposure > self.config.max_symbol_exposure:
            violations.append(f"Symbol exposure {new_exposure:.1%} > max {self.config.max_symbol_exposure:.1%}")

        # Check sector exposure
        sector = self.symbol_sectors.get(signal.symbol, "unknown")
        current_sector_exposure = state.sector_exposure.get(sector, 0.0)
        new_sector_exposure = current_sector_exposure + getattr(signal, 'position_size_pct', 0.05)
        if new_sector_exposure > self.config.max_sector_exposure:
            violations.append(f"Sector {sector} exposure {new_sector_exposure:.1%} > max")

        # Check direction exposure
        position_size = getattr(signal, 'position_size_pct', 0.05)
        if signal.direction == "BUY":
            new_long = state.long_exposure + position_size
            if new_long > self.config.max_long_exposure:
                violations.append(f"Long exposure {new_long:.1%} > max {self.config.max_long_exposure:.1%}")
        else:
            new_short = state.short_exposure + position_size
            if new_short > self.config.max_short_exposure:
                violations.append(f"Short exposure {new_short:.1%} > max {self.config.max_short_exposure:.1%}")

        # Check cash reserve
        new_cash = state.cash_available - position_size
        if new_cash < self.config.min_cash_reserve:
            violations.append(f"Cash reserve {new_cash:.1%} < min {self.config.min_cash_reserve:.1%}")

        # Check correlation constraints
        if state.correlations and signal.symbol in state.correlations:
            correlated_count = self._count_correlated_positions(
                signal.symbol, state.positions.keys(), state.correlations
            )
            if correlated_count >= self.config.max_correlated_positions:
                violations.append(f"Too many correlated positions ({correlated_count})")

        return violations

    def _count_correlated_positions(
        self,
        symbol: str,
        existing_symbols: Set[str],
        correlations: Dict[str, Dict[str, float]],
    ) -> int:
        """Count positions correlated with a symbol.

        Args:
            symbol: Symbol to check
            existing_symbols: Existing position symbols
            correlations: Correlation matrix

        Returns:
            Number of correlated positions
        """
        if symbol not in correlations:
            return 0

        count = 0
        for existing in existing_symbols:
            if existing in correlations.get(symbol, {}):
                if abs(correlations[symbol][existing]) >= self.config.correlation_threshold:
                    count += 1

        return count

    def _update_simulated_state(
        self,
        signal: Signal,
        state: PortfolioState,
    ):
        """Update portfolio state after accepting a signal.

        Args:
            signal: Accepted signal
            state: Portfolio state to update
        """
        position_size = getattr(signal, 'position_size_pct', 0.05)
        symbol = signal.symbol

        # Update position
        if symbol in state.positions:
            state.positions[symbol] += position_size
            state.position_adds[symbol] = state.position_adds.get(symbol, 0) + 1
        else:
            state.positions[symbol] = position_size
            state.position_adds[symbol] = 0

        # Update sector exposure
        sector = self.symbol_sectors.get(symbol, "unknown")
        state.sector_exposure[sector] = state.sector_exposure.get(sector, 0.0) + position_size

        # Update direction exposure
        if signal.direction == "BUY":
            state.long_exposure += position_size
        else:
            state.short_exposure += position_size

        # Update cash
        state.cash_available -= position_size

    def get_available_capacity(
        self,
        state: PortfolioState,
    ) -> Dict[str, float]:
        """Get available capacity for new positions.

        Args:
            state: Current portfolio state

        Returns:
            Dictionary with capacity metrics
        """
        return {
            "positions_available": max(0, self.config.max_positions - len(state.positions)),
            "long_capacity": max(0, self.config.max_long_exposure - state.long_exposure),
            "short_capacity": max(0, self.config.max_short_exposure - state.short_exposure),
            "cash_available": max(0, state.cash_available - self.config.min_cash_reserve),
        }

    def prioritize_for_portfolio(
        self,
        signals: List[Signal],
        state: PortfolioState,
    ) -> List[Signal]:
        """Prioritize signals for portfolio fit.

        Args:
            signals: List of signals
            state: Current portfolio state

        Returns:
            Signals sorted by portfolio fit score
        """
        scored = []
        for signal in signals:
            fit_score = self._calculate_portfolio_fit(signal, state)
            scored.append((signal, fit_score))

        # Sort by fit score descending
        sorted_signals = sorted(scored, key=lambda x: x[1], reverse=True)
        return [signal for signal, _ in sorted_signals]

    def _calculate_portfolio_fit(
        self,
        signal: Signal,
        state: PortfolioState,
    ) -> float:
        """Calculate how well a signal fits the portfolio.

        Args:
            signal: Signal to evaluate
            state: Current portfolio state

        Returns:
            Fit score (0 to 1, higher = better fit)
        """
        scores = []

        # Diversification score (new symbol = higher score)
        if signal.symbol not in state.positions:
            scores.append(1.0)
        else:
            scores.append(0.3)  # Adding to existing is less diverse

        # Sector diversification
        sector = self.symbol_sectors.get(signal.symbol, "unknown")
        sector_exposure = state.sector_exposure.get(sector, 0.0)
        sector_score = max(0, 1 - sector_exposure / self.config.max_sector_exposure)
        scores.append(sector_score)

        # Direction balance
        if signal.direction == "BUY":
            direction_ratio = state.long_exposure / max(state.short_exposure, 0.01)
        else:
            direction_ratio = state.short_exposure / max(state.long_exposure, 0.01)

        # Prefer signals that balance the portfolio
        balance_score = 1.0 if direction_ratio < 2 else 0.5 if direction_ratio < 5 else 0.2
        scores.append(balance_score)

        # Correlation score (if available)
        if state.correlations and signal.symbol in state.correlations:
            correlated = self._count_correlated_positions(
                signal.symbol, state.positions.keys(), state.correlations
            )
            corr_score = max(0, 1 - correlated / self.config.max_correlated_positions)
            scores.append(corr_score)

        return sum(scores) / len(scores)

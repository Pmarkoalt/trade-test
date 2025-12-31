"""Position queue selection logic when signals exceed available slots."""

from typing import List, Dict, Optional, Callable
import logging
from trading_system.models.signals import Signal
from trading_system.models.portfolio import Portfolio
from trading_system.portfolio.position_sizing import estimate_position_size
from trading_system.portfolio.correlation import compute_correlation_to_portfolio

logger = logging.getLogger(__name__)


def violates_correlation_guard(
    signal: Signal,
    portfolio: Portfolio,
    candidate_returns: Dict[str, List[float]],
    portfolio_returns: Dict[str, List[float]],
    lookback: int = 20
) -> bool:
    """Check if signal violates correlation guard.
    
    Correlation guard applies only if:
    - Portfolio has >= 4 positions
    - Average pairwise correlation > 0.70
    
    If guard is active, reject candidates with correlation > 0.75 to portfolio.
    
    Args:
        signal: Signal to check
        portfolio: Portfolio with existing positions
        candidate_returns: Dictionary mapping symbol to returns list
        portfolio_returns: Dictionary mapping symbol to returns list for existing positions
        lookback: Number of days for correlation calculation
    
    Returns:
        True if signal violates correlation guard, False otherwise
    """
    # Guard only applies if >= 4 positions
    if len(portfolio.positions) < 4:
        return False
    
    # Guard only applies if avg pairwise correlation > 0.70
    if portfolio.avg_pairwise_corr is None or portfolio.avg_pairwise_corr <= 0.70:
        return False
    
    # Get candidate returns
    if signal.symbol not in candidate_returns:
        return False  # Cannot compute correlation, allow signal
    
    candidate_ret = candidate_returns[signal.symbol]
    
    # Compute correlation to portfolio
    avg_corr = compute_correlation_to_portfolio(
        candidate_symbol=signal.symbol,
        candidate_returns=candidate_ret,
        portfolio_returns=portfolio_returns,
        lookback=lookback
    )
    
    if avg_corr is None:
        return False  # Cannot compute correlation, allow signal
    
    # Reject if correlation > 0.75
    return avg_corr > 0.75


def select_signals_from_queue(
    signals: List[Signal],
    portfolio: Portfolio,
    max_positions: int,
    max_exposure: float,
    risk_per_trade: float,
    max_position_notional: float,
    candidate_returns: Dict[str, List[float]],
    portfolio_returns: Dict[str, List[float]],
    lookback: int = 20
) -> List[Signal]:
    """Select signals from queue when signals exceed available slots.
    
    Selection process:
    1. Sort signals by score (descending)
    2. Apply constraints in order:
       a. Max positions constraint
       b. Max exposure constraint
       c. Capacity constraint (already checked in signal generation, but verify)
       d. Correlation guard (if >= 4 positions and avg_pairwise_corr > 0.70)
    3. Return selected signals
    
    Args:
        signals: List of signals to select from (should already be scored)
        portfolio: Portfolio with existing positions
        max_positions: Maximum number of positions allowed
        max_exposure: Maximum exposure as fraction of equity (e.g., 0.80 for 80%)
        risk_per_trade: Base risk per trade (e.g., 0.0075 for 0.75%)
        max_position_notional: Maximum position notional as fraction of equity (e.g., 0.15 for 15%)
        candidate_returns: Dictionary mapping symbol to returns list for candidates
        portfolio_returns: Dictionary mapping symbol to returns list for existing positions
        lookback: Number of days for correlation calculation
    
    Returns:
        List of selected signals
    """
    if not signals:
        return []
    
    # Sort by score (descending)
    sorted_signals = sorted(signals, key=lambda s: s.score, reverse=True)
    
    selected = []
    rejected_count = 0
    
    for signal in sorted_signals:
        # Constraint 1: Max positions
        current_positions = len(portfolio.positions)
        if current_positions + len(selected) >= max_positions:
            rejected_count += 1
            continue
        
        # Constraint 2: Max exposure
        # Estimate position size to compute notional
        estimated_qty = estimate_position_size(
            equity=portfolio.equity,
            risk_pct=risk_per_trade,
            entry_price=signal.entry_price,
            stop_price=signal.stop_price,
            max_position_notional=max_position_notional,
            risk_multiplier=portfolio.risk_multiplier
        )
        estimated_notional = signal.entry_price * estimated_qty
        
        # Check if adding this position would exceed max exposure
        # Note: portfolio.gross_exposure is current exposure, but we need to add
        # selected signals' exposure as well
        total_exposure = portfolio.gross_exposure + sum(
            s.entry_price * estimate_position_size(
                equity=portfolio.equity,
                risk_pct=risk_per_trade,
                entry_price=s.entry_price,
                stop_price=s.stop_price,
                max_position_notional=max_position_notional,
                risk_multiplier=portfolio.risk_multiplier
            )
            for s in selected
        ) + estimated_notional
        
        if total_exposure > portfolio.equity * max_exposure:
            rejected_count += 1
            continue
        
        # Constraint 3: Capacity constraint (should already be checked, but verify)
        if not signal.capacity_passed:
            rejected_count += 1
            continue
        
        # Constraint 4: Correlation guard
        if violates_correlation_guard(
            signal, portfolio, candidate_returns, portfolio_returns, lookback
        ):
            rejected_count += 1
            continue
        
        # All constraints passed
        selected.append(signal)
    
    if rejected_count > 0:
        logger.warning(
            f"REJECTED_SIGNALS: {rejected_count} signals rejected due to constraints "
            f"(selected {len(selected)} from {len(signals)} candidates)"
        )
    
    return selected


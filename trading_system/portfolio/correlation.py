"""Correlation metrics for portfolio diversification."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def compute_average_pairwise_correlation(
    returns_data: Dict[str, List[float]], lookback: int = 20, min_positions: int = 4
) -> tuple[Optional[float], Optional[np.ndarray]]:
    """Compute average pairwise correlation of portfolio positions.

    Computes correlation matrix for all positions and returns average pairwise
    correlation (excluding diagonal).

    Args:
        returns_data: Dictionary mapping symbol to list of daily returns
        lookback: Number of days to use for correlation calculation (default: 20)
        min_positions: Minimum number of positions required to compute correlation (default: 4)

    Returns:
        Tuple of (avg_pairwise_corr, correlation_matrix):
        - avg_pairwise_corr: Average pairwise correlation (None if insufficient data)
        - correlation_matrix: Full correlation matrix as numpy array (None if insufficient data)
    """
    if len(returns_data) < min_positions:
        return None, None

    # Get returns for all symbols with sufficient history
    position_returns = {}
    for symbol, returns in returns_data.items():
        if len(returns) >= lookback:
            position_returns[symbol] = returns[-lookback:]

    if len(position_returns) < 2:
        return None, None

    # Align returns by length (use minimum common length)
    min_length = min(len(returns) for returns in position_returns.values())
    aligned_returns = {symbol: returns[-min_length:] for symbol, returns in position_returns.items()}

    # Create DataFrame for correlation calculation
    returns_df = pd.DataFrame(aligned_returns)

    # Compute correlation matrix
    corr_matrix = returns_df.corr().values

    # Compute average pairwise correlation (exclude diagonal)
    n = len(corr_matrix)
    off_diagonal = []
    for i in range(n):
        for j in range(i + 1, n):
            corr_value = corr_matrix[i, j]
            if not np.isnan(corr_value):
                off_diagonal.append(corr_value)

    if not off_diagonal:
        return None, None

    avg_pairwise_corr = np.mean(off_diagonal)

    return avg_pairwise_corr, corr_matrix


def compute_correlation_to_portfolio(
    candidate_symbol: str,
    candidate_returns: List[float],
    portfolio_returns: Dict[str, List[float]],
    lookback: int = 20,
    min_days: int = 10,
) -> Optional[float]:
    """Compute correlation of candidate symbol to existing portfolio.

    Computes average correlation of candidate to all existing positions.

    Args:
        candidate_symbol: Symbol of candidate position
        candidate_returns: List of daily returns for candidate
        portfolio_returns: Dictionary mapping symbol to list of returns for existing positions
        lookback: Number of days to use for correlation (default: 20)
        min_days: Minimum days required for correlation calculation (default: 10)

    Returns:
        Average correlation to portfolio (None if insufficient data)
    """
    if len(candidate_returns) < min_days:
        return None  # Insufficient history

    # Get candidate returns window
    candidate_window = candidate_returns[-lookback:] if len(candidate_returns) >= lookback else candidate_returns

    # Get returns for existing positions
    portfolio_windows = []
    for symbol, returns in portfolio_returns.items():
        if symbol == candidate_symbol:
            continue  # Skip if candidate already in portfolio

        if len(returns) >= lookback:
            portfolio_windows.append(returns[-lookback:])
        elif len(returns) >= min_days:
            # Use available returns if at least min_days
            portfolio_windows.append(returns)

    if not portfolio_windows:
        return None  # No portfolio returns available

    # Compute correlations
    correlations = []
    for pos_returns in portfolio_windows:
        # Align lengths
        min_len = min(len(candidate_window), len(pos_returns))
        if min_len < min_days:
            continue

        candidate_aligned = candidate_window[-min_len:]
        pos_aligned = pos_returns[-min_len:]

        corr = np.corrcoef(candidate_aligned, pos_aligned)[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)

    if not correlations:
        return None

    return np.mean(correlations)

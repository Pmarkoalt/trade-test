"""Signal scoring functions for position queue ranking."""

from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from trading_system.models.features import FeatureRow
from trading_system.models.signals import Signal
from trading_system.portfolio.correlation import compute_correlation_to_portfolio

if TYPE_CHECKING:
    from trading_system.portfolio.portfolio import Portfolio


def compute_breakout_strength(signal: Signal, features: FeatureRow) -> float:
    """Compute breakout strength component for scoring.

    Breakout strength = (close - MA) / ATR14
    - For 20D breakout: use MA20
    - For 55D breakout: use MA50

    Args:
        signal: Signal object with triggered_on field
        features: FeatureRow with close, ma20, ma50, atr14

    Returns:
        Breakout strength value
    """
    if features.atr14 is None or features.atr14 <= 0:
        return 0.0

    close = features.close

    # Use appropriate MA based on breakout type
    if signal.triggered_on is None:
        return 0.0
    if signal.triggered_on.value == "20D":
        ma_value = features.ma20
    else:  # 55D
        ma_value = features.ma50

    if ma_value is None or np.isnan(ma_value):
        return 0.0

    breakout_strength = (close - ma_value) / features.atr14
    return breakout_strength


def compute_momentum_strength(features: FeatureRow) -> float:
    """Compute relative momentum strength vs benchmark.

    Momentum strength = roc60 - benchmark_roc60
    (relative strength: positive means outperforming benchmark)

    Args:
        features: FeatureRow with roc60 and benchmark_roc60

    Returns:
        Relative momentum strength value
    """
    if features.roc60 is None or np.isnan(features.roc60):
        return 0.0

    if features.benchmark_roc60 is None or np.isnan(features.benchmark_roc60):
        # If benchmark data unavailable, use absolute momentum
        return features.roc60

    relative_strength = features.roc60 - features.benchmark_roc60
    return relative_strength


def compute_diversification_bonus(
    signal: Signal,
    portfolio: "Portfolio",
    candidate_returns: Dict[str, List[float]],
    portfolio_returns: Dict[str, List[float]],
    lookback: int = 20,
) -> float:
    """Compute diversification bonus for scoring.

    Diversification bonus = 1 - avg_corr_to_portfolio
    If no existing positions: return 0.5 (neutral)

    Args:
        signal: Signal object
        portfolio: Portfolio with existing positions
        candidate_returns: Dictionary mapping symbol to returns list
        portfolio_returns: Dictionary mapping symbol to returns list for existing positions
        lookback: Number of days for correlation calculation

    Returns:
        Diversification bonus value (0-1 range, higher = more diversified)
    """
    if len(portfolio.positions) == 0:
        return 0.5  # Neutral if no existing positions

    # Get candidate returns
    if signal.symbol not in candidate_returns:
        return 0.5  # Neutral if no return data

    candidate_ret = candidate_returns[signal.symbol]

    # Compute correlation to portfolio
    avg_corr = compute_correlation_to_portfolio(
        candidate_symbol=signal.symbol, candidate_returns=candidate_ret, portfolio_returns=portfolio_returns, lookback=lookback
    )

    if avg_corr is None:
        return 0.5  # Neutral if correlation cannot be computed

    diversification_bonus = 1.0 - avg_corr
    return diversification_bonus


def rank_normalize(values: List[float]) -> List[float]:
    """Normalize values to 0-1 scale using rank-based normalization.

    Assigns ranks to values, handling ties by averaging ranks.
    Handles NaN values by assigning lowest rank (0.0).

    Args:
        values: List of values to normalize

    Returns:
        List of normalized values in [0, 1] range
    """
    if not values:
        return []

    # Convert to numpy array for easier handling
    arr = np.array(values, dtype=float)

    # Handle NaN/inf values - mark invalid
    valid_mask = np.isfinite(arr)

    if not np.any(valid_mask):
        # All values are invalid, return all zeros
        return [0.0] * len(values)

    # Create array for ranks
    ranks = np.zeros(len(arr))

    # For valid values, compute ranks
    valid_values = arr[valid_mask]
    valid_indices = np.where(valid_mask)[0]

    if len(valid_values) == 1:
        # Single valid value gets rank 1.0 (normalized to 1.0)
        ranks[valid_indices[0]] = 1.0
    else:
        # Sort valid values to get order
        sorted_indices = np.argsort(valid_values)
        sorted_values = valid_values[sorted_indices]

        # Assign ranks, handling ties by averaging
        current_rank = 1.0
        i = 0
        while i < len(sorted_values):
            # Count how many values are tied at this position
            tie_value = sorted_values[i]

            # Count ties
            tie_count = 1
            while i + tie_count < len(sorted_values) and sorted_values[i + tie_count] == tie_value:
                tie_count += 1

            # Assign average rank to all tied values
            average_rank = current_rank + (tie_count - 1) / 2.0

            for j in range(tie_count):
                original_idx = valid_indices[sorted_indices[i + j]]
                ranks[original_idx] = average_rank

            current_rank += tie_count
            i += tie_count

    # Normalize ranks to [0, 1] range
    n_valid = len(valid_values)
    if n_valid == 1:
        normalized = ranks.copy()
        normalized[valid_mask] = 1.0
    else:
        normalized = (ranks - 1.0) / (n_valid - 1.0)

    # Invalid values get 0.0
    normalized[~valid_mask] = 0.0

    return [float(x) for x in normalized.tolist()]


def get_sentiment_score(
    symbol: str,
    date: pd.Timestamp,
    sentiment_data: Optional[pd.DataFrame],
) -> float:
    """Get sentiment score for a symbol on a given date.

    Args:
        symbol: Stock symbol
        date: Date to get sentiment for
        sentiment_data: DataFrame with columns: date, symbol, sentiment_score

    Returns:
        Sentiment score (-1 to 1), or 0.0 if not available
    """
    if sentiment_data is None or sentiment_data.empty:
        return 0.0

    # Normalize date for comparison
    date_normalized = pd.Timestamp(date.date()) if hasattr(date, 'date') else pd.Timestamp(date)

    # Filter for symbol and date
    mask = (
        (sentiment_data["symbol"].str.upper() == symbol.upper()) &
        (pd.to_datetime(sentiment_data["date"]).dt.normalize() == date_normalized)
    )

    matching = sentiment_data[mask]

    if len(matching) > 0:
        return float(matching.iloc[0]["sentiment_score"])

    return 0.0


def score_signals(
    signals: List[Signal],
    get_features: Callable[[Signal], Optional[FeatureRow]],
    portfolio: "Portfolio",
    candidate_returns: Dict[str, List[float]],
    portfolio_returns: Dict[str, List[float]],
    lookback: int = 20,
    sentiment_data: Optional[pd.DataFrame] = None,
    sentiment_weight: float = 0.0,
) -> None:
    """Compute and assign scores to signals.

    Computes breakout strength, momentum strength, diversification bonus,
    and optionally sentiment score for each signal, then rank-normalizes
    each component across all signals, and computes final weighted score.

    Score weights (when sentiment_weight=0):
    - 0.50 * breakout_rank
    - 0.30 * momentum_rank
    - 0.20 * diversification_rank

    When sentiment is enabled (sentiment_weight > 0), weights are adjusted:
    - (0.50 - sentiment_weight/2) * breakout_rank
    - (0.30 - sentiment_weight/3) * momentum_rank
    - (0.20 - sentiment_weight/6) * diversification_rank
    - sentiment_weight * sentiment_rank

    Updates signal objects in-place by setting:
    - signal.breakout_strength
    - signal.momentum_strength
    - signal.diversification_bonus
    - signal.sentiment_score (if sentiment enabled)
    - signal.score

    Args:
        signals: List of signals to score
        get_features: Function to get FeatureRow for a signal (signal -> FeatureRow or None)
        portfolio: Portfolio with existing positions
        candidate_returns: Dictionary mapping symbol to returns list for candidates
        portfolio_returns: Dictionary mapping symbol to returns list for existing positions
        lookback: Number of days for correlation calculation
        sentiment_data: Optional DataFrame with sentiment scores
        sentiment_weight: Weight for sentiment in scoring (0.0-0.5)
    """
    if not signals:
        return

    # Determine if sentiment is enabled
    use_sentiment = sentiment_data is not None and sentiment_weight > 0

    # Step 1: Compute raw components for all signals
    breakout_strengths = []
    momentum_strengths = []
    diversification_bonuses = []
    sentiment_scores = []

    for signal in signals:
        # Get features for this signal
        features = get_features(signal)
        if features is None:
            # If features unavailable, use default values
            breakout_strengths.append(0.0)
            momentum_strengths.append(0.0)
            diversification_bonuses.append(0.5)
            sentiment_scores.append(0.0)
            continue

        # Compute breakout strength
        breakout_strength = compute_breakout_strength(signal, features)
        breakout_strengths.append(breakout_strength)

        # Compute momentum strength
        momentum_strength = compute_momentum_strength(features)
        momentum_strengths.append(momentum_strength)

        # Compute diversification bonus
        div_bonus = compute_diversification_bonus(signal, portfolio, candidate_returns, portfolio_returns, lookback)
        diversification_bonuses.append(div_bonus)

        # Compute sentiment score if enabled
        if use_sentiment:
            sent_score = get_sentiment_score(signal.symbol, signal.date, sentiment_data)
            # Normalize from [-1, 1] to [0, 1] for ranking
            normalized_sent = (sent_score + 1.0) / 2.0
            sentiment_scores.append(normalized_sent)
        else:
            sentiment_scores.append(0.5)  # Neutral

    # Step 2: Rank-normalize each component
    breakout_ranks = rank_normalize(breakout_strengths)
    momentum_ranks = rank_normalize(momentum_strengths)
    diversification_ranks = rank_normalize(diversification_bonuses)
    sentiment_ranks = rank_normalize(sentiment_scores) if use_sentiment else [0.5] * len(signals)

    # Step 3: Compute weighted scores and update signals
    # Adjust weights based on sentiment_weight
    if use_sentiment and sentiment_weight > 0:
        # Reduce other weights proportionally to make room for sentiment
        breakout_weight = 0.50 - (sentiment_weight * 0.50 / 0.85)
        momentum_weight = 0.30 - (sentiment_weight * 0.30 / 0.85)
        diversification_weight = 0.20 - (sentiment_weight * 0.20 / 0.85)
        # Ensure weights sum to ~1.0
        total = breakout_weight + momentum_weight + diversification_weight + sentiment_weight
        breakout_weight /= total
        momentum_weight /= total
        diversification_weight /= total
        sentiment_weight_adj = sentiment_weight / total
    else:
        breakout_weight = 0.50
        momentum_weight = 0.30
        diversification_weight = 0.20
        sentiment_weight_adj = 0.0

    for i, signal in enumerate(signals):
        signal.breakout_strength = breakout_strengths[i]
        signal.momentum_strength = momentum_strengths[i]
        signal.diversification_bonus = diversification_bonuses[i]

        # Store sentiment score on signal if available
        if use_sentiment:
            # Store original score (-1 to 1)
            original_sent = (sentiment_scores[i] * 2.0) - 1.0
            if hasattr(signal, 'sentiment_score'):
                signal.sentiment_score = original_sent  # type: ignore

        # Weighted score
        signal.score = (
            breakout_weight * breakout_ranks[i] +
            momentum_weight * momentum_ranks[i] +
            diversification_weight * diversification_ranks[i] +
            sentiment_weight_adj * sentiment_ranks[i]
        )

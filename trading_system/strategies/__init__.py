"""Signal scoring and queue selection strategies."""

from .scoring import (
    compute_breakout_strength,
    compute_momentum_strength,
    compute_diversification_bonus,
    rank_normalize,
    score_signals,
)
from .queue import (
    violates_correlation_guard,
    select_signals_from_queue,
)

__all__ = [
    "compute_breakout_strength",
    "compute_momentum_strength",
    "compute_diversification_bonus",
    "rank_normalize",
    "score_signals",
    "violates_correlation_guard",
    "select_signals_from_queue",
]

"""Correlation indicators (rolling correlation)."""

import numpy as np
import pandas as pd


def rolling_corr(returns_a: pd.Series, returns_b: pd.Series, window: int = 20) -> pd.Series:
    """Compute rolling correlation between two return series.

    Used for correlation guard to avoid over-concentration in correlated positions.
    Correlation is computed over a rolling window of returns.

    Args:
        returns_a: First return series (daily returns: (close[t]/close[t-1]) - 1)
        returns_b: Second return series (daily returns)
        window: Rolling window size (default 20)

    Returns:
        Series with rolling correlation values. Returns NaN until window is filled
        or if either series has insufficient data.

    Example:
        >>> returns_a = pd.Series([0.01, -0.02, 0.03, 0.01])
        >>> returns_b = pd.Series([0.02, -0.01, 0.02, 0.01])
        >>> corr20 = rolling_corr(returns_a, returns_b, window=20)
        >>> # First 19 values will be NaN
    """
    if len(returns_a) == 0 or len(returns_b) == 0:
        return pd.Series(dtype=float, index=returns_a.index)

    # Align indices (use intersection)
    common_index = returns_a.index.intersection(returns_b.index)
    if len(common_index) == 0:
        return pd.Series(dtype=float, index=returns_a.index)

    returns_a_aligned = returns_a.loc[common_index]
    returns_b_aligned = returns_b.loc[common_index]

    # Compute rolling correlation
    corr_series = returns_a_aligned.rolling(window=window, min_periods=window).corr(returns_b_aligned)

    # Explicitly set NaN for insufficient lookback (first window-1 values)
    if len(corr_series) > 0 and len(corr_series) >= window:
        corr_series.iloc[: window - 1] = np.nan

    # Reindex to original index (fill missing with NaN)
    if len(corr_series) < len(returns_a):
        corr_series = corr_series.reindex(returns_a.index)

    return corr_series

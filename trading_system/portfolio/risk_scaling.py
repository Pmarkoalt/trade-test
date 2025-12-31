"""Volatility scaling for portfolio risk management."""

from typing import List, Optional
import numpy as np


def compute_volatility_scaling(
    portfolio_returns: List[float],
    lookback_20d: int = 20,
    lookback_252d: int = 252
) -> tuple[float, Optional[float], Optional[float]]:
    """Compute volatility scaling multiplier based on portfolio returns.
    
    Computes 20D realized vol of portfolio equity returns, compares to median vol
    over 252D, and returns risk multiplier.
    
    Formula:
        vol_ratio = vol_20 / median_vol_252
        risk_multiplier = clip(1 / max(vol_ratio, 1), 0.33, 1.0)
    
    Args:
        portfolio_returns: List of daily portfolio returns (as decimals, e.g., 0.01 for 1%)
        lookback_20d: Lookback period for short-term volatility (default: 20)
        lookback_252d: Lookback period for median volatility calculation (default: 252)
    
    Returns:
        Tuple of (risk_multiplier, vol_20d, median_vol_252d):
        - risk_multiplier: Volatility scaling multiplier (0.33 to 1.0)
        - vol_20d: 20-day rolling portfolio volatility (annualized), None if insufficient data
        - median_vol_252d: Median vol over last 252 days, None if insufficient data
    """
    if len(portfolio_returns) < lookback_20d:
        # Insufficient history: use default multiplier
        return 1.0, None, None
    
    # Compute 20D volatility (annualized)
    returns_20d = portfolio_returns[-lookback_20d:]
    vol_20d = np.std(returns_20d) * np.sqrt(252)
    
    # Compute median over last 252D (if available)
    all_returns = portfolio_returns[-lookback_252d:] if len(portfolio_returns) >= lookback_252d else portfolio_returns
    
    if len(all_returns) >= lookback_20d:
        # Compute rolling 20D volatilities
        rolling_vols = []
        for i in range(len(all_returns) - lookback_20d + 1):
            window_returns = all_returns[i:i + lookback_20d]
            window_vol = np.std(window_returns) * np.sqrt(252)
            rolling_vols.append(window_vol)
        
        if rolling_vols:
            median_vol_252d = np.median(rolling_vols)
        else:
            median_vol_252d = vol_20d  # Use current vol as baseline
    else:
        # Use current vol as baseline if insufficient history
        median_vol_252d = vol_20d
    
    # Compute risk multiplier
    # If vol_20d > median_vol_252d, reduce risk (multiplier < 1.0)
    # If vol_20d < median_vol_252d, increase risk (multiplier = 1.0, capped)
    vol_ratio = vol_20d / median_vol_252d if median_vol_252d > 0 else 1.0
    risk_multiplier = max(0.33, min(1.0, 1.0 / max(vol_ratio, 1.0)))
    
    return risk_multiplier, vol_20d, median_vol_252d


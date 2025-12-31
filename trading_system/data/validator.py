"""Data validation functions for OHLCV data quality checks."""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def validate_ohlcv(df: pd.DataFrame, symbol: str) -> bool:
    """
    Validate OHLCV data.
    
    Checks:
    1. Required columns present
    2. OHLC relationships valid
    3. No negative prices/volumes
    4. No extreme moves (>50% in one day)
    5. Dates in chronological order
    6. No duplicate dates
    
    Args:
        df: DataFrame with OHLCV data (date as index)
        symbol: Symbol name for logging
    
    Returns:
        True if valid, False otherwise
    """
    # Check required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"{symbol}: Missing columns: {missing_cols}")
        return False
    
    # Check OHLC relationships
    invalid_ohlc = (
        (df['low'] > df['high']) |
        (df['open'] < df['low']) | (df['open'] > df['high']) |
        (df['close'] < df['low']) | (df['close'] > df['high'])
    )
    
    if invalid_ohlc.any():
        invalid_dates = df.index[invalid_ohlc].tolist()
        logger.error(
            f"{symbol}: Invalid OHLC at dates: {invalid_dates[:10]} "
            f"(showing first 10 of {invalid_ohlc.sum()} total)"
        )
        return False
    
    # Check for negative or zero values
    price_cols = ['open', 'high', 'low', 'close']
    if (df[price_cols] <= 0).any().any():
        logger.error(f"{symbol}: Non-positive prices found")
        return False
    
    if (df['volume'] < 0).any():
        logger.error(f"{symbol}: Negative volume found")
        return False
    
    # Check for extreme moves (>50% in one day)
    if len(df) > 1:
        returns = df['close'].pct_change().dropna()
        extreme_moves = abs(returns) > 0.50
        if extreme_moves.any():
            # Align extreme_moves with df.index (extreme_moves is indexed by returns.index)
            # Get the dates where extreme moves occurred
            extreme_dates = returns.index[extreme_moves].tolist()
            logger.warning(
                f"{symbol}: Extreme moves (>50%) at dates: {extreme_dates[:10]} "
                f"(showing first 10 of {extreme_moves.sum()} total). "
                "These may be data errors."
            )
            # Mark as warnings but don't fail validation (see EDGE_CASES.md)
    
    # Check date order
    if not df.index.is_monotonic_increasing:
        logger.error(f"{symbol}: Dates not in chronological order")
        return False
    
    # Check for duplicate dates
    if df.index.duplicated().any():
        duplicates = df.index[df.index.duplicated()].unique().tolist()
        logger.error(
            f"{symbol}: Duplicate dates: {duplicates[:10]} "
            f"(showing first 10 of {len(duplicates)} total)"
        )
        return False
    
    return True


def detect_missing_data(
    df: pd.DataFrame,
    symbol: str,
    asset_class: str = "equity",
    expected_frequency: str = "D"
) -> Dict[str, Any]:
    """
    Detect missing data periods.
    
    Args:
        df: DataFrame with OHLCV data (date as index)
        symbol: Symbol name for logging
        asset_class: "equity" or "crypto"
        expected_frequency: Expected frequency ("D" for daily)
    
    Returns:
        Dictionary with:
        - missing_dates: List of missing dates
        - consecutive_gaps: List of (start, end) tuples for consecutive gaps
        - gap_lengths: List of gap lengths in days
    """
    if len(df) == 0:
        return {
            'missing_dates': [],
            'consecutive_gaps': [],
            'gap_lengths': []
        }
    
    # Create expected date range
    start_date = df.index.min()
    end_date = df.index.max()
    
    if asset_class == "crypto":
        # Crypto: continuous calendar days (365 days)
        expected_dates = pd.date_range(
            start=start_date,
            end=end_date,
            freq='D'
        )
    else:
        # Equity: use trading calendar (weekdays only)
        # For simplicity, we use weekdays here
        # In production, could use pandas_market_calendars
        expected_dates = pd.date_range(
            start=start_date,
            end=end_date,
            freq='B'  # Business days
        )
    
    # Find missing dates
    missing_dates = expected_dates.difference(df.index)
    
    # Find consecutive gaps
    consecutive_gaps = []
    if len(missing_dates) > 0:
        sorted_missing = sorted(missing_dates)
        gap_start = sorted_missing[0]
        
        for i in range(1, len(sorted_missing)):
            days_diff = (sorted_missing[i] - sorted_missing[i-1]).days
            if days_diff > 1:
                # Gap ended, start new gap
                consecutive_gaps.append((gap_start, sorted_missing[i-1]))
                gap_start = sorted_missing[i]
        
        # Last gap
        consecutive_gaps.append((gap_start, sorted_missing[-1]))
    
    gap_lengths = [(end - start).days + 1 for start, end in consecutive_gaps]
    
    return {
        'missing_dates': missing_dates.tolist(),
        'consecutive_gaps': consecutive_gaps,
        'gap_lengths': gap_lengths
    }


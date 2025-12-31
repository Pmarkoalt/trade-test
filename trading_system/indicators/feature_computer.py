"""Main feature computation function that computes all indicators for a symbol."""

import pandas as pd
import numpy as np
from typing import Dict, Optional

from trading_system.models.features import FeatureRow
from trading_system.indicators.ma import ma
from trading_system.indicators.atr import atr
from trading_system.indicators.momentum import roc
from trading_system.indicators.breakouts import highest_close
from trading_system.indicators.volume import adv
from trading_system.indicators.correlation import rolling_corr


def compute_features(
    df_ohlc: pd.DataFrame,
    symbol: str,
    asset_class: str,
    benchmark_roc60: Optional[pd.Series] = None,
    benchmark_returns: Optional[pd.Series] = None
) -> pd.DataFrame:
    """Compute all indicators for a symbol and return as DataFrame of FeatureRow objects.
    
    This function computes all required indicators:
    - Moving averages: MA20, MA50, MA200
    - Volatility: ATR14
    - Momentum: ROC60
    - Breakout levels: highest_close_20d, highest_close_55d
    - Volume: ADV20
    - Returns: returns_1d
    - Benchmark data: benchmark_roc60, benchmark_returns_1d (if provided)
    
    Args:
        df_ohlc: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                 Index should be datetime
        symbol: Symbol name (e.g., "AAPL", "BTC")
        asset_class: "equity" or "crypto"
        benchmark_roc60: Optional series of benchmark ROC60 values (aligned by date)
        benchmark_returns: Optional series of benchmark daily returns (aligned by date)
    
    Returns:
        DataFrame with one row per date, containing all computed indicators.
        Each row can be converted to a FeatureRow object.
    
    Example:
        >>> df = pd.DataFrame({
        ...     'open': [100, 101, 102],
        ...     'high': [102, 103, 104],
        ...     'low': [99, 100, 101],
        ...     'close': [101, 102, 103],
        ...     'volume': [1e6, 1.1e6, 1.2e6]
        ... }, index=pd.date_range('2024-01-01', periods=3))
        >>> features = compute_features(df, 'AAPL', 'equity')
        >>> # Returns DataFrame with all indicators
    """
    # Validate required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df_ohlc.columns for col in required_cols):
        raise ValueError(
            f"DataFrame must contain columns: {required_cols}. "
            f"Found: {list(df_ohlc.columns)}"
        )
    
    # Validate asset_class
    if asset_class not in ["equity", "crypto"]:
        raise ValueError(f"asset_class must be 'equity' or 'crypto', got: {asset_class}")
    
    # Create a copy to avoid modifying original
    df = df_ohlc.copy()
    
    # Compute dollar volume
    if 'dollar_volume' not in df.columns:
        df['dollar_volume'] = df['close'] * df['volume']
    
    # Compute moving averages
    df['ma20'] = ma(df['close'], window=20)
    df['ma50'] = ma(df['close'], window=50)
    df['ma200'] = ma(df['close'], window=200)
    
    # Compute ATR
    df['atr14'] = atr(df[['high', 'low', 'close']], period=14)
    
    # Compute ROC60
    df['roc60'] = roc(df['close'], window=60)
    
    # Compute highest close (excluding today)
    df['highest_close_20d'] = highest_close(df['close'], window=20)
    df['highest_close_55d'] = highest_close(df['close'], window=55)
    
    # Compute ADV20
    df['adv20'] = adv(df['dollar_volume'], window=20)
    
    # Compute daily returns: (close[t] / close[t-1]) - 1
    df['returns_1d'] = (df['close'] / df['close'].shift(1)) - 1
    # First value is NaN (no previous close)
    
    # Compute MA50 slope (for equity eligibility)
    # Requires 70 bars: 50 for MA50 + 20 for slope lookback
    if len(df) >= 70:
        df['ma50_slope'] = (df['ma50'] / df['ma50'].shift(20)) - 1
    else:
        df['ma50_slope'] = np.nan
    
    # Add benchmark data if provided
    if benchmark_roc60 is not None:
        # Align by index (date)
        df['benchmark_roc60'] = benchmark_roc60.reindex(df.index)
    else:
        df['benchmark_roc60'] = np.nan
    
    if benchmark_returns is not None:
        # Align by index (date)
        df['benchmark_returns_1d'] = benchmark_returns.reindex(df.index)
    else:
        df['benchmark_returns_1d'] = np.nan
    
    # Create output DataFrame with FeatureRow-compatible columns
    feature_cols = [
        'date', 'symbol', 'asset_class',
        'close', 'open', 'high', 'low',
        'ma20', 'ma50', 'ma200',
        'atr14',
        'roc60',
        'highest_close_20d', 'highest_close_55d',
        'adv20',
        'returns_1d',
        'ma50_slope',
        'benchmark_roc60', 'benchmark_returns_1d'
    ]
    
    # Create result DataFrame
    result = pd.DataFrame(index=df.index)
    result['date'] = df.index
    result['symbol'] = symbol
    result['asset_class'] = asset_class
    
    # Copy price data
    for col in ['close', 'open', 'high', 'low']:
        result[col] = df[col]
    
    # Copy indicator data
    for col in [
        'ma20', 'ma50', 'ma200',
        'atr14',
        'roc60',
        'highest_close_20d', 'highest_close_55d',
        'adv20',
        'returns_1d',
        'ma50_slope',
        'benchmark_roc60', 'benchmark_returns_1d'
    ]:
        result[col] = df[col]
    
    return result


def compute_features_for_date(
    features_df: pd.DataFrame,
    date: pd.Timestamp
) -> Optional[FeatureRow]:
    """Convert a row from features DataFrame to a FeatureRow object.
    
    Args:
        features_df: DataFrame returned by compute_features()
        date: Date to extract features for
    
    Returns:
        FeatureRow object if date exists, None otherwise
    """
    if date not in features_df.index:
        return None
    
    row = features_df.loc[date]
    
    # Convert to FeatureRow
    return FeatureRow(
        date=row['date'],
        symbol=row['symbol'],
        asset_class=row['asset_class'],
        close=float(row['close']),
        open=float(row['open']),
        high=float(row['high']),
        low=float(row['low']),
        ma20=float(row['ma20']) if pd.notna(row['ma20']) else None,
        ma50=float(row['ma50']) if pd.notna(row['ma50']) else None,
        ma200=float(row['ma200']) if pd.notna(row['ma200']) else None,
        atr14=float(row['atr14']) if pd.notna(row['atr14']) else None,
        roc60=float(row['roc60']) if pd.notna(row['roc60']) else None,
        highest_close_20d=float(row['highest_close_20d']) if pd.notna(row['highest_close_20d']) else None,
        highest_close_55d=float(row['highest_close_55d']) if pd.notna(row['highest_close_55d']) else None,
        adv20=float(row['adv20']) if pd.notna(row['adv20']) else None,
        returns_1d=float(row['returns_1d']) if pd.notna(row['returns_1d']) else None,
        ma50_slope=float(row['ma50_slope']) if pd.notna(row['ma50_slope']) else None,
        benchmark_roc60=float(row['benchmark_roc60']) if pd.notna(row['benchmark_roc60']) else None,
        benchmark_returns_1d=float(row['benchmark_returns_1d']) if pd.notna(row['benchmark_returns_1d']) else None,
    )


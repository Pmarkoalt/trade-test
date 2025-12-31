"""Main feature computation function that computes all indicators for a symbol."""

import pandas as pd
import numpy as np
from typing import Dict, Optional

from trading_system.models.features import FeatureRow
from trading_system.exceptions import IndicatorError
from trading_system.indicators.ma import ma
from trading_system.indicators.atr import atr
from trading_system.indicators.momentum import roc
from trading_system.indicators.breakouts import highest_close
from trading_system.indicators.volume import adv
from trading_system.indicators.correlation import rolling_corr
from trading_system.data.memory_profiler import optimize_dataframe_dtypes
from trading_system.indicators.cache import enable_caching, get_cache


def compute_features(
    df_ohlc: pd.DataFrame,
    symbol: str,
    asset_class: str,
    benchmark_roc60: Optional[pd.Series] = None,
    benchmark_returns: Optional[pd.Series] = None,
    use_cache: bool = True,
    optimize_memory: bool = True
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
    - Mean reversion: zscore, ma_lookback, std_lookback (20-day default)
    
    Args:
        df_ohlc: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                 Index should be datetime
        symbol: Symbol name (e.g., "AAPL", "BTC")
        asset_class: "equity" or "crypto"
        benchmark_roc60: Optional series of benchmark ROC60 values (aligned by date)
        benchmark_returns: Optional series of benchmark daily returns (aligned by date)
        use_cache: If True, use indicator caching (default: True)
        optimize_memory: If True, optimize DataFrame dtypes to reduce memory (default: True)
    
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
        raise IndicatorError(
            f"DataFrame must contain columns: {required_cols}. "
            f"Found: {list(df_ohlc.columns)}",
            indicator_name="compute_features",
            symbol=symbol
        )
    
    # Validate asset_class
    if asset_class not in ["equity", "crypto"]:
        raise IndicatorError(
            f"asset_class must be 'equity' or 'crypto', got: {asset_class}",
            indicator_name="compute_features",
            symbol=symbol
        )
    
    # Optimized: Work with views where possible, only copy when necessary
    # For indicators, we'll create a new DataFrame to avoid modifying input
    # but compute indicators efficiently
    
    # Pre-compute dollar volume if needed (vectorized)
    dollar_volume = df_ohlc['close'] * df_ohlc['volume'] if 'dollar_volume' not in df_ohlc.columns else df_ohlc['dollar_volume']
    
    # Compute all indicators in batch (vectorized operations)
    # Moving averages - compute all at once using same close series
    close_series = df_ohlc['close']
    ma20 = ma(close_series, window=20, use_cache=use_cache)
    ma50 = ma(close_series, window=50, use_cache=use_cache)
    ma200 = ma(close_series, window=200, use_cache=use_cache)
    
    # ATR - requires high, low, close
    atr14 = atr(df_ohlc[['high', 'low', 'close']], period=14, use_cache=use_cache)
    
    # ROC60
    roc60 = roc(close_series, window=60, use_cache=use_cache)
    
    # Highest close (excluding today) - compute both windows
    highest_close_20d = highest_close(close_series, window=20, use_cache=use_cache)
    highest_close_55d = highest_close(close_series, window=55, use_cache=use_cache)
    
    # ADV20
    adv20 = adv(dollar_volume, window=20, use_cache=use_cache)
    
    # Daily returns: (close[t] / close[t-1]) - 1
    # Optimized: use vectorized division with proper NaN handling
    close_shifted = close_series.shift(1)
    returns_1d = pd.Series(
        np.divide(close_series.values, close_shifted.values, out=np.full(len(close_series), np.nan), where=close_shifted.values != 0) - 1,
        index=close_series.index
    )
    
    # MA50 slope (for equity eligibility)
    # Requires 70 bars: 50 for MA50 + 20 for slope lookback
    if len(ma50) >= 70:
        ma50_shifted = ma50.shift(20)
        ma50_slope = pd.Series(
            np.divide(ma50.values, ma50_shifted.values, out=np.full(len(ma50), np.nan), where=ma50_shifted.values != 0) - 1,
            index=ma50.index
        )
    else:
        ma50_slope = pd.Series(np.nan, index=df_ohlc.index)
    
    # Mean reversion indicators (z-score)
    # Default lookback of 20 days for mean reversion
    # Can be customized per strategy via parameters
    lookback_mean_reversion = 20
    ma_lookback = close_series.rolling(window=lookback_mean_reversion, min_periods=lookback_mean_reversion).mean()
    std_lookback = close_series.rolling(window=lookback_mean_reversion, min_periods=lookback_mean_reversion).std()
    
    # Z-score: (close - MA) / STD
    # Use np.divide for safe division (handles NaN and zero std)
    zscore = pd.Series(
        np.divide(
            (close_series.values - ma_lookback.values),
            std_lookback.values,
            out=np.full(len(close_series), np.nan),
            where=(std_lookback.values != 0) & ~np.isnan(std_lookback.values)
        ),
        index=close_series.index
    )
    
    # Explicitly set NaN for insufficient lookback (first lookback-1 values)
    if len(zscore) > 0 and len(zscore) >= lookback_mean_reversion:
        zscore.iloc[:lookback_mean_reversion-1] = np.nan
        ma_lookback.iloc[:lookback_mean_reversion-1] = np.nan
        std_lookback.iloc[:lookback_mean_reversion-1] = np.nan
    
    # Add benchmark data if provided (optimized alignment)
    if benchmark_roc60 is not None:
        benchmark_roc60_aligned = benchmark_roc60.reindex(df_ohlc.index)
    else:
        benchmark_roc60_aligned = pd.Series(np.nan, index=df_ohlc.index)
    
    if benchmark_returns is not None:
        benchmark_returns_aligned = benchmark_returns.reindex(df_ohlc.index)
    else:
        benchmark_returns_aligned = pd.Series(np.nan, index=df_ohlc.index)
    
    # Create result DataFrame efficiently using dict constructor
    # This is faster than creating empty DataFrame and assigning columns one by one
    # Use float32 for numeric columns to reduce memory usage
    result = pd.DataFrame({
        'date': df_ohlc.index,
        'symbol': symbol,
        'asset_class': asset_class,
        'close': df_ohlc['close'].values.astype('float32') if df_ohlc['close'].dtype != 'float32' else df_ohlc['close'].values,
        'open': df_ohlc['open'].values.astype('float32') if df_ohlc['open'].dtype != 'float32' else df_ohlc['open'].values,
        'high': df_ohlc['high'].values.astype('float32') if df_ohlc['high'].dtype != 'float32' else df_ohlc['high'].values,
        'low': df_ohlc['low'].values.astype('float32') if df_ohlc['low'].dtype != 'float32' else df_ohlc['low'].values,
        'ma20': ma20.values.astype('float32'),
        'ma50': ma50.values.astype('float32'),
        'ma200': ma200.values.astype('float32'),
        'atr14': atr14.values.astype('float32'),
        'roc60': roc60.values.astype('float32'),
        'highest_close_20d': highest_close_20d.values.astype('float32'),
        'highest_close_55d': highest_close_55d.values.astype('float32'),
        'adv20': adv20.values.astype('float32'),
        'returns_1d': returns_1d.values.astype('float32'),
        'ma50_slope': ma50_slope.values.astype('float32'),
        'benchmark_roc60': benchmark_roc60_aligned.values.astype('float32'),
        'benchmark_returns_1d': benchmark_returns_aligned.values.astype('float32'),
        # Mean reversion indicators
        'zscore': zscore.values.astype('float32'),
        'ma_lookback': ma_lookback.values.astype('float32'),
        'std_lookback': std_lookback.values.astype('float32')
    }, index=df_ohlc.index)
    
    # Optimize dtypes (convert string columns to category if beneficial)
    if optimize_memory:
        result = optimize_dataframe_dtypes(result)
    
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
    
    try:
        row = features_df.loc[date]
    except (KeyError, IndexError) as e:
        raise IndicatorError(
            f"Date {date} not found in features DataFrame",
            indicator_name="compute_features_for_date",
            symbol=None
        ) from e
    
    # Convert to FeatureRow
    try:
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
        # Mean reversion indicators
        zscore=float(row['zscore']) if pd.notna(row['zscore']) else None,
        ma_lookback=float(row['ma_lookback']) if pd.notna(row['ma_lookback']) else None,
        std_lookback=float(row['std_lookback']) if pd.notna(row['std_lookback']) else None,
    )
    except (KeyError, ValueError, TypeError) as e:
        raise IndicatorError(
            f"Error converting row to FeatureRow for date {date}: {e}",
            indicator_name="compute_features_for_date",
            symbol=row.get('symbol', None) if 'row' in locals() else None
        ) from e


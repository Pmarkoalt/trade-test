"""Synthetic data generator for testing.

This module provides utilities to generate synthetic OHLCV data with:
- Known patterns (trends, breakouts)
- Edge cases (extreme moves, missing data, invalid OHLC)
- Specific characteristics (volatility, correlation)
- Large datasets for performance testing
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class TrendType(Enum):
    """Types of price trends."""

    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"


class BreakoutType(Enum):
    """Types of breakout patterns."""

    FAST_20D = "fast_20d"  # Price breaks above 20-day high
    SLOW_55D = "slow_55d"  # Price breaks above 55-day high
    NONE = "none"


class DataPattern(Enum):
    """Predefined data patterns for testing."""

    NORMAL = "normal"
    EXTREME_MOVE = "extreme_move"
    FLASH_CRASH = "flash_crash"
    BREAKOUT_FAST = "breakout_fast"
    BREAKOUT_SLOW = "breakout_slow"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    MISSING_DAYS = "missing_days"
    INVALID_OHLC = "invalid_ohlc"


class SyntheticDataGenerator:
    """Generate synthetic OHLCV data for testing."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional random seed.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed

    def generate_ohlcv(
        self,
        symbol: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        base_price: float = 100.0,
        base_volume: float = 50000000.0,
        volatility: float = 0.02,
        trend: float = 0.0,
        asset_class: str = "equity",
        pattern: Optional[DataPattern] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate OHLCV DataFrame with specified characteristics.

        Args:
            symbol: Symbol name
            start_date: Start date
            end_date: End date
            base_price: Starting price
            base_volume: Base volume level
            volatility: Daily volatility (0.02 = 2% daily)
            trend: Daily trend (0.0 = no trend, 0.001 = 0.1% daily upward)
            asset_class: "equity" or "crypto" (affects trading calendar)
            pattern: Optional predefined pattern to apply
            **kwargs: Additional pattern-specific parameters

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        # Generate trading calendar
        if asset_class == "crypto":
            dates = pd.date_range(start_date, end_date, freq="D")
        else:
            dates = pd.bdate_range(start_date, end_date)

        # Apply pattern if specified
        if pattern == DataPattern.EXTREME_MOVE:
            return self._generate_extreme_move(symbol, dates, base_price, base_volume, **kwargs)
        elif pattern == DataPattern.FLASH_CRASH:
            return self._generate_flash_crash(symbol, dates, base_price, base_volume, **kwargs)
        elif pattern == DataPattern.BREAKOUT_FAST:
            return self._generate_breakout(symbol, dates, base_price, base_volume, BreakoutType.FAST_20D, **kwargs)
        elif pattern == DataPattern.BREAKOUT_SLOW:
            return self._generate_breakout(symbol, dates, base_price, base_volume, BreakoutType.SLOW_55D, **kwargs)
        elif pattern == DataPattern.HIGH_VOLATILITY:
            return self._generate_ohlcv_base(symbol, dates, base_price, base_volume, volatility=0.05, trend=trend)
        elif pattern == DataPattern.LOW_VOLATILITY:
            return self._generate_ohlcv_base(symbol, dates, base_price, base_volume, volatility=0.005, trend=trend)
        elif pattern == DataPattern.MISSING_DAYS:
            return self._generate_with_missing_days(symbol, dates, base_price, base_volume, volatility, trend, **kwargs)
        elif pattern == DataPattern.INVALID_OHLC:
            return self._generate_invalid_ohlc(symbol, dates, base_price, base_volume, **kwargs)
        else:
            # Normal pattern
            return self._generate_ohlcv_base(symbol, dates, base_price, base_volume, volatility, trend)

    def _generate_ohlcv_base(
        self, symbol: str, dates: pd.DatetimeIndex, base_price: float, base_volume: float, volatility: float, trend: float
    ) -> pd.DataFrame:
        """Generate base OHLCV data with random walk."""
        data = []
        current_price = base_price

        for date in dates:
            # Generate random walk with trend
            daily_return = np.random.normal(trend, volatility)
            open_price = current_price
            close_price = current_price * (1 + daily_return)

            # Generate high and low with intraday volatility
            intraday_vol = volatility * 0.5
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, intraday_vol)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, intraday_vol)))

            # Ensure OHLC relationships are valid
            high_price = max(open_price, close_price, high_price)
            low_price = min(open_price, close_price, low_price)

            # Generate volume with some randomness
            volume = base_volume * (1 + np.random.uniform(-0.2, 0.2))

            data.append(
                {
                    "open": round(open_price, 2),
                    "high": round(high_price, 2),
                    "low": round(low_price, 2),
                    "close": round(close_price, 2),
                    "volume": round(volume, 0),
                }
            )

            current_price = close_price

        df = pd.DataFrame(data, index=dates)
        df.index.name = "date"
        return df

    def _generate_extreme_move(
        self,
        symbol: str,
        dates: pd.DatetimeIndex,
        base_price: float,
        base_volume: float,
        move_date: Optional[pd.Timestamp] = None,
        move_pct: float = 0.60,
        direction: str = "up",
    ) -> pd.DataFrame:
        """Generate data with an extreme price move (>50%).

        Args:
            move_date: Date of extreme move (defaults to middle of range)
            move_pct: Percentage move (default 60%)
            direction: "up" or "down"
        """
        if move_date is None:
            move_date = dates[len(dates) // 2]

        # Generate normal data up to move date
        move_idx = dates.get_loc(move_date) if move_date in dates else len(dates) // 2

        data = []
        current_price = base_price

        for i, date in enumerate(dates):
            if i == move_idx:
                # Extreme move
                if direction == "up":
                    close_price = current_price * (1 + move_pct)
                else:
                    close_price = current_price * (1 - move_pct)

                open_price = current_price
                high_price = close_price * 1.01 if direction == "up" else current_price * 1.01
                low_price = current_price * 0.99 if direction == "up" else close_price * 0.99
                volume = base_volume * 3.0  # High volume on extreme move
            else:
                # Normal day
                daily_return = np.random.normal(0, 0.02)
                open_price = current_price
                close_price = current_price * (1 + daily_return)
                high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
                low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
                volume = base_volume * (1 + np.random.uniform(-0.2, 0.2))

            data.append(
                {
                    "open": round(open_price, 2),
                    "high": round(high_price, 2),
                    "low": round(low_price, 2),
                    "close": round(close_price, 2),
                    "volume": round(volume, 0),
                }
            )

            current_price = close_price

        df = pd.DataFrame(data, index=dates)
        df.index.name = "date"
        return df

    def _generate_flash_crash(
        self,
        symbol: str,
        dates: pd.DatetimeIndex,
        base_price: float,
        base_volume: float,
        crash_date: Optional[pd.Timestamp] = None,
        crash_pct: float = 0.20,
    ) -> pd.DataFrame:
        """Generate data with a flash crash scenario.

        Flash crash: sudden large drop followed by partial recovery.
        """
        if crash_date is None:
            crash_date = dates[len(dates) // 2]

        crash_idx = dates.get_loc(crash_date) if crash_date in dates else len(dates) // 2

        data = []
        current_price = base_price

        for i, date in enumerate(dates):
            if i == crash_idx:
                # Flash crash: gap down
                open_price = current_price * (1 - crash_pct)
                low_price = open_price * 0.95  # Further drop intraday
                close_price = open_price * 1.05  # Partial recovery
                high_price = open_price * 1.08
                volume = base_volume * 5.0  # Extreme volume
            elif i == crash_idx + 1:
                # Recovery day
                open_price = current_price * 0.90
                close_price = current_price * 0.92
                high_price = current_price * 0.95
                low_price = current_price * 0.88
                volume = base_volume * 2.0
            else:
                # Normal day
                daily_return = np.random.normal(0, 0.02)
                open_price = current_price
                close_price = current_price * (1 + daily_return)
                high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
                low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
                volume = base_volume * (1 + np.random.uniform(-0.2, 0.2))

            data.append(
                {
                    "open": round(open_price, 2),
                    "high": round(high_price, 2),
                    "low": round(low_price, 2),
                    "close": round(close_price, 2),
                    "volume": round(volume, 0),
                }
            )

            current_price = close_price

        df = pd.DataFrame(data, index=dates)
        df.index.name = "date"
        return df

    def _generate_breakout(
        self,
        symbol: str,
        dates: pd.DatetimeIndex,
        base_price: float,
        base_volume: float,
        breakout_type: BreakoutType,
        breakout_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Generate data with a breakout pattern.

        For FAST_20D: price consolidates, then breaks above 20-day high
        For SLOW_55D: price consolidates, then breaks above 55-day high
        """
        if breakout_date is None:
            # Place breakout after enough history for pattern
            if breakout_type == BreakoutType.FAST_20D:
                breakout_idx = 25  # After 20 days + buffer
            else:
                breakout_idx = 60  # After 55 days + buffer
        else:
            breakout_idx = dates.get_loc(breakout_date) if breakout_date in dates else len(dates) // 2

        data = []
        current_price = base_price

        # Calculate the high that will be broken
        if breakout_type == BreakoutType.FAST_20D:
            lookback = 20
        else:
            lookback = 55

        for i, date in enumerate(dates):
            if i < breakout_idx:
                # Consolidation phase: sideways movement
                daily_return = np.random.normal(0, 0.01)  # Low volatility
                open_price = current_price
                close_price = current_price * (1 + daily_return)
                high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
                low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
                volume = base_volume * 0.8  # Lower volume during consolidation
            elif i == breakout_idx:
                # Breakout day: price breaks above recent high
                # Calculate recent high from previous days
                if i >= lookback:
                    recent_high = max([data[j]["high"] for j in range(i - lookback, i)])
                else:
                    recent_high = current_price * 1.02

                # Break above with clearance
                open_price = current_price
                close_price = recent_high * 1.02  # 2% above recent high
                high_price = close_price * 1.01
                low_price = min(open_price, recent_high * 0.99)
                volume = base_volume * 2.0  # High volume on breakout
            else:
                # Post-breakout: continue upward trend
                daily_return = np.random.normal(0.001, 0.015)  # Slight upward bias
                open_price = current_price
                close_price = current_price * (1 + daily_return)
                high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
                low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
                volume = base_volume * 1.2  # Elevated volume

            data.append(
                {
                    "open": round(open_price, 2),
                    "high": round(high_price, 2),
                    "low": round(low_price, 2),
                    "close": round(close_price, 2),
                    "volume": round(volume, 0),
                }
            )

            current_price = close_price

        df = pd.DataFrame(data, index=dates)
        df.index.name = "date"
        return df

    def _generate_with_missing_days(
        self,
        symbol: str,
        dates: pd.DatetimeIndex,
        base_price: float,
        base_volume: float,
        volatility: float,
        trend: float,
        missing_dates: Optional[List[pd.Timestamp]] = None,
        consecutive_missing: int = 0,
    ) -> pd.DataFrame:
        """Generate data with missing days.

        Args:
            missing_dates: Specific dates to exclude
            consecutive_missing: Number of consecutive days to remove (if > 0)
        """
        # Generate full dataset first
        full_df = self._generate_ohlcv_base(symbol, dates, base_price, base_volume, volatility, trend)

        # Remove missing dates
        if missing_dates:
            for missing_date in missing_dates:
                if missing_date in full_df.index:
                    full_df.drop(missing_date, inplace=True)

        # Remove consecutive days
        if consecutive_missing > 0:
            start_idx = len(full_df) // 2
            for i in range(consecutive_missing):
                if start_idx + i < len(full_df):
                    full_df.drop(full_df.index[start_idx + i], inplace=True)

        return full_df

    def _generate_invalid_ohlc(
        self,
        symbol: str,
        dates: pd.DatetimeIndex,
        base_price: float,
        base_volume: float,
        invalid_date: Optional[pd.Timestamp] = None,
        invalid_type: str = "close_out_of_range",
    ) -> pd.DataFrame:
        """Generate data with invalid OHLC relationships.

        Args:
            invalid_type: "close_out_of_range", "low_greater_than_high", "open_out_of_range"
        """
        # Generate normal data first
        normal_df = self._generate_ohlcv_base(symbol, dates, base_price, base_volume, 0.02, 0.0)

        if invalid_date is None:
            invalid_date = dates[len(dates) // 2]

        if invalid_date in normal_df.index:
            if invalid_type == "close_out_of_range":
                # Close price outside [low, high]
                normal_df.loc[invalid_date, "close"] = normal_df.loc[invalid_date, "high"] * 1.1
            elif invalid_type == "low_greater_than_high":
                # Low > High
                normal_df.loc[invalid_date, "low"] = normal_df.loc[invalid_date, "high"] * 1.1
            elif invalid_type == "open_out_of_range":
                # Open price outside [low, high]
                normal_df.loc[invalid_date, "open"] = normal_df.loc[invalid_date, "high"] * 1.1

        return normal_df

    def generate_correlated_data(
        self,
        symbols: List[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        base_prices: Optional[Dict[str, float]] = None,
        correlation_matrix: Optional[np.ndarray] = None,
        base_volatility: float = 0.02,
        asset_class: str = "equity",
    ) -> Dict[str, pd.DataFrame]:
        """Generate correlated OHLCV data for multiple symbols.

        Args:
            symbols: List of symbols to generate
            start_date: Start date
            end_date: End date
            base_prices: Optional dict of symbol -> base price
            correlation_matrix: Correlation matrix (n x n) for n symbols.
                               If None, generates random correlations.
            base_volatility: Base volatility level
            asset_class: "equity" or "crypto"

        Returns:
            Dict mapping symbol -> DataFrame
        """
        if base_prices is None:
            base_prices = {symbol: 100.0 for symbol in symbols}

        # Generate trading calendar
        if asset_class == "crypto":
            dates = pd.date_range(start_date, end_date, freq="D")
        else:
            dates = pd.bdate_range(start_date, end_date)

        n_symbols = len(symbols)

        # Generate or use provided correlation matrix
        if correlation_matrix is None:
            # Generate random positive definite correlation matrix
            A = np.random.rand(n_symbols, n_symbols)
            correlation_matrix = np.dot(A, A.T)
            # Normalize to correlation range [-1, 1]
            correlation_matrix = (correlation_matrix - correlation_matrix.min()) / (
                correlation_matrix.max() - correlation_matrix.min()
            ) * 2 - 1
            # Make symmetric
            correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
            np.fill_diagonal(correlation_matrix, 1.0)

        # Generate correlated returns using Cholesky decomposition
        L = np.linalg.cholesky(correlation_matrix)

        # Generate random returns for all symbols
        random_returns = np.random.normal(0, base_volatility, (len(dates), n_symbols))
        correlated_returns = np.dot(random_returns, L.T)

        # Generate OHLCV for each symbol
        result = {}
        for i, symbol in enumerate(symbols):
            data = []
            current_price = base_prices[symbol]

            for j, date in enumerate(dates):
                daily_return = correlated_returns[j, i]
                open_price = current_price
                close_price = current_price * (1 + daily_return)

                intraday_vol = base_volatility * 0.5
                high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, intraday_vol)))
                low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, intraday_vol)))

                high_price = max(open_price, close_price, high_price)
                low_price = min(open_price, close_price, low_price)

                volume = 50000000.0 * (1 + np.random.uniform(-0.2, 0.2))

                data.append(
                    {
                        "open": round(open_price, 2),
                        "high": round(high_price, 2),
                        "low": round(low_price, 2),
                        "close": round(close_price, 2),
                        "volume": round(volume, 0),
                    }
                )

                current_price = close_price

            df = pd.DataFrame(data, index=dates)
            df.index.name = "date"
            result[symbol] = df

        return result

    def generate_large_dataset(
        self,
        symbols: List[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        base_prices: Optional[Dict[str, float]] = None,
        asset_class: str = "equity",
    ) -> Dict[str, pd.DataFrame]:
        """Generate large dataset for performance testing.

        Optimized for speed when generating many symbols over long periods.

        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            base_prices: Optional base prices
            asset_class: "equity" or "crypto"

        Returns:
            Dict mapping symbol -> DataFrame
        """
        if base_prices is None:
            base_prices = {symbol: 100.0 for symbol in symbols}

        # Generate trading calendar
        if asset_class == "crypto":
            dates = pd.date_range(start_date, end_date, freq="D")
        else:
            dates = pd.bdate_range(start_date, end_date)

        result = {}
        for symbol in symbols:
            df = self._generate_ohlcv_base(symbol, dates, base_prices[symbol], 50000000.0, 0.02, 0.0)
            result[symbol] = df

        return result

    def save_to_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """Save DataFrame to CSV file in the expected format.

        Args:
            df: DataFrame with OHLCV data
            filepath: Path to save CSV file
        """
        # Reset index to make date a column
        df_to_save = df.reset_index()
        df_to_save.to_csv(filepath, index=False)


def generate_trend_data(
    symbol: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    base_price: float = 100.0,
    trend_pct: float = 0.001,
    direction: str = "up",
) -> pd.DataFrame:
    """Convenience function to generate trending data.

    Args:
        symbol: Symbol name
        start_date: Start date
        end_date: End date
        base_price: Starting price
        trend_pct: Daily trend percentage (0.001 = 0.1% per day)
        direction: "up" or "down"

    Returns:
        DataFrame with OHLCV data
    """
    generator = SyntheticDataGenerator()
    trend = trend_pct if direction == "up" else -trend_pct
    return generator.generate_ohlcv(symbol, start_date, end_date, base_price, trend=trend)


def generate_breakout_data(
    symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp, base_price: float = 100.0, breakout_type: str = "fast_20d"
) -> pd.DataFrame:
    """Convenience function to generate breakout data.

    Args:
        symbol: Symbol name
        start_date: Start date
        end_date: End date
        base_price: Starting price
        breakout_type: "fast_20d" or "slow_55d"

    Returns:
        DataFrame with OHLCV data
    """
    generator = SyntheticDataGenerator()
    pattern = DataPattern.BREAKOUT_FAST if breakout_type == "fast_20d" else DataPattern.BREAKOUT_SLOW
    return generator.generate_ohlcv(symbol, start_date, end_date, base_price, pattern=pattern)


def generate_edge_case_data(
    symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp, edge_case: str, **kwargs
) -> pd.DataFrame:
    """Convenience function to generate edge case data.

    Args:
        symbol: Symbol name
        start_date: Start date
        end_date: End date
        edge_case: One of "extreme_move", "flash_crash", "missing_days", "invalid_ohlc"
        **kwargs: Additional parameters for specific edge case

    Returns:
        DataFrame with OHLCV data
    """
    generator = SyntheticDataGenerator()

    pattern_map = {
        "extreme_move": DataPattern.EXTREME_MOVE,
        "flash_crash": DataPattern.FLASH_CRASH,
        "missing_days": DataPattern.MISSING_DAYS,
        "invalid_ohlc": DataPattern.INVALID_OHLC,
    }

    pattern = pattern_map.get(edge_case, DataPattern.NORMAL)
    return generator.generate_ohlcv(symbol, start_date, end_date, pattern=pattern, **kwargs)

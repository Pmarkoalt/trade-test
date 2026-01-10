"""Crypto universe selection, filtering, and rebalancing logic."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# Fixed crypto universe fallback (from spec)
FIXED_CRYPTO_UNIVERSE = ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOT", "MATIC", "LTC", "LINK"]


@dataclass
class UniverseConfig:
    """Configuration for crypto universe selection."""

    # Selection mode
    mode: Literal["fixed", "custom", "dynamic"] = "fixed"

    # Fixed/custom list (for mode="fixed" or "custom")
    symbols: Optional[List[str]] = None

    # Dynamic selection filters (for mode="dynamic")
    min_market_cap_usd: Optional[float] = None  # Minimum market cap in USD
    min_volume_usd: Optional[float] = None  # Minimum daily volume in USD
    min_liquidity_score: Optional[float] = None  # Minimum liquidity score (0-1)
    max_symbols: Optional[int] = None  # Maximum number of symbols to select

    # Rebalancing configuration
    rebalance_frequency: Optional[Literal["monthly", "quarterly", "never"]] = "never"
    rebalance_lookback_days: int = 30  # Days to look back for metrics

    # Custom universe file path (for mode="custom")
    universe_file_path: Optional[str] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.mode == "custom" and not self.symbols and not self.universe_file_path:
            raise ValueError("custom mode requires either symbols or universe_file_path")
        if self.mode == "dynamic" and not any([self.min_market_cap_usd, self.min_volume_usd, self.min_liquidity_score]):
            raise ValueError(
                "dynamic mode requires at least one filter: " "min_market_cap_usd, min_volume_usd, or min_liquidity_score"
            )


class CryptoUniverseManager:
    """Manages crypto universe selection and rebalancing."""

    def __init__(self, config: UniverseConfig):
        """Initialize universe manager.

        Args:
            config: Universe configuration
        """
        self.config = config
        self.current_universe: List[str] = []
        self.last_rebalance_date: Optional[pd.Timestamp] = None

    def select_universe(
        self, available_data: Dict[str, pd.DataFrame], reference_date: Optional[pd.Timestamp] = None
    ) -> List[str]:
        """Select universe based on configuration.

        Args:
            available_data: Dictionary mapping symbol -> DataFrame with OHLCV data
            reference_date: Date to use for filtering (uses latest available if None)

        Returns:
            List of selected symbols
        """
        if self.config.mode == "fixed":
            return self._select_fixed_universe(available_data)

        elif self.config.mode == "custom":
            return self._select_custom_universe(available_data)

        elif self.config.mode == "dynamic":
            return self._select_dynamic_universe(available_data, reference_date)

        else:
            raise ValueError(f"Unknown universe mode: {self.config.mode}")

    def _select_fixed_universe(self, available_data: Dict[str, pd.DataFrame]) -> List[str]:
        """Select fixed universe (use config symbols or fallback to default).

        Args:
            available_data: Available data to filter against

        Returns:
            List of symbols from fixed universe that have data
        """
        if self.config.symbols:
            fixed_list = self.config.symbols
        else:
            fixed_list = FIXED_CRYPTO_UNIVERSE

        # Filter to only symbols that have data
        selected = [s for s in fixed_list if s in available_data]

        logger.info(f"Fixed universe: {len(selected)}/{len(fixed_list)} symbols " f"have data: {selected}")

        return selected

    def _select_custom_universe(self, available_data: Dict[str, pd.DataFrame]) -> List[str]:
        """Select custom universe from file or config list.

        Args:
            available_data: Available data to filter against

        Returns:
            List of symbols from custom universe that have data
        """
        symbols = self.config.symbols

        # Load from file if provided
        if self.config.universe_file_path:
            file_path = Path(self.config.universe_file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Universe file not found: {self.config.universe_file_path}")

            try:
                df = pd.read_csv(file_path)

                if "symbol" in df.columns:
                    symbols_raw = df["symbol"].tolist()
                else:
                    # Assume first column is symbols
                    symbols_raw = df.iloc[:, 0].tolist()

                # Clean and normalize
                if symbols_raw is None or not isinstance(symbols_raw, list):
                    symbols = []
                else:
                    symbols = [str(s).upper().strip() for s in symbols_raw if pd.notna(s)]

            except Exception as e:
                raise ValueError(f"Error loading universe from {self.config.universe_file_path}: {e}")

        if not symbols:
            raise ValueError("No symbols provided for custom universe")

        # Filter to only symbols that have data
        selected = [s for s in symbols if s in available_data]

        logger.info(f"Custom universe: {len(selected)}/{len(symbols)} symbols " f"have data: {selected}")

        return selected

    def _select_dynamic_universe(
        self, available_data: Dict[str, pd.DataFrame], reference_date: Optional[pd.Timestamp] = None
    ) -> List[str]:
        """Select universe dynamically based on filters.

        Filters by:
        - Market cap (if market_cap column exists)
        - Volume (dollar_volume from OHLCV data)
        - Liquidity score (computed from volume and volatility)

        Args:
            available_data: Available data to filter against
            reference_date: Date to use for filtering (uses latest available if None)

        Returns:
            List of selected symbols sorted by market cap (descending)
        """
        if not available_data:
            logger.warning("No available data for dynamic universe selection")
            return []

        # Calculate metrics for each symbol
        symbol_metrics = []

        for symbol, df in available_data.items():
            if df.empty:
                continue

            # Determine reference date
            if reference_date is None:
                # Use latest available date
                eval_date = df.index.max()
            else:
                # Use provided date or latest before it
                available_dates = df.index[df.index <= reference_date]
                if len(available_dates) == 0:
                    continue
                eval_date = available_dates.max()

            # Get data up to eval_date for lookback
            lookback_start = eval_date - pd.Timedelta(days=self.config.rebalance_lookback_days)
            lookback_data = df[(df.index >= lookback_start) & (df.index <= eval_date)]

            if len(lookback_data) == 0:
                continue

            # Get latest bar
            latest_bar = lookback_data.iloc[-1]

            # Calculate metrics
            metrics = {
                "symbol": symbol,
                "date": eval_date,
            }

            # Market cap (if available in data)
            if "market_cap" in df.columns:
                metrics["market_cap"] = latest_bar.get("market_cap", 0.0)
            else:
                # Estimate from price * supply if available, otherwise None
                metrics["market_cap"] = None

            # Average dollar volume over lookback
            if "dollar_volume" in lookback_data.columns:
                metrics["avg_dollar_volume"] = lookback_data["dollar_volume"].mean()
            elif "volume" in lookback_data.columns and "close" in lookback_data.columns:
                # Compute dollar volume
                dollar_vol = lookback_data["close"] * lookback_data["volume"]
                metrics["avg_dollar_volume"] = dollar_vol.mean()
            else:
                metrics["avg_dollar_volume"] = 0.0

            # Liquidity score (0-1): normalized combination of volume and price stability
            if "dollar_volume" in lookback_data.columns:
                avg_dollar_vol = lookback_data["dollar_volume"].mean()
            elif "volume" in lookback_data.columns and "close" in lookback_data.columns:
                dollar_vol = lookback_data["close"] * lookback_data["volume"]
                avg_dollar_vol = dollar_vol.mean()
            else:
                avg_dollar_vol = 0.0

            # Price volatility (coefficient of variation)
            if "close" in lookback_data.columns and len(lookback_data) > 1:
                returns = lookback_data["close"].pct_change().dropna()
                if len(returns) > 0 and returns.mean() != 0:
                    vol = returns.std()
                    mean_return = abs(returns.mean())
                    cv = vol / mean_return if mean_return > 0 else float("inf")
                    # Lower CV = more stable = better liquidity score
                    # Normalize: score = 1 / (1 + cv) so higher volume + lower CV = higher score
                    liquidity_score = avg_dollar_vol / (1.0 + cv * 1e6)  # Scale for reasonable range
                else:
                    liquidity_score = avg_dollar_vol / 1e6
            else:
                liquidity_score = 0.0

            # Normalize liquidity score to 0-1 range (using percentile ranking)
            metrics["liquidity_score"] = liquidity_score

            symbol_metrics.append(metrics)

        if not symbol_metrics:
            logger.warning("No valid metrics computed for dynamic universe selection")
            return []

        # Convert to DataFrame for easier filtering
        metrics_df = pd.DataFrame(symbol_metrics)

        # Normalize liquidity scores to 0-1 range using percentile ranking
        if len(metrics_df) > 1 and metrics_df["liquidity_score"].max() > metrics_df["liquidity_score"].min():
            metrics_df["liquidity_score_normalized"] = (
                metrics_df["liquidity_score"] - metrics_df["liquidity_score"].min()
            ) / (metrics_df["liquidity_score"].max() - metrics_df["liquidity_score"].min())
        else:
            metrics_df["liquidity_score_normalized"] = 0.5  # Default if all same

        # Apply filters
        filtered = metrics_df.copy()

        if self.config.min_market_cap_usd is not None:
            # Only filter if market cap data is available
            if metrics_df["market_cap"].notna().any():
                filtered = filtered[
                    (filtered["market_cap"].isna()) | (filtered["market_cap"] >= self.config.min_market_cap_usd)
                ]
            else:
                logger.warning("min_market_cap_usd filter specified but no market cap data available")

        if self.config.min_volume_usd is not None:
            filtered = filtered[filtered["avg_dollar_volume"] >= self.config.min_volume_usd]

        if self.config.min_liquidity_score is not None:
            filtered = filtered[filtered["liquidity_score_normalized"] >= self.config.min_liquidity_score]

        # Sort by market cap (descending) or dollar volume if market cap not available
        if filtered["market_cap"].notna().any():
            filtered = filtered.sort_values("market_cap", ascending=False, na_last=True)
        else:
            filtered = filtered.sort_values("avg_dollar_volume", ascending=False)

        # Apply max_symbols limit
        if self.config.max_symbols is not None:
            filtered = filtered.head(self.config.max_symbols)

        selected = filtered["symbol"].tolist()

        logger.info(f"Dynamic universe: selected {len(selected)}/{len(available_data)} symbols. " f"Selected: {selected}")

        if len(selected) == 0:
            logger.warning("Dynamic universe selection resulted in empty universe. " "Consider relaxing filters.")

        return [str(s) for s in selected]

    def should_rebalance(self, current_date: pd.Timestamp) -> bool:
        """Check if universe should be rebalanced.

        Args:
            current_date: Current date to check

        Returns:
            True if rebalancing should occur
        """
        if self.config.rebalance_frequency == "never":
            return False

        if self.last_rebalance_date is None:
            return True  # First time, should initialize

        if self.config.rebalance_frequency == "monthly":
            # Rebalance if we've passed into a new month
            return bool(
                current_date.year != self.last_rebalance_date.year or current_date.month != self.last_rebalance_date.month
            )

        elif self.config.rebalance_frequency == "quarterly":
            # Rebalance if we've passed into a new quarter
            current_quarter = (current_date.month - 1) // 3
            last_quarter = (self.last_rebalance_date.month - 1) // 3
            return bool(current_date.year != self.last_rebalance_date.year or current_quarter != last_quarter)

        return False

    def rebalance_universe(
        self, available_data: Dict[str, pd.DataFrame], current_date: pd.Timestamp
    ) -> Tuple[List[str], bool]:
        """Rebalance universe if needed.

        Args:
            available_data: Available data to select from
            current_date: Current date

        Returns:
            Tuple of (new_universe, was_rebalanced)
        """
        if self.should_rebalance(current_date):
            new_universe = self.select_universe(available_data, current_date)
            self.current_universe = new_universe
            self.last_rebalance_date = current_date
            logger.info(f"Universe rebalanced on {current_date.date()}: " f"{len(new_universe)} symbols")
            return new_universe, True
        else:
            # Return current universe if initialized, otherwise select now
            if not self.current_universe:
                self.current_universe = self.select_universe(available_data, current_date)
                self.last_rebalance_date = current_date
                logger.info(f"Universe initialized on {current_date.date()}: " f"{len(self.current_universe)} symbols")
            return self.current_universe, False

    def validate_universe(
        self, universe: List[str], available_data: Dict[str, pd.DataFrame], min_symbols: int = 1
    ) -> Tuple[bool, List[str]]:
        """Validate universe against available data.

        Args:
            universe: List of symbols to validate
            available_data: Available data to check against
            min_symbols: Minimum number of symbols required

        Returns:
            Tuple of (is_valid, warnings)
        """
        warnings = []

        if len(universe) < min_symbols:
            return False, [f"Universe has {len(universe)} symbols, minimum {min_symbols} required"]

        # Check that all symbols have data
        missing_data = [s for s in universe if s not in available_data]
        if missing_data:
            warnings.append(f"Symbols in universe but missing data: {missing_data}")

        # Check data quality (at least some bars)
        low_quality = []
        for symbol in universe:
            if symbol in available_data:
                df = available_data[symbol]
                if len(df) < 20:  # At least 20 bars
                    low_quality.append(symbol)

        if low_quality:
            warnings.append(f"Symbols with insufficient data (<20 bars): {low_quality}")

        is_valid = len(warnings) == 0
        return is_valid, warnings


def create_universe_config_from_dict(config_dict: dict) -> UniverseConfig:
    """Create UniverseConfig from dictionary (e.g., from YAML).

    Args:
        config_dict: Dictionary with universe configuration

    Returns:
        UniverseConfig instance
    """
    return UniverseConfig(**config_dict)


def select_crypto_universe(
    available_data: Dict[str, pd.DataFrame],
    config: Optional[UniverseConfig] = None,
    reference_date: Optional[pd.Timestamp] = None,
) -> List[str]:
    """Convenience function to select crypto universe.

    Args:
        available_data: Available OHLCV data
        config: Universe configuration (defaults to fixed universe)
        reference_date: Reference date for dynamic selection

    Returns:
        List of selected symbols
    """
    if config is None:
        config = UniverseConfig(mode="fixed")

    manager = CryptoUniverseManager(config)
    return manager.select_universe(available_data, reference_date)


def select_top_crypto_by_volume(
    available_data: Dict[str, pd.DataFrame],
    top_n: int = 10,
    lookback_days: int = 30,
    reference_date: Optional[pd.Timestamp] = None,
) -> List[str]:
    """Select top N crypto by average dollar volume.

    Args:
        available_data: Available OHLCV data
        top_n: Number of top coins to select
        lookback_days: Days to look back for volume calculation
        reference_date: Reference date for selection

    Returns:
        List of top N symbols by volume
    """
    config = UniverseConfig(
        mode="dynamic",
        min_volume_usd=0.0,
        max_symbols=top_n,
        rebalance_lookback_days=lookback_days,
    )

    manager = CryptoUniverseManager(config)
    return manager.select_universe(available_data, reference_date)

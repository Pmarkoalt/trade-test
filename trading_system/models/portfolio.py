"""Portfolio state data model."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .positions import Position


@dataclass
class Portfolio:
    """Portfolio state at a specific date."""

    date: pd.Timestamp

    # Cash and equity
    cash: float  # Available cash
    starting_equity: float  # Initial equity (100,000)
    equity: float  # Current equity = cash + sum(position_values)

    # Positions
    positions: Dict[str, Position] = field(default_factory=dict)  # symbol -> Position

    # Equity curve
    equity_curve: List[float] = field(default_factory=list)  # Historical equity values
    daily_returns: List[float] = field(default_factory=list)  # Daily portfolio returns

    # Exposure
    gross_exposure: float = 0.0  # Sum of all position notional values
    gross_exposure_pct: float = 0.0  # gross_exposure / equity
    per_position_exposure: Dict[str, float] = field(default_factory=dict)  # symbol -> pct

    # P&L
    realized_pnl: float = 0.0  # Cumulative realized P&L
    unrealized_pnl: float = 0.0  # Sum of all position unrealized P&L

    # Risk metrics
    portfolio_vol_20d: Optional[float] = None  # 20-day rolling portfolio volatility (annualized)
    median_vol_252d: Optional[float] = None  # Median vol over last 252 days
    risk_multiplier: float = 1.0  # Volatility scaling multiplier (0.33 to 1.0)

    # Correlation metrics
    avg_pairwise_corr: Optional[float] = None  # Average pairwise correlation (if >= 4 positions)
    correlation_matrix: Optional[np.ndarray] = None  # Full correlation matrix

    # Trade statistics
    total_trades: int = 0  # Total trades closed
    open_trades: int = 0  # Current open positions

    def update_equity(self, current_prices: Dict[str, float]) -> None:
        """Update equity based on current market prices.

        Args:
            current_prices: Dictionary mapping symbol to current price
        """
        # Update unrealized P&L for all open positions
        total_unrealized = 0.0
        total_exposure = 0.0
        open_count = 0

        for symbol, position in self.positions.items():
            # Only process open positions
            if not position.is_open():
                continue

            open_count += 1

            if symbol in current_prices:
                current_price = current_prices[symbol]
                position.update_unrealized_pnl(current_price)
                total_unrealized += position.unrealized_pnl
                total_exposure += current_price * position.quantity

        self.unrealized_pnl = total_unrealized
        self.gross_exposure = total_exposure
        self.gross_exposure_pct = total_exposure / self.equity if self.equity > 0 else 0.0
        self.equity = self.cash + total_exposure
        self.open_trades = open_count

    def compute_portfolio_returns(self, lookback: Optional[int] = None) -> List[float]:
        """Compute portfolio returns for volatility calculation.

        Args:
            lookback: Number of recent returns to return (None for all)

        Returns:
            List of daily returns
        """
        if len(self.equity_curve) < 2:
            return []

        returns = []
        for i in range(1, len(self.equity_curve)):
            ret = (self.equity_curve[i] / self.equity_curve[i - 1]) - 1
            returns.append(ret)

        if lookback is not None and len(returns) >= lookback:
            return returns[-lookback:]

        return returns

    def update_volatility_scaling(self) -> None:
        """Update risk multiplier based on portfolio volatility."""
        returns = self.compute_portfolio_returns(lookback=20)

        if len(returns) < 20:
            # Insufficient history: use default multiplier
            self.risk_multiplier = 1.0
            self.portfolio_vol_20d = None
            self.median_vol_252d = None
            return

        # Compute 20D volatility (annualized)
        vol_20d = np.std(returns) * np.sqrt(252)
        self.portfolio_vol_20d = vol_20d

        # Compute median over last 252 days
        all_returns = self.compute_portfolio_returns(lookback=252)
        if len(all_returns) >= 252:
            # Compute rolling median (simplified: use all available)
            rolling_vols = []
            for i in range(len(all_returns) - 19):
                window_returns = all_returns[i : i + 20]
                window_vol = np.std(window_returns) * np.sqrt(252)
                rolling_vols.append(window_vol)
            self.median_vol_252d = np.median(rolling_vols)
        else:
            # Use current vol as baseline if insufficient history
            self.median_vol_252d = vol_20d

        # Compute risk multiplier
        median_vol_value: Optional[float] = self.median_vol_252d
        if median_vol_value is not None and isinstance(median_vol_value, (int, float)) and median_vol_value > 0.0:
            vol_ratio = vol_20d / median_vol_value
        else:
            vol_ratio = 1.0
        self.risk_multiplier = max(0.33, min(1.0, 1.0 / max(vol_ratio, 1.0)))

    def update_correlation_metrics(self, returns_data: Dict[str, List[float]], lookback: int = 20) -> None:
        """Update correlation metrics for existing positions.

        Args:
            returns_data: Dictionary mapping symbol to list of daily returns
            lookback: Number of days to use for correlation calculation
        """
        if len(self.positions) < 4:
            self.avg_pairwise_corr = None
            self.correlation_matrix = None
            return

        # Get returns for all positions
        position_symbols = list(self.positions.keys())
        position_returns = {}

        for symbol in position_symbols:
            if symbol in returns_data and len(returns_data[symbol]) >= lookback:
                position_returns[symbol] = returns_data[symbol][-lookback:]

        if len(position_returns) < 2:
            self.avg_pairwise_corr = None
            self.correlation_matrix = None
            return

        # Compute correlation matrix
        try:
            returns_df = pd.DataFrame(position_returns)
            corr_matrix = returns_df.corr().values

            # Compute average pairwise correlation (exclude diagonal)
            n = len(corr_matrix)
            off_diagonal = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not np.isnan(corr_matrix[i, j]):
                        off_diagonal.append(corr_matrix[i, j])

            self.avg_pairwise_corr = np.mean(off_diagonal) if off_diagonal else None
            self.correlation_matrix = corr_matrix
        except Exception:
            # If correlation calculation fails, set to None
            self.avg_pairwise_corr = None
            self.correlation_matrix = None

    def __post_init__(self):
        """Validate portfolio data."""
        if self.starting_equity <= 0:
            raise ValueError(f"Invalid starting_equity: {self.starting_equity}, must be positive")

        if self.cash < 0:
            raise ValueError(f"Invalid cash: {self.cash}, must be >= 0")

        if self.equity <= 0:
            raise ValueError(f"Invalid equity: {self.equity}, must be positive")

        if not (0.33 <= self.risk_multiplier <= 1.0):
            raise ValueError(f"Invalid risk_multiplier: {self.risk_multiplier}, must be between 0.33 and 1.0")

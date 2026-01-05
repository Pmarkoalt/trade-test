"""Metrics calculation for backtest results."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..models.positions import Position


class MetricsCalculator:
    """Calculate performance metrics from portfolio and trade data."""

    def __init__(
        self,
        equity_curve: List[float],
        daily_returns: List[float],
        closed_trades: List[Position],
        dates: Optional[List[pd.Timestamp]] = None,
        benchmark_returns: Optional[List[float]] = None,
    ):
        """Initialize metrics calculator.

        Args:
            equity_curve: List of daily equity values
            daily_returns: List of daily portfolio returns (as decimals)
            closed_trades: List of closed Position objects
            dates: Optional list of dates corresponding to equity_curve
            benchmark_returns: Optional list of benchmark daily returns for correlation
        """
        self.equity_curve = np.array(equity_curve)
        self.daily_returns = np.array(daily_returns)
        self.closed_trades = closed_trades
        # Handle dates: can be None, empty list, or pandas Index/Series
        # Use .empty for pandas objects, len() for lists, and None check
        if dates is None:
            self.dates = []
        elif hasattr(dates, "empty"):
            # pandas Index/Series
            self.dates = [] if dates.empty else dates
        elif len(dates) == 0:
            # empty list
            self.dates = []
        else:
            self.dates = dates
        self.benchmark_returns = np.array(benchmark_returns) if benchmark_returns else None

        # Validate inputs
        if len(self.equity_curve) != len(self.daily_returns) + 1:
            raise ValueError(
                f"equity_curve length ({len(self.equity_curve)}) must be "
                f"daily_returns length ({len(self.daily_returns)}) + 1"
            )

        if self.benchmark_returns is not None:
            if len(self.benchmark_returns) != len(self.daily_returns):
                raise ValueError(
                    f"benchmark_returns length ({len(self.benchmark_returns)}) must match "
                    f"daily_returns length ({len(self.daily_returns)})"
                )

    # Primary Metrics

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calculate annualized Sharpe ratio.

        Args:
            risk_free_rate: Annual risk-free rate (default: 0.0)

        Returns:
            Annualized Sharpe ratio
        """
        if len(self.daily_returns) == 0:
            return 0.0

        mean_return = np.mean(self.daily_returns)
        std_return = np.std(self.daily_returns)

        if std_return == 0:
            return 0.0

        # Annualize: mean * 252, std * sqrt(252)
        annualized_return = mean_return * 252
        annualized_std = std_return * np.sqrt(252)

        excess_return = annualized_return - risk_free_rate

        return excess_return / annualized_std if annualized_std > 0 else 0.0

    def max_drawdown(self) -> float:
        """Calculate maximum drawdown as a percentage.

        Returns:
            Maximum drawdown as decimal (e.g., 0.15 for 15%)
        """
        if len(self.equity_curve) == 0:
            return 0.0

        # Calculate running maximum
        running_max = np.maximum.accumulate(self.equity_curve)

        # Calculate drawdown at each point
        drawdowns = (self.equity_curve - running_max) / running_max

        # Return maximum drawdown (most negative, so we take the minimum)
        max_dd = np.min(drawdowns)

        return float(abs(max_dd))  # Return as positive percentage

    def calmar_ratio(self) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown).

        Returns:
            Calmar ratio
        """
        max_dd = self.max_drawdown()
        if max_dd == 0:
            return 0.0

        # Calculate annualized return
        if len(self.equity_curve) < 2:
            return 0.0

        total_return = (self.equity_curve[-1] / self.equity_curve[0]) - 1

        # Annualize based on number of trading days
        num_days = len(self.daily_returns)
        if num_days == 0:
            return 0.0

        years = num_days / 252.0
        if years == 0:
            return 0.0

        annualized_return = (1 + total_return) ** (1 / years) - 1

        return annualized_return / max_dd if max_dd > 0 else 0.0

    def total_trades(self) -> int:
        """Get total number of closed trades.

        Returns:
            Total trades count
        """
        return len(self.closed_trades)

    # Secondary Metrics

    def expectancy(self) -> float:
        """Calculate expectancy in R-multiples.

        Expectancy = average R-multiple across all trades.
        R-multiple = (exit_price - entry_price) / (entry_price - initial_stop_price)

        Returns:
            Average R-multiple (expectancy)
        """
        if len(self.closed_trades) == 0:
            return 0.0

        r_multiples = []
        for trade in self.closed_trades:
            if trade.exit_price is None or trade.entry_price is None:
                continue

            # Calculate R-multiple
            price_change = trade.exit_price - trade.entry_price
            risk = trade.entry_price - trade.initial_stop_price

            if risk > 0:
                r_multiple = price_change / risk
                r_multiples.append(r_multiple)

        if len(r_multiples) == 0:
            return 0.0

        return float(np.mean(r_multiples))

    def profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss).

        Returns:
            Profit factor (1.0 if no losses)
        """
        if len(self.closed_trades) == 0:
            return 0.0

        gross_profit = 0.0
        gross_loss = 0.0

        for trade in self.closed_trades:
            if trade.realized_pnl > 0:
                gross_profit += trade.realized_pnl
            else:
                gross_loss += abs(trade.realized_pnl)

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def correlation_to_benchmark(self) -> Optional[float]:
        """Calculate correlation to benchmark returns.

        Returns:
            Correlation coefficient, or None if benchmark not provided
        """
        if self.benchmark_returns is None or len(self.daily_returns) == 0:
            return None

        if len(self.daily_returns) != len(self.benchmark_returns):
            return None

        # Calculate correlation
        correlation = np.corrcoef(self.daily_returns, self.benchmark_returns)[0, 1]

        return correlation if not np.isnan(correlation) else None

    def percentile_daily_loss(self, percentile: float = 99.0) -> float:
        """Calculate percentile daily loss.

        Args:
            percentile: Percentile to calculate (default: 99.0 for 99th percentile)

        Returns:
            Percentile daily loss as decimal (e.g., 0.05 for 5%)
        """
        if len(self.daily_returns) == 0:
            return 0.0

        # Get negative returns (losses)
        losses = self.daily_returns[self.daily_returns < 0]

        if len(losses) == 0:
            return 0.0

        # Calculate percentile (convert to positive for loss)
        percentile_loss = np.percentile(np.abs(losses), percentile)

        return float(percentile_loss)

    # Tertiary Metrics

    def recovery_factor(self) -> float:
        """Calculate recovery factor (net profit / max drawdown).

        Returns:
            Recovery factor
        """
        max_dd = self.max_drawdown()
        if max_dd == 0:
            return 0.0

        if len(self.equity_curve) < 2:
            return 0.0

        net_profit = self.equity_curve[-1] - self.equity_curve[0]

        return net_profit / (self.equity_curve[0] * max_dd) if max_dd > 0 else 0.0

    def drawdown_duration(self) -> int:
        """Calculate maximum drawdown duration in days.

        Returns:
            Maximum number of consecutive days in drawdown
        """
        if len(self.equity_curve) == 0:
            return 0

        # Calculate running maximum
        running_max = np.maximum.accumulate(self.equity_curve)

        # Calculate drawdown periods
        in_drawdown = self.equity_curve < running_max

        # Find longest consecutive drawdown period
        max_duration = 0
        current_duration = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration

    def turnover(self) -> float:
        """Calculate turnover (trades per month).

        Returns:
            Average number of trades per month
        """
        if len(self.daily_returns) == 0:
            return 0.0

        total_trades = self.total_trades()

        # Calculate number of months
        num_days = len(self.daily_returns)
        months = num_days / 21.0  # ~21 trading days per month

        if months == 0:
            return 0.0

        return total_trades / months

    def average_holding_period(self) -> float:
        """Calculate average holding period in days.

        Returns:
            Average holding period in days
        """
        if len(self.closed_trades) == 0:
            return 0.0

        holding_periods = []
        for trade in self.closed_trades:
            if trade.entry_date is not None and trade.exit_date is not None:
                holding_days = (trade.exit_date - trade.entry_date).days
                if holding_days > 0:
                    holding_periods.append(holding_days)

        if len(holding_periods) == 0:
            return 0.0

        return float(np.mean(holding_periods))

    def max_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losses.

        Returns:
            Maximum number of consecutive losing trades
        """
        if len(self.closed_trades) == 0:
            return 0

        max_consecutive = 0
        current_consecutive = 0

        for trade in self.closed_trades:
            if trade.realized_pnl < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def win_rate(self) -> float:
        """Calculate win rate (percentage of winning trades).

        Returns:
            Win rate as decimal (e.g., 0.55 for 55%)
        """
        if len(self.closed_trades) == 0:
            return 0.0

        winning_trades = sum(1 for trade in self.closed_trades if trade.realized_pnl > 0)

        return winning_trades / len(self.closed_trades)

    def compute_all_metrics(self) -> Dict[str, float]:
        """Compute all metrics and return as dictionary.

        Returns:
            Dictionary of all metrics
        """
        metrics = {
            # Primary
            "sharpe_ratio": self.sharpe_ratio(),
            "max_drawdown": self.max_drawdown(),
            "calmar_ratio": self.calmar_ratio(),
            "total_trades": float(self.total_trades()),
            # Secondary
            "expectancy": self.expectancy(),
            "profit_factor": self.profit_factor(),
            "correlation_to_benchmark": self.correlation_to_benchmark() or 0.0,
            "percentile_99_daily_loss": self.percentile_daily_loss(99.0),
            # Tertiary
            "recovery_factor": self.recovery_factor(),
            "drawdown_duration": float(self.drawdown_duration()),
            "turnover": self.turnover(),
            "average_holding_period": self.average_holding_period(),
            "max_consecutive_losses": float(self.max_consecutive_losses()),
            "win_rate": self.win_rate(),
        }

        return metrics

"""Portfolio analytics including risk attribution and performance attribution."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class RiskAttribution:
    """Risk attribution breakdown by asset."""

    asset: str
    weight: float  # Portfolio weight
    contribution_to_risk: float  # Contribution to portfolio volatility
    contribution_pct: float  # Percentage contribution to total risk
    marginal_risk: float  # Marginal contribution to risk
    standalone_volatility: float  # Standalone asset volatility


@dataclass
class PerformanceAttribution:
    """Performance attribution breakdown."""

    asset: str
    weight: float  # Portfolio weight
    asset_return: float  # Asset return over period
    contribution_to_return: float  # Weight * return
    contribution_pct: float  # Percentage contribution to total return
    selection_effect: Optional[float] = None  # Selection effect (active vs benchmark)
    allocation_effect: Optional[float] = None  # Allocation effect (weight vs benchmark)


@dataclass
class PortfolioAnalytics:
    """Comprehensive portfolio analytics."""

    # Basic metrics
    total_return: float  # Total return over period
    annualized_return: float  # Annualized return
    volatility: float  # Annualized volatility
    sharpe_ratio: Optional[float]  # Sharpe ratio
    max_drawdown: float  # Maximum drawdown
    calmar_ratio: Optional[float]  # Return / Max DD

    # Risk metrics
    var_95: Optional[float] = None  # 95% Value at Risk
    cvar_95: Optional[float] = None  # 95% Conditional VaR (Expected Shortfall)
    beta: Optional[float] = None  # Beta vs benchmark (if provided)
    alpha: Optional[float] = None  # Alpha vs benchmark (if provided)
    tracking_error: Optional[float] = None  # Tracking error vs benchmark

    # Risk attribution
    risk_attribution: List[RiskAttribution] = field(default_factory=list)

    # Performance attribution
    performance_attribution: List[PerformanceAttribution] = field(default_factory=list)

    # Concentration metrics
    herfindahl_index: Optional[float] = None  # Concentration index
    effective_positions: Optional[float] = None  # Effective number of positions
    largest_position_pct: Optional[float] = None  # Largest position weight

    # Turnover
    turnover: Optional[float] = None  # Portfolio turnover


class PortfolioAnalyticsCalculator:
    """Calculate comprehensive portfolio analytics."""

    def __init__(self, risk_free_rate: float = 0.0, trading_days_per_year: int = 252):
        """Initialize analytics calculator.

        Args:
            risk_free_rate: Risk-free rate for Sharpe calculation (default: 0.0)
            trading_days_per_year: Trading days per year for annualization (default: 252)
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year

    def calculate_basic_metrics(
        self, equity_curve: List[float], returns: Optional[List[float]] = None
    ) -> Tuple[float, float, float, Optional[float], float, Optional[float]]:
        """Calculate basic portfolio metrics.

        Args:
            equity_curve: List of equity values over time
            returns: Optional list of daily returns (if None, computed from equity_curve)

        Returns:
            Tuple of (total_return, annualized_return, volatility, sharpe_ratio, max_drawdown, calmar_ratio)
        """
        if len(equity_curve) < 2:
            return 0.0, 0.0, 0.0, None, 0.0, None

        # Compute returns if not provided
        if returns is None:
            returns = []
            for i in range(1, len(equity_curve)):
                ret = (equity_curve[i] / equity_curve[i - 1]) - 1
                returns.append(ret)

        if not returns:
            return 0.0, 0.0, 0.0, None, 0.0, None

        returns_array = np.array(returns)
        n_periods = len(returns)

        # Total return
        total_return = (equity_curve[-1] / equity_curve[0]) - 1

        # Annualized return
        if n_periods > 0:
            annualized_return = (1 + total_return) ** (self.trading_days_per_year / n_periods) - 1
        else:
            annualized_return = 0.0

        # Volatility (annualized)
        if len(returns_array) > 1:
            volatility = np.std(returns_array, ddof=1) * np.sqrt(self.trading_days_per_year)
        else:
            volatility = 0.0

        # Sharpe ratio
        if volatility > 0:
            excess_return = annualized_return - self.risk_free_rate
            sharpe_ratio = excess_return / volatility
        else:
            sharpe_ratio = None

        # Maximum drawdown
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdowns = (equity_array - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Calmar ratio
        if max_drawdown > 0:
            calmar_ratio = annualized_return / max_drawdown
        else:
            calmar_ratio = None

        return total_return, annualized_return, volatility, sharpe_ratio, max_drawdown, calmar_ratio

    def calculate_risk_metrics(
        self, returns: List[float], benchmark_returns: Optional[List[float]] = None, confidence_level: float = 0.95
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Calculate risk metrics including VaR, CVaR, beta, alpha, tracking error.

        Args:
            returns: Portfolio returns
            benchmark_returns: Optional benchmark returns for beta/alpha/tracking error
            confidence_level: Confidence level for VaR/CVaR (default: 0.95)

        Returns:
            Tuple of (var, cvar, beta, alpha, tracking_error)
        """
        if not returns:
            return None, None, None, None, None

        returns_array = np.array(returns)

        # Value at Risk (VaR) - percentile method
        var = np.percentile(returns_array, (1 - confidence_level) * 100)
        var = abs(var)  # Make positive

        # Conditional VaR (Expected Shortfall)
        var_threshold = np.percentile(returns_array, (1 - confidence_level) * 100)
        tail_returns = returns_array[returns_array <= var_threshold]
        cvar = abs(np.mean(tail_returns)) if len(tail_returns) > 0 else None

        # Beta, alpha, tracking error (if benchmark provided)
        beta = None
        alpha = None
        tracking_error = None

        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            benchmark_array = np.array(benchmark_returns)

            # Remove NaN or infinite values
            valid_mask = np.isfinite(returns_array) & np.isfinite(benchmark_array)
            if np.sum(valid_mask) > 1:
                clean_returns = returns_array[valid_mask]
                clean_benchmark = benchmark_array[valid_mask]

                # Beta (covariance / variance of benchmark)
                if np.var(clean_benchmark) > 0:
                    beta = np.cov(clean_returns, clean_benchmark)[0, 1] / np.var(clean_benchmark)

                # Alpha (excess return after adjusting for beta)
                if beta is not None:
                    expected_return = beta * np.mean(clean_benchmark)
                    alpha = np.mean(clean_returns) - expected_return
                    # Annualize alpha
                    alpha = alpha * self.trading_days_per_year

                # Tracking error (std of active returns)
                active_returns = clean_returns - clean_benchmark
                tracking_error = np.std(active_returns, ddof=1) * np.sqrt(self.trading_days_per_year)

        return var, cvar, beta, alpha, tracking_error

    def calculate_risk_attribution(
        self, weights: Dict[str, float], returns_data: pd.DataFrame, portfolio_volatility: Optional[float] = None
    ) -> List[RiskAttribution]:
        """Calculate risk attribution by asset.

        Args:
            weights: Portfolio weights (symbol -> weight)
            returns_data: DataFrame with columns as symbols, rows as dates (returns)
            portfolio_volatility: Optional pre-calculated portfolio volatility

        Returns:
            List of RiskAttribution objects
        """
        if returns_data.empty:
            return []

        symbols = list(weights.keys())
        if not symbols:
            return []

        # Filter to symbols we have weights for
        available_symbols = [s for s in symbols if s in returns_data.columns]
        if not available_symbols:
            return []

        # Calculate covariance matrix
        cov_matrix = returns_data[available_symbols].cov().values * self.trading_days_per_year

        # Calculate portfolio weights array
        weights_array = np.array([weights[s] for s in available_symbols])

        # Calculate portfolio volatility if not provided
        if portfolio_volatility is None:
            portfolio_vol = np.sqrt(np.dot(weights_array, np.dot(cov_matrix, weights_array)))
        else:
            portfolio_vol = portfolio_volatility

        if portfolio_vol == 0:
            return []

        # Calculate risk contributions
        # Marginal contribution to risk = d(vol)/d(w_i) = (Cov * w) / vol
        marginal_contrib = np.dot(cov_matrix, weights_array) / portfolio_vol
        risk_contributions = weights_array * marginal_contrib

        # Standalone volatilities
        standalone_vols = np.sqrt(np.diag(cov_matrix))

        attribution = []
        for i, symbol in enumerate(available_symbols):
            attribution.append(
                RiskAttribution(
                    asset=symbol,
                    weight=weights[symbol],
                    contribution_to_risk=risk_contributions[i],
                    contribution_pct=risk_contributions[i] / portfolio_vol if portfolio_vol > 0 else 0.0,
                    marginal_risk=marginal_contrib[i],
                    standalone_volatility=standalone_vols[i],
                )
            )

        return attribution

    def calculate_performance_attribution(
        self,
        weights: Dict[str, float],
        asset_returns: Dict[str, float],
        portfolio_return: Optional[float] = None,
        benchmark_weights: Optional[Dict[str, float]] = None,
        benchmark_returns: Optional[Dict[str, float]] = None,
    ) -> List[PerformanceAttribution]:
        """Calculate performance attribution by asset.

        Args:
            weights: Portfolio weights (symbol -> weight)
            asset_returns: Asset returns over period (symbol -> return)
            portfolio_return: Optional pre-calculated portfolio return
            benchmark_weights: Optional benchmark weights for allocation effect
            benchmark_returns: Optional benchmark returns for selection effect

        Returns:
            List of PerformanceAttribution objects
        """
        if not weights or not asset_returns:
            return []

        # Calculate portfolio return if not provided
        if portfolio_return is None:
            portfolio_return = sum(
                weights.get(s, 0.0) * asset_returns.get(s, 0.0) for s in set(weights.keys()) | set(asset_returns.keys())
            )

        if portfolio_return == 0:
            return []

        attribution = []
        for symbol in set(weights.keys()) | set(asset_returns.keys()):
            weight = weights.get(symbol, 0.0)
            asset_return = asset_returns.get(symbol, 0.0)

            # Contribution to return
            contribution = weight * asset_return
            contribution_pct = contribution / portfolio_return if portfolio_return != 0 else 0.0

            # Selection and allocation effects (if benchmark provided)
            selection_effect = None
            allocation_effect = None

            if benchmark_weights is not None and benchmark_returns is not None:
                bench_weight = benchmark_weights.get(symbol, 0.0)
                bench_return = benchmark_returns.get(symbol, 0.0)

                # Selection effect: (asset_return - bench_return) * bench_weight
                selection_effect = (asset_return - bench_return) * bench_weight

                # Allocation effect: (weight - bench_weight) * bench_return
                allocation_effect = (weight - bench_weight) * bench_return

            attribution.append(
                PerformanceAttribution(
                    asset=symbol,
                    weight=weight,
                    asset_return=asset_return,
                    contribution_to_return=contribution,
                    contribution_pct=contribution_pct,
                    selection_effect=selection_effect,
                    allocation_effect=allocation_effect,
                )
            )

        return attribution

    def calculate_concentration_metrics(self, weights: Dict[str, float]) -> Tuple[float, float, float]:
        """Calculate concentration metrics.

        Args:
            weights: Portfolio weights (symbol -> weight)

        Returns:
            Tuple of (herfindahl_index, effective_positions, largest_position_pct)
        """
        if not weights:
            return 0.0, 0.0, 0.0

        weights_array = np.array(list(weights.values()))
        weights_array = weights_array[weights_array > 0]  # Only non-zero weights

        if len(weights_array) == 0:
            return 0.0, 0.0, 0.0

        # Herfindahl index (sum of squared weights)
        herfindahl = np.sum(weights_array**2)

        # Effective number of positions (1 / Herfindahl)
        effective_positions = 1.0 / herfindahl if herfindahl > 0 else 0.0

        # Largest position percentage
        largest_position_pct = np.max(weights_array)

        return herfindahl, effective_positions, largest_position_pct

    def calculate_turnover(self, prev_weights: Dict[str, float], curr_weights: Dict[str, float]) -> float:
        """Calculate portfolio turnover.

        Turnover = 0.5 * sum(|w_i_new - w_i_old|)
        (Factor of 0.5 because each trade is counted once, not twice)

        Args:
            prev_weights: Previous period weights
            curr_weights: Current period weights

        Returns:
            Turnover rate (0.0 to 1.0+)
        """
        all_symbols = set(prev_weights.keys()) | set(curr_weights.keys())
        turnover = 0.0

        for symbol in all_symbols:
            prev_w = prev_weights.get(symbol, 0.0)
            curr_w = curr_weights.get(symbol, 0.0)
            turnover += abs(curr_w - prev_w)

        return 0.5 * turnover

    def calculate_full_analytics(
        self,
        equity_curve: List[float],
        returns: List[float],
        weights: Dict[str, float],
        returns_data: Optional[pd.DataFrame] = None,
        asset_returns: Optional[Dict[str, float]] = None,
        benchmark_returns: Optional[List[float]] = None,
        benchmark_weights: Optional[Dict[str, float]] = None,
        prev_weights: Optional[Dict[str, float]] = None,
    ) -> PortfolioAnalytics:
        """Calculate comprehensive portfolio analytics.

        Args:
            equity_curve: List of equity values over time
            returns: List of daily portfolio returns
            weights: Current portfolio weights
            returns_data: Optional DataFrame of asset returns for risk attribution
            asset_returns: Optional dict of asset returns for performance attribution
            benchmark_returns: Optional benchmark returns for beta/alpha/tracking error
            benchmark_weights: Optional benchmark weights for attribution
            prev_weights: Optional previous weights for turnover calculation

        Returns:
            PortfolioAnalytics object with all metrics
        """
        # Basic metrics
        total_return, ann_return, vol, sharpe, max_dd, calmar = self.calculate_basic_metrics(equity_curve, returns)

        # Risk metrics
        var, cvar, beta, alpha, tracking_error = self.calculate_risk_metrics(returns, benchmark_returns)

        # Risk attribution
        risk_attribution = []
        if returns_data is not None:
            risk_attribution = self.calculate_risk_attribution(weights, returns_data, vol)

        # Performance attribution
        performance_attribution = []
        if asset_returns is not None:
            # Convert benchmark_returns from list to dict if needed
            benchmark_returns_dict: Optional[Dict[str, float]] = None
            if benchmark_returns is not None:
                if not isinstance(benchmark_returns, list):
                    benchmark_returns_dict = benchmark_returns
            performance_attribution = self.calculate_performance_attribution(
                weights, asset_returns, total_return, benchmark_weights, benchmark_returns_dict
            )

        # Concentration metrics
        herfindahl, effective_positions, largest_pct = self.calculate_concentration_metrics(weights)

        # Turnover
        turnover = None
        if prev_weights is not None:
            turnover = self.calculate_turnover(prev_weights, weights)

        return PortfolioAnalytics(
            total_return=total_return,
            annualized_return=ann_return,
            volatility=vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            var_95=var,
            cvar_95=cvar,
            beta=beta,
            alpha=alpha,
            tracking_error=tracking_error,
            risk_attribution=risk_attribution,
            performance_attribution=performance_attribution,
            herfindahl_index=herfindahl,
            effective_positions=effective_positions,
            largest_position_pct=largest_pct,
            turnover=turnover,
        )

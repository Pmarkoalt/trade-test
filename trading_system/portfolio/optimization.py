"""Portfolio optimization using Markowitz mean-variance and risk parity methods."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgError
from scipy.optimize import differential_evolution, minimize


@dataclass
class OptimizationResult:
    """Result of portfolio optimization."""

    weights: Dict[str, float]  # symbol -> weight (0.0 to 1.0)
    expected_return: Optional[float] = None  # Portfolio expected return
    volatility: Optional[float] = None  # Portfolio volatility (annualized)
    sharpe_ratio: Optional[float] = None  # Sharpe ratio (if risk-free rate provided)
    method: str = "markowitz"  # Optimization method used

    def to_array(self, symbols: List[str]) -> np.ndarray:
        """Convert weights dict to array in symbol order.

        Args:
            symbols: Ordered list of symbols

        Returns:
            Numpy array of weights
        """
        return np.array([self.weights.get(s, 0.0) for s in symbols])


@dataclass
class RebalanceTarget:
    """Target allocation for rebalancing."""

    symbol: str
    target_weight: float  # Target weight (0.0 to 1.0)
    current_weight: float  # Current weight in portfolio
    target_notional: float  # Target notional value
    current_notional: float  # Current notional value
    delta_notional: float  # Required change (positive = buy, negative = sell)
    delta_quantity: int  # Required quantity change (rounded)

    @property
    def is_rebalance_needed(self) -> bool:
        """Check if rebalancing is needed (weight deviation > threshold)."""
        # Default threshold: 5% absolute deviation
        return abs(self.target_weight - self.current_weight) > 0.05


class PortfolioOptimizer:
    """Portfolio optimization using Markowitz mean-variance and risk parity."""

    def __init__(self, risk_free_rate: float = 0.0, optimization_method: str = "markowitz"):
        """Initialize portfolio optimizer.

        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation (default: 0.0)
            optimization_method: "markowitz" or "risk_parity" (default: "markowitz")
        """
        self.risk_free_rate = risk_free_rate
        self.optimization_method = optimization_method

    def optimize_markowitz(
        self,
        returns_data: pd.DataFrame,
        target_return: Optional[float] = None,
        risk_aversion: float = 1.0,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        long_only: bool = True,
    ) -> OptimizationResult:
        """Optimize portfolio using Markowitz mean-variance optimization.

        Maximizes: E[R] - risk_aversion * Var[R]

        Or maximizes Sharpe ratio if target_return is None.

        Args:
            returns_data: DataFrame with columns as symbols, rows as dates (returns)
            target_return: Optional target return (if None, maximizes Sharpe ratio)
            risk_aversion: Risk aversion parameter (higher = more risk averse)
            min_weight: Minimum weight per asset (default: 0.0)
            max_weight: Maximum weight per asset (default: 1.0)
            long_only: If True, enforces long-only constraint (min_weight >= 0)

        Returns:
            OptimizationResult with optimal weights
        """
        if returns_data.empty:
            raise ValueError("returns_data cannot be empty")

        symbols = returns_data.columns.tolist()
        n_assets = len(symbols)

        if n_assets == 0:
            raise ValueError("No assets in returns_data")

        # Calculate expected returns and covariance matrix
        expected_returns = returns_data.mean().values * 252  # Annualized
        cov_matrix = returns_data.cov().values * 252  # Annualized

        # Check for valid covariance matrix
        try:
            np.linalg.cholesky(cov_matrix)
        except LinAlgError:
            # Add small regularization if matrix is not positive definite
            cov_matrix += np.eye(n_assets) * 1e-6

        # Initial weights (equal weight)
        x0 = np.ones(n_assets) / n_assets

        # Constraints: weights sum to 1
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        # Target return constraint (if specified)
        if target_return is not None:
            constraints.append({"type": "eq", "fun": lambda w: np.dot(w, expected_returns) - target_return})

        # Bounds
        bounds = [(min_weight, max_weight) for _ in range(n_assets)]
        if long_only:
            bounds = [(max(0.0, min_weight), max_weight) for _ in range(n_assets)]

        # Objective function
        if target_return is None:
            # Maximize Sharpe ratio
            def objective(w):
                port_return = np.dot(w, expected_returns)
                port_vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
                if port_vol == 0:
                    return -np.inf
                sharpe = (port_return - self.risk_free_rate) / port_vol
                return -sharpe  # Minimize negative Sharpe

        else:
            # Minimize variance for given return
            def objective(w):
                return np.dot(w, np.dot(cov_matrix, w))

        # Optimize
        try:
            result = minimize(
                objective, x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 1000, "ftol": 1e-9}
            )

            if not result.success:
                # Fallback to differential evolution for better global search
                try:
                    result = differential_evolution(objective, bounds=bounds, constraints=constraints, maxiter=1000, seed=42)
                    # Check if differential_evolution succeeded
                    if not hasattr(result, "x") or result.x is None:
                        raise ValueError("Differential evolution failed")
                except Exception:
                    # If differential evolution also fails, use equal weights
                    weights = {s: 1.0 / n_assets for s in symbols}
                    return OptimizationResult(weights=weights, method="markowitz_fallback")
        except Exception as e:
            # Fallback to equal weights if optimization fails
            weights = {s: 1.0 / n_assets for s in symbols}
            return OptimizationResult(weights=weights, method="markowitz_fallback")

        # Extract weights - ensure result.x exists
        if not hasattr(result, "x") or result.x is None:
            weights = {s: 1.0 / n_assets for s in symbols}
            return OptimizationResult(weights=weights, method="markowitz_fallback")

        optimal_weights = result.x
        weights_dict = {symbols[i]: max(0.0, optimal_weights[i]) for i in range(n_assets)}

        # Normalize to ensure sum = 1.0
        total = sum(weights_dict.values())
        if total > 0:
            weights_dict = {k: v / total for k, v in weights_dict.items()}
        else:
            # Fallback to equal weights
            weights_dict = {s: 1.0 / n_assets for s in symbols}

        # Calculate portfolio metrics
        weights_array = np.array([weights_dict[s] for s in symbols])
        port_return = np.dot(weights_array, expected_returns)
        port_vol = np.sqrt(np.dot(weights_array, np.dot(cov_matrix, weights_array)))
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0.0

        return OptimizationResult(
            weights=weights_dict, expected_return=port_return, volatility=port_vol, sharpe_ratio=sharpe, method="markowitz"
        )

    def optimize_risk_parity(
        self, returns_data: pd.DataFrame, min_weight: float = 0.0, max_weight: float = 1.0, long_only: bool = True
    ) -> OptimizationResult:
        """Optimize portfolio using risk parity (equal risk contribution).

        Risk parity equalizes the risk contribution of each asset to the portfolio.
        Each asset contributes equally to portfolio risk.

        Args:
            returns_data: DataFrame with columns as symbols, rows as dates (returns)
            min_weight: Minimum weight per asset (default: 0.0)
            max_weight: Maximum weight per asset (default: 1.0)
            long_only: If True, enforces long-only constraint

        Returns:
            OptimizationResult with optimal weights
        """
        if returns_data.empty:
            raise ValueError("returns_data cannot be empty")

        symbols = returns_data.columns.tolist()
        n_assets = len(symbols)

        if n_assets == 0:
            raise ValueError("No assets in returns_data")

        # Calculate covariance matrix
        cov_matrix = returns_data.cov().values * 252  # Annualized

        # Check for valid covariance matrix
        try:
            np.linalg.cholesky(cov_matrix)
        except LinAlgError:
            cov_matrix += np.eye(n_assets) * 1e-6

        # Initial weights (equal weight)
        x0 = np.ones(n_assets) / n_assets

        # Bounds
        bounds = [(min_weight, max_weight) for _ in range(n_assets)]
        if long_only:
            bounds = [(max(0.0, min_weight), max_weight) for _ in range(n_assets)]

        # Constraint: weights sum to 1
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        # Objective: minimize sum of squared differences in risk contributions
        def risk_contribution(w):
            """Calculate risk contribution of each asset."""
            port_vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
            if port_vol == 0:
                return np.zeros(n_assets)
            # Marginal contribution to risk
            marginal_contrib = np.dot(cov_matrix, w) / port_vol
            # Risk contribution
            contrib = w * marginal_contrib
            return contrib

        def objective(w):
            """Minimize variance of risk contributions."""
            contrib = risk_contribution(w)
            target_contrib = np.ones(n_assets) / n_assets  # Equal risk contribution
            return np.sum((contrib - target_contrib) ** 2)

        # Optimize
        try:
            result = minimize(
                objective, x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 1000, "ftol": 1e-9}
            )

            if not result.success:
                try:
                    result = differential_evolution(objective, bounds=bounds, constraints=constraints, maxiter=1000, seed=42)
                    # Check if differential_evolution succeeded
                    if not hasattr(result, "x") or result.x is None:
                        raise ValueError("Differential evolution failed")
                except Exception:
                    # If differential evolution also fails, use equal weights
                    weights = {s: 1.0 / n_assets for s in symbols}
                    return OptimizationResult(weights=weights, method="risk_parity_fallback")
        except Exception as e:
            # Fallback to equal weights
            weights = {s: 1.0 / n_assets for s in symbols}
            return OptimizationResult(weights=weights, method="risk_parity_fallback")

        # Extract weights - ensure result.x exists
        if not hasattr(result, "x") or result.x is None:
            weights = {s: 1.0 / n_assets for s in symbols}
            return OptimizationResult(weights=weights, method="risk_parity_fallback")

        optimal_weights = result.x
        weights_dict = {symbols[i]: max(0.0, optimal_weights[i]) for i in range(n_assets)}

        # Normalize
        total = sum(weights_dict.values())
        if total > 0:
            weights_dict = {k: v / total for k, v in weights_dict.items()}
        else:
            weights_dict = {s: 1.0 / n_assets for s in symbols}

        # Calculate portfolio metrics
        weights_array = np.array([weights_dict[s] for s in symbols])
        expected_returns = returns_data.mean().values * 252
        port_return = np.dot(weights_array, expected_returns)
        port_vol = np.sqrt(np.dot(weights_array, np.dot(cov_matrix, weights_array)))
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0.0

        return OptimizationResult(
            weights=weights_dict, expected_return=port_return, volatility=port_vol, sharpe_ratio=sharpe, method="risk_parity"
        )

    def optimize(self, returns_data: pd.DataFrame, **kwargs) -> OptimizationResult:
        """Optimize portfolio using specified method.

        Args:
            returns_data: DataFrame with columns as symbols, rows as dates (returns)
            **kwargs: Additional arguments passed to optimization method

        Returns:
            OptimizationResult with optimal weights
        """
        if self.optimization_method == "markowitz":
            return self.optimize_markowitz(returns_data, **kwargs)
        elif self.optimization_method == "risk_parity":
            return self.optimize_risk_parity(returns_data, **kwargs)
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")


def compute_rebalance_targets(
    portfolio_equity: float,
    current_positions: Dict[str, float],  # symbol -> current_notional
    target_weights: Dict[str, float],  # symbol -> target_weight
    current_prices: Dict[str, float],  # symbol -> current_price
    rebalance_threshold: float = 0.05,  # 5% deviation triggers rebalance
    min_trade_size: float = 100.0,  # Minimum notional trade size
) -> List[RebalanceTarget]:
    """Compute rebalancing targets for current portfolio.

    Args:
        portfolio_equity: Current portfolio equity
        current_positions: Current position notional values
        target_weights: Target weights for each symbol
        current_prices: Current market prices
        rebalance_threshold: Minimum weight deviation to trigger rebalance (default: 0.05)
        min_trade_size: Minimum notional trade size (default: $100)

    Returns:
        List of RebalanceTarget objects
    """
    targets = []

    # Calculate total current notional (including positions not in target_weights)
    total_current_notional = sum(current_positions.values())

    # For each target symbol
    for symbol, target_weight in target_weights.items():
        if symbol not in current_prices:
            continue  # Skip if no price data

        target_notional = portfolio_equity * target_weight
        current_notional = current_positions.get(symbol, 0.0)
        current_weight = current_notional / portfolio_equity if portfolio_equity > 0 else 0.0

        delta_notional = target_notional - current_notional

        # Calculate quantity change
        price = current_prices[symbol]
        if price > 0:
            delta_quantity = int(np.round(delta_notional / price))
        else:
            delta_quantity = 0

        target = RebalanceTarget(
            symbol=symbol,
            target_weight=target_weight,
            current_weight=current_weight,
            target_notional=target_notional,
            current_notional=current_notional,
            delta_notional=delta_notional,
            delta_quantity=delta_quantity,
        )

        # Only include if rebalancing is needed and trade size is significant
        if abs(target.current_weight - target.target_weight) >= rebalance_threshold and abs(delta_notional) >= min_trade_size:
            targets.append(target)

    # Handle positions that should be closed (not in target_weights)
    for symbol, current_notional in current_positions.items():
        if symbol not in target_weights and symbol in current_prices:
            current_weight = current_notional / portfolio_equity if portfolio_equity > 0 else 0.0
            price = current_prices[symbol]
            delta_quantity = -int(np.round(current_notional / price)) if price > 0 else 0

            target = RebalanceTarget(
                symbol=symbol,
                target_weight=0.0,
                current_weight=current_weight,
                target_notional=0.0,
                current_notional=current_notional,
                delta_notional=-current_notional,
                delta_quantity=delta_quantity,
            )
            targets.append(target)

    return targets


def should_rebalance(current_weights: Dict[str, float], target_weights: Dict[str, float], threshold: float = 0.05) -> bool:
    """Check if portfolio needs rebalancing.

    Args:
        current_weights: Current portfolio weights
        target_weights: Target portfolio weights
        threshold: Minimum weight deviation to trigger rebalance (default: 0.05)

    Returns:
        True if rebalancing is needed
    """
    # Check all target symbols
    for symbol, target_weight in target_weights.items():
        current_weight = current_weights.get(symbol, 0.0)
        if abs(current_weight - target_weight) >= threshold:
            return True

    # Check for positions that should be closed
    for symbol in current_weights:
        if symbol not in target_weights and current_weights[symbol] > threshold:
            return True

    return False

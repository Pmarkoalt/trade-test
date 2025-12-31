"""Model evaluation enhancements for trading system.

This module provides:
- Backtesting-specific metrics
- Model performance tracking
- Feature importance analysis
- Prediction confidence intervals
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from trading_system.ml.models import MLModel


def compute_backtest_metrics(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
    positions: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute backtesting-specific metrics.

    Args:
        predictions: Model predictions (expected returns or signal strength)
        actual_returns: Actual returns
        positions: Optional position sizes (if None, assumes binary positions)

    Returns:
        Dictionary of backtesting metrics
    """
    if len(predictions) != len(actual_returns):
        raise ValueError("Predictions and actual_returns must have same length")

    if positions is None:
        # Binary positions: 1 if prediction > 0, -1 if < 0, 0 otherwise
        positions = np.sign(predictions)

    # Portfolio returns
    portfolio_returns = positions * actual_returns

    # Basic metrics
    total_return = np.prod(1 + portfolio_returns) - 1
    sharpe_ratio = (
        np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252) if np.std(portfolio_returns) > 0 else 0.0
    )

    # Maximum drawdown
    cumulative_returns = np.cumprod(1 + portfolio_returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = np.min(drawdown)

    # Win rate
    winning_trades = np.sum(portfolio_returns > 0)
    total_trades = np.sum(positions != 0)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

    # Average win/loss
    winning_returns = portfolio_returns[portfolio_returns > 0]
    losing_returns = portfolio_returns[portfolio_returns < 0]
    avg_win = np.mean(winning_returns) if len(winning_returns) > 0 else 0.0
    avg_loss = np.mean(losing_returns) if len(losing_returns) > 0 else 0.0

    # Profit factor
    total_profit = np.sum(winning_returns) if len(winning_returns) > 0 else 0.0
    total_loss = abs(np.sum(losing_returns)) if len(losing_returns) > 0 else 0.0
    profit_factor = total_profit / total_loss if total_loss > 0 else np.inf

    # Calmar ratio
    calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    return {
        "total_return": float(total_return),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "profit_factor": float(profit_factor),
        "calmar_ratio": float(calmar_ratio),
        "total_trades": int(total_trades),
    }


def compute_feature_importance(
    model: MLModel,
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "permutation",  # "permutation" or "builtin"
    n_repeats: int = 10,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """Compute feature importance.

    Args:
        model: Trained MLModel instance
        X: Feature matrix
        y: Target vector
        method: Importance method ("permutation" or "builtin")
        n_repeats: Number of repeats for permutation importance
        random_state: Random state for reproducibility

    Returns:
        Dictionary mapping feature names to importance scores
    """
    if method == "builtin":
        # Use model's built-in feature importance
        if hasattr(model._model, "feature_importances_"):
            importances = model._model.feature_importances_
            feature_names = list(X.columns)
            return dict(zip(feature_names, importances))
        else:
            # Fallback to permutation importance
            method = "permutation"

    if method == "permutation":
        from sklearn.inspection import permutation_importance

        # Compute permutation importance
        perm_importance = permutation_importance(
            model._model,
            X,
            y,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring="r2" if len(np.unique(y)) > 10 else "accuracy",
        )

        feature_names = list(X.columns)
        importances = perm_importance.importances_mean

        return dict(zip(feature_names, importances))

    raise ValueError(f"Unknown importance method: {method}")


def compute_confidence_intervals(
    model: MLModel,
    X: pd.DataFrame,
    confidence_level: float = 0.95,
    method: str = "bootstrap",  # "bootstrap" or "quantile"
    n_bootstrap: int = 100,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute prediction confidence intervals.

    Args:
        model: Trained MLModel instance
        X: Feature matrix
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        method: Method for computing intervals ("bootstrap" or "quantile")
        n_bootstrap: Number of bootstrap samples
        random_state: Random state for reproducibility

    Returns:
        Tuple of (lower_bound, upper_bound) arrays
    """
    if method == "bootstrap":
        np.random.seed(random_state)

        predictions_list = []
        n_samples = len(X)

        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X.iloc[indices]

            # Make predictions
            pred = model.predict(X_boot)
            predictions_list.append(pred)

        predictions_array = np.array(predictions_list)  # Shape: (n_bootstrap, n_samples)

        # Compute percentiles
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bound = np.percentile(predictions_array, lower_percentile, axis=0)
        upper_bound = np.percentile(predictions_array, upper_percentile, axis=0)

        return lower_bound, upper_bound

    elif method == "quantile":
        # For models that support quantile regression
        # This is a placeholder - would need quantile regression models
        predictions = model.predict(X)
        std_estimate = np.std(predictions)

        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin = z_score * std_estimate

        lower_bound = predictions - margin
        upper_bound = predictions + margin

        return lower_bound, upper_bound

    raise ValueError(f"Unknown confidence interval method: {method}")


class ModelPerformanceTracker:
    """Track model performance over time."""

    def __init__(self):
        """Initialize performance tracker."""
        self._metrics_history: List[Dict[str, float]] = []
        self._predictions_history: List[np.ndarray] = []
        self._actuals_history: List[np.ndarray] = []

    def update(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update performance tracking.

        Args:
            predictions: Model predictions
            actuals: Actual values
            metrics: Optional pre-computed metrics
        """
        self._predictions_history.append(predictions.copy())
        self._actuals_history.append(actuals.copy())

        if metrics is None:
            # Compute basic metrics
            if len(np.unique(actuals)) <= 10:
                # Classification metrics
                from sklearn.metrics import accuracy_score

                metrics = {
                    "accuracy": accuracy_score(actuals, predictions),
                }
            else:
                # Regression metrics
                from sklearn.metrics import mean_squared_error, r2_score

                metrics = {
                    "mse": mean_squared_error(actuals, predictions),
                    "r2": r2_score(actuals, predictions),
                }

        self._metrics_history.append(metrics)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance over time.

        Returns:
            Dictionary with performance summary
        """
        if not self._metrics_history:
            return {}

        # Aggregate metrics
        summary = {}
        for key in self._metrics_history[0].keys():
            values = [m[key] for m in self._metrics_history if key in m]
            if values:
                summary[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "latest": values[-1],
                }

        return summary

    def get_trend(self, metric_name: str) -> str:
        """Get trend for a specific metric.

        Args:
            metric_name: Name of metric

        Returns:
            Trend string ("improving", "degrading", or "stable")
        """
        values = [m[metric_name] for m in self._metrics_history if metric_name in m]

        if len(values) < 2:
            return "stable"

        # Compare recent vs earlier performance
        recent = np.mean(values[-5:]) if len(values) >= 5 else values[-1]
        earlier = np.mean(values[:5]) if len(values) >= 5 else values[0]

        threshold = 0.05  # 5% change threshold
        if recent > earlier * (1 + threshold):
            return "improving"
        elif recent < earlier * (1 - threshold):
            return "degrading"
        else:
            return "stable"

    def clear(self) -> None:
        """Clear all tracking data."""
        self._metrics_history.clear()
        self._predictions_history.clear()
        self._actuals_history.clear()

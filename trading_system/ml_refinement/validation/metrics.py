"""Evaluation metrics for ML models."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calculate classification metrics.

    Args:
        y_true: True labels (0/1).
        y_pred: Predicted labels (0/1).
        y_proba: Predicted probabilities (optional).

    Returns:
        Dictionary of metrics.
    """
    metrics = {}

    # Basic metrics
    n = len(y_true)
    if n == 0:
        return {"error": "No samples"}

    # Confusion matrix elements
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Accuracy
    metrics["accuracy"] = (tp + tn) / n

    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1"] = f1

    # Class balance
    metrics["positive_rate"] = np.mean(y_true)
    metrics["pred_positive_rate"] = np.mean(y_pred)

    # AUC-ROC if probabilities available
    if y_proba is not None:
        metrics["auc"] = calculate_auc(y_true, y_proba)
        metrics["log_loss"] = calculate_log_loss(y_true, y_proba)

        # Brier score
        metrics["brier"] = np.mean((y_proba - y_true) ** 2)

    return metrics


def calculate_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Calculate AUC-ROC.

    Simple implementation without sklearn dependency.
    """
    # Sort by predicted probability
    desc_score_indices = np.argsort(y_proba)[::-1]
    y_true_sorted = y_true[desc_score_indices]

    # Calculate ROC curve points
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)

    total_pos = np.sum(y_true)
    total_neg = len(y_true) - total_pos

    if total_pos == 0 or total_neg == 0:
        return 0.5

    tpr = tps / total_pos
    fpr = fps / total_neg

    # AUC via trapezoidal rule
    auc = np.trapz(tpr, fpr)
    return auc


def calculate_log_loss(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """Calculate log loss (cross-entropy)."""
    # Clip probabilities
    y_proba = np.clip(y_proba, eps, 1 - eps)

    # Log loss
    loss = -np.mean(y_true * np.log(y_proba) + (1 - y_true) * np.log(1 - y_proba))
    return loss


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Dictionary of metrics.
    """
    metrics = {}

    if len(y_true) == 0:
        return {"error": "No samples"}

    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)
    metrics["mse"] = mse
    metrics["rmse"] = np.sqrt(mse)

    # Mean Absolute Error
    metrics["mae"] = np.mean(np.abs(y_true - y_pred))

    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    metrics["r2"] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Correlation
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        metrics["correlation"] = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        metrics["correlation"] = 0

    return metrics


def calculate_trading_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calculate trading-specific metrics.

    These metrics evaluate how useful predictions are for trading.
    """
    metrics = {}

    # Filter to high-conviction predictions
    high_conf_mask = y_proba >= threshold

    if np.sum(high_conf_mask) == 0:
        return {"high_conf_count": 0}

    # High confidence accuracy
    y_pred_high = (y_proba[high_conf_mask] >= 0.5).astype(int)
    y_true_high = y_true[high_conf_mask]

    metrics["high_conf_count"] = int(np.sum(high_conf_mask))
    metrics["high_conf_accuracy"] = np.mean(y_pred_high == y_true_high)

    # Expected value calculation
    # Assuming we trade when p > threshold
    # EV = p * avg_win - (1-p) * avg_loss
    # Simplified: use 2:1 risk/reward
    avg_win = 2.0
    avg_loss = 1.0

    ev_per_trade = y_proba[high_conf_mask] * avg_win - (1 - y_proba[high_conf_mask]) * avg_loss
    metrics["expected_value"] = float(np.mean(ev_per_trade))

    # Calibration
    metrics["calibration_error"] = calculate_calibration_error(y_true, y_proba)

    return metrics


def calculate_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Calculate Expected Calibration Error.

    Lower is better - means probabilities are well calibrated.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_proba[mask])
            bin_size = np.sum(mask) / len(y_true)
            ece += bin_size * abs(bin_acc - bin_conf)

    return ece


def calculate_feature_importance_stability(
    importance_history: List[Dict[str, float]],
) -> Dict[str, float]:
    """
    Calculate stability of feature importances across folds.

    High stability means features are consistently important.
    """
    if not importance_history:
        return {}

    # Get all feature names
    all_features = set()
    for imp_dict in importance_history:
        all_features.update(imp_dict.keys())

    stability = {}
    for feature in all_features:
        values = [d.get(feature, 0) for d in importance_history]
        mean_imp = np.mean(values)
        std_imp = np.std(values)

        # Coefficient of variation (lower = more stable)
        cv = std_imp / mean_imp if mean_imp > 0 else float("inf")

        stability[feature] = {
            "mean": mean_imp,
            "std": std_imp,
            "cv": cv,
        }

    return stability

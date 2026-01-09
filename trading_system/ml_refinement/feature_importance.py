"""
Feature Importance Tracking Module

Track and analyze which features drive ML model predictions:
- Model-based importance (tree-based models)
- Permutation importance
- SHAP values (if available)
- Feature correlation analysis
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportance:
    """Feature importance result."""

    feature_name: str
    importance: float
    std: float = 0.0
    rank: int = 0
    category: str = "unknown"


class FeatureImportanceTracker:
    """Track and analyze feature importance over time."""

    def __init__(self):
        self.importance_history: List[Dict[str, float]] = []
        self.current_importance: Dict[str, FeatureImportance] = {}
        self.feature_categories = {
            "price": ["close", "open", "high", "low", "return"],
            "volume": ["volume", "dollar_volume", "adv"],
            "momentum": ["roc", "momentum", "rsi"],
            "trend": ["ma", "sma", "ema", "trend"],
            "volatility": ["atr", "std", "volatility", "vol"],
            "regime": ["regime", "bull", "bear"],
        }

    def extract_from_model(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from a trained model."""
        importance = {}

        # Try different model types
        if hasattr(model, "feature_importances_"):
            # Tree-based models (sklearn)
            imp = model.feature_importances_
            for name, value in zip(feature_names, imp):
                importance[name] = float(value)

        elif hasattr(model, "coef_"):
            # Linear models
            coef = np.abs(model.coef_).flatten()
            for name, value in zip(feature_names, coef):
                importance[name] = float(value)

        elif hasattr(model, "_model") and hasattr(model._model, "feature_importances_"):
            # Wrapped models
            imp = model._model.feature_importances_
            for name, value in zip(feature_names, imp):
                importance[name] = float(value)

        else:
            logger.warning("Could not extract feature importance from model")
            return {}

        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        return importance

    def calculate_permutation_importance(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 10,
    ) -> Dict[str, float]:
        """Calculate permutation importance."""
        try:
            from sklearn.inspection import permutation_importance
        except ImportError:
            logger.warning("sklearn not available for permutation importance")
            return {}

        result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42)

        importance = {}
        for name, imp_mean in zip(X.columns, result.importances_mean):
            importance[name] = float(imp_mean)

        # Normalize
        total = sum(abs(v) for v in importance.values())
        if total > 0:
            importance = {k: abs(v) / total for k, v in importance.items()}

        return importance

    def update_importance(
        self,
        importance: Dict[str, float],
        timestamp: Optional[pd.Timestamp] = None,
    ) -> None:
        """Update importance tracking with new values."""
        self.importance_history.append(importance)

        # Update current importance with ranking
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        self.current_importance = {}
        for rank, (name, value) in enumerate(sorted_features, 1):
            category = self._categorize_feature(name)
            self.current_importance[name] = FeatureImportance(
                feature_name=name,
                importance=value,
                rank=rank,
                category=category,
            )

    def _categorize_feature(self, feature_name: str) -> str:
        """Categorize feature by name."""
        name_lower = feature_name.lower()
        for category, keywords in self.feature_categories.items():
            if any(kw in name_lower for kw in keywords):
                return category
        return "other"

    def get_top_features(self, n: int = 10) -> List[FeatureImportance]:
        """Get top N most important features."""
        sorted_features = sorted(
            self.current_importance.values(),
            key=lambda x: x.importance,
            reverse=True,
        )
        return sorted_features[:n]

    def get_importance_by_category(self) -> Dict[str, float]:
        """Get aggregated importance by category."""
        category_importance = {}

        for fi in self.current_importance.values():
            if fi.category not in category_importance:
                category_importance[fi.category] = 0.0
            category_importance[fi.category] += fi.importance

        return category_importance

    def analyze_stability(self, window: int = 10) -> Dict[str, Dict[str, float]]:
        """Analyze feature importance stability over time."""
        if len(self.importance_history) < window:
            return {}

        recent = self.importance_history[-window:]

        stability = {}
        all_features = set()
        for imp in recent:
            all_features.update(imp.keys())

        for feature in all_features:
            values = [imp.get(feature, 0) for imp in recent]
            stability[feature] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "cv": np.std(values) / np.mean(values) if np.mean(values) > 0 else 0,
                "trend": self._calculate_trend(values),
            }

        return stability

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction (-1 to 1)."""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        # Normalize by mean
        mean_val = np.mean(values)
        if mean_val == 0:
            return 0.0
        return np.clip(slope / mean_val, -1, 1)

    def detect_drift(self, threshold: float = 0.3) -> List[str]:
        """Detect features with significant importance drift."""
        stability = self.analyze_stability()

        drifting = []
        for feature, stats in stability.items():
            if abs(stats["trend"]) > threshold:
                direction = "increasing" if stats["trend"] > 0 else "decreasing"
                drifting.append(f"{feature} ({direction})")

        return drifting

    def generate_report(self) -> str:
        """Generate feature importance report."""
        lines = []
        lines.append("=" * 60)
        lines.append("FEATURE IMPORTANCE REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Top features
        lines.append("TOP 10 FEATURES")
        lines.append("-" * 40)
        for fi in self.get_top_features(10):
            lines.append(f"  {fi.rank:>2}. {fi.feature_name:<25} " f"{fi.importance:>6.2%} [{fi.category}]")
        lines.append("")

        # Category breakdown
        lines.append("IMPORTANCE BY CATEGORY")
        lines.append("-" * 40)
        for category, importance in sorted(
            self.get_importance_by_category().items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            lines.append(f"  {category:<15} {importance:>6.2%}")
        lines.append("")

        # Drift detection
        drifting = self.detect_drift()
        if drifting:
            lines.append("⚠️ FEATURES WITH DRIFT")
            lines.append("-" * 40)
            for feature in drifting:
                lines.append(f"  • {feature}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert current importance to DataFrame."""
        data = []
        for fi in self.current_importance.values():
            data.append(
                {
                    "feature": fi.feature_name,
                    "importance": fi.importance,
                    "rank": fi.rank,
                    "category": fi.category,
                }
            )
        return pd.DataFrame(data).sort_values("rank")

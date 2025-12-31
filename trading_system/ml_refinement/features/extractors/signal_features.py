"""Signal metadata feature extractors."""

from typing import Any, Dict, List

from trading_system.ml_refinement.features.extractors.base_extractor import (
    BaseFeatureExtractor,
)


class SignalMetadataFeatures(BaseFeatureExtractor):
    """Extract features from signal metadata."""

    @property
    def name(self) -> str:
        return "signal_metadata"

    @property
    def category(self) -> str:
        return "signal"

    @property
    def feature_names(self) -> List[str]:
        return [
            "technical_score",
            "news_score",
            "combined_score",
            "conviction_high",
            "conviction_medium",
            "conviction_low",
            "risk_reward_ratio",
            "position_size",
            "is_equity",
            "is_crypto",
            "is_breakout",
            "is_momentum",
        ]

    def extract(self, signal_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract features from signal metadata.

        Args:
            signal_data: Dictionary with signal information including:
                - technical_score
                - news_score
                - combined_score
                - conviction
                - entry_price
                - target_price
                - stop_price
                - position_size_pct
                - asset_class
                - signal_type
        """
        features = {}

        # Scores (normalize to 0-1)
        features["technical_score"] = signal_data.get("technical_score", 0) / 10
        features["news_score"] = (signal_data.get("news_score") or 0) / 10
        features["combined_score"] = signal_data.get("combined_score", 0) / 10

        # Conviction one-hot encoding
        conviction = signal_data.get("conviction", "").upper()
        features["conviction_high"] = 1.0 if conviction == "HIGH" else 0.0
        features["conviction_medium"] = 1.0 if conviction == "MEDIUM" else 0.0
        features["conviction_low"] = 1.0 if conviction == "LOW" else 0.0

        # Risk/reward ratio
        entry = signal_data.get("entry_price", 0)
        target = signal_data.get("target_price", 0)
        stop = signal_data.get("stop_price", 0)

        if entry > 0 and stop > 0 and target > 0:
            risk = abs(entry - stop)
            reward = abs(target - entry)
            if risk > 0:
                features["risk_reward_ratio"] = reward / risk
            else:
                features["risk_reward_ratio"] = 0.0
        else:
            features["risk_reward_ratio"] = 0.0

        # Position size
        features["position_size"] = signal_data.get("position_size_pct", 0)

        # Asset class one-hot
        asset_class = signal_data.get("asset_class", "").lower()
        features["is_equity"] = 1.0 if asset_class == "equity" else 0.0
        features["is_crypto"] = 1.0 if asset_class == "crypto" else 0.0

        # Signal type one-hot
        signal_type = signal_data.get("signal_type", "").lower()
        features["is_breakout"] = 1.0 if "breakout" in signal_type else 0.0
        features["is_momentum"] = 1.0 if "momentum" in signal_type else 0.0

        return features

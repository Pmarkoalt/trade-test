"""Registry of available features."""

from typing import Dict, List, Type

from trading_system.ml_refinement.features.extractors.base_extractor import (
    BaseFeatureExtractor,
)
from trading_system.ml_refinement.features.extractors.technical_features import (
    MomentumFeatures,
    TrendFeatures,
    VolatilityFeatures,
)
from trading_system.ml_refinement.features.extractors.market_features import (
    MarketRegimeFeatures,
)
from trading_system.ml_refinement.features.extractors.signal_features import (
    SignalMetadataFeatures,
)


class FeatureRegistry:
    """
    Registry of available feature extractors.

    Example:
        registry = FeatureRegistry()
        registry.register(CustomExtractor())

        extractor = registry.get("trend_features")
        all_names = registry.get_all_feature_names()
    """

    # Default extractors
    DEFAULT_EXTRACTORS: List[Type[BaseFeatureExtractor]] = [
        TrendFeatures,
        MomentumFeatures,
        VolatilityFeatures,
        MarketRegimeFeatures,
        SignalMetadataFeatures,
    ]

    def __init__(self):
        """Initialize with default extractors."""
        self._extractors: Dict[str, BaseFeatureExtractor] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register default extractors."""
        for extractor_class in self.DEFAULT_EXTRACTORS:
            extractor = extractor_class()
            self.register(extractor)

    def register(self, extractor: BaseFeatureExtractor):
        """Register an extractor."""
        self._extractors[extractor.name] = extractor

    def get(self, name: str) -> BaseFeatureExtractor:
        """Get extractor by name."""
        if name not in self._extractors:
            raise KeyError(f"Extractor '{name}' not found")
        return self._extractors[name]

    def get_all(self) -> List[BaseFeatureExtractor]:
        """Get all registered extractors."""
        return list(self._extractors.values())

    def get_by_category(self, category: str) -> List[BaseFeatureExtractor]:
        """Get extractors by category."""
        return [e for e in self._extractors.values() if e.category == category]

    def get_all_feature_names(self) -> List[str]:
        """Get all feature names across all extractors."""
        names = []
        for extractor in self._extractors.values():
            names.extend(extractor.feature_names)
        return names

    def get_feature_count(self) -> int:
        """Get total feature count."""
        return len(self.get_all_feature_names())


"""Base class for feature extractors."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pandas as pd


class BaseFeatureExtractor(ABC):
    """
    Abstract base class for feature extractors.

    All extractors should:
    1. Have a unique name
    2. Define what features they provide
    3. Extract features from input data
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this extractor."""
        pass

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """List of feature names this extractor provides."""
        pass

    @property
    def category(self) -> str:
        """Category of features (technical, market, signal, news)."""
        return "general"

    @abstractmethod
    def extract(self, data: Any) -> Dict[str, float]:
        """
        Extract features from input data.

        Args:
            data: Input data (format depends on extractor type)

        Returns:
            Dictionary of feature_name -> value
        """
        pass

    def validate_output(self, features: Dict[str, float]) -> bool:
        """Validate extracted features."""
        expected = set(self.feature_names)
        actual = set(features.keys())
        return expected == actual


class OHLCVExtractor(BaseFeatureExtractor):
    """Base class for extractors that work with OHLCV data."""

    @abstractmethod
    def extract(self, ohlcv: pd.DataFrame, current_idx: int = -1) -> Dict[str, float]:
        """
        Extract features from OHLCV dataframe.

        Args:
            ohlcv: DataFrame with columns [open, high, low, close, volume]
            current_idx: Index of current bar (default: last bar)

        Returns:
            Dictionary of feature values
        """
        pass

    def _safe_get(
        self,
        series: pd.Series,
        idx: int,
        default: float = 0.0,
    ) -> float:
        """Safely get value from series."""
        try:
            val = series.iloc[idx]
            return float(val) if pd.notna(val) else default
        except (IndexError, KeyError):
            return default


"""Configuration for ML refinement module."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ModelType(str, Enum):
    """Types of ML models."""

    SIGNAL_QUALITY = "signal_quality"
    RETURN_PREDICTOR = "return_predictor"
    REGIME_CLASSIFIER = "regime_classifier"
    RISK_PREDICTOR = "risk_predictor"


class FeatureSet(str, Enum):
    """Predefined feature sets."""

    MINIMAL = "minimal"  # Basic features only
    STANDARD = "standard"  # Standard feature set
    EXTENDED = "extended"  # All available features
    CUSTOM = "custom"  # User-defined


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""

    # Feature sets to use
    feature_set: FeatureSet = FeatureSet.STANDARD

    # Custom features (if feature_set is CUSTOM)
    custom_features: List[str] = field(default_factory=list)

    # Technical feature parameters
    technical_lookbacks: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 200])

    # Market feature parameters
    market_regime_window: int = 20
    volatility_window: int = 20

    # News feature parameters
    news_lookback_hours: int = 48
    include_news_features: bool = True

    # Feature scaling
    scale_features: bool = True
    scaling_method: str = "standard"  # "standard", "minmax", "robust"


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Walk-forward parameters
    train_window_days: int = 252  # ~1 year
    validation_window_days: int = 63  # ~3 months
    step_size_days: int = 21  # ~1 month

    # Minimum data requirements
    min_training_samples: int = 100
    min_validation_samples: int = 20

    # Model parameters
    model_type: str = "gradient_boosting"
    hyperparameters: Dict = field(default_factory=dict)

    # Training options
    early_stopping: bool = True
    early_stopping_rounds: int = 50

    # Regularization
    max_features: int = 50  # Limit feature count
    feature_selection: bool = True
    feature_importance_threshold: float = 0.01


@dataclass
class MLConfig:
    """Main ML configuration."""

    # Enable/disable ML
    enabled: bool = True

    # Feature configuration
    features: FeatureConfig = field(default_factory=FeatureConfig)

    # Training configuration
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Model paths
    model_dir: str = "models/"
    feature_db_path: str = "features.db"

    # Retraining schedule
    retrain_frequency_days: int = 7  # Weekly retraining
    min_new_samples_for_retrain: int = 20

    # Prediction thresholds
    quality_threshold_high: float = 0.7
    quality_threshold_low: float = 0.3

    # Integration
    use_ml_scores: bool = True
    ml_score_weight: float = 0.3  # Weight in combined score


@dataclass
class FeatureVector:
    """A single feature vector for a signal."""

    signal_id: str
    timestamp: str
    features: Dict[str, float]
    target: Optional[float] = None  # R-multiple outcome
    target_binary: Optional[int] = None  # 1 = win, 0 = loss

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "signal_id": self.signal_id,
            "timestamp": self.timestamp,
            "features": self.features,
            "target": self.target,
            "target_binary": self.target_binary,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "FeatureVector":
        """Create from dictionary."""
        return cls(
            signal_id=data["signal_id"],
            timestamp=data["timestamp"],
            features=data["features"],
            target=data.get("target"),
            target_binary=data.get("target_binary"),
        )


@dataclass
class ModelMetadata:
    """Metadata for a trained model."""

    model_id: str
    model_type: ModelType
    version: str
    created_at: str

    # Training info
    train_start_date: str
    train_end_date: str
    train_samples: int
    validation_samples: int

    # Performance metrics
    train_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)

    # Feature info
    feature_names: List[str] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)

    # Status
    is_active: bool = False
    deployed_at: Optional[str] = None

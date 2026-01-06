"""Run configuration Pydantic models for backtest runs."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional

import pandas as pd
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from ..research.config import ResearchConfig
from ..signals.config import SignalConfig
from .validation import (
    ConfigValidationError,
    validate_date_format,
    validate_date_range,
    validate_file_exists,
    validate_yaml_format,
    wrap_validation_error,
)


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    equity_path: str
    crypto_path: str
    benchmark_path: str
    format: str = "csv"
    start_date: str
    end_date: str
    min_lookback_days: int = Field(default=250, ge=1, le=1000)

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        allowed = ["csv", "parquet", "database"]
        if v not in allowed:
            raise ValueError(f"format must be one of {allowed}, got '{v}'")
        return v

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        validate_date_format(v)
        return v

    @model_validator(mode="after")
    def validate_date_range(self):
        validate_date_range(self.start_date, self.end_date, "dataset")
        return self


class SplitsConfig(BaseModel):
    """Walk-forward splits configuration."""

    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    holdout_start: str
    holdout_end: str

    @field_validator("train_start", "train_end", "validation_start", "validation_end", "holdout_start", "holdout_end")
    @classmethod
    def validate_date(cls, v: str) -> str:
        validate_date_format(v)
        return v

    @model_validator(mode="after")
    def validate_splits(self):
        # Validate each period's date range
        validate_date_range(self.train_start, self.train_end, "train")
        validate_date_range(self.validation_start, self.validation_end, "validation")
        validate_date_range(self.holdout_start, self.holdout_end, "holdout")

        # Validate split order: train -> validation -> holdout
        train_end = datetime.strptime(self.train_end, "%Y-%m-%d")
        validation_start = datetime.strptime(self.validation_start, "%Y-%m-%d")
        validation_end = datetime.strptime(self.validation_end, "%Y-%m-%d")
        holdout_start = datetime.strptime(self.holdout_start, "%Y-%m-%d")

        if train_end >= validation_start:
            raise ValueError(
                f"Train end date ({self.train_end}) must be before validation start date ({self.validation_start})"
            )
        if validation_end >= holdout_start:
            raise ValueError(
                f"Validation end date ({self.validation_end}) must be before holdout start date ({self.holdout_start})"
            )

        return self

    def get_dates(self, period: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Get start and end dates for a period.

        Args:
            period: One of "train", "validation", "holdout"

        Returns:
            Tuple of (start_date, end_date) as pandas Timestamps
        """
        if period == "train":
            return (pd.Timestamp(self.train_start), pd.Timestamp(self.train_end))
        elif period == "validation":
            return (pd.Timestamp(self.validation_start), pd.Timestamp(self.validation_end))
        elif period == "holdout":
            return (pd.Timestamp(self.holdout_start), pd.Timestamp(self.holdout_end))
        else:
            raise ValueError(f"Invalid period: {period}")


class StrategyConfigRef(BaseModel):
    """Reference to a strategy configuration file."""

    config_path: str
    enabled: bool = True

    @field_validator("config_path")
    @classmethod
    def validate_config_path(cls, v: str) -> str:
        # Note: We don't validate file exists here to allow relative paths
        # Actual validation happens when the strategy config is loaded
        if not v or not v.strip():
            raise ValueError("config_path cannot be empty")
        return v


class StrategiesConfig(BaseModel):
    """Strategies configuration."""

    equity: Optional[StrategyConfigRef] = None
    crypto: Optional[StrategyConfigRef] = None

    @model_validator(mode="after")
    def validate_at_least_one_strategy(self):
        if not self.equity and not self.crypto:
            raise ValueError("At least one strategy (equity or crypto) must be enabled")

        if self.equity and not self.equity.enabled and self.crypto and not self.crypto.enabled:
            raise ValueError("At least one strategy must be enabled")

        return self


class PortfolioConfig(BaseModel):
    """Portfolio configuration."""

    starting_equity: float = Field(default=100000.0, gt=0, description="Starting capital in USD")


class VolatilityScalingConfig(BaseModel):
    """Volatility scaling configuration."""

    enabled: bool = True
    mode: Literal["continuous", "regime", "of"] = "continuous"
    lookback: int = Field(default=20, ge=1, le=252)
    baseline_lookback: int = Field(default=252, ge=1)
    min_multiplier: float = Field(default=0.33, ge=0.0, le=1.0)
    max_multiplier: float = Field(default=1.0, ge=0.0, le=5.0)

    @model_validator(mode="after")
    def validate_multipliers(self):
        if self.min_multiplier > self.max_multiplier:
            raise ValueError(f"min_multiplier ({self.min_multiplier}) must be <= max_multiplier ({self.max_multiplier})")
        return self


class CorrelationGuardConfig(BaseModel):
    """Correlation guard configuration."""

    enabled: bool = True
    min_positions: int = Field(default=4, ge=1)
    avg_pairwise_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    candidate_threshold: float = Field(default=0.75, ge=0.0, le=1.0)


class ScoringConfig(BaseModel):
    """Scoring function configuration."""

    weights: Dict[str, float] = Field(default_factory=lambda: {"breakout": 0.50, "momentum": 0.30, "diversification": 0.20})

    @model_validator(mode="after")
    def validate_weights(self):
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.001:  # Allow small floating point errors
            raise ValueError(f"Scoring weights must sum to 1.0, got {total}")

        for key, value in self.weights.items():
            if value < 0 or value > 1:
                raise ValueError(f"Scoring weight '{key}' must be between 0 and 1, got {value}")

        return self


class ExecutionConfig(BaseModel):
    """Execution configuration."""

    signal_timing: str = Field(default="close")
    execution_timing: str = Field(default="next_open")
    slippage_model: str = Field(default="full")

    @field_validator("signal_timing")
    @classmethod
    def validate_signal_timing(cls, v: str) -> str:
        allowed = ["close"]
        if v not in allowed:
            raise ValueError(f"signal_timing must be one of {allowed}, got '{v}'")
        return v

    @field_validator("execution_timing")
    @classmethod
    def validate_execution_timing(cls, v: str) -> str:
        allowed = ["next_open"]
        if v not in allowed:
            raise ValueError(f"execution_timing must be one of {allowed}, got '{v}'")
        return v

    @field_validator("slippage_model")
    @classmethod
    def validate_slippage_model(cls, v: str) -> str:
        allowed = ["full", "simple", "none"]
        if v not in allowed:
            raise ValueError(f"slippage_model must be one of {allowed}, got '{v}'")
        return v


class OutputConfig(BaseModel):
    """Output configuration."""

    base_path: str = "results/"
    run_id: Optional[str] = None
    config_name: Optional[str] = None  # Optional name to include in output directory
    equity_curve: str = "equity_curve.csv"
    trade_log: str = "trade_log.csv"
    weekly_summary: str = "weekly_summary.csv"
    monthly_report: str = "monthly_report.json"
    scenario_comparison: str = "scenario_comparison.json"
    log_level: str = Field(default="INFO")
    log_file: str = "backtest.log"
    log_json_format: bool = False  # Use JSON format for file logs
    log_use_rich: bool = True  # Use rich for console output (if available)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return v.upper()

    def get_run_id(self) -> str:
        """Get or generate run ID.

        Returns:
            Run ID string (format: run_YYYYMMDD_HHMMSS or run_YYYYMMDD_HHMMSS_configname)
        """
        if self.run_id:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.config_name:
            # Sanitize config_name: replace spaces/special chars with underscores
            safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in self.config_name)
            return f"run_{timestamp}_{safe_name}"
        return f"run_{timestamp}"


class SensitivityConfig(BaseModel):
    """Parameter sensitivity grid configuration."""

    enabled: bool = True
    equity_atr_mult: List[float] = Field(default_factory=lambda: [2.0, 2.5, 3.0, 3.5])
    equity_breakout_clearance: List[float] = Field(default_factory=lambda: [0.000, 0.005, 0.010, 0.015])
    equity_exit_ma: List[int] = Field(default_factory=lambda: [20, 50])
    crypto_atr_mult: List[float] = Field(default_factory=lambda: [2.5, 3.0, 3.5, 4.0])
    crypto_breakout_clearance: List[float] = Field(default_factory=lambda: [0.000, 0.005, 0.010, 0.015])
    crypto_exit_mode: List[str] = Field(default_factory=lambda: ["MA20", "MA50", "staged"])
    vol_scaling_mode: List[str] = Field(default_factory=lambda: ["continuous", "regime", "off"])


class StressTestsConfig(BaseModel):
    """Stress tests configuration."""

    slippage_multipliers: List[float] = Field(default_factory=lambda: [1.0, 2.0, 3.0])
    bear_market_test: bool = True
    range_market_test: bool = True
    flash_crash_test: bool = True


class StatisticalConfig(BaseModel):
    """Statistical tests configuration."""

    bootstrap_iterations: int = 1000
    permutation_iterations: int = 1000
    bootstrap_5th_percentile_threshold: float = 0.4


class ValidationConfig(BaseModel):
    """Validation suite configuration."""

    sensitivity: Optional[SensitivityConfig] = None
    stress_tests: Optional[StressTestsConfig] = None
    statistical: Optional[StatisticalConfig] = None


class MetricsTargetsConfig(BaseModel):
    """Metrics targets configuration."""

    sharpe_ratio_min: float = 1.0
    max_drawdown_max: float = 0.15
    calmar_ratio_min: float = 1.5
    min_trades: int = 50


class SecondaryMetricsConfig(BaseModel):
    """Secondary metrics configuration."""

    expectancy_R_min: float = 0.3
    profit_factor_min: float = 1.4
    correlation_to_benchmark_max: float = 0.80
    percentile_99_daily_loss_max: float = 0.05


class RejectionCriteriaConfig(BaseModel):
    """Rejection criteria configuration."""

    max_drawdown_max: float = 0.20
    sharpe_ratio_min: float = 0.75
    calmar_ratio_min: float = 1.0


class MetricsConfig(BaseModel):
    """Metrics configuration."""

    primary: Optional[MetricsTargetsConfig] = None
    secondary: Optional[SecondaryMetricsConfig] = None
    rejection: Optional[RejectionCriteriaConfig] = None


class MLDataCollectionConfig(BaseModel):
    """ML data collection configuration for feature accumulation.

    When enabled, the backtest engine will extract features at signal generation
    and record outcomes when trades close, storing them in a SQLite database
    for ML model training.
    """

    enabled: bool = Field(default=False, description="Enable ML feature collection during backtest")
    db_path: str = Field(default="features.db", description="Path to SQLite database for ML features")


class RunConfig(BaseModel):
    """Complete run configuration."""

    dataset: DatasetConfig
    splits: SplitsConfig
    strategies: StrategiesConfig
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    volatility_scaling: VolatilityScalingConfig = Field(default_factory=VolatilityScalingConfig)
    correlation_guard: CorrelationGuardConfig = Field(default_factory=CorrelationGuardConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    signals: Optional[SignalConfig] = Field(default_factory=SignalConfig)
    random_seed: int = 42
    validation: Optional[ValidationConfig] = None
    metrics: Optional[MetricsConfig] = None
    ml_data_collection: MLDataCollectionConfig = Field(default_factory=MLDataCollectionConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        """Load run configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            RunConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ConfigValidationError: If configuration validation fails
        """
        try:
            validate_file_exists(path, "run config")
            data = validate_yaml_format(path)
        except (FileNotFoundError, ValueError):
            raise

        try:
            return cls(**data)
        except Exception as e:
            if isinstance(e, Exception) and hasattr(e, "errors"):
                # Pydantic validation error
                raise wrap_validation_error(e, "Run configuration", config_path=path) from e
            raise ConfigValidationError(f"Failed to load run configuration: {str(e)}", config_path=path) from e

    def to_yaml(self, path: str) -> None:
        """Save run configuration to YAML file.

        Args:
            path: Path to save YAML configuration file
        """
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def get_output_dir(self) -> Path:
        """Get output directory path.

        Returns:
            Path object for output directory
        """
        run_id = self.output.get_run_id()
        return Path(self.output.base_path) / run_id

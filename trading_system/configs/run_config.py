"""Run configuration Pydantic models for backtest runs."""

from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Literal
from datetime import datetime
import yaml
import pandas as pd
from pathlib import Path


class DatasetConfig(BaseModel):
    """Dataset configuration."""
    equity_path: str
    crypto_path: str
    benchmark_path: str
    format: str = "csv"
    start_date: str
    end_date: str
    min_lookback_days: int = 250


class SplitsConfig(BaseModel):
    """Walk-forward splits configuration."""
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    holdout_start: str
    holdout_end: str
    
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


class StrategiesConfig(BaseModel):
    """Strategies configuration."""
    equity: Optional[StrategyConfigRef] = None
    crypto: Optional[StrategyConfigRef] = None


class PortfolioConfig(BaseModel):
    """Portfolio configuration."""
    starting_equity: float = 100000.0


class VolatilityScalingConfig(BaseModel):
    """Volatility scaling configuration."""
    enabled: bool = True
    mode: Literal["continuous", "regime", "off"] = "continuous"
    lookback: int = 20
    baseline_lookback: int = 252
    min_multiplier: float = 0.33
    max_multiplier: float = 1.0


class CorrelationGuardConfig(BaseModel):
    """Correlation guard configuration."""
    enabled: bool = True
    min_positions: int = 4
    avg_pairwise_threshold: float = 0.70
    candidate_threshold: float = 0.75


class ScoringConfig(BaseModel):
    """Scoring function configuration."""
    weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "breakout": 0.50,
            "momentum": 0.30,
            "diversification": 0.20
        }
    )


class ExecutionConfig(BaseModel):
    """Execution configuration."""
    signal_timing: str = "close"
    execution_timing: str = "next_open"
    slippage_model: str = "full"


class OutputConfig(BaseModel):
    """Output configuration."""
    base_path: str = "results/"
    run_id: Optional[str] = None
    equity_curve: str = "equity_curve.csv"
    trade_log: str = "trade_log.csv"
    weekly_summary: str = "weekly_summary.csv"
    monthly_report: str = "monthly_report.json"
    scenario_comparison: str = "scenario_comparison.json"
    log_level: str = "INFO"
    log_file: str = "backtest.log"
    
    def get_run_id(self) -> str:
        """Get or generate run ID.
        
        Returns:
            Run ID string (format: run_YYYYMMDD_HHMMSS)
        """
        if self.run_id:
            return self.run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    random_seed: int = 42
    validation: Optional[ValidationConfig] = None
    metrics: Optional[MetricsConfig] = None
    
    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        """Load run configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            RunConfig instance
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(**data)
    
    def to_yaml(self, path: str) -> None:
        """Save run configuration to YAML file.
        
        Args:
            path: Path to save YAML configuration file
        """
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
    
    def get_output_dir(self) -> Path:
        """Get output directory path.
        
        Returns:
            Path object for output directory
        """
        run_id = self.output.get_run_id()
        return Path(self.output.base_path) / run_id


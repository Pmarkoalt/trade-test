"""Strategy configuration Pydantic models."""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Literal, Union, Dict, Any
import yaml
from .validation import (
    validate_file_exists, validate_yaml_format, wrap_validation_error, ConfigValidationError
)


class EligibilityConfig(BaseModel):
    """Eligibility filter configuration."""
    trend_ma: int = 50  # MA period for trend filter
    ma_slope_lookback: int = 20
    ma_slope_min: float = 0.005
    require_close_above_trend_ma: bool = True
    require_close_above_ma200: bool = False  # For crypto
    relative_strength_enabled: bool = False
    relative_strength_min: float = 0.0


class EntryConfig(BaseModel):
    """Entry trigger configuration."""
    fast_clearance: float = Field(default=0.005, ge=0.0, le=0.1, description="0.5% clearance")
    slow_clearance: float = Field(default=0.010, ge=0.0, le=0.1, description="1.0% clearance")
    
    @model_validator(mode='after')
    def validate_clearance(self):
        if self.fast_clearance >= self.slow_clearance:
            raise ValueError(f"fast_clearance ({self.fast_clearance}) must be < slow_clearance ({self.slow_clearance})")
        return self


class ExitConfig(BaseModel):
    """Exit configuration."""
    mode: Literal["ma_cross", "staged"] = "ma_cross"
    exit_ma: int = Field(default=20, ge=1, le=200)
    hard_stop_atr_mult: float = Field(default=2.5, gt=0.0, le=10.0)
    tightened_stop_atr_mult: Optional[float] = Field(default=None, gt=0.0, le=10.0)
    
    @field_validator('exit_ma')
    @classmethod
    def validate_exit_ma(cls, v: int) -> int:
        if v not in [20, 50] and v < 200:
            # Allow custom values but warn about standard values
            pass
        return v
    
    @model_validator(mode='after')
    def validate_staged_exit(self):
        if self.mode == "staged" and self.tightened_stop_atr_mult is None:
            raise ValueError("tightened_stop_atr_mult is required when exit.mode is 'staged'")
        if self.tightened_stop_atr_mult is not None and self.tightened_stop_atr_mult >= self.hard_stop_atr_mult:
            raise ValueError(f"tightened_stop_atr_mult ({self.tightened_stop_atr_mult}) must be < hard_stop_atr_mult ({self.hard_stop_atr_mult})")
        return self


class RiskConfig(BaseModel):
    """Risk management configuration."""
    risk_per_trade: float = Field(default=0.0075, gt=0.0, le=0.05, description="0.75% risk per trade")
    max_positions: int = Field(default=8, ge=1, le=50)
    max_exposure: float = Field(default=0.80, gt=0.0, le=1.0, description="80% max exposure")
    max_position_notional: float = Field(default=0.15, gt=0.0, le=1.0, description="15% max position size")
    
    @model_validator(mode='after')
    def validate_exposure(self):
        if self.max_position_notional > self.max_exposure:
            raise ValueError(f"max_position_notional ({self.max_position_notional}) should not exceed max_exposure ({self.max_exposure})")
        return self


class CapacityConfig(BaseModel):
    """Capacity constraint configuration."""
    max_order_pct_adv: float = 0.005  # 0.5% for equity, 0.25% for crypto


class CostsConfig(BaseModel):
    """Execution costs configuration."""
    fee_bps: int = 1  # 1 for equity, 8 for crypto
    slippage_base_bps: int = 8  # 8 for equity, 10 for crypto
    slippage_std_mult: float = 0.75
    weekend_penalty: float = 1.0  # 1.5 for crypto weekends
    stress_threshold: float = -0.03  # -3% for equity, -5% for crypto
    stress_slippage_mult: float = 2.0


class MLConfig(BaseModel):
    """ML model integration configuration."""
    enabled: bool = False
    model_path: Optional[str] = None  # Path to saved ML model directory
    prediction_mode: Literal["score_enhancement", "filter", "replace"] = "score_enhancement"
    ml_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Weight for ML prediction in score_enhancement mode")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence for filtering mode")
    
    @model_validator(mode='after')
    def validate_model_path(self):
        if self.enabled and not self.model_path:
            raise ValueError("model_path is required when ML is enabled")
        return self


class IndicatorsConfig(BaseModel):
    """Indicator calculation configuration."""
    ma_periods: List[int] = Field(default_factory=lambda: [20, 50, 200])
    atr_period: int = Field(default=14, ge=1, le=252)
    roc_period: int = Field(default=60, ge=1, le=252)
    breakout_fast: int = Field(default=20, ge=1, le=252)
    breakout_slow: int = Field(default=55, ge=1, le=252)
    adv_lookback: int = Field(default=20, ge=1, le=252)
    corr_lookback: int = Field(default=20, ge=1, le=252)
    
    @model_validator(mode='after')
    def validate_breakout_periods(self):
        if self.breakout_fast >= self.breakout_slow:
            raise ValueError(f"breakout_fast ({self.breakout_fast}) must be < breakout_slow ({self.breakout_slow})")
        if len(self.ma_periods) == 0:
            raise ValueError("ma_periods cannot be empty")
        if any(p < 1 or p > 252 for p in self.ma_periods):
            raise ValueError("All ma_periods must be between 1 and 252")
        return self


class CryptoUniverseConfig(BaseModel):
    """Crypto universe configuration (optional, for dynamic universe selection)."""
    mode: Literal["fixed", "custom", "dynamic"] = "fixed"
    symbols: Optional[List[str]] = None
    min_market_cap_usd: Optional[float] = None
    min_volume_usd: Optional[float] = None
    min_liquidity_score: Optional[float] = None
    max_symbols: Optional[int] = None
    rebalance_frequency: Optional[Literal["monthly", "quarterly", "never"]] = "never"
    rebalance_lookback_days: int = 30
    universe_file_path: Optional[str] = None


class StrategyConfig(BaseModel):
    """Complete strategy configuration."""
    name: str
    asset_class: Literal["equity", "crypto"]
    universe: Union[str, List[str]]  # "NASDAQ-100", "SP500", "crypto", or explicit list
    benchmark: str  # "SPY" for equity, "BTC" for crypto

    # Multi-strategy allocation (default: 1.0 = 100% if single strategy)
    risk_allocation: float = Field(default=1.0, gt=0.0, le=1.0, description="Capital allocation fraction (0.0-1.0)")

    # Optional crypto universe configuration (for dynamic selection)
    universe_config: Optional[CryptoUniverseConfig] = None

    # Strategy-specific parameters (for mean reversion, pairs, etc.)
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Strategy-specific parameters")

    indicators: IndicatorsConfig = Field(default_factory=IndicatorsConfig)
    eligibility: EligibilityConfig = Field(default_factory=EligibilityConfig)
    entry: EntryConfig = Field(default_factory=EntryConfig)
    exit: ExitConfig = Field(default_factory=ExitConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    capacity: CapacityConfig = Field(default_factory=CapacityConfig)
    costs: CostsConfig = Field(default_factory=CostsConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "StrategyConfig":
        """Load from YAML file.
        
        Args:
            path: Path to YAML configuration file
        
        Returns:
            StrategyConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ConfigValidationError: If configuration validation fails
        """
        try:
            validate_file_exists(path, "strategy config")
            data = validate_yaml_format(path)
        except (FileNotFoundError, ValueError) as e:
            raise
        
        try:
            return cls(**data)
        except Exception as e:
            if isinstance(e, Exception) and hasattr(e, 'errors'):
                # Pydantic validation error
                raise wrap_validation_error(e, "Strategy configuration", config_path=path) from e
            raise ConfigValidationError(
                f"Failed to load strategy configuration: {str(e)}",
                config_path=path
            ) from e
    
    def to_yaml(self, path: str) -> None:
        """Save to YAML file.
        
        Args:
            path: Path to save YAML configuration file
        """
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)


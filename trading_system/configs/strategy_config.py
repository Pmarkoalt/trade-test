"""Strategy configuration Pydantic models."""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import yaml


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
    fast_clearance: float = 0.005  # 0.5%
    slow_clearance: float = 0.010  # 1.0%


class ExitConfig(BaseModel):
    """Exit configuration."""
    mode: Literal["ma_cross", "staged"] = "ma_cross"
    exit_ma: int = 20  # 20 or 50
    hard_stop_atr_mult: float = 2.5
    tightened_stop_atr_mult: Optional[float] = None  # For crypto staged exit


class RiskConfig(BaseModel):
    """Risk management configuration."""
    risk_per_trade: float = 0.0075  # 0.75%
    max_positions: int = 8
    max_exposure: float = 0.80  # 80%
    max_position_notional: float = 0.15  # 15%


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


class IndicatorsConfig(BaseModel):
    """Indicator calculation configuration."""
    ma_periods: List[int] = [20, 50, 200]
    atr_period: int = 14
    roc_period: int = 60
    breakout_fast: int = 20
    breakout_slow: int = 55
    adv_lookback: int = 20
    corr_lookback: int = 20


class StrategyConfig(BaseModel):
    """Complete strategy configuration."""
    name: str
    asset_class: Literal["equity", "crypto"]
    universe: str  # "NASDAQ-100", "SP500", or explicit list
    benchmark: str  # "SPY" for equity, "BTC" for crypto
    
    indicators: IndicatorsConfig = Field(default_factory=IndicatorsConfig)
    eligibility: EligibilityConfig = Field(default_factory=EligibilityConfig)
    entry: EntryConfig = Field(default_factory=EntryConfig)
    exit: ExitConfig = Field(default_factory=ExitConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    capacity: CapacityConfig = Field(default_factory=CapacityConfig)
    costs: CostsConfig = Field(default_factory=CostsConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "StrategyConfig":
        """Load from YAML file.
        
        Args:
            path: Path to YAML configuration file
        
        Returns:
            StrategyConfig instance
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str) -> None:
        """Save to YAML file.
        
        Args:
            path: Path to save YAML configuration file
        """
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)


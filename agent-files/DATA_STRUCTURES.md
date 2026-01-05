# Data Structures Specification

Complete definitions of all data models, classes, and structures used throughout the trading system.

---

## Core Data Models

### Bar (OHLCV Data)

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class Bar:
    """Single OHLCV bar for a symbol at a specific date."""
    date: pd.Timestamp  # Date/time of the bar
    symbol: str  # Ticker symbol (e.g., "AAPL", "BTC")
    open: float  # Opening price
    high: float  # High price
    low: float  # Low price
    close: float  # Closing price
    volume: float  # Volume (shares or units)
    dollar_volume: float  # Computed: close * volume

    def __post_init__(self):
        """Validate OHLC relationships."""
        assert self.low <= self.open <= self.high, f"Invalid OHLC: {self.symbol} {self.date}"
        assert self.low <= self.close <= self.high, f"Invalid OHLC: {self.symbol} {self.date}"
        assert self.volume >= 0, f"Negative volume: {self.symbol} {self.date}"
        if not hasattr(self, 'dollar_volume') or self.dollar_volume == 0:
            self.dollar_volume = self.close * self.volume
```

**Validation Rules:**
- `low <= open <= high`
- `low <= close <= high`
- `volume >= 0`
- `dollar_volume = close * volume` (computed if not provided)

---

### FeatureRow (Indicators)

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class FeatureRow:
    """Computed indicators for a symbol at a specific date."""
    date: pd.Timestamp
    symbol: str
    asset_class: str  # "equity" | "crypto"

    # Price data
    close: float
    open: float
    high: float
    low: float

    # Moving averages
    ma20: Optional[float] = None  # NaN until 20 bars available
    ma50: Optional[float] = None  # NaN until 50 bars available
    ma200: Optional[float] = None  # NaN until 200 bars available

    # Volatility
    atr14: Optional[float] = None  # NaN until 14 bars available

    # Momentum
    roc60: Optional[float] = None  # NaN if close[t-60] missing

    # Breakout levels
    highest_close_20d: Optional[float] = None  # Highest close over last 20 days (exclusive of today)
    highest_close_55d: Optional[float] = None  # Highest close over last 55 days (exclusive of today)

    # Volume
    adv20: Optional[float] = None  # 20-day average dollar volume

    # Returns
    returns_1d: Optional[float] = None  # (close[t] / close[t-1]) - 1

    # Benchmark data (for relative strength)
    benchmark_roc60: Optional[float] = None  # SPY or BTC ROC60
    benchmark_returns_1d: Optional[float] = None  # SPY or BTC daily return

    def is_valid_for_entry(self) -> bool:
        """Check if sufficient data exists for signal generation."""
        return (
            self.ma20 is not None and not np.isnan(self.ma20) and
            self.ma50 is not None and not np.isnan(self.ma50) and
            self.atr14 is not None and not np.isnan(self.atr14) and
            self.highest_close_20d is not None and not np.isnan(self.highest_close_20d) and
            self.highest_close_55d is not None and not np.isnan(self.highest_close_55d) and
            self.adv20 is not None and not np.isnan(self.adv20)
        )
```

**Indicator Calculation Rules:**
- All indicators return `NaN` until sufficient lookback is available
- Indicators are NOT forward-filled (use NaN to prevent lookahead)
- `highest_close_20d` and `highest_close_55d` exclude today's close (use prior N days only)
- ATR uses Wilder's smoothing method (standard implementation)

---

### Signal (Entry Intent)

```python
from dataclasses import dataclass
from enum import Enum

class SignalSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"  # For exits, though system is long-only

class BreakoutType(str, Enum):
    FAST_20D = "20D"
    SLOW_55D = "55D"

@dataclass
class Signal:
    """Entry signal generated at day close."""
    symbol: str
    asset_class: str  # "equity" | "crypto"
    date: pd.Timestamp  # Date when signal was generated (at close)
    side: SignalSide  # Always BUY for this system

    # Entry details
    entry_price: float  # Close price at signal time (will execute at next open)
    stop_price: float  # Calculated stop: entry_price - (ATR_mult * ATR14)
    atr_mult: float  # ATR multiplier used (2.5 for equity, 3.0 for crypto)

    # Trigger information
    triggered_on: BreakoutType  # Which breakout triggered: "20D" or "55D"
    breakout_clearance: float  # Actual clearance above prior high (for logging)

    # Scoring components (for queue ranking)
    breakout_strength: float  # (close - MA) / ATR14, normalized
    momentum_strength: float  # Relative strength vs benchmark
    diversification_bonus: float  # 1 - avg_corr_to_portfolio
    score: float  # Final weighted score (0-1 after rank normalization)

    # Eligibility status
    passed_eligibility: bool  # True if all filters passed
    eligibility_failures: list[str]  # Reasons if failed (e.g., ["below_MA50", "insufficient_slope"])

    # Capacity check
    order_notional: float  # Estimated order size (for capacity check)
    adv20: float  # ADV20 at signal time
    capacity_passed: bool  # True if order_notional <= max_pct * ADV20

    def is_valid(self) -> bool:
        """Check if signal is valid for execution."""
        return (
            self.passed_eligibility and
            self.capacity_passed and
            self.side == SignalSide.BUY
        )
```

**Signal Generation Rules:**
- Signals are generated at day `t` close
- `entry_price` = close price at `t`
- Execution will occur at `t+1` open
- Stop price is calculated at signal time (not updated until position is opened)

---

### Order (Execution Intent)

```python
from dataclasses import dataclass
from enum import Enum

class OrderStatus(str, Enum):
    PENDING = "PENDING"  # Created, awaiting execution
    FILLED = "FILLED"  # Executed
    REJECTED = "REJECTED"  # Rejected due to constraint
    CANCELLED = "CANCELLED"  # Cancelled before execution

@dataclass
class Order:
    """Order to execute at next session open."""
    order_id: str  # Unique identifier
    symbol: str
    asset_class: str
    date: pd.Timestamp  # Date when order was created (signal date)
    execution_date: pd.Timestamp  # Date when order should execute (next open)

    side: SignalSide  # BUY or SELL
    quantity: int  # Number of shares/units (calculated from risk sizing)
    limit_price: Optional[float] = None  # Not used (market orders only)

    # Derived from signal
    signal_date: pd.Timestamp  # Original signal date
    expected_fill_price: float  # Next open price (estimated)
    stop_price: float  # Stop price for position

    # Status
    status: OrderStatus = OrderStatus.PENDING
    rejection_reason: Optional[str] = None  # If REJECTED

    # Constraints checked
    capacity_checked: bool = False
    correlation_checked: bool = False
    max_positions_checked: bool = False
    max_exposure_checked: bool = False
```

**Order Lifecycle:**
1. Created from Signal at day `t` close
2. Status: PENDING
3. Executed at day `t+1` open → Status: FILLED
4. Or rejected → Status: REJECTED with reason

---

### Fill (Execution Result)

```python
from dataclasses import dataclass

@dataclass
class Fill:
    """Realized execution result."""
    fill_id: str  # Unique identifier
    order_id: str  # Parent order ID
    symbol: str
    asset_class: str
    date: pd.Timestamp  # Execution date (open of day t+1)

    side: SignalSide
    quantity: int  # Actual filled quantity (may differ from order if partial)
    fill_price: float  # Actual execution price (open + slippage)
    open_price: float  # Market open price (before slippage)

    # Costs
    slippage_bps: float  # Actual slippage in basis points
    fee_bps: float  # Fee in basis points (1 for equity, 8 for crypto)
    total_cost: float  # (slippage + fee) * notional

    # Slippage model components (for diagnostics)
    vol_mult: float  # Volatility multiplier
    size_penalty: float  # Size penalty
    weekend_penalty: float  # Weekend penalty (crypto only)
    stress_mult: float  # Stress multiplier

    # Notional
    notional: float  # fill_price * quantity

    def compute_total_cost(self) -> float:
        """Compute total execution cost."""
        notional = self.fill_price * self.quantity
        slippage_cost = notional * (self.slippage_bps / 10000)
        fee_cost = notional * (self.fee_bps / 10000)
        return slippage_cost + fee_cost
```

**Fill Rules:**
- Fill price = open_price * (1 ± slippage_bps/10000) depending on side
- BUY: fill_price = open * (1 + slippage_bps/10000)
- SELL: fill_price = open * (1 - slippage_bps/10000)
- Quantity is always integer (shares/units)

---

### Position (Open Position)

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class ExitReason(str, Enum):
    TRAILING_MA_CROSS = "trailing_ma_cross"  # MA20 or MA50 cross
    HARD_STOP = "hard_stop"  # ATR-based stop hit
    DATA_MISSING = "data_missing"  # 2+ consecutive missing days
    CAPACITY_REJECT = "capacity_reject"  # Should not happen for open positions
    MANUAL = "manual"  # Manual exit (not used in MVP)

@dataclass
class Position:
    """Open position in portfolio."""
    symbol: str
    asset_class: str

    # Entry details
    entry_date: pd.Timestamp  # Date when position was opened
    entry_price: float  # Fill price at entry
    entry_fill_id: str  # Reference to entry fill
    quantity: int  # Number of shares/units

    # Stop management
    stop_price: float  # Current stop price
    initial_stop_price: float  # Original stop (for R-multiple calculation)
    hard_stop_atr_mult: float  # ATR multiplier (2.5 equity, 3.0 crypto)
    tightened_stop: bool = False  # True if stop was tightened (crypto staged exit)
    tightened_stop_atr_mult: Optional[float] = None  # 2.0 for crypto after MA20 break

    # Exit tracking
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_fill_id: Optional[str] = None
    exit_reason: Optional[ExitReason] = None

    # Cost tracking
    entry_slippage_bps: float
    entry_fee_bps: float
    entry_total_cost: float
    exit_slippage_bps: Optional[float] = None
    exit_fee_bps: Optional[float] = None
    exit_total_cost: Optional[float] = None

    # P&L
    realized_pnl: float = 0.0  # Only set when position is closed
    unrealized_pnl: float = 0.0  # Updated daily: (current_price - entry_price) * quantity - costs

    # Metadata
    triggered_on: BreakoutType  # Which breakout triggered entry
    adv20_at_entry: float  # ADV20 at entry (for diagnostics)

    def is_open(self) -> bool:
        """Check if position is still open."""
        return self.exit_date is None

    def update_unrealized_pnl(self, current_price: float) -> None:
        """Update unrealized P&L based on current price."""
        if not self.is_open():
            return

        # Unrealized P&L = (current_price - entry_price) * quantity - entry_costs
        price_pnl = (current_price - self.entry_price) * self.quantity
        self.unrealized_pnl = price_pnl - self.entry_total_cost

    def compute_r_multiple(self, exit_price: float) -> float:
        """Compute R-multiple for closed position."""
        if self.exit_price is None:
            # Use provided exit_price for calculation
            price_change = exit_price - self.entry_price
        else:
            price_change = self.exit_price - self.entry_price

        risk = self.entry_price - self.initial_stop_price
        if risk <= 0:
            return 0.0

        return price_change / risk

    def update_stop(self, new_stop_price: float, reason: str = "") -> None:
        """Update stop price (for trailing stops or tightening)."""
        # Stop can only move up (for long positions) or stay same
        # Never moves down (trailing stop logic)
        if new_stop_price > self.stop_price:
            self.stop_price = new_stop_price
            if "tighten" in reason.lower():
                self.tightened_stop = True
                self.tightened_stop_atr_mult = 2.0  # For crypto
```

**Position Rules:**
- Stop price can only move up (trailing stop) or be tightened (crypto)
- Unrealized P&L updated daily at close
- Realized P&L computed only on exit
- R-multiple = (exit_price - entry_price) / (entry_price - initial_stop_price)

---

### Portfolio (Portfolio State)

```python
from dataclasses import dataclass, field
from typing import Dict, List
from datetime import datetime

@dataclass
class Portfolio:
    """Portfolio state at a specific date."""
    date: pd.Timestamp

    # Cash and equity
    cash: float  # Available cash
    starting_equity: float  # Initial equity (100,000)
    equity: float  # Current equity = cash + sum(position_values)

    # Positions
    positions: Dict[str, Position] = field(default_factory=dict)  # symbol -> Position

    # Equity curve
    equity_curve: List[float] = field(default_factory=list)  # Historical equity values
    daily_returns: List[float] = field(default_factory=list)  # Daily portfolio returns

    # Exposure
    gross_exposure: float = 0.0  # Sum of all position notional values
    gross_exposure_pct: float = 0.0  # gross_exposure / equity
    per_position_exposure: Dict[str, float] = field(default_factory=dict)  # symbol -> pct

    # P&L
    realized_pnl: float = 0.0  # Cumulative realized P&L
    unrealized_pnl: float = 0.0  # Sum of all position unrealized P&L

    # Risk metrics
    portfolio_vol_20d: Optional[float] = None  # 20-day rolling portfolio volatility (annualized)
    median_vol_252d: Optional[float] = None  # Median vol over last 252 days
    risk_multiplier: float = 1.0  # Volatility scaling multiplier (0.33 to 1.0)

    # Correlation metrics
    avg_pairwise_corr: Optional[float] = None  # Average pairwise correlation (if >= 4 positions)
    correlation_matrix: Optional[np.ndarray] = None  # Full correlation matrix

    # Trade statistics
    total_trades: int = 0  # Total trades closed
    open_trades: int = 0  # Current open positions

    def update_equity(self, current_prices: Dict[str, float]) -> None:
        """Update equity based on current market prices."""
        # Update unrealized P&L for all positions
        total_unrealized = 0.0
        total_exposure = 0.0

        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position.update_unrealized_pnl(current_prices[symbol])
                total_unrealized += position.unrealized_pnl
                total_exposure += current_prices[symbol] * position.quantity

        self.unrealized_pnl = total_unrealized
        self.gross_exposure = total_exposure
        self.gross_exposure_pct = total_exposure / self.equity if self.equity > 0 else 0.0
        self.equity = self.cash + total_exposure
        self.open_trades = len(self.positions)

    def compute_portfolio_returns(self, lookback: int = 20) -> List[float]:
        """Compute portfolio returns for volatility calculation."""
        if len(self.equity_curve) < 2:
            return []

        returns = []
        for i in range(1, len(self.equity_curve)):
            ret = (self.equity_curve[i] / self.equity_curve[i-1]) - 1
            returns.append(ret)

        return returns[-lookback:] if len(returns) >= lookback else returns

    def update_volatility_scaling(self) -> None:
        """Update risk multiplier based on portfolio volatility."""
        returns = self.compute_portfolio_returns(lookback=20)

        if len(returns) < 20:
            # Insufficient history: use default multiplier
            self.risk_multiplier = 1.0
            return

        # Compute 20D volatility (annualized)
        vol_20d = np.std(returns) * np.sqrt(252)
        self.portfolio_vol_20d = vol_20d

        # Compute median over last 252 days
        all_returns = self.compute_portfolio_returns(lookback=252)
        if len(all_returns) >= 252:
            # Compute rolling median (simplified: use all available)
            median_vol = np.median([np.std(all_returns[i:i+20]) * np.sqrt(252)
                                    for i in range(len(all_returns) - 19)])
            self.median_vol_252d = median_vol
        else:
            # Use current vol as baseline if insufficient history
            self.median_vol_252d = vol_20d

        # Compute risk multiplier
        vol_ratio = vol_20d / self.median_vol_252d if self.median_vol_252d > 0 else 1.0
        self.risk_multiplier = max(0.33, min(1.0, 1.0 / max(vol_ratio, 1.0)))

    def update_correlation_metrics(self, returns_data: Dict[str, List[float]], lookback: int = 20) -> None:
        """Update correlation metrics for existing positions."""
        if len(self.positions) < 4:
            self.avg_pairwise_corr = None
            self.correlation_matrix = None
            return

        # Get returns for all positions
        position_symbols = list(self.positions.keys())
        position_returns = {}

        for symbol in position_symbols:
            if symbol in returns_data and len(returns_data[symbol]) >= lookback:
                position_returns[symbol] = returns_data[symbol][-lookback:]

        if len(position_returns) < 2:
            self.avg_pairwise_corr = None
            return

        # Compute correlation matrix
        returns_df = pd.DataFrame(position_returns)
        corr_matrix = returns_df.corr().values

        # Compute average pairwise correlation (exclude diagonal)
        n = len(corr_matrix)
        off_diagonal = []
        for i in range(n):
            for j in range(i+1, n):
                if not np.isnan(corr_matrix[i, j]):
                    off_diagonal.append(corr_matrix[i, j])

        self.avg_pairwise_corr = np.mean(off_diagonal) if off_diagonal else None
        self.correlation_matrix = corr_matrix
```

**Portfolio Update Rules:**
- Equity = cash + sum(position_values at current prices)
- Volatility scaling computed daily at close
- Correlation metrics computed only if >= 4 positions
- Risk multiplier applied to new entries only (not existing positions)

---

## Data Container Classes

### MarketData (Data Storage)

```python
from typing import Dict, List, Optional
import pandas as pd

class MarketData:
    """Container for all market data (bars, features, benchmarks)."""

    def __init__(self):
        self.bars: Dict[str, pd.DataFrame] = {}  # symbol -> DataFrame of bars
        self.features: Dict[str, pd.DataFrame] = {}  # symbol -> DataFrame of features
        self.benchmarks: Dict[str, pd.DataFrame] = {}  # "SPY" or "BTC" -> DataFrame

    def get_bar(self, symbol: str, date: pd.Timestamp) -> Optional[Bar]:
        """Get bar for symbol at date."""
        if symbol not in self.bars:
            return None
        df = self.bars[symbol]
        if date not in df.index:
            return None
        row = df.loc[date]
        return Bar(
            date=date,
            symbol=symbol,
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume'],
            dollar_volume=row.get('dollar_volume', row['close'] * row['volume'])
        )

    def get_features(self, symbol: str, date: pd.Timestamp) -> Optional[FeatureRow]:
        """Get features for symbol at date."""
        if symbol not in self.features:
            return None
        df = self.features[symbol]
        if date not in df.index:
            return None
        row = df.loc[date]
        return FeatureRow(**row.to_dict())
```

---

## Configuration Classes

### StrategyConfig (Pydantic Model)

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

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
        """Load from YAML file."""
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
```

---

## Summary

All data structures are now explicitly defined with:
- Field names and types
- Validation rules
- Default values
- Helper methods
- Usage examples

These can be directly implemented as Python classes using `dataclasses` or `pydantic` models.

# Canonical Data Contracts

This document describes the canonical data contracts used across all system components to ensure consistent communication between backtests, optimization, daily signal generation, newsletter, paper trading, and manual trade tracking.

## Overview

The canonical contracts are defined in `trading_system/models/contracts.py` and provide a single source of truth for data structures used throughout the system.

## Core Contracts

### 1. Signal

Represents a trading opportunity identified by a strategy.

```python
from trading_system.models.contracts import Signal, AssetClass, SignalIntent

signal = Signal(
    symbol="AAPL",
    asset_class=AssetClass.EQUITY,
    timestamp=pd.Timestamp.now(),
    side="BUY",
    intent=SignalIntent.EXECUTE_NEXT_OPEN,
    confidence=0.85,  # 0.0 to 1.0
    rationale_tags={
        "technical": "Momentum breakout above 20-day MA",
        "news": "Positive earnings report",
        "conviction": "HIGH"
    },
    entry_price=150.0,
    stop_price=145.0,
    bucket="safe_sp500",  # Strategy bucket
    strategy_name="momentum"
)
```

**Fields:**
- `symbol`: Stock/crypto ticker
- `asset_class`: AssetClass.EQUITY or AssetClass.CRYPTO
- `timestamp`: When signal was generated
- `side`: "BUY" or "SELL"
- `intent`: Execution intent (EXECUTE_NEXT_OPEN, EXECUTE_MARKET, etc.)
- `confidence`: Signal confidence score (0.0 to 1.0)
- `rationale_tags`: Dictionary of reasoning (technical, news, etc.)
- `entry_price`: Expected entry price
- `stop_price`: Initial stop price
- `bucket`: Strategy bucket identifier (optional)
- `strategy_name`: Strategy that generated the signal (optional)

### 2. Allocation

Defines position sizing and risk allocation for a signal.

```python
from trading_system.models.contracts import Allocation

allocation = Allocation(
    symbol="AAPL",
    signal_timestamp=pd.Timestamp.now(),
    recommended_position_size_dollars=10000.0,
    recommended_position_size_percent=2.0,  # % of portfolio
    risk_budget_used=1.0,
    max_positions_constraint_applied=False,
    liquidity_flags=[],
    capacity_flags=[],
    quantity=65,  # Calculated shares
    max_adv_percent=0.5,  # % of ADV20
    notes="Strong technical setup"
)
```

**Fields:**
- `symbol`: Stock/crypto ticker
- `signal_timestamp`: Timestamp of associated signal
- `recommended_position_size_dollars`: Dollar amount to allocate
- `recommended_position_size_percent`: Percentage of portfolio
- `risk_budget_used`: Amount of risk budget consumed
- `max_positions_constraint_applied`: Whether position limits were hit
- `liquidity_flags`: List of liquidity warnings
- `capacity_flags`: List of capacity warnings
- `quantity`: Number of shares/units to buy
- `max_adv_percent`: Maximum % of average daily volume

### 3. TradePlan

Defines how to execute and manage a position.

```python
from trading_system.models.contracts import TradePlan, OrderMethod, StopLogicType

trade_plan = TradePlan(
    symbol="AAPL",
    signal_timestamp=pd.Timestamp.now(),
    entry_method=OrderMethod.MOO,  # Market-on-open
    entry_price=150.0,
    stop_logic=StopLogicType.ATR_TRAILING,
    stop_price=145.0,
    stop_params={"atr_mult": 2.5},
    exit_logic="ma_cross",
    exit_params={"ma_period": 20},
    time_stop_days=30,  # Optional max holding period
    allocation=allocation,  # Link to allocation
    notes="Execute at market open"
)
```

**Fields:**
- `symbol`: Stock/crypto ticker
- `signal_timestamp`: Timestamp of associated signal
- `entry_method`: OrderMethod (MOO, MKT, LIMIT)
- `entry_price`: Expected entry price
- `stop_logic`: StopLogicType (ATR_TRAILING, MA_CROSS, etc.)
- `stop_price`: Initial stop price
- `stop_params`: Dictionary of stop parameters
- `exit_logic`: Exit logic type
- `exit_params`: Dictionary of exit parameters
- `time_stop_days`: Optional maximum holding period
- `allocation`: Associated Allocation object

### 4. PositionRecord

Unified format for tracking all positions (system, paper, manual).

```python
from trading_system.models.contracts import PositionRecord, PositionSource

position = PositionRecord(
    symbol="AAPL",
    asset_class=AssetClass.EQUITY,
    source=PositionSource.SYSTEM,  # or PAPER, MANUAL
    open_timestamp=pd.Timestamp.now(),
    entry_price=150.0,
    quantity=65,
    side="LONG",
    stop_price=145.0,
    initial_stop_price=145.0,
    bucket="safe_sp500",
    strategy_name="momentum",
    notes="Entered on momentum breakout",
    tags=["high_conviction", "earnings_play"]
)
```

**Fields:**
- `symbol`: Stock/crypto ticker
- `asset_class`: AssetClass.EQUITY or AssetClass.CRYPTO
- `source`: PositionSource (SYSTEM, PAPER, MANUAL)
- `open_timestamp`: When position was opened
- `close_timestamp`: When position was closed (optional)
- `entry_price`: Entry fill price
- `exit_price`: Exit fill price (optional)
- `quantity`: Number of shares/units
- `side`: "LONG" or "SHORT"
- `fills`: List of fill records
- `pnl`: Realized P&L (if closed)
- `r_multiple`: R-multiple (if closed)
- `stop_price`: Current stop price
- `initial_stop_price`: Original stop price
- `notes`: Free-form notes
- `tags`: List of tags for categorization
- `bucket`: Strategy bucket
- `strategy_name`: Strategy that generated the position

### 5. DailySignalBatch

Container for a day's worth of signals, allocations, and trade plans.

```python
from trading_system.models.contracts import DailySignalBatch

batch = DailySignalBatch(
    generation_date=pd.Timestamp.now(),
    signals=[signal1, signal2, ...],
    allocations=[alloc1, alloc2, ...],
    trade_plans=[plan1, plan2, ...],
    bucket_summaries={
        "safe_sp500": {
            "total_signals": 3,
            "asset_class": "equity",
            "avg_confidence": 0.75
        }
    },
    metadata={
        "asset_class": "equity",
        "universe_type": "NASDAQ-100",
        "symbols_analyzed": 100
    }
)

# Get top signals by confidence
top_signals = batch.get_top_signals(n=10)

# Get signals for specific bucket
bucket_signals = batch.get_signals_by_bucket("safe_sp500")
```

## Enumerations

### AssetClass
- `EQUITY`: Equity/stock assets
- `CRYPTO`: Cryptocurrency assets

### SignalIntent
- `EXECUTE_NEXT_OPEN`: Execute at next market open
- `EXECUTE_MARKET`: Execute immediately at market
- `EXECUTE_LIMIT`: Execute with limit order
- `EXECUTE_MOO`: Execute with market-on-open order

### OrderMethod
- `MOO`: Market-on-open
- `MKT`: Market order
- `LIMIT`: Limit order

### PositionSource
- `SYSTEM`: System-generated (backtest/live)
- `PAPER`: Paper trading account
- `MANUAL`: Manually entered by user

### StopLogicType
- `ATR_TRAILING`: ATR-based trailing stop
- `MA_CROSS`: Moving average cross
- `FIXED_PERCENT`: Fixed percentage stop
- `TIGHTENED_ATR`: Tightened ATR stop (crypto)

### ExitLogicType
- `MA_CROSS`: Moving average cross
- `TIME_STOP`: Time-based exit
- `PROFIT_TARGET`: Profit target hit
- `MANUAL`: Manual exit

## Usage Examples

### Daily Signal Generation

```python
from trading_system.integration.daily_signal_service import DailySignalService

# Initialize service
service = DailySignalService()

# Generate signals for equity bucket
batch = await service.generate_daily_signals(
    asset_class="equity",
    bucket="safe_sp500"
)

# Access signals
for signal in batch.signals:
    print(f"{signal.symbol}: {signal.side} @ ${signal.entry_price}")
    print(f"  Confidence: {signal.confidence:.2f}")
    print(f"  Rationale: {signal.rationale_tags}")
```

### CLI Usage

```bash
# Generate daily signals for equity
python -m trading_system generate-daily-signals --asset-class equity --bucket safe_sp500

# Generate signals and save to file
python -m trading_system generate-daily-signals \
    --asset-class crypto \
    --bucket aggressive_crypto \
    --output signals.json

# Use custom config
python -m trading_system generate-daily-signals \
    --asset-class equity \
    --config configs/production_run_config.yaml \
    --output daily_signals.json
```

### Integration with Newsletter

```python
from trading_system.integration.daily_signal_service import DailySignalService
from trading_system.output.email.email_service import EmailService

# Generate signals
service = DailySignalService()
batch = await service.generate_daily_signals(asset_class="equity")

# Convert to newsletter format
newsletter_data = {
    "date": batch.generation_date,
    "top_picks": batch.get_top_signals(n=5),
    "bucket_summaries": batch.bucket_summaries,
    "total_signals": len(batch.signals)
}

# Send newsletter (implementation by Agent 3)
# email_service.send_newsletter(newsletter_data)
```

### Integration with Paper Trading

```python
from trading_system.integration.daily_signal_service import DailySignalService
from trading_system.execution.paper_trader import PaperTrader

# Generate signals
service = DailySignalService()
batch = await service.generate_daily_signals(asset_class="equity")

# Execute trade plans (implementation by Agent 4)
# paper_trader = PaperTrader()
# for trade_plan in batch.trade_plans:
#     paper_trader.execute_trade_plan(trade_plan)
```

## Migration from Legacy Models

The canonical contracts coexist with the existing `Signal`, `Position`, `Order`, and `Fill` models in `trading_system/models/`. 

- **Legacy Signal** (`trading_system.models.Signal`): Used internally by strategies
- **Canonical Signal** (`trading_system.models.contracts.Signal`): Used for cross-component communication

To import canonical contracts:

```python
from trading_system.models.contracts import (
    Signal,
    Allocation,
    TradePlan,
    PositionRecord,
    DailySignalBatch
)
```

To import legacy models:

```python
from trading_system.models import (
    Signal,  # Legacy signal
    Position,
    Order,
    Fill
)
```

## Testing

Integration tests are located in `tests/integration/test_daily_signal_service.py`.

Run tests:

```bash
# Run all integration tests
pytest tests/integration/test_daily_signal_service.py -v

# Run specific test
pytest tests/integration/test_daily_signal_service.py::TestDailySignalService::test_generate_daily_signals_basic -v
```

## Next Steps

1. **Agent 2**: Implement Bucket A (Safe S&P) and Bucket B (Aggressive Crypto) strategies that output canonical signals
2. **Agent 3**: Build newsletter service that consumes `DailySignalBatch`
3. **Agent 4**: Build paper trading service that executes `TradePlan` objects
4. **All Agents**: Update existing code to use canonical contracts where appropriate

## Benefits

1. **Single Source of Truth**: All components use the same data structures
2. **Type Safety**: Dataclasses with validation ensure data integrity
3. **Extensibility**: Easy to add new fields without breaking existing code
4. **Testability**: Mock-friendly interfaces for unit and integration tests
5. **Documentation**: Self-documenting code with clear field definitions

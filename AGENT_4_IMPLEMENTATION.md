# Agent 4 Implementation: Paper Trading + Manual Trades

## Overview

Agent 4 has implemented the foundation for paper trading execution and manual trade tracking, as specified in `PROJECT_NEXT_STEPS.md`. This implementation provides:

1. **Paper Trading Execution Pipeline** - Execute orders via broker adapter with order lifecycle tracking
2. **Manual Trade Storage** - Track user-managed positions with CRUD operations
3. **Unified Positions View** - Merge positions from backtest, paper trading, and manual sources

## Components Delivered

### 1. Paper Trading Execution Pipeline

**Location**: `trading_system/execution/paper_trading.py`

**Key Classes**:
- `PaperTradingConfig` - Configuration for paper trading execution
- `PaperTradingRunner` - Manages order lifecycle from submission to fill
- `OrderLifecycle` - Tracks individual order status and retry logic

**Features**:
- Order submission via broker adapter (Alpaca)
- Automatic retry logic for transient failures
- Order lifecycle tracking (pending → filled/rejected)
- Daily order limits and safety checks
- Account reconciliation with broker
- Fill export to CSV for analysis

**Usage Example**:
```python
from trading_system.adapters.alpaca_adapter import AlpacaAdapter
from trading_system.adapters.base_adapter import AdapterConfig
from trading_system.execution.paper_trading import PaperTradingConfig, PaperTradingRunner

# Setup adapter
config = AdapterConfig(
    api_key="your_alpaca_key",
    api_secret="your_alpaca_secret",
    paper_trading=True
)
adapter = AlpacaAdapter(config)

# Create runner
paper_config = PaperTradingConfig(adapter_config=config)
runner = PaperTradingRunner(config=paper_config, adapter=adapter)

# Submit orders
with adapter:
    results = runner.submit_orders([order1, order2])
    
    # Check status
    summary = runner.get_order_summary()
    print(f"Filled: {summary['filled']}, Pending: {summary['pending']}")
    
    # Reconcile positions
    positions = runner.reconcile_positions()
```

### 2. Manual Trade Storage

**Location**: `trading_system/storage/manual_trades.py`

**Key Classes**:
- `ManualTrade` - Data model for user-managed trades
- `ManualTradeDatabase` - SQLite database for CRUD operations

**Features**:
- Full CRUD operations (create, read, update, delete)
- Automatic P&L calculation (realized and unrealized)
- Open/closed trade filtering
- Date range queries
- Tag and notes support
- Conversion to Position model for unified reporting

**Database Schema**:
```sql
CREATE TABLE manual_trades (
    trade_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_date TEXT NOT NULL,
    entry_price REAL NOT NULL,
    quantity INTEGER NOT NULL,
    stop_price REAL NOT NULL,
    initial_stop_price REAL NOT NULL,
    exit_date TEXT,
    exit_price REAL,
    exit_reason TEXT,
    realized_pnl REAL DEFAULT 0.0,
    unrealized_pnl REAL DEFAULT 0.0,
    notes TEXT,
    tags TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
```

**Usage Example**:
```python
from trading_system.storage.manual_trades import ManualTrade, ManualTradeDatabase
from trading_system.models.positions import PositionSide
from datetime import datetime

db = ManualTradeDatabase()

# Create a new trade
trade = ManualTrade(
    trade_id=str(uuid.uuid4()),
    symbol="AAPL",
    asset_class="equity",
    side=PositionSide.LONG,
    entry_date=datetime.now(),
    entry_price=150.0,
    quantity=100,
    stop_price=145.0,
    initial_stop_price=145.0,
    notes="Earnings play"
)
db.create_trade(trade)

# Update stop price
trade.stop_price = 148.0
db.update_trade(trade)

# Close trade
db.close_trade(trade.trade_id, datetime.now(), 160.0, "profit_target")

# Query trades
open_trades = db.get_open_trades()
closed_trades = db.get_closed_trades()
```

### 3. Unified Positions View

**Location**: `trading_system/reporting/unified_positions.py`

**Key Classes**:
- `UnifiedPositionView` - Aggregates positions from multiple sources
- `UnifiedPosition` - Position with source tracking
- `PositionSource` - Enum for position sources (backtest, paper, manual)

**Features**:
- Merge positions from backtest, paper trading, and manual sources
- Exposure summary (gross, net, by asset class)
- Position grouping by symbol
- Export to CSV and DataFrame
- Console summary printing

**Usage Example**:
```python
from trading_system.reporting.unified_positions import UnifiedPositionView
from trading_system.storage.manual_trades import ManualTradeDatabase
from trading_system.adapters.alpaca_adapter import AlpacaAdapter

# Setup
manual_db = ManualTradeDatabase()
paper_adapter = AlpacaAdapter(config)
paper_adapter.connect()

# Create unified view
view = UnifiedPositionView(
    manual_db=manual_db,
    paper_adapter=paper_adapter
)

# Get all positions
positions = view.get_all_positions(
    include_paper=True,
    include_manual=True,
    open_only=True
)

# Get exposure summary
exposure = view.get_exposure_summary()
print(f"Gross Exposure: ${exposure['gross_exposure']:,.2f}")
print(f"Net Exposure: ${exposure['net_exposure']:,.2f}")

# Export to CSV
view.export_to_csv("positions.csv", open_only=True)

# Print summary
view.print_summary()
```

## CLI Commands

### Paper Trading Commands

**Status**: Show account status and positions
```bash
python -m trading_system paper status \
    --api-key YOUR_KEY \
    --api-secret YOUR_SECRET
```

**Positions**: List current paper trading positions
```bash
python -m trading_system paper positions \
    --export positions.csv
```

**Reconcile**: Reconcile positions with broker
```bash
python -m trading_system paper reconcile
```

### Manual Trade Commands

**Add**: Create a new manual trade
```bash
python -m trading_system manual add AAPL LONG 100 150.0 145.0 \
    --asset-class equity \
    --notes "Earnings play" \
    --tags "tech,earnings"
```

**Close**: Close an existing trade
```bash
python -m trading_system manual close TRADE_ID 160.0 \
    --reason "profit_target"
```

**Update**: Update trade details
```bash
python -m trading_system manual update TRADE_ID \
    --stop-price 148.0 \
    --notes "Updated stop"
```

**List**: List all trades
```bash
python -m trading_system manual list \
    --open-only \
    --export trades.csv
```

**Show**: Show trade details
```bash
python -m trading_system manual show TRADE_ID
```

**Delete**: Delete a trade
```bash
python -m trading_system manual delete TRADE_ID --confirm
```

### Unified Positions Command

**View all positions**:
```bash
python -m trading_system positions \
    --include-paper \
    --include-manual \
    --open-only \
    --export all_positions.csv
```

## Environment Setup

### Required Environment Variables

For paper trading with Alpaca:
```bash
export ALPACA_API_KEY="your_alpaca_api_key"
export ALPACA_API_SECRET="your_alpaca_api_secret"
export ALPACA_BASE_URL="https://paper-api.alpaca.markets"
export ALPACA_PAPER=true
```

### Dependencies

The following dependency is required for Alpaca integration:
```bash
pip install alpaca-trade-api
```

## Database Files

The implementation creates the following database files:

- `results/manual_trades.db` - Manual trade storage
- `results/backtest_results.db` - Backtest results (existing)
- `logs/paper_trading/` - Paper trading execution logs

## Testing

Comprehensive test suites have been created:

- `tests/test_manual_trades.py` - Manual trade CRUD operations
- `tests/test_paper_trading.py` - Paper trading execution pipeline
- `tests/test_unified_positions.py` - Unified positions view

**Run tests**:
```bash
pytest tests/test_manual_trades.py -v
pytest tests/test_paper_trading.py -v
pytest tests/test_unified_positions.py -v
```

## Integration with Other Agents

### Agent 1 (Contracts + Orchestration)
- Uses existing `Order`, `Fill`, and `Position` models
- Compatible with signal generation contracts
- Ready to consume `TradePlan` objects when defined

### Agent 2 (Strategy & Data)
- Can consume signals from strategy buckets
- Paper trading runner accepts `Order` objects from any strategy

### Agent 3 (Newsletter + Scheduler)
- Unified positions view can be used in newsletter generation
- Paper trading can be triggered by scheduler after signal generation

## Next Steps

### Immediate Integration Tasks

1. **Wire CLI commands into main CLI** - Add paper/manual/positions commands to main CLI parser
2. **Create daily execution workflow** - Combine signal generation → paper trading → newsletter
3. **Add position monitoring** - Daily reconciliation and P&L updates
4. **Implement TradePlan contract** - Once Agent 1 defines the contract

### Future Enhancements

1. **Order types** - Support limit orders, stop orders
2. **Position sizing** - Integrate with risk management
3. **Multi-broker support** - Add Interactive Brokers adapter
4. **Real-time monitoring** - WebSocket integration for live updates
5. **Alerts** - Email/SMS alerts for fills, stops hit, etc.
6. **Performance tracking** - Compare paper vs backtest performance

## File Structure

```
trading_system/
├── adapters/
│   ├── base_adapter.py          # Existing broker interface
│   └── alpaca_adapter.py        # Existing Alpaca implementation
├── cli/
│   └── commands/
│       ├── paper_trading.py     # NEW: Paper trading CLI
│       ├── manual_trades.py     # NEW: Manual trades CLI
│       └── positions.py         # NEW: Unified positions CLI
├── execution/
│   └── paper_trading.py         # NEW: Paper trading runner
├── reporting/
│   └── unified_positions.py     # NEW: Unified positions view
└── storage/
    └── manual_trades.py         # NEW: Manual trade storage

tests/
├── test_manual_trades.py        # NEW: Manual trade tests
├── test_paper_trading.py        # NEW: Paper trading tests
└── test_unified_positions.py    # NEW: Unified positions tests
```

## Notes

- All new code follows existing patterns and conventions
- Comprehensive error handling and logging
- Type hints throughout
- Docstrings for all public APIs
- SQLite for lightweight storage (can migrate to PostgreSQL later)
- Mock-first development approach for testing

## Contact

For questions or issues with Agent 4 deliverables, refer to:
- `PROJECT_NEXT_STEPS.md` - Overall roadmap
- `agent-files/` - Detailed architecture documentation
- This file - Agent 4 specific implementation details

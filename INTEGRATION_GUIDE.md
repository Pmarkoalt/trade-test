# Integration Guide: Complete System Workflow

## Overview

This guide demonstrates how Agent 4's paper trading and manual trade components integrate with the work from Agents 1-3 to create a complete daily trading workflow.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DAILY WORKFLOW                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Agent 1: Signal Generation (Canonical Contracts)               │
│  - Generate daily signals for equity/crypto buckets             │
│  - Output: Signal, Allocation, TradePlan objects                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Agent 2: Strategy Execution (Bucket A/B)                       │
│  - Safe S&P 500 bucket                                          │
│  - Aggressive crypto bucket                                     │
│  - Output: Orders ready for execution                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Agent 3: Newsletter Generation                                 │
│  - Format signals into HTML email                               │
│  - Send daily digest                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Agent 4: Paper Trading Execution (THIS IMPLEMENTATION)         │
│  - Submit orders to Alpaca paper account                        │
│  - Track order lifecycle                                        │
│  - Reconcile positions                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Unified Positions View                                         │
│  - Merge paper + manual + backtest positions                    │
│  - Calculate exposure metrics                                   │
│  - Export for analysis                                          │
└─────────────────────────────────────────────────────────────────┘
```

## Complete Daily Workflow Example

### Step 1: Generate Daily Signals (Agent 1 + Agent 2)

```bash
# Generate signals for equity bucket
python -m trading_system generate-daily-signals \
    --asset-class equity \
    --bucket safe_sp500 \
    --config configs/equity_bucket_a.yaml \
    --output signals/equity_$(date +%Y%m%d).json

# Generate signals for crypto bucket
python -m trading_system generate-daily-signals \
    --asset-class crypto \
    --bucket aggressive_crypto \
    --config configs/crypto_bucket_b.yaml \
    --output signals/crypto_$(date +%Y%m%d).json
```

### Step 2: Send Newsletter (Agent 3)

```bash
# Send daily newsletter with all signals
python -m trading_system send-newsletter
```

### Step 3: Execute Paper Trading (Agent 4)

```python
# Example: Execute orders from signal generation
from trading_system.execution.paper_trading import PaperTradingConfig, PaperTradingRunner
from trading_system.adapters.alpaca_adapter import AlpacaAdapter
from trading_system.adapters.base_adapter import AdapterConfig
import os

# Setup Alpaca adapter
adapter_config = AdapterConfig(
    api_key=os.getenv("ALPACA_API_KEY"),
    api_secret=os.getenv("ALPACA_API_SECRET"),
    paper_trading=True
)

# Create paper trading runner
paper_config = PaperTradingConfig(adapter_config=adapter_config)
adapter = AlpacaAdapter(adapter_config)

with adapter:
    runner = PaperTradingRunner(config=paper_config, adapter=adapter)

    # Submit orders (from signal generation)
    results = runner.submit_orders(orders)

    # Check status
    summary = runner.get_order_summary()
    print(f"Filled: {summary['filled']}, Pending: {summary['pending']}")

    # Reconcile positions
    positions = runner.reconcile_positions()
```

### Step 4: Track Manual Trades (Agent 4)

```bash
# Add a manual trade you executed yourself
python -m trading_system manual add AAPL LONG 100 150.0 145.0 \
    --asset-class equity \
    --notes "Manual entry on breakout" \
    --tags "tech,manual"

# Update stop price as position moves
python -m trading_system manual update TRADE_ID --stop-price 152.0

# Close when exited
python -m trading_system manual close TRADE_ID 160.0 --reason "profit_target"
```

### Step 5: View Unified Positions (Agent 4)

```bash
# View all positions across all sources
python -m trading_system positions \
    --include-paper \
    --include-manual \
    --open-only \
    --export daily_positions_$(date +%Y%m%d).csv
```

## Integration Points

### Agent 1 → Agent 4

Agent 1 generates `Order` objects using canonical contracts. Agent 4's `PaperTradingRunner` consumes these orders:

```python
from trading_system.models.orders import Order, SignalSide
from trading_system.execution.paper_trading import PaperTradingRunner

# Agent 1 generates orders
orders = [
    Order(
        order_id=str(uuid.uuid4()),
        symbol="AAPL",
        asset_class="equity",
        date=pd.Timestamp.now(),
        execution_date=pd.Timestamp.now() + pd.Timedelta(days=1),
        side=SignalSide.BUY,
        quantity=100,
        signal_date=pd.Timestamp.now(),
        expected_fill_price=150.0,
        stop_price=145.0,
    )
]

# Agent 4 executes them
runner.submit_orders(orders)
```

### Agent 3 → Agent 4

Agent 3's newsletter can include positions from Agent 4's unified view:

```python
from trading_system.reporting.unified_positions import UnifiedPositionView

# Get positions for newsletter
view = UnifiedPositionView(manual_db=db, paper_adapter=adapter)
exposure = view.get_exposure_summary()

# Include in newsletter
newsletter_data = {
    "signals": daily_signals,
    "current_positions": exposure,
    "open_positions_count": exposure['total_positions'],
    "gross_exposure": exposure['gross_exposure'],
}
```

### Agent 4 Internal Integration

Manual trades and paper trades merge seamlessly:

```python
from trading_system.reporting.unified_positions import UnifiedPositionView
from trading_system.storage.manual_trades import ManualTradeDatabase

# Setup
manual_db = ManualTradeDatabase()
paper_adapter = AlpacaAdapter(config)
paper_adapter.connect()

# Create unified view
view = UnifiedPositionView(manual_db=manual_db, paper_adapter=paper_adapter)

# Get all positions (paper + manual)
all_positions = view.get_all_positions(
    include_paper=True,
    include_manual=True,
    open_only=True
)

# Calculate total exposure
exposure = view.get_exposure_summary()
print(f"Total Exposure: ${exposure['gross_exposure']:,.2f}")
print(f"Paper Positions: {len([p for p in all_positions if p.source == 'paper'])}")
print(f"Manual Positions: {len([p for p in all_positions if p.source == 'manual'])}")
```

## Automated Daily Workflow Script

Create a script that runs the complete workflow:

```python
#!/usr/bin/env python
"""Daily trading workflow automation."""

import os
import sys
from datetime import datetime
from pathlib import Path

from trading_system.adapters.alpaca_adapter import AlpacaAdapter
from trading_system.adapters.base_adapter import AdapterConfig
from trading_system.execution.paper_trading import PaperTradingConfig, PaperTradingRunner
from trading_system.reporting.unified_positions import UnifiedPositionView
from trading_system.storage.manual_trades import ManualTradeDatabase


def run_daily_workflow():
    """Run complete daily workflow."""

    print(f"\n{'='*80}")
    print(f"DAILY TRADING WORKFLOW - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    # Step 1: Generate signals (Agent 1 + Agent 2)
    print("Step 1: Generating daily signals...")
    # This would call the signal generation service
    # signals = generate_daily_signals()

    # Step 2: Send newsletter (Agent 3)
    print("Step 2: Sending newsletter...")
    # This would call the newsletter service
    # send_newsletter(signals)

    # Step 3: Execute paper trading (Agent 4)
    print("Step 3: Executing paper trading orders...")
    adapter_config = AdapterConfig(
        api_key=os.getenv("ALPACA_API_KEY"),
        api_secret=os.getenv("ALPACA_API_SECRET"),
        paper_trading=True
    )

    paper_config = PaperTradingConfig(adapter_config=adapter_config)
    adapter = AlpacaAdapter(adapter_config)

    try:
        with adapter:
            runner = PaperTradingRunner(config=paper_config, adapter=adapter)

            # Submit orders (would come from signal generation)
            # results = runner.submit_orders(orders)

            # Reconcile positions
            positions = runner.reconcile_positions()
            print(f"  ✓ Reconciled {len(positions)} paper positions")

            # Get account status
            account = runner.get_account_info()
            print(f"  ✓ Account equity: ${account.equity:,.2f}")

    except Exception as e:
        print(f"  ✗ Error in paper trading: {e}")
        return 1

    # Step 4: Update unified positions view
    print("Step 4: Updating unified positions view...")
    manual_db = ManualTradeDatabase()
    view = UnifiedPositionView(manual_db=manual_db, paper_adapter=adapter)

    exposure = view.get_exposure_summary()
    print(f"  ✓ Total positions: {exposure['total_positions']}")
    print(f"  ✓ Gross exposure: ${exposure['gross_exposure']:,.2f}")
    print(f"  ✓ Net exposure: ${exposure['net_exposure']:,.2f}")

    # Export daily snapshot
    output_dir = Path("results/daily_snapshots")
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_file = output_dir / f"positions_{datetime.now().strftime('%Y%m%d')}.csv"
    view.export_to_csv(str(snapshot_file), open_only=True)
    print(f"  ✓ Exported snapshot to {snapshot_file}")

    print(f"\n{'='*80}")
    print("DAILY WORKFLOW COMPLETE")
    print(f"{'='*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(run_daily_workflow())
```

Save this as `scripts/daily_workflow.py` and run:

```bash
python scripts/daily_workflow.py
```

## CLI Quick Reference

### Paper Trading
```bash
# Check account status
python -m trading_system paper status

# View positions
python -m trading_system paper positions --export positions.csv

# Reconcile with broker
python -m trading_system paper reconcile
```

### Manual Trades
```bash
# Add trade
python -m trading_system manual add SYMBOL SIDE QTY PRICE STOP

# List trades
python -m trading_system manual list --open-only

# Close trade
python -m trading_system manual close TRADE_ID EXIT_PRICE

# Update trade
python -m trading_system manual update TRADE_ID --stop-price NEW_STOP
```

### Unified Positions
```bash
# View all positions
python -m trading_system positions --open-only --export all_positions.csv
```

## Environment Variables Required

```bash
# Alpaca Paper Trading
export ALPACA_API_KEY="your_alpaca_api_key"
export ALPACA_API_SECRET="your_alpaca_api_secret"
export ALPACA_BASE_URL="https://paper-api.alpaca.markets"

# Email (for newsletter - Agent 3)
export SMTP_HOST="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USER="your_email@gmail.com"
export SMTP_PASSWORD="your_app_password"
export EMAIL_FROM="your_email@gmail.com"
export EMAIL_TO="recipient@example.com"
```

## Testing the Integration

1. **Test signal generation**:
   ```bash
   python -m trading_system generate-daily-signals --asset-class equity --bucket safe_sp500
   ```

2. **Test newsletter**:
   ```bash
   python -m trading_system send-newsletter --test
   ```

3. **Test paper trading**:
   ```bash
   python -m trading_system paper status
   ```

4. **Test manual trades**:
   ```bash
   python -m trading_system manual add TEST LONG 1 100.0 95.0
   python -m trading_system manual list
   ```

5. **Test unified view**:
   ```bash
   python -m trading_system positions --open-only
   ```

## Next Steps

1. Set up cron job or scheduler for daily execution
2. Configure email alerts for fills and stops
3. Add monitoring dashboard
4. Implement risk limits and position sizing
5. Add performance tracking and comparison

## Support

- See `AGENT_4_IMPLEMENTATION.md` for detailed Agent 4 documentation
- See `PROJECT_NEXT_STEPS.md` for overall roadmap
- See `agent-files/` for architecture details

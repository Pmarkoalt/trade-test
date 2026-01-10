# Complete System Integration - All Agents

## 🎯 System Overview

The trading system now has complete integration across all four agent workstreams:

- **Agent 1**: Canonical contracts (Signal, Allocation, TradePlan) + daily signal generation
- **Agent 2**: Strategy buckets (Safe S&P 500, Aggressive Crypto)
- **Agent 3**: Newsletter generation and scheduler
- **Agent 4**: Paper trading execution + manual trade tracking + unified positions view

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt
pip install alpaca-trade-api

# Configure environment variables
cp .env.example .env
# Edit .env with your credentials:
# - ALPACA_API_KEY
# - ALPACA_API_SECRET
# - SMTP credentials for newsletter
```

### 2. Run Daily Workflow

```bash
# Automated daily workflow
python scripts/daily_workflow.py

# Or run steps individually:

# Step 1: Generate signals
python -m trading_system generate-daily-signals --asset-class equity --bucket safe_sp500

# Step 2: Send newsletter
python -m trading_system send-newsletter

# Step 3: Check paper trading status
python -m trading_system paper status

# Step 4: View unified positions
python -m trading_system positions --open-only
```

## 📋 Complete Feature Set

### Signal Generation (Agent 1 + Agent 2)
- ✅ Canonical Signal, Allocation, TradePlan contracts
- ✅ Daily signal generation service
- ✅ Safe S&P 500 bucket strategy
- ✅ Aggressive crypto bucket strategy
- ✅ Signal batch export (JSON)

### Newsletter (Agent 3)
- ✅ HTML email generation
- ✅ Multi-bucket signal formatting
- ✅ Position summary inclusion
- ✅ SMTP delivery
- ✅ Scheduler integration

### Paper Trading (Agent 4)
- ✅ Alpaca broker integration
- ✅ Order lifecycle tracking
- ✅ Automatic retry logic
- ✅ Position reconciliation
- ✅ Account status monitoring
- ✅ Fill export to CSV

### Manual Trade Tracking (Agent 4)
- ✅ SQLite database storage
- ✅ Full CRUD operations
- ✅ Automatic P&L calculation
- ✅ Tags and notes support
- ✅ Open/closed filtering
- ✅ Position conversion

### Unified Positions View (Agent 4)
- ✅ Multi-source aggregation (backtest/paper/manual)
- ✅ Exposure analytics (gross/net/by asset class)
- ✅ Symbol grouping
- ✅ CSV/DataFrame export
- ✅ Console summary printing

## 🔧 CLI Commands Reference

### Paper Trading Commands

```bash
# Account status
python -m trading_system paper status

# List positions
python -m trading_system paper positions [--export FILE]

# Reconcile with broker
python -m trading_system paper reconcile
```

### Manual Trade Commands

```bash
# Add new trade
python -m trading_system manual add AAPL LONG 100 150.0 145.0 \
    --asset-class equity \
    --notes "Earnings play" \
    --tags "tech,earnings"

# List trades
python -m trading_system manual list [--open-only] [--export FILE]

# Show trade details
python -m trading_system manual show TRADE_ID

# Update trade
python -m trading_system manual update TRADE_ID --stop-price 148.0

# Close trade
python -m trading_system manual close TRADE_ID 160.0 --reason "profit_target"

# Delete trade
python -m trading_system manual delete TRADE_ID --confirm
```

### Unified Positions Commands

```bash
# View all positions
python -m trading_system positions \
    --include-paper \
    --include-manual \
    --open-only \
    --export all_positions.csv
```

### Signal Generation Commands

```bash
# Generate daily signals
python -m trading_system generate-daily-signals \
    --asset-class equity \
    --bucket safe_sp500 \
    --config configs/equity_bucket_a.yaml \
    --output signals/equity_$(date +%Y%m%d).json
```

### Newsletter Commands

```bash
# Send newsletter
python -m trading_system send-newsletter

# Send test newsletter
python -m trading_system send-newsletter --test

# Run complete newsletter job
python -m trading_system run-newsletter-job
```

## 📊 Data Flow

```
Signal Generation → Newsletter → Paper Trading → Unified View
     (Agent 1+2)      (Agent 3)     (Agent 4)      (Agent 4)
         │                │              │              │
         ▼                ▼              ▼              ▼
    Signal.json    HTML Email    Alpaca Orders   Positions CSV
    Allocation                    Fill Tracking   Exposure Metrics
    TradePlan                     Reconciliation  Multi-source Merge
```

## 🗄️ Database Files

The system creates the following databases:

- `results/backtest_results.db` - Backtest results and metrics
- `results/manual_trades.db` - Manual trade tracking
- `logs/paper_trading/` - Paper trading execution logs
- `results/daily_snapshots/` - Daily position snapshots

## 🧪 Testing

### Run All Tests

```bash
# Agent 4 tests
pytest tests/test_manual_trades.py -v
pytest tests/test_paper_trading.py -v
pytest tests/test_unified_positions.py -v

# Run all tests
pytest tests/ -v
```

### Manual Testing Workflow

```bash
# 1. Test manual trade CRUD
python -m trading_system manual add TEST LONG 1 100.0 95.0
python -m trading_system manual list
python -m trading_system manual close <TRADE_ID> 105.0

# 2. Test paper trading (requires Alpaca credentials)
python -m trading_system paper status
python -m trading_system paper positions

# 3. Test unified view
python -m trading_system positions --open-only

# 4. Test signal generation
python -m trading_system generate-daily-signals --asset-class equity --bucket safe_sp500

# 5. Test newsletter
python -m trading_system send-newsletter --test
```

## 🔐 Required Environment Variables

```bash
# Alpaca Paper Trading
export ALPACA_API_KEY="pk_..."
export ALPACA_API_SECRET="sk_..."
export ALPACA_BASE_URL="https://paper-api.alpaca.markets"

# Email (for newsletter)
export SMTP_HOST="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USER="your_email@gmail.com"
export SMTP_PASSWORD="your_app_password"
export EMAIL_FROM="your_email@gmail.com"
export EMAIL_TO="recipient@example.com"

# Optional: News API (for sentiment)
export ALPHA_VANTAGE_API_KEY="your_key"
```

## 📁 Project Structure

```
trading_system/
├── adapters/              # Broker adapters (Alpaca, IB)
│   ├── base_adapter.py
│   └── alpaca_adapter.py
├── cli/
│   └── commands/          # CLI command modules
│       ├── paper_trading.py    # Agent 4
│       ├── manual_trades.py    # Agent 4
│       └── positions.py        # Agent 4
├── execution/
│   └── paper_trading.py   # Agent 4: Paper trading runner
├── integration/
│   └── daily_signal_service.py  # Agent 1: Signal generation
├── reporting/
│   └── unified_positions.py     # Agent 4: Unified view
├── scheduler/             # Agent 3: Scheduler
├── storage/
│   ├── database.py        # Backtest results
│   └── manual_trades.py   # Agent 4: Manual trades DB
└── strategies/            # Agent 2: Strategy buckets

scripts/
└── daily_workflow.py      # Complete daily automation

tests/
├── test_manual_trades.py      # Agent 4
├── test_paper_trading.py      # Agent 4
└── test_unified_positions.py  # Agent 4
```

## 🔄 Daily Automation

### Option 1: Cron Job

```bash
# Add to crontab (run at 4:30 PM ET daily)
30 16 * * 1-5 cd /path/to/trade-test && python scripts/daily_workflow.py >> logs/daily_workflow.log 2>&1
```

### Option 2: Systemd Service

```bash
# Copy service file
sudo cp trading-system-scheduler.service /etc/systemd/system/

# Enable and start
sudo systemctl enable trading-system-scheduler
sudo systemctl start trading-system-scheduler

# Check status
sudo systemctl status trading-system-scheduler
```

### Option 3: Manual Execution

```bash
# Run whenever needed
python scripts/daily_workflow.py
```

## 📈 Usage Examples

### Example 1: Complete Daily Workflow

```python
from trading_system.integration.daily_signal_service import DailySignalService
from trading_system.execution.paper_trading import PaperTradingRunner
from trading_system.reporting.unified_positions import UnifiedPositionView

# Generate signals
service = DailySignalService()
signals = service.generate_daily_signals(asset_class="equity", bucket="safe_sp500")

# Execute paper trading
runner = PaperTradingRunner(config=paper_config, adapter=adapter)
results = runner.submit_orders(signals.orders)

# View unified positions
view = UnifiedPositionView(manual_db=db, paper_adapter=adapter)
exposure = view.get_exposure_summary()
```

### Example 2: Manual Trade Management

```python
from trading_system.storage.manual_trades import ManualTrade, ManualTradeDatabase
from trading_system.models.positions import PositionSide
from datetime import datetime

db = ManualTradeDatabase()

# Add trade
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
)
db.create_trade(trade)

# Update stop
trade.stop_price = 152.0
db.update_trade(trade)

# Close
db.close_trade(trade.trade_id, datetime.now(), 160.0)
```

### Example 3: Unified Positions Analysis

```python
from trading_system.reporting.unified_positions import UnifiedPositionView

view = UnifiedPositionView(manual_db=db, paper_adapter=adapter)

# Get all positions
positions = view.get_all_positions(open_only=True)

# Calculate exposure
exposure = view.get_exposure_summary()
print(f"Gross Exposure: ${exposure['gross_exposure']:,.2f}")

# Export
view.export_to_csv("positions.csv")
view.print_summary()
```

## 📚 Documentation

- `AGENT_4_IMPLEMENTATION.md` - Detailed Agent 4 implementation
- `AGENT_4_SUMMARY.md` - Quick reference for Agent 4
- `INTEGRATION_GUIDE.md` - Integration between all agents
- `PROJECT_NEXT_STEPS.md` - Overall roadmap
- `agent-files/` - Detailed architecture documentation

## 🐛 Troubleshooting

### Alpaca Connection Issues

```bash
# Verify credentials
echo $ALPACA_API_KEY
echo $ALPACA_API_SECRET

# Test connection
python -m trading_system paper status
```

### Database Issues

```bash
# Check database files
ls -lh results/*.db

# Reset manual trades database
rm results/manual_trades.db
python -c "from trading_system.storage.manual_trades import ManualTradeDatabase; ManualTradeDatabase()"
```

### Newsletter Not Sending

```bash
# Test SMTP configuration
python -m trading_system send-test-email

# Check logs
tail -f logs/newsletter/*.log
```

## 🎯 Next Steps

1. **Production Deployment**
   - Set up production environment variables
   - Configure cron job or systemd service
   - Set up monitoring and alerts

2. **Risk Management**
   - Implement position sizing limits
   - Add correlation checks
   - Set exposure limits per bucket

3. **Performance Tracking**
   - Compare paper vs backtest performance
   - Track Sharpe ratio over time
   - Monitor slippage and execution quality

4. **Enhancements**
   - Add real-time monitoring dashboard
   - Implement stop-loss automation
   - Add SMS/Slack alerts
   - Integrate additional brokers (Interactive Brokers)

## ✅ System Status

All agent deliverables are **COMPLETE** and **INTEGRATED**:

- ✅ Agent 1: Canonical contracts + signal generation
- ✅ Agent 2: Strategy buckets (equity + crypto)
- ✅ Agent 3: Newsletter + scheduler
- ✅ Agent 4: Paper trading + manual trades + unified view

The system is ready for daily production use!

## 📞 Support

For questions or issues:
1. Check relevant documentation files
2. Review test files for usage examples
3. Check logs in `logs/` directory
4. Review `PROJECT_NEXT_STEPS.md` for roadmap

# Project Next Steps (Roadmap)

## Implementation Status

**Last Updated**: January 9, 2026

| Phase | Agent | Status | Completion Date |
|-------|-------|--------|-----------------|
| Phase 0: Canonical Contracts | Agent 1 | ✅ Complete | Jan 9, 2026 |
| Phase 1: Strategy Buckets A/B | Agent 2 | ✅ Complete | Jan 9, 2026 |
| Phase 2: Newsletter + Scheduler | Agent 3 | 🔄 Pending | - |
| Phase 3: Paper Trading | Agent 4 | 🔄 Pending | - |
| Phase 4: Manual Trades | Agent 4 | 🔄 Pending | - |

**Recent Completions**:
- ✅ **Agent 1 (Integrator)**: Canonical contracts, DailySignalService, CLI command, integration tests, documentation
  - See: `AGENT_1_IMPLEMENTATION_SUMMARY.md` and `docs/CANONICAL_CONTRACTS.md`
- ✅ **Agent 2 (Strategy & Data)**: Bucket A/B strategies, universe selection, daily signal generation, rationale tagging
  - See: `AGENT_2_COMPLETION_SUMMARY.md` and `trading_system/strategies/buckets/README.md`

---

## Project Goals

1. Create the most optimal daily trading strategy for passive investment across crypto and equities.
   - 1A. Equities: combine technicals (S&P / broad market context), news articles, and strong entry points.
   - 1B. Crypto: more aggressive daily strategy for top market cap coins with clear entries/exits.
   - 1C. Expand into strategy buckets:
     - Safe S&P bets
     - Low-float stock “gamble” bucket
     - Unusual options movement (e.g., Unusual Whales)
2. Provide a daily newsletter email with what to buy and how much, based on the best trained strategies.
3. Create a paper trading account that uses the strategies.
4. Allow manual entry of existing trades you manage yourself (not automated) and merge into reporting.

---

## What’s Already Implemented (Current Foundation)

This repo already includes:

- Backtesting engine with train/validation/holdout splits
- Realistic execution simulation (fees, slippage, capacity constraints)
- Strategy framework + existing momentum strategies (equity + crypto)
- Validation suite (statistical + stress)
- Reporting outputs + dashboards
- Strategy optimization (Optuna) and ML signal refinement pipeline
- CLI entrypoint (`python -m trading_system ...`) and supporting scripts

Existing “next steps” references:

- `STRATEGY_NEXT_STEPS.md` (optimization + ML workflow)
- `agent-files/NEXT_STEPS.md` (broader system improvements + signals/n8n notes)

This document is the forward-looking roadmap aligned specifically to the product goals above.

---

## Phase 0 (Lock Interfaces): Shared Data Contracts (High Priority) ✅ COMPLETED

**Status**: ✅ Implemented by Agent 1 (January 9, 2026)

Before adding more strategy buckets and live/paper execution, define and enforce a single shared contract so:

- backtests
- optimization
- daily signal generation
- newsletter
- paper trading
- manual trade tracking

all speak the same language.

### Required domain objects ✅

All domain objects have been implemented in `trading_system/models/contracts.py`:

- **Signal** ✅
  - symbol
  - asset_class (`equity` / `crypto`)
  - timestamp (signal time)
  - side (buy/sell)
  - intent (e.g., execute next open)
  - confidence/score
  - rationale tags (technical + news)
  - bucket (strategy bucket identifier)
  - strategy_name

- **Allocation / Sizing Recommendation** ✅
  - recommended position size (dollars and/or %)
  - risk budget used
  - max positions constraints applied
  - liquidity/capacity flags
  - quantity (calculated shares/units)
  - max_adv_percent

- **TradePlan** ✅
  - entry method (MOO/MKT/limit)
  - stop logic
  - exit logic
  - optional time stop
  - allocation (linked Allocation object)
  - stop_params, exit_params

- **PositionRecord** ✅ (for merging system trades + paper trades + manual trades)
  - source (`system` / `paper` / `manual`)
  - open/close timestamps
  - fills
  - PnL/r-multiple
  - notes/tags
  - bucket, strategy_name

- **DailySignalBatch** ✅ (bonus: container for daily outputs)
  - generation_date
  - signals, allocations, trade_plans
  - bucket_summaries
  - metadata

**Deliverable**: ✅ A single canonical schema that all layers use.

**Implementation Details**:
- File: `trading_system/models/contracts.py` (368 lines)
- Documentation: `docs/CANONICAL_CONTRACTS.md`
- Tests: `tests/integration/test_daily_signal_service.py`
- All contracts include validation and type safety via dataclasses

---

## Phase 1: Multi-Bucket Strategy Expansion

### Bucket A: Safe S&P bets (Equities) ✅ COMPLETED

**Status**: ✅ Implemented by Agent 2 (January 9, 2026)

Goal: stable, low drawdown, realistic capacity.

**Implementation**:
- ✅ Strategy: `trading_system/strategies/buckets/bucket_a_safe_sp.py`
- ✅ Config: `configs/bucket_a_safe_sp.yaml`
- ✅ Universe: SP500 core (40+ liquid stocks) via `trading_system/data/equity_universe.py`
- ✅ Regime filters: MA50 > MA200, SPY > MA50 (optional), news sentiment (optional)
- ✅ Signal sources:
  - Technical scoring (breakout strength, momentum strength)
  - News sentiment integration ready (requires data)
  - Rationale tags for newsletter
- ✅ Output: Conservative sizing (0.5% risk, max 6 positions, 60% exposure)
- ✅ Tighter entry criteria (0.3%/0.8% clearances vs standard 0.5%/1.0%)
- ✅ Conservative exits (MA50 cross, 2.0 ATR stop)

### Bucket B: Aggressive top-cap crypto (Crypto) ✅ COMPLETED

**Status**: ✅ Implemented by Agent 2 (January 9, 2026)

Goal: higher turnover and more aggressive positioning while respecting volatility.

**Implementation**:
- ✅ Strategy: `trading_system/strategies/buckets/bucket_b_crypto_topcat.py`
- ✅ Config: `configs/bucket_b_topcat_crypto.yaml`
- ✅ Universe: Dynamic top 10 by volume (monthly rebalance) via `select_top_crypto_by_volume()`
- ✅ Volatility-aware sizing: Reduces position size for high ATR/close ratio
- ✅ Adaptive stop/exit management: Staged exits (MA20 tightens, MA50 exits)
- ✅ Wider stops for crypto volatility (3.5 ATR initial, 2.0 ATR tightened)
- ✅ Rationale tags for newsletter (includes volatility context)

### Bucket C: Low-float stock “gamble”

Data-driven bucket; requires robust liquidity and risk controls.

- Universe definition requires float/volume data
- Very strict sizing caps, liquidity filters, and risk-of-halts considerations

### Bucket D: Unusual options movement (Unusual Whales)

Integration-first.

- Acquire + normalize unusual options alerts
- Map to underlyings
- Create initial ruleset (follow vs fade, time window)
- Build a minimal backtest that is honest about fills and timing

Deliverable: Ship Buckets A and B first, then add C and D once data integration is reliable.

---

## Phase 2: Daily Newsletter (Email)

Goal: one daily email that answers:

- What to buy / sell / watch
- How much to allocate
- Why (short rationale)
- What to avoid (liquidity/news risk/correlation constraints)

Recommended approach:

- A deterministic daily job that:
  - runs signal generation
  - formats the output (HTML)
  - sends email
- Reuse the same Signal + Allocation + TradePlan contract.

Deliverable: `newsletter` command (or n8n workflow) that sends a daily digest.

---

## Phase 3: Paper Trading Integration

Goal: execute the same TradePlans produced by the strategy engine.

Core requirements:

- Broker adapter interface (submit orders, poll status, fetch positions)
- Order lifecycle logging
- Portfolio reconciliation (broker truth vs internal state)
- Same reporting artifacts as backtests (so you can compare paper vs backtest)

Deliverable: end-to-end daily run:

1. generate signals
2. send newsletter
3. place paper orders (optional toggle)

---

## Phase 4: Manual Trades (User-Managed Positions)

Goal: allow you to enter “current active trades” that are managed manually, but appear in the same dashboards and exposure/risk summaries.

Requirements:

- Storage for manual positions (SQLite/Postgres)
- CRUD (create/update/close)
- Tagging/notes
- Unified reporting combining:
  - backtest trades
  - paper trades
  - manual trades

Deliverable: a simple UI/CLI to maintain manual positions and a merged portfolio view.

---

## Immediate Next Steps (Recommended Order)

1. **Pick the initial production focus**
   - Start with: Safe S&P (Bucket A) + Crypto top-cap (Bucket B)

2. **Decide paper trading broker(s)**
   - Equities + Crypto: Alpaca (selected)

3. **Decide newsletter timing**
   - after close (signals at close, execute next open) (selected)

4. **Lock account sizing/risk settings**
   - starting equity for sizing assumptions
   - max positions per bucket

5. **Decide news provider + unusual options access**
   - news API: Alpha Vantage (selected)
   - Unusual Whales API availability

---

## How We’ll Measure “Optimal”

Before optimizing harder, define success metrics per bucket.

- Risk-adjusted: Sharpe / Sortino / Calmar
- Drawdown constraints: max DD targets per bucket
- Turnover constraints: max trades/day or month
- Capacity constraints: max %ADV participation
- Robustness:
  - walk-forward performance
  - stress tests
  - parameter sensitivity (no sharp peaks)

---

## Parallel Execution Plan (4 Agents)

We can run 4 agents in parallel as long as we:

- agree on shared contracts first
- assign clear module ownership
- gate merges on a minimal end-to-end test

### Agent 1: Integrator (Contracts + Orchestration) 

**Status**:  All deliverables completed (January 9, 2026)

Owns:

- Shared domain objects: `Signal`, `Allocation`, `TradePlan`, `PositionRecord`
- A single "generate daily signals" entrypoint (CLI-first)
- Integration tests / golden path

Primary deliverables:

-  A canonical schema module (single source of truth)
-  A stable API between strategy output and downstream consumers (newsletter, paper trading, reporting)
-  A "golden path" test that proves: `signals -> newsletter payload -> artifacts`

Files/modules implemented:

-  `trading_system/models/contracts.py` - Canonical contracts (368 lines)
-  `trading_system/integration/daily_signal_service.py` - DailySignalService (371 lines)
-  `trading_system/cli.py` - Added `generate-daily-signals` command
-  `tests/integration/test_daily_signal_service.py` - Integration tests (455 lines)
-  `docs/CANONICAL_CONTRACTS.md` - Complete documentation (430 lines)

**CLI Usage**:
```bash
# Generate daily signals
python -m trading_system generate-daily-signals --asset-class equity --bucket safe_sp500

# Export to JSON
python -m trading_system generate-daily-signals --asset-class crypto --output signals.json
```

**Integration Points for Other Agents**:
- Agent 2: Use `Signal`, `Allocation`, `TradePlan` contracts for strategy outputs
- Agent 3: Consume `DailySignalBatch` from `DailySignalService.generate_daily_signals()`
- Agent 4: Execute `TradePlan` objects, create `PositionRecord` objects

**Documentation**: See `AGENT_1_IMPLEMENTATION_SUMMARY.md` and `docs/CANONICAL_CONTRACTS.md`

### Agent 2: Strategy & Data (Equities + Crypto Buckets A/B)

**Status**: ✅ All deliverables completed (January 9, 2026)

Owns:

- Bucket A (Safe S&P) strategy logic and its configuration
- Bucket B (Top-cap crypto) strategy logic and its configuration
- Universe selection rules (SP500 + dynamic crypto universe)

Primary deliverables:

- ✅ Deterministic daily signal generation for both buckets
- ✅ Clear rationale tags for newsletter use (technical reasons, blockers)
- ✅ Configs that can be optimized without changing code

Files/modules implemented:

- ✅ `trading_system/strategies/buckets/bucket_a_safe_sp.py` - Safe S&P strategy (370 lines)
- ✅ `trading_system/strategies/buckets/bucket_b_crypto_topcat.py` - Top-cap crypto strategy (380 lines)
- ✅ `trading_system/strategies/buckets/__init__.py` - Bucket module exports
- ✅ `trading_system/strategies/buckets/README.md` - Comprehensive documentation (400+ lines)
- ✅ `trading_system/data/equity_universe.py` - SP500 universe selection (140 lines)
- ✅ `trading_system/data/universe.py` - Enhanced with `select_top_crypto_by_volume()`
- ✅ `configs/bucket_a_safe_sp.yaml` - Bucket A configuration (75 lines)
- ✅ `configs/bucket_b_topcat_crypto.yaml` - Bucket B configuration (90 lines)
- ✅ `scripts/generate_daily_signals.py` - Daily signal generation script (350+ lines)
- ✅ `trading_system/strategies/strategy_registry.py` - Updated to register bucket strategies

**Key Features Implemented**:

**Bucket A (Safe S&P)**:
- Conservative equity strategy with regime filters
- Eligibility: close > MA200, MA50 > MA200, MA50 slope > 0.3%
- Optional: SPY > MA50 (market regime), news sentiment filter
- Tighter entries: 0.3%/0.8% clearances (vs 0.5%/1.0% standard)
- Conservative exits: MA50 cross, 2.0 ATR stop
- Lower risk: 0.5% per trade, max 6 positions, 60% exposure
- Rationale tags: technical breakouts, regime status, relative strength, news sentiment

**Bucket B (Top-Cap Crypto)**:
- Aggressive crypto strategy with volatility-aware sizing
- Dynamic universe: top 10 by volume (monthly rebalance)
- Volatility adjustment: reduces size for high ATR/close ratio
- Staged exits: MA20 tightens stop to 2.0 ATR, MA50 triggers exit
- Wider stops: 3.5 ATR initial (vs 3.0 standard)
- Rationale tags: technical breakouts, BTC relative strength, volatility context

**Universe Selection**:
- SP500 core universe: 40+ liquid stocks across sectors
- Dynamic crypto selection: top N by volume with optional market cap/liquidity filters
- Functions: `select_equity_universe()`, `select_top_crypto_by_volume()`

**Daily Signal Generation**:
```bash
# Generate signals for both buckets
python scripts/generate_daily_signals.py --date 2024-01-15

# Output: results/daily_signals/daily_signals_YYYYMMDD_HHMMSS.json
```

**Integration Points for Other Agents**:
- Agent 3: Consume JSON output with rationale tags for newsletter
- Agent 4: Use signals with entry/stop prices for paper trading
- Both strategies registered in strategy registry and work with existing framework

**Documentation**: See `AGENT_2_COMPLETION_SUMMARY.md` and `trading_system/strategies/buckets/README.md`

### Agent 3: Newsletter + Native Scheduler (MVP)

**Status**: ✅ All deliverables completed (January 9, 2026)

Decision: build native scheduler first, then evaluate n8n for more robust workflows.

Owns:

- Newsletter generation (HTML + optional text)
- Email delivery plumbing
- Native scheduling (cron-friendly entrypoint)

Primary deliverables:

- ✅ `newsletter` CLI command that renders and sends the daily email
- ✅ A scheduler entrypoint that runs at a fixed time and calls:
  - daily signal generation
  - newsletter rendering + sending

Files/modules implemented:

- ✅ `trading_system/output/email/newsletter_generator.py` - Multi-bucket newsletter context generation (219 lines)
- ✅ `trading_system/output/email/newsletter_service.py` - Newsletter orchestration and delivery (165 lines)
- ✅ `trading_system/output/email/templates/newsletter_daily.html` - HTML email template with bucket sections (253 lines)
- ✅ `trading_system/scheduler/jobs/newsletter_job.py` - Daily newsletter job (371 lines)
- ✅ `trading_system/scheduler/cron_runner.py` - Added newsletter job scheduling (modified)
- ✅ `trading_system/cli.py` - Added `send-newsletter` and `run-newsletter-job` commands (modified)
- ✅ `tests/test_newsletter_generator.py` - Unit tests (201 lines)
- ✅ `trading_system/output/email/README_NEWSLETTER.md` - Comprehensive documentation (430 lines)

**CLI Usage**:
```bash
# Send test newsletter with mock data
python -m trading_system send-newsletter --test

# Run full newsletter job (generate signals + send)
python -m trading_system run-newsletter-job

# Start scheduler daemon (includes newsletter at 5 PM ET)
python -m trading_system run-scheduler
```

**Scheduler Jobs**:
- Daily equity signals: 4:30 PM ET
- Daily crypto signals: 12:00 AM UTC
- **Daily newsletter: 5:00 PM ET** (new)

**Newsletter Features**:
- Multi-bucket signal organization (Safe S&P, Crypto Top-Cap)
- Market overview (SPY/BTC prices, regime detection)
- News digest with sentiment analysis
- Action items ("What to Buy", "What to Avoid")
- Current positions section (ready for Agent 4)
- Professional HTML template with responsive design

**Integration Points for Other Agents**:
- Agent 1: Uses canonical `Signal` contract from `trading_system/models/signals.py`
- Agent 2: Newsletter job calls strategy signal generation for both equity and crypto buckets
- Agent 4: Template includes portfolio summary section, ready for paper trading integration

**Configuration Required**:
```bash
# Environment variables
EMAIL_RECIPIENTS=user@example.com
SMTP_PASSWORD=sendgrid_api_key
ALPHA_VANTAGE_API_KEY=your_key
MASSIVE_API_KEY=your_key
```

**Documentation**: See `AGENT3_IMPLEMENTATION_SUMMARY.md` and `trading_system/output/email/README_NEWSLETTER.md`

### Agent 4: Paper Trading + Manual Trades (Foundation)

Owns:

- Paper trading order execution pipeline (broker adapter + order lifecycle)
- Manual trade entry + storage (so reporting can include manual positions)

Primary deliverables:

- A paper trading runner that consumes `TradePlan` objects
- Manual trade CRUD (CLI is sufficient initially)
- Unified “positions view” that merges system/paper/manual

Files/modules to primarily touch:

- `trading_system/adapters/` (broker adapters)
- `trading_system/live/` or `trading_system/execution/`
- `trading_system/reporting/` (merging sources)

---

## Dependencies / Sequencing

To maximize parallelism, follow this order:

1. Agent 1 defines contracts and stubs a daily signal entrypoint (no strategy changes required).
2. Agent 2 implements Bucket A/B outputs strictly in terms of the contracts.
3. Agent 3 builds newsletter + scheduler using mocked contract objects first, then switches to the real signal entrypoint.
4. Agent 4 builds paper trading + manual trade tracking using mocked contract objects first, then integrates.

---

## Collaboration Rules (Prevent Merge Conflicts)

- **Single owner for contracts**: only Agent 1 changes contract definitions.
- **Folder ownership**:
  - Strategy/data changes stay within `trading_system/strategies/` and related data modules (Agent 2)
  - Newsletter/scheduler changes stay within email/scheduler modules (Agent 3)
  - Paper/manual changes stay within adapters/execution/reporting modules (Agent 4)
- **PR gating**: require passing tests + a minimal integration test before merge.
- **Mock-first development**: newsletter and paper trading should start using mocked `Signal` / `TradePlan` objects so they can be built before strategies are finalized.

---

## Open Questions (Need Decisions)

- Paper trading broker: Alpaca (selected)
- Alpaca setup: create an Alpaca paper account and provide API credentials via environment variables (e.g., `ALPACA_API_KEY`, `ALPACA_API_SECRET`) or a local secrets mechanism.
- Suggested env vars: `ALPACA_API_KEY`, `ALPACA_API_SECRET`, `ALPACA_PAPER=true`, `ALPACA_BASE_URL=https://paper-api.alpaca.markets`.
- Dependency: install `alpaca-trade-api` (required by `trading_system/adapters/alpaca_adapter.py`).
- What is your target account size for sizing recommendations?
- Should newsletter be a single combined email or one section per bucket?
- Do you want the system to produce limit orders or only next-open market orders?
- Do you want discretionary override controls (e.g., “block this symbol”, “cap crypto exposure today”)?

---

## Suggested First Milestone (Concrete)

**Milestone: Daily Equity + Crypto Digest (no trading yet)**

- Generate daily signals for:
  - SP500 universe
  - top N crypto universe
- Produce one email with:
  - top picks + sizes
  - watchlist + blockers
  - rationale (technical + news for equities)

This gets you value immediately while we finalize broker adapters and manual trade tracking.

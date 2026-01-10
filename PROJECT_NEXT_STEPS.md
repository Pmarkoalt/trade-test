# Project Next Steps (Roadmap)

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

## Phase 0 (Lock Interfaces): Shared Data Contracts (High Priority)

Before adding more strategy buckets and live/paper execution, define and enforce a single shared contract so:

- backtests
- optimization
- daily signal generation
- newsletter
- paper trading
- manual trade tracking

all speak the same language.

### Required domain objects

- **Signal**
  - symbol
  - asset_class (`equity` / `crypto`)
  - timestamp (signal time)
  - side (buy/sell)
  - intent (e.g., execute next open)
  - confidence/score
  - rationale tags (technical + news)

- **Allocation / Sizing Recommendation**
  - recommended position size (dollars and/or %)
  - risk budget used
  - max positions constraints applied
  - liquidity/capacity flags

- **TradePlan**
  - entry method (MOO/MKT/limit)
  - stop logic
  - exit logic
  - optional time stop

- **PositionRecord** (for merging system trades + paper trades + manual trades)
  - source (`system` / `paper` / `manual`)
  - open/close timestamps
  - fills
  - PnL/r-multiple
  - notes/tags

Deliverable: A single canonical schema that all layers use.

---

## Phase 1: Multi-Bucket Strategy Expansion

### Bucket A: Safe S&P bets (Equities)

Goal: stable, low drawdown, realistic capacity.

- Universe: SP500 (or SPY + sector ETFs first)
- Regime filters: market trend / risk-off gating
- Signal sources:
  - technical scoring (existing)
  - news sentiment and event-risk (earnings, major headline flags)
- Output: limited number of positions, strict correlation and concentration limits

### Bucket B: Aggressive top-cap crypto (Crypto)

Goal: higher turnover and more aggressive positioning while respecting volatility.

- Universe: dynamic top market cap coins
- Stronger volatility-aware sizing
- Adaptive stop/exit management

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

Owns:

- Shared domain objects: `Signal`, `Allocation`, `TradePlan`, `PositionRecord`
- A single “generate daily signals” entrypoint (CLI-first)
- Integration tests / golden path

Primary deliverables:

- A canonical schema module (single source of truth)
- A stable API between strategy output and downstream consumers (newsletter, paper trading, reporting)
- A “golden path” test that proves: `signals -> newsletter payload -> artifacts`

Files/modules to primarily touch:

- `trading_system/models/` (or a dedicated `trading_system/contracts/` module)
- `trading_system/cli.py` (new command groups if needed)
- `trading_system/integration/`
- `tests/integration/`

### Agent 2: Strategy & Data (Equities + Crypto Buckets A/B)

Owns:

- Bucket A (Safe S&P) strategy logic and its configuration
- Bucket B (Top-cap crypto) strategy logic and its configuration
- Universe selection rules (SP500 + dynamic crypto universe)

Primary deliverables:

- Deterministic daily signal generation for both buckets
- Clear rationale tags for newsletter use (technical reasons, blockers)
- Configs that can be optimized without changing code

Files/modules to primarily touch:

- `trading_system/strategies/`
- `trading_system/data/` (universe selection)
- `EXAMPLE_CONFIGS/` (or `configs/`) for bucket configs

### Agent 3: Newsletter + Native Scheduler (MVP)

Decision: build native scheduler first, then evaluate n8n for more robust workflows.

Owns:

- Newsletter generation (HTML + optional text)
- Email delivery plumbing
- Native scheduling (cron-friendly entrypoint)

Primary deliverables:

- `newsletter` CLI command that renders and sends the daily email
- A scheduler entrypoint that runs at a fixed time and calls:
  - daily signal generation
  - newsletter rendering + sending

Files/modules to primarily touch:

- `trading_system/output/email/` (templates)
- `trading_system/scheduler/` (jobs)
- `trading_system/cli.py`

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

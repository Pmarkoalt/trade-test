# Complete System Documentation for 9+/10 Implementation

---

# STRATEGY_crypto_equities_FINAL.md

## Purpose
Two spot-only, daily-timeframe momentum strategies (equities + crypto) designed to hold positions for days-to-weeks with robust risk-adjusted performance.

**Objective:** Sharpe > 1.0, Max DD < 15%, Calmar > 1.5 on holdout data.

---

## Design Principles
- Daily bars only (no intraday signals)
- Decisions made at day D close
- Orders execute at day D+1 open
- Simple entry logic, disciplined exits
- Minimal parameters, maximum robustness
- Strategy factory architecture (config-driven, not hardcoded)

---

## Universes (MVP, FIXED)

### Equities Universe
**NASDAQ-100 constituents** (preferred) OR **S&P 500 constituents**

Rationale:
- Liquid, tight spreads
- Reduces survivorship bias
- Realistic execution modeling
- No earnings filter in MVP

### Crypto Universe (Fixed List)
**Static 10-asset list to avoid survivorship bias:**

BTC, ETH, BNB, XRP, ADA, SOL, DOT, MATIC, LTC, LINK

Notes:
- Daily candles in UTC
- No dynamic "top N by market cap"
- These assets were consistently large-cap over 2022-2024

---

## Shared Indicators (Daily Timeframe)

Computed per asset:
- **Moving Averages:** MA20, MA50, MA200
- **Volatility:** ATR14
- **Momentum:** ROC60D (60-day rate of change)
- **Breakout Levels:** Highest close over 20D and 55D
- **Volume:** ADV20 (20-day average dollar volume)
- **Relative Strength:** Asset ROC60D vs benchmark (SPY for equities, BTC for crypto)

---

## Equities Strategy (NASDAQ-100 / S&P 500)

### Eligibility Filter (Trend Qualification)

Asset must meet ALL of:

1. **Price above MA50:**
   ```
   close > MA50
   ```

2. **MA50 uptrend (rate of change method):**
   ```
   ma50_roc_20d = (MA50[t] / MA50[t-20]) - 1
   ma50_uptrend = ma50_roc_20d > 0.005  # >0.5% rise over 20 days
   ```
   
3. **Relative strength vs SPY (optional for MVP, recommended for v1.1):**
   ```
   spy_roc60 = (SPY_close / SPY_close_60d_ago) - 1
   stock_roc60 = (close / close_60d_ago) - 1
   relative_strength = stock_roc60 - spy_roc60
   
   # Filter: relative_strength > 0 (outperforming market)
   ```

### Entry Logic

**Two independent breakout triggers (OR logic):**

1. **Fast breakout (20-day):**
   ```
   close >= highest_close_20D * 1.005  # 0.5% clearance above prior high
   ```

2. **Slow breakout (55-day):**
   ```
   close >= highest_close_55D * 1.010  # 1.0% clearance above prior high
   ```

**Clearance rationale:** Filters microbreakouts that immediately fail. Ensures momentum, not just marginal new highs.

**Capacity constraint (hard reject):**
```
order_notional > 0.5% of ADV20 → reject trade
```

**Entry execution:**
- Buy at next session open

### Exit Logic

**Primary trailing exit:**
```
If close < MA20 → exit at next session open
```

**Alternative (test during validation):**
```
If close < MA50 → exit at next session open
```

**Recommendation:** Run both on train/validate. Expect MA50 to have lower win rate but higher Sharpe (fewer false exits).

**Catastrophic hard stop (evaluated daily on close):**
```
stop_price = entry_price - (2.5 × ATR14)

If close < stop_price → exit at next session open
```

**No time-based stops. No fixed profit targets.**

### Position Sizing

- **Base risk per trade:** 0.75% of equity
- **Max positions:** 8
- **Max gross exposure:** 80% of equity
- **Max position notional:** 15% of equity

**Risk calculation:**
```python
stop_distance = entry_price - stop_price
position_size = floor((equity × 0.0075) / stop_distance)

# Clamp to constraints
position_size = min(position_size, equity × 0.15 / entry_price)
```

---

## Crypto Strategy (Fixed 10-Asset Universe)

### Eligibility Filter (Trend Qualification)

**Strict structural trend requirement:**
```
close > MA200
```

No exceptions in MVP. No impulse bypass mode.

**Optional relative strength filter (v1.1):**
```
crypto_roc60 = (close / close_60d_ago) - 1
btc_roc60 = (BTC_close / BTC_close_60d_ago) - 1
relative_strength = crypto_roc60 - btc_roc60

# Filter: relative_strength > 0 (outperforming BTC)
```

### Entry Logic

**Same breakout triggers as equities:**

1. **Fast breakout (20-day):**
   ```
   close >= highest_close_20D * 1.005
   ```

2. **Slow breakout (55-day):**
   ```
   close >= highest_close_55D * 1.010
   ```

**Capacity constraint (stricter than equities):**
```
order_notional > 0.25% of ADV20 → reject trade
```

**Entry execution:**
- Buy at next UTC daily open

### Exit Logic (More Forgiving Than Equities)

**Primary trailing exit (two-stage approach):**

**Recommended approach (staged):**
```
# Stage 1: Warning signal
If close < MA20:
    → Tighten hard stop to 2.0 × ATR14 (from 3.0×)

# Stage 2: Exit signal  
If close < MA50 OR tightened stop hit:
    → Exit at next UTC daily open
```

**Alternative (simpler, test on validation):**
```
If close < MA50 → exit at next UTC daily open
```

**Rationale:** Crypto trends are violent and choppy. MA20 generates too many false exits. MA50 or staged approach provides more breathing room.

**Catastrophic hard stop (evaluated daily on close):**
```
stop_price = entry_price - (3.0 × ATR14)  # Wider than equities

If close < stop_price → exit at next UTC daily open
```

**After MA20 broken, tighten to:**
```
stop_price = entry_price - (2.0 × ATR14)
```

### Position Sizing

- **Base risk per trade:** 0.75% of equity
- **Max positions:** 8
- **Max gross exposure:** 80% of equity  
- **Max position notional:** 15% of equity

Same sizing calculation as equities.

---

## Portfolio-Level Risk Controls

### 1. Volatility Scaling (Continuous, Not Binary)

**Compute portfolio volatility:**
```python
portfolio_returns_20d = daily equity changes over last 20 days
portfolio_vol_20d = std(portfolio_returns_20d) × sqrt(252)  # annualized

# Compute baseline (median over trailing 252 days)
median_vol_252d = median(portfolio_vol_20d over last 252 days)
```

**Scaling rule (continuous):**
```python
vol_ratio = portfolio_vol_20d / median_vol_252d

risk_multiplier = 1.0 / max(vol_ratio, 1.0)  # inverse scaling
risk_multiplier = clip(risk_multiplier, 0.33, 1.0)

# Apply to all new positions
adjusted_risk = 0.0075 × risk_multiplier
```

**Simpler alternative (regime-based, test both):**
```python
if portfolio_vol_20d > 2.0 × median_vol_252d:
    risk_multiplier = 0.50
else:
    risk_multiplier = 1.0
```

**No stop-trading based on weekly P&L.**

### 2. Correlation Guard (Prevent Clustering)

**Compute rolling 20D correlation matrix:**
```python
# For each pair of positions, compute correlation of daily returns over last 20 days
correlation_matrix = compute_correlations(positions, lookback=20)

# Average pairwise correlation (exclude diagonal)
avg_pairwise_corr = mean(off_diagonal(correlation_matrix))
```

**Guard rules (only apply if n_positions >= 4):**

```python
# Rule 1: Portfolio-level guard
if avg_pairwise_corr > 0.70:
    # For each new candidate:
    candidate_corr_to_portfolio = mean(correlations(candidate, existing_positions))
    
    if candidate_corr_to_portfolio > 0.75:
        reject_entry  # or deprioritize in scoring

# Rule 2: Cluster concentration guard (optional, v1.1)
# Group assets by cluster (L1 crypto, mega-cap tech, etc.)
if cluster_exposure > 0.60 × total_exposure:
    reject new entries in that cluster
```

**Correlation floor:** Minimum 4 positions required before applying guard (avoid noise with small portfolios).

---

## Position Queue Logic (When Signals > Slots)

### Scenario
You generate 10 buy signals but only have room for 3 positions.

### Scoring Function (Rank-Based Normalization)

**Components:**

1. **Breakout strength:**
   ```python
   # For 20D breakout
   breakout_strength_20 = (close - MA20) / ATR14
   
   # For 55D breakout
   breakout_strength_55 = (close - MA50) / ATR14
   
   # Use whichever triggered
   ```

2. **Momentum strength (relative):**
   ```python
   # Equities
   spy_roc60 = (SPY_close / SPY_close_60) - 1
   stock_roc60 = (close / close_60) - 1
   relative_strength = stock_roc60 - spy_roc60
   
   # Crypto  
   btc_roc60 = (BTC_close / BTC_close_60) - 1
   crypto_roc60 = (close / close_60) - 1
   relative_strength = crypto_roc60 - btc_roc60
   ```

3. **Diversification bonus:**
   ```python
   if n_existing_positions > 0:
       avg_corr_to_portfolio = mean(correlations(candidate, existing_positions))
       diversification_bonus = 1.0 - avg_corr_to_portfolio
   else:
       diversification_bonus = 0.5  # neutral
   ```

**Normalization (rank-based to avoid scale issues):**

```python
def rank_normalize(values):
    """Convert values to 0-1 ranks"""
    ranks = rankdata(values)
    return ranks / len(values)

# Normalize each component
breakout_rank = rank_normalize([candidate.breakout_strength for candidate in candidates])
momentum_rank = rank_normalize([candidate.relative_strength for candidate in candidates])
diversification_rank = rank_normalize([candidate.diversification_bonus for candidate in candidates])

# Weighted score
for i, candidate in enumerate(candidates):
    candidate.score = (
        0.50 × breakout_rank[i] +
        0.30 × momentum_rank[i] +
        0.20 × diversification_rank[i]
    )
```

**Selection:**
```python
# Sort by score descending
ranked_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)

# Take top N until max_positions or max_exposure constraints hit
selected = []
for candidate in ranked_candidates:
    if len(selected) + n_existing_positions >= max_positions:
        break
    if total_exposure + candidate.notional > max_exposure:
        break
    if candidate violates correlation guard:
        continue
    if candidate violates capacity constraint:
        continue
    
    selected.append(candidate)
```

---

## Data Quality & Error Handling

### Missing Data Policy

**1 day missing:**
```python
# Skip signal updates for that symbol
# Do not update stops
# Log warning
```

**2+ consecutive days missing:**
```python
# Mark symbol as "unhealthy"
# If currently in position:
#     Attempt to exit at next available open
#     Send alert
# If not in position:
#     Exclude from universe until data resumes
```

### Gap Handling (Equities)

**Gap sanity check:**
```python
gap_pct = abs(open[t] - close[t-1]) / close[t-1]

if gap_pct > 0.10:  # >10% gap
    log("GAP_EVENT", symbol=symbol, gap_pct=gap_pct, reason="earnings/news")
    # Still execute (momentum systems must handle gaps)
    # Flag in trade log for post-analysis
```

### Price Validation

**Basic sanity checks:**
```python
# Reject if OHLC relationships violated
if not (low <= open <= high and low <= close <= high):
    mark_data_error(symbol, date)
    
# Reject if price moved >50% in one day (likely data error, not flash crash)
if abs(close / close[-1] - 1) > 0.50:
    mark_data_error(symbol, date)
```

---

## Metrics & Targets

### Primary Metrics (Holdout Evaluation)

**Must pass ALL:**
- **Sharpe ratio > 1.0** (annualized)
- **Max drawdown < 15%**
- **Calmar ratio > 1.5** (annual return / max DD)
- **Minimum 50 total trades** (combined equities + crypto)

### Secondary Metrics (Must Pass 3 of 4)

- **Expectancy > 0.3R** after costs
  ```python
  R_multiple per trade = (exit_price - entry_price) / (entry_price - stop_price)
  expectancy = mean(R_multiples)
  ```
  
- **Profit factor > 1.4**
  ```python
  profit_factor = sum(winning_trades) / abs(sum(losing_trades))
  ```
  
- **Correlation to benchmark < 0.80**
  ```python
  # Equities: correlation(portfolio_returns, SPY_returns)
  # Crypto: correlation(portfolio_returns, BTC_returns)
  ```
  
- **99th percentile daily loss < 5%**
  ```python
  worst_daily_loss = percentile(daily_returns, 1)
  ```

### Tertiary Metrics (Monitoring Only)

- **Recovery factor > 3.0**
  ```python
  recovery_factor = total_net_profit / max_drawdown
  ```
  
- **Drawdown duration**
  ```python
  # Average days from peak to new peak
  # Median and max drawdown duration
  ```
  
- **Turnover (trades per month)**
  - Target: 5-15 trades/month combined (days-to-weeks holding)
  - If >20 trades/month, system has become high-frequency by accident
  
- **Average holding period**
  - Target: 7-21 days
  - Track median and mean
  
- **Max consecutive losing trades**
  - Psychological limit: <10 (most traders quit after this)
  
- **Win rate**
  - Expected: 40-50% for momentum systems
  - Not a target, just diagnostic

### Regime-Conditional Performance (Diagnostic)

Report separately for:

**Equities:**
- SPY uptrend months (SPY > MA200): expect strong profits
- SPY downtrend months (SPY < MA200): expect small losses, no catastrophic DD
- SPY ranging months (ADX < 20): expect breakeven to small loss

**Crypto:**
- BTC uptrend months: expect strong profits
- BTC downtrend months: expect controlled losses
- BTC ranging months: expect breakeven

---

## Execution Cost Model (Conservative & Realistic)

### Fees (Fixed)

**Equities:**
```python
fee_bps = 1  # 1 basis point per side
fee_cost = notional × 0.0001
```

**Crypto:**
```python
fee_bps = 8  # 8 basis points per side
fee_cost = notional × 0.0008
```

### Slippage Model (Dynamic, Vol-Scaled, Size-Penalized)

**Components:**

1. **Base slippage:**
   ```python
   base_slippage_bps = 8   # equities
   base_slippage_bps = 10  # crypto
   ```

2. **Volatility multiplier:**
   ```python
   vol_mult = ATR14 / mean(ATR14 over last 60 days)
   vol_mult = clip(vol_mult, 0.5, 3.0)  # prevent extreme outliers
   ```

3. **Size penalty:**
   ```python
   size_ratio = order_notional / (0.01 × ADV20)  # order as % of 1% ADV
   size_penalty = clip(size_ratio, 0.5, 2.0)
   ```

4. **Session penalty (crypto only):**
   ```python
   if day_of_week in [Saturday, Sunday]:  # UTC
       weekend_penalty = 1.5
   else:
       weekend_penalty = 1.0
   ```

5. **Stress multiplier:**
   ```python
   # Measure on prior week's return
   if last_week_spy_return < -0.03:  # equities
       stress_mult = 2.0
   elif VIX > 30:  # real-time stress (optional, v1.1)
       stress_mult = 2.0
   else:
       stress_mult = 1.0
       
   # Similar for crypto with BTC returns < -5%
   ```

**Final slippage calculation:**

```python
# Equities
slippage_mean_bps = (
    base_slippage_bps × 
    vol_mult × 
    size_penalty × 
    stress_mult
)

# Crypto
slippage_mean_bps = (
    base_slippage_bps × 
    vol_mult × 
    size_penalty × 
    weekend_penalty ×
    stress_mult
)

# Add variance (normal distribution)
slippage_std_bps = slippage_mean_bps × 0.75
slippage_actual_bps = sample_normal(slippage_mean_bps, slippage_std_bps)
slippage_actual_bps = max(slippage_actual_bps, 0)  # no negative slippage

# Apply to fill price
if side == BUY:
    fill_price = open_price × (1 + slippage_actual_bps / 10000)
else:  # SELL
    fill_price = open_price × (1 - slippage_actual_bps / 10000)
```

### Slippage Clustering (Realistic Stress Behavior)

**Key insight:** Bad fills cluster during stress periods.

**Implementation:**
```python
# During stress weeks, sample from worse distribution
if stress_mult == 2.0:
    # Sample from fatter tail distribution
    slippage_std_bps = slippage_mean_bps × 1.5  # higher variance
```

### Execution Quality Tracking (Live Trading)

Track per trade:
```python
model_slippage_bps = computed from formula above
actual_slippage_bps = actual_fill - expected_fill (basis points)
slippage_error = actual_slippage_bps - model_slippage_bps

# Alert if slippage_error consistently > 5 bps (model is too optimistic)
```

---

## Out of Scope (Explicitly Deferred)

**MVP excludes:**
- Low-float equities
- Earnings calendar filters
- Time-based exits (max hold periods)
- Intraday execution logic
- Leverage or derivatives
- Dynamic crypto universe (top-N by market cap)
- Sector rotation (equities)
- News/sentiment analysis
- Machine learning signal generation

**Post-MVP (v1.1+):**
- Relative strength filters (vs SPY/BTC)
- Earnings proximity filter (0-1 days before)
- Sector momentum bias (equities)
- VIX-based regime detection
- Cluster definitions (L1 crypto, mega-cap tech, etc.)

---

## Expected Behavior (Realistic Expectations)

### What This Strategy Does Well
- Participates in sustained trends (weeks to months)
- Limits losses via disciplined stops
- Diversifies across uncorrelated opportunities
- Survives drawdowns without catastrophic loss

### What This Strategy Does Poorly
- Misses early trend beginnings (enters late)
- Exits before trend tops (leaves money on table)
- Underperforms in ranging/choppy markets
- Generates 45-55% win rate (many small losses, fewer large wins)

### Psychological Challenges
- **Expect 3-5 consecutive losses** (normal for momentum)
- **Expect to exit just before continuation** (happens frequently)
- **Drawdowns of 10-12% are normal** (within tolerance)
- **Most trades will be scratches** (small wins, small losses)

**This is working as designed.** Profitability comes from asymmetric risk/reward, not high win rate.

---

# SIMULATION_walkforward_backtest_FINAL.md

## Purpose
Rigorous walk-forward simulation with explicit train/validate/holdout protocol, realistic execution costs, and comprehensive robustness testing.

**Goal:** Validate strategy logic before committing to paper/live trading.

---

## Data Requirements

### Minimum Dataset
- **Duration:** 24 months (18 months minimum)
- **Assets:**
  - Equities: NASDAQ-100 or S&P 500 constituents
  - Crypto: BTC, ETH, BNB, XRP, ADA, SOL, DOT, MATIC, LTC, LINK
  - Benchmark: SPY (equities), BTC (crypto benchmark for correlation)

### Required Fields (Per Asset, Per Day)
- Date (timestamp)
- Open, High, Low, Close (OHLC)
- Volume
- Dollar volume (price × volume)

### Data Quality Requirements
- No gaps >1 day (crypto: 365 days/year; equities: market calendar)
- OHLC relationships valid (low ≤ open, close ≤ high)
- No single-day moves >50% (likely data errors)
- Minimum lookback: 250 days before test start (for MA200, baseline vol, etc.)

---

## Protocol (Pre-Registered, LOCKED Before Any Testing)

### Data Splits

**24-month dataset (preferred):**
```
Train/Sensitivity: Months 1-15 (parameter sensitivity only, NO optimization)
Validation: Months 16-18 (light tuning if needed)
Holdout: Months 19-24 (SACRED, touched ONCE at end)
```

**18-month dataset (minimum acceptable):**
```
Train/Sensitivity: Months 1-12
Validation: Months 13-15
Holdout: Months 16-18
```

### Pre-Registration Document

**Create and commit BEFORE any backtest run:**

```markdown
# PREREGISTRATION.md

Date: [TODAY]
Strategies: Equities Momentum + Crypto Momentum
Dataset: [2023-01-01 to 2024-12-31] or specify your period

## Data Splits
- Train: [start] to [end]
- Validation: [start] to [end]  
- Holdout: [start] to [end] **LOCKED**

## Declared Parameters (Frozen)
- Risk per trade: 0.75%
- Max positions: 8 per strategy
- Max exposure: 80%
- Execution: Close signal → next open fill

Equities:
- Universe: NASDAQ-100 (or S&P 500)
- ATR stop: 2.5x
- Exit: MA20 cross (or MA50, to be tested)
- Trend filter: MA50 with RoC >0.5% over 20D
- Breakout: 20D (0.5% clearance) OR 55D (1.0% clearance)

Crypto:
- Universe: BTC, ETH, BNB, XRP, ADA, SOL, DOT, MATIC, LTC, LINK
- ATR stop: 3.0x (tighten to 2.0x after MA20 break)
- Exit: MA50 cross (or staged MA20→MA50, to be tested)
- Trend filter: Close > MA200
- Breakout: 20D (0.5% clearance) OR 55D (1.0% clearance)

Slippage:
- Model: vol-scaled, size-penalized, stress-clustered
- Base: 8 bps (equities), 10 bps (crypto)
- Stress: 2x during SPY <-3% weeks or BTC <-5% weeks

## Success Criteria (Measured on HOLDOUT ONLY)

Primary (must pass ALL 4):
1. Sharpe ratio > 1.0
2. Max drawdown < 15%
3. Calmar ratio > 1.5
4. Minimum 50 total trades (combined)

Secondary (must pass 3 of 4):
1. Expectancy > 0.3R after costs
2. Profit factor > 1.4
3. Correlation to benchmark < 0.80
4. 99th percentile daily loss < 5%

Stress tests (must pass 2 of 3):
1. 2x slippage: Sharpe > 0.75
2. 3x slippage: Calmar > 1.0
3. Bear market months only: Expectancy > 0 (or DD < 20%)

## Rejection Criteria (ANY on holdout = FAIL)
- Max DD > 20%
- Sharpe < 0.75
- Calmar < 1.0
- Profits depend on <3 trades
- Bootstrap 5th percentile Sharpe < 0.4

## Tuning Allowed (Train + Validation ONLY)
- ATR stop multiplier (from grid: 2.0/2.5/3.0/3.5)
- Breakout clearance threshold (from grid: 0%/0.5%/1.0%/1.5%)
- Exit logic (MA20 vs MA50 vs staged)
- Volatility scaling (continuous vs regime-based vs off)
- Relative strength filter (enable/disable)

## Never Tuned (Structural Constraints)
- Risk per trade (0.75%)
- Max positions (8)
- Execution model (close→open)
- Universe definitions
- Capacity constraints (ADV-based)

Signed: [Your Name]
Date: [Today]
```

**Once signed, holdout period is LOCKED. Never look at holdout data until final evaluation.**

---

## Walk-Forward Engine (Event-Driven, No Lookahead)

### Daily Event Loop

For each calendar day `t` from start to end:

```python
# 1. Update data up to close[t]
update_data_through_date(t)

# 2. Compute indicators using data ≤ t
features[t] = compute_features(data[start:t+1])

# 3. Generate signals at close[t]
signals[t] = generate_signals(features[t], portfolio_state[t], date=t)

# 4. Create orders for execution at open[t+1]
orders[t+1] = create_orders(signals[t], portfolio_state[t])

# 5. Execute fills at open[t+1] with slippage/fees
fills[t+1] = execute_orders(orders[t+1], market_data[t+1], slippage_model)

# 6. Update portfolio state
portfolio_state[t+1] = update_portfolio(portfolio_state[t], fills[t+1])

# 7. Update stops (trailing) based on close[t+1]
stop_orders[t+2] = update_stops(portfolio_state[t+1], market_data[t+1])

# 8. Log metrics
log_daily_metrics(t+1, portfolio_state[t+1])

# 9. Advance to next day
t = t + 1
```

**Calendar handling:**
- Equities: Use market calendar (skip weekends/holidays)
- Crypto: Run 365 days/year (UTC daily candles)

---

## Execution Model (Realistic, Conservative)

### Fill Price Simulation

**Default fill:** Next-day open price

**Slippage application:**
```python
# Entry (buy)
slippage_bps = compute_slippage(order, market_state)
fill_price = open_price × (1 + slippage_bps / 10000)

# Exit (sell)
fill_price = open_price × (1 - slippage_bps / 10000)

# Fees
fee_cost = fill_price × quantity × (fee_bps / 10000)
```

### Capacity Constraints (Hard Rejections)

**Equities:**
```python
if order_notional > 0.005 × ADV20:  # 0.5% of 20D avg dollar volume
    reject_order("CAPACITY_LIMIT_EQUITIES")
    log_rejection(symbol, reason="exceeds 0.5% ADV")
```

**Crypto:**
```python
if order_notional > 0.0025 × ADV20:  # 0.25% of 20D avg dollar volume
    reject_order("CAPACITY_LIMIT_CRYPTO")
    log_rejection(symbol, reason="exceeds 0.25% ADV")
```

### Slippage Calculation (Full Model)

```python
def compute_slippage_bps(order, market_state, stress_state):
    """
    Compute slippage in basis points for a given order.
    
    Args:
        order: Order object with symbol, side, notional
        market_state: Current ATR, ADV, day_of_week
        stress_state: Recent market returns, VIX (optional)
    
    Returns:
        slippage_bps: Float, basis points of slippage
    """
    # 1. Base slippage
    if order.asset_class == 'equity':
        base = 8
    elif order.asset_class == 'crypto':
        base = 10
    
    # 2. Volatility multiplier
    atr_60d_avg = mean(market_state.atr14[-60:])
    vol_mult = market_state.atr14 / atr_60d_avg
    vol_mult = clip(vol_mult, 0.5, 3.0)
    
    # 3. Size penalty
    size_ratio = order.notional / (0.01 × market_state.adv20)
    size_penalty = clip(size_ratio, 0.5, 2.0)
    
    # 4. Session penalty (crypto weekends)
    if order.asset_class == 'crypto' and market_state.day_of_week in [5, 6]:  # Sat, Sun
        session_penalty = 1.5
    else:
        session_penalty = 1.0
    
    # 5. Stress multiplier
    stress_mult = 1.0
    
    if order.asset_class == 'equity':
        if stress_state.spy_weekly_return < -0.03:
            stress_mult = 2.0
        elif stress_state.vix > 30:  # optional, requires VIX data
            stress_mult = 2.0
            
    elif order.asset_class == 'crypto':
        if stress_state.btc_weekly_return < -0.05:
            stress_mult = 2.0
    
    # 6. Compute mean
    slippage_mean = base × vol_mult × size_penalty × session_penalty × stress_mult
    
    # 7. Add variance (normal distribution)
    slippage_std = slippage_mean × 0.75
    slippage_draw = sample_normal(slippage_mean, slippage_std)
    slippage_bps = max(slippage_draw, 0)  # no negative slippage
    
    return slippage_bps
```

### Fee Model

```python
def compute_fees(fill_price, quantity, asset_class):
    """Compute transaction fees."""
    notional = fill_price × quantity
    
    if asset_class == 'equity':
        fee_bps = 1
    elif asset_class == 'crypto':
        fee_bps = 8
    
    fee_cost = notional × (fee_bps / 10000)
    return fee_cost
```

### Total Cost Per Trade

```python
# Entry
entry_slippage_cost = entry_price × quantity × (slippage_bps / 10000)
entry_fee = entry_price × quantity × (fee_bps / 10000)
entry_total_cost = entry_slippage_cost + entry_fee

# Exit
exit_slippage_cost = exit_price × quantity × (slippage_bps / 10000)
exit_fee = exit_price × quantity × (fee_bps / 10000)
exit_total_cost = exit_slippage_cost + exit_fee

# Round-trip cost
round_trip_cost = entry_total_cost + exit_total_cost
round_trip_bps = (round_trip_cost / entry_notional) × 10000
```

---

## Validation Tests (Required Before Holdout)

### 1. Parameter Sensitivity Analysis (Train + Validation ONLY)

**Test grid (coarse):**

```python
# Equities
equity_params = {
    'atr_stop_mult': [2.0, 2.5, 3.0, 3.5],
    'breakout_clearance': [0.000, 0.005, 0.010, 0.015],  # 0%, 0.5%, 1.0%, 1.5%
    'exit_ma': ['MA20', 'MA50', 'staged'],
    'trend_filter': ['MA50_roc', 'MA200', 'none'],  # optional
}

# Crypto
crypto_params = {
    'atr_stop_mult': [2.5, 3.0, 3.5, 4.0],
    'breakout_clearance': [0.000, 0.005, 0.010, 0.015],
    'exit_ma': ['MA20', 'MA50', 'staged'],
}

# Shared
shared_params = {
    'vol_scaling': ['continuous', 'regime_based', 'off'],
    'relative_strength_filter': [True, False],  # optional
}
```

**Run combinations and evaluate:**
```python
for equity_combo in product(*equity_params.values()):
    for crypto_combo in product(*crypto_params.values()):
        results = run_backtest(
            equity_params=equity_combo,
            crypto_params=crypto_combo,
            period='train_validate'
        )
        store_results(results)

# Analyze
plot_heatmaps(param='atr_stop_mult', metric='sharpe')
plot_heatmaps(param='breakout_clearance', metric='calmar')

# Check for sharp peaks (overfitting indicator)
if max(sharpe_values) - median(sharpe_values) > 0.5:
    flag_overfitting_risk()
```

**Selection criteria:**
- Choose parameters where performance is **stable across a neighborhood**
- Avoid sharp peaks (fragile)
- Prefer simpler models if performance is similar

**Document final choices:**
```
Final parameters (from sensitivity analysis):
- Equities ATR stop: 2.5x (stable from 2.0-3.0)
- Crypto ATR stop: 3.0x (stable from 2.5-3.5)
- Breakout clearance: 0.5% (improves quality vs 0%, no benefit beyond 1.0%)
- Exit logic: MA20 for equities, staged for crypto
- Vol scaling: continuous (marginal improvement over regime-based)
```

### 2. Slippage Stress Tests (Train + Validation)

**Run entire backtest with slippage multipliers:**

```python
scenarios = {
    'baseline': 1.0,
    '2x_slippage': 2.0,
    '3x_slippage': 3.0,
}

for scenario_name, multiplier in scenarios.items():
    results = run_backtest(
        period='train_validate',
        slippage_multiplier=multiplier
    )
    
    print(f"{scenario_name}:")
    print(f"  Sharpe: {results.sharpe:.2f}")
    print(f"  Calmar: {results.calmar:.2f}")
    print(f"  Max DD: {results.max_dd:.1%}")
    print(f"  Expectancy: {results.expectancy:.2f}R")
```

**Acceptance criteria:**
- 2x slippage: Sharpe should remain >0.75
- 3x slippage: Calmar should remain >1.0
- If 3x slippage flips expectancy negative, edge is too thin

### 3. Bootstrap Resampling (Trade Returns)

**Test statistical robustness:**

```python
def bootstrap_analysis(trade_returns, n_iterations=1000):
    """
    Bootstrap trade returns to build confidence intervals.
    
    Args:
        trade_returns: List of R-multiples per trade
        n_iterations: Number of bootstrap samples
    
    Returns:
        Dictionary with percentile results
    """
    sharpe_samples = []
    max_dd_samples = []
    calmar_samples = []
    
    for i in range(n_iterations):
        # Resample with replacement
        sample = resample(trade_returns, n=len(trade_returns), replace=True)
        
        # Compute metrics
        sharpe = compute_sharpe(sample)
        max_dd = compute_max_drawdown(sample)
        calmar = compute_calmar(sample)
        
        sharpe_samples.append(sharpe)
        max_dd_samples.append(max_dd)
        calmar_samples.append(calmar)
    
    results = {
        'sharpe_5th': percentile(sharpe_samples, 5),
        'sharpe_50th': percentile(sharpe_samples, 50),
        'sharpe_95th': percentile(sharpe_samples, 95),
        'max_dd_95th': percentile(max_dd_samples, 95),
        'calmar_5th': percentile(calmar_samples, 5),
    }
    
    return results

# Run on train + validation
bootstrap_results = bootstrap_analysis(all_trade_returns)

# Check fragility
if bootstrap_results['sharpe_5th'] < 0.5:
    flag_warning("Strategy is statistically fragile")
    
if bootstrap_results['max_dd_95th'] > 0.25:
    flag_warning("Tail risk: 95th percentile DD > 25%")
```

### 4. Permutation Test (Entry Timing)

**Test if entry timing matters vs random:**

```python
def permutation_test(strategy, data, n_iterations=1000):
    """
    Randomize entry dates while preserving exit logic.
    
    Tests if strategy performance is due to entry skill or luck.
    """
    # Get actual results
    actual_results = run_backtest(strategy, data)
    actual_sharpe = actual_results.sharpe
    
    # Run randomized versions
    random_sharpes = []
    
    for i in range(n_iterations):
        # Randomize entry dates (preserve distribution of holding periods)
        randomized_strategy = randomize_entries(strategy, data)
        random_results = run_backtest(randomized_strategy, data)
        random_sharpes.append(random_results.sharpe)
    
    # Compute percentile rank
    percentile_rank = percentileofscore(random_sharpes, actual_sharpe)
    
    print(f"Actual Sharpe: {actual_sharpe:.2f}")
    print(f"Percentile rank: {percentile_rank:.1f}%")
    print(f"95th percentile random: {percentile(random_sharpes, 95):.2f}")
    
    if percentile_rank < 95:
        flag_warning("Strategy not statistically significant vs random")
    
    return percentile_rank

# Run on train + validation
permutation_results = permutation_test(strategy, train_validate_data)
```

**Acceptance:** Actual Sharpe must be >95th percentile of randomized.

### 5. Correlation Stress Diagnostics

**Check if diversification fails during drawdowns:**

```python
def correlation_stress_analysis(portfolio_history):
    """
    Analyze correlation behavior during drawdowns.
    """
    # Compute rolling 20D correlations
    for date in portfolio_history:
        positions = portfolio_history[date].positions
        if len(positions) >= 2:
            corr_matrix = compute_correlation_matrix(positions, lookback=20)
            avg_pairwise_corr = mean(off_diagonal(corr_matrix))
            
            # Track correlation vs drawdown
            dd_from_peak = compute_dd_from_peak(date)
            
            log_correlation(date, avg_pairwise_corr, dd_from_peak)
    
    # Analyze
    during_drawdowns = filter(lambda x: x.dd_from_peak < -0.05, logs)
    avg_corr_during_dd = mean([x.avg_pairwise_corr for x in during_drawdowns])
    
    during_normal = filter(lambda x: x.dd_from_peak >= -0.05, logs)
    avg_corr_normal = mean([x.avg_pairwise_corr for x in during_normal])
    
    print(f"Avg correlation during normal: {avg_corr_normal:.2f}")
    print(f"Avg correlation during DD: {avg_corr_during_dd:.2f}")
    
    if avg_corr_during_dd > 0.70:
        flag_warning("Diversification fails during stress")
```

### 6. Adverse Scenario Testing

**Test performance in specific market regimes:**

#### A. Bear Market Test

```python
# Isolate months where SPY < MA200 (equities) or BTC < MA200 (crypto)
bear_months = filter(lambda m: benchmark_below_ma200(m), all_months)

bear_results = run_backtest(strategy, months=bear_months)

print(f"Bear market results:")
print(f"  Expectancy: {bear_results.expectancy:.2f}R")
print(f"  Max DD: {bear_results.max_dd:.1%}")
print(f"  Win rate: {bear_results.win_rate:.1%}")

# Acceptance: Expectancy > 0 or Max DD < 20%
```

#### B. Range-Bound Market Test

```python
# Isolate months where benchmark range < 10% (high-to-low)
range_months = filter(lambda m: monthly_range(m) < 0.10, all_months)

range_results = run_backtest(strategy, months=range_months)

# Expect breakeven to small loss (momentum struggles in ranges)
print(f"Range market results:")
print(f"  Return: {range_results.total_return:.1%}")
print(f"  Max DD: {range_results.max_dd:.1%}")

# Acceptance: Max DD < 15%, return > -5%
```

#### C. Flash Crash Simulation

```python
# Simulate extreme stress: one random day per quarter with:
# - Slippage × 5
# - All stops hit at worst possible price

for quarter in quarters:
    crash_day = random.choice(quarter.days)
    
    simulate_crash(
        date=crash_day,
        slippage_mult=5.0,
        all_stops_hit=True
    )

crash_results = run_backtest_with_crashes(strategy, data)

print(f"Flash crash scenario:")
print(f"  Max DD: {crash_results.max_dd:.1%}")
print(f"  Recovery time: {crash_results.recovery_days} days")

# Acceptance: Max DD < 25%, portfolio survives
```

---

## Outputs & Artifacts

### Daily Equity Curve (CSV)

```csv
date,equity,cash,n_positions_equity,n_positions_crypto,exposure_pct,portfolio_vol_20d,avg_pairwise_corr,dd_from_peak
2023-01-03,100000.00,20000.00,5,3,80.0,15.2,0.35,0.0
2023-01-04,101250.50,18500.00,6,3,82.5,15.4,0.38,-0.0
...
```

### Trade Log (CSV)

```csv
symbol,asset_class,entry_date,entry_price,entry_notional,exit_date,exit_price,hold_days,pnl,pnl_pct,R_multiple,exit_reason,entry_slippage_bps,exit_slippage_bps,entry_fee,exit_fee,adv20_at_entry,vol_mult,size_penalty
AAPL,equity,2023-01-05,150.25,12000,2023-01-18,157.80,13,750.50,6.25,2.1,MA20_cross,7.2,8.5,1.20,1.26,500000000,1.1,0.8
BTC,crypto,2023-01-10,42500,15000,2023-01-25,48200,15,2125,14.2,3.8,MA50_cross,9.8,12.1,120,144,2000000000,1.3,0.6
...
```

**Fields:**
- R_multiple: (exit - entry) / (entry - stop)
- exit_reason: MA20_cross, MA50_cross, hard_stop, capacity_reject, data_missing
- vol_mult, size_penalty: slippage model components (for diagnostics)

### Weekly Summary (CSV)

```csv
week_start,weekly_return_pct,weekly_vol_pct,sharpe_ytd,calmar_ytd,dd_from_peak,n_trades_closed,avg_hold_days,turnover,corr_to_spy,corr_to_btc,exposure_avg
2023-01-02,2.5,12.0,1.25,1.80,-0.0,3,12.5,0.15,0.42,0.35,78.5
2023-01-09,-1.2,14.5,1.18,1.72,-1.2,2,10.0,0.12,0.45,0.38,81.0
...
```

### Monthly Report (JSON)

```json
{
  "month": "2023-01",
  "total_return_pct": 5.2,
  "sharpe_12m": 1.15,
  "calmar_12m": 1.68,
  "max_dd_month_pct": -2.5,
  "win_rate": 0.48,
  "profit_factor": 1.75,
  "expectancy_R": 0.42,
  "avg_correlation": 0.38,
  "trades_count": 18,
  "largest_win_pct": 12.5,
  "largest_loss_pct": -4.2,
  "avg_hold_days": 11.5,
  "turnover": 0.68,
  "exposure_avg_pct": 79.5
}
```

### Scenario Comparison (JSON)

```json
{
  "baseline": {
    "sharpe": 1.22,
    "calmar": 1.85,
    "max_dd": 11.5,
    "expectancy": 0.45
  },
  "2x_slippage": {
    "sharpe": 0.95,
    "calmar": 1.42,
    "max_dd": 12.8,
    "expectancy": 0.32
  },
  "3x_slippage": {
    "sharpe": 0.78,
    "calmar": 1.15,
    "max_dd": 14.2,
    "expectancy": 0.22
  },
  "bear_months": {
    "expectancy": 0.05,
    "max_dd": 18.5,
    "win_rate": 0.35
  }
}
```

---

## Rejection Conditions (Hard Failures)

**Strategy is REJECTED if ANY of these on holdout:**

1. **Max drawdown > 20%**
   - Risk of ruin too high
   
2. **Sharpe ratio < 0.75**
   - Insufficient risk-adjusted returns
   
3. **Calmar ratio < 1.0**
   - Returns don't justify drawdown
   
4. **Profits depend on <3 trades**
   - Fragile, not systematic
   - Test: Remove top 3 trades, is expectancy still positive?
   
5. **2x slippage flips expectancy negative**
   - Edge is too thin for real-world execution
   
6. **Bootstrap 5th percentile Sharpe < 0.4**
   - Statistically fragile
   
7. **Permutation test fails (actual < 95th percentile random)**
   - Entry timing is not adding value

**If ANY rejection condition triggers, DO NOT TUNE ON HOLDOUT.**

Go back to strategy design or accept failure.

---

## Post-Holdout: Decision Tree

```
Holdout Results Ready
    |
    ├─> ALL primary criteria passed?
    |       |
    |       ├─> YES: Check secondary criteria
    |       |       |
    |       |       ├─> 3+ of 4 passed?
    |       |       |       |
    |       |       |       ├─> YES: Check stress tests
    |       |       |       |       |
    |       |       |       |       ├─> 2+ of 3 passed?
    |       |       |       |       |       |
    |       |       |       |       |       ├─> YES: PROCEED TO PAPER TRADING
    |       |       |       |       |       |
    |       |       |       |       |       └─> NO: CONDITIONAL PASS (reduce size, monitor)
    |       |       |       |       |
    |       |       |       |       └─> Stress tests failed
    |       |       |       |               └─> REJECT or reduce to $10k test
    |       |       |       |
    |       |       |       └─> NO: Secondary criteria failed
    |       |       |               └─> CONDITIONAL PASS or REJECT
    |       |       |
    |       |       └─> Secondary results
    |       |
    |       └─> NO: Primary criteria failed
    |               └─> REJECT (do not tune on holdout)
    |
    └─> Rejection condition triggered?
            └─> YES: HARD REJECT
```

---

## Recommended Timeline

**Week 1-2:** Data acquisition & validation
- Acquire OHLCV for 24 months
- Verify data quality
- Build indicator pipeline

**Week 3-4:** Strategy implementation
- Implement strategy factory
- Build portfolio state machine
- Unit test signal generation

**Week 5-6:** Execution model
- Implement slippage model
- Add capacity constraints
- Test on toy data

**Week 7:** Parameter sensitivity (Train + Validation)
- Run grid search
- Plot heatmaps
- Select robust parameters

**Week 8:** Validation suite (Train + Validation)
- Bootstrap test
- Permutation test
- Slippage stress tests
- Scenario tests

**Week 9:** Holdout evaluation (ONCE)
- Run with final parameters
- Compute all metrics
- Make go/no-go decision

**Total: 9 weeks to validated strategy**

If holdout passes → paper trade for 12 weeks
If holdout fails → redesign or abandon

---

# PARAMS_defaults_FINAL.md

## Frozen Parameters (DO NOT TUNE)

**Structural constraints that define the strategy:**

```python
# Portfolio
STARTING_EQUITY = 100_000  # USD
RISK_PER_TRADE_BASE = 0.0075  # 0.75% of equity
MAX_POSITIONS_PER_STRATEGY = 8
MAX_GROSS_EXPOSURE = 0.80  # 80% of equity
MAX_POSITION_NOTIONAL = 0.15  # 15% of equity per position

# Execution
SIGNAL_TIMING = "close"  # signals generated at day close
EXECUTION_TIMING = "next_open"  # orders filled at next day open

# Universe
EQUITY_UNIVERSE = "NASDAQ-100"  # or "SP500"
CRYPTO_UNIVERSE = ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOT", "MATIC", "LTC", "LINK"]

# Capacity (hard rejects)
EQUITY_MAX_ORDER_PCT_OF_ADV = 0.005  # 0.5% of 20D avg dollar volume
CRYPTO_MAX_ORDER_PCT_OF_ADV = 0.0025  # 0.25% of 20D avg dollar volume
```

**These define the strategy identity. Changing them = different strategy.**

---

## Common Indicators (Both Strategies)

```python
# Moving averages
MA_PERIODS = [20, 50, 200]

# Volatility
ATR_PERIOD = 14

# Momentum
ROC_PERIOD = 60  # 60-day rate of change

# Breakout windows
BREAKOUT_FAST = 20  # days
BREAKOUT_SLOW = 55  # days

# Volume
VOLUME_LOOKBACK = 20  # days for average

# Correlation
CORRELATION_LOOKBACK = 20  # days for rolling correlation
```

---

## Equities Parameters (NASDAQ-100 / S&P 500)

### Eligibility Filter

```python
# Trend qualification
EQUITY_TREND_MA = 50  # must be above MA50

# MA slope (rate of change method)
EQUITY_MA_SLOPE_LOOKBACK = 20  # days
EQUITY_MA_SLOPE_MIN = 0.005  # >0.5% rise over 20D

# Relative strength filter (optional, test on/off)
EQUITY_RELATIVE_STRENGTH_ENABLED = False  # MVP: off, v1.1: on
EQUITY_RELATIVE_STRENGTH_MIN = 0.0  # must outperform SPY
```

### Entry

```python
# Breakout clearance thresholds
EQUITY_BREAKOUT_20D_CLEARANCE = 0.005  # 0.5% above prior 20D high
EQUITY_BREAKOUT_55D_CLEARANCE = 0.010  # 1.0% above prior 55D high

# Trigger: close >= highest_close_N * (1 + clearance)
```

**Tunable during sensitivity:** clearance from [0%, 0.5%, 1.0%, 1.5%]

### Exit

```python
# Trailing exit (test both)
EQUITY_EXIT_MA = 20  # MA20 or MA50 (run both, compare)

# Exit if: close < MA_N → sell next open

# Hard stop
EQUITY_STOP_ATR_MULT = 2.5  # stop = entry - 2.5 × ATR14
```

**Tunable during sensitivity:** ATR mult from [2.0, 2.5, 3.0, 3.5]

### Costs

```python
# Fees
EQUITY_FEE_BPS = 1  # 1 basis point per side

# Slippage (base)
EQUITY_SLIPPAGE_BASE_BPS = 8

# Slippage variance
EQUITY_SLIPPAGE_STD_MULT = 0.75  # std = mean × 0.75

# Stress multiplier
EQUITY_STRESS_THRESHOLD = -0.03  # SPY weekly return < -3%
EQUITY_STRESS_SLIPPAGE_MULT = 2.0  # double slippage during stress weeks
```

---

## Crypto Parameters (Fixed 10-Asset Universe)

### Eligibility Filter

```python
# Trend qualification (strict)
CRYPTO_TREND_MA = 200  # must be above MA200

# No impulse bypass in MVP

# Relative strength filter (optional)
CRYPTO_RELATIVE_STRENGTH_ENABLED = False  # v1.1: test on/off
CRYPTO_RELATIVE_STRENGTH_MIN = 0.0  # must outperform BTC
```

### Entry

```python
# Breakout clearance (same as equities)
CRYPTO_BREAKOUT_20D_CLEARANCE = 0.005  # 0.5%
CRYPTO_BREAKOUT_55D_CLEARANCE = 0.010  # 1.0%
```

### Exit

```python
# Trailing exit (test three modes)
# Mode 1: MA50 only
# Mode 2: Staged (MA20 warning → tighten stop → MA50 exit)
# Mode 3: MA20 AND MA50 (conservative)

CRYPTO_EXIT_MODE = "staged"  # MVP default after testing

# Staged exit logic
# Stage 1: close < MA20 → tighten stop to 2.0 × ATR
# Stage 2: close < MA50 OR stop hit → exit

# Hard stop
CRYPTO_STOP_ATR_MULT = 3.0  # wider than equities (crypto is choppier)
CRYPTO_STOP_ATR_MULT_TIGHTENED = 2.0  # after MA20 break
```

**Tunable during sensitivity:** ATR mult from [2.5, 3.0, 3.5, 4.0]

### Costs

```python
# Fees
CRYPTO_FEE_BPS = 8  # 8 basis points per side

# Slippage (base)
CRYPTO_SLIPPAGE_BASE_BPS = 10

# Weekend penalty
CRYPTO_WEEKEND_PENALTY = 1.5  # Sat/Sun UTC

# Slippage variance
CRYPTO_SLIPPAGE_STD_MULT = 0.75

# Stress multiplier
CRYPTO_STRESS_THRESHOLD = -0.05  # BTC weekly return < -5%
CRYPTO_STRESS_SLIPPAGE_MULT = 2.0
```

---

## Volatility Scaling (Portfolio-Level)

### Continuous Scaling (Recommended)

```python
# Compute portfolio vol (20D rolling)
PORTFOLIO_VOL_LOOKBACK = 20  # days

# Baseline (median over trailing year)
PORTFOLIO_VOL_BASELINE_LOOKBACK = 252  # days

# Scaling formula
vol_ratio = portfolio_vol_20d / median_vol_252d
risk_multiplier = 1.0 / max(vol_ratio, 1.0)  # inverse scaling
risk_multiplier = clip(risk_multiplier, 0.33, 1.0)

# Apply to new positions
adjusted_risk = RISK_PER_TRADE_BASE × risk_multiplier
```

### Regime-Based Alternative (Simpler, Test Both)

```python
REGIME_VOL_THRESHOLD = 2.0  # 2x median vol

if portfolio_vol_20d > REGIME_VOL_THRESHOLD × median_vol_252d:
    risk_multiplier = 0.50
else:
    risk_multiplier = 1.0
```

**Tunable during sensitivity:** test continuous vs regime vs off

---

## Correlation Guard

```python
# Portfolio-level
CORRELATION_GUARD_ENABLED = True
CORRELATION_GUARD_MIN_POSITIONS = 4  # don't apply if <4 positions
CORRELATION_GUARD_AVG_THRESHOLD = 0.70  # avg pairwise correlation

# Individual candidate
CORRELATION_GUARD_CANDIDATE_THRESHOLD = 0.75  # correlation to portfolio

# Logic
if n_positions >= 4 and avg_pairwise_corr > 0.70:
    for candidate in new_signals:
        if correlation(candidate, portfolio) > 0.75:
            reject_or_deprioritize(candidate)
```

---

## Scoring Function (Position Queue)

```python
# Weights (normalized via ranking)
SCORE_WEIGHT_BREAKOUT = 0.50
SCORE_WEIGHT_MOMENTUM = 0.30
SCORE_WEIGHT_DIVERSIFICATION = 0.20

# Computation (rank-based normalization)
def score_candidate(candidate, existing_positions):
    # 1. Breakout strength
    if candidate.triggered_on == '20D':
        breakout_strength = (candidate.close - candidate.ma20) / candidate.atr14
    else:  # 55D
        breakout_strength = (candidate.close - candidate.ma50) / candidate.atr14
    
    # 2. Momentum strength (relative)
    if candidate.asset_class == 'equity':
        benchmark_roc60 = SPY_roc60
    else:  # crypto
        benchmark_roc60 = BTC_roc60
    
    momentum_strength = candidate.roc60 - benchmark_roc60
    
    # 3. Diversification bonus
    if len(existing_positions) > 0:
        avg_corr = mean([correlation(candidate, pos) for pos in existing_positions])
        diversification_bonus = 1.0 - avg_corr
    else:
        diversification_bonus = 0.5  # neutral
    
    return {
        'breakout_strength': breakout_strength,
        'momentum_strength': momentum_strength,
        'diversification_bonus': diversification_bonus
    }

# Normalize via ranking across all candidates
# Then combine: 0.5×breakout_rank + 0.3×momentum_rank + 0.2×div_rank
```

---

## Sensitivity Testing Grid

### To Test on Train + Validation ONLY

```python
sensitivity_grid = {
    # Equities
    'equity_atr_mult': [2.0, 2.5, 3.0, 3.5],
    'equity_breakout_clearance': [0.000, 0.005, 0.010, 0.015],
    'equity_exit_ma': [20, 50],
    
    # Crypto
    'crypto_atr_mult': [2.5, 3.0, 3.5, 4.0],
    'crypto_breakout_clearance': [0.000, 0.005, 0.010, 0.015],
    'crypto_exit_mode': ['MA20', 'MA50', 'staged'],
    
    # Shared
    'vol_scaling_mode': ['continuous', 'regime', 'off'],
    'relative_strength_enabled': [False, True],  # optional for v1.1
}
```

**Selection criteria:**
- Choose parameters with stable performance across neighbors
- Avoid sharp peaks (overfitting)
- Prefer simpler if performance similar

---

## Metrics Targets (Holdout Evaluation)

### Primary (Must Pass ALL)

```python
PRIMARY_TARGETS = {
    'sharpe_ratio': 1.0,  # minimum
    'max_drawdown': 0.15,  # maximum (15%)
    'calmar_ratio': 1.5,  # minimum
    'min_trades': 50,  # combined equities + crypto
}
```

### Secondary (Must Pass 3 of 4)

```python
SECONDARY_TARGETS = {
    'expectancy_R': 0.3,  # minimum, after costs
    'profit_factor': 1.4,  # minimum
    'correlation_to_benchmark': 0.80,  # maximum
    'percentile_99_daily_loss': 0.05,  # maximum (5%)
}
```

### Stress Tests (Must Pass 2 of 3)

```python
STRESS_TEST_TARGETS = {
    '2x_slippage_sharpe': 0.75,  # minimum
    '3x_slippage_calmar': 1.0,  # minimum
    'bear_market_expectancy': 0.0,  # minimum (or max_dd < 20%)
}
```

### Rejection (ANY = Fail)

```python
REJECTION_CRITERIA = {
    'max_drawdown': 0.20,  # >20% = reject
    'sharpe_ratio': 0.75,  # <0.75 = reject
    'calmar_ratio': 1.0,  # <1.0 = reject
    'top_3_trade_dependency': True,  # if removing top 3 flips expectancy negative
    '2x_slippage_negative_expectancy': True,
    'bootstrap_5th_percentile_sharpe': 0.4,  # <0.4 = reject
    'permutation_test_percentile': 95,  # <95th percentile = reject
}
```

---

## Additional Metrics to Track

```python
# Recovery
RECOVERY_FACTOR_TARGET = 3.0  # net profit / max DD

# Drawdown duration
MAX_DRAWDOWN_DURATION_DAYS = 90  # alert if recovery takes longer

# Turnover
TURNOVER_MONTHLY_TARGET = (5, 15)  # min, max trades per month

# Holding period
AVG_HOLD_DAYS_TARGET = (7, 21)  # days

# Consecutive losses
MAX_CONSECUTIVE_LOSSES_ALERT = 10  # psychological threshold

# Win rate
EXPECTED_WIN_RATE = (0.40, 0.55)  # momentum systems typically 40-50%
```

---

## Implementation Notes

### Strategy Factory Pattern

```python
class MomentumStrategy:
    """Base class for both equity and crypto strategies."""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.universe = config.universe
        self.params = config.params
        
    def compute_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicators."""
        pass
        
    def generate_signals(
        self, 
        features: pd.DataFrame,
        portfolio: Portfolio,
        date: datetime
    ) -> List[Signal]:
        """Generate entry signals."""
        pass
        
    def update_stops(
        self,
        portfolio: Portfolio,
        current_data: pd.DataFrame
    ) -> List[Order]:
        """Update trailing stops daily."""
        pass
        
    def score_candidates(
        self,
        signals: List[Signal],
        portfolio: Portfolio
    ) -> Dict[str, float]:
        """Rank signals when slots limited."""
        pass
```

### Config-Driven Design

```python
# equity_config.yaml
universe: "NASDAQ-100"
trend_filter:
  ma_period: 50
  slope_lookback: 20
  slope_min: 0.005
breakout:
  fast_period: 20
  fast_clearance: 0.005
  slow_period: 55
  slow_clearance: 0.010
exit:
  ma_period: 20  # or 50, test both
  atr_mult: 2.5
risk:
  base_pct: 0.0075
  max_positions: 8
  max_exposure: 0.80
capacity:
  max_order_pct_adv: 0.005
costs:
  fee_bps: 1
  slippage_base_bps: 8
```

**Allows A/B testing without code changes.**

---

## Development Checklist

### Before Writing Any Code

- [ ] Sign pre-registration document
- [ ] Acquire 24 months of data (verified quality)
- [ ] Define train/validate/holdout splits
- [ ] Commit to frozen parameters

### During Implementation

- [ ] Build indicator pipeline (unit tested)
- [ ] Implement strategy factory (config-driven)
- [ ] Build slippage model (vol/size/stress scaled)
- [ ] Add capacity constraints (hard rejects)
- [ ] Implement scoring function (rank-normalized)
- [ ] Add correlation guard
- [ ] Build event logging system

### Validation Phase (Train + Validate Only)

- [ ] Run parameter sensitivity grid
- [ ] Plot heatmaps (check for sharp peaks)
- [ ] Run slippage stress tests (2x, 3x)
- [ ] Run bootstrap test (1,000 iterations)
- [ ] Run permutation test (1,000 iterations)
- [ ] Run scenario tests (bear, range, crash)
- [ ] Document final parameter choices

### Holdout Evaluation (ONCE)

- [ ] Review pre-registration criteria
- [ ] Run with final parameters
- [ ] Compute all metrics
- [ ] Check rejection criteria
- [ ] Make go/no-go decision
- [ ] If PASS → proceed to paper trading
- [ ] If FAIL → do NOT retune on holdout

---

This completes the comprehensive documentation for a 9+/10 implementation-ready momentum trading system. All major, minor, and nice-to-have improvements from the review are incorporated.
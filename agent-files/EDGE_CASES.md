# Edge Cases & Special Handling

Complete documentation of all edge cases, error conditions, and special handling logic throughout the trading system.

---

## Data Quality Edge Cases

### 1. Missing Data (Single Day)

**Scenario:** One day of data is missing for a symbol.

**Handling:**
```python
def handle_missing_data_single_day(symbol: str, date: pd.Timestamp):
    """
    Handle single day missing data.

    Actions:
    1. Skip signal generation for that symbol on that date
    2. Do NOT update stops for positions in that symbol
    3. Log warning: "MISSING_DATA_1DAY: {symbol} {date}"
    4. Continue processing other symbols normally
    """
    log.warning(f"MISSING_DATA_1DAY: {symbol} {date}")
    # Skip this symbol for this date
    return None
```

**Impact:**
- Indicators are not updated (use previous day's values)
- Signals are not generated
- Stops are not checked (use previous stop price)
- Position remains open

---

### 2. Missing Data (2+ Consecutive Days)

**Scenario:** Two or more consecutive days of data are missing.

**Handling:**
```python
def handle_missing_data_consecutive(symbol: str, missing_dates: List[pd.Timestamp], portfolio: Portfolio):
    """
    Handle 2+ consecutive missing days.

    Actions:
    1. Mark symbol as "unhealthy"
    2. If position exists:
       a. Attempt exit at next available open price
       b. Use last known close price if next open unavailable
       c. Set exit_reason = ExitReason.DATA_MISSING
    3. If no position:
       a. Exclude from universe until data resumes
       b. Remove from signal generation
    4. Log alert: "DATA_UNHEALTHY: {symbol}, missing {n} days"
    5. Send alert notification
    """
    if symbol in portfolio.positions:
        position = portfolio.positions[symbol]
        # Try to exit at next available data
        next_date = find_next_available_date(symbol, missing_dates[-1])
        if next_date:
            exit_price = get_bar(symbol, next_date).open
            close_position(position, exit_price, ExitReason.DATA_MISSING)
        else:
            # Force exit at last known close
            last_close = get_last_known_close(symbol, missing_dates[0])
            close_position(position, last_close, ExitReason.DATA_MISSING)

    mark_symbol_unhealthy(symbol)
    log.alert(f"DATA_UNHEALTHY: {symbol}, missing {len(missing_dates)} days")
    send_alert(f"Symbol {symbol} marked unhealthy due to missing data")
```

**Impact:**
- Position is closed immediately
- Symbol excluded from universe
- Alert sent to operator

**Maximum Wait:** Attempt exit for up to 3 consecutive missing days, then force exit at last known price.

---

### 3. Invalid OHLC Data

**Scenario:** OHLC relationships are violated (e.g., low > high).

**Handling:**
```python
def validate_bar(bar: Bar) -> bool:
    """
    Validate OHLC relationships.

    Checks:
    1. low <= open <= high
    2. low <= close <= high
    3. volume >= 0
    4. prices > 0

    Returns: True if valid, False otherwise
    """
    if not (bar.low <= bar.open <= bar.high):
        log.error(f"INVALID_OHLC: {bar.symbol} {bar.date}, open out of range")
        return False

    if not (bar.low <= bar.close <= bar.high):
        log.error(f"INVALID_OHLC: {bar.symbol} {bar.date}, close out of range")
        return False

    if bar.volume < 0:
        log.error(f"INVALID_VOLUME: {bar.symbol} {bar.date}, negative volume")
        return False

    if bar.close <= 0 or bar.open <= 0:
        log.error(f"INVALID_PRICE: {bar.symbol} {bar.date}, non-positive price")
        return False

    return True

def handle_invalid_bar(bar: Bar):
    """
    Handle invalid bar data.

    Actions:
    1. Mark as data error
    2. Skip this bar (do not use for indicators)
    3. Log error
    4. If in position: treat as missing data (see above)
    """
    mark_data_error(bar.symbol, bar.date)
    # Treat as missing data for that day
    handle_missing_data_single_day(bar.symbol, bar.date)
```

---

### 4. Extreme Price Moves (>50% in One Day)

**Scenario:** Price moves more than 50% in a single day (likely data error, not flash crash).

**Handling:**
```python
def detect_extreme_move(symbol: str, current_bar: Bar, previous_bar: Bar) -> bool:
    """
    Detect extreme price moves (>50%).

    Returns: True if move is extreme (likely data error)
    """
    if previous_bar is None:
        return False

    move_pct = abs(current_bar.close / previous_bar.close - 1)
    return move_pct > 0.50

def handle_extreme_move(symbol: str, bar: Bar):
    """
    Handle extreme price move.

    Actions:
    1. Mark as data error
    2. Skip this bar
    3. Log error with details
    4. Treat as missing data
    """
    log.error(f"EXTREME_MOVE: {symbol} {bar.date}, move: {move_pct:.1%}")
    mark_data_error(symbol, bar.date)
    handle_missing_data_single_day(symbol, bar.date)
```

**Note:** In real trading, flash crashes can occur, but for backtesting, we assume >50% moves are data errors.

---

## Indicator Calculation Edge Cases

### 5. Insufficient Lookback for Indicators

**Scenario:** Not enough historical data to compute indicators (e.g., MA200 needs 200 days).

**Handling:**
```python
def compute_ma(series: pd.Series, window: int) -> pd.Series:
    """
    Compute moving average with NaN handling.

    Rules:
    - Return NaN for all dates before window is filled
    - Do NOT forward-fill NaN values
    - Use NaN to prevent lookahead
    """
    ma = series.rolling(window=window).mean()
    # Explicitly set NaN for insufficient lookback
    ma.iloc[:window-1] = np.nan
    return ma

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute ATR using Wilder's smoothing.

    Rules:
    - Return NaN for first (period-1) bars
    - Use Wilder's exponential smoothing (not simple average)
    """
    # True Range calculation
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's smoothing
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    atr.iloc[:period-1] = np.nan
    return atr

def compute_roc(close: pd.Series, window: int = 60) -> pd.Series:
    """
    Compute rate of change.

    Rules:
    - Return NaN if close[t-window] is missing
    - Formula: (close[t] / close[t-window]) - 1
    """
    roc = (close / close.shift(window)) - 1
    return roc

def compute_highest_close(close: pd.Series, window: int) -> pd.Series:
    """
    Compute highest close over window (EXCLUDING today).

    Rules:
    - Use prior N days only (exclude today to avoid lookahead)
    - Return NaN until window is filled
    """
    # Shift by 1 to exclude today, then rolling max
    highest = close.shift(1).rolling(window=window).max()
    highest.iloc[:window] = np.nan
    return highest
```

**Key Rules:**
- All indicators return `NaN` until sufficient lookback
- Never forward-fill NaN values
- Use NaN to prevent signal generation when data is insufficient

---

### 6. NaN Values in Feature Calculation

**Scenario:** Some indicators are NaN (insufficient lookback) but others are valid.

**Handling:**
```python
def is_valid_for_signal(features: FeatureRow) -> bool:
    """
    Check if features are valid for signal generation.

    Required indicators:
    - ma20, ma50 (for eligibility)
    - atr14 (for stop calculation)
    - highest_close_20d, highest_close_55d (for entry triggers)
    - adv20 (for capacity check)

    Returns: True if all required indicators are valid
    """
    required = [
        features.ma20, features.ma50, features.atr14,
        features.highest_close_20d, features.highest_close_55d, features.adv20
    ]

    return all(x is not None and not np.isnan(x) for x in required)

def generate_signals(features: FeatureRow) -> Optional[Signal]:
    """
    Generate signals only if features are valid.
    """
    if not is_valid_for_signal(features):
        return None  # Skip signal generation

    # Proceed with signal generation
    ...
```

---

## Position Management Edge Cases

### 7. Position Sizing: Insufficient Cash

**Scenario:** Calculated position size requires more cash than available.

**Handling:**
```python
def calculate_position_size(
    equity: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
    max_position_notional: float,
    available_cash: float
) -> int:
    """
    Calculate position size with cash constraint.

    Steps:
    1. Calculate desired quantity from risk
    2. Check max position notional constraint
    3. Check available cash constraint
    4. Return minimum of all constraints
    """
    # Risk-based sizing
    risk_dollars = equity * risk_pct
    stop_distance = entry_price - stop_price
    if stop_distance <= 0:
        return 0  # Invalid stop

    qty_risk = int(risk_dollars / stop_distance)

    # Max position notional constraint
    max_qty_notional = int((equity * max_position_notional) / entry_price)

    # Cash constraint
    max_qty_cash = int(available_cash / entry_price)

    # Take minimum
    qty = min(qty_risk, max_qty_notional, max_qty_cash)

    # Minimum position size check (optional: reject if too small)
    if qty < 1:
        return 0  # Cannot afford even 1 share

    return qty
```

**Impact:**
- Position size is reduced to fit available cash
- Trade still executes if qty >= 1
- Log warning if size is significantly reduced

---

### 8. Position Sizing: Stop Price Above Entry Price

**Scenario:** Calculated stop price is above entry price (invalid for long positions).

**Handling:**
```python
def validate_stop_price(entry_price: float, stop_price: float) -> bool:
    """
    Validate stop price for long position.

    Rules:
    - Stop must be below entry price
    - Stop must be positive
    """
    if stop_price >= entry_price:
        log.error(f"INVALID_STOP: stop {stop_price} >= entry {entry_price}")
        return False

    if stop_price <= 0:
        log.error(f"INVALID_STOP: stop {stop_price} <= 0")
        return False

    return True

def handle_invalid_stop(signal: Signal) -> Optional[Signal]:
    """
    Handle invalid stop price.

    Actions:
    1. Reject signal
    2. Log error
    3. Return None
    """
    log.error(f"INVALID_STOP: {signal.symbol} {signal.date}, stop={signal.stop_price}, entry={signal.entry_price}")
    return None
```

**Impact:**
- Signal is rejected
- No position is opened
- Error is logged

---

### 9. Stop Price Update: Trailing Stop Logic

**Scenario:** How to update trailing stops (can only move up, never down).

**Handling:**
```python
def update_trailing_stop(
    position: Position,
    current_price: float,
    ma_level: float,  # MA20 or MA50
    atr14: float,
    atr_mult: float
) -> Optional[float]:
    """
    Update trailing stop for long position.

    Rules:
    1. Stop can only move UP (never down)
    2. If MA cross triggers exit, use MA level as stop
    3. Hard stop (ATR-based) can only tighten (crypto staged exit)
    4. Never widen hard stop

    Returns: New stop price if updated, None if unchanged
    """
    # Calculate potential new stop from ATR
    atr_stop = current_price - (atr_mult * atr14)

    # For trailing: use higher of current stop or ATR stop
    # But never go below current stop
    new_stop = max(position.stop_price, atr_stop)

    # If MA cross triggers exit, stop is set to MA level
    # (This is for exit logic, not stop update)

    # For crypto staged exit: if MA20 broken, tighten to 2.0x ATR
    if position.asset_class == "crypto" and not position.tightened_stop:
        if current_price < ma_level:  # MA20
            # Tighten stop
            tightened_stop = position.entry_price - (2.0 * atr14)
            new_stop = max(position.stop_price, tightened_stop)
            position.tightened_stop = True
            position.tightened_stop_atr_mult = 2.0

    # Only update if new stop is higher than current
    if new_stop > position.stop_price:
        position.stop_price = new_stop
        return new_stop

    return None
```

**Key Rules:**
- Stop price can only increase (for long positions)
- Never decrease stop price
- Crypto staged exit: tighten once when MA20 breaks, then never reset

---

### 10. Multiple Exit Signals on Same Day

**Scenario:** Both trailing MA cross and hard stop are triggered on the same day.

**Handling:**
```python
def check_exit_signals(
    position: Position,
    current_price: float,
    ma_level: float,
    stop_price: float
) -> Optional[ExitReason]:
    """
    Check exit signals with priority.

    Priority order:
    1. Hard stop (highest priority)
    2. Trailing MA cross
    3. Data missing (handled separately)

    Returns: ExitReason if exit triggered, None otherwise
    """
    # Priority 1: Hard stop
    if current_price <= position.stop_price:
        return ExitReason.HARD_STOP

    # Priority 2: Trailing MA cross
    if current_price < ma_level:
        return ExitReason.TRAILING_MA_CROSS

    return None
```

**Impact:**
- Hard stop takes precedence
- Exit reason is logged as "hard_stop" even if MA also crossed
- Consistent logging for analysis

---

## Portfolio-Level Edge Cases

### 11. Volatility Scaling: Insufficient History

**Scenario:** Portfolio has less than 20 days of history (can't compute 20D volatility).

**Handling:**
```python
def update_volatility_scaling(portfolio: Portfolio) -> float:
    """
    Update risk multiplier based on portfolio volatility.

    Rules:
    - If < 20 days history: use risk_multiplier = 1.0
    - If < 252 days history: use current vol as baseline
    - Compute 20D rolling vol (annualized)
    - Compare to median over 252D
    """
    returns = portfolio.compute_portfolio_returns(lookback=20)

    if len(returns) < 20:
        # Insufficient history: use default
        portfolio.risk_multiplier = 1.0
        portfolio.portfolio_vol_20d = None
        return 1.0

    # Compute 20D volatility (annualized)
    vol_20d = np.std(returns) * np.sqrt(252)
    portfolio.portfolio_vol_20d = vol_20d

    # Compute median over 252D (if available)
    all_returns = portfolio.compute_portfolio_returns(lookback=252)
    if len(all_returns) >= 252:
        # Compute rolling median
        rolling_vols = []
        for i in range(len(all_returns) - 19):
            window_returns = all_returns[i:i+20]
            window_vol = np.std(window_returns) * np.sqrt(252)
            rolling_vols.append(window_vol)
        median_vol = np.median(rolling_vols)
    else:
        # Use current vol as baseline if insufficient history
        median_vol = vol_20d

    portfolio.median_vol_252d = median_vol

    # Compute risk multiplier
    vol_ratio = vol_20d / median_vol if median_vol > 0 else 1.0
    risk_multiplier = max(0.33, min(1.0, 1.0 / max(vol_ratio, 1.0)))
    portfolio.risk_multiplier = risk_multiplier

    return risk_multiplier
```

---

### 12. Correlation Guard: Insufficient History

**Scenario:** Position has less than 20 days of return history (can't compute correlation).

**Handling:**
```python
def compute_correlation_to_portfolio(
    candidate_symbol: str,
    candidate_returns: List[float],
    portfolio_positions: Dict[str, Position],
    returns_data: Dict[str, List[float]],
    lookback: int = 20
) -> Optional[float]:
    """
    Compute correlation of candidate to existing portfolio.

    Rules:
    - Require minimum 10 days of returns for correlation
    - If insufficient: return None (skip correlation guard)
    - Use daily returns: (close[t] / close[t-1]) - 1
    """
    if len(candidate_returns) < 10:
        return None  # Insufficient history

    # Get returns for existing positions
    portfolio_returns = []
    for symbol, position in portfolio_positions.items():
        if symbol in returns_data and len(returns_data[symbol]) >= lookback:
            portfolio_returns.append(returns_data[symbol][-lookback:])

    if len(portfolio_returns) == 0:
        return None  # No portfolio returns available

    # Compute average correlation
    correlations = []
    candidate_returns_window = candidate_returns[-lookback:]

    for pos_returns in portfolio_returns:
        if len(pos_returns) == len(candidate_returns_window):
            corr = np.corrcoef(candidate_returns_window, pos_returns)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

    if len(correlations) == 0:
        return None

    return np.mean(correlations)
```

**Impact:**
- Correlation guard is skipped if insufficient data
- Candidate is not penalized for lack of correlation data
- Guard only applies when >= 4 positions AND sufficient return history

---

### 13. Position Queue: All Candidates Fail Constraints

**Scenario:** All signals fail capacity, correlation, or other constraints.

**Handling:**
```python
def select_positions_from_queue(
    signals: List[Signal],
    portfolio: Portfolio,
    max_positions: int,
    max_exposure: float
) -> List[Signal]:
    """
    Select positions from signal queue.

    Rules:
    1. Sort by score (descending)
    2. Apply constraints in order:
       a. Max positions
       b. Max exposure
       c. Capacity constraint
       d. Correlation guard
    3. If all candidates fail: return empty list (no new positions)
    4. Log warning if many signals rejected
    """
    # Sort by score
    sorted_signals = sorted(signals, key=lambda s: s.score, reverse=True)

    selected = []
    rejected_count = 0

    for signal in sorted_signals:
        # Check max positions
        if len(portfolio.positions) + len(selected) >= max_positions:
            rejected_count += 1
            continue

        # Check max exposure
        estimated_notional = signal.entry_price * estimate_quantity(signal, portfolio)
        if portfolio.gross_exposure + estimated_notional > portfolio.equity * max_exposure:
            rejected_count += 1
            continue

        # Check capacity
        if not signal.capacity_passed:
            rejected_count += 1
            continue

        # Check correlation guard
        if violates_correlation_guard(signal, portfolio):
            rejected_count += 1
            continue

        # All checks passed
        selected.append(signal)

    if rejected_count > 0:
        log.warning(f"REJECTED_SIGNALS: {rejected_count} signals rejected due to constraints")

    return selected
```

**Impact:**
- No new positions are opened if all fail constraints
- System continues normally (not an error condition)
- Warning is logged for monitoring

---

## Execution Edge Cases

### 14. Slippage Calculation: Extreme Values

**Scenario:** Slippage calculation produces extreme values (negative or very large).

**Handling:**
```python
def compute_slippage_bps(
    order: Order,
    market_state: MarketState,
    stress_state: StressState
) -> float:
    """
    Compute slippage with bounds checking.

    Rules:
    1. Slippage must be >= 0 (no negative slippage for market orders)
    2. Cap maximum slippage at 500 bps (5%) to prevent outliers
    3. Use normal distribution with clipping
    """
    # Compute mean slippage
    slippage_mean = compute_slippage_mean(order, market_state, stress_state)

    # Add variance
    slippage_std = slippage_mean * 0.75
    if stress_state.stress_mult == 2.0:
        slippage_std *= 1.5  # Fatter tails during stress

    # Sample from normal distribution
    slippage_draw = np.random.normal(slippage_mean, slippage_std)

    # Clip to valid range
    slippage_bps = max(0.0, min(500.0, slippage_draw))

    return slippage_bps
```

**Bounds:**
- Minimum: 0 bps (no negative slippage)
- Maximum: 500 bps (5%) to prevent data errors from causing extreme values

---

### 15. Weekly Return Calculation for Stress Multiplier

**Scenario:** How to compute weekly returns for stress detection.

**Handling:**
```python
def compute_weekly_return(
    benchmark: str,
    current_date: pd.Timestamp,
    bars: pd.DataFrame,
    asset_class: str
) -> float:
    """
    Compute weekly return for stress multiplier.

    Rules:
    - Equities: Last 5 trading days (Mon-Fri, skip weekends/holidays)
    - Crypto: Last 7 calendar days (UTC, continuous)
    - Formula: (close[t] / close[t-N]) - 1

    Returns: Weekly return as decimal (e.g., -0.03 for -3%)
    """
    if asset_class == "equity":
        # Get last 5 trading days
        trading_days = get_trading_days(bars.index, current_date, lookback=5)
        if len(trading_days) < 5:
            return 0.0  # Insufficient data

        start_date = trading_days[0]
        end_date = trading_days[-1]

    else:  # crypto
        # Get last 7 calendar days
        end_date = current_date
        start_date = end_date - pd.Timedelta(days=7)

    # Get closes
    start_close = bars.loc[start_date, 'close']
    end_close = bars.loc[end_date, 'close']

    if start_close <= 0:
        return 0.0

    weekly_return = (end_close / start_close) - 1
    return weekly_return
```

**Key Rules:**
- Equities: 5 trading days (use market calendar)
- Crypto: 7 calendar days (UTC)
- Return 0.0 if insufficient data (no stress multiplier)

---

## Data Loading Edge Cases

### 16. Symbol Not in Universe

**Scenario:** Data file contains symbols not in the defined universe.

**Handling:**
```python
def load_data(
    data_path: str,
    universe: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Load data and filter by universe.

    Rules:
    1. Load all available data
    2. Filter to universe symbols only
    3. Log warning for symbols in data but not in universe
    4. Log error for symbols in universe but missing from data
    """
    all_data = load_from_file(data_path)

    # Filter to universe
    universe_data = {}
    missing_symbols = []
    extra_symbols = []

    for symbol in universe:
        if symbol in all_data:
            universe_data[symbol] = all_data[symbol]
        else:
            missing_symbols.append(symbol)
            log.error(f"MISSING_UNIVERSE_SYMBOL: {symbol} not in data")

    for symbol in all_data:
        if symbol not in universe:
            extra_symbols.append(symbol)
            log.warning(f"EXTRA_SYMBOL: {symbol} in data but not in universe")

    return universe_data
```

---

### 17. Benchmark Data Missing

**Scenario:** SPY or BTC benchmark data is missing.

**Handling:**
```python
def load_benchmark(
    benchmark_symbol: str,
    data_path: str
) -> pd.DataFrame:
    """
    Load benchmark data with validation.

    Rules:
    1. Benchmark is REQUIRED (system cannot run without it)
    2. If missing: raise error and stop execution
    3. Validate benchmark has sufficient history
    """
    benchmark_data = load_from_file(data_path, symbols=[benchmark_symbol])

    if benchmark_symbol not in benchmark_data:
        raise ValueError(f"BENCHMARK_MISSING: {benchmark_symbol} not found in data")

    if len(benchmark_data[benchmark_symbol]) < 250:
        raise ValueError(f"BENCHMARK_INSUFFICIENT: {benchmark_symbol} has < 250 days of data")

    return benchmark_data[benchmark_symbol]
```

**Impact:**
- System cannot start without benchmark data
- Error is raised immediately
- User must provide benchmark data

---

## Summary

All edge cases are now documented with:
- Specific scenarios
- Handling logic
- Code examples
- Impact assessment
- Logging requirements

These edge cases should be handled explicitly in the implementation to ensure robust operation.

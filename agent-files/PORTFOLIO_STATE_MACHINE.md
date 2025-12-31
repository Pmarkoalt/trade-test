# Portfolio State Machine & Update Sequence

Explicit documentation of how portfolio state is updated throughout the daily event loop.

---

## Daily Event Loop Sequence

### Overview

For each calendar day `t` from start to end:

```
Day t Close → Generate Signals → Create Orders
    ↓
Day t+1 Open → Execute Orders → Update Portfolio
    ↓
Day t+1 Close → Update Stops → Check Exits → Log Metrics
    ↓
Advance to Day t+2
```

---

## Detailed Step-by-Step Sequence

### Step 1: Update Data Through Day t Close

**Time:** End of day `t` (after market close)

```python
def update_data_through_date(date: pd.Timestamp, market_data: MarketData):
    """
    Update all market data up to and including day t close.
    
    Actions:
    1. Load/update bars for all symbols up to date
    2. Compute indicators using data <= date (no lookahead)
    3. Update benchmark data (SPY, BTC)
    4. Validate data quality (OHLC, missing data, extreme moves)
    """
    # Load bars for date
    for symbol in universe:
        bar = load_bar(symbol, date)
        if bar:
            validate_bar(bar)  # Check OHLC relationships
            market_data.bars[symbol].loc[date] = bar
    
    # Compute indicators (using data <= date only)
    for symbol in universe:
        features = compute_features(market_data.bars[symbol], end_date=date)
        market_data.features[symbol].loc[date] = features
    
    # Update benchmarks
    update_benchmark("SPY", date)
    update_benchmark("BTC", date)
```

**Key Rule:** All indicators use data ≤ `date` (no lookahead).

---

### Step 2: Generate Signals at Day t Close

**Time:** End of day `t` (after data update)

```python
def generate_signals(date: pd.Timestamp, portfolio: Portfolio, strategies: List[Strategy]) -> List[Signal]:
    """
    Generate entry signals for all strategies at day t close.
    
    Actions:
    1. For each strategy (equity, crypto):
       a. Get features for all symbols in universe
       b. Check eligibility filters
       c. Check entry triggers (20D or 55D breakout)
       d. Calculate stop price
       e. Check capacity constraint
       f. Create signal if all pass
    2. Score signals (for queue ranking)
    3. Return list of valid signals
    """
    all_signals = []
    
    for strategy in strategies:
        for symbol in strategy.universe:
            # Get features
            features = market_data.features[symbol].loc[date]
            
            # Skip if insufficient data
            if not features.is_valid_for_entry():
                continue
            
            # Check eligibility
            if not strategy.check_eligibility(features):
                continue
            
            # Check entry triggers
            breakout_type = strategy.check_entry_triggers(features)
            if breakout_type is None:
                continue
            
            # Calculate stop price
            stop_price = features.close - (strategy.config.exit.hard_stop_atr_mult * features.atr14)
            
            # Validate stop
            if not validate_stop_price(features.close, stop_price):
                continue
            
            # Estimate position size (for capacity check)
            estimated_qty = estimate_position_size(
                equity=portfolio.equity,
                risk_pct=strategy.config.risk.risk_per_trade * portfolio.risk_multiplier,
                entry_price=features.close,
                stop_price=stop_price,
                max_notional=portfolio.equity * strategy.config.risk.max_position_notional
            )
            order_notional = features.close * estimated_qty
            
            # Check capacity
            if order_notional > strategy.config.capacity.max_order_pct_adv * features.adv20:
                continue  # Reject due to capacity
            
            # Create signal
            signal = Signal(
                symbol=symbol,
                asset_class=strategy.config.asset_class,
                date=date,
                side=SignalSide.BUY,
                entry_price=features.close,
                stop_price=stop_price,
                atr_mult=strategy.config.exit.hard_stop_atr_mult,
                triggered_on=breakout_type,
                breakout_clearance=compute_clearance(features, breakout_type),
                passed_eligibility=True,
                capacity_passed=True,
                order_notional=order_notional,
                adv20=features.adv20
            )
            
            all_signals.append(signal)
    
    # Score signals (for queue ranking)
    score_signals(all_signals, portfolio)
    
    return all_signals
```

**Key Rules:**
- Signals generated at day `t` close
- Entry price = close price at `t`
- Stop price calculated at signal time
- Capacity check uses estimated position size

---

### Step 3: Create Orders for Day t+1 Open

**Time:** End of day `t` (after signal generation)

```python
def create_orders(signals: List[Signal], portfolio: Portfolio, strategies: List[Strategy]) -> List[Order]:
    """
    Create orders from signals for execution at day t+1 open.
    
    Actions:
    1. Filter signals by constraints (max positions, max exposure, correlation guard)
    2. Select top N signals from queue
    3. Calculate exact position sizes
    4. Create orders
    5. Update portfolio state (reserve cash)
    """
    # Filter by constraints
    valid_signals = filter_signals_by_constraints(signals, portfolio, strategies)
    
    # Select from queue (if more signals than slots)
    selected_signals = select_from_queue(valid_signals, portfolio, strategies)
    
    orders = []
    
    for signal in selected_signals:
        strategy = get_strategy_for_signal(signal, strategies)
        
        # Calculate exact position size
        qty = calculate_position_size(
            equity=portfolio.equity,
            risk_pct=strategy.config.risk.risk_per_trade * portfolio.risk_multiplier,
            entry_price=signal.entry_price,
            stop_price=signal.stop_price,
            max_notional=portfolio.equity * strategy.config.risk.max_position_notional,
            available_cash=portfolio.cash
        )
        
        if qty < 1:
            continue  # Cannot afford position
        
        # Create order
        order = Order(
            order_id=generate_order_id(),
            symbol=signal.symbol,
            asset_class=signal.asset_class,
            date=signal.date,  # Signal date
            execution_date=get_next_open(signal.date),  # Day t+1 open
            side=SignalSide.BUY,
            quantity=qty,
            signal_date=signal.date,
            expected_fill_price=None,  # Will be set at execution
            stop_price=signal.stop_price,
            status=OrderStatus.PENDING
        )
        
        orders.append(order)
        
        # Reserve cash (estimate)
        estimated_cost = signal.entry_price * qty * 1.01  # Add 1% buffer for slippage/fees
        portfolio.cash -= estimated_cost  # Will be adjusted at execution
    
    return orders
```

**Key Rules:**
- Orders created at day `t` close
- Execution scheduled for day `t+1` open
- Cash is reserved (will be adjusted at execution)
- Position sizes calculated exactly

---

### Step 4: Execute Orders at Day t+1 Open

**Time:** Day `t+1` market open

```python
def execute_orders(orders: List[Order], market_data: MarketData, date: pd.Timestamp) -> List[Fill]:
    """
    Execute orders at day t+1 open with slippage and fees.
    
    Actions:
    1. Get open price for each symbol
    2. Calculate slippage (vol/size/stress scaled)
    3. Calculate fees
    4. Create fills
    5. Update order status
    """
    fills = []
    
    for order in orders:
        if order.status != OrderStatus.PENDING:
            continue
        
        # Get open price
        bar = market_data.get_bar(order.symbol, date)
        if bar is None:
            # Missing data: reject order
            order.status = OrderStatus.REJECTED
            order.rejection_reason = "MISSING_DATA"
            continue
        
        open_price = bar.open
        
        # Calculate slippage
        market_state = get_market_state(order.symbol, date, market_data)
        stress_state = get_stress_state(date, market_data)
        slippage_bps = compute_slippage_bps(order, market_state, stress_state)
        
        # Apply slippage to fill price
        if order.side == SignalSide.BUY:
            fill_price = open_price * (1 + slippage_bps / 10000)
        else:  # SELL
            fill_price = open_price * (1 - slippage_bps / 10000)
        
        # Calculate fees
        notional = fill_price * order.quantity
        fee_bps = get_fee_bps(order.asset_class)  # 1 for equity, 8 for crypto
        fee_cost = notional * (fee_bps / 10000)
        
        # Create fill
        fill = Fill(
            fill_id=generate_fill_id(),
            order_id=order.order_id,
            symbol=order.symbol,
            asset_class=order.asset_class,
            date=date,
            side=order.side,
            quantity=order.quantity,
            fill_price=fill_price,
            open_price=open_price,
            slippage_bps=slippage_bps,
            fee_bps=fee_bps,
            total_cost=(slippage_bps + fee_bps) * notional / 10000,
            vol_mult=market_state.vol_mult,
            size_penalty=market_state.size_penalty,
            weekend_penalty=market_state.weekend_penalty,
            stress_mult=stress_state.stress_mult,
            notional=notional
        )
        
        fills.append(fill)
        
        # Update order
        order.status = OrderStatus.FILLED
    
    return fills
```

**Key Rules:**
- Execution at day `t+1` open
- Fill price = open_price ± slippage
- Fees calculated on fill price
- Missing data → order rejected

---

### Step 5: Update Portfolio with Fills

**Time:** Day `t+1` (after execution)

```python
def update_portfolio_with_fills(portfolio: Portfolio, fills: List[Fill], date: pd.Timestamp):
    """
    Update portfolio state with executed fills.
    
    Actions:
    1. For each fill:
       a. Deduct cash (fill_price * quantity + fees)
       b. Create position
       c. Add to portfolio.positions
    2. Update equity
    3. Update exposure
    """
    for fill in fills:
        # Deduct cash
        total_cost = fill.fill_price * fill.quantity + fill.total_cost
        portfolio.cash -= total_cost
        
        # Create position
        position = Position(
            symbol=fill.symbol,
            asset_class=fill.asset_class,
            entry_date=fill.date,
            entry_price=fill.fill_price,
            entry_fill_id=fill.fill_id,
            quantity=fill.quantity,
            stop_price=get_stop_from_order(fill.order_id),  # From original signal
            initial_stop_price=get_stop_from_order(fill.order_id),
            hard_stop_atr_mult=get_atr_mult_from_order(fill.order_id),
            entry_slippage_bps=fill.slippage_bps,
            entry_fee_bps=fill.fee_bps,
            entry_total_cost=fill.total_cost,
            triggered_on=get_triggered_on_from_order(fill.order_id),
            adv20_at_entry=get_adv20_at_entry(fill.symbol, fill.date)
        )
        
        # Add to portfolio
        portfolio.positions[fill.symbol] = position
    
    # Update equity
    portfolio.update_equity(get_current_prices(date, portfolio.positions.keys(), market_data))
```

**Key Rules:**
- Cash deducted immediately
- Position created with entry details
- Equity updated after all fills

---

### Step 6: Update Stops at Day t+1 Close

**Time:** End of day `t+1` (after market close)

```python
def update_stops(portfolio: Portfolio, market_data: MarketData, date: pd.Timestamp) -> List[Order]:
    """
    Update trailing stops and check exit signals at day t+1 close.
    
    Actions:
    1. For each open position:
       a. Get current close price
       b. Update trailing stop (if applicable)
       c. Check exit signals (hard stop, trailing MA cross)
       d. Create exit order if triggered
    2. Return list of exit orders
    """
    exit_orders = []
    
    for symbol, position in portfolio.positions.items():
        if not position.is_open():
            continue
        
        # Get current bar
        bar = market_data.get_bar(symbol, date)
        if bar is None:
            # Missing data: handle separately
            handle_missing_data(symbol, date, portfolio)
            continue
        
        current_price = bar.close
        
        # Get features
        features = market_data.features[symbol].loc[date]
        
        # Update trailing stop (if applicable)
        # Note: For long positions, stop can only move up
        strategy = get_strategy_for_symbol(symbol, strategies)
        
        if strategy.config.exit.mode == "staged" and position.asset_class == "crypto":
            # Crypto staged exit logic
            if not position.tightened_stop and current_price < features.ma20:
                # Stage 1: Tighten stop
                tightened_stop = position.entry_price - (2.0 * features.atr14)
                if tightened_stop > position.stop_price:
                    position.stop_price = tightened_stop
                    position.tightened_stop = True
                    position.tightened_stop_atr_mult = 2.0
        
        # Check exit signals (priority: hard stop > trailing MA)
        exit_reason = None
        
        # Priority 1: Hard stop
        if current_price <= position.stop_price:
            exit_reason = ExitReason.HARD_STOP
        
        # Priority 2: Trailing MA cross
        elif strategy.config.exit.mode == "ma_cross":
            ma_level = features.ma20 if strategy.config.exit.exit_ma == 20 else features.ma50
            if current_price < ma_level:
                exit_reason = ExitReason.TRAILING_MA_CROSS
        
        elif strategy.config.exit.mode == "staged" and position.asset_class == "crypto":
            # Stage 2: MA50 cross OR tightened stop hit
            if current_price < features.ma50 or (position.tightened_stop and current_price <= position.stop_price):
                exit_reason = ExitReason.TRAILING_MA_CROSS
        
        # Create exit order if triggered
        if exit_reason:
            exit_order = Order(
                order_id=generate_order_id(),
                symbol=symbol,
                asset_class=position.asset_class,
                date=date,
                execution_date=get_next_open(date),  # Day t+2 open
                side=SignalSide.SELL,
                quantity=position.quantity,
                signal_date=date,
                stop_price=None,  # Not used for exits
                status=OrderStatus.PENDING
            )
            exit_order.exit_reason = exit_reason
            exit_orders.append(exit_order)
    
    return exit_orders
```

**Key Rules:**
- Stops updated at day `t+1` close
- Exit orders execute at day `t+2` open
- Hard stop takes priority over trailing MA
- Crypto staged exit: tighten once, then exit on MA50 or tightened stop

---

### Step 7: Execute Exit Orders at Day t+2 Open

**Time:** Day `t+2` market open

```python
def execute_exit_orders(exit_orders: List[Order], portfolio: Portfolio, market_data: MarketData, date: pd.Timestamp):
    """
    Execute exit orders and close positions.
    
    Actions:
    1. Execute exit fills (same as entry execution)
    2. Close positions
    3. Update cash
    4. Update realized P&L
    5. Remove from portfolio.positions
    """
    for order in exit_orders:
        # Execute fill (same logic as entry)
        fill = execute_order(order, market_data, date)
        
        if fill is None:
            continue  # Execution failed
        
        # Get position
        position = portfolio.positions[order.symbol]
        
        # Calculate realized P&L
        price_pnl = (fill.fill_price - position.entry_price) * position.quantity
        total_costs = position.entry_total_cost + fill.total_cost
        realized_pnl = price_pnl - total_costs
        
        # Update position
        position.exit_date = fill.date
        position.exit_price = fill.fill_price
        position.exit_fill_id = fill.fill_id
        position.exit_reason = order.exit_reason
        position.exit_slippage_bps = fill.slippage_bps
        position.exit_fee_bps = fill.fee_bps
        position.exit_total_cost = fill.total_cost
        position.realized_pnl = realized_pnl
        
        # Update cash
        portfolio.cash += fill.fill_price * fill.quantity - fill.total_cost
        
        # Update portfolio realized P&L
        portfolio.realized_pnl += realized_pnl
        
        # Remove from positions
        del portfolio.positions[order.symbol]
        
        # Log trade
        log_trade(position)
```

**Key Rules:**
- Exit execution same as entry (slippage + fees)
- Realized P&L = (exit_price - entry_price) * quantity - total_costs
- Position removed from portfolio after exit

---

### Step 8: Update Portfolio Metrics at Day t+1 Close

**Time:** End of day `t+1` (after all updates)

```python
def update_portfolio_metrics(portfolio: Portfolio, market_data: MarketData, date: pd.Timestamp):
    """
    Update all portfolio-level metrics at day t+1 close.
    
    Actions:
    1. Update equity (cash + position values at current prices)
    2. Update unrealized P&L for all positions
    3. Update exposure metrics
    4. Update volatility scaling
    5. Update correlation metrics
    6. Append to equity curve
    7. Compute daily return
    """
    # Get current prices
    current_prices = {}
    for symbol in portfolio.positions.keys():
        bar = market_data.get_bar(symbol, date)
        if bar:
            current_prices[symbol] = bar.close
    
    # Update equity
    portfolio.update_equity(current_prices)
    
    # Update volatility scaling
    portfolio.update_volatility_scaling()
    
    # Update correlation metrics
    returns_data = get_returns_data(portfolio.positions.keys(), market_data)
    portfolio.update_correlation_metrics(returns_data)
    
    # Append to equity curve
    portfolio.equity_curve.append(portfolio.equity)
    
    # Compute daily return
    if len(portfolio.equity_curve) > 1:
        daily_return = (portfolio.equity_curve[-1] / portfolio.equity_curve[-2]) - 1
        portfolio.daily_returns.append(daily_return)
    
    # Log daily metrics
    log_daily_metrics(portfolio, date)
```

**Key Rules:**
- All metrics updated at close
- Equity = cash + sum(position_values)
- Volatility scaling computed daily
- Correlation metrics only if >= 4 positions

---

## State Transition Diagram

```
[Day t Close]
    ↓
[Update Data] → [Generate Signals] → [Create Orders]
    ↓
[Day t+1 Open]
    ↓
[Execute Orders] → [Update Portfolio] → [Update Stops] → [Check Exits]
    ↓
[Day t+1 Close]
    ↓
[Update Metrics] → [Log Daily State]
    ↓
[Day t+2 Open]
    ↓
[Execute Exit Orders] → [Close Positions]
    ↓
[Advance to Next Day]
```

---

## Key Invariants

1. **No Lookahead:** Indicators at date `t` use only data ≤ `t`
2. **Execution Timing:** Orders created at `t` close execute at `t+1` open
3. **Stop Updates:** Stops updated at `t+1` close, exit orders execute at `t+2` open
4. **Cash Management:** Cash reserved at order creation, adjusted at execution
5. **Equity Consistency:** Equity = cash + sum(position_values) at all times
6. **Position Lifecycle:** Create → Update → Exit (never skip steps)

---

## Summary

The portfolio state machine follows a strict daily sequence:
1. Data update and signal generation at close
2. Order creation and cash reservation
3. Execution at next open
4. Stop updates and exit checks at close
5. Metric updates and logging

All state transitions are deterministic and traceable through the event log.


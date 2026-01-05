# Algorithm Pseudocode

Detailed pseudocode for complex algorithms used in validation and analysis.

---

## Bootstrap Resampling (Trade Returns)

### Purpose
Test statistical robustness by resampling trade R-multiples to build confidence intervals.

### Algorithm

```python
def bootstrap_analysis(trade_returns: List[float], n_iterations: int = 1000) -> Dict:
    """
    Bootstrap trade returns to build confidence intervals.

    Args:
        trade_returns: List of R-multiples per trade (closed positions only)
        n_iterations: Number of bootstrap samples

    Returns:
        Dictionary with percentile results and statistics
    """
    # Initialize storage
    sharpe_samples = []
    max_dd_samples = []
    calmar_samples = []
    expectancy_samples = []

    # Convert R-multiples to equity curve (for DD calculation)
    # Assume starting equity = 100,000, risk per trade = 0.75%
    starting_equity = 100000
    risk_per_trade = 0.0075

    # Original metrics (for comparison)
    original_sharpe = compute_sharpe_from_returns(trade_returns)
    original_max_dd = compute_max_drawdown_from_returns(trade_returns)
    original_calmar = original_sharpe / abs(original_max_dd) if original_max_dd != 0 else 0
    original_expectancy = np.mean(trade_returns)

    # Bootstrap loop
    for i in range(n_iterations):
        # Resample with replacement (same size as original)
        sample_returns = np.random.choice(trade_returns, size=len(trade_returns), replace=True)

        # Compute metrics on resampled data
        sharpe = compute_sharpe_from_returns(sample_returns)
        max_dd = compute_max_drawdown_from_returns(sample_returns)
        calmar = sharpe / abs(max_dd) if max_dd != 0 else 0
        expectancy = np.mean(sample_returns)

        # Store results
        sharpe_samples.append(sharpe)
        max_dd_samples.append(max_dd)
        calmar_samples.append(calmar)
        expectancy_samples.append(expectancy)

    # Compute percentiles
    results = {
        # Sharpe ratio
        'sharpe_5th': np.percentile(sharpe_samples, 5),
        'sharpe_25th': np.percentile(sharpe_samples, 25),
        'sharpe_50th': np.percentile(sharpe_samples, 50),
        'sharpe_75th': np.percentile(sharpe_samples, 75),
        'sharpe_95th': np.percentile(sharpe_samples, 95),

        # Max drawdown
        'max_dd_5th': np.percentile(max_dd_samples, 5),
        'max_dd_50th': np.percentile(max_dd_samples, 50),
        'max_dd_95th': np.percentile(max_dd_samples, 95),

        # Calmar ratio
        'calmar_5th': np.percentile(calmar_samples, 5),
        'calmar_50th': np.percentile(calmar_samples, 50),
        'calmar_95th': np.percentile(calmar_samples, 95),

        # Expectancy
        'expectancy_5th': np.percentile(expectancy_samples, 5),
        'expectancy_50th': np.percentile(expectancy_samples, 50),
        'expectancy_95th': np.percentile(expectancy_samples, 95),

        # Original values
        'original_sharpe': original_sharpe,
        'original_max_dd': original_max_dd,
        'original_calmar': original_calmar,
        'original_expectancy': original_expectancy,

        # Percentile ranks (where original falls in distribution)
        'sharpe_percentile_rank': percentileofscore(sharpe_samples, original_sharpe),
        'calmar_percentile_rank': percentileofscore(calmar_samples, original_calmar),
    }

    return results

def compute_sharpe_from_returns(r_multiples: List[float]) -> float:
    """
    Compute Sharpe ratio from R-multiples.

    Assumes:
    - Risk per trade = 0.75% of equity
    - R-multiple = profit/loss in units of risk
    - Convert to equity returns, then annualize
    """
    # Convert R-multiples to equity returns
    # Each trade: equity_change = R_multiple * risk_per_trade * equity
    # For simplicity, assume constant equity (or use actual equity curve)

    # Alternative: Use R-multiples directly as "returns"
    # Sharpe = mean(R) / std(R) * sqrt(252 / avg_hold_days)
    # But this requires holding period data

    # Simplified: Use R-multiples as if they were daily returns
    # (This is approximate but standard in trading literature)
    mean_r = np.mean(r_multiples)
    std_r = np.std(r_multiples)

    if std_r == 0:
        return 0.0

    # Annualize (assume ~15 trades per year, or use actual frequency)
    # For momentum system: ~10-20 trades/year, so sqrt(15) ≈ 3.87
    sharpe = mean_r / std_r * np.sqrt(15)  # Approximate

    return sharpe

def compute_max_drawdown_from_returns(r_multiples: List[float]) -> float:
    """
    Compute max drawdown from R-multiples.

    Build equity curve from R-multiples, then compute DD.
    """
    # Build equity curve
    starting_equity = 100000
    risk_per_trade = 0.0075

    equity_curve = [starting_equity]
    for r_mult in r_multiples:
        # Each trade changes equity by: R_multiple * risk_per_trade * current_equity
        equity_change = r_mult * risk_per_trade * equity_curve[-1]
        new_equity = equity_curve[-1] + equity_change
        equity_curve.append(new_equity)

    # Compute drawdown
    peak = equity_curve[0]
    max_dd = 0.0

    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = (equity - peak) / peak
        if dd < max_dd:
            max_dd = dd

    return max_dd
```

### Acceptance Criteria

```python
def check_bootstrap_results(results: Dict) -> Tuple[bool, List[str]]:
    """
    Check bootstrap results against acceptance criteria.

    Returns:
        (passed, warnings)
    """
    warnings = []

    # Rejection: Sharpe 5th percentile < 0.4
    if results['sharpe_5th'] < 0.4:
        return (False, ["REJECT: Bootstrap Sharpe 5th percentile < 0.4"])

    # Warning: Sharpe 5th percentile < 0.6
    if results['sharpe_5th'] < 0.6:
        warnings.append("WARNING: Bootstrap Sharpe 5th percentile < 0.6 (fragile)")

    # Warning: Max DD 95th percentile > 25%
    if results['max_dd_95th'] > 0.25:
        warnings.append("WARNING: Bootstrap Max DD 95th percentile > 25% (tail risk)")

    # Check percentile rank (original should be near median)
    if results['sharpe_percentile_rank'] < 40 or results['sharpe_percentile_rank'] > 60:
        warnings.append(f"WARNING: Original Sharpe at {results['sharpe_percentile_rank']:.1f}th percentile (unusual)")

    return (True, warnings)
```

---

## Permutation Test (Entry Timing)

### Purpose
Test if entry timing adds value vs random entry dates (preserving exit logic and holding period distribution).

### Algorithm

```python
def permutation_test(
    strategy: Strategy,
    data: MarketData,
    portfolio: Portfolio,
    period: Tuple[pd.Timestamp, pd.Timestamp],
    n_iterations: int = 1000
) -> Dict:
    """
    Permutation test: Randomize entry dates while preserving exit logic.

    Tests if strategy performance is due to entry skill or luck.

    Args:
        strategy: Strategy to test
        data: Market data
        portfolio: Initial portfolio state
        period: (start_date, end_date) for test period
        n_iterations: Number of randomized runs

    Returns:
        Dictionary with results and percentile rank
    """
    # Run actual strategy
    actual_results = run_backtest(strategy, data, portfolio, period)
    actual_sharpe = actual_results.sharpe
    actual_trades = actual_results.trades

    # Extract actual entry/exit pairs
    actual_entries = [(t.entry_date, t.exit_date, t.symbol) for t in actual_trades]

    # Store randomized Sharpe ratios
    random_sharpes = []

    for i in range(n_iterations):
        # Randomize entry dates
        randomized_entries = randomize_entry_dates(actual_entries, period)

        # Run backtest with randomized entries (preserve exit logic)
        random_results = run_backtest_randomized(
            strategy, data, portfolio, period, randomized_entries
        )

        random_sharpes.append(random_results.sharpe)

    # Compute percentile rank
    percentile_rank = percentileofscore(random_sharpes, actual_sharpe)

    results = {
        'actual_sharpe': actual_sharpe,
        'random_sharpe_5th': np.percentile(random_sharpes, 5),
        'random_sharpe_50th': np.percentile(random_sharpes, 50),
        'random_sharpe_95th': np.percentile(random_sharpes, 95),
        'percentile_rank': percentile_rank,
        'passed': percentile_rank >= 95  # Must be >= 95th percentile
    }

    return results

def randomize_entry_dates(
    actual_entries: List[Tuple[pd.Timestamp, pd.Timestamp, str]],
    period: Tuple[pd.Timestamp, pd.Timestamp]
) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
    """
    Randomize entry dates while preserving:
    - Exit logic (same exit dates relative to entry)
    - Holding period distribution
    - Symbol assignment
    """
    start_date, end_date = period

    randomized = []

    for entry_date, exit_date, symbol in actual_entries:
        # Compute holding period
        hold_days = (exit_date - entry_date).days

        # Randomize entry date (within period, ensuring exit is also in period)
        max_entry_date = end_date - pd.Timedelta(days=hold_days)

        if max_entry_date < start_date:
            # Cannot fit: skip this trade
            continue

        # Random entry date
        random_entry = start_date + pd.Timedelta(
            days=np.random.randint(0, (max_entry_date - start_date).days + 1)
        )

        # Preserve exit date relative to entry
        random_exit = random_entry + pd.Timedelta(days=hold_days)

        randomized.append((random_entry, random_exit, symbol))

    return randomized

def run_backtest_randomized(
    strategy: Strategy,
    data: MarketData,
    portfolio: Portfolio,
    period: Tuple[pd.Timestamp, pd.Timestamp],
    randomized_entries: List[Tuple[pd.Timestamp, pd.Timestamp, str]]
) -> BacktestResults:
    """
    Run backtest with randomized entry dates.

    For each randomized entry:
    1. Check if entry conditions were met at that date
    2. If yes: enter at that date
    3. Use original exit logic (MA cross, stop, etc.)
    4. Compute returns
    """
    # Simplified: Just compute returns from randomized entries
    # (In practice, need to check entry conditions at randomized dates)

    trades = []

    for entry_date, exit_date, symbol in randomized_entries:
        # Get entry/exit prices
        entry_bar = data.get_bar(symbol, entry_date)
        exit_bar = data.get_bar(symbol, exit_date)

        if entry_bar is None or exit_bar is None:
            continue

        # Create trade (simplified)
        trade = Trade(
            symbol=symbol,
            entry_date=entry_date,
            entry_price=entry_bar.close,  # Simplified: use close
            exit_date=exit_date,
            exit_price=exit_bar.close,
            # ... other fields
        )

        trades.append(trade)

    # Compute metrics
    returns = compute_returns_from_trades(trades)
    sharpe = compute_sharpe(returns)

    return BacktestResults(sharpe=sharpe, trades=trades)
```

### Acceptance Criteria

```python
def check_permutation_test(results: Dict) -> Tuple[bool, List[str]]:
    """
    Check permutation test results.

    Returns:
        (passed, warnings)
    """
    if not results['passed']:
        return (False, ["REJECT: Permutation test failed (actual Sharpe < 95th percentile random)"])

    return (True, [])
```

---

## Correlation Stress Analysis

### Purpose
Analyze correlation behavior during drawdowns to check if diversification fails during stress.

### Algorithm

```python
def correlation_stress_analysis(
    portfolio_history: Dict[pd.Timestamp, Portfolio],
    returns_data: Dict[str, pd.Series]
) -> Dict:
    """
    Analyze correlation behavior during drawdowns.

    Args:
        portfolio_history: Historical portfolio states
        returns_data: Daily returns for all symbols

    Returns:
        Dictionary with correlation statistics
    """
    # Compute drawdown for each date
    equity_curve = [p.equity for p in portfolio_history.values()]
    peaks = compute_rolling_peaks(equity_curve)
    drawdowns = [(eq - peak) / peak for eq, peak in zip(equity_curve, peaks)]

    # Classify periods
    normal_periods = []  # DD >= -5%
    drawdown_periods = []  # DD < -5%

    for date, dd in zip(portfolio_history.keys(), drawdowns):
        portfolio = portfolio_history[date]

        if len(portfolio.positions) < 2:
            continue  # Need at least 2 positions for correlation

        # Compute correlation
        position_symbols = list(portfolio.positions.keys())
        corr_matrix = compute_correlation_matrix(position_symbols, returns_data, date, lookback=20)

        if corr_matrix is None:
            continue

        avg_pairwise_corr = compute_avg_pairwise_corr(corr_matrix)

        if dd >= -0.05:
            normal_periods.append(avg_pairwise_corr)
        else:
            drawdown_periods.append(avg_pairwise_corr)

    # Compute statistics
    results = {
        'normal_avg_corr': np.mean(normal_periods) if normal_periods else None,
        'normal_std_corr': np.std(normal_periods) if normal_periods else None,
        'drawdown_avg_corr': np.mean(drawdown_periods) if drawdown_periods else None,
        'drawdown_std_corr': np.std(drawdown_periods) if drawdown_periods else None,
        'correlation_increase': None,
        'warning': False
    }

    if results['normal_avg_corr'] and results['drawdown_avg_corr']:
        results['correlation_increase'] = results['drawdown_avg_corr'] - results['normal_avg_corr']

        # Warning if correlation increases significantly during drawdowns
        if results['drawdown_avg_corr'] > 0.70:
            results['warning'] = True

    return results

def compute_correlation_matrix(
    symbols: List[str],
    returns_data: Dict[str, pd.Series],
    date: pd.Timestamp,
    lookback: int = 20
) -> Optional[np.ndarray]:
    """
    Compute correlation matrix for symbols at date.

    Uses rolling 20D window of daily returns.
    """
    # Get returns for each symbol
    symbol_returns = []

    for symbol in symbols:
        if symbol not in returns_data:
            return None

        returns = returns_data[symbol]

        # Get window ending at date
        date_idx = returns.index.get_loc(date, method='nearest')
        if date_idx < lookback:
            return None  # Insufficient history

        window_returns = returns.iloc[date_idx - lookback + 1:date_idx + 1]
        symbol_returns.append(window_returns.values)

    if len(symbol_returns) < 2:
        return None

    # Compute correlation matrix
    returns_array = np.array(symbol_returns)
    corr_matrix = np.corrcoef(returns_array)

    return corr_matrix

def compute_avg_pairwise_corr(corr_matrix: np.ndarray) -> float:
    """
    Compute average pairwise correlation (exclude diagonal).
    """
    n = len(corr_matrix)
    off_diagonal = []

    for i in range(n):
        for j in range(i + 1, n):
            if not np.isnan(corr_matrix[i, j]):
                off_diagonal.append(corr_matrix[i, j])

    return np.mean(off_diagonal) if off_diagonal else 0.0
```

---

## Summary

All complex algorithms are now documented with:
- Purpose and rationale
- Detailed pseudocode
- Acceptance criteria
- Implementation notes

These algorithms can be directly implemented in Python using the provided pseudocode as a guide.

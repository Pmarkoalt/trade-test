"""Helper functions for creating test data and fixtures."""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from trading_system.models.bar import Bar
from trading_system.models.features import FeatureRow
from trading_system.models.signals import Signal, SignalSide, BreakoutType
from trading_system.models.orders import Order, OrderStatus, Fill
from trading_system.models.portfolio import Portfolio
from trading_system.models.positions import Position, ExitReason
from trading_system.models.market_data import MarketData


def create_sample_bar(
    date: pd.Timestamp,
    symbol: str = "AAPL",
    base_price: float = 150.0,
    volatility: float = 0.02,
    volume: float = 50000000.0
) -> Bar:
    """Create a sample Bar with realistic OHLCV data.
    
    Args:
        date: Date for the bar
        symbol: Symbol name
        base_price: Base price level
        volatility: Volatility multiplier (0.02 = 2% daily volatility)
        volume: Volume for the day
    
    Returns:
        Bar object with OHLCV data
    """
    # Generate realistic OHLC from base price
    open_price = base_price * (1 + np.random.uniform(-volatility, volatility))
    close_price = base_price * (1 + np.random.uniform(-volatility, volatility))
    high_price = max(open_price, close_price) * (1 + np.random.uniform(0, volatility))
    low_price = min(open_price, close_price) * (1 - np.random.uniform(0, volatility))
    
    return Bar(
        date=date,
        symbol=symbol,
        open=round(open_price, 2),
        high=round(high_price, 2),
        low=round(low_price, 2),
        close=round(close_price, 2),
        volume=volume
    )


def create_sample_feature_row(
    date: pd.Timestamp,
    symbol: str = "AAPL",
    asset_class: str = "equity",
    close: float = 150.0,
    ma20: Optional[float] = None,
    ma50: Optional[float] = None,
    ma200: Optional[float] = None,
    atr14: Optional[float] = None,
    **kwargs
) -> FeatureRow:
    """Create a sample FeatureRow with indicators.
    
    Args:
        date: Date for the features
        symbol: Symbol name
        asset_class: "equity" or "crypto"
        close: Close price
        ma20: MA20 value (defaults to close * 0.98)
        ma50: MA50 value (defaults to close * 0.95)
        ma200: MA200 value (defaults to close * 0.90)
        atr14: ATR14 value (defaults to close * 0.02)
        **kwargs: Additional feature values
    
    Returns:
        FeatureRow object
    """
    # Set defaults if not provided
    if ma20 is None:
        ma20 = close * 0.98
    if ma50 is None:
        ma50 = close * 0.95
    if ma200 is None:
        ma200 = close * 0.90
    if atr14 is None:
        atr14 = close * 0.02
    
    # Set defaults for other features
    open_price = kwargs.get('open', close * 0.99)
    high = kwargs.get('high', close * 1.01)
    low = kwargs.get('low', close * 0.99)
    roc60 = kwargs.get('roc60', 0.05)
    highest_close_20d = kwargs.get('highest_close_20d', close * 0.98)
    highest_close_55d = kwargs.get('highest_close_55d', close * 0.95)
    adv20 = kwargs.get('adv20', close * 50000000)  # Assuming 50M volume
    returns_1d = kwargs.get('returns_1d', 0.001)
    benchmark_roc60 = kwargs.get('benchmark_roc60', 0.03)
    benchmark_returns_1d = kwargs.get('benchmark_returns_1d', 0.0005)
    
    return FeatureRow(
        date=date,
        symbol=symbol,
        asset_class=asset_class,
        close=close,
        open=open_price,
        high=high,
        low=low,
        ma20=ma20,
        ma50=ma50,
        ma200=ma200,
        atr14=atr14,
        roc60=roc60,
        highest_close_20d=highest_close_20d,
        highest_close_55d=highest_close_55d,
        adv20=adv20,
        returns_1d=returns_1d,
        benchmark_roc60=benchmark_roc60,
        benchmark_returns_1d=benchmark_returns_1d,
    )


def create_sample_signal(
    date: pd.Timestamp,
    symbol: str = "AAPL",
    asset_class: str = "equity",
    entry_price: float = 150.0,
    atr14: float = 3.0,
    atr_mult: float = 2.5,
    triggered_on: BreakoutType = BreakoutType.FAST_20D,
    **kwargs
) -> Signal:
    """Create a sample Signal.
    
    Args:
        date: Signal date
        symbol: Symbol name
        asset_class: "equity" or "crypto"
        entry_price: Entry price
        atr14: ATR14 value
        atr_mult: ATR multiplier for stop
        triggered_on: Breakout type
        **kwargs: Additional signal parameters
    
    Returns:
        Signal object
    """
    stop_price = entry_price - (atr_mult * atr14)
    
    return Signal(
        symbol=symbol,
        asset_class=asset_class,
        date=date,
        side=SignalSide.BUY,
        entry_price=entry_price,
        stop_price=stop_price,
        atr_mult=atr_mult,
        triggered_on=triggered_on,
        breakout_clearance=kwargs.get('breakout_clearance', 0.005),
        breakout_strength=kwargs.get('breakout_strength', 0.5),
        momentum_strength=kwargs.get('momentum_strength', 0.5),
        diversification_bonus=kwargs.get('diversification_bonus', 0.5),
        score=kwargs.get('score', 0.5),
        passed_eligibility=kwargs.get('passed_eligibility', True),
        eligibility_failures=kwargs.get('eligibility_failures', []),
        order_notional=kwargs.get('order_notional', 100000.0),
        adv20=kwargs.get('adv20', 100000000.0),
        capacity_passed=kwargs.get('capacity_passed', True),
    )


def create_sample_order(
    order_id: str,
    symbol: str,
    asset_class: str,
    date: pd.Timestamp,
    execution_date: pd.Timestamp,
    quantity: int,
    expected_fill_price: float,
    stop_price: float,
    signal_date: Optional[pd.Timestamp] = None,
    **kwargs
) -> Order:
    """Create a sample Order.
    
    Args:
        order_id: Unique order ID
        symbol: Symbol name
        asset_class: "equity" or "crypto"
        date: Order creation date
        execution_date: Execution date
        quantity: Order quantity
        expected_fill_price: Expected fill price
        stop_price: Stop price
        signal_date: Original signal date (defaults to date)
        **kwargs: Additional order parameters
    
    Returns:
        Order object
    """
    if signal_date is None:
        signal_date = date
    
    return Order(
        order_id=order_id,
        symbol=symbol,
        asset_class=asset_class,
        date=date,
        execution_date=execution_date,
        side=kwargs.get('side', SignalSide.BUY),
        quantity=quantity,
        limit_price=kwargs.get('limit_price', None),
        signal_date=signal_date,
        expected_fill_price=expected_fill_price,
        stop_price=stop_price,
        status=kwargs.get('status', OrderStatus.PENDING),
        rejection_reason=kwargs.get('rejection_reason', None),
        capacity_checked=kwargs.get('capacity_checked', False),
        correlation_checked=kwargs.get('correlation_checked', False),
        max_positions_checked=kwargs.get('max_positions_checked', False),
        max_exposure_checked=kwargs.get('max_exposure_checked', False),
    )


def create_sample_fill(
    fill_id: str,
    order_id: str,
    symbol: str,
    asset_class: str,
    date: pd.Timestamp,
    quantity: int,
    fill_price: float,
    open_price: float,
    slippage_bps: float = 10.0,
    fee_bps: float = 1.0,
    **kwargs
) -> Fill:
    """Create a sample Fill.
    
    Args:
        fill_id: Unique fill ID
        order_id: Parent order ID
        symbol: Symbol name
        asset_class: "equity" or "crypto"
        date: Fill date
        quantity: Filled quantity
        fill_price: Actual fill price
        open_price: Market open price
        slippage_bps: Slippage in basis points
        fee_bps: Fee in basis points
        **kwargs: Additional fill parameters
    
    Returns:
        Fill object
    """
    return Fill(
        fill_id=fill_id,
        order_id=order_id,
        symbol=symbol,
        asset_class=asset_class,
        date=date,
        side=kwargs.get('side', SignalSide.BUY),
        quantity=quantity,
        fill_price=fill_price,
        open_price=open_price,
        slippage_bps=slippage_bps,
        fee_bps=fee_bps,
        total_cost=kwargs.get('total_cost', None),  # Will be computed if None
        vol_mult=kwargs.get('vol_mult', 1.0),
        size_penalty=kwargs.get('size_penalty', 1.0),
        weekend_penalty=kwargs.get('weekend_penalty', 1.0),
        stress_mult=kwargs.get('stress_mult', 1.0),
        notional=kwargs.get('notional', fill_price * quantity),
    )


def create_sample_position(
    symbol: str,
    asset_class: str,
    entry_date: pd.Timestamp,
    entry_price: float,
    quantity: int,
    stop_price: float,
    **kwargs
) -> Position:
    """Create a sample Position.
    
    Args:
        symbol: Symbol name
        asset_class: "equity" or "crypto"
        entry_date: Entry date
        entry_price: Entry price
        quantity: Position quantity
        stop_price: Stop price
        **kwargs: Additional position parameters
    
    Returns:
        Position object
    """
    from trading_system.models.signals import BreakoutType
    
    return Position(
        symbol=symbol,
        asset_class=asset_class,
        entry_date=entry_date,
        entry_price=entry_price,
        entry_fill_id=kwargs.get('entry_fill_id', f"fill_{symbol}_{entry_date.date()}"),
        quantity=quantity,
        stop_price=stop_price,
        initial_stop_price=kwargs.get('initial_stop_price', stop_price),
        hard_stop_atr_mult=kwargs.get('hard_stop_atr_mult', 2.5),
        entry_slippage_bps=kwargs.get('entry_slippage_bps', 10.0),
        entry_fee_bps=kwargs.get('entry_fee_bps', 1.0),
        entry_total_cost=kwargs.get('entry_total_cost', 0.0),
        triggered_on=kwargs.get('triggered_on', BreakoutType.FAST_20D),
        adv20_at_entry=kwargs.get('adv20_at_entry', 100000000.0),
    )


def create_sample_portfolio(
    date: pd.Timestamp,
    starting_equity: float = 100000.0,
    cash: Optional[float] = None,
    **kwargs
) -> Portfolio:
    """Create a sample Portfolio.
    
    Args:
        date: Portfolio date
        starting_equity: Starting equity
        cash: Cash amount (defaults to starting_equity)
        **kwargs: Additional portfolio parameters
    
    Returns:
        Portfolio object
    """
    if cash is None:
        cash = starting_equity
    
    portfolio = Portfolio(
        date=date,
        cash=cash,
        starting_equity=starting_equity,
        equity=starting_equity,
        positions=kwargs.get('positions', {}),
        equity_curve=kwargs.get('equity_curve', [starting_equity]),
        daily_returns=kwargs.get('daily_returns', []),
        gross_exposure=kwargs.get('gross_exposure', 0.0),
        gross_exposure_pct=kwargs.get('gross_exposure_pct', 0.0),
        per_position_exposure=kwargs.get('per_position_exposure', {}),
        realized_pnl=kwargs.get('realized_pnl', 0.0),
        unrealized_pnl=kwargs.get('unrealized_pnl', 0.0),
        portfolio_vol_20d=kwargs.get('portfolio_vol_20d', None),
        median_vol_252d=kwargs.get('median_vol_252d', None),
        risk_multiplier=kwargs.get('risk_multiplier', 1.0),
        avg_pairwise_corr=kwargs.get('avg_pairwise_corr', None),
        correlation_matrix=kwargs.get('correlation_matrix', None),
        total_trades=kwargs.get('total_trades', 0),
        open_trades=kwargs.get('open_trades', 0),
    )
    
    return portfolio


def create_mock_market_data(
    symbols: List[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    base_prices: Optional[Dict[str, float]] = None
) -> MarketData:
    """Create mock MarketData with OHLCV bars for multiple symbols.
    
    Args:
        symbols: List of symbols
        start_date: Start date
        end_date: End date
        base_prices: Optional dict of symbol -> base price
    
    Returns:
        MarketData object
    """
    if base_prices is None:
        base_prices = {symbol: 100.0 for symbol in symbols}
    
    market_data = MarketData()
    
    # Generate trading days (weekdays only for equity, all days for crypto)
    dates = pd.bdate_range(start_date, end_date)
    
    for symbol in symbols:
        bars_dict = {}
        current_price = base_prices[symbol]
        
        for date in dates:
            bar = create_sample_bar(
                date=date,
                symbol=symbol,
                base_price=current_price,
                volatility=0.02,
                volume=50000000.0
            )
            bars_dict[date] = bar
            # Update current price for next bar (random walk)
            current_price = bar.close
        
        market_data.bars[symbol] = bars_dict
    
    return market_data


def generate_ohlcv_data(
    symbol: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    base_price: float = 100.0,
    volatility: float = 0.02,
    base_volume: float = 50000000.0,
    trend: float = 0.0
) -> pd.DataFrame:
    """Generate OHLCV DataFrame for testing.
    
    Args:
        symbol: Symbol name
        start_date: Start date
        end_date: End date
        base_price: Starting price
        volatility: Daily volatility
        base_volume: Base volume
        trend: Daily trend (0.0 = no trend, 0.001 = 0.1% daily upward)
    
    Returns:
        DataFrame with OHLCV columns and date index
    """
    dates = pd.bdate_range(start_date, end_date)
    data = []
    
    current_price = base_price
    
    for date in dates:
        # Generate OHLC from current price
        open_price = current_price * (1 + np.random.uniform(-volatility, volatility))
        close_price = current_price * (1 + np.random.uniform(-volatility, volatility) + trend)
        high_price = max(open_price, close_price) * (1 + np.random.uniform(0, volatility * 0.5))
        low_price = min(open_price, close_price) * (1 - np.random.uniform(0, volatility * 0.5))
        
        volume = base_volume * (1 + np.random.uniform(-0.2, 0.2))
        
        data.append({
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': round(volume, 2),
        })
        
        current_price = close_price
    
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'date'
    return df


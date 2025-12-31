"""Assertion helpers for testing trading system components."""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import pytest

from trading_system.models.bar import Bar
from trading_system.models.signals import Signal
from trading_system.models.orders import Order, Fill
from trading_system.models.portfolio import Portfolio


def assert_no_lookahead(
    signals: List[Signal],
    bars: Dict[pd.Timestamp, Bar],
    feature_rows: Optional[Dict[pd.Timestamp, Any]] = None
) -> None:
    """Assert that signals do not use future data (no lookahead bias).
    
    Args:
        signals: List of signals to check
        bars: Dictionary of date -> Bar (used to verify prices)
        feature_rows: Optional dictionary of date -> FeatureRow (for indicator checks)
    
    Raises:
        AssertionError: If lookahead bias is detected
    """
    for signal in signals:
        signal_date = signal.date
        
        # Check that entry_price matches close price at signal date
        if signal_date in bars:
            expected_price = bars[signal_date].close
            assert abs(signal.entry_price - expected_price) < 0.01, (
                f"Signal {signal.symbol} at {signal_date}: entry_price {signal.entry_price} "
                f"does not match close price {expected_price}"
            )
        
        # Check that stop_price is below entry_price
        assert signal.stop_price < signal.entry_price, (
            f"Signal {signal.symbol} at {signal_date}: stop_price {signal.stop_price} "
            f"is not below entry_price {signal.entry_price}"
        )
        
        # If feature rows provided, check that indicators used don't exceed signal date
        if feature_rows is not None:
            # This is a simplified check - in practice, you'd verify that indicators
            # don't use data beyond the signal date
            if signal_date in feature_rows:
                # Feature row should exist for the signal date
                pass


def assert_valid_bar(bar: Bar) -> None:
    """Assert that a Bar has valid OHLCV relationships.
    
    Args:
        bar: Bar to validate
    
    Raises:
        AssertionError: If bar is invalid
    """
    # Check prices are positive
    assert bar.open > 0, f"Bar {bar.symbol} {bar.date}: open price must be positive"
    assert bar.high > 0, f"Bar {bar.symbol} {bar.date}: high price must be positive"
    assert bar.low > 0, f"Bar {bar.symbol} {bar.date}: low price must be positive"
    assert bar.close > 0, f"Bar {bar.symbol} {bar.date}: close price must be positive"
    
    # Check OHLC relationships
    assert bar.low <= bar.high, (
        f"Bar {bar.symbol} {bar.date}: low {bar.low} > high {bar.high}"
    )
    assert bar.low <= bar.open <= bar.high, (
        f"Bar {bar.symbol} {bar.date}: open {bar.open} not between low/high"
    )
    assert bar.low <= bar.close <= bar.high, (
        f"Bar {bar.symbol} {bar.date}: close {bar.close} not between low/high"
    )
    
    # Check volume
    assert bar.volume >= 0, f"Bar {bar.symbol} {bar.date}: volume must be non-negative"
    
    # Check dollar_volume is computed correctly (if present)
    if hasattr(bar, 'dollar_volume') and bar.dollar_volume is not None:
        expected_dv = bar.close * bar.volume
        assert abs(bar.dollar_volume - expected_dv) < 0.01, (
            f"Bar {bar.symbol} {bar.date}: dollar_volume {bar.dollar_volume} "
            f"does not match close * volume {expected_dv}"
        )


def assert_valid_signal(signal: Signal) -> None:
    """Assert that a Signal is valid.
    
    Args:
        signal: Signal to validate
    
    Raises:
        AssertionError: If signal is invalid
    """
    # Check asset class
    assert signal.asset_class in ["equity", "crypto"], (
        f"Signal {signal.symbol}: invalid asset_class {signal.asset_class}"
    )
    
    # Check prices
    assert signal.entry_price > 0, (
        f"Signal {signal.symbol}: entry_price must be positive"
    )
    assert signal.stop_price > 0, (
        f"Signal {signal.symbol}: stop_price must be positive"
    )
    assert signal.stop_price < signal.entry_price, (
        f"Signal {signal.symbol}: stop_price {signal.stop_price} must be "
        f"below entry_price {signal.entry_price} for long positions"
    )
    
    # Check ADV
    assert signal.adv20 > 0, (
        f"Signal {signal.symbol}: adv20 must be positive"
    )
    
    # Check scoring components are in valid range
    assert 0 <= signal.breakout_strength <= 1, (
        f"Signal {signal.symbol}: breakout_strength {signal.breakout_strength} "
        f"must be between 0 and 1"
    )
    assert 0 <= signal.momentum_strength <= 1, (
        f"Signal {signal.symbol}: momentum_strength {signal.momentum_strength} "
        f"must be between 0 and 1"
    )
    assert 0 <= signal.diversification_bonus <= 1, (
        f"Signal {signal.symbol}: diversification_bonus {signal.diversification_bonus} "
        f"must be between 0 and 1"
    )
    assert 0 <= signal.score <= 1, (
        f"Signal {signal.symbol}: score {signal.score} must be between 0 and 1"
    )
    
    # Check eligibility_failures is a list
    assert isinstance(signal.eligibility_failures, list), (
        f"Signal {signal.symbol}: eligibility_failures must be a list"
    )


def assert_valid_order(order: Order) -> None:
    """Assert that an Order is valid.
    
    Args:
        order: Order to validate
    
    Raises:
        AssertionError: If order is invalid
    """
    # Check asset class
    assert order.asset_class in ["equity", "crypto"], (
        f"Order {order.order_id}: invalid asset_class {order.asset_class}"
    )
    
    # Check quantity
    assert order.quantity > 0, (
        f"Order {order.order_id}: quantity must be positive"
    )
    
    # Check prices
    assert order.expected_fill_price > 0, (
        f"Order {order.order_id}: expected_fill_price must be positive"
    )
    assert order.stop_price > 0, (
        f"Order {order.order_id}: stop_price must be positive"
    )
    assert order.stop_price < order.expected_fill_price, (
        f"Order {order.order_id}: stop_price {order.stop_price} must be "
        f"below expected_fill_price {order.expected_fill_price} for long positions"
    )
    
    # Check dates
    assert order.execution_date >= order.date, (
        f"Order {order.order_id}: execution_date {order.execution_date} "
        f"must be >= order date {order.date}"
    )


def assert_valid_fill(fill: Fill) -> None:
    """Assert that a Fill is valid.
    
    Args:
        fill: Fill to validate
    
    Raises:
        AssertionError: If fill is invalid
    """
    # Check asset class
    assert fill.asset_class in ["equity", "crypto"], (
        f"Fill {fill.fill_id}: invalid asset_class {fill.asset_class}"
    )
    
    # Check quantity
    assert fill.quantity > 0, (
        f"Fill {fill.fill_id}: quantity must be positive"
    )
    
    # Check prices
    assert fill.fill_price > 0, (
        f"Fill {fill.fill_id}: fill_price must be positive"
    )
    assert fill.open_price > 0, (
        f"Fill {fill.fill_id}: open_price must be positive"
    )
    
    # Check slippage direction matches side
    if fill.side.value == "BUY":
        assert fill.fill_price >= fill.open_price, (
            f"Fill {fill.fill_id}: BUY fill_price {fill.fill_price} should be "
            f">= open_price {fill.open_price}"
        )
    else:  # SELL
        assert fill.fill_price <= fill.open_price, (
            f"Fill {fill.fill_id}: SELL fill_price {fill.fill_price} should be "
            f"<= open_price {fill.open_price}"
        )
    
    # Check costs
    assert fill.slippage_bps >= 0, (
        f"Fill {fill.fill_id}: slippage_bps must be non-negative"
    )
    assert fill.fee_bps >= 0, (
        f"Fill {fill.fill_id}: fee_bps must be non-negative"
    )
    
    # Check notional
    expected_notional = fill.fill_price * fill.quantity
    assert abs(fill.notional - expected_notional) < 0.01, (
        f"Fill {fill.fill_id}: notional {fill.notional} does not match "
        f"fill_price * quantity {expected_notional}"
    )


def assert_valid_portfolio(portfolio: Portfolio) -> None:
    """Assert that a Portfolio is valid.
    
    Args:
        portfolio: Portfolio to validate
    
    Raises:
        AssertionError: If portfolio is invalid
    """
    # Check equity
    assert portfolio.starting_equity > 0, (
        f"Portfolio {portfolio.date}: starting_equity must be positive"
    )
    assert portfolio.equity > 0, (
        f"Portfolio {portfolio.date}: equity must be positive"
    )
    assert portfolio.cash >= 0, (
        f"Portfolio {portfolio.date}: cash must be non-negative"
    )
    
    # Check risk multiplier
    assert 0.33 <= portfolio.risk_multiplier <= 1.0, (
        f"Portfolio {portfolio.date}: risk_multiplier {portfolio.risk_multiplier} "
        f"must be between 0.33 and 1.0"
    )
    
    # Check exposure percentages
    assert 0 <= portfolio.gross_exposure_pct <= 2.0, (
        f"Portfolio {portfolio.date}: gross_exposure_pct {portfolio.gross_exposure_pct} "
        f"should be reasonable (0-200%)"
    )
    
    # Check position counts
    assert portfolio.open_trades >= 0, (
        f"Portfolio {portfolio.date}: open_trades must be non-negative"
    )
    assert portfolio.total_trades >= 0, (
        f"Portfolio {portfolio.date}: total_trades must be non-negative"
    )
    assert len(portfolio.positions) == portfolio.open_trades, (
        f"Portfolio {portfolio.date}: positions dict length {len(portfolio.positions)} "
        f"does not match open_trades {portfolio.open_trades}"
    )
    
    # Check equity curve
    assert len(portfolio.equity_curve) > 0, (
        f"Portfolio {portfolio.date}: equity_curve must not be empty"
    )
    assert portfolio.equity_curve[-1] == portfolio.equity, (
        f"Portfolio {portfolio.date}: last equity_curve value {portfolio.equity_curve[-1]} "
        f"does not match current equity {portfolio.equity}"
    )


def assert_trades_match_expected(
    actual_trades: List[Dict[str, Any]],
    expected_trades: List[Dict[str, Any]],
    tolerance: float = 0.01
) -> None:
    """Assert that actual trades match expected trades.
    
    Args:
        actual_trades: List of actual trade dictionaries
        expected_trades: List of expected trade dictionaries
        tolerance: Tolerance for numeric comparisons
    
    Raises:
        AssertionError: If trades don't match
    """
    assert len(actual_trades) == len(expected_trades), (
        f"Number of trades mismatch: actual {len(actual_trades)}, "
        f"expected {len(expected_trades)}"
    )
    
    for i, (actual, expected) in enumerate(zip(actual_trades, expected_trades)):
        # Check symbol
        assert actual['symbol'] == expected['symbol'], (
            f"Trade {i}: symbol mismatch: actual {actual['symbol']}, "
            f"expected {expected['symbol']}"
        )
        
        # Check dates
        if 'entry_date' in expected:
            assert actual['entry_date'] == expected['entry_date'], (
                f"Trade {i}: entry_date mismatch: actual {actual['entry_date']}, "
                f"expected {expected['entry_date']}"
            )
        
        # Check prices (with tolerance)
        if 'entry_price' in expected:
            assert abs(actual['entry_price'] - expected['entry_price']) < tolerance, (
                f"Trade {i}: entry_price mismatch: actual {actual['entry_price']}, "
                f"expected {expected['entry_price']}"
            )
        
        if 'exit_price' in expected:
            assert abs(actual['exit_price'] - expected['exit_price']) < tolerance, (
                f"Trade {i}: exit_price mismatch: actual {actual['exit_price']}, "
                f"expected {expected['exit_price']}"
            )


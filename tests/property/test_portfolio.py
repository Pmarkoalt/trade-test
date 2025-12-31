"""Property-based tests for portfolio using hypothesis."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from trading_system.models.orders import Fill
from trading_system.models.positions import ExitReason, Position
from trading_system.models.signals import BreakoutType, SignalSide
from trading_system.portfolio.portfolio import Portfolio


# Strategies for generating test data
def valid_equity():
    """Generate valid equity values."""
    return st.floats(min_value=1000.0, max_value=10000000.0, allow_nan=False, allow_infinity=False)


def valid_price():
    """Generate valid price values."""
    return st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False)


def valid_quantity():
    """Generate valid quantity values."""
    return st.integers(min_value=1, max_value=100000)


def valid_date():
    """Generate valid dates."""
    return st.datetimes(min_value=pd.Timestamp("2020-01-01"), max_value=pd.Timestamp("2025-12-31")).map(
        lambda x: pd.Timestamp(x)
    )


class TestPortfolioProperties:
    """Property-based tests for portfolio operations."""

    @given(valid_equity(), valid_date())
    @settings(max_examples=50, deadline=5000)
    def test_portfolio_initialization(self, equity, date):
        """Property: Portfolio initializes with correct equity."""
        portfolio = Portfolio(date=date, starting_equity=equity, cash=equity, equity=equity)
        assert portfolio.equity == equity
        assert portfolio.cash == equity
        assert portfolio.starting_equity == equity
        assert portfolio.equity_curve[0] == equity

    @given(valid_equity(), valid_date(), valid_price(), valid_quantity())
    @settings(max_examples=50, deadline=5000)
    def test_portfolio_equity_always_positive(self, equity, date, price, quantity):
        """Property: Portfolio equity is always positive after operations."""
        portfolio = Portfolio(date=date, starting_equity=equity, cash=equity, equity=equity)

        # Create a fill
        notional = price * quantity
        assume(notional <= equity * 0.9)  # Don't exceed available cash

        fill = Fill(
            fill_id="test_fill",
            order_id="test_order",
            symbol="TEST",
            asset_class="equity",
            date=date,
            side=SignalSide.BUY,
            quantity=quantity,
            fill_price=price,
            open_price=price,
            slippage_bps=10.0,
            fee_bps=5.0,
            total_cost=notional * 1.0015,  # Include fees
            vol_mult=1.0,
            size_penalty=1.0,
            weekend_penalty=1.0,
            stress_mult=1.0,
            notional=notional,
        )

        # Process fill
        position = portfolio.process_fill(
            fill=fill, stop_price=price * 0.95, atr_mult=2.5, triggered_on=BreakoutType.FAST_20D, adv20_at_entry=notional
        )

        # Equity should still be positive
        assert portfolio.equity > 0
        assert portfolio.cash >= 0

    @given(
        valid_equity(),
        valid_date(),
        st.lists(
            st.tuples(
                st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"))),
                valid_price(),
                valid_quantity(),
            ),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=30, deadline=5000)
    def test_portfolio_exposure_limits(self, equity, date, positions_data):
        """Property: Portfolio respects exposure limits."""
        portfolio = Portfolio(date=date, starting_equity=equity, cash=equity, equity=equity)

        # Add positions
        for symbol, price, quantity in positions_data:
            notional = price * quantity
            assume(notional <= equity * 0.15)  # Per-position limit
            assume(portfolio.cash >= notional * 1.0015)  # Have enough cash

            fill = Fill(
                fill_id=f"fill_{symbol}",
                order_id=f"order_{symbol}",
                symbol=symbol,
                asset_class="equity",
                date=date,
                side=SignalSide.BUY,
                quantity=quantity,
                fill_price=price,
                open_price=price,
                slippage_bps=10.0,
                fee_bps=5.0,
                total_cost=notional * 1.0015,
                vol_mult=1.0,
                size_penalty=1.0,
                weekend_penalty=1.0,
                stress_mult=1.0,
                notional=notional,
            )

            portfolio.process_fill(
                fill=fill, stop_price=price * 0.95, atr_mult=2.5, triggered_on=BreakoutType.FAST_20D, adv20_at_entry=notional
            )

        # Update equity to calculate exposure metrics
        current_prices = {symbol: price for symbol, price, _ in positions_data}
        portfolio.update_equity(current_prices)

        # Check exposure limits
        if portfolio.gross_exposure_pct is not None:
            assert portfolio.gross_exposure_pct <= 0.80  # Max 80% exposure

        # Check per-position limits
        for symbol, exposure_pct in portfolio.per_position_exposure.items():
            assert exposure_pct <= 0.15  # Max 15% per position

    @given(
        valid_equity(),
        valid_date(),
        valid_price(),
        valid_quantity(),
        st.floats(min_value=0.5, max_value=2.0),  # Price multiplier
    )
    @settings(max_examples=50, deadline=5000)
    def test_portfolio_equity_updates_with_prices(self, equity, date, entry_price, quantity, price_mult):
        """Property: Portfolio equity updates correctly with price changes."""
        portfolio = Portfolio(date=date, starting_equity=equity, cash=equity, equity=equity)

        notional = entry_price * quantity
        assume(notional <= equity * 0.9)

        fill = Fill(
            fill_id="test_fill",
            order_id="test_order",
            symbol="TEST",
            asset_class="equity",
            date=date,
            side=SignalSide.BUY,
            quantity=quantity,
            fill_price=entry_price,
            open_price=entry_price,
            slippage_bps=10.0,
            fee_bps=5.0,
            total_cost=notional * 1.0015,
            vol_mult=1.0,
            size_penalty=1.0,
            weekend_penalty=1.0,
            stress_mult=1.0,
            notional=notional,
        )

        portfolio.process_fill(
            fill=fill, stop_price=entry_price * 0.95, atr_mult=2.5, triggered_on=BreakoutType.FAST_20D, adv20_at_entry=notional
        )

        initial_equity = portfolio.equity

        # Update with new price
        new_price = entry_price * price_mult
        current_prices = {"TEST": new_price}
        portfolio.update_equity(current_prices)

        # Equity should change based on price change
        if price_mult > 1.0:
            assert portfolio.equity >= initial_equity
        elif price_mult < 1.0:
            assert portfolio.equity <= initial_equity

    @given(valid_equity(), valid_date(), st.integers(min_value=0, max_value=20))
    @settings(max_examples=50, deadline=5000)
    def test_portfolio_open_trades_matches_positions(self, equity, date, num_positions):
        """Property: open_trades count matches number of positions."""
        portfolio = Portfolio(date=date, starting_equity=equity, cash=equity, equity=equity)

        # Add positions
        for i in range(num_positions):
            symbol = f"SYM{i}"
            price = 100.0
            quantity = 10
            notional = price * quantity

            if notional * 1.0015 > portfolio.cash:
                break  # Out of cash

            fill = Fill(
                fill_id=f"fill_{i}",
                order_id=f"order_{i}",
                symbol=symbol,
                asset_class="equity",
                date=date,
                side=SignalSide.BUY,
                quantity=quantity,
                fill_price=price,
                open_price=price,
                slippage_bps=10.0,
                fee_bps=5.0,
                total_cost=notional * 1.0015,
                vol_mult=1.0,
                size_penalty=1.0,
                weekend_penalty=1.0,
                stress_mult=1.0,
                notional=notional,
            )

            portfolio.process_fill(
                fill=fill, stop_price=price * 0.95, atr_mult=2.5, triggered_on=BreakoutType.FAST_20D, adv20_at_entry=notional
            )

        assert portfolio.open_trades == len(portfolio.positions)

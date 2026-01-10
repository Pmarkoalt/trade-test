"""Tests for manual trade tracking and storage."""

import uuid
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from trading_system.models.positions import PositionSide
from trading_system.storage.manual_trades import ManualTrade, ManualTradeDatabase


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_manual_trades.db"
        yield ManualTradeDatabase(db_path=db_path)


def test_create_manual_trade(temp_db):
    """Test creating a manual trade."""
    trade = ManualTrade(
        trade_id=str(uuid.uuid4()),
        symbol="AAPL",
        asset_class="equity",
        side=PositionSide.LONG,
        entry_date=datetime(2024, 1, 15, 10, 0, 0),
        entry_price=150.0,
        quantity=100,
        stop_price=145.0,
        initial_stop_price=145.0,
        notes="Test trade",
        tags="test,equity",
    )

    trade_id = temp_db.create_trade(trade)
    assert trade_id == trade.trade_id

    # Retrieve and verify
    retrieved = temp_db.get_trade(trade_id)
    assert retrieved is not None
    assert retrieved.symbol == "AAPL"
    assert retrieved.side == PositionSide.LONG
    assert retrieved.quantity == 100
    assert retrieved.entry_price == 150.0
    assert retrieved.stop_price == 145.0
    assert retrieved.is_open()


def test_close_manual_trade(temp_db):
    """Test closing a manual trade."""
    trade = ManualTrade(
        trade_id=str(uuid.uuid4()),
        symbol="TSLA",
        asset_class="equity",
        side=PositionSide.LONG,
        entry_date=datetime(2024, 1, 15),
        entry_price=200.0,
        quantity=50,
        stop_price=190.0,
        initial_stop_price=190.0,
    )

    trade_id = temp_db.create_trade(trade)

    # Close the trade
    exit_date = datetime(2024, 1, 20)
    exit_price = 220.0
    temp_db.close_trade(trade_id, exit_date, exit_price, "profit_target")

    # Verify closure
    closed_trade = temp_db.get_trade(trade_id)
    assert closed_trade is not None
    assert not closed_trade.is_open()
    assert closed_trade.exit_price == 220.0
    assert closed_trade.exit_reason == "profit_target"
    assert closed_trade.realized_pnl == (220.0 - 200.0) * 50  # $1000


def test_update_manual_trade(temp_db):
    """Test updating a manual trade."""
    trade = ManualTrade(
        trade_id=str(uuid.uuid4()),
        symbol="BTC",
        asset_class="crypto",
        side=PositionSide.LONG,
        entry_date=datetime(2024, 1, 15),
        entry_price=40000.0,
        quantity=1,
        stop_price=38000.0,
        initial_stop_price=38000.0,
    )

    trade_id = temp_db.create_trade(trade)

    # Update stop price
    trade.stop_price = 39000.0
    trade.notes = "Updated stop"
    temp_db.update_trade(trade)

    # Verify update
    updated = temp_db.get_trade(trade_id)
    assert updated is not None
    assert updated.stop_price == 39000.0
    assert updated.notes == "Updated stop"


def test_get_open_trades(temp_db):
    """Test retrieving open trades."""
    # Create open trades
    for i in range(3):
        trade = ManualTrade(
            trade_id=str(uuid.uuid4()),
            symbol=f"STOCK{i}",
            asset_class="equity",
            side=PositionSide.LONG,
            entry_date=datetime(2024, 1, 15),
            entry_price=100.0 + i,
            quantity=10,
            stop_price=95.0 + i,
            initial_stop_price=95.0 + i,
        )
        temp_db.create_trade(trade)

    # Create closed trade
    closed_trade = ManualTrade(
        trade_id=str(uuid.uuid4()),
        symbol="CLOSED",
        asset_class="equity",
        side=PositionSide.LONG,
        entry_date=datetime(2024, 1, 10),
        entry_price=100.0,
        quantity=10,
        stop_price=95.0,
        initial_stop_price=95.0,
    )
    closed_id = temp_db.create_trade(closed_trade)
    temp_db.close_trade(closed_id, datetime(2024, 1, 15), 105.0)

    # Get open trades
    open_trades = temp_db.get_open_trades()
    assert len(open_trades) == 3
    assert all(t.is_open() for t in open_trades)


def test_get_closed_trades(temp_db):
    """Test retrieving closed trades."""
    # Create and close trades
    for i in range(2):
        trade = ManualTrade(
            trade_id=str(uuid.uuid4()),
            symbol=f"STOCK{i}",
            asset_class="equity",
            side=PositionSide.LONG,
            entry_date=datetime(2024, 1, 10),
            entry_price=100.0,
            quantity=10,
            stop_price=95.0,
            initial_stop_price=95.0,
        )
        trade_id = temp_db.create_trade(trade)
        temp_db.close_trade(trade_id, datetime(2024, 1, 15 + i), 105.0 + i)

    # Get closed trades
    closed_trades = temp_db.get_closed_trades()
    assert len(closed_trades) == 2
    assert all(not t.is_open() for t in closed_trades)


def test_delete_trade(temp_db):
    """Test deleting a manual trade."""
    trade = ManualTrade(
        trade_id=str(uuid.uuid4()),
        symbol="DELETE_ME",
        asset_class="equity",
        side=PositionSide.LONG,
        entry_date=datetime(2024, 1, 15),
        entry_price=100.0,
        quantity=10,
        stop_price=95.0,
        initial_stop_price=95.0,
    )

    trade_id = temp_db.create_trade(trade)
    assert temp_db.get_trade(trade_id) is not None

    # Delete
    temp_db.delete_trade(trade_id)
    assert temp_db.get_trade(trade_id) is None


def test_manual_trade_to_position(temp_db):
    """Test converting ManualTrade to Position."""
    trade = ManualTrade(
        trade_id=str(uuid.uuid4()),
        symbol="AAPL",
        asset_class="equity",
        side=PositionSide.LONG,
        entry_date=datetime(2024, 1, 15),
        entry_price=150.0,
        quantity=100,
        stop_price=145.0,
        initial_stop_price=145.0,
    )

    position = trade.to_position()
    assert position.symbol == "AAPL"
    assert position.asset_class == "equity"
    assert position.side == PositionSide.LONG
    assert position.quantity == 100
    assert position.entry_price == 150.0
    assert position.stop_price == 145.0
    assert position.strategy_name == "manual"


def test_short_position_pnl(temp_db):
    """Test P&L calculation for short positions."""
    trade = ManualTrade(
        trade_id=str(uuid.uuid4()),
        symbol="SPY",
        asset_class="equity",
        side=PositionSide.SHORT,
        entry_date=datetime(2024, 1, 15),
        entry_price=400.0,
        quantity=10,
        stop_price=410.0,
        initial_stop_price=410.0,
    )

    trade_id = temp_db.create_trade(trade)

    # Close at profit (price went down)
    temp_db.close_trade(trade_id, datetime(2024, 1, 20), 390.0)

    closed_trade = temp_db.get_trade(trade_id)
    assert closed_trade.realized_pnl == (400.0 - 390.0) * 10  # $100 profit


def test_update_unrealized_pnl(temp_db):
    """Test updating unrealized P&L."""
    trade = ManualTrade(
        trade_id=str(uuid.uuid4()),
        symbol="AAPL",
        asset_class="equity",
        side=PositionSide.LONG,
        entry_date=datetime(2024, 1, 15),
        entry_price=150.0,
        quantity=100,
        stop_price=145.0,
        initial_stop_price=145.0,
    )

    trade_id = temp_db.create_trade(trade)

    # Update unrealized P&L with current price
    current_price = 155.0
    temp_db.update_unrealized_pnl(trade_id, current_price)

    updated_trade = temp_db.get_trade(trade_id)
    assert updated_trade.unrealized_pnl == (155.0 - 150.0) * 100  # $500

"""Tests for unified positions view."""

import uuid
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest

from trading_system.models.positions import Position, PositionSide
from trading_system.models.signals import BreakoutType
from trading_system.reporting.unified_positions import PositionSource, UnifiedPosition, UnifiedPositionView
from trading_system.storage.manual_trades import ManualTrade, ManualTradeDatabase


@pytest.fixture
def temp_manual_db():
    """Create a temporary manual trades database."""
    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_manual_trades.db"
        yield ManualTradeDatabase(db_path=db_path)


@pytest.fixture
def mock_paper_adapter():
    """Create a mock paper trading adapter."""
    adapter = MagicMock()
    adapter.is_connected.return_value = True
    return adapter


def test_unified_position_view_initialization(temp_manual_db):
    """Test unified position view initialization."""
    view = UnifiedPositionView(manual_db=temp_manual_db)
    
    assert view.manual_db == temp_manual_db
    assert view.paper_adapter is None


def test_get_manual_positions_only(temp_manual_db):
    """Test getting manual positions only."""
    # Create manual trades
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
        temp_manual_db.create_trade(trade)
    
    view = UnifiedPositionView(manual_db=temp_manual_db)
    positions = view.get_all_positions(include_paper=False, include_manual=True, open_only=True)
    
    assert len(positions) == 3
    assert all(p.source == PositionSource.MANUAL for p in positions)
    assert all(p.position.is_open() for p in positions)


def test_get_paper_positions_only(temp_manual_db, mock_paper_adapter):
    """Test getting paper positions only."""
    # Mock paper positions
    mock_positions = {
        "AAPL": Position(
            symbol="AAPL",
            asset_class="equity",
            entry_date=pd.Timestamp("2024-01-15"),
            entry_price=150.0,
            entry_fill_id="fill_123",
            quantity=100,
            side=PositionSide.LONG,
            stop_price=145.0,
            initial_stop_price=145.0,
            hard_stop_atr_mult=2.5,
            entry_slippage_bps=3.0,
            entry_fee_bps=1.0,
            entry_total_cost=60.0,
            triggered_on=BreakoutType.FAST_20D,
            adv20_at_entry=1000000.0,
        ),
        "TSLA": Position(
            symbol="TSLA",
            asset_class="equity",
            entry_date=pd.Timestamp("2024-01-15"),
            entry_price=200.0,
            entry_fill_id="fill_456",
            quantity=50,
            side=PositionSide.LONG,
            stop_price=190.0,
            initial_stop_price=190.0,
            hard_stop_atr_mult=2.5,
            entry_slippage_bps=3.0,
            entry_fee_bps=1.0,
            entry_total_cost=30.0,
            triggered_on=BreakoutType.FAST_20D,
            adv20_at_entry=2000000.0,
        ),
    }
    mock_paper_adapter.get_positions.return_value = mock_positions
    
    view = UnifiedPositionView(manual_db=temp_manual_db, paper_adapter=mock_paper_adapter)
    positions = view.get_all_positions(include_paper=True, include_manual=False, open_only=True)
    
    assert len(positions) == 2
    assert all(p.source == PositionSource.PAPER for p in positions)


def test_get_all_positions_mixed(temp_manual_db, mock_paper_adapter):
    """Test getting positions from multiple sources."""
    # Create manual trades
    for i in range(2):
        trade = ManualTrade(
            trade_id=str(uuid.uuid4()),
            symbol=f"MANUAL{i}",
            asset_class="equity",
            side=PositionSide.LONG,
            entry_date=datetime(2024, 1, 15),
            entry_price=100.0,
            quantity=10,
            stop_price=95.0,
            initial_stop_price=95.0,
        )
        temp_manual_db.create_trade(trade)
    
    # Mock paper positions
    mock_positions = {
        "PAPER1": Position(
            symbol="PAPER1",
            asset_class="equity",
            entry_date=pd.Timestamp("2024-01-15"),
            entry_price=150.0,
            entry_fill_id="fill_123",
            quantity=100,
            side=PositionSide.LONG,
            stop_price=145.0,
            initial_stop_price=145.0,
            hard_stop_atr_mult=2.5,
            entry_slippage_bps=3.0,
            entry_fee_bps=1.0,
            entry_total_cost=60.0,
            triggered_on=BreakoutType.FAST_20D,
            adv20_at_entry=1000000.0,
        ),
    }
    mock_paper_adapter.get_positions.return_value = mock_positions
    
    view = UnifiedPositionView(manual_db=temp_manual_db, paper_adapter=mock_paper_adapter)
    positions = view.get_all_positions(include_paper=True, include_manual=True, open_only=True)
    
    assert len(positions) == 3
    manual_positions = [p for p in positions if p.source == PositionSource.MANUAL]
    paper_positions = [p for p in positions if p.source == PositionSource.PAPER]
    
    assert len(manual_positions) == 2
    assert len(paper_positions) == 1


def test_get_open_positions_by_symbol(temp_manual_db):
    """Test grouping positions by symbol."""
    # Create multiple positions for same symbol
    for i in range(2):
        trade = ManualTrade(
            trade_id=str(uuid.uuid4()),
            symbol="AAPL",
            asset_class="equity",
            side=PositionSide.LONG,
            entry_date=datetime(2024, 1, 15 + i),
            entry_price=150.0 + i,
            quantity=10,
            stop_price=145.0 + i,
            initial_stop_price=145.0 + i,
        )
        temp_manual_db.create_trade(trade)
    
    # Different symbol
    trade = ManualTrade(
        trade_id=str(uuid.uuid4()),
        symbol="TSLA",
        asset_class="equity",
        side=PositionSide.LONG,
        entry_date=datetime(2024, 1, 15),
        entry_price=200.0,
        quantity=10,
        stop_price=190.0,
        initial_stop_price=190.0,
    )
    temp_manual_db.create_trade(trade)
    
    view = UnifiedPositionView(manual_db=temp_manual_db)
    by_symbol = view.get_open_positions_by_symbol()
    
    assert len(by_symbol) == 2
    assert "AAPL" in by_symbol
    assert "TSLA" in by_symbol
    assert len(by_symbol["AAPL"]) == 2
    assert len(by_symbol["TSLA"]) == 1


def test_get_exposure_summary(temp_manual_db):
    """Test calculating exposure summary."""
    # Create long positions
    for i in range(2):
        trade = ManualTrade(
            trade_id=str(uuid.uuid4()),
            symbol=f"LONG{i}",
            asset_class="equity",
            side=PositionSide.LONG,
            entry_date=datetime(2024, 1, 15),
            entry_price=100.0,
            quantity=100,
            stop_price=95.0,
            initial_stop_price=95.0,
        )
        temp_manual_db.create_trade(trade)
    
    # Create short position
    trade = ManualTrade(
        trade_id=str(uuid.uuid4()),
        symbol="SHORT1",
        asset_class="equity",
        side=PositionSide.SHORT,
        entry_date=datetime(2024, 1, 15),
        entry_price=200.0,
        quantity=50,
        stop_price=210.0,
        initial_stop_price=210.0,
    )
    temp_manual_db.create_trade(trade)
    
    # Create crypto position
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
    temp_manual_db.create_trade(trade)
    
    view = UnifiedPositionView(manual_db=temp_manual_db)
    exposure = view.get_exposure_summary()
    
    assert exposure["total_positions"] == 4
    assert exposure["total_long_notional"] == 100.0 * 100 * 2 + 40000.0 * 1  # 60000
    assert exposure["total_short_notional"] == 200.0 * 50  # 10000
    assert exposure["net_exposure"] == 60000.0 - 10000.0  # 50000
    assert exposure["gross_exposure"] == 60000.0 + 10000.0  # 70000
    assert exposure["equity_notional"] == 100.0 * 100 * 2 + 200.0 * 50  # 30000
    assert exposure["crypto_notional"] == 40000.0 * 1  # 40000


def test_get_positions_dataframe(temp_manual_db):
    """Test converting positions to DataFrame."""
    # Create trades
    for i in range(3):
        trade = ManualTrade(
            trade_id=str(uuid.uuid4()),
            symbol=f"STOCK{i}",
            asset_class="equity",
            side=PositionSide.LONG,
            entry_date=datetime(2024, 1, 15 + i),
            entry_price=100.0 + i,
            quantity=10,
            stop_price=95.0 + i,
            initial_stop_price=95.0 + i,
        )
        temp_manual_db.create_trade(trade)
    
    view = UnifiedPositionView(manual_db=temp_manual_db)
    df = view.get_positions_dataframe(open_only=True)
    
    assert len(df) == 3
    assert "symbol" in df.columns
    assert "source" in df.columns
    assert "side" in df.columns
    assert "quantity" in df.columns
    assert all(df["source"] == "manual")


def test_export_to_csv(temp_manual_db):
    """Test exporting positions to CSV."""
    with TemporaryDirectory() as tmpdir:
        # Create trades
        for i in range(2):
            trade = ManualTrade(
                trade_id=str(uuid.uuid4()),
                symbol=f"STOCK{i}",
                asset_class="equity",
                side=PositionSide.LONG,
                entry_date=datetime(2024, 1, 15),
                entry_price=100.0,
                quantity=10,
                stop_price=95.0,
                initial_stop_price=95.0,
            )
            temp_manual_db.create_trade(trade)
        
        view = UnifiedPositionView(manual_db=temp_manual_db)
        output_path = Path(tmpdir) / "positions.csv"
        
        view.export_to_csv(str(output_path), open_only=True)
        
        assert output_path.exists()
        
        # Read back and verify
        df = pd.read_csv(output_path)
        assert len(df) == 2
        assert "symbol" in df.columns


def test_open_only_filter(temp_manual_db):
    """Test filtering for open positions only."""
    # Create open trades
    for i in range(2):
        trade = ManualTrade(
            trade_id=str(uuid.uuid4()),
            symbol=f"OPEN{i}",
            asset_class="equity",
            side=PositionSide.LONG,
            entry_date=datetime(2024, 1, 15),
            entry_price=100.0,
            quantity=10,
            stop_price=95.0,
            initial_stop_price=95.0,
        )
        temp_manual_db.create_trade(trade)
    
    # Create and close a trade
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
    closed_id = temp_manual_db.create_trade(closed_trade)
    temp_manual_db.close_trade(closed_id, datetime(2024, 1, 15), 105.0)
    
    view = UnifiedPositionView(manual_db=temp_manual_db)
    
    # Get all positions
    all_positions = view.get_all_positions(include_manual=True, open_only=False)
    assert len(all_positions) == 3
    
    # Get open only
    open_positions = view.get_all_positions(include_manual=True, open_only=True)
    assert len(open_positions) == 2
    assert all(p.position.is_open() for p in open_positions)


def test_unified_position_to_dict(temp_manual_db):
    """Test converting UnifiedPosition to dictionary."""
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
    
    unified = UnifiedPosition(
        position=trade.to_position(),
        source=PositionSource.MANUAL,
        source_id=trade.trade_id,
    )
    
    data = unified.to_dict()
    
    assert data["source"] == "manual"
    assert data["source_id"] == trade.trade_id
    assert data["symbol"] == "AAPL"
    assert data["side"] == "LONG"
    assert data["quantity"] == 100
    assert data["entry_price"] == 150.0
    assert data["is_open"] is True

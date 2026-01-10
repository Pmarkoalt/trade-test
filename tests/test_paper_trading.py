"""Tests for paper trading execution pipeline."""

import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest

from trading_system.adapters.base_adapter import AccountInfo, AdapterConfig
from trading_system.execution.paper_trading import PaperTradingConfig, PaperTradingRunner
from trading_system.models.orders import Fill, Order, OrderStatus, SignalSide


@pytest.fixture
def mock_adapter():
    """Create a mock broker adapter."""
    adapter = MagicMock()
    adapter.is_connected.return_value = True
    return adapter


@pytest.fixture
def paper_config():
    """Create paper trading configuration."""
    with TemporaryDirectory() as tmpdir:
        adapter_config = AdapterConfig(
            api_key="test_key",
            api_secret="test_secret",
            paper_trading=True,
        )
        yield PaperTradingConfig(
            adapter_config=adapter_config,
            log_dir=Path(tmpdir) / "logs",
            db_path=Path(tmpdir) / "paper_trading.db",
        )


def test_paper_trading_runner_initialization(mock_adapter, paper_config):
    """Test paper trading runner initialization."""
    runner = PaperTradingRunner(config=paper_config, adapter=mock_adapter)

    assert runner.adapter == mock_adapter
    assert runner.config == paper_config
    assert len(runner.pending_orders) == 0
    assert len(runner.completed_orders) == 0


def test_submit_single_order_success(mock_adapter, paper_config):
    """Test successful order submission."""
    runner = PaperTradingRunner(config=paper_config, adapter=mock_adapter)

    # Create test order
    order = Order(
        order_id=str(uuid.uuid4()),
        symbol="AAPL",
        asset_class="equity",
        date=pd.Timestamp("2024-01-15"),
        execution_date=pd.Timestamp("2024-01-16"),
        side=SignalSide.BUY,
        quantity=100,
        signal_date=pd.Timestamp("2024-01-15"),
        expected_fill_price=150.0,
        stop_price=145.0,
    )

    # Mock successful fill
    mock_fill = Fill(
        fill_id=str(uuid.uuid4()),
        order_id=order.order_id,
        symbol="AAPL",
        asset_class="equity",
        date=pd.Timestamp("2024-01-16"),
        side=SignalSide.BUY,
        quantity=100,
        fill_price=150.05,
        open_price=150.0,
        slippage_bps=3.33,
        fee_bps=1.0,
        total_cost=65.0,
        vol_mult=1.0,
        size_penalty=1.0,
        weekend_penalty=1.0,
        stress_mult=1.0,
        notional=15005.0,
    )
    mock_adapter.submit_order.return_value = mock_fill

    # Submit order
    results = runner.submit_orders([order])

    assert len(results) == 1
    assert order.order_id in results

    lifecycle = results[order.order_id]
    assert lifecycle.status == OrderStatus.FILLED
    assert lifecycle.fill == mock_fill
    assert lifecycle.order == order

    # Check completed orders
    assert order.order_id in runner.completed_orders
    assert order.order_id not in runner.pending_orders


def test_submit_order_rejected(mock_adapter, paper_config):
    """Test order rejection due to insufficient funds."""
    runner = PaperTradingRunner(config=paper_config, adapter=mock_adapter)

    order = Order(
        order_id=str(uuid.uuid4()),
        symbol="AAPL",
        asset_class="equity",
        date=pd.Timestamp("2024-01-15"),
        execution_date=pd.Timestamp("2024-01-16"),
        side=SignalSide.BUY,
        quantity=100,
        signal_date=pd.Timestamp("2024-01-15"),
        expected_fill_price=150.0,
        stop_price=145.0,
    )

    # Mock rejection
    mock_adapter.submit_order.side_effect = ValueError("Insufficient funds")

    # Submit order
    results = runner.submit_orders([order])

    lifecycle = results[order.order_id]
    assert lifecycle.status == OrderStatus.REJECTED
    assert "Insufficient funds" in lifecycle.error_message
    assert order.order_id in runner.completed_orders


def test_submit_order_network_error(mock_adapter, paper_config):
    """Test order submission with network error."""
    runner = PaperTradingRunner(config=paper_config, adapter=mock_adapter)

    order = Order(
        order_id=str(uuid.uuid4()),
        symbol="AAPL",
        asset_class="equity",
        date=pd.Timestamp("2024-01-15"),
        execution_date=pd.Timestamp("2024-01-16"),
        side=SignalSide.BUY,
        quantity=100,
        signal_date=pd.Timestamp("2024-01-15"),
        expected_fill_price=150.0,
        stop_price=145.0,
    )

    # Mock network error
    mock_adapter.submit_order.side_effect = ConnectionError("Network timeout")

    # Submit order
    results = runner.submit_orders([order])

    lifecycle = results[order.order_id]
    assert lifecycle.status == OrderStatus.PENDING
    assert "Network timeout" in lifecycle.error_message
    assert order.order_id in runner.pending_orders


def test_retry_pending_orders(mock_adapter, paper_config):
    """Test retrying pending orders."""
    runner = PaperTradingRunner(config=paper_config, adapter=mock_adapter)

    order = Order(
        order_id=str(uuid.uuid4()),
        symbol="AAPL",
        asset_class="equity",
        date=pd.Timestamp("2024-01-15"),
        execution_date=pd.Timestamp("2024-01-16"),
        side=SignalSide.BUY,
        quantity=100,
        signal_date=pd.Timestamp("2024-01-15"),
        expected_fill_price=150.0,
        stop_price=145.0,
    )

    # First attempt fails
    mock_adapter.submit_order.side_effect = ConnectionError("Network timeout")
    runner.submit_orders([order])

    assert order.order_id in runner.pending_orders

    # Second attempt succeeds
    mock_fill = Fill(
        fill_id=str(uuid.uuid4()),
        order_id=order.order_id,
        symbol="AAPL",
        asset_class="equity",
        date=pd.Timestamp("2024-01-16"),
        side=SignalSide.BUY,
        quantity=100,
        fill_price=150.05,
        open_price=150.0,
        slippage_bps=3.33,
        fee_bps=1.0,
        total_cost=65.0,
        vol_mult=1.0,
        size_penalty=1.0,
        weekend_penalty=1.0,
        stress_mult=1.0,
        notional=15005.0,
    )
    mock_adapter.submit_order.side_effect = None
    mock_adapter.submit_order.return_value = mock_fill

    # Retry
    retried = runner.retry_pending_orders()

    assert order.order_id in retried
    assert retried[order.order_id].status == OrderStatus.FILLED
    assert order.order_id not in runner.pending_orders
    assert order.order_id in runner.completed_orders


def test_max_retries_exceeded(mock_adapter, paper_config):
    """Test that orders are rejected after max retries."""
    runner = PaperTradingRunner(config=paper_config, adapter=mock_adapter)

    order = Order(
        order_id=str(uuid.uuid4()),
        symbol="AAPL",
        asset_class="equity",
        date=pd.Timestamp("2024-01-15"),
        execution_date=pd.Timestamp("2024-01-16"),
        side=SignalSide.BUY,
        quantity=100,
        signal_date=pd.Timestamp("2024-01-15"),
        expected_fill_price=150.0,
        stop_price=145.0,
    )

    # Always fail
    mock_adapter.submit_order.side_effect = ConnectionError("Network timeout")

    runner.submit_orders([order])

    # Retry max times
    for _ in range(paper_config.max_retries):
        runner.retry_pending_orders()

    # Should be rejected now
    assert order.order_id not in runner.pending_orders
    assert order.order_id in runner.completed_orders
    assert runner.completed_orders[order.order_id].status == OrderStatus.REJECTED


def test_get_account_info(mock_adapter, paper_config):
    """Test getting account information."""
    runner = PaperTradingRunner(config=paper_config, adapter=mock_adapter)

    mock_account = AccountInfo(
        equity=100000.0,
        cash=50000.0,
        buying_power=100000.0,
        margin_used=0.0,
        broker_account_id="TEST123",
        currency="USD",
    )
    mock_adapter.get_account_info.return_value = mock_account

    account = runner.get_account_info()

    assert account.equity == 100000.0
    assert account.cash == 50000.0
    assert account.broker_account_id == "TEST123"


def test_get_positions(mock_adapter, paper_config):
    """Test getting current positions."""
    runner = PaperTradingRunner(config=paper_config, adapter=mock_adapter)

    mock_positions = {
        "AAPL": Mock(symbol="AAPL", quantity=100, entry_price=150.0),
        "TSLA": Mock(symbol="TSLA", quantity=50, entry_price=200.0),
    }
    mock_adapter.get_positions.return_value = mock_positions

    positions = runner.get_positions()

    assert len(positions) == 2
    assert "AAPL" in positions
    assert "TSLA" in positions


def test_order_summary(mock_adapter, paper_config):
    """Test getting order summary."""
    runner = PaperTradingRunner(config=paper_config, adapter=mock_adapter)

    # Create some orders
    for i in range(3):
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=f"STOCK{i}",
            asset_class="equity",
            date=pd.Timestamp("2024-01-15"),
            execution_date=pd.Timestamp("2024-01-16"),
            side=SignalSide.BUY,
            quantity=100,
            signal_date=pd.Timestamp("2024-01-15"),
            expected_fill_price=100.0,
            stop_price=95.0,
        )

        if i == 0:
            # Filled
            mock_fill = Fill(
                fill_id=str(uuid.uuid4()),
                order_id=order.order_id,
                symbol=order.symbol,
                asset_class="equity",
                date=pd.Timestamp("2024-01-16"),
                side=SignalSide.BUY,
                quantity=100,
                fill_price=100.0,
                open_price=100.0,
                slippage_bps=0.0,
                fee_bps=1.0,
                total_cost=10.0,
                vol_mult=1.0,
                size_penalty=1.0,
                weekend_penalty=1.0,
                stress_mult=1.0,
                notional=10000.0,
            )
            mock_adapter.submit_order.return_value = mock_fill
        elif i == 1:
            # Rejected
            mock_adapter.submit_order.side_effect = ValueError("Invalid order")
        else:
            # Pending
            mock_adapter.submit_order.side_effect = ConnectionError("Network error")

        runner.submit_orders([order])
        mock_adapter.submit_order.side_effect = None

    summary = runner.get_order_summary()

    assert summary["total"] == 3
    assert summary["filled"] == 1
    assert summary["rejected"] == 1
    assert summary["pending"] == 1


def test_daily_order_limit(mock_adapter, paper_config):
    """Test daily order limit enforcement."""
    paper_config.max_orders_per_day = 2
    runner = PaperTradingRunner(config=paper_config, adapter=mock_adapter)

    # Create 3 orders
    orders = []
    for i in range(3):
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=f"STOCK{i}",
            asset_class="equity",
            date=pd.Timestamp("2024-01-15"),
            execution_date=pd.Timestamp("2024-01-16"),
            side=SignalSide.BUY,
            quantity=100,
            signal_date=pd.Timestamp("2024-01-15"),
            expected_fill_price=100.0,
            stop_price=95.0,
        )
        orders.append(order)

    # Should raise error due to limit
    with pytest.raises(ValueError, match="Daily order limit exceeded"):
        runner.submit_orders(orders)

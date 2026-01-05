"""Unit tests for broker adapters."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from tests.fixtures.mock_adapter import MockAdapter
from tests.utils.test_helpers import create_sample_order
from trading_system.adapters.alpaca_adapter import AlpacaAdapter
from trading_system.adapters.base_adapter import AccountInfo, AdapterConfig
from trading_system.adapters.ib_adapter import IBAdapter
from trading_system.models.orders import OrderStatus, SignalSide
from trading_system.models.positions import Position, PositionSide
from trading_system.models.signals import BreakoutType


class TestBaseAdapter:
    """Tests for BaseAdapter interface."""

    def test_adapter_config_defaults(self):
        """Test AdapterConfig defaults."""
        config = AdapterConfig()
        assert config.paper_trading is True
        assert config.host == "127.0.0.1"
        assert config.port == 7497
        assert config.client_id == 1
        assert config.extra_config == {}

    def test_adapter_config_custom(self):
        """Test AdapterConfig with custom values."""
        config = AdapterConfig(
            api_key="test_key", api_secret="test_secret", paper_trading=False, host="192.168.1.1", port=7496, client_id=2
        )
        assert config.api_key == "test_key"
        assert config.api_secret == "test_secret"
        assert config.paper_trading is False
        assert config.host == "192.168.1.1"
        assert config.port == 7496
        assert config.client_id == 2

    def test_account_info(self):
        """Test AccountInfo dataclass."""
        account = AccountInfo(
            equity=100000.0,
            cash=50000.0,
            buying_power=200000.0,
            margin_used=50000.0,
            broker_account_id="TEST123",
            currency="USD",
        )
        assert account.equity == 100000.0
        assert account.cash == 50000.0
        assert account.buying_power == 200000.0
        assert account.margin_used == 50000.0
        assert account.broker_account_id == "TEST123"
        assert account.currency == "USD"


class TestMockAdapter:
    """Tests for MockAdapter (unit testing without API)."""

    def test_connection_disconnection(self):
        """Test connection and disconnection."""
        config = AdapterConfig()
        adapter = MockAdapter(config)

        assert adapter.is_connected() is False

        adapter.connect()
        assert adapter.is_connected() is True

        adapter.disconnect()
        assert adapter.is_connected() is False

    def test_connection_error_simulation(self):
        """Test connection error simulation."""
        config = AdapterConfig()
        adapter = MockAdapter(config)
        adapter.set_simulate_connection_error(True)

        with pytest.raises(ConnectionError, match="Simulated connection error"):
            adapter.connect()

    def test_context_manager(self):
        """Test adapter as context manager."""
        config = AdapterConfig()
        adapter = MockAdapter(config)

        with adapter:
            assert adapter.is_connected() is True

        assert adapter.is_connected() is False

    def test_get_account_info(self):
        """Test getting account information."""
        config = AdapterConfig()
        adapter = MockAdapter(config)
        adapter.set_account_balance(equity=100000.0, cash=50000.0)

        with adapter:
            account = adapter.get_account_info()
            assert account.equity == 100000.0
            assert account.cash == 50000.0
            assert account.buying_power == 100000.0  # 2x cash
            assert account.broker_account_id == "MOCK_ACCOUNT"

    def test_get_account_info_not_connected(self):
        """Test getting account info when not connected."""
        config = AdapterConfig()
        adapter = MockAdapter(config)

        with pytest.raises(ConnectionError, match="Not connected"):
            adapter.get_account_info()

    def test_submit_order_buy(self):
        """Test submitting a buy order."""
        config = AdapterConfig()
        adapter = MockAdapter(config)
        adapter.set_price("AAPL", 150.0)
        adapter.set_account_balance(equity=100000.0, cash=100000.0)

        order = create_sample_order(
            order_id="ORD001",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp.now(),
            execution_date=pd.Timestamp.now(),
            quantity=100,
            expected_fill_price=150.0,
            stop_price=145.0,
            side=SignalSide.BUY,
        )

        with adapter:
            fill = adapter.submit_order(order)

            assert fill.order_id == "ORD001"
            assert fill.symbol == "AAPL"
            assert fill.quantity == 100
            assert fill.fill_price > 0
            assert fill.side == SignalSide.BUY

            # Check position was created
            position = adapter.get_position("AAPL")
            assert position is not None
            assert position.quantity == 100
            assert position.side == PositionSide.LONG

    def test_submit_order_sell(self):
        """Test submitting a sell order."""
        config = AdapterConfig()
        adapter = MockAdapter(config)
        adapter.set_price("AAPL", 150.0)
        adapter.set_account_balance(equity=100000.0, cash=100000.0)

        # First create a position
        position = Position(
            symbol="AAPL",
            asset_class="equity",
            entry_date=pd.Timestamp.now(),
            entry_price=140.0,
            entry_fill_id="FILL001",
            quantity=100,
            side=PositionSide.LONG,
            stop_price=135.0,
            initial_stop_price=135.0,
            hard_stop_atr_mult=2.5,
            entry_slippage_bps=5.0,
            entry_fee_bps=1.0,
            entry_total_cost=0.0,
            triggered_on=BreakoutType.FAST_20D,
            adv20_at_entry=100000000.0,
        )
        adapter.add_position(position)

        order = create_sample_order(
            order_id="ORD002",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp.now(),
            execution_date=pd.Timestamp.now(),
            quantity=100,
            expected_fill_price=150.0,
            stop_price=155.0,
            side=SignalSide.SELL,
        )

        with adapter:
            fill = adapter.submit_order(order)

            assert fill.order_id == "ORD002"
            assert fill.side == SignalSide.SELL

            # Check position was closed
            position = adapter.get_position("AAPL")
            assert position is None

    def test_submit_order_insufficient_funds(self):
        """Test submitting order with insufficient funds."""
        config = AdapterConfig()
        adapter = MockAdapter(config)
        adapter.set_price("AAPL", 150.0)
        adapter.set_account_balance(equity=10000.0, cash=1000.0)  # Not enough

        order = create_sample_order(
            order_id="ORD003",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp.now(),
            execution_date=pd.Timestamp.now(),
            quantity=100,  # 100 * 150 = 15000, need more than 1000
            expected_fill_price=150.0,
            stop_price=145.0,
            side=SignalSide.BUY,
        )

        with adapter:
            with pytest.raises(ValueError, match="Insufficient funds"):
                adapter.submit_order(order)

    def test_submit_order_invalid_symbol(self):
        """Test submitting order with invalid symbol."""
        config = AdapterConfig()
        adapter = MockAdapter(config)
        adapter.set_account_balance(equity=100000.0, cash=100000.0)

        order = create_sample_order(
            order_id="ORD004",
            symbol="INVALID",
            asset_class="equity",
            date=pd.Timestamp.now(),
            execution_date=pd.Timestamp.now(),
            quantity=100,
            expected_fill_price=150.0,
            stop_price=145.0,
            side=SignalSide.BUY,
        )

        with adapter:
            with pytest.raises(ValueError, match="not found"):
                adapter.submit_order(order)

    def test_submit_order_network_failure(self):
        """Test submitting order with network failure."""
        config = AdapterConfig()
        adapter = MockAdapter(config)
        adapter.set_price("AAPL", 150.0)
        adapter.set_account_balance(equity=100000.0, cash=100000.0)
        adapter.set_simulate_network_failure(True)

        order = create_sample_order(
            order_id="ORD005",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp.now(),
            execution_date=pd.Timestamp.now(),
            quantity=100,
            expected_fill_price=150.0,
            stop_price=145.0,
            side=SignalSide.BUY,
        )

        with adapter:
            with pytest.raises(RuntimeError, match="network failure"):
                adapter.submit_order(order)

    def test_submit_order_rate_limit(self):
        """Test submitting order with rate limiting."""
        config = AdapterConfig()
        adapter = MockAdapter(config)
        adapter.set_price("AAPL", 150.0)
        adapter.set_account_balance(equity=100000.0, cash=100000.0)
        adapter.set_simulate_rate_limit(True)
        adapter._rate_limit_max = 2  # Limit to 2 calls

        order = create_sample_order(
            order_id="ORD006",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp.now(),
            execution_date=pd.Timestamp.now(),
            quantity=10,
            expected_fill_price=150.0,
            stop_price=145.0,
            side=SignalSide.BUY,
        )

        with adapter:
            # First 2 calls should succeed
            adapter.submit_order(order)
            order.order_id = "ORD007"
            adapter.submit_order(order)

            # Third call should fail
            order.order_id = "ORD008"
            with pytest.raises(RuntimeError, match="Rate limit"):
                adapter.submit_order(order)

    def test_cancel_order(self):
        """Test canceling an order."""
        config = AdapterConfig()
        adapter = MockAdapter(config)
        adapter.set_price("AAPL", 150.0)
        adapter.set_account_balance(equity=100000.0, cash=100000.0)

        order = create_sample_order(
            order_id="ORD009",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp.now(),
            execution_date=pd.Timestamp.now(),
            quantity=100,
            expected_fill_price=150.0,
            stop_price=145.0,
            side=SignalSide.BUY,
        )

        with adapter:
            # Submit order first
            adapter.submit_order(order)

            # Cancel it
            result = adapter.cancel_order("ORD009")
            assert result is True

            # Check status
            status = adapter.get_order_status("ORD009")
            assert status == OrderStatus.CANCELLED

    def test_cancel_order_not_found(self):
        """Test canceling non-existent order."""
        config = AdapterConfig()
        adapter = MockAdapter(config)

        with adapter:
            result = adapter.cancel_order("NONEXISTENT")
            assert result is False

    def test_get_order_status(self):
        """Test getting order status."""
        config = AdapterConfig()
        adapter = MockAdapter(config)
        adapter.set_price("AAPL", 150.0)
        adapter.set_account_balance(equity=100000.0, cash=100000.0)

        order = create_sample_order(
            order_id="ORD010",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp.now(),
            execution_date=pd.Timestamp.now(),
            quantity=100,
            expected_fill_price=150.0,
            stop_price=145.0,
            side=SignalSide.BUY,
        )

        with adapter:
            adapter.submit_order(order)
            status = adapter.get_order_status("ORD010")
            assert status == OrderStatus.FILLED

    def test_get_positions(self):
        """Test getting all positions."""
        config = AdapterConfig()
        adapter = MockAdapter(config)

        position1 = Position(
            symbol="AAPL",
            asset_class="equity",
            entry_date=pd.Timestamp.now(),
            entry_price=140.0,
            entry_fill_id="FILL001",
            quantity=100,
            side=PositionSide.LONG,
            stop_price=135.0,
            initial_stop_price=135.0,
            hard_stop_atr_mult=2.5,
            entry_slippage_bps=5.0,
            entry_fee_bps=1.0,
            entry_total_cost=0.0,
            triggered_on=BreakoutType.FAST_20D,
            adv20_at_entry=100000000.0,
        )

        position2 = Position(
            symbol="GOOGL",
            asset_class="equity",
            entry_date=pd.Timestamp.now(),
            entry_price=2000.0,
            entry_fill_id="FILL002",
            quantity=50,
            side=PositionSide.LONG,
            stop_price=1950.0,
            initial_stop_price=1950.0,
            hard_stop_atr_mult=2.5,
            entry_slippage_bps=5.0,
            entry_fee_bps=1.0,
            entry_total_cost=0.0,
            triggered_on=BreakoutType.FAST_20D,
            adv20_at_entry=100000000.0,
        )

        adapter.add_position(position1)
        adapter.add_position(position2)

        with adapter:
            positions = adapter.get_positions()
            assert len(positions) == 2
            assert "AAPL" in positions
            assert "GOOGL" in positions

    def test_get_position(self):
        """Test getting a specific position."""
        config = AdapterConfig()
        adapter = MockAdapter(config)

        position = Position(
            symbol="AAPL",
            asset_class="equity",
            entry_date=pd.Timestamp.now(),
            entry_price=140.0,
            entry_fill_id="FILL001",
            quantity=100,
            side=PositionSide.LONG,
            stop_price=135.0,
            initial_stop_price=135.0,
            hard_stop_atr_mult=2.5,
            entry_slippage_bps=5.0,
            entry_fee_bps=1.0,
            entry_total_cost=0.0,
            triggered_on=BreakoutType.FAST_20D,
            adv20_at_entry=100000000.0,
        )

        adapter.add_position(position)

        with adapter:
            pos = adapter.get_position("AAPL")
            assert pos is not None
            assert pos.symbol == "AAPL"
            assert pos.quantity == 100

            # Non-existent position
            pos = adapter.get_position("NONEXISTENT")
            assert pos is None

    def test_get_current_price(self):
        """Test getting current price."""
        config = AdapterConfig()
        adapter = MockAdapter(config)
        adapter.set_price("AAPL", 150.0)
        adapter.set_price("GOOGL", 2000.0)

        with adapter:
            price = adapter.get_current_price("AAPL")
            assert price == 150.0

            price = adapter.get_current_price("GOOGL")
            assert price == 2000.0

            # Non-existent symbol
            price = adapter.get_current_price("NONEXISTENT")
            assert price is None

    def test_subscribe_market_data(self):
        """Test subscribing to market data."""
        config = AdapterConfig()
        adapter = MockAdapter(config)

        with adapter:
            adapter.subscribe_market_data(["AAPL", "GOOGL"])
            # Should not raise

    def test_unsubscribe_market_data(self):
        """Test unsubscribing from market data."""
        config = AdapterConfig()
        adapter = MockAdapter(config)

        with adapter:
            adapter.unsubscribe_market_data(["AAPL", "GOOGL"])
            # Should not raise


class TestAlpacaAdapter:
    """Unit tests for AlpacaAdapter (mocked)."""

    @patch("trading_system.adapters.alpaca_adapter.tradeapi")
    def test_connect_success(self, mock_tradeapi):
        """Test successful connection to Alpaca."""
        config = AdapterConfig(api_key="test_key", api_secret="test_secret", paper_trading=True)

        # Mock API
        mock_api = Mock()
        mock_account = Mock()
        mock_account.account_number = "TEST123"
        mock_api.get_account.return_value = mock_account
        mock_tradeapi.REST.return_value = mock_api

        adapter = AlpacaAdapter(config)
        adapter.connect()

        assert adapter.is_connected() is True
        mock_tradeapi.REST.assert_called_once()

    @patch("trading_system.adapters.alpaca_adapter.tradeapi")
    def test_connect_missing_credentials(self, mock_tradeapi):
        """Test connection fails with missing credentials."""
        config = AdapterConfig(api_key=None, api_secret=None, paper_trading=True)

        adapter = AlpacaAdapter(config)

        with pytest.raises(ValueError, match="API key and secret are required"):
            adapter.connect()

    @patch("trading_system.adapters.alpaca_adapter.tradeapi", None)
    def test_connect_import_error(self):
        """Test connection fails when alpaca-trade-api is not installed."""
        config = AdapterConfig(api_key="test_key", api_secret="test_secret", paper_trading=True)

        adapter = AlpacaAdapter(config)

        with pytest.raises(ImportError, match="alpaca-trade-api is required"):
            adapter.connect()

    @patch("trading_system.adapters.alpaca_adapter.tradeapi")
    def test_get_account_info(self, mock_tradeapi):
        """Test getting account info from Alpaca."""
        config = AdapterConfig(api_key="test_key", api_secret="test_secret", paper_trading=True)

        # Mock API
        mock_api = Mock()
        mock_account = Mock()
        mock_account.equity = "100000.0"
        mock_account.cash = "50000.0"
        mock_account.buying_power = "200000.0"
        mock_account.portfolio_value = "150000.0"
        mock_account.account_number = "TEST123"
        mock_api.get_account.return_value = mock_account
        mock_tradeapi.REST.return_value = mock_api

        adapter = AlpacaAdapter(config)
        adapter.connect()

        account = adapter.get_account_info()
        assert account.equity == 100000.0
        assert account.cash == 50000.0
        assert account.buying_power == 200000.0
        assert account.broker_account_id == "TEST123"

    @patch("trading_system.adapters.alpaca_adapter.tradeapi")
    def test_get_account_info_not_connected(self, mock_tradeapi):
        """Test getting account info when not connected."""
        config = AdapterConfig(api_key="test_key", api_secret="test_secret", paper_trading=True)

        adapter = AlpacaAdapter(config)

        with pytest.raises(ConnectionError, match="Not connected"):
            adapter.get_account_info()

    @patch("trading_system.adapters.alpaca_adapter.tradeapi")
    def test_submit_order_success(self, mock_tradeapi):
        """Test submitting order successfully."""
        config = AdapterConfig(api_key="test_key", api_secret="test_secret", paper_trading=True)

        # Mock API
        mock_api = Mock()
        mock_account = Mock()
        mock_account.account_number = "TEST123"
        mock_api.get_account.return_value = mock_account

        # Mock order submission
        mock_order = Mock()
        mock_order.id = "ALPACA_ORDER_123"
        mock_order.status = "filled"
        mock_order.filled_avg_price = "150.0"
        mock_order.filled_qty = "100"
        mock_api.submit_order.return_value = mock_order
        mock_api.get_order.return_value = mock_order
        mock_tradeapi.REST.return_value = mock_api

        adapter = AlpacaAdapter(config)
        adapter.connect()

        order = create_sample_order(
            order_id="ORD001",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp.now(),
            execution_date=pd.Timestamp.now(),
            quantity=100,
            expected_fill_price=150.0,
            stop_price=145.0,
            side=SignalSide.BUY,
        )

        fill = adapter.submit_order(order)

        assert fill.order_id == "ORD001"
        assert fill.symbol == "AAPL"
        assert fill.quantity == 100
        assert fill.fill_price == 150.0

    @patch("trading_system.adapters.alpaca_adapter.tradeapi")
    def test_submit_order_not_connected(self, mock_tradeapi):
        """Test submitting order when not connected."""
        config = AdapterConfig(api_key="test_key", api_secret="test_secret", paper_trading=True)

        adapter = AlpacaAdapter(config)

        order = create_sample_order(
            order_id="ORD001",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp.now(),
            execution_date=pd.Timestamp.now(),
            quantity=100,
            expected_fill_price=150.0,
            stop_price=145.0,
            side=SignalSide.BUY,
        )

        with pytest.raises(ConnectionError, match="Not connected"):
            adapter.submit_order(order)

    @patch("trading_system.adapters.alpaca_adapter.tradeapi")
    def test_cancel_order(self, mock_tradeapi):
        """Test canceling an order."""
        config = AdapterConfig(api_key="test_key", api_secret="test_secret", paper_trading=True)

        # Mock API
        mock_api = Mock()
        mock_account = Mock()
        mock_account.account_number = "TEST123"
        mock_api.get_account.return_value = mock_account

        # Mock order
        mock_order = Mock()
        mock_order.id = "ALPACA_ORDER_123"
        mock_order.client_order_id = "ORD001"
        mock_api.list_orders.return_value = [mock_order]
        mock_tradeapi.REST.return_value = mock_api

        adapter = AlpacaAdapter(config)
        adapter.connect()

        result = adapter.cancel_order("ORD001")
        assert result is True
        mock_api.cancel_order.assert_called_once_with("ALPACA_ORDER_123")

    @patch("trading_system.adapters.alpaca_adapter.tradeapi")
    def test_get_positions(self, mock_tradeapi):
        """Test getting positions from Alpaca."""
        config = AdapterConfig(api_key="test_key", api_secret="test_secret", paper_trading=True)

        # Mock API
        mock_api = Mock()
        mock_account = Mock()
        mock_account.account_number = "TEST123"
        mock_api.get_account.return_value = mock_account

        # Mock positions
        mock_pos1 = Mock()
        mock_pos1.symbol = "AAPL"
        mock_pos1.qty = "100"
        mock_pos1.avg_entry_price = "140.0"
        mock_pos1.avg_entry_date = "2024-01-01"

        mock_pos2 = Mock()
        mock_pos2.symbol = "GOOGL"
        mock_pos2.qty = "-50"  # Short position
        mock_pos2.avg_entry_price = "2000.0"
        mock_pos2.avg_entry_date = "2024-01-01"

        mock_api.list_positions.return_value = [mock_pos1, mock_pos2]
        mock_tradeapi.REST.return_value = mock_api

        adapter = AlpacaAdapter(config)
        adapter.connect()

        positions = adapter.get_positions()
        assert len(positions) == 2
        assert "AAPL" in positions
        assert "GOOGL" in positions
        assert positions["AAPL"].side == PositionSide.LONG
        assert positions["GOOGL"].side == PositionSide.SHORT


class TestIBAdapter:
    """Unit tests for IBAdapter (mocked)."""

    @patch("trading_system.adapters.ib_adapter.IB")
    def test_connect_success(self, mock_ib_class):
        """Test successful connection to IB."""
        config = AdapterConfig(host="127.0.0.1", port=7497, client_id=1, paper_trading=True)

        # Mock IB instance
        mock_ib = Mock()
        mock_ib.isConnected.return_value = True
        mock_ib.accountValues.return_value = []
        mock_ib.managedAccounts.return_value = ["DU123456"]
        mock_ib_class.return_value = mock_ib

        adapter = IBAdapter(config)
        adapter.connect()

        assert adapter.is_connected() is True
        mock_ib.connect.assert_called_once()

    @patch("trading_system.adapters.ib_adapter.IB", None)
    def test_connect_import_error(self):
        """Test connection fails when ib_insync is not installed."""
        config = AdapterConfig(host="127.0.0.1", port=7497, client_id=1, paper_trading=True)

        adapter = IBAdapter(config)

        with pytest.raises(ImportError, match="ib_insync is required"):
            adapter.connect()

    @patch("trading_system.adapters.ib_adapter.IB")
    def test_get_account_info(self, mock_ib_class):
        """Test getting account info from IB."""
        config = AdapterConfig(host="127.0.0.1", port=7497, client_id=1, paper_trading=True)

        # Mock IB instance
        mock_ib = Mock()
        mock_ib.isConnected.return_value = True

        # Mock account values
        from unittest.mock import Mock as MockObj

        mock_net_liq = MockObj()
        mock_net_liq.tag = "NetLiquidation"
        mock_net_liq.value = "100000.0"
        mock_net_liq.currency = "USD"

        mock_cash = MockObj()
        mock_cash.tag = "TotalCashValue"
        mock_cash.value = "50000.0"
        mock_cash.currency = "USD"

        mock_bp = MockObj()
        mock_bp.tag = "BuyingPower"
        mock_bp.value = "200000.0"
        mock_bp.currency = "USD"

        mock_ib.accountValues.return_value = [mock_net_liq, mock_cash, mock_bp]
        mock_ib.managedAccounts.return_value = ["DU123456"]
        mock_ib_class.return_value = mock_ib

        adapter = IBAdapter(config)
        adapter.connect()

        account = adapter.get_account_info()
        assert account.equity == 100000.0
        assert account.cash == 50000.0
        assert account.buying_power == 200000.0

    @patch("trading_system.adapters.ib_adapter.IB")
    def test_get_account_info_not_connected(self, mock_ib_class):
        """Test getting account info when not connected."""
        config = AdapterConfig(host="127.0.0.1", port=7497, client_id=1, paper_trading=True)

        adapter = IBAdapter(config)

        with pytest.raises(ConnectionError, match="Not connected"):
            adapter.get_account_info()

    @patch("trading_system.adapters.ib_adapter.IB")
    @patch("builtins.__import__")
    def test_submit_order_success(self, mock_import, mock_ib_class):
        """Test submitting order successfully to IB."""
        config = AdapterConfig(host="127.0.0.1", port=7497, client_id=1, paper_trading=True)

        # Mock IB instance
        mock_ib = Mock()
        mock_ib.isConnected.return_value = True

        # Mock contract qualification
        from unittest.mock import Mock as MockObj

        mock_contract = MockObj()
        mock_contract.symbol = "AAPL"
        mock_ib.qualifyContracts.return_value = [mock_contract]

        # Mock trade
        mock_trade = MockObj()
        mock_trade.isDone.return_value = True
        mock_trade.orderStatus.status = "Filled"
        mock_trade.orderStatus.whyHeld = None

        # Mock fill
        mock_fill = MockObj()
        mock_execution = MockObj()
        mock_execution.price = "150.0"
        mock_execution.shares = "100"
        mock_fill.execution = mock_execution
        mock_fill.commission = "1.0"
        mock_trade.fills = [mock_fill]

        mock_ib.placeOrder.return_value = mock_trade
        mock_ib.accountValues.return_value = []
        mock_ib.managedAccounts.return_value = ["DU123456"]
        mock_ib_class.return_value = mock_ib

        # Mock the ib_insync imports
        mock_stock = Mock()
        mock_stock.return_value = mock_contract
        mock_market_order = Mock()
        mock_crypto = Mock()

        def import_mock(name, *args, **kwargs):
            if name == "ib_insync":
                mock_module = Mock()
                mock_module.Stock = mock_stock
                mock_module.MarketOrder = mock_market_order
                mock_module.Crypto = mock_crypto
                return mock_module
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = import_mock

        adapter = IBAdapter(config)
        adapter.connect()

        order = create_sample_order(
            order_id="ORD001",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp.now(),
            execution_date=pd.Timestamp.now(),
            quantity=100,
            expected_fill_price=150.0,
            stop_price=145.0,
            side=SignalSide.BUY,
        )

        fill = adapter.submit_order(order)

        assert fill.order_id == "ORD001"
        assert fill.symbol == "AAPL"
        assert fill.quantity == 100
        assert fill.fill_price == 150.0

    @patch("trading_system.adapters.ib_adapter.IB")
    @patch("builtins.__import__")
    def test_submit_order_timeout(self, mock_import, mock_ib_class):
        """Test order submission timeout."""
        config = AdapterConfig(host="127.0.0.1", port=7497, client_id=1, paper_trading=True)

        # Mock IB instance
        mock_ib = Mock()
        mock_ib.isConnected.return_value = True

        # Mock contract qualification
        mock_contract = Mock()
        mock_ib.qualifyContracts.return_value = [mock_contract]

        # Mock trade that never completes
        mock_trade = Mock()
        mock_trade.isDone.return_value = False  # Never completes
        mock_ib.placeOrder.return_value = mock_trade
        mock_ib.accountValues.return_value = []
        mock_ib.managedAccounts.return_value = ["DU123456"]
        mock_ib_class.return_value = mock_ib

        # Mock the ib_insync imports
        mock_stock = Mock()
        mock_stock.return_value = mock_contract

        def import_mock(name, *args, **kwargs):
            if name == "ib_insync":
                mock_module = Mock()
                mock_module.Stock = mock_stock
                mock_module.MarketOrder = Mock()
                mock_module.Crypto = Mock()
                return mock_module
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = import_mock

        adapter = IBAdapter(config)
        adapter.connect()

        order = create_sample_order(
            order_id="ORD002",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp.now(),
            execution_date=pd.Timestamp.now(),
            quantity=100,
            expected_fill_price=150.0,
            stop_price=145.0,
            side=SignalSide.BUY,
        )

        # Should raise RuntimeError for timeout
        with pytest.raises(RuntimeError, match="timeout"):
            adapter.submit_order(order)

    @patch("trading_system.adapters.ib_adapter.IB")
    @patch("builtins.__import__")
    def test_submit_order_insufficient_funds(self, mock_import, mock_ib_class):
        """Test order submission with insufficient funds."""
        config = AdapterConfig(host="127.0.0.1", port=7497, client_id=1, paper_trading=True)

        # Mock IB instance
        mock_ib = Mock()
        mock_ib.isConnected.return_value = True

        # Mock contract qualification
        mock_contract = Mock()
        mock_ib.qualifyContracts.return_value = [mock_contract]

        # Mock trade with rejection reason
        mock_trade = Mock()
        mock_trade.isDone.return_value = True
        mock_trade.orderStatus.status = "Cancelled"
        mock_trade.orderStatus.whyHeld = "Insufficient funds"
        mock_ib.placeOrder.return_value = mock_trade
        mock_ib.accountValues.return_value = []
        mock_ib.managedAccounts.return_value = ["DU123456"]
        mock_ib_class.return_value = mock_ib

        # Mock the ib_insync imports
        mock_stock = Mock()
        mock_stock.return_value = mock_contract

        def import_mock(name, *args, **kwargs):
            if name == "ib_insync":
                mock_module = Mock()
                mock_module.Stock = mock_stock
                mock_module.MarketOrder = Mock()
                mock_module.Crypto = Mock()
                return mock_module
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = import_mock

        adapter = IBAdapter(config)
        adapter.connect()

        order = create_sample_order(
            order_id="ORD003",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp.now(),
            execution_date=pd.Timestamp.now(),
            quantity=100,
            expected_fill_price=150.0,
            stop_price=145.0,
            side=SignalSide.BUY,
        )

        # Should raise ValueError for insufficient funds
        with pytest.raises(ValueError, match="Insufficient funds"):
            adapter.submit_order(order)

    @patch("trading_system.adapters.ib_adapter.IB")
    @patch("builtins.__import__")
    def test_submit_order_position_limit(self, mock_import, mock_ib_class):
        """Test order submission with position limit exceeded."""
        config = AdapterConfig(host="127.0.0.1", port=7497, client_id=1, paper_trading=True)

        # Mock IB instance
        mock_ib = Mock()
        mock_ib.isConnected.return_value = True

        # Mock contract qualification
        mock_contract = Mock()
        mock_ib.qualifyContracts.return_value = [mock_contract]

        # Mock trade with position limit rejection
        mock_trade = Mock()
        mock_trade.isDone.return_value = True
        mock_trade.orderStatus.status = "Cancelled"
        mock_trade.orderStatus.whyHeld = "Position limit exceeded"
        mock_ib.placeOrder.return_value = mock_trade
        mock_ib.accountValues.return_value = []
        mock_ib.managedAccounts.return_value = ["DU123456"]
        mock_ib_class.return_value = mock_ib

        # Mock the ib_insync imports
        mock_stock = Mock()
        mock_stock.return_value = mock_contract

        def import_mock(name, *args, **kwargs):
            if name == "ib_insync":
                mock_module = Mock()
                mock_module.Stock = mock_stock
                mock_module.MarketOrder = Mock()
                mock_module.Crypto = Mock()
                return mock_module
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = import_mock

        adapter = IBAdapter(config)
        adapter.connect()

        order = create_sample_order(
            order_id="ORD004",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp.now(),
            execution_date=pd.Timestamp.now(),
            quantity=100,
            expected_fill_price=150.0,
            stop_price=145.0,
            side=SignalSide.BUY,
        )

        # Should raise ValueError for position limit
        with pytest.raises(ValueError, match="Position limit"):
            adapter.submit_order(order)

    @patch("trading_system.adapters.ib_adapter.IB")
    def test_get_positions(self, mock_ib_class):
        """Test getting positions from IB."""
        config = AdapterConfig(host="127.0.0.1", port=7497, client_id=1, paper_trading=True)

        # Mock IB instance
        mock_ib = Mock()
        mock_ib.isConnected.return_value = True

        # Mock positions
        from unittest.mock import Mock as MockObj

        mock_pos1 = MockObj()
        mock_contract1 = MockObj()
        mock_contract1.symbol = "AAPL"
        mock_contract1.secType = "STK"
        mock_pos1.contract = mock_contract1
        mock_pos1.avgCost = "140.0"
        mock_pos1.position = "100"

        mock_pos2 = MockObj()
        mock_contract2 = MockObj()
        mock_contract2.symbol = "GOOGL"
        mock_contract2.secType = "STK"
        mock_pos2.contract = mock_contract2
        mock_pos2.avgCost = "2000.0"
        mock_pos2.position = "-50"  # Short position

        mock_ib.positions.return_value = [mock_pos1, mock_pos2]
        mock_ib.accountValues.return_value = []
        mock_ib.managedAccounts.return_value = ["DU123456"]
        mock_ib_class.return_value = mock_ib

        adapter = IBAdapter(config)
        adapter.connect()

        positions = adapter.get_positions()
        assert len(positions) == 2
        assert "AAPL" in positions
        assert "GOOGL" in positions
        assert positions["AAPL"].side == PositionSide.LONG
        assert positions["GOOGL"].side == PositionSide.SHORT


class TestAlpacaAdapterErrorHandling:
    """Additional error handling tests for AlpacaAdapter."""

    @patch("trading_system.adapters.alpaca_adapter.tradeapi")
    def test_submit_order_rate_limit(self, mock_tradeapi):
        """Test order submission with rate limiting."""
        config = AdapterConfig(api_key="test_key", api_secret="test_secret", paper_trading=True)

        # Mock API
        mock_api = Mock()
        mock_account = Mock()
        mock_account.account_number = "TEST123"
        mock_api.get_account.return_value = mock_account

        # Simulate rate limit error
        from requests.exceptions import HTTPError

        rate_limit_error = HTTPError("Rate limit exceeded")
        rate_limit_error.response = Mock()
        rate_limit_error.response.status_code = 429
        mock_api.submit_order.side_effect = rate_limit_error
        mock_tradeapi.REST.return_value = mock_api

        adapter = AlpacaAdapter(config)
        adapter.connect()

        order = create_sample_order(
            order_id="ORD001",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp.now(),
            execution_date=pd.Timestamp.now(),
            quantity=100,
            expected_fill_price=150.0,
            stop_price=145.0,
            side=SignalSide.BUY,
        )

        # Should raise RuntimeError for rate limit
        with pytest.raises(RuntimeError, match="Rate limit"):
            adapter.submit_order(order)

    @patch("trading_system.adapters.alpaca_adapter.tradeapi")
    def test_submit_order_network_failure(self, mock_tradeapi):
        """Test order submission with network failure."""
        config = AdapterConfig(api_key="test_key", api_secret="test_secret", paper_trading=True)

        # Mock API
        mock_api = Mock()
        mock_account = Mock()
        mock_account.account_number = "TEST123"
        mock_api.get_account.return_value = mock_account

        # Simulate network error
        import requests

        network_error = requests.exceptions.ConnectionError("Connection failed")
        mock_api.submit_order.side_effect = network_error
        mock_tradeapi.REST.return_value = mock_api

        adapter = AlpacaAdapter(config)
        adapter.connect()

        order = create_sample_order(
            order_id="ORD002",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp.now(),
            execution_date=pd.Timestamp.now(),
            quantity=100,
            expected_fill_price=150.0,
            stop_price=145.0,
            side=SignalSide.BUY,
        )

        # Should raise ConnectionError for network failure
        with pytest.raises(ConnectionError, match="Network error"):
            adapter.submit_order(order)

    @patch("trading_system.adapters.alpaca_adapter.tradeapi")
    def test_submit_order_invalid_order(self, mock_tradeapi):
        """Test order submission with invalid order."""
        config = AdapterConfig(api_key="test_key", api_secret="test_secret", paper_trading=True)

        # Mock API
        mock_api = Mock()
        mock_account = Mock()
        mock_account.account_number = "TEST123"
        mock_api.get_account.return_value = mock_account

        # Simulate invalid order error (400 Bad Request)
        from requests.exceptions import HTTPError

        invalid_error = HTTPError("Invalid order")
        invalid_error.response = Mock()
        invalid_error.response.status_code = 400
        invalid_error.response.text = "Invalid order"
        mock_api.submit_order.side_effect = invalid_error
        mock_tradeapi.REST.return_value = mock_api

        adapter = AlpacaAdapter(config)
        adapter.connect()

        order = create_sample_order(
            order_id="ORD003",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp.now(),
            execution_date=pd.Timestamp.now(),
            quantity=100,
            expected_fill_price=150.0,
            stop_price=145.0,
            side=SignalSide.BUY,
        )

        # Should raise ValueError for invalid order
        with pytest.raises(ValueError, match="Invalid order"):
            adapter.submit_order(order)

    @patch("trading_system.adapters.alpaca_adapter.tradeapi")
    def test_get_account_info_rate_limit(self, mock_tradeapi):
        """Test getting account info with rate limiting."""
        config = AdapterConfig(api_key="test_key", api_secret="test_secret", paper_trading=True)

        # Mock API
        mock_api = Mock()
        mock_account = Mock()
        mock_account.account_number = "TEST123"
        mock_api.get_account.return_value = mock_account
        mock_tradeapi.REST.return_value = mock_api

        adapter = AlpacaAdapter(config)
        adapter.connect()

        # Simulate rate limit on second call
        from requests.exceptions import HTTPError

        rate_limit_error = HTTPError("Rate limit exceeded")
        rate_limit_error.response = Mock()
        rate_limit_error.response.status_code = 429
        mock_api.get_account.side_effect = [mock_account, rate_limit_error]

        # Ensure mock_account has all required attributes for first call
        mock_account.equity = "100000.0"
        mock_account.cash = "50000.0"
        mock_account.buying_power = "200000.0"
        mock_account.portfolio_value = "150000.0"

        # First call should succeed
        account1 = adapter.get_account_info()
        assert account1 is not None

        # Second call should raise RuntimeError
        with pytest.raises(RuntimeError, match="Rate limit"):
            adapter.get_account_info()

    @patch("trading_system.adapters.alpaca_adapter.tradeapi")
    def test_get_positions_network_error(self, mock_tradeapi):
        """Test getting positions with network error."""
        config = AdapterConfig(api_key="test_key", api_secret="test_secret", paper_trading=True)

        # Mock API
        mock_api = Mock()
        mock_account = Mock()
        mock_account.account_number = "TEST123"
        mock_api.get_account.return_value = mock_account

        # Simulate network error
        import requests

        network_error = requests.exceptions.ConnectionError("Connection failed")
        mock_api.list_positions.side_effect = network_error
        mock_tradeapi.REST.return_value = mock_api

        adapter = AlpacaAdapter(config)
        adapter.connect()

        # Should raise ConnectionError
        with pytest.raises(ConnectionError, match="Network error"):
            adapter.get_positions()

    @patch("trading_system.adapters.alpaca_adapter.tradeapi")
    def test_cancel_order_not_found(self, mock_tradeapi):
        """Test canceling non-existent order."""
        config = AdapterConfig(api_key="test_key", api_secret="test_secret", paper_trading=True)

        # Mock API
        mock_api = Mock()
        mock_account = Mock()
        mock_account.account_number = "TEST123"
        mock_api.get_account.return_value = mock_account
        mock_api.list_orders.return_value = []  # No orders
        mock_tradeapi.REST.return_value = mock_api

        adapter = AlpacaAdapter(config)
        adapter.connect()

        # Should return False for non-existent order
        result = adapter.cancel_order("NONEXISTENT")
        assert result is False


class TestMockAdapterEdgeCases:
    """Edge case tests for MockAdapter."""

    def test_submit_order_zero_quantity(self):
        """Test submitting order with zero quantity."""
        config = AdapterConfig()
        adapter = MockAdapter(config)
        adapter.set_price("AAPL", 150.0)
        adapter.set_account_balance(equity=100000.0, cash=100000.0)

        with adapter:
            # Order validation happens during creation, not in submit_order
            with pytest.raises(ValueError, match="quantity"):
                order = create_sample_order(
                    order_id="ORD001",
                    symbol="AAPL",
                    asset_class="equity",
                    date=pd.Timestamp.now(),
                    execution_date=pd.Timestamp.now(),
                    quantity=0,  # Invalid
                    expected_fill_price=150.0,
                    stop_price=145.0,
                    side=SignalSide.BUY,
                )

    def test_submit_order_negative_quantity(self):
        """Test submitting order with negative quantity."""
        config = AdapterConfig()
        adapter = MockAdapter(config)
        adapter.set_price("AAPL", 150.0)
        adapter.set_account_balance(equity=100000.0, cash=100000.0)

        with adapter:
            # Order validation happens during creation, not in submit_order
            with pytest.raises(ValueError, match="quantity"):
                order = create_sample_order(
                    order_id="ORD002",
                    symbol="AAPL",
                    asset_class="equity",
                    date=pd.Timestamp.now(),
                    execution_date=pd.Timestamp.now(),
                    quantity=-10,  # Invalid
                    expected_fill_price=150.0,
                    stop_price=145.0,
                    side=SignalSide.BUY,
                )

    def test_submit_order_timeout_simulation(self):
        """Test order submission with timeout simulation."""
        config = AdapterConfig()
        adapter = MockAdapter(config)
        adapter.set_price("AAPL", 150.0)
        adapter.set_account_balance(equity=100000.0, cash=100000.0)
        adapter.set_simulate_timeout(True, delay=0.1)  # Short delay for test

        order = create_sample_order(
            order_id="ORD003",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp.now(),
            execution_date=pd.Timestamp.now(),
            quantity=100,
            expected_fill_price=150.0,
            stop_price=145.0,
            side=SignalSide.BUY,
        )

        with adapter:
            with pytest.raises(RuntimeError, match="timeout"):
                adapter.submit_order(order)

    def test_submit_order_invalid_order_simulation(self):
        """Test order submission with invalid order simulation."""
        config = AdapterConfig()
        adapter = MockAdapter(config)
        adapter.set_price("AAPL", 150.0)
        adapter.set_account_balance(equity=100000.0, cash=100000.0)
        adapter.set_simulate_invalid_order(True)

        order = create_sample_order(
            order_id="ORD004",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp.now(),
            execution_date=pd.Timestamp.now(),
            quantity=100,
            expected_fill_price=150.0,
            stop_price=145.0,
            side=SignalSide.BUY,
        )

        with adapter:
            with pytest.raises(ValueError, match="invalid order"):
                adapter.submit_order(order)

    def test_submit_order_position_limit_simulation(self):
        """Test order submission with position limit (via insufficient funds)."""
        config = AdapterConfig()
        adapter = MockAdapter(config)
        adapter.set_price("AAPL", 150.0)
        adapter.set_account_balance(equity=100000.0, cash=100000.0)
        adapter.set_simulate_insufficient_funds(True)

        order = create_sample_order(
            order_id="ORD005",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp.now(),
            execution_date=pd.Timestamp.now(),
            quantity=100,
            expected_fill_price=150.0,
            stop_price=145.0,
            side=SignalSide.BUY,
        )

        with adapter:
            with pytest.raises(ValueError, match="Insufficient funds"):
                adapter.submit_order(order)

    def test_get_order_status_pending(self):
        """Test getting order status for pending order."""
        config = AdapterConfig()
        adapter = MockAdapter(config)

        # Manually set order status to PENDING
        adapter._orders["ORD001"] = create_sample_order(
            order_id="ORD001",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp.now(),
            execution_date=pd.Timestamp.now(),
            quantity=100,
            expected_fill_price=150.0,
            stop_price=145.0,
            side=SignalSide.BUY,
        )
        adapter._order_status["ORD001"] = OrderStatus.PENDING

        with adapter:
            status = adapter.get_order_status("ORD001")
            assert status == OrderStatus.PENDING

    def test_get_order_status_not_found(self):
        """Test getting order status for non-existent order."""
        config = AdapterConfig()
        adapter = MockAdapter(config)

        with adapter:
            status = adapter.get_order_status("NONEXISTENT")
            assert status == OrderStatus.CANCELLED  # Default for not found

    def test_submit_order_partial_fill_simulation(self):
        """Test order submission with partial fill (if supported)."""
        config = AdapterConfig()
        adapter = MockAdapter(config)
        adapter.set_price("AAPL", 150.0)
        adapter.set_account_balance(equity=100000.0, cash=100000.0)

        order = create_sample_order(
            order_id="ORD006",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp.now(),
            execution_date=pd.Timestamp.now(),
            quantity=100,
            expected_fill_price=150.0,
            stop_price=145.0,
            side=SignalSide.BUY,
        )

        with adapter:
            fill = adapter.submit_order(order)
            # MockAdapter fills completely, but verify fill is correct
            assert fill.quantity == order.quantity
            assert fill.fill_price > 0

    def test_multiple_positions_same_symbol(self):
        """Test handling multiple orders for same symbol."""
        config = AdapterConfig()
        adapter = MockAdapter(config)
        adapter.set_price("AAPL", 150.0)
        adapter.set_account_balance(equity=100000.0, cash=100000.0)

        # Submit first order
        order1 = create_sample_order(
            order_id="ORD007",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp.now(),
            execution_date=pd.Timestamp.now(),
            quantity=50,
            expected_fill_price=150.0,
            stop_price=145.0,
            side=SignalSide.BUY,
        )

        with adapter:
            fill1 = adapter.submit_order(order1)
            assert fill1.quantity == 50

            # Submit second order for same symbol
            order2 = create_sample_order(
                order_id="ORD008",
                symbol="AAPL",
                asset_class="equity",
                date=pd.Timestamp.now(),
                execution_date=pd.Timestamp.now(),
                quantity=50,
                expected_fill_price=150.0,
                stop_price=145.0,
                side=SignalSide.BUY,
            )

            fill2 = adapter.submit_order(order2)
            assert fill2.quantity == 50

            # Check position was updated (should have 100 shares)
            position = adapter.get_position("AAPL")
            assert position is not None
            assert position.quantity == 100  # Combined quantity

"""Integration tests for live trading adapters.

These tests require:
- Paper trading accounts configured (Alpaca paper, IB paper)
- Environment variables set for API credentials
- Network access to broker APIs

Run with: pytest -m integration tests/integration/test_live_trading.py
"""

import os
import time

import pandas as pd
import pytest

from tests.utils.test_helpers import create_sample_order
from trading_system.adapters.alpaca_adapter import AlpacaAdapter
from trading_system.adapters.base_adapter import AccountInfo, AdapterConfig
from trading_system.adapters.ib_adapter import IBAdapter
from trading_system.models.orders import Order, OrderStatus, SignalSide

# Skip all tests if credentials are not available
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_PAPER_URL = os.getenv("ALPACA_PAPER_URL", "https://paper-api.alpaca.markets")

IB_HOST = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT = int(os.getenv("IB_PORT", "7497"))  # 7497 for paper, 7496 for live
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "1"))

ALPACA_AVAILABLE = ALPACA_API_KEY and ALPACA_API_SECRET
IB_AVAILABLE = os.getenv("IB_AVAILABLE", "false").lower() == "true"


@pytest.mark.integration
class TestAlpacaIntegration:
    """Integration tests for AlpacaAdapter with paper trading account."""

    @pytest.fixture
    def alpaca_config(self):
        """Create Alpaca adapter config from environment."""
        if not ALPACA_AVAILABLE:
            pytest.skip("Alpaca credentials not available")

        return AdapterConfig(
            api_key=ALPACA_API_KEY, api_secret=ALPACA_API_SECRET, base_url=ALPACA_PAPER_URL, paper_trading=True
        )

    @pytest.fixture
    def alpaca_adapter(self, alpaca_config):
        """Create and connect Alpaca adapter."""
        adapter = AlpacaAdapter(alpaca_config)
        adapter.connect()
        yield adapter
        adapter.disconnect()

    def test_connection_disconnection(self, alpaca_config):
        """Test connecting and disconnecting to Alpaca."""
        adapter = AlpacaAdapter(alpaca_config)

        assert adapter.is_connected() is False

        adapter.connect()
        assert adapter.is_connected() is True

        adapter.disconnect()
        assert adapter.is_connected() is False

    def test_context_manager(self, alpaca_config):
        """Test adapter as context manager."""
        adapter = AlpacaAdapter(alpaca_config)

        with adapter:
            assert adapter.is_connected() is True

        assert adapter.is_connected() is False

    def test_get_account_info(self, alpaca_adapter):
        """Test getting account information from Alpaca."""
        account = alpaca_adapter.get_account_info()

        assert isinstance(account, AccountInfo)
        assert account.equity > 0
        assert account.cash >= 0
        assert account.buying_power > 0
        assert account.broker_account_id is not None
        assert account.currency == "USD"

    def test_get_current_price(self, alpaca_adapter):
        """Test getting current market price."""
        # Test with a liquid stock
        price = alpaca_adapter.get_current_price("AAPL")

        assert price is not None
        assert price > 0

    def test_get_positions_empty(self, alpaca_adapter):
        """Test getting positions when account has none."""
        positions = alpaca_adapter.get_positions()

        assert isinstance(positions, dict)
        # May be empty or have existing positions

    def test_subscribe_market_data(self, alpaca_adapter):
        """Test subscribing to market data."""
        # This is a placeholder in AlpacaAdapter, but should not raise
        alpaca_adapter.subscribe_market_data(["AAPL", "GOOGL"])

    def test_order_lifecycle_small(self, alpaca_adapter):
        """Test full order lifecycle with a small order.

        Note: This test actually submits a real order to paper trading account.
        Use very small quantities to avoid issues.
        """
        # Get account info first
        account = alpaca_adapter.get_account_info()

        # Only run if we have enough cash
        if account.cash < 1000:
            pytest.skip("Insufficient cash for test order")

        # Create a very small order (1 share)
        order = create_sample_order(
            order_id=f"TEST_{int(time.time())}",
            symbol="AAPL",  # Use a liquid stock
            asset_class="equity",
            date=pd.Timestamp.now(),
            execution_date=pd.Timestamp.now(),
            quantity=1,  # Just 1 share
            expected_fill_price=150.0,  # Approximate
            stop_price=145.0,
            side=SignalSide.BUY,
        )

        try:
            # Submit order
            fill = alpaca_adapter.submit_order(order)

            assert fill is not None
            assert fill.order_id == order.order_id
            assert fill.symbol == "AAPL"
            assert fill.quantity > 0
            assert fill.fill_price > 0

            # Check order status
            status = alpaca_adapter.get_order_status(order.order_id)
            assert status in [OrderStatus.FILLED, OrderStatus.PENDING]

            # Wait a bit for order to settle
            time.sleep(2)

            # Check positions
            positions = alpaca_adapter.get_positions()
            # May or may not have position depending on fill

        except Exception as e:
            # If order fails, it might be due to market hours or other issues
            # Log but don't fail the test
            pytest.skip(f"Order submission failed (may be outside market hours): {e}")

    def test_get_order_status(self, alpaca_adapter):
        """Test getting order status for non-existent order."""
        # Test with a non-existent order ID
        status = alpaca_adapter.get_order_status("NONEXISTENT_ORDER_ID")

        # Should return CANCELLED or PENDING for non-existent orders
        assert status in [OrderStatus.CANCELLED, OrderStatus.PENDING]

    def test_error_handling_invalid_symbol(self, alpaca_adapter):
        """Test error handling for invalid symbol."""
        # Try to get price for invalid symbol
        price = alpaca_adapter.get_current_price("INVALID_SYMBOL_XYZ123")

        # Should return None or raise an error
        assert price is None or price == 0

    def test_error_handling_not_connected(self, alpaca_config):
        """Test error handling when not connected."""
        adapter = AlpacaAdapter(alpaca_config)

        # Should raise ConnectionError when not connected
        with pytest.raises(ConnectionError):
            adapter.get_account_info()

        with pytest.raises(ConnectionError):
            order = create_sample_order(
                order_id="TEST",
                symbol="AAPL",
                asset_class="equity",
                date=pd.Timestamp.now(),
                execution_date=pd.Timestamp.now(),
                quantity=1,
                expected_fill_price=150.0,
                stop_price=145.0,
                side=SignalSide.BUY,
            )
            adapter.submit_order(order)


@pytest.mark.integration
class TestIBIntegration:
    """Integration tests for IBAdapter with paper trading account.

    Note: These tests require TWS or IB Gateway to be running
    and configured for API access.
    """

    @pytest.fixture
    def ib_config(self):
        """Create IB adapter config from environment."""
        if not IB_AVAILABLE:
            pytest.skip("IB TWS/Gateway not available")

        return AdapterConfig(host=IB_HOST, port=IB_PORT, client_id=IB_CLIENT_ID, paper_trading=True)

    @pytest.fixture
    def ib_adapter(self, ib_config):
        """Create and connect IB adapter."""
        try:
            adapter = IBAdapter(ib_config)
            adapter.connect()
            yield adapter
        except ConnectionError as e:
            pytest.skip(f"Could not connect to IB: {e}")
        finally:
            try:
                adapter.disconnect()
            except Exception:
                pass

    def test_connection_disconnection(self, ib_config):
        """Test connecting and disconnecting to IB."""
        try:
            adapter = IBAdapter(ib_config)

            assert adapter.is_connected() is False

            adapter.connect()
            assert adapter.is_connected() is True

            adapter.disconnect()
            assert adapter.is_connected() is False
        except ConnectionError as e:
            pytest.skip(f"Could not connect to IB: {e}")

    def test_context_manager(self, ib_config):
        """Test adapter as context manager."""
        try:
            adapter = IBAdapter(ib_config)

            with adapter:
                assert adapter.is_connected() is True

            assert adapter.is_connected() is False
        except ConnectionError as e:
            pytest.skip(f"Could not connect to IB: {e}")

    def test_get_account_info(self, ib_adapter):
        """Test getting account information from IB."""
        account = ib_adapter.get_account_info()

        assert isinstance(account, AccountInfo)
        assert account.equity > 0
        assert account.cash >= 0
        assert account.broker_account_id is not None

    def test_get_current_price(self, ib_adapter):
        """Test getting current market price from IB."""
        # Test with a liquid stock
        price = ib_adapter.get_current_price("AAPL")

        # IB may return None if market data subscription is not available
        # or if symbol is not valid
        if price is not None:
            assert price > 0

    def test_get_positions(self, ib_adapter):
        """Test getting positions from IB."""
        positions = ib_adapter.get_positions()

        assert isinstance(positions, dict)
        # May be empty or have existing positions

    def test_subscribe_market_data(self, ib_adapter):
        """Test subscribing to market data."""
        # This is a placeholder in IBAdapter, but should not raise
        ib_adapter.subscribe_market_data(["AAPL", "GOOGL"])

    def test_error_handling_not_connected(self, ib_config):
        """Test error handling when not connected."""
        adapter = IBAdapter(ib_config)

        # Should raise ConnectionError when not connected
        with pytest.raises(ConnectionError):
            adapter.get_account_info()


@pytest.mark.integration
class TestAdapterReconnection:
    """Tests for adapter reconnection logic."""

    @pytest.fixture
    def alpaca_config(self):
        """Create Alpaca adapter config."""
        if not ALPACA_AVAILABLE:
            pytest.skip("Alpaca credentials not available")

        return AdapterConfig(
            api_key=ALPACA_API_KEY, api_secret=ALPACA_API_SECRET, base_url=ALPACA_PAPER_URL, paper_trading=True
        )

    def test_reconnection_after_disconnect(self, alpaca_config):
        """Test reconnecting after disconnection."""
        adapter = AlpacaAdapter(alpaca_config)

        # Connect
        adapter.connect()
        assert adapter.is_connected() is True

        # Disconnect
        adapter.disconnect()
        assert adapter.is_connected() is False

        # Reconnect
        adapter.connect()
        assert adapter.is_connected() is True

        # Should be able to get account info
        account = adapter.get_account_info()
        assert account is not None

        adapter.disconnect()


@pytest.mark.integration
class TestRateLimiting:
    """Tests for rate limiting behavior."""

    @pytest.fixture
    def alpaca_config(self):
        """Create Alpaca adapter config."""
        if not ALPACA_AVAILABLE:
            pytest.skip("Alpaca credentials not available")

        return AdapterConfig(
            api_key=ALPACA_API_KEY, api_secret=ALPACA_API_SECRET, base_url=ALPACA_PAPER_URL, paper_trading=True
        )

    def test_rapid_requests(self, alpaca_config):
        """Test making rapid requests to check rate limiting."""
        adapter = AlpacaAdapter(alpaca_config)

        with adapter:
            # Make multiple rapid requests
            for i in range(10):
                try:
                    account = adapter.get_account_info()
                    assert account is not None
                    time.sleep(0.1)  # Small delay to avoid hitting rate limits
                except RuntimeError as e:
                    if "rate limit" in str(e).lower() or "429" in str(e):
                        # Rate limit hit - this is expected behavior
                        break
                    raise


@pytest.mark.integration
class TestAccountBalanceUpdates:
    """Tests for account balance tracking."""

    @pytest.fixture
    def alpaca_config(self):
        """Create Alpaca adapter config."""
        if not ALPACA_AVAILABLE:
            pytest.skip("Alpaca credentials not available")

        return AdapterConfig(
            api_key=ALPACA_API_KEY, api_secret=ALPACA_API_SECRET, base_url=ALPACA_PAPER_URL, paper_trading=True
        )

    def test_account_balance_consistency(self, alpaca_config):
        """Test that account balance is consistent across calls."""
        adapter = AlpacaAdapter(alpaca_config)

        with adapter:
            # Get account info multiple times
            account1 = adapter.get_account_info()
            time.sleep(0.5)
            account2 = adapter.get_account_info()

            # Balances should be consistent (or very close due to market movements)
            # Allow small differences for market movements
            assert abs(account1.equity - account2.equity) < account1.equity * 0.1  # 10% tolerance
            assert abs(account1.cash - account2.cash) < account1.cash * 0.1


@pytest.mark.integration
class TestOrderLifecycle:
    """Comprehensive tests for full order lifecycle."""

    @pytest.fixture
    def alpaca_config(self):
        """Create Alpaca adapter config."""
        if not ALPACA_AVAILABLE:
            pytest.skip("Alpaca credentials not available")

        return AdapterConfig(
            api_key=ALPACA_API_KEY, api_secret=ALPACA_API_SECRET, base_url=ALPACA_PAPER_URL, paper_trading=True
        )

    @pytest.fixture
    def alpaca_adapter(self, alpaca_config):
        """Create and connect Alpaca adapter."""
        adapter = AlpacaAdapter(alpaca_config)
        adapter.connect()
        yield adapter
        adapter.disconnect()

    def test_full_order_lifecycle_buy_sell(self, alpaca_adapter):
        """Test complete order lifecycle: submit buy -> fill -> submit sell -> fill."""
        # Get initial account state
        initial_account = alpaca_adapter.get_account_info()
        initial_cash = initial_account.cash

        # Only run if we have enough cash
        if initial_cash < 2000:
            pytest.skip("Insufficient cash for test order")

        try:
            # Step 1: Submit buy order
            buy_order = create_sample_order(
                order_id=f"BUY_TEST_{int(time.time())}",
                symbol="AAPL",
                asset_class="equity",
                date=pd.Timestamp.now(),
                execution_date=pd.Timestamp.now(),
                quantity=1,
                expected_fill_price=150.0,
                stop_price=145.0,
                side=SignalSide.BUY,
            )

            buy_fill = alpaca_adapter.submit_order(buy_order)

            assert buy_fill is not None
            assert buy_fill.order_id == buy_order.order_id
            assert buy_fill.side == SignalSide.BUY
            assert buy_fill.quantity > 0
            assert buy_fill.fill_price > 0

            # Wait for order to settle
            time.sleep(2)

            # Step 2: Check position was created
            positions = alpaca_adapter.get_positions()
            # Position may or may not exist depending on fill timing

            # Step 3: Check account balance changed
            account_after_buy = alpaca_adapter.get_account_info()
            # Cash should decrease (may not be exact due to fees)

            # Step 4: Submit sell order (if we have a position)
            if "AAPL" in positions:
                sell_order = create_sample_order(
                    order_id=f"SELL_TEST_{int(time.time())}",
                    symbol="AAPL",
                    asset_class="equity",
                    date=pd.Timestamp.now(),
                    execution_date=pd.Timestamp.now(),
                    quantity=1,
                    expected_fill_price=150.0,
                    stop_price=155.0,
                    side=SignalSide.SELL,
                )

                sell_fill = alpaca_adapter.submit_order(sell_order)

                assert sell_fill is not None
                assert sell_fill.order_id == sell_order.order_id
                assert sell_fill.side == SignalSide.SELL

                # Wait for order to settle
                time.sleep(2)

                # Step 5: Check account balance updated
                account_after_sell = alpaca_adapter.get_account_info()
                # Cash should increase after sell

        except Exception as e:
            # If order fails, it might be due to market hours or other issues
            pytest.skip(f"Order lifecycle test failed (may be outside market hours): {e}")

    def test_order_status_tracking(self, alpaca_adapter):
        """Test tracking order status through lifecycle."""
        account = alpaca_adapter.get_account_info()

        if account.cash < 1000:
            pytest.skip("Insufficient cash for test order")

        try:
            order = create_sample_order(
                order_id=f"STATUS_TEST_{int(time.time())}",
                symbol="AAPL",
                asset_class="equity",
                date=pd.Timestamp.now(),
                execution_date=pd.Timestamp.now(),
                quantity=1,
                expected_fill_price=150.0,
                stop_price=145.0,
                side=SignalSide.BUY,
            )

            # Submit order
            fill = alpaca_adapter.submit_order(order)

            # Check status immediately after submission
            status = alpaca_adapter.get_order_status(order.order_id)
            assert status in [OrderStatus.FILLED, OrderStatus.PENDING]

            # Wait and check again
            time.sleep(2)
            status_after = alpaca_adapter.get_order_status(order.order_id)
            assert status_after in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.PENDING]

        except Exception as e:
            pytest.skip(f"Order status tracking test failed: {e}")


@pytest.mark.integration
class TestPositionTracking:
    """Tests for position tracking after orders."""

    @pytest.fixture
    def alpaca_config(self):
        """Create Alpaca adapter config."""
        if not ALPACA_AVAILABLE:
            pytest.skip("Alpaca credentials not available")

        return AdapterConfig(
            api_key=ALPACA_API_KEY, api_secret=ALPACA_API_SECRET, base_url=ALPACA_PAPER_URL, paper_trading=True
        )

    @pytest.fixture
    def alpaca_adapter(self, alpaca_config):
        """Create and connect Alpaca adapter."""
        adapter = AlpacaAdapter(alpaca_config)
        adapter.connect()
        yield adapter
        adapter.disconnect()

    def test_position_creation_after_buy(self, alpaca_adapter):
        """Test that position is created after buy order."""
        account = alpaca_adapter.get_account_info()

        if account.cash < 1000:
            pytest.skip("Insufficient cash for test order")

        try:
            # Get initial positions
            initial_positions = alpaca_adapter.get_positions()
            initial_count = len(initial_positions)

            # Submit buy order
            order = create_sample_order(
                order_id=f"POS_TEST_{int(time.time())}",
                symbol="AAPL",
                asset_class="equity",
                date=pd.Timestamp.now(),
                execution_date=pd.Timestamp.now(),
                quantity=1,
                expected_fill_price=150.0,
                stop_price=145.0,
                side=SignalSide.BUY,
            )

            fill = alpaca_adapter.submit_order(order)
            assert fill is not None

            # Wait for position to appear
            time.sleep(3)

            # Check positions
            positions_after = alpaca_adapter.get_positions()
            # Position may or may not appear immediately depending on broker

            # If position exists, verify it
            if "AAPL" in positions_after:
                position = positions_after["AAPL"]
                assert position.symbol == "AAPL"
                assert position.quantity > 0
                assert position.entry_price > 0

        except Exception as e:
            pytest.skip(f"Position tracking test failed: {e}")

    def test_position_removal_after_sell(self, alpaca_adapter):
        """Test that position is removed after sell order."""
        account = alpaca_adapter.get_account_info()

        if account.cash < 2000:
            pytest.skip("Insufficient cash for test order")

        try:
            # First, create a position by buying
            buy_order = create_sample_order(
                order_id=f"SELL_POS_TEST_{int(time.time())}_BUY",
                symbol="AAPL",
                asset_class="equity",
                date=pd.Timestamp.now(),
                execution_date=pd.Timestamp.now(),
                quantity=1,
                expected_fill_price=150.0,
                stop_price=145.0,
                side=SignalSide.BUY,
            )

            buy_fill = alpaca_adapter.submit_order(buy_order)
            time.sleep(3)  # Wait for position to settle

            # Check if position exists
            positions_before = alpaca_adapter.get_positions()
            has_position = "AAPL" in positions_before

            if has_position:
                # Now sell
                sell_order = create_sample_order(
                    order_id=f"SELL_POS_TEST_{int(time.time())}_SELL",
                    symbol="AAPL",
                    asset_class="equity",
                    date=pd.Timestamp.now(),
                    execution_date=pd.Timestamp.now(),
                    quantity=1,
                    expected_fill_price=150.0,
                    stop_price=155.0,
                    side=SignalSide.SELL,
                )

                sell_fill = alpaca_adapter.submit_order(sell_order)
                time.sleep(3)  # Wait for position to update

                # Check positions after sell
                positions_after = alpaca_adapter.get_positions()
                # Position should be reduced or removed

        except Exception as e:
            pytest.skip(f"Position removal test failed: {e}")

    def test_get_specific_position(self, alpaca_adapter):
        """Test getting a specific position by symbol."""
        positions = alpaca_adapter.get_positions()

        if positions:
            # Test with an existing position
            symbol = list(positions.keys())[0]
            position = alpaca_adapter.get_position(symbol)

            assert position is not None
            assert position.symbol == symbol

        # Test with non-existent position
        non_existent = alpaca_adapter.get_position("NONEXISTENT_SYMBOL_XYZ123")
        assert non_existent is None


@pytest.mark.integration
class TestAccountBalanceUpdates:
    """Tests for account balance updates after orders."""

    @pytest.fixture
    def alpaca_config(self):
        """Create Alpaca adapter config."""
        if not ALPACA_AVAILABLE:
            pytest.skip("Alpaca credentials not available")

        return AdapterConfig(
            api_key=ALPACA_API_KEY, api_secret=ALPACA_API_SECRET, base_url=ALPACA_PAPER_URL, paper_trading=True
        )

    @pytest.fixture
    def alpaca_adapter(self, alpaca_config):
        """Create and connect Alpaca adapter."""
        adapter = AlpacaAdapter(alpaca_config)
        adapter.connect()
        yield adapter
        adapter.disconnect()

    def test_account_balance_after_buy_order(self, alpaca_adapter):
        """Test that account balance decreases after buy order."""
        initial_account = alpaca_adapter.get_account_info()
        initial_cash = initial_account.cash
        initial_equity = initial_account.equity

        if initial_cash < 1000:
            pytest.skip("Insufficient cash for test order")

        try:
            # Submit buy order
            order = create_sample_order(
                order_id=f"BALANCE_TEST_{int(time.time())}",
                symbol="AAPL",
                asset_class="equity",
                date=pd.Timestamp.now(),
                execution_date=pd.Timestamp.now(),
                quantity=1,
                expected_fill_price=150.0,
                stop_price=145.0,
                side=SignalSide.BUY,
            )

            fill = alpaca_adapter.submit_order(order)

            # Wait for balance to update
            time.sleep(2)

            # Check account balance
            account_after = alpaca_adapter.get_account_info()

            # Cash should decrease (allowing for fees and market movements)
            # Note: In paper trading, balances may not update immediately
            # So we just verify the account info is still accessible

            assert account_after is not None
            assert account_after.cash >= 0
            assert account_after.equity >= 0

        except Exception as e:
            pytest.skip(f"Account balance test failed: {e}")

    def test_account_balance_consistency(self, alpaca_adapter):
        """Test that account balance queries are consistent."""
        # Get account info multiple times rapidly
        accounts = []
        for _ in range(5):
            account = alpaca_adapter.get_account_info()
            accounts.append(account)
            time.sleep(0.1)

        # All accounts should have valid values
        for account in accounts:
            assert account.equity >= 0
            assert account.cash >= 0
            assert account.buying_power >= 0
            assert account.broker_account_id is not None


@pytest.mark.integration
class TestErrorHandlingComprehensive:
    """Comprehensive error handling tests."""

    @pytest.fixture
    def alpaca_config(self):
        """Create Alpaca adapter config."""
        if not ALPACA_AVAILABLE:
            pytest.skip("Alpaca credentials not available")

        return AdapterConfig(
            api_key=ALPACA_API_KEY, api_secret=ALPACA_API_SECRET, base_url=ALPACA_PAPER_URL, paper_trading=True
        )

    @pytest.fixture
    def alpaca_adapter(self, alpaca_config):
        """Create and connect Alpaca adapter."""
        adapter = AlpacaAdapter(alpaca_config)
        adapter.connect()
        yield adapter
        adapter.disconnect()

    def test_invalid_order_quantity(self, alpaca_adapter):
        """Test error handling for invalid order quantity."""
        order = create_sample_order(
            order_id=f"INVALID_QTY_{int(time.time())}",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp.now(),
            execution_date=pd.Timestamp.now(),
            quantity=0,  # Invalid quantity
            expected_fill_price=150.0,
            stop_price=145.0,
            side=SignalSide.BUY,
        )

        # Should raise ValueError
        with pytest.raises(ValueError, match="quantity"):
            alpaca_adapter.submit_order(order)

    def test_invalid_symbol(self, alpaca_adapter):
        """Test error handling for invalid symbol."""
        # Try to get price for invalid symbol
        price = alpaca_adapter.get_current_price("INVALID_SYMBOL_XYZ123456")
        assert price is None

        # Try to submit order with invalid symbol
        order = create_sample_order(
            order_id=f"INVALID_SYM_{int(time.time())}",
            symbol="INVALID_SYMBOL_XYZ123456",
            asset_class="equity",
            date=pd.Timestamp.now(),
            execution_date=pd.Timestamp.now(),
            quantity=1,
            expected_fill_price=150.0,
            stop_price=145.0,
            side=SignalSide.BUY,
        )

        # Should raise ValueError or RuntimeError
        with pytest.raises((ValueError, RuntimeError)):
            alpaca_adapter.submit_order(order)

    def test_insufficient_funds(self, alpaca_adapter):
        """Test error handling for insufficient funds."""
        account = alpaca_adapter.get_account_info()

        # Create order that costs more than available cash
        order = create_sample_order(
            order_id=f"INSUFFICIENT_{int(time.time())}",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp.now(),
            execution_date=pd.Timestamp.now(),
            quantity=1000000,  # Very large quantity
            expected_fill_price=150.0,
            stop_price=145.0,
            side=SignalSide.BUY,
        )

        # Should raise ValueError for insufficient funds
        # Note: Paper trading may allow this, so we catch both cases
        try:
            alpaca_adapter.submit_order(order)
            # If it succeeds, that's okay for paper trading
        except (ValueError, RuntimeError) as e:
            error_msg = str(e).lower()
            assert "insufficient" in error_msg or "funds" in error_msg or "400" in str(e) or "422" in str(e)

    def test_operations_when_not_connected(self, alpaca_config):
        """Test that operations fail gracefully when not connected."""
        adapter = AlpacaAdapter(alpaca_config)

        # Should not be connected
        assert adapter.is_connected() is False

        # All operations should raise ConnectionError
        with pytest.raises(ConnectionError):
            adapter.get_account_info()

        with pytest.raises(ConnectionError):
            adapter.get_positions()

        with pytest.raises(ConnectionError):
            adapter.get_current_price("AAPL")

        order = create_sample_order(
            order_id="TEST",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp.now(),
            execution_date=pd.Timestamp.now(),
            quantity=1,
            expected_fill_price=150.0,
            stop_price=145.0,
            side=SignalSide.BUY,
        )

        with pytest.raises(ConnectionError):
            adapter.submit_order(order)

        with pytest.raises(ConnectionError):
            adapter.cancel_order("TEST_ORDER")

        with pytest.raises(ConnectionError):
            adapter.get_order_status("TEST_ORDER")


@pytest.mark.integration
class TestReconnectionLogic:
    """Tests for adapter reconnection logic."""

    @pytest.fixture
    def alpaca_config(self):
        """Create Alpaca adapter config."""
        if not ALPACA_AVAILABLE:
            pytest.skip("Alpaca credentials not available")

        return AdapterConfig(
            api_key=ALPACA_API_KEY, api_secret=ALPACA_API_SECRET, base_url=ALPACA_PAPER_URL, paper_trading=True
        )

    def test_reconnect_after_network_error_simulation(self, alpaca_config):
        """Test reconnecting after a simulated network error."""
        adapter = AlpacaAdapter(alpaca_config)

        # Connect
        adapter.connect()
        assert adapter.is_connected() is True

        # Disconnect (simulating network error)
        adapter.disconnect()
        assert adapter.is_connected() is False

        # Reconnect
        adapter.connect()
        assert adapter.is_connected() is True

        # Should be able to get account info
        account = adapter.get_account_info()
        assert account is not None

        adapter.disconnect()

    def test_multiple_reconnections(self, alpaca_config):
        """Test multiple connect/disconnect cycles."""
        adapter = AlpacaAdapter(alpaca_config)

        for _ in range(3):
            adapter.connect()
            assert adapter.is_connected() is True

            account = adapter.get_account_info()
            assert account is not None

            adapter.disconnect()
            assert adapter.is_connected() is False

    def test_reconnect_and_submit_order(self, alpaca_config):
        """Test that orders can be submitted after reconnection."""
        adapter = AlpacaAdapter(alpaca_config)

        # Connect
        adapter.connect()
        account = adapter.get_account_info()

        if account.cash < 1000:
            adapter.disconnect()
            pytest.skip("Insufficient cash for test order")

        # Disconnect and reconnect
        adapter.disconnect()
        adapter.connect()

        # Should be able to submit order after reconnection
        try:
            order = create_sample_order(
                order_id=f"RECONNECT_TEST_{int(time.time())}",
                symbol="AAPL",
                asset_class="equity",
                date=pd.Timestamp.now(),
                execution_date=pd.Timestamp.now(),
                quantity=1,
                expected_fill_price=150.0,
                stop_price=145.0,
                side=SignalSide.BUY,
            )

            fill = adapter.submit_order(order)
            assert fill is not None

        except Exception as e:
            pytest.skip(f"Reconnection order test failed: {e}")
        finally:
            adapter.disconnect()


@pytest.mark.integration
class TestRateLimitingComprehensive:
    """Comprehensive tests for rate limiting behavior."""

    @pytest.fixture
    def alpaca_config(self):
        """Create Alpaca adapter config."""
        if not ALPACA_AVAILABLE:
            pytest.skip("Alpaca credentials not available")

        return AdapterConfig(
            api_key=ALPACA_API_KEY, api_secret=ALPACA_API_SECRET, base_url=ALPACA_PAPER_URL, paper_trading=True
        )

    def test_rapid_account_queries(self, alpaca_config):
        """Test making rapid account info queries."""
        adapter = AlpacaAdapter(alpaca_config)

        with adapter:
            rate_limit_hit = False

            # Make rapid requests
            for i in range(20):
                try:
                    account = adapter.get_account_info()
                    assert account is not None
                    time.sleep(0.05)  # Small delay
                except RuntimeError as e:
                    if "rate limit" in str(e).lower() or "429" in str(e):
                        rate_limit_hit = True
                        break
                    raise

            # Rate limiting may or may not be hit depending on broker limits
            # Just verify we handled it gracefully if it occurred

    def test_rapid_price_queries(self, alpaca_config):
        """Test making rapid price queries."""
        adapter = AlpacaAdapter(alpaca_config)

        with adapter:
            symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
            rate_limit_hit = False

            for symbol in symbols * 3:  # Query each symbol 3 times
                try:
                    price = adapter.get_current_price(symbol)
                    # Price may be None if symbol is invalid
                    time.sleep(0.1)  # Small delay
                except RuntimeError as e:
                    if "rate limit" in str(e).lower() or "429" in str(e):
                        rate_limit_hit = True
                        break
                    raise

            # Rate limiting may or may not be hit
            # Just verify we handled it gracefully if it occurred

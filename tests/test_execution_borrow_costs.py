"""Unit tests for execution/borrow_costs.py."""

import pandas as pd
import pytest

from trading_system.execution.borrow_costs import compute_borrow_cost_bps, compute_borrow_cost_dollars, is_hard_to_borrow


class TestComputeBorrowCostBps:
    """Tests for compute_borrow_cost_bps function."""

    def test_equity_borrow_cost(self):
        """Test borrow cost for equity."""
        cost = compute_borrow_cost_bps("equity")
        assert cost == 5.0  # 5 bps/day
        assert isinstance(cost, float)

    def test_crypto_borrow_cost(self):
        """Test borrow cost for crypto."""
        cost = compute_borrow_cost_bps("crypto")
        assert cost == 1.0  # 1 bps/day
        assert isinstance(cost, float)

    def test_equity_with_symbol(self):
        """Test equity borrow cost with symbol (should still return default)."""
        cost = compute_borrow_cost_bps("equity", symbol="AAPL")
        assert cost == 5.0

    def test_crypto_with_symbol(self):
        """Test crypto borrow cost with symbol (should still return default)."""
        cost = compute_borrow_cost_bps("crypto", symbol="BTC")
        assert cost == 1.0

    def test_equity_with_date(self):
        """Test equity borrow cost with date (should still return default)."""
        date = pd.Timestamp("2024-01-15")
        cost = compute_borrow_cost_bps("equity", date=date)
        assert cost == 5.0

    def test_invalid_asset_class(self):
        """Test that invalid asset class raises ValueError."""
        with pytest.raises(ValueError, match="Invalid asset_class"):
            compute_borrow_cost_bps("invalid")


class TestComputeBorrowCostDollars:
    """Tests for compute_borrow_cost_dollars function."""

    def test_equity_borrow_cost_dollars(self):
        """Test borrow cost in dollars for equity."""
        notional = 100_000.0
        days_held = 5.0

        cost = compute_borrow_cost_dollars(notional, "equity", days_held)

        # 100_000 * (5 / 10000) * 5 = 250.0
        expected = 100_000.0 * (5.0 / 10000.0) * 5.0
        assert cost == expected
        assert cost == 250.0

    def test_crypto_borrow_cost_dollars(self):
        """Test borrow cost in dollars for crypto."""
        notional = 100_000.0
        days_held = 5.0

        cost = compute_borrow_cost_dollars(notional, "crypto", days_held)

        # 100_000 * (1 / 10000) * 5 = 50.0
        expected = 100_000.0 * (1.0 / 10000.0) * 5.0
        assert cost == expected
        assert cost == 50.0

    def test_zero_days_held(self):
        """Test that zero days held returns zero cost."""
        cost = compute_borrow_cost_dollars(100_000.0, "equity", 0.0)
        assert cost == 0.0

    def test_negative_days_held(self):
        """Test that negative days held returns zero cost."""
        cost = compute_borrow_cost_dollars(100_000.0, "equity", -5.0)
        assert cost == 0.0

    def test_large_notional(self):
        """Test borrow cost with large notional."""
        notional = 1_000_000.0
        days_held = 10.0

        cost = compute_borrow_cost_dollars(notional, "equity", days_held)
        expected = 1_000_000.0 * (5.0 / 10000.0) * 10.0
        assert cost == expected
        assert cost == 5000.0

    def test_fractional_days(self):
        """Test borrow cost with fractional days."""
        notional = 100_000.0
        days_held = 2.5

        cost = compute_borrow_cost_dollars(notional, "equity", days_held)
        expected = 100_000.0 * (5.0 / 10000.0) * 2.5
        assert cost == expected
        assert cost == 125.0

    def test_with_symbol(self):
        """Test borrow cost with symbol parameter."""
        cost = compute_borrow_cost_dollars(100_000.0, "equity", 5.0, symbol="AAPL")
        assert cost == 250.0  # Should use default rate


class TestIsHardToBorrow:
    """Tests for is_hard_to_borrow function."""

    def test_crypto_never_hard_to_borrow(self):
        """Test that crypto is never hard to borrow."""
        assert is_hard_to_borrow("BTC", "crypto") is False
        assert is_hard_to_borrow("BTC", "crypto", adv20=1_000_000) is False
        assert is_hard_to_borrow("ETH", "crypto", adv20=100_000) is False

    def test_equity_low_liquidity_hard_to_borrow(self):
        """Test that equity with low ADV20 is hard to borrow."""
        # Below threshold ($5M)
        assert is_hard_to_borrow("XYZ", "equity", adv20=1_000_000) is True
        assert is_hard_to_borrow("XYZ", "equity", adv20=4_999_999) is True

    def test_equity_high_liquidity_not_hard_to_borrow(self):
        """Test that equity with high ADV20 is not hard to borrow."""
        # Above threshold ($5M)
        assert is_hard_to_borrow("AAPL", "equity", adv20=10_000_000) is False
        assert is_hard_to_borrow("AAPL", "equity", adv20=5_000_000) is False
        assert is_hard_to_borrow("AAPL", "equity", adv20=100_000_000) is False

    def test_equity_no_adv20_default(self):
        """Test that equity without ADV20 defaults to not hard-to-borrow."""
        assert is_hard_to_borrow("XYZ", "equity", adv20=None) is False

    def test_equity_exactly_at_threshold(self):
        """Test equity exactly at threshold ($5M)."""
        # At threshold, should not be hard to borrow (>= threshold)
        assert is_hard_to_borrow("XYZ", "equity", adv20=5_000_000) is False

    def test_different_symbols(self):
        """Test with different symbols."""
        assert is_hard_to_borrow("AAPL", "equity", adv20=20_000_000) is False
        assert is_hard_to_borrow("MSFT", "equity", adv20=15_000_000) is False
        assert is_hard_to_borrow("ILLIQUID", "equity", adv20=1_000_000) is True

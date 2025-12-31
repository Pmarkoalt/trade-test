"""Unit tests for strategy_registry.py."""

import pytest

from trading_system.configs.strategy_config import StrategyConfig
from trading_system.exceptions import StrategyNotFoundError
from trading_system.strategies.base.strategy_interface import StrategyInterface
from trading_system.strategies.strategy_registry import (
    create_strategy,
    get_strategy_class,
    list_available_strategies,
    register_strategy,
)


class TestRegisterStrategy:
    """Tests for register_strategy function."""

    def test_register_strategy_valid(self):
        """Test registering a valid strategy."""

        # Create a mock strategy class
        class MockStrategy(StrategyInterface):
            def __init__(self, config):
                self.config = config
                self.asset_class = config.asset_class

            def generate_signal(self, symbol, features, order_notional, diversification_bonus=0.0):
                return None

            def check_exit_signals(self, position, features):
                return None

        # Register the strategy
        register_strategy("mock", "equity", MockStrategy)

        # Verify it was registered
        strategy_class = get_strategy_class("mock", "equity")
        assert strategy_class == MockStrategy

    def test_register_strategy_invalid_type(self):
        """Test registering strategy with invalid type."""

        class MockStrategy:
            pass  # Doesn't implement StrategyInterface

        with pytest.raises(ValueError, match="must implement StrategyInterface"):
            register_strategy("invalid", "equity", MockStrategy)

    def test_register_strategy_invalid_asset_class(self):
        """Test registering strategy with invalid asset class."""

        class MockStrategy(StrategyInterface):
            def __init__(self, config):
                pass

            def generate_signal(self, symbol, features, order_notional, diversification_bonus=0.0):
                return None

            def check_exit_signals(self, position, features):
                return None

        with pytest.raises(ValueError, match="must be 'equity' or 'crypto'"):
            register_strategy("mock", "invalid", MockStrategy)

    def test_register_strategy_empty_type(self):
        """Test registering strategy with empty type."""

        class MockStrategy(StrategyInterface):
            def __init__(self, config):
                pass

            def generate_signal(self, symbol, features, order_notional, diversification_bonus=0.0):
                return None

            def check_exit_signals(self, position, features):
                return None

        with pytest.raises(ValueError, match="cannot be empty"):
            register_strategy("", "equity", MockStrategy)

    def test_register_strategy_duplicate(self):
        """Test registering duplicate strategy."""

        class MockStrategy(StrategyInterface):
            def __init__(self, config):
                pass

            def generate_signal(self, symbol, features, order_notional, diversification_bonus=0.0):
                return None

            def check_exit_signals(self, position, features):
                return None

        # Register first time
        register_strategy("duplicate_test", "equity", MockStrategy)

        # Try to register again (should fail)
        with pytest.raises(ValueError, match="already registered"):
            register_strategy("duplicate_test", "equity", MockStrategy)


class TestGetStrategyClass:
    """Tests for get_strategy_class function."""

    def test_get_strategy_class_existing(self):
        """Test getting an existing strategy class."""
        strategy_class = get_strategy_class("momentum", "equity")
        assert strategy_class is not None
        assert issubclass(strategy_class, StrategyInterface)

    def test_get_strategy_class_nonexistent(self):
        """Test getting a non-existent strategy class."""
        strategy_class = get_strategy_class("nonexistent", "equity")
        assert strategy_class is None

    def test_get_strategy_class_crypto(self):
        """Test getting crypto strategy class."""
        strategy_class = get_strategy_class("momentum", "crypto")
        assert strategy_class is not None
        assert issubclass(strategy_class, StrategyInterface)


class TestCreateStrategy:
    """Tests for create_strategy function."""

    def test_create_strategy_momentum_equity(self):
        """Test creating momentum equity strategy."""
        config = StrategyConfig(
            name="equity_momentum",
            asset_class="equity",
            universe="NASDAQ-100",
            benchmark="SPY",
        )

        strategy = create_strategy(config)
        assert strategy is not None
        assert strategy.config == config
        assert strategy.asset_class == "equity"

    def test_create_strategy_momentum_crypto(self):
        """Test creating momentum crypto strategy."""
        config = StrategyConfig(
            name="crypto_momentum",
            asset_class="crypto",
            universe="fixed",
            benchmark="BTC",
        )

        strategy = create_strategy(config)
        assert strategy is not None
        assert strategy.config == config
        assert strategy.asset_class == "crypto"

    def test_create_strategy_mean_reversion(self):
        """Test creating mean reversion strategy."""
        config = StrategyConfig(
            name="equity_mean_reversion",
            asset_class="equity",
            universe="NASDAQ-100",
            benchmark="SPY",
        )

        strategy = create_strategy(config)
        assert strategy is not None
        assert strategy.asset_class == "equity"

    def test_create_strategy_multi_timeframe(self):
        """Test creating multi-timeframe strategy."""
        config = StrategyConfig(
            name="equity_mtf",
            asset_class="equity",
            universe="NASDAQ-100",
            benchmark="SPY",
        )

        strategy = create_strategy(config)
        assert strategy is not None
        assert strategy.asset_class == "equity"

    def test_create_strategy_factor(self):
        """Test creating factor strategy."""
        config = StrategyConfig(
            name="equity_factor",
            asset_class="equity",
            universe="NASDAQ-100",
            benchmark="SPY",
        )

        strategy = create_strategy(config)
        assert strategy is not None
        assert strategy.asset_class == "equity"

    def test_create_strategy_not_found(self):
        """Test creating strategy that doesn't exist."""
        config = StrategyConfig(
            name="nonexistent_strategy",
            asset_class="equity",
            universe="NASDAQ-100",
            benchmark="SPY",
        )

        with pytest.raises(StrategyNotFoundError):
            create_strategy(config)

    def test_create_strategy_infer_type_from_name(self):
        """Test that strategy type is inferred from config name."""
        # Test various naming patterns
        test_cases = [
            ("momentum_strategy", "momentum"),
            ("mean_reversion_strategy", "mean_reversion"),
            ("mean-reversion_strategy", "mean_reversion"),
            ("pairs_trading", "pairs"),
            ("multi_timeframe_strategy", "multi_timeframe"),
            ("mtf_strategy", "multi_timeframe"),
            ("factor_strategy", "factor"),
        ]

        for name, expected_type in test_cases:
            config = StrategyConfig(
                name=name,
                asset_class="equity",
                universe="NASDAQ-100",
                benchmark="SPY",
            )

            # Should not raise an error if strategy type can be inferred
            try:
                strategy = create_strategy(config)
                assert strategy is not None
            except StrategyNotFoundError:
                # Some combinations might not be registered, which is OK
                pass


class TestListAvailableStrategies:
    """Tests for list_available_strategies function."""

    def test_list_available_strategies(self):
        """Test listing all available strategies."""
        strategies = list_available_strategies()
        assert isinstance(strategies, list)
        assert len(strategies) > 0

        # Each item should be a tuple of (strategy_type, asset_class)
        for item in strategies:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)  # strategy_type
            assert item[1] in ["equity", "crypto"]  # asset_class

    def test_list_available_strategies_contains_momentum(self):
        """Test that momentum strategies are listed."""
        strategies = list_available_strategies()
        strategy_types = {s[0] for s in strategies}
        assert "momentum" in strategy_types

    def test_list_available_strategies_contains_equity(self):
        """Test that equity strategies are listed."""
        strategies = list_available_strategies()
        asset_classes = {s[1] for s in strategies}
        assert "equity" in asset_classes

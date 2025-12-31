"""Unit tests for portfolio optimization module."""

import pytest
import pandas as pd
import numpy as np

from trading_system.portfolio.optimization import (
    PortfolioOptimizer,
    OptimizationResult,
    RebalanceTarget,
    compute_rebalance_targets,
    should_rebalance,
)


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    def test_optimization_result_creation(self):
        """Test creating an OptimizationResult."""
        weights = {"AAPL": 0.5, "MSFT": 0.3, "GOOGL": 0.2}
        result = OptimizationResult(
            weights=weights, expected_return=0.12, volatility=0.15, sharpe_ratio=0.8, method="markowitz"
        )

        assert result.weights == weights
        assert result.expected_return == 0.12
        assert result.volatility == 0.15
        assert result.sharpe_ratio == 0.8
        assert result.method == "markowitz"

    def test_optimization_result_to_array(self):
        """Test converting weights dict to array."""
        weights = {"AAPL": 0.5, "MSFT": 0.3, "GOOGL": 0.2}
        result = OptimizationResult(weights=weights)

        symbols = ["AAPL", "MSFT", "GOOGL"]
        arr = result.to_array(symbols)

        assert len(arr) == 3
        assert arr[0] == 0.5
        assert arr[1] == 0.3
        assert arr[2] == 0.2

    def test_optimization_result_to_array_missing_symbol(self):
        """Test converting weights with missing symbol."""
        weights = {"AAPL": 0.5, "MSFT": 0.3}
        result = OptimizationResult(weights=weights)

        symbols = ["AAPL", "MSFT", "GOOGL"]
        arr = result.to_array(symbols)

        assert len(arr) == 3
        assert arr[0] == 0.5
        assert arr[1] == 0.3
        assert arr[2] == 0.0  # Missing symbol gets 0.0


class TestRebalanceTarget:
    """Tests for RebalanceTarget dataclass."""

    def test_rebalance_target_creation(self):
        """Test creating a RebalanceTarget."""
        target = RebalanceTarget(
            symbol="AAPL",
            target_weight=0.5,
            current_weight=0.3,
            target_notional=50000.0,
            current_notional=30000.0,
            delta_notional=20000.0,
            delta_quantity=200,
        )

        assert target.symbol == "AAPL"
        assert target.target_weight == 0.5
        assert target.current_weight == 0.3
        assert target.delta_notional == 20000.0
        assert target.delta_quantity == 200

    def test_rebalance_target_is_rebalance_needed_true(self):
        """Test is_rebalance_needed when rebalancing is needed."""
        target = RebalanceTarget(
            symbol="AAPL",
            target_weight=0.5,
            current_weight=0.3,  # 20% deviation > 5% threshold
            target_notional=50000.0,
            current_notional=30000.0,
            delta_notional=20000.0,
            delta_quantity=200,
        )

        assert target.is_rebalance_needed is True

    def test_rebalance_target_is_rebalance_needed_false(self):
        """Test is_rebalance_needed when rebalancing is not needed."""
        target = RebalanceTarget(
            symbol="AAPL",
            target_weight=0.5,
            current_weight=0.48,  # 2% deviation < 5% threshold
            target_notional=50000.0,
            current_notional=48000.0,
            delta_notional=2000.0,
            delta_quantity=20,
        )

        assert target.is_rebalance_needed is False


class TestPortfolioOptimizer:
    """Tests for PortfolioOptimizer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = PortfolioOptimizer(risk_free_rate=0.02, optimization_method="markowitz")

    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.risk_free_rate == 0.02
        assert self.optimizer.optimization_method == "markowitz"

    def test_optimize_markowitz_sharpe_maximization(self):
        """Test Markowitz optimization maximizing Sharpe ratio."""
        # Create sample returns data
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        returns_data = pd.DataFrame(
            {
                "AAPL": np.random.normal(0.001, 0.02, 100),
                "MSFT": np.random.normal(0.0008, 0.018, 100),
                "GOOGL": np.random.normal(0.0009, 0.019, 100),
            },
            index=dates,
        )

        result = self.optimizer.optimize_markowitz(returns_data)

        assert isinstance(result, OptimizationResult)
        assert len(result.weights) == 3
        assert "AAPL" in result.weights
        assert "MSFT" in result.weights
        assert "GOOGL" in result.weights

        # Weights should sum to approximately 1.0
        total_weight = sum(result.weights.values())
        assert abs(total_weight - 1.0) < 0.01

        # All weights should be between 0 and 1
        for weight in result.weights.values():
            assert 0.0 <= weight <= 1.0

    def test_optimize_markowitz_target_return(self):
        """Test Markowitz optimization with target return."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        returns_data = pd.DataFrame(
            {
                "AAPL": np.random.normal(0.001, 0.02, 100),
                "MSFT": np.random.normal(0.0008, 0.018, 100),
            },
            index=dates,
        )

        target_return = 0.10  # 10% annual return
        result = self.optimizer.optimize_markowitz(returns_data, target_return=target_return)

        assert isinstance(result, OptimizationResult)
        assert result.expected_return is not None

    def test_optimize_markowitz_long_only(self):
        """Test Markowitz optimization with long-only constraint."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        returns_data = pd.DataFrame(
            {
                "AAPL": np.random.normal(0.001, 0.02, 100),
                "MSFT": np.random.normal(0.0008, 0.018, 100),
            },
            index=dates,
        )

        result = self.optimizer.optimize_markowitz(returns_data, long_only=True)

        # All weights should be >= 0
        for weight in result.weights.values():
            assert weight >= 0.0

    def test_optimize_markowitz_with_bounds(self):
        """Test Markowitz optimization with custom weight bounds."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        returns_data = pd.DataFrame(
            {
                "AAPL": np.random.normal(0.001, 0.02, 100),
                "MSFT": np.random.normal(0.0008, 0.018, 100),
            },
            index=dates,
        )

        result = self.optimizer.optimize_markowitz(returns_data, min_weight=0.1, max_weight=0.6)

        # All weights should be within bounds
        for weight in result.weights.values():
            assert 0.1 <= weight <= 0.6

    def test_optimize_markowitz_empty_data(self):
        """Test optimization with empty returns data."""
        empty_data = pd.DataFrame()

        with pytest.raises(ValueError, match="returns_data cannot be empty"):
            self.optimizer.optimize_markowitz(empty_data)

    def test_optimize_markowitz_no_assets(self):
        """Test optimization with no assets."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        empty_data = pd.DataFrame(index=dates)

        with pytest.raises(ValueError, match="No assets in returns_data"):
            self.optimizer.optimize_markowitz(empty_data)

    def test_optimize_risk_parity(self):
        """Test risk parity optimization."""
        optimizer = PortfolioOptimizer(optimization_method="risk_parity")
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        returns_data = pd.DataFrame(
            {
                "AAPL": np.random.normal(0.001, 0.02, 100),
                "MSFT": np.random.normal(0.0008, 0.018, 100),
            },
            index=dates,
        )

        result = optimizer.optimize_risk_parity(returns_data)

        assert isinstance(result, OptimizationResult)
        assert result.method == "risk_parity"
        assert len(result.weights) == 2

        # Weights should sum to approximately 1.0
        total_weight = sum(result.weights.values())
        assert abs(total_weight - 1.0) < 0.01


class TestComputeRebalanceTargets:
    """Tests for compute_rebalance_targets function."""

    def test_compute_rebalance_targets_basic(self):
        """Test computing rebalance targets."""
        current_weights = {"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3}
        target_weights = {"AAPL": 0.5, "MSFT": 0.3, "GOOGL": 0.2}
        equity = 100000.0
        prices = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 2500.0}

        targets = compute_rebalance_targets(current_weights, target_weights, equity, prices)

        assert len(targets) == 3
        assert all(isinstance(t, RebalanceTarget) for t in targets)

        # Check AAPL target (needs to increase)
        aapl_target = next(t for t in targets if t.symbol == "AAPL")
        assert aapl_target.target_weight == 0.5
        assert aapl_target.current_weight == 0.4
        assert aapl_target.delta_notional > 0  # Should buy more

    def test_compute_rebalance_targets_reduce_position(self):
        """Test rebalance targets when reducing position."""
        current_weights = {"AAPL": 0.6, "MSFT": 0.4}
        target_weights = {"AAPL": 0.4, "MSFT": 0.6}
        equity = 100000.0
        prices = {"AAPL": 150.0, "MSFT": 300.0}

        targets = compute_rebalance_targets(current_weights, target_weights, equity, prices)

        # AAPL should be reduced
        aapl_target = next(t for t in targets if t.symbol == "AAPL")
        assert aapl_target.delta_notional < 0  # Should sell

        # MSFT should be increased
        msft_target = next(t for t in targets if t.symbol == "MSFT")
        assert msft_target.delta_notional > 0  # Should buy

    def test_compute_rebalance_targets_new_symbol(self):
        """Test rebalance targets with new symbol not in current portfolio."""
        current_weights = {"AAPL": 0.5, "MSFT": 0.5}
        target_weights = {"AAPL": 0.4, "MSFT": 0.4, "GOOGL": 0.2}  # GOOGL is new
        equity = 100000.0
        prices = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 2500.0}

        targets = compute_rebalance_targets(current_weights, target_weights, equity, prices)

        # Should have 3 targets
        assert len(targets) == 3

        # GOOGL should be new position (current_weight = 0)
        googl_target = next(t for t in targets if t.symbol == "GOOGL")
        assert googl_target.current_weight == 0.0
        assert googl_target.target_weight == 0.2
        assert googl_target.delta_notional > 0  # Should buy

    def test_compute_rebalance_targets_remove_symbol(self):
        """Test rebalance targets when removing a symbol."""
        current_weights = {"AAPL": 0.5, "MSFT": 0.3, "GOOGL": 0.2}
        target_weights = {"AAPL": 0.6, "MSFT": 0.4}  # GOOGL removed
        equity = 100000.0
        prices = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 2500.0}

        targets = compute_rebalance_targets(current_weights, target_weights, equity, prices)

        # Should have 3 targets (including GOOGL to remove)
        assert len(targets) == 3

        # GOOGL should be removed (target_weight = 0)
        googl_target = next(t for t in targets if t.symbol == "GOOGL")
        assert googl_target.target_weight == 0.0
        assert googl_target.current_weight == 0.2
        assert googl_target.delta_notional < 0  # Should sell


class TestShouldRebalance:
    """Tests for should_rebalance function."""

    def test_should_rebalance_true(self):
        """Test should_rebalance when rebalancing is needed."""
        current_weights = {"AAPL": 0.4, "MSFT": 0.6}
        target_weights = {"AAPL": 0.5, "MSFT": 0.5}
        threshold = 0.05  # 5% threshold

        # AAPL deviation: |0.4 - 0.5| = 0.1 > 0.05, so should rebalance
        result = should_rebalance(current_weights, target_weights, threshold)

        assert result is True

    def test_should_rebalance_false(self):
        """Test should_rebalance when rebalancing is not needed."""
        current_weights = {"AAPL": 0.48, "MSFT": 0.52}
        target_weights = {"AAPL": 0.5, "MSFT": 0.5}
        threshold = 0.05  # 5% threshold

        # Max deviation: |0.48 - 0.5| = 0.02 < 0.05, so should not rebalance
        result = should_rebalance(current_weights, target_weights, threshold)

        assert result is False

    def test_should_rebalance_custom_threshold(self):
        """Test should_rebalance with custom threshold."""
        current_weights = {"AAPL": 0.45, "MSFT": 0.55}
        target_weights = {"AAPL": 0.5, "MSFT": 0.5}
        threshold = 0.10  # 10% threshold

        # Max deviation: |0.45 - 0.5| = 0.05 < 0.10, so should not rebalance
        result = should_rebalance(current_weights, target_weights, threshold)

        assert result is False

    def test_should_rebalance_missing_symbols(self):
        """Test should_rebalance with missing symbols in current weights."""
        current_weights = {"AAPL": 0.5}
        target_weights = {"AAPL": 0.5, "MSFT": 0.5}  # MSFT missing in current

        # MSFT has deviation of 0.5 (from 0 to 0.5), so should rebalance
        result = should_rebalance(current_weights, target_weights, threshold=0.05)

        assert result is True


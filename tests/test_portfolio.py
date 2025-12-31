"""Unit tests for portfolio management system."""

from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from trading_system.models.orders import Fill, OrderStatus
from trading_system.models.positions import ExitReason, Position, PositionSide
from trading_system.models.signals import BreakoutType, SignalSide
from trading_system.portfolio import (
    Portfolio,
    calculate_position_size,
    compute_average_pairwise_correlation,
    compute_correlation_to_portfolio,
    compute_volatility_scaling,
    estimate_position_size,
)


class TestPositionSizing:
    """Tests for position sizing calculations."""

    def test_basic_position_sizing(self):
        """Test basic risk-based position sizing."""
        equity = 100000.0
        risk_pct = 0.0075  # 0.75%
        entry_price = 100.0
        stop_price = 95.0  # 5% stop
        max_position_notional = 0.15  # 15%
        max_exposure = 0.80  # 80%
        available_cash = 50000.0

        qty = calculate_position_size(
            equity=equity,
            risk_pct=risk_pct,
            entry_price=entry_price,
            stop_price=stop_price,
            max_position_notional=max_position_notional,
            max_exposure=max_exposure,
            available_cash=available_cash,
        )

        # Risk-based: 100000 * 0.0075 / 5 = 150
        # Max notional: 100000 * 0.15 / 100 = 150
        # Max exposure: 100000 * 0.80 / 100 = 800
        # Cash: 50000 / 100 = 500
        # Should be min(150, 150, 800, 500) = 150
        assert qty == 150

    def test_position_sizing_with_volatility_scaling(self):
        """Test position sizing with volatility scaling multiplier."""
        equity = 100000.0
        risk_pct = 0.0075
        entry_price = 100.0
        stop_price = 95.0
        max_position_notional = 0.15
        max_exposure = 0.80
        available_cash = 50000.0
        risk_multiplier = 0.5  # Reduce risk by 50%

        qty = calculate_position_size(
            equity=equity,
            risk_pct=risk_pct,
            entry_price=entry_price,
            stop_price=stop_price,
            max_position_notional=max_position_notional,
            max_exposure=max_exposure,
            available_cash=available_cash,
            risk_multiplier=risk_multiplier,
        )

        # Risk-based: 100000 * 0.0075 * 0.5 / 5 = 75
        # Max notional: 150
        # Should be min(75, 150, 800, 500) = 75
        assert qty == 75

    def test_insufficient_cash(self):
        """Test position sizing when cash is insufficient."""
        equity = 100000.0
        risk_pct = 0.0075
        entry_price = 100.0
        stop_price = 95.0
        max_position_notional = 0.15
        max_exposure = 0.80
        available_cash = 10.0  # Very little cash

        qty = calculate_position_size(
            equity=equity,
            risk_pct=risk_pct,
            entry_price=entry_price,
            stop_price=stop_price,
            max_position_notional=max_position_notional,
            max_exposure=max_exposure,
            available_cash=available_cash,
        )

        # Should be limited by cash: 10 / 100 = 0, so return 0
        assert qty == 0

    def test_max_position_notional_constraint(self):
        """Test that max position notional constraint is respected."""
        equity = 100000.0
        risk_pct = 0.0075
        entry_price = 50.0
        stop_price = 45.0
        max_position_notional = 0.15  # 15%
        max_exposure = 0.80
        available_cash = 100000.0

        qty = calculate_position_size(
            equity=equity,
            risk_pct=risk_pct,
            entry_price=entry_price,
            stop_price=stop_price,
            max_position_notional=max_position_notional,
            max_exposure=max_exposure,
            available_cash=available_cash,
        )

        # Risk-based: 100000 * 0.0075 / 5 = 150
        # Max notional: 100000 * 0.15 / 50 = 300
        # Should be min(150, 300, ...) = 150
        assert qty == 150

    def test_invalid_stop_price(self):
        """Test that invalid stop prices return 0."""
        equity = 100000.0
        risk_pct = 0.0075
        entry_price = 100.0
        stop_price = 105.0  # Stop above entry (invalid)
        max_position_notional = 0.15
        max_exposure = 0.80
        available_cash = 50000.0

        qty = calculate_position_size(
            equity=equity,
            risk_pct=risk_pct,
            entry_price=entry_price,
            stop_price=stop_price,
            max_position_notional=max_position_notional,
            max_exposure=max_exposure,
            available_cash=available_cash,
        )

        assert qty == 0

    def test_estimate_position_size(self):
        """Test position size estimation (without cash/exposure constraints)."""
        equity = 100000.0
        risk_pct = 0.0075
        entry_price = 100.0
        stop_price = 95.0
        max_position_notional = 0.15

        qty = estimate_position_size(
            equity=equity,
            risk_pct=risk_pct,
            entry_price=entry_price,
            stop_price=stop_price,
            max_position_notional=max_position_notional,
        )

        # Risk-based: 100000 * 0.0075 / 5 = 150
        # Max notional: 100000 * 0.15 / 100 = 150
        # Should be min(150, 150) = 150
        assert qty == 150


class TestVolatilityScaling:
    """Tests for volatility scaling calculations."""

    def test_insufficient_history(self):
        """Test volatility scaling with insufficient history."""
        returns = [0.01, 0.02, -0.01]  # Only 3 returns

        risk_mult, vol_20d, median_vol_252d = compute_volatility_scaling(returns)

        assert risk_mult == 1.0
        assert vol_20d is None
        assert median_vol_252d is None

    def test_normal_volatility(self):
        """Test volatility scaling with normal volatility."""
        # Create 20 days of returns with ~15% annualized vol
        np.random.seed(42)
        daily_vol = 0.15 / np.sqrt(252)  # ~0.0095
        returns = np.random.normal(0, daily_vol, 20).tolist()

        risk_mult, vol_20d, median_vol_252d = compute_volatility_scaling(returns)

        assert 0.33 <= risk_mult <= 1.0
        assert vol_20d is not None
        assert median_vol_252d is not None

    def test_high_volatility_reduces_risk(self):
        """Test that high volatility reduces risk multiplier."""
        # Create high volatility returns
        np.random.seed(42)
        high_daily_vol = 0.30 / np.sqrt(252)  # ~0.019
        high_returns = np.random.normal(0, high_daily_vol, 20).tolist()

        # Create low volatility returns for median
        low_daily_vol = 0.10 / np.sqrt(252)  # ~0.0063
        low_returns = np.random.normal(0, low_daily_vol, 20).tolist()

        # Combine: high recent, low historical
        all_returns = low_returns * 10 + high_returns  # 200 low + 20 high

        risk_mult, vol_20d, median_vol_252d = compute_volatility_scaling(all_returns)

        # High recent vol should reduce multiplier
        assert risk_mult < 1.0
        assert vol_20d > median_vol_252d

    def test_low_volatility_increases_risk(self):
        """Test that low volatility allows full risk (multiplier = 1.0)."""
        # Create low volatility returns
        np.random.seed(42)
        low_daily_vol = 0.10 / np.sqrt(252)
        low_returns = np.random.normal(0, low_daily_vol, 252).tolist()

        risk_mult, vol_20d, median_vol_252d = compute_volatility_scaling(low_returns)

        # Low vol should allow high risk (close to 1.0)
        assert risk_mult >= 0.9  # Close to 1.0
        assert vol_20d is not None
        assert median_vol_252d is not None


class TestCorrelation:
    """Tests for correlation calculations."""

    def test_insufficient_positions(self):
        """Test correlation with insufficient positions."""
        returns_data = {"AAPL": [0.01, 0.02, -0.01], "MSFT": [0.015, 0.025, -0.005]}

        avg_corr, corr_matrix = compute_average_pairwise_correlation(returns_data, min_positions=4)

        assert avg_corr is None
        assert corr_matrix is None

    def test_correlation_calculation(self):
        """Test correlation calculation with sufficient data."""
        np.random.seed(42)

        # Create correlated returns
        base_returns = np.random.normal(0, 0.01, 20)
        returns_data = {
            "AAPL": (base_returns + np.random.normal(0, 0.005, 20)).tolist(),
            "MSFT": (base_returns + np.random.normal(0, 0.005, 20)).tolist(),
            "GOOGL": (base_returns + np.random.normal(0, 0.005, 20)).tolist(),
            "AMZN": (base_returns + np.random.normal(0, 0.005, 20)).tolist(),
        }

        avg_corr, corr_matrix = compute_average_pairwise_correlation(returns_data, min_positions=4)

        assert avg_corr is not None
        assert -1.0 <= avg_corr <= 1.0
        assert corr_matrix is not None
        assert corr_matrix.shape == (4, 4)

    def test_correlation_to_portfolio(self):
        """Test correlation of candidate to portfolio."""
        np.random.seed(42)

        base_returns = np.random.normal(0, 0.01, 20)
        portfolio_returns = {
            "AAPL": (base_returns + np.random.normal(0, 0.005, 20)).tolist(),
            "MSFT": (base_returns + np.random.normal(0, 0.005, 20)).tolist(),
        }

        candidate_returns = (base_returns + np.random.normal(0, 0.005, 20)).tolist()

        corr = compute_correlation_to_portfolio(
            candidate_symbol="GOOGL", candidate_returns=candidate_returns, portfolio_returns=portfolio_returns
        )

        assert corr is not None
        assert -1.0 <= corr <= 1.0

    def test_insufficient_history_for_correlation(self):
        """Test correlation with insufficient return history."""
        portfolio_returns = {"AAPL": [0.01, 0.02], "MSFT": [0.015, 0.025]}

        candidate_returns = [0.01, 0.02]  # Only 2 days

        corr = compute_correlation_to_portfolio(
            candidate_symbol="GOOGL", candidate_returns=candidate_returns, portfolio_returns=portfolio_returns, min_days=10
        )

        assert corr is None


class TestPortfolio:
    """Tests for Portfolio class."""

    @pytest.fixture
    def portfolio(self):
        """Create a test portfolio."""
        return Portfolio(date=pd.Timestamp("2024-01-01"), cash=100000.0, starting_equity=100000.0, equity=100000.0)

    @pytest.fixture
    def sample_position(self):
        """Create a sample position."""
        return Position(
            symbol="AAPL",
            asset_class="equity",
            entry_date=pd.Timestamp("2024-01-02"),
            entry_price=150.0,
            entry_fill_id="fill_1",
            quantity=100,
            stop_price=140.0,
            initial_stop_price=140.0,
            hard_stop_atr_mult=2.5,
            entry_slippage_bps=10.0,
            entry_fee_bps=1.0,
            entry_total_cost=15.0,
            triggered_on=BreakoutType.FAST_20D,
            adv20_at_entry=1000000.0,
        )

    def test_portfolio_initialization(self, portfolio):
        """Test portfolio initialization."""
        assert portfolio.cash == 100000.0
        assert portfolio.equity == 100000.0
        assert portfolio.starting_equity == 100000.0
        assert len(portfolio.positions) == 0
        assert len(portfolio.equity_curve) == 1
        assert portfolio.equity_curve[0] == 100000.0

    def test_add_position(self, portfolio, sample_position):
        """Test adding a position."""
        portfolio.add_position(sample_position)

        assert len(portfolio.positions) == 1
        assert "AAPL" in portfolio.positions
        assert portfolio.open_trades == 1

    def test_add_duplicate_position(self, portfolio, sample_position):
        """Test that adding duplicate position raises error."""
        portfolio.add_position(sample_position)

        with pytest.raises(ValueError):
            portfolio.add_position(sample_position)

    def test_remove_position(self, portfolio, sample_position):
        """Test removing a position."""
        portfolio.add_position(sample_position)
        removed = portfolio.remove_position("AAPL")

        assert removed is not None
        assert removed.symbol == "AAPL"
        assert len(portfolio.positions) == 0
        assert portfolio.open_trades == 0

    def test_update_equity(self, portfolio, sample_position):
        """Test updating equity with current prices."""
        portfolio.add_position(sample_position)

        # Update equity with current price
        current_prices = {"AAPL": 160.0}
        portfolio.update_equity(current_prices)

        # Note: add_position does NOT reduce cash (use process_fill for that)
        # Equity = cash + position_value (exposure)
        # Cash = 100000 (unchanged since add_position doesn't modify cash)
        # Position value = 160 * 100 = 16000
        # Equity = 100000 + 16000 = 116000
        expected_equity = 100000.0 + 16000.0
        assert abs(portfolio.equity - expected_equity) < 1.0
        assert portfolio.gross_exposure == 16000.0
        # exposure_pct is calculated before equity update (based on starting equity)
        assert abs(portfolio.gross_exposure_pct - 16000.0 / 100000.0) < 0.01

    def test_process_fill(self, portfolio):
        """Test processing a fill to create a position."""
        fill = Fill(
            fill_id="fill_1",
            order_id="order_1",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp("2024-01-02"),
            side=SignalSide.BUY,
            quantity=100,
            fill_price=150.0,
            open_price=149.5,
            slippage_bps=10.0,
            fee_bps=1.0,
            total_cost=15.0,
            vol_mult=1.0,
            size_penalty=1.0,
            weekend_penalty=1.0,
            stress_mult=1.0,
            notional=15000.0,
        )

        position = portfolio.process_fill(
            fill=fill, stop_price=140.0, atr_mult=2.5, triggered_on=BreakoutType.FAST_20D, adv20_at_entry=1000000.0
        )

        assert position is not None
        assert position.symbol == "AAPL"
        assert position.quantity == 100
        assert portfolio.cash == 100000.0 - 15000.0 - 15.0
        assert "AAPL" in portfolio.positions

    def test_close_position(self, portfolio, sample_position):
        """Test closing a position."""
        portfolio.add_position(sample_position)

        # Update cash to reflect entry
        portfolio.cash = 84985.0

        exit_fill = Fill(
            fill_id="fill_2",
            order_id="order_2",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp("2024-01-10"),
            side=SignalSide.SELL,
            quantity=100,
            fill_price=160.0,
            open_price=160.5,
            slippage_bps=10.0,
            fee_bps=1.0,
            total_cost=16.0,
            vol_mult=1.0,
            size_penalty=1.0,
            weekend_penalty=1.0,
            stress_mult=1.0,
            notional=16000.0,
        )

        closed = portfolio.close_position(symbol="AAPL", exit_fill=exit_fill, exit_reason=ExitReason.TRAILING_MA_CROSS)

        assert closed is not None
        assert closed.exit_reason == ExitReason.TRAILING_MA_CROSS
        assert closed.realized_pnl > 0
        assert "AAPL" not in portfolio.positions
        assert portfolio.total_trades == 1
        # Cash should increase by proceeds minus costs
        assert portfolio.cash > 84985.0

    def test_update_volatility_scaling(self, portfolio):
        """Test updating volatility scaling."""
        # Add some returns to equity curve
        portfolio.equity_curve = [100000.0]
        for i in range(25):
            daily_return = np.random.normal(0, 0.01)
            new_equity = portfolio.equity_curve[-1] * (1 + daily_return)
            portfolio.equity_curve.append(new_equity)
            portfolio.daily_returns.append(daily_return)

        portfolio.update_volatility_scaling()

        assert 0.33 <= portfolio.risk_multiplier <= 1.0
        assert portfolio.portfolio_vol_20d is not None
        assert portfolio.median_vol_252d is not None

    def test_update_correlation_metrics(self, portfolio):
        """Test updating correlation metrics."""
        # Need at least 4 positions
        np.random.seed(42)
        base_returns = np.random.normal(0, 0.01, 20)

        for i, symbol in enumerate(["AAPL", "MSFT", "GOOGL", "AMZN"]):
            position = Position(
                symbol=symbol,
                asset_class="equity",
                entry_date=pd.Timestamp("2024-01-02"),
                entry_price=100.0,
                entry_fill_id=f"fill_{i}",
                quantity=100,
                stop_price=90.0,
                initial_stop_price=90.0,
                hard_stop_atr_mult=2.5,
                entry_slippage_bps=10.0,
                entry_fee_bps=1.0,
                entry_total_cost=10.0,
                triggered_on=BreakoutType.FAST_20D,
                adv20_at_entry=1000000.0,
            )
            portfolio.add_position(position)

        returns_data = {
            "AAPL": (base_returns + np.random.normal(0, 0.005, 20)).tolist(),
            "MSFT": (base_returns + np.random.normal(0, 0.005, 20)).tolist(),
            "GOOGL": (base_returns + np.random.normal(0, 0.005, 20)).tolist(),
            "AMZN": (base_returns + np.random.normal(0, 0.005, 20)).tolist(),
        }

        portfolio.update_correlation_metrics(returns_data)

        assert portfolio.avg_pairwise_corr is not None
        assert -1.0 <= portfolio.avg_pairwise_corr <= 1.0
        assert portfolio.correlation_matrix is not None

    def test_update_stops_hard_stop(self, portfolio, sample_position):
        """Test stop update detects hard stop exit."""
        portfolio.add_position(sample_position)

        current_prices = {"AAPL": 135.0}  # Below stop of 140.0
        features_data = {"AAPL": {"ma20": 145.0, "ma50": 140.0, "atr14": 5.0}}

        exit_signals = portfolio.update_stops(
            current_prices=current_prices, features_data=features_data, exit_mode="ma_cross", exit_ma=20
        )

        assert len(exit_signals) == 1
        assert exit_signals[0][0] == "AAPL"
        assert exit_signals[0][1] == ExitReason.HARD_STOP

    def test_update_stops_ma_cross(self, portfolio, sample_position):
        """Test stop update detects MA cross exit."""
        portfolio.add_position(sample_position)

        # Price is below MA20, should trigger MA cross exit
        current_prices = {"AAPL": 145.0}  # Above stop (140.0), but below MA20 (150.0)
        features_data = {"AAPL": {"ma20": 150.0, "ma50": 145.0, "atr14": 5.0}}

        exit_signals = portfolio.update_stops(
            current_prices=current_prices, features_data=features_data, exit_mode="ma_cross", exit_ma=20
        )

        assert len(exit_signals) == 1
        assert exit_signals[0][0] == "AAPL"
        assert exit_signals[0][1] == ExitReason.TRAILING_MA_CROSS

    def test_crypto_staged_exit(self, portfolio):
        """Test crypto staged exit logic."""
        crypto_position = Position(
            symbol="BTC",
            asset_class="crypto",
            entry_date=pd.Timestamp("2024-01-02"),
            entry_price=50000.0,
            entry_fill_id="fill_1",
            quantity=1,
            stop_price=45000.0,
            initial_stop_price=45000.0,
            hard_stop_atr_mult=3.0,
            entry_slippage_bps=10.0,
            entry_fee_bps=8.0,
            entry_total_cost=50.0,
            triggered_on=BreakoutType.FAST_20D,
            adv20_at_entry=10000000.0,
        )

        portfolio.add_position(crypto_position)

        # Price breaks below MA20 - should tighten stop
        current_prices = {"BTC": 48000.0}
        features_data = {"BTC": {"ma20": 49000.0, "ma50": 47000.0, "atr14": 2000.0}}

        exit_signals = portfolio.update_stops(
            current_prices=current_prices, features_data=features_data, exit_mode="staged", exit_ma=50
        )

        # Should not exit yet, but stop should be tightened
        assert len(exit_signals) == 0
        assert crypto_position.tightened_stop is True
        # Tightened stop = entry_price - 2.0 * atr14 = 50000 - 4000 = 46000
        assert crypto_position.stop_price == 46000.0

    def test_append_daily_metrics(self, portfolio):
        """Test appending daily metrics."""
        portfolio.equity = 105000.0
        portfolio.append_daily_metrics()

        assert len(portfolio.equity_curve) == 2
        assert portfolio.equity_curve[-1] == 105000.0
        assert len(portfolio.daily_returns) == 1
        assert abs(portfolio.daily_returns[0] - 0.05) < 0.001  # 5% return

    def test_get_exposure_summary(self, portfolio, sample_position):
        """Test getting exposure summary."""
        portfolio.add_position(sample_position)
        current_prices = {"AAPL": 160.0}
        portfolio.update_equity(current_prices)

        summary = portfolio.get_exposure_summary()

        assert "gross_exposure" in summary
        assert "gross_exposure_pct" in summary
        assert "cash" in summary
        assert "equity" in summary
        assert summary["equity"] == portfolio.equity

    def test_short_position_sizing(self):
        """Test position sizing for short positions."""
        equity = 100000.0
        risk_pct = 0.0075
        entry_price = 100.0
        stop_price = 105.0  # Stop above entry for short
        max_position_notional = 0.15
        max_exposure = 0.80
        available_cash = 50000.0

        qty = calculate_position_size(
            equity=equity,
            risk_pct=risk_pct,
            entry_price=entry_price,
            stop_price=stop_price,  # Stop above = short
            max_position_notional=max_position_notional,
            max_exposure=max_exposure,
            available_cash=available_cash,
        )

        # Risk-based: 100000 * 0.0075 / 5 = 150
        # Should calculate correctly for shorts
        assert qty > 0

    def test_short_position_creation(self, portfolio):
        """Test creating a short position."""
        fill = Fill(
            fill_id="test_short_fill",
            symbol="TSLA",
            asset_class="equity",
            date=pd.Timestamp("2024-01-15"),
            side=SignalSide.SELL,  # Short
            fill_price=250.0,
            quantity=100,
            slippage_bps=10.0,
            fee_bps=1.0,
            total_cost=275.0,
        )

        # Stop above entry for short
        stop_price = 260.0  # 4% stop above entry
        position = portfolio.process_fill(
            fill=fill, stop_price=stop_price, atr_mult=2.0, triggered_on=BreakoutType.FAST_20D, adv20_at_entry=10_000_000
        )

        assert position.side == PositionSide.SHORT
        assert position.entry_price == 250.0
        assert position.stop_price == 260.0
        assert position.quantity == 100

        # Short: cash should increase (receive proceeds)
        assert portfolio.cash > 100000.0  # Initial cash was 100000

    def test_short_position_pnl(self, portfolio):
        """Test P&L calculation for short positions."""
        fill = Fill(
            fill_id="test_short_fill",
            symbol="TSLA",
            asset_class="equity",
            date=pd.Timestamp("2024-01-15"),
            side=SignalSide.SELL,
            fill_price=250.0,
            quantity=100,
            slippage_bps=10.0,
            fee_bps=1.0,
            total_cost=275.0,
        )

        position = portfolio.process_fill(
            fill=fill, stop_price=260.0, atr_mult=2.0, triggered_on=BreakoutType.FAST_20D, adv20_at_entry=10_000_000
        )

        # Price drops to 240 (profit for short)
        current_prices = {"TSLA": 240.0}
        portfolio.update_equity(current_prices)

        # Short P&L: (entry_price - current_price) * quantity
        # (250 - 240) * 100 = 1000 profit
        assert position.unrealized_pnl > 0  # Should be positive when price drops

    def test_short_position_exit(self, portfolio):
        """Test closing a short position."""
        fill = Fill(
            fill_id="test_short_fill",
            symbol="TSLA",
            asset_class="equity",
            date=pd.Timestamp("2024-01-15"),
            side=SignalSide.SELL,
            fill_price=250.0,
            quantity=100,
            slippage_bps=10.0,
            fee_bps=1.0,
            total_cost=275.0,
        )

        position = portfolio.process_fill(
            fill=fill, stop_price=260.0, atr_mult=2.0, triggered_on=BreakoutType.FAST_20D, adv20_at_entry=10_000_000
        )

        # Close at 240 (profit)
        exit_fill = Fill(
            fill_id="test_exit_fill",
            symbol="TSLA",
            asset_class="equity",
            date=pd.Timestamp("2024-01-20"),
            side=SignalSide.BUY,  # Buy to cover
            fill_price=240.0,
            quantity=100,
            slippage_bps=10.0,
            fee_bps=1.0,
            total_cost=264.0,
        )

        closed_position = portfolio.close_position(symbol="TSLA", exit_fill=exit_fill, exit_reason=ExitReason.MANUAL)

        assert closed_position is not None
        assert closed_position.realized_pnl > 0  # Profit from short
        assert portfolio.cash < portfolio.equity  # Cash decreased (paid to cover)

    def test_short_stop_loss(self, portfolio):
        """Test stop loss for short positions."""
        fill = Fill(
            fill_id="test_short_fill",
            symbol="TSLA",
            asset_class="equity",
            date=pd.Timestamp("2024-01-15"),
            side=SignalSide.SELL,
            fill_price=250.0,
            quantity=100,
            slippage_bps=10.0,
            fee_bps=1.0,
            total_cost=275.0,
        )

        position = portfolio.process_fill(
            fill=fill,
            stop_price=260.0,  # Stop above entry
            atr_mult=2.0,
            triggered_on=BreakoutType.FAST_20D,
            adv20_at_entry=10_000_000,
        )

        # Price rises to stop (loss for short)
        current_prices = {"TSLA": 260.0}
        exit_signals = portfolio.update_stops(
            current_prices=current_prices,
            features_data={"TSLA": {"ma20": 255.0, "ma50": 245.0, "atr14": 5.0}},
            exit_mode="ma_cross",
        )

        # Should trigger exit
        assert len(exit_signals) > 0
        assert exit_signals[0][0] == "TSLA"
        assert exit_signals[0][1] == ExitReason.HARD_STOP

    def test_short_exposure_limits(self, portfolio):
        """Test short exposure limits."""
        from trading_system.configs.strategy_config import RiskConfig, StrategyConfig
        from trading_system.strategies.base.strategy_interface import StrategyInterface

        # Create a mock strategy with short limits
        risk_config = RiskConfig(
            max_exposure=0.80, max_short_exposure=0.40, max_net_exposure=0.40  # 40% max short exposure  # ±40% net exposure
        )

        # Create a simple mock strategy
        class MockStrategy(StrategyInterface):
            def __init__(self):
                from trading_system.configs.strategy_config import StrategyConfig

                self.config = StrategyConfig(
                    name="test", asset_class="equity", universe=["TSLA"], benchmark="SPY", risk=risk_config
                )

        strategy = MockStrategy()
        portfolio.strategies["test"] = strategy

        # Try to create a short position that would exceed limits
        fill = Fill(
            fill_id="test_short_fill",
            symbol="TSLA",
            asset_class="equity",
            date=pd.Timestamp("2024-01-15"),
            side=SignalSide.SELL,
            fill_price=250.0,
            quantity=200,  # Large position
            slippage_bps=10.0,
            fee_bps=1.0,
            total_cost=550.0,
        )

        # This should work if within limits
        try:
            position = portfolio.process_fill(
                fill=fill,
                stop_price=260.0,
                atr_mult=2.0,
                triggered_on=BreakoutType.FAST_20D,
                adv20_at_entry=10_000_000,
                strategy_name="test",
            )
            # If successful, check exposure
            portfolio.update_equity({"TSLA": 250.0})
            short_exposure_pct = portfolio.short_exposure / portfolio.equity if portfolio.equity > 0 else 0.0
            assert short_exposure_pct <= 0.40 + 0.0001  # Within limit
        except ValueError as e:
            # If it fails due to limits, that's also valid
            assert "exceed" in str(e).lower() or "limit" in str(e).lower()

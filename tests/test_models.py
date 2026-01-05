"""Unit tests for all data models."""


import numpy as np
import pandas as pd
import pytest

from trading_system.configs.strategy_config import EligibilityConfig, IndicatorsConfig, StrategyConfig
from trading_system.models.bar import Bar
from trading_system.models.features import FeatureRow
from trading_system.models.market_data import MarketData
from trading_system.models.orders import Fill, Order, OrderStatus
from trading_system.models.portfolio import Portfolio
from trading_system.models.positions import Position, PositionSide
from trading_system.models.signals import BreakoutType, Signal, SignalSide, SignalType


class TestBar:
    """Tests for Bar model."""

    def test_valid_bar(self):
        """Test creating a valid bar."""
        date = pd.Timestamp("2024-01-01")
        bar = Bar(date=date, symbol="AAPL", open=100.0, high=105.0, low=99.0, close=103.0, volume=1000000.0)

        assert bar.date == date
        assert bar.symbol == "AAPL"
        assert bar.open == 100.0
        assert bar.high == 105.0
        assert bar.low == 99.0
        assert bar.close == 103.0
        assert bar.volume == 1000000.0
        assert bar.dollar_volume == 103.0 * 1000000.0

    def test_dollar_volume_computed(self):
        """Test dollar_volume is computed if not provided."""
        bar = Bar(
            date=pd.Timestamp("2024-01-01"), symbol="AAPL", open=100.0, high=105.0, low=99.0, close=103.0, volume=1000000.0
        )
        assert bar.dollar_volume == 103.0 * 1000000.0

    def test_dollar_volume_provided(self):
        """Test dollar_volume can be provided explicitly."""
        bar = Bar(
            date=pd.Timestamp("2024-01-01"),
            symbol="AAPL",
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000000.0,
            dollar_volume=103000000.0,
        )
        assert bar.dollar_volume == 103000000.0

    def test_invalid_ohlc_open(self):
        """Test validation fails if open is out of range."""
        with pytest.raises(ValueError, match="open.*must be between"):
            Bar(
                date=pd.Timestamp("2024-01-01"),
                symbol="AAPL",
                open=110.0,  # Above high
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000000.0,
            )

    def test_invalid_ohlc_close(self):
        """Test validation fails if close is out of range."""
        with pytest.raises(ValueError, match="close.*must be between"):
            Bar(
                date=pd.Timestamp("2024-01-01"),
                symbol="AAPL",
                open=100.0,
                high=105.0,
                low=99.0,
                close=98.0,  # Below low
                volume=1000000.0,
            )

    def test_negative_volume(self):
        """Test validation fails for negative volume."""
        with pytest.raises(ValueError, match="Negative volume"):
            Bar(date=pd.Timestamp("2024-01-01"), symbol="AAPL", open=100.0, high=105.0, low=99.0, close=103.0, volume=-1000.0)

    def test_non_positive_price(self):
        """Test validation fails for non-positive prices."""
        with pytest.raises(ValueError, match="prices must be positive"):
            Bar(
                date=pd.Timestamp("2024-01-01"),
                symbol="AAPL",
                open=100.0,
                high=105.0,
                low=99.0,
                close=0.0,  # Invalid
                volume=1000000.0,
            )


class TestFeatureRow:
    """Tests for FeatureRow model."""

    def test_valid_feature_row(self):
        """Test creating a valid feature row."""
        date = pd.Timestamp("2024-01-01")
        features = FeatureRow(
            date=date,
            symbol="AAPL",
            asset_class="equity",
            close=100.0,
            open=99.0,
            high=101.0,
            low=98.0,
            ma20=99.5,
            ma50=98.0,
            atr14=2.0,
            highest_close_20d=99.0,
            highest_close_55d=98.0,
            adv20=10000000.0,
        )

        assert features.date == date
        assert features.symbol == "AAPL"
        assert features.asset_class == "equity"
        assert features.is_valid_for_entry() is True

    def test_invalid_asset_class(self):
        """Test validation fails for invalid asset_class."""
        with pytest.raises(ValueError, match="Invalid asset_class"):
            FeatureRow(
                date=pd.Timestamp("2024-01-01"),
                symbol="AAPL",
                asset_class="invalid",
                close=100.0,
                open=99.0,
                high=101.0,
                low=98.0,
            )

    def test_is_valid_for_entry_all_present(self):
        """Test is_valid_for_entry returns True when all required fields present."""
        features = FeatureRow(
            date=pd.Timestamp("2024-01-01"),
            symbol="AAPL",
            asset_class="equity",
            close=100.0,
            open=99.0,
            high=101.0,
            low=98.0,
            ma20=99.5,
            ma50=98.0,
            atr14=2.0,
            highest_close_20d=99.0,
            highest_close_55d=98.0,
            adv20=10000000.0,
        )
        assert features.is_valid_for_entry() is True

    def test_is_valid_for_entry_missing_field(self):
        """Test is_valid_for_entry returns False when required field missing."""
        features = FeatureRow(
            date=pd.Timestamp("2024-01-01"),
            symbol="AAPL",
            asset_class="equity",
            close=100.0,
            open=99.0,
            high=101.0,
            low=98.0,
            ma20=99.5,
            ma50=98.0,
            atr14=2.0,
            highest_close_20d=99.0,
            highest_close_55d=98.0,
            adv20=None,  # Missing
        )
        assert features.is_valid_for_entry() is False

    def test_is_valid_for_entry_nan_field(self):
        """Test is_valid_for_entry returns False when field is NaN."""
        features = FeatureRow(
            date=pd.Timestamp("2024-01-01"),
            symbol="AAPL",
            asset_class="equity",
            close=100.0,
            open=99.0,
            high=101.0,
            low=98.0,
            ma20=np.nan,  # NaN
            ma50=98.0,
            atr14=2.0,
            highest_close_20d=99.0,
            highest_close_55d=98.0,
            adv20=10000000.0,
        )
        assert features.is_valid_for_entry() is False


class TestSignal:
    """Tests for Signal model."""

    def test_valid_signal(self):
        """Test creating a valid signal."""
        signal = Signal(
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp("2024-01-01"),
            side=SignalSide.BUY,
            signal_type=SignalType.ENTRY_LONG,
            trigger_reason="test_breakout",
            entry_price=100.0,
            stop_price=95.0,
            atr_mult=2.5,
            triggered_on=BreakoutType.FAST_20D,
            breakout_clearance=0.01,
            breakout_strength=0.5,
            momentum_strength=0.3,
            diversification_bonus=0.2,
            score=0.8,
            passed_eligibility=True,
            eligibility_failures=[],
            order_notional=10000.0,
            adv20=2000000.0,
            capacity_passed=True,
        )

        assert signal.symbol == "AAPL"
        assert signal.is_valid() is True

    def test_invalid_stop_price(self):
        """Test validation fails if stop_price >= entry_price."""
        with pytest.raises(ValueError, match="stop_price.*>= entry_price"):
            Signal(
                symbol="AAPL",
                asset_class="equity",
                date=pd.Timestamp("2024-01-01"),
                side=SignalSide.BUY,
                signal_type=SignalType.ENTRY_LONG,
                trigger_reason="test_breakout",
                entry_price=100.0,
                stop_price=105.0,  # Above entry
                atr_mult=2.5,
                triggered_on=BreakoutType.FAST_20D,
                breakout_clearance=0.01,
                breakout_strength=0.5,
                momentum_strength=0.3,
                diversification_bonus=0.2,
                score=0.8,
                passed_eligibility=True,
                eligibility_failures=[],
                order_notional=10000.0,
                adv20=2000000.0,
                capacity_passed=True,
            )

    def test_is_valid_fails_eligibility(self):
        """Test is_valid returns False if eligibility failed."""
        signal = Signal(
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp("2024-01-01"),
            side=SignalSide.BUY,
            signal_type=SignalType.ENTRY_LONG,
            trigger_reason="test_breakout",
            entry_price=100.0,
            stop_price=95.0,
            atr_mult=2.5,
            triggered_on=BreakoutType.FAST_20D,
            breakout_clearance=0.01,
            breakout_strength=0.5,
            momentum_strength=0.3,
            diversification_bonus=0.2,
            score=0.8,
            passed_eligibility=False,  # Failed
            eligibility_failures=["below_MA50"],
            order_notional=10000.0,
            adv20=2000000.0,
            capacity_passed=True,
        )
        assert signal.is_valid() is False

    def test_is_valid_fails_capacity(self):
        """Test is_valid returns False if capacity check failed."""
        signal = Signal(
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp("2024-01-01"),
            side=SignalSide.BUY,
            signal_type=SignalType.ENTRY_LONG,
            trigger_reason="test_breakout",
            entry_price=100.0,
            stop_price=95.0,
            atr_mult=2.5,
            triggered_on=BreakoutType.FAST_20D,
            breakout_clearance=0.01,
            breakout_strength=0.5,
            momentum_strength=0.3,
            diversification_bonus=0.2,
            score=0.8,
            passed_eligibility=True,
            eligibility_failures=[],
            order_notional=10000.0,
            adv20=2000000.0,
            capacity_passed=False,  # Failed
        )
        assert signal.is_valid() is False


class TestOrder:
    """Tests for Order model."""

    def test_valid_order(self):
        """Test creating a valid order."""
        order = Order(
            order_id="ORD001",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp("2024-01-01"),
            execution_date=pd.Timestamp("2024-01-02"),
            side=SignalSide.BUY,
            quantity=100,
            signal_date=pd.Timestamp("2024-01-01"),
            expected_fill_price=100.0,
            stop_price=95.0,
        )

        assert order.order_id == "ORD001"
        assert order.status == OrderStatus.PENDING
        assert order.quantity == 100

    def test_invalid_quantity(self):
        """Test validation fails for non-positive quantity."""
        with pytest.raises(ValueError, match="Invalid quantity"):
            Order(
                order_id="ORD001",
                symbol="AAPL",
                asset_class="equity",
                date=pd.Timestamp("2024-01-01"),
                execution_date=pd.Timestamp("2024-01-02"),
                side=SignalSide.BUY,
                quantity=0,  # Invalid
                signal_date=pd.Timestamp("2024-01-01"),
                expected_fill_price=100.0,
                stop_price=95.0,
            )


class TestFill:
    """Tests for Fill model."""

    def test_valid_fill(self):
        """Test creating a valid fill."""
        fill = Fill(
            fill_id="FILL001",
            order_id="ORD001",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp("2024-01-02"),
            side=SignalSide.BUY,
            quantity=100,
            fill_price=100.5,
            open_price=100.0,
            slippage_bps=5.0,
            fee_bps=1.0,
            total_cost=60.5,
            vol_mult=1.0,
            size_penalty=1.0,
            weekend_penalty=1.0,
            stress_mult=1.0,
            notional=10050.0,
        )

        assert fill.fill_id == "FILL001"
        # slippage_cost = 10050 * (5.0 / 10000) = 5.025
        # fee_cost = 10050 * (1.0 / 10000) = 1.005
        # total = 6.03
        assert fill.compute_total_cost() == pytest.approx(6.03, rel=1e-2)

    def test_compute_total_cost(self):
        """Test total cost computation."""
        fill = Fill(
            fill_id="FILL001",
            order_id="ORD001",
            symbol="AAPL",
            asset_class="equity",
            date=pd.Timestamp("2024-01-02"),
            side=SignalSide.BUY,
            quantity=100,
            fill_price=100.0,
            open_price=100.0,
            slippage_bps=10.0,
            fee_bps=1.0,
            total_cost=0.0,  # Will be computed
            vol_mult=1.0,
            size_penalty=1.0,
            weekend_penalty=1.0,
            stress_mult=1.0,
            notional=10000.0,
        )

        # slippage_cost = 10000 * 0.001 = 10.0
        # fee_cost = 10000 * 0.0001 = 1.0
        # total = 11.0
        cost = fill.compute_total_cost()
        assert cost == pytest.approx(11.0, rel=1e-2)

    def test_invalid_fill_price_buy(self):
        """Test validation fails if BUY fill_price < open_price."""
        with pytest.raises(ValueError, match="fill_price for BUY"):
            Fill(
                fill_id="FILL001",
                order_id="ORD001",
                symbol="AAPL",
                asset_class="equity",
                date=pd.Timestamp("2024-01-02"),
                side=SignalSide.BUY,
                quantity=100,
                fill_price=99.0,  # Below open (invalid for BUY)
                open_price=100.0,
                slippage_bps=5.0,
                fee_bps=1.0,
                total_cost=60.5,
                vol_mult=1.0,
                size_penalty=1.0,
                weekend_penalty=1.0,
                stress_mult=1.0,
                notional=9900.0,
            )


class TestPosition:
    """Tests for Position model."""

    def test_valid_position(self):
        """Test creating a valid position."""
        position = Position(
            symbol="AAPL",
            asset_class="equity",
            entry_date=pd.Timestamp("2024-01-02"),
            entry_price=100.0,
            entry_fill_id="FILL001",
            quantity=100,
            side=PositionSide.LONG,
            stop_price=95.0,
            initial_stop_price=95.0,
            hard_stop_atr_mult=2.5,
            entry_slippage_bps=5.0,
            entry_fee_bps=1.0,
            entry_total_cost=60.0,
            triggered_on=BreakoutType.FAST_20D,
            adv20_at_entry=2000000.0,
        )

        assert position.symbol == "AAPL"
        assert position.is_open() is True
        assert position.quantity == 100

    def test_is_open(self):
        """Test is_open returns correct value."""
        position = Position(
            symbol="AAPL",
            asset_class="equity",
            entry_date=pd.Timestamp("2024-01-02"),
            entry_price=100.0,
            entry_fill_id="FILL001",
            quantity=100,
            side=PositionSide.LONG,
            stop_price=95.0,
            initial_stop_price=95.0,
            hard_stop_atr_mult=2.5,
            entry_slippage_bps=5.0,
            entry_fee_bps=1.0,
            entry_total_cost=60.0,
            triggered_on=BreakoutType.FAST_20D,
            adv20_at_entry=2000000.0,
        )

        assert position.is_open() is True

        position.exit_date = pd.Timestamp("2024-01-10")
        assert position.is_open() is False

    def test_update_unrealized_pnl(self):
        """Test updating unrealized P&L."""
        position = Position(
            symbol="AAPL",
            asset_class="equity",
            entry_date=pd.Timestamp("2024-01-02"),
            entry_price=100.0,
            entry_fill_id="FILL001",
            quantity=100,
            side=PositionSide.LONG,
            stop_price=95.0,
            initial_stop_price=95.0,
            hard_stop_atr_mult=2.5,
            entry_slippage_bps=5.0,
            entry_fee_bps=1.0,
            entry_total_cost=60.0,
            triggered_on=BreakoutType.FAST_20D,
            adv20_at_entry=2000000.0,
        )

        # Price up to 105, P&L = (105 - 100) * 100 - 60 = 440
        position.update_unrealized_pnl(105.0)
        assert position.unrealized_pnl == pytest.approx(440.0, rel=1e-2)

        # Price down to 98, P&L = (98 - 100) * 100 - 60 = -260
        position.update_unrealized_pnl(98.0)
        assert position.unrealized_pnl == pytest.approx(-260.0, rel=1e-2)

    def test_compute_r_multiple(self):
        """Test R-multiple calculation."""
        position = Position(
            symbol="AAPL",
            asset_class="equity",
            entry_date=pd.Timestamp("2024-01-02"),
            entry_price=100.0,
            entry_fill_id="FILL001",
            quantity=100,
            side=PositionSide.LONG,
            stop_price=95.0,
            initial_stop_price=95.0,
            hard_stop_atr_mult=2.5,
            entry_slippage_bps=5.0,
            entry_fee_bps=1.0,
            entry_total_cost=60.0,
            triggered_on=BreakoutType.FAST_20D,
            adv20_at_entry=2000000.0,
            exit_price=110.0,
        )

        # R-multiple = (110 - 100) / (100 - 95) = 10 / 5 = 2.0
        r_mult = position.compute_r_multiple()
        assert r_mult == pytest.approx(2.0, rel=1e-2)

    def test_update_stop_trailing(self):
        """Test updating stop price (trailing)."""
        position = Position(
            symbol="AAPL",
            asset_class="equity",
            entry_date=pd.Timestamp("2024-01-02"),
            entry_price=100.0,
            entry_fill_id="FILL001",
            quantity=100,
            side=PositionSide.LONG,
            stop_price=95.0,
            initial_stop_price=95.0,
            hard_stop_atr_mult=2.5,
            entry_slippage_bps=5.0,
            entry_fee_bps=1.0,
            entry_total_cost=60.0,
            triggered_on=BreakoutType.FAST_20D,
            adv20_at_entry=2000000.0,
        )

        # Update stop higher (trailing)
        position.update_stop(97.0, reason="trailing")
        assert position.stop_price == 97.0

        # Try to update lower (should not change)
        position.update_stop(96.0, reason="trailing")
        assert position.stop_price == 97.0  # Unchanged

    def test_update_stop_tighten(self):
        """Test tightening stop (crypto staged exit)."""
        position = Position(
            symbol="BTC",
            asset_class="crypto",
            entry_date=pd.Timestamp("2024-01-02"),
            entry_price=100.0,
            entry_fill_id="FILL001",
            quantity=100,
            side=PositionSide.LONG,
            stop_price=95.0,
            initial_stop_price=95.0,
            hard_stop_atr_mult=3.0,
            entry_slippage_bps=5.0,
            entry_fee_bps=8.0,
            entry_total_cost=130.0,
            triggered_on=BreakoutType.FAST_20D,
            adv20_at_entry=2000000.0,
        )

        # Tighten stop
        position.update_stop(96.0, reason="tighten")
        assert position.stop_price == 96.0
        assert position.tightened_stop is True
        assert position.tightened_stop_atr_mult == 2.0


class TestPortfolio:
    """Tests for Portfolio model."""

    def test_valid_portfolio(self):
        """Test creating a valid portfolio."""
        portfolio = Portfolio(date=pd.Timestamp("2024-01-01"), cash=50000.0, starting_equity=100000.0, equity=100000.0)

        assert portfolio.date == pd.Timestamp("2024-01-01")
        assert portfolio.cash == 50000.0
        assert portfolio.starting_equity == 100000.0
        assert portfolio.equity == 100000.0
        assert len(portfolio.positions) == 0

    def test_update_equity(self):
        """Test updating equity with current prices."""
        portfolio = Portfolio(date=pd.Timestamp("2024-01-01"), cash=50000.0, starting_equity=100000.0, equity=100000.0)

        # Add a position
        position = Position(
            symbol="AAPL",
            asset_class="equity",
            entry_date=pd.Timestamp("2024-01-01"),
            entry_price=100.0,
            entry_fill_id="FILL001",
            quantity=100,
            side=PositionSide.LONG,
            stop_price=95.0,
            initial_stop_price=95.0,
            hard_stop_atr_mult=2.5,
            entry_slippage_bps=5.0,
            entry_fee_bps=1.0,
            entry_total_cost=60.0,
            triggered_on=BreakoutType.FAST_20D,
            adv20_at_entry=2000000.0,
        )
        portfolio.positions["AAPL"] = position

        # Update equity with current price
        current_prices = {"AAPL": 105.0}
        portfolio.update_equity(current_prices)

        # Equity = cash + position_value = 50000 + 10500 = 60500
        # But we need to account for unrealized P&L
        assert portfolio.gross_exposure == pytest.approx(10500.0, rel=1e-2)
        assert portfolio.open_trades == 1

    def test_compute_portfolio_returns(self):
        """Test computing portfolio returns."""
        portfolio = Portfolio(date=pd.Timestamp("2024-01-01"), cash=50000.0, starting_equity=100000.0, equity=100000.0)

        # Add equity curve
        portfolio.equity_curve = [100000.0, 101000.0, 102000.0, 101500.0]

        returns = portfolio.compute_portfolio_returns()
        assert len(returns) == 3
        assert returns[0] == pytest.approx(0.01, rel=1e-2)  # 1% gain
        assert returns[1] == pytest.approx(0.0099, rel=1e-2)  # ~1% gain
        assert returns[2] == pytest.approx(-0.0049, rel=1e-2)  # ~-0.5% loss

    def test_update_volatility_scaling_insufficient_history(self):
        """Test volatility scaling with insufficient history."""
        portfolio = Portfolio(date=pd.Timestamp("2024-01-01"), cash=50000.0, starting_equity=100000.0, equity=100000.0)

        # Add only 10 days of equity curve
        portfolio.equity_curve = [100000.0 + i * 100 for i in range(10)]

        portfolio.update_volatility_scaling()

        # Should use default multiplier
        assert portfolio.risk_multiplier == 1.0
        assert portfolio.portfolio_vol_20d is None

    def test_update_correlation_metrics_insufficient_positions(self):
        """Test correlation metrics with insufficient positions."""
        portfolio = Portfolio(date=pd.Timestamp("2024-01-01"), cash=50000.0, starting_equity=100000.0, equity=100000.0)

        # Add only 2 positions (need 4+)
        portfolio.positions["AAPL"] = Position(
            symbol="AAPL",
            asset_class="equity",
            entry_date=pd.Timestamp("2024-01-01"),
            entry_price=100.0,
            entry_fill_id="FILL001",
            quantity=100,
            side=PositionSide.LONG,
            stop_price=95.0,
            initial_stop_price=95.0,
            hard_stop_atr_mult=2.5,
            entry_slippage_bps=5.0,
            entry_fee_bps=1.0,
            entry_total_cost=60.0,
            triggered_on=BreakoutType.FAST_20D,
            adv20_at_entry=2000000.0,
        )

        portfolio.update_correlation_metrics({})

        assert portfolio.avg_pairwise_corr is None
        assert portfolio.correlation_matrix is None


class TestMarketData:
    """Tests for MarketData container."""

    def test_get_bar(self):
        """Test getting a bar from MarketData."""
        market_data = MarketData()

        # Create sample bars DataFrame
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        bars_df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "high": [105.0, 106.0, 107.0, 108.0, 109.0],
                "low": [99.0, 100.0, 101.0, 102.0, 103.0],
                "close": [103.0, 104.0, 105.0, 106.0, 107.0],
                "volume": [1000000.0] * 5,
            },
            index=dates,
        )

        market_data.add_bars("AAPL", bars_df)

        # Get bar
        bar = market_data.get_bar("AAPL", dates[0])
        assert bar is not None
        assert bar.symbol == "AAPL"
        assert bar.close == 103.0

        # Get non-existent bar
        bar = market_data.get_bar("AAPL", pd.Timestamp("2024-01-10"))
        assert bar is None

        # Get bar for non-existent symbol
        bar = market_data.get_bar("MSFT", dates[0])
        assert bar is None

    def test_get_features(self):
        """Test getting features from MarketData."""
        market_data = MarketData()

        # Create sample features DataFrame
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        features_df = pd.DataFrame(
            {
                "asset_class": ["equity"] * 5,
                "close": [100.0, 101.0, 102.0, 103.0, 104.0],
                "open": [99.0, 100.0, 101.0, 102.0, 103.0],
                "high": [101.0, 102.0, 103.0, 104.0, 105.0],
                "low": [98.0, 99.0, 100.0, 101.0, 102.0],
                "ma20": [99.0, 99.5, 100.0, 100.5, 101.0],
                "ma50": [98.0, 98.5, 99.0, 99.5, 100.0],
                "atr14": [2.0] * 5,
                "highest_close_20d": [99.0, 100.0, 101.0, 102.0, 103.0],
                "highest_close_55d": [98.0, 99.0, 100.0, 101.0, 102.0],
                "adv20": [10000000.0] * 5,
            },
            index=dates,
        )

        market_data.add_features("AAPL", features_df)

        # Get features
        features = market_data.get_features("AAPL", dates[0])
        assert features is not None
        assert features.symbol == "AAPL"
        assert features.close == 100.0
        assert features.is_valid_for_entry() is True


class TestStrategyConfig:
    """Tests for StrategyConfig Pydantic models."""

    def test_valid_strategy_config(self):
        """Test creating a valid strategy config."""
        config = StrategyConfig(name="test_strategy", asset_class="equity", universe="NASDAQ-100", benchmark="SPY")

        assert config.name == "test_strategy"
        assert config.asset_class == "equity"
        assert config.universe == "NASDAQ-100"
        assert config.benchmark == "SPY"
        assert isinstance(config.indicators, IndicatorsConfig)
        assert isinstance(config.eligibility, EligibilityConfig)

    def test_invalid_asset_class(self):
        """Test validation fails for invalid asset_class."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            StrategyConfig(name="test_strategy", asset_class="invalid", universe="NASDAQ-100", benchmark="SPY")

    def test_default_configs(self):
        """Test default configuration values."""
        config = StrategyConfig(name="test_strategy", asset_class="equity", universe="NASDAQ-100", benchmark="SPY")

        assert config.risk.risk_per_trade == 0.0075
        assert config.risk.max_positions == 8
        assert config.costs.fee_bps == 1
        assert config.indicators.ma_periods == [20, 50, 200]

"""Integration tests for daily signal service and canonical contracts.

Tests the golden path: signals -> newsletter payload -> artifacts
"""

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from trading_system.integration.daily_signal_service import DailySignalService
from trading_system.models.contracts import (
    Allocation,
    AssetClass,
    DailySignalBatch,
    OrderMethod,
    Signal,
    SignalIntent,
    StopLogicType,
    TradePlan,
)


@pytest.fixture
def mock_ohlcv_data():
    """Create mock OHLCV data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    data = {
        "AAPL": pd.DataFrame(
            {
                "open": [150.0] * 100,
                "high": [155.0] * 100,
                "low": [145.0] * 100,
                "close": [152.0] * 100,
                "volume": [1000000] * 100,
            },
            index=dates,
        ),
        "MSFT": pd.DataFrame(
            {
                "open": [300.0] * 100,
                "high": [310.0] * 100,
                "low": [290.0] * 100,
                "close": [305.0] * 100,
                "volume": [800000] * 100,
            },
            index=dates,
        ),
    }
    return data


@pytest.fixture
def mock_recommendations():
    """Create mock recommendations for testing."""
    return [
        {
            "symbol": "AAPL",
            "date": date.today(),
            "side": "BUY",
            "conviction": "HIGH",
            "conviction_score": 0.85,
            "entry_price": 152.0,
            "stop_price": 145.0,
            "technical_reason": "Momentum breakout above 20-day MA",
            "news_summary": "Positive earnings report",
            "strategy": "momentum",
            "position_size_dollars": 10000.0,
            "position_size_pct": 2.0,
            "quantity": 65,
            "risk_budget_used": 1.0,
            "max_positions_applied": False,
            "liquidity_flags": [],
            "capacity_flags": [],
            "max_adv_pct": 0.5,
            "atr_mult": 2.5,
            "notes": "Strong technical setup",
        },
        {
            "symbol": "MSFT",
            "date": date.today(),
            "side": "BUY",
            "conviction": "MEDIUM",
            "conviction_score": 0.65,
            "entry_price": 305.0,
            "stop_price": 295.0,
            "technical_reason": "Bullish reversal pattern",
            "news_summary": "New product launch",
            "strategy": "momentum",
            "position_size_dollars": 8000.0,
            "position_size_pct": 1.5,
            "quantity": 26,
            "risk_budget_used": 0.8,
            "max_positions_applied": False,
            "liquidity_flags": [],
            "capacity_flags": [],
            "max_adv_pct": 0.4,
            "atr_mult": 2.5,
            "notes": "Good risk/reward",
        },
    ]


class TestDailySignalService:
    """Test suite for DailySignalService."""

    @pytest.mark.asyncio
    async def test_generate_daily_signals_basic(self, mock_ohlcv_data, mock_recommendations):
        """Test basic daily signal generation."""
        with patch("trading_system.integration.daily_signal_service.LiveDataFetcher") as mock_fetcher_class, patch(
            "trading_system.integration.daily_signal_service.LiveSignalGenerator"
        ) as mock_generator_class, patch(
            "trading_system.integration.daily_signal_service.load_universe"
        ) as mock_load_universe, patch(
            "trading_system.integration.daily_signal_service.load_strategies_from_run_config"
        ) as mock_load_strategies:
            # Setup mocks
            mock_fetcher = AsyncMock()
            mock_fetcher.fetch_daily_data = AsyncMock(return_value=mock_ohlcv_data)
            mock_fetcher_class.return_value = mock_fetcher

            mock_generator = MagicMock()
            mock_generator.generate_recommendations = AsyncMock(return_value=mock_recommendations)
            mock_generator_class.return_value = mock_generator

            mock_load_universe.return_value = ["AAPL", "MSFT"]
            mock_load_strategies.return_value = [MagicMock()]

            # Create service and generate signals
            service = DailySignalService()
            batch = await service.generate_daily_signals(asset_class="equity", bucket="safe_sp500")

            # Verify batch structure
            assert isinstance(batch, DailySignalBatch)
            assert len(batch.signals) == 2
            assert len(batch.allocations) == 2
            assert len(batch.trade_plans) == 2

            # Verify signals
            for signal in batch.signals:
                assert isinstance(signal, Signal)
                assert signal.asset_class == AssetClass.EQUITY
                assert signal.intent == SignalIntent.EXECUTE_NEXT_OPEN
                assert 0.0 <= signal.confidence <= 1.0
                assert signal.bucket == "safe_sp500"

            # Verify allocations
            for allocation in batch.allocations:
                assert isinstance(allocation, Allocation)
                assert allocation.recommended_position_size_dollars > 0
                assert allocation.quantity > 0

            # Verify trade plans
            for trade_plan in batch.trade_plans:
                assert isinstance(trade_plan, TradePlan)
                assert trade_plan.entry_method == OrderMethod.MOO
                assert trade_plan.stop_logic == StopLogicType.ATR_TRAILING
                assert trade_plan.entry_price > 0
                assert trade_plan.stop_price > 0

    @pytest.mark.asyncio
    async def test_signal_conversion_accuracy(self, mock_ohlcv_data, mock_recommendations):
        """Test that signal conversion preserves all important data."""
        with patch("trading_system.integration.daily_signal_service.LiveDataFetcher") as mock_fetcher_class, patch(
            "trading_system.integration.daily_signal_service.LiveSignalGenerator"
        ) as mock_generator_class, patch(
            "trading_system.integration.daily_signal_service.load_universe"
        ) as mock_load_universe, patch(
            "trading_system.integration.daily_signal_service.load_strategies_from_run_config"
        ) as mock_load_strategies:
            # Setup mocks
            mock_fetcher = AsyncMock()
            mock_fetcher.fetch_daily_data = AsyncMock(return_value=mock_ohlcv_data)
            mock_fetcher_class.return_value = mock_fetcher

            mock_generator = MagicMock()
            mock_generator.generate_recommendations = AsyncMock(return_value=mock_recommendations)
            mock_generator_class.return_value = mock_generator

            mock_load_universe.return_value = ["AAPL", "MSFT"]
            mock_load_strategies.return_value = [MagicMock()]

            # Create service and generate signals
            service = DailySignalService()
            batch = await service.generate_daily_signals(asset_class="equity")

            # Verify first signal
            signal = batch.signals[0]
            rec = mock_recommendations[0]

            assert signal.symbol == rec["symbol"]
            assert signal.side == rec["side"]
            assert signal.confidence == rec["conviction_score"]
            assert signal.entry_price == rec["entry_price"]
            assert signal.stop_price == rec["stop_price"]
            assert signal.rationale_tags["technical"] == rec["technical_reason"]
            assert signal.rationale_tags["news"] == rec["news_summary"]

    @pytest.mark.asyncio
    async def test_batch_summaries(self, mock_ohlcv_data, mock_recommendations):
        """Test that batch summaries are correctly generated."""
        with patch("trading_system.integration.daily_signal_service.LiveDataFetcher") as mock_fetcher_class, patch(
            "trading_system.integration.daily_signal_service.LiveSignalGenerator"
        ) as mock_generator_class, patch(
            "trading_system.integration.daily_signal_service.load_universe"
        ) as mock_load_universe, patch(
            "trading_system.integration.daily_signal_service.load_strategies_from_run_config"
        ) as mock_load_strategies:
            # Setup mocks
            mock_fetcher = AsyncMock()
            mock_fetcher.fetch_daily_data = AsyncMock(return_value=mock_ohlcv_data)
            mock_fetcher_class.return_value = mock_fetcher

            mock_generator = MagicMock()
            mock_generator.generate_recommendations = AsyncMock(return_value=mock_recommendations)
            mock_generator_class.return_value = mock_generator

            mock_load_universe.return_value = ["AAPL", "MSFT"]
            mock_load_strategies.return_value = [MagicMock()]

            # Create service and generate signals
            service = DailySignalService()
            batch = await service.generate_daily_signals(asset_class="equity", bucket="test_bucket")

            # Verify bucket summaries
            assert "test_bucket" in batch.bucket_summaries
            summary = batch.bucket_summaries["test_bucket"]
            assert summary["total_signals"] == 2
            assert summary["asset_class"] == "equity"
            assert 0.0 <= summary["avg_confidence"] <= 1.0

            # Verify metadata
            assert batch.metadata["asset_class"] == "equity"
            assert batch.metadata["bucket"] == "test_bucket"
            assert batch.metadata["symbols_analyzed"] == 2
            assert batch.metadata["data_symbols"] == 2

    @pytest.mark.asyncio
    async def test_get_top_signals(self, mock_ohlcv_data, mock_recommendations):
        """Test getting top signals by confidence."""
        with patch("trading_system.integration.daily_signal_service.LiveDataFetcher") as mock_fetcher_class, patch(
            "trading_system.integration.daily_signal_service.LiveSignalGenerator"
        ) as mock_generator_class, patch(
            "trading_system.integration.daily_signal_service.load_universe"
        ) as mock_load_universe, patch(
            "trading_system.integration.daily_signal_service.load_strategies_from_run_config"
        ) as mock_load_strategies:
            # Setup mocks
            mock_fetcher = AsyncMock()
            mock_fetcher.fetch_daily_data = AsyncMock(return_value=mock_ohlcv_data)
            mock_fetcher_class.return_value = mock_fetcher

            mock_generator = MagicMock()
            mock_generator.generate_recommendations = AsyncMock(return_value=mock_recommendations)
            mock_generator_class.return_value = mock_generator

            mock_load_universe.return_value = ["AAPL", "MSFT"]
            mock_load_strategies.return_value = [MagicMock()]

            # Create service and generate signals
            service = DailySignalService()
            batch = await service.generate_daily_signals(asset_class="equity")

            # Get top signals
            top_signals = batch.get_top_signals(n=1)
            assert len(top_signals) == 1
            assert top_signals[0].symbol == "AAPL"  # Highest confidence
            assert top_signals[0].confidence == 0.85


class TestCanonicalContracts:
    """Test suite for canonical contract models."""

    def test_signal_validation(self):
        """Test Signal validation."""
        signal = Signal(
            symbol="AAPL",
            asset_class=AssetClass.EQUITY,
            timestamp=pd.Timestamp.now(),
            side="BUY",
            intent=SignalIntent.EXECUTE_NEXT_OPEN,
            confidence=0.75,
            entry_price=150.0,
            stop_price=145.0,
        )

        assert signal.symbol == "AAPL"
        assert signal.confidence == 0.75

        # Test invalid confidence
        with pytest.raises(ValueError):
            Signal(
                symbol="AAPL",
                asset_class=AssetClass.EQUITY,
                timestamp=pd.Timestamp.now(),
                side="BUY",
                intent=SignalIntent.EXECUTE_NEXT_OPEN,
                confidence=1.5,  # Invalid
                entry_price=150.0,
                stop_price=145.0,
            )

    def test_allocation_validation(self):
        """Test Allocation validation."""
        allocation = Allocation(
            symbol="AAPL",
            signal_timestamp=pd.Timestamp.now(),
            recommended_position_size_dollars=10000.0,
            recommended_position_size_percent=2.0,
            risk_budget_used=1.0,
            max_positions_constraint_applied=False,
            quantity=65,
        )

        assert allocation.recommended_position_size_dollars == 10000.0
        assert allocation.quantity == 65

        # Test invalid position size
        with pytest.raises(ValueError):
            Allocation(
                symbol="AAPL",
                signal_timestamp=pd.Timestamp.now(),
                recommended_position_size_dollars=-1000.0,  # Invalid
                recommended_position_size_percent=2.0,
                risk_budget_used=1.0,
                max_positions_constraint_applied=False,
            )

    def test_trade_plan_validation(self):
        """Test TradePlan validation."""
        trade_plan = TradePlan(
            symbol="AAPL",
            signal_timestamp=pd.Timestamp.now(),
            entry_method=OrderMethod.MOO,
            entry_price=150.0,
            stop_logic=StopLogicType.ATR_TRAILING,
            stop_price=145.0,
            exit_logic="ma_cross",
        )

        assert trade_plan.entry_method == OrderMethod.MOO
        assert trade_plan.stop_logic == StopLogicType.ATR_TRAILING

        # Test invalid entry price
        with pytest.raises(ValueError):
            TradePlan(
                symbol="AAPL",
                signal_timestamp=pd.Timestamp.now(),
                entry_method=OrderMethod.MOO,
                entry_price=0.0,  # Invalid
                stop_logic=StopLogicType.ATR_TRAILING,
                stop_price=145.0,
                exit_logic="ma_cross",
            )

    def test_daily_signal_batch(self):
        """Test DailySignalBatch functionality."""
        signals = [
            Signal(
                symbol="AAPL",
                asset_class=AssetClass.EQUITY,
                timestamp=pd.Timestamp.now(),
                side="BUY",
                intent=SignalIntent.EXECUTE_NEXT_OPEN,
                confidence=0.85,
                entry_price=150.0,
                stop_price=145.0,
                bucket="test_bucket",
            ),
            Signal(
                symbol="MSFT",
                asset_class=AssetClass.EQUITY,
                timestamp=pd.Timestamp.now(),
                side="BUY",
                intent=SignalIntent.EXECUTE_NEXT_OPEN,
                confidence=0.65,
                entry_price=305.0,
                stop_price=295.0,
                bucket="test_bucket",
            ),
        ]

        batch = DailySignalBatch(generation_date=pd.Timestamp.now(), signals=signals)

        # Test get_signals_by_bucket
        bucket_signals = batch.get_signals_by_bucket("test_bucket")
        assert len(bucket_signals) == 2

        # Test get_top_signals
        top_signals = batch.get_top_signals(n=1)
        assert len(top_signals) == 1
        assert top_signals[0].symbol == "AAPL"  # Highest confidence

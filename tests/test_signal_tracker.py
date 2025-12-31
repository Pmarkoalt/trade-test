"""Unit tests for SignalTracker."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import pytest

from trading_system.tracking.models import (
    ConvictionLevel,
    SignalDirection,
    SignalStatus,
)
from trading_system.tracking.storage.sqlite_store import SQLiteTrackingStore
from trading_system.tracking.signal_tracker import SignalTracker


# Define Recommendation locally to avoid import issues
@dataclass
class Recommendation:
    """A trading recommendation to deliver to user."""

    id: str
    symbol: str
    asset_class: str  # 'equity' or 'crypto'
    direction: str  # 'BUY' or 'SELL'
    conviction: str  # 'HIGH', 'MEDIUM', 'LOW'

    # Prices
    current_price: float
    entry_price: float  # Expected fill price (next open)
    target_price: float
    stop_price: float

    # Sizing
    position_size_pct: float
    risk_pct: float

    # Scores
    technical_score: float

    # Context
    signal_type: str  # 'breakout_20d', 'breakout_55d', etc.
    reasoning: str

    # Optional scores
    news_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    combined_score: float = 0.0

    # Optional metadata
    news_headlines: List[str] = field(default_factory=list)
    news_reasoning: Optional[str] = None
    news_sentiment: Optional[str] = None  # 'positive', 'negative', 'neutral'
    generated_at: datetime = field(default_factory=datetime.now)
    strategy_name: Optional[str] = None


class TestSignalTracker:
    """Tests for SignalTracker."""

    def test_record_signal(self, tmp_path):
        """Test recording a new signal."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        tracker = SignalTracker(store)

        signal_id = tracker.record_signal(
            symbol="AAPL",
            asset_class="equity",
            direction=SignalDirection.BUY,
            signal_type="breakout_20d",
            conviction=ConvictionLevel.HIGH,
            signal_price=150.0,
            entry_price=150.0,
            target_price=165.0,
            stop_price=145.0,
        )

        assert signal_id is not None
        signal = tracker.get_signal(signal_id)
        assert signal.symbol == "AAPL"
        assert signal.status == SignalStatus.PENDING
        assert signal.direction == SignalDirection.BUY
        assert signal.conviction == ConvictionLevel.HIGH
        assert signal.entry_price == 150.0
        store.close()

    def test_record_signal_with_all_fields(self, tmp_path):
        """Test recording signal with all optional fields."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        tracker = SignalTracker(store)

        signal_id = tracker.record_signal(
            symbol="BTC",
            asset_class="crypto",
            direction=SignalDirection.BUY,
            signal_type="breakout_55d",
            conviction=ConvictionLevel.MEDIUM,
            signal_price=50000.0,
            entry_price=50050.0,
            target_price=55000.0,
            stop_price=48000.0,
            technical_score=0.85,
            news_score=0.75,
            combined_score=0.80,
            position_size_pct=5.0,
            reasoning="Strong breakout with positive news",
            news_headlines=["Bitcoin adoption increases"],
            tags=["momentum", "crypto"],
        )

        signal = tracker.get_signal(signal_id)
        assert signal.technical_score == 0.85
        assert signal.news_score == 0.75
        assert signal.combined_score == 0.80
        assert signal.position_size_pct == 5.0
        assert signal.reasoning == "Strong breakout with positive news"
        assert signal.news_headlines == ["Bitcoin adoption increases"]
        assert signal.tags == ["momentum", "crypto"]

        store.close()

    def test_mark_delivered(self, tmp_path):
        """Test delivery marking."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        tracker = SignalTracker(store)

        signal_id = tracker.record_signal(
            symbol="AAPL",
            asset_class="equity",
            direction=SignalDirection.BUY,
            signal_type="breakout",
            conviction=ConvictionLevel.MEDIUM,
            signal_price=150.0,
            entry_price=150.0,
            target_price=165.0,
            stop_price=145.0,
        )

        # Mark as delivered
        timestamp = datetime.now()
        success = tracker.mark_delivered(signal_id, method="email", timestamp=timestamp)
        assert success is True

        signal = tracker.get_signal(signal_id)
        assert signal.was_delivered is True
        assert signal.delivery_method == "email"
        assert signal.delivered_at == timestamp

        # Try to mark non-existent signal
        success = tracker.mark_delivered("non-existent-id")
        assert success is False

        store.close()

    def test_signal_lifecycle(self, tmp_path):
        """Test full signal lifecycle: pending -> active -> closed."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        tracker = SignalTracker(store)

        # Create signal
        signal_id = tracker.record_signal(
            symbol="AAPL",
            asset_class="equity",
            direction=SignalDirection.BUY,
            signal_type="breakout",
            conviction=ConvictionLevel.MEDIUM,
            signal_price=150.0,
            entry_price=150.0,
            target_price=165.0,
            stop_price=145.0,
        )

        # Check initial state
        signal = tracker.get_signal(signal_id)
        assert signal.status == SignalStatus.PENDING

        # Mark as delivered
        tracker.mark_delivered(signal_id, method="email")
        signal = tracker.get_signal(signal_id)
        assert signal.was_delivered is True

        # Mark entry filled (position opened)
        entry_timestamp = datetime.now()
        success = tracker.mark_entry_filled(signal_id, timestamp=entry_timestamp)
        assert success is True

        signal = tracker.get_signal(signal_id)
        assert signal.status == SignalStatus.ACTIVE
        assert signal.entry_filled_at == entry_timestamp

        # Mark expired (alternative path)
        expired_id = tracker.record_signal(
            symbol="MSFT",
            asset_class="equity",
            direction=SignalDirection.BUY,
            signal_type="breakout",
            conviction=ConvictionLevel.LOW,
            signal_price=200.0,
            entry_price=200.0,
            target_price=220.0,
            stop_price=190.0,
        )
        success = tracker.mark_expired(expired_id)
        assert success is True

        expired_signal = tracker.get_signal(expired_id)
        assert expired_signal.status == SignalStatus.EXPIRED

        store.close()

    def test_record_from_recommendation(self, tmp_path):
        """Test recording signal from Recommendation object."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        tracker = SignalTracker(store)

        recommendation = Recommendation(
            id="rec-123",
            symbol="AAPL",
            asset_class="equity",
            direction="BUY",
            conviction="HIGH",
            current_price=150.0,
            entry_price=150.0,
            target_price=165.0,
            stop_price=145.0,
            position_size_pct=5.0,
            risk_pct=3.0,
            technical_score=0.85,
            news_score=0.75,
            combined_score=0.80,
            signal_type="breakout_20d",
            reasoning="Strong momentum breakout",
            news_headlines=["Apple announces new product"],
        )

        signal_id = tracker.record_from_recommendation(recommendation)

        signal = tracker.get_signal(signal_id)
        assert signal.symbol == "AAPL"
        assert signal.direction == SignalDirection.BUY
        assert signal.conviction == ConvictionLevel.HIGH
        assert signal.signal_price == 150.0  # current_price -> signal_price
        assert signal.entry_price == 150.0
        assert signal.technical_score == 0.85
        assert signal.news_score == 0.75
        assert signal.combined_score == 0.80
        assert signal.reasoning == "Strong momentum breakout"
        assert signal.news_headlines == ["Apple announces new product"]

        store.close()

    def test_record_from_recommendation_with_sentiment_score(self, tmp_path):
        """Test recording from Recommendation with sentiment_score but no news_score."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        tracker = SignalTracker(store)

        recommendation = Recommendation(
            id="rec-456",
            symbol="BTC",
            asset_class="crypto",
            direction="BUY",
            conviction="MEDIUM",
            current_price=50000.0,
            entry_price=50050.0,
            target_price=55000.0,
            stop_price=48000.0,
            position_size_pct=3.0,
            risk_pct=4.0,
            technical_score=0.70,
            news_score=None,  # No news_score
            sentiment_score=0.65,  # But has sentiment_score
            combined_score=0.675,
            signal_type="breakout_55d",
            reasoning="Crypto momentum",
        )

        signal_id = tracker.record_from_recommendation(recommendation)

        signal = tracker.get_signal(signal_id)
        # sentiment_score should be used as news_score
        assert signal.news_score == 0.65

        store.close()

    def test_get_pending_signals(self, tmp_path):
        """Test getting pending signals."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        tracker = SignalTracker(store)

        # Create multiple signals
        for i in range(5):
            tracker.record_signal(
                symbol=f"SYM{i}",
                asset_class="equity",
                direction=SignalDirection.BUY,
                signal_type="breakout",
                conviction=ConvictionLevel.MEDIUM,
                signal_price=100.0 + i,
                entry_price=100.0 + i,
                target_price=110.0 + i,
                stop_price=95.0 + i,
            )

        # Mark some as active
        pending = tracker.get_pending_signals()
        assert len(pending) == 5

        # Mark one as active
        if pending:
            tracker.mark_entry_filled(pending[0].id)
            pending_after = tracker.get_pending_signals()
            assert len(pending_after) == 4

        store.close()

    def test_get_active_signals(self, tmp_path):
        """Test getting active signals."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        tracker = SignalTracker(store)

        # Create and activate some signals
        signal_ids = []
        for i in range(3):
            signal_id = tracker.record_signal(
                symbol=f"SYM{i}",
                asset_class="equity",
                direction=SignalDirection.BUY,
                signal_type="breakout",
                conviction=ConvictionLevel.MEDIUM,
                signal_price=100.0,
                entry_price=100.0,
                target_price=110.0,
                stop_price=95.0,
            )
            signal_ids.append(signal_id)
            tracker.mark_entry_filled(signal_id)

        active = tracker.get_active_signals()
        assert len(active) == 3
        assert all(s.status == SignalStatus.ACTIVE for s in active)

        store.close()

    def test_get_recent_signals(self, tmp_path):
        """Test getting recent signals."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        tracker = SignalTracker(store)

        # Create signals
        for i in range(5):
            tracker.record_signal(
                symbol=f"SYM{i}",
                asset_class="equity",
                direction=SignalDirection.BUY,
                signal_type="breakout",
                conviction=ConvictionLevel.MEDIUM,
                signal_price=100.0,
                entry_price=100.0,
                target_price=110.0,
                stop_price=95.0,
            )

        recent = tracker.get_recent_signals(days=7)
        assert len(recent) == 5

        # Filter by symbol
        recent_symbol = tracker.get_recent_signals(days=7, symbol="SYM0")
        assert len(recent_symbol) == 1
        assert recent_symbol[0].symbol == "SYM0"

        store.close()

    def test_get_signal_counts(self, tmp_path):
        """Test getting signal counts by status."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        tracker = SignalTracker(store)

        # Create signals with different statuses
        pending_ids = []
        for i in range(3):
            signal_id = tracker.record_signal(
                symbol=f"PENDING{i}",
                asset_class="equity",
                direction=SignalDirection.BUY,
                signal_type="breakout",
                conviction=ConvictionLevel.MEDIUM,
                signal_price=100.0,
                entry_price=100.0,
                target_price=110.0,
                stop_price=95.0,
            )
            pending_ids.append(signal_id)

        active_ids = []
        for i in range(2):
            signal_id = tracker.record_signal(
                symbol=f"ACTIVE{i}",
                asset_class="equity",
                direction=SignalDirection.BUY,
                signal_type="breakout",
                conviction=ConvictionLevel.MEDIUM,
                signal_price=100.0,
                entry_price=100.0,
                target_price=110.0,
                stop_price=95.0,
            )
            tracker.mark_entry_filled(signal_id)
            active_ids.append(signal_id)

        counts = tracker.get_signal_counts()
        assert counts["pending"] == 3
        assert counts["active"] == 2

        store.close()

    def test_mark_entry_filled_with_fill_price(self, tmp_path):
        """Test marking entry filled with fill price (for slippage tracking)."""
        store = SQLiteTrackingStore(str(tmp_path / "test.db"))
        store.initialize()
        tracker = SignalTracker(store)

        signal_id = tracker.record_signal(
            symbol="AAPL",
            asset_class="equity",
            direction=SignalDirection.BUY,
            signal_type="breakout",
            conviction=ConvictionLevel.MEDIUM,
            signal_price=150.0,
            entry_price=150.0,
            target_price=165.0,
            stop_price=145.0,
        )

        # Note: fill_price parameter is accepted but not stored in TrackedSignal
        # This is for future enhancement - the method signature supports it
        success = tracker.mark_entry_filled(signal_id, fill_price=150.5)
        assert success is True

        signal = tracker.get_signal(signal_id)
        assert signal.status == SignalStatus.ACTIVE

        store.close()

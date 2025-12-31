"""Data models for performance tracking."""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional
from enum import Enum
import uuid


class SignalDirection(str, Enum):
    """Signal direction."""

    BUY = "BUY"
    SELL = "SELL"


class ConvictionLevel(str, Enum):
    """Signal conviction level."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class SignalStatus(str, Enum):
    """Signal lifecycle status."""

    PENDING = "pending"  # Generated, waiting for entry
    ACTIVE = "active"  # Position entered
    CLOSED = "closed"  # Position exited
    EXPIRED = "expired"  # Never entered, signal expired
    CANCELLED = "cancelled"  # Manually cancelled


class ExitReason(str, Enum):
    """Reason for exiting a position."""

    TARGET_HIT = "target_hit"
    STOP_HIT = "stop_hit"
    TRAILING_STOP = "trailing_stop"
    TIME_EXIT = "time_exit"
    MANUAL = "manual"
    SIGNAL_REVERSAL = "signal_reversal"


@dataclass
class TrackedSignal:
    """A signal tracked for performance measurement."""

    # Identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Signal details
    symbol: str = ""
    asset_class: str = ""  # "equity" or "crypto"
    direction: SignalDirection = SignalDirection.BUY
    signal_type: str = ""  # e.g., "breakout_20d", "news_sentiment"
    conviction: ConvictionLevel = ConvictionLevel.MEDIUM

    # Prices at signal generation
    signal_price: float = 0.0  # Price when signal generated
    entry_price: float = 0.0  # Recommended entry price
    target_price: float = 0.0  # Target price
    stop_price: float = 0.0  # Stop loss price

    # Scores
    technical_score: float = 0.0
    news_score: Optional[float] = None
    combined_score: float = 0.0

    # Sizing
    position_size_pct: float = 0.0  # % of portfolio

    # Status
    status: SignalStatus = SignalStatus.PENDING

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    delivered_at: Optional[datetime] = None
    entry_filled_at: Optional[datetime] = None
    exit_filled_at: Optional[datetime] = None

    # Delivery
    was_delivered: bool = False
    delivery_method: str = ""  # "email", "sms", "push"

    # Metadata
    reasoning: str = ""
    news_headlines: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "asset_class": self.asset_class,
            "direction": self.direction.value,
            "signal_type": self.signal_type,
            "conviction": self.conviction.value,
            "signal_price": self.signal_price,
            "entry_price": self.entry_price,
            "target_price": self.target_price,
            "stop_price": self.stop_price,
            "technical_score": self.technical_score,
            "news_score": self.news_score,
            "combined_score": self.combined_score,
            "position_size_pct": self.position_size_pct,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "entry_filled_at": self.entry_filled_at.isoformat() if self.entry_filled_at else None,
            "exit_filled_at": self.exit_filled_at.isoformat() if self.exit_filled_at else None,
            "was_delivered": self.was_delivered,
            "delivery_method": self.delivery_method,
            "reasoning": self.reasoning,
            "news_headlines": self.news_headlines,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TrackedSignal":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            symbol=data["symbol"],
            asset_class=data["asset_class"],
            direction=SignalDirection(data["direction"]),
            signal_type=data["signal_type"],
            conviction=ConvictionLevel(data["conviction"]),
            signal_price=data["signal_price"],
            entry_price=data["entry_price"],
            target_price=data["target_price"],
            stop_price=data["stop_price"],
            technical_score=data["technical_score"],
            news_score=data.get("news_score"),
            combined_score=data["combined_score"],
            position_size_pct=data["position_size_pct"],
            status=SignalStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            delivered_at=datetime.fromisoformat(data["delivered_at"]) if data.get("delivered_at") else None,
            entry_filled_at=datetime.fromisoformat(data["entry_filled_at"]) if data.get("entry_filled_at") else None,
            exit_filled_at=datetime.fromisoformat(data["exit_filled_at"]) if data.get("exit_filled_at") else None,
            was_delivered=data["was_delivered"],
            delivery_method=data["delivery_method"],
            reasoning=data["reasoning"],
            news_headlines=data.get("news_headlines", []),
            tags=data.get("tags", []),
        )


@dataclass
class SignalOutcome:
    """Outcome of a tracked signal."""

    # Link to signal
    signal_id: str = ""

    # Actual execution
    actual_entry_price: Optional[float] = None
    actual_entry_date: Optional[date] = None
    actual_exit_price: Optional[float] = None
    actual_exit_date: Optional[date] = None

    # Trade result
    exit_reason: Optional[ExitReason] = None
    holding_days: int = 0

    # Returns
    return_pct: float = 0.0  # Percentage return
    return_dollars: float = 0.0  # Dollar return (if position size known)
    r_multiple: float = 0.0  # Return in R-multiples

    # Benchmark comparison
    benchmark_return_pct: float = 0.0  # SPY/BTC return over same period
    alpha: float = 0.0  # Signal return - benchmark return

    # User feedback
    was_followed: bool = False  # Did user take the trade?
    user_notes: str = ""

    # Timestamps
    recorded_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "signal_id": self.signal_id,
            "actual_entry_price": self.actual_entry_price,
            "actual_entry_date": self.actual_entry_date.isoformat() if self.actual_entry_date else None,
            "actual_exit_price": self.actual_exit_price,
            "actual_exit_date": self.actual_exit_date.isoformat() if self.actual_exit_date else None,
            "exit_reason": self.exit_reason.value if self.exit_reason else None,
            "holding_days": self.holding_days,
            "return_pct": self.return_pct,
            "return_dollars": self.return_dollars,
            "r_multiple": self.r_multiple,
            "benchmark_return_pct": self.benchmark_return_pct,
            "alpha": self.alpha,
            "was_followed": self.was_followed,
            "user_notes": self.user_notes,
            "recorded_at": self.recorded_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SignalOutcome":
        """Create from dictionary."""
        return cls(
            signal_id=data["signal_id"],
            actual_entry_price=data.get("actual_entry_price"),
            actual_entry_date=date.fromisoformat(data["actual_entry_date"]) if data.get("actual_entry_date") else None,
            actual_exit_price=data.get("actual_exit_price"),
            actual_exit_date=date.fromisoformat(data["actual_exit_date"]) if data.get("actual_exit_date") else None,
            exit_reason=ExitReason(data["exit_reason"]) if data.get("exit_reason") else None,
            holding_days=data.get("holding_days", 0),
            return_pct=data.get("return_pct", 0.0),
            return_dollars=data.get("return_dollars", 0.0),
            r_multiple=data.get("r_multiple", 0.0),
            benchmark_return_pct=data.get("benchmark_return_pct", 0.0),
            alpha=data.get("alpha", 0.0),
            was_followed=data.get("was_followed", False),
            user_notes=data.get("user_notes", ""),
            recorded_at=datetime.fromisoformat(data["recorded_at"]) if data.get("recorded_at") else datetime.now(),
        )


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics."""

    # Period
    period_start: date = field(default_factory=date.today)
    period_end: date = field(default_factory=date.today)

    # Counts
    total_signals: int = 0
    signals_followed: int = 0
    signals_won: int = 0
    signals_lost: int = 0

    # Rates
    win_rate: float = 0.0  # signals_won / total closed
    follow_rate: float = 0.0  # signals_followed / total delivered

    # Returns
    total_return_pct: float = 0.0
    avg_return_pct: float = 0.0
    avg_winner_pct: float = 0.0
    avg_loser_pct: float = 0.0

    # R-multiples
    total_r: float = 0.0
    avg_r: float = 0.0
    avg_winner_r: float = 0.0
    avg_loser_r: float = 0.0
    expectancy_r: float = 0.0  # (win_rate * avg_winner_r) - (loss_rate * abs(avg_loser_r))

    # Risk metrics
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Benchmark comparison
    benchmark_return_pct: float = 0.0
    alpha: float = 0.0

    # By category
    metrics_by_asset_class: Dict[str, Dict] = field(default_factory=dict)
    metrics_by_signal_type: Dict[str, Dict] = field(default_factory=dict)
    metrics_by_conviction: Dict[str, Dict] = field(default_factory=dict)


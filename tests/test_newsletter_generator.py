"""Tests for newsletter generator."""

from datetime import date

import pandas as pd
import pytest

from trading_system.models.signals import Signal, SignalSide, SignalType
from trading_system.output.email.newsletter_generator import NewsletterGenerator


@pytest.fixture
def newsletter_generator():
    """Create newsletter generator instance."""
    return NewsletterGenerator()


@pytest.fixture
def sample_signals():
    """Create sample signals for testing."""
    return {
        "safe_sp": [
            Signal(
                symbol="AAPL",
                asset_class="equity",
                date=pd.Timestamp("2024-01-15"),
                side=SignalSide.BUY,
                signal_type=SignalType.ENTRY_LONG,
                trigger_reason="momentum_breakout_20D",
                entry_price=150.0,
                stop_price=145.0,
                score=0.85,
                urgency=0.7,
                adv20=1000000.0,
                breakout_strength=0.8,
                momentum_strength=0.75,
                diversification_bonus=0.6,
            ),
            Signal(
                symbol="MSFT",
                asset_class="equity",
                date=pd.Timestamp("2024-01-15"),
                side=SignalSide.BUY,
                signal_type=SignalType.ENTRY_LONG,
                trigger_reason="technical_breakout",
                entry_price=380.0,
                stop_price=370.0,
                score=0.78,
                urgency=0.6,
                adv20=2000000.0,
            ),
        ],
        "crypto_topCap": [
            Signal(
                symbol="BTC",
                asset_class="crypto",
                date=pd.Timestamp("2024-01-15"),
                side=SignalSide.BUY,
                signal_type=SignalType.ENTRY_LONG,
                trigger_reason="crypto_momentum",
                entry_price=45000.0,
                stop_price=43500.0,
                score=0.82,
                urgency=0.8,
                adv20=5000000.0,
            ),
        ],
    }


def test_newsletter_generator_init(newsletter_generator):
    """Test newsletter generator initialization."""
    assert newsletter_generator is not None


def test_generate_newsletter_context(newsletter_generator, sample_signals):
    """Test newsletter context generation."""
    context = newsletter_generator.generate_newsletter_context(
        signals_by_bucket=sample_signals,
        market_summary={"spy_price": 450.0, "spy_pct": 0.5},
        date_obj=date(2024, 1, 15),
    )

    assert context is not None
    assert "bucket_sections" in context
    assert "total_signals" in context
    assert "market" in context
    assert "date" in context
    assert "generated_at" in context

    # Check total signals
    assert context["total_signals"] == 3

    # Check bucket sections
    assert len(context["bucket_sections"]) == 2
    bucket_names = [section["name"] for section in context["bucket_sections"]]
    assert "safe_sp" in bucket_names
    assert "crypto_topCap" in bucket_names


def test_bucket_description(newsletter_generator):
    """Test bucket description generation."""
    desc = newsletter_generator._get_bucket_description("safe_sp")
    assert "Safe S&P 500" in desc

    desc = newsletter_generator._get_bucket_description("crypto_topCap")
    assert "crypto" in desc.lower()

    desc = newsletter_generator._get_bucket_description("unknown_bucket")
    assert desc == "unknown_bucket"


def test_format_signal_for_email(newsletter_generator, sample_signals):
    """Test signal formatting for email."""
    signal = sample_signals["safe_sp"][0]
    formatted = newsletter_generator.format_signal_for_email(signal, index=1)

    assert formatted["index"] == 1
    assert formatted["symbol"] == "AAPL"
    assert formatted["side"] == "BUY"
    assert formatted["entry_price"] == 150.0
    assert formatted["stop_price"] == 145.0
    assert formatted["score"] == 0.85
    assert formatted["asset_class"] == "equity"
    assert "risk_pct" in formatted
    assert formatted["risk_pct"] > 0


def test_generate_plain_text_summary(newsletter_generator, sample_signals):
    """Test plain text summary generation."""
    summary = newsletter_generator.generate_plain_text_summary(
        signals_by_bucket=sample_signals,
        date_obj=date(2024, 1, 15),
    )

    assert summary is not None
    assert "Daily Trading Signals" in summary
    assert "January 15, 2024" in summary
    assert "AAPL" in summary
    assert "BTC" in summary
    assert "Total: 3 signals" in summary


def test_empty_signals(newsletter_generator):
    """Test newsletter generation with empty signals."""
    context = newsletter_generator.generate_newsletter_context(
        signals_by_bucket={},
        date_obj=date(2024, 1, 15),
    )

    assert context["total_signals"] == 0
    assert len(context["bucket_sections"]) == 0


def test_mixed_valid_invalid_signals(newsletter_generator):
    """Test newsletter with mix of valid and invalid signals."""
    signals = {
        "safe_sp": [
            Signal(
                symbol="AAPL",
                asset_class="equity",
                date=pd.Timestamp("2024-01-15"),
                side=SignalSide.BUY,
                signal_type=SignalType.ENTRY_LONG,
                trigger_reason="test",
                entry_price=150.0,
                stop_price=145.0,
                score=0.85,
                adv20=1000000.0,
                passed_eligibility=True,
                capacity_passed=True,
            ),
            Signal(
                symbol="MSFT",
                asset_class="equity",
                date=pd.Timestamp("2024-01-15"),
                side=SignalSide.BUY,
                signal_type=SignalType.ENTRY_LONG,
                trigger_reason="test",
                entry_price=380.0,
                stop_price=370.0,
                score=0.78,
                adv20=2000000.0,
                passed_eligibility=False,  # Invalid
                capacity_passed=True,
            ),
        ],
    }

    context = newsletter_generator.generate_newsletter_context(
        signals_by_bucket=signals,
        date_obj=date(2024, 1, 15),
    )

    # Only valid signals should be counted
    assert context["total_signals"] == 1
    assert len(context["bucket_sections"][0]["buy_signals"]) == 1


def test_signal_metadata_in_rationale(newsletter_generator):
    """Test that signal metadata is included in rationale."""
    signal = Signal(
        symbol="AAPL",
        asset_class="equity",
        date=pd.Timestamp("2024-01-15"),
        side=SignalSide.BUY,
        signal_type=SignalType.ENTRY_LONG,
        trigger_reason="momentum_breakout",
        entry_price=150.0,
        stop_price=145.0,
        score=0.85,
        adv20=1000000.0,
        metadata={
            "technical_reason": "20D breakout with strong volume",
            "news_reason": "Positive earnings report",
        },
    )

    formatted = newsletter_generator.format_signal_for_email(signal, index=1)
    rationale = formatted["rationale"]

    assert "momentum_breakout" in rationale
    assert "20D breakout with strong volume" in rationale
    assert "Positive earnings report" in rationale

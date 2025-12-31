"""Tests for ML refinement feature extractors."""

import pandas as pd
import pytest

from trading_system.ml_refinement.features.extractors import (
    BaseFeatureExtractor,
    MomentumFeatures,
    OHLCVExtractor,
    SignalMetadataFeatures,
    TrendFeatures,
    VolatilityFeatures,
    MarketRegimeFeatures,
)


def test_trend_features():
    """Test trend feature extraction."""
    ohlcv = pd.DataFrame(
        {
            "open": [100] * 250,
            "high": [101] * 250,
            "low": [99] * 250,
            "close": list(range(100, 350)),  # Uptrend
            "volume": [1000] * 250,
        }
    )

    extractor = TrendFeatures()
    features = extractor.extract(ohlcv)

    assert "price_vs_ma20" in features
    assert features["price_vs_ma20"] > 0  # Price above MA in uptrend
    assert "trend_strength" in features
    assert "higher_highs" in features
    assert "lower_lows" in features
    assert extractor.validate_output(features)


def test_trend_features_insufficient_data():
    """Test trend features with insufficient data."""
    ohlcv = pd.DataFrame(
        {
            "open": [100] * 10,
            "high": [101] * 10,
            "low": [99] * 10,
            "close": [100] * 10,
            "volume": [1000] * 10,
        }
    )

    extractor = TrendFeatures()
    features = extractor.extract(ohlcv)

    # Should return zeros for features requiring more data
    assert features["price_vs_ma200"] == 0.0
    assert features["ma200_slope"] == 0.0


def test_momentum_features_rsi():
    """Test RSI calculation."""
    # Create data with clear uptrend (RSI should be high)
    ohlcv = pd.DataFrame(
        {
            "open": [100] * 50,
            "high": [102] * 50,
            "low": [99] * 50,
            "close": list(range(100, 150)),  # Strong uptrend
            "volume": [1000] * 50,
        }
    )

    extractor = MomentumFeatures()
    features = extractor.extract(ohlcv)

    assert "rsi_14" in features
    assert 0 <= features["rsi_14"] <= 1  # Normalized
    assert "rsi_deviation" in features
    assert -1 <= features["rsi_deviation"] <= 1
    assert "roc_5" in features
    assert "momentum_divergence" in features
    assert "acceleration" in features
    assert extractor.validate_output(features)


def test_volatility_features():
    """Test volatility feature extraction."""
    ohlcv = pd.DataFrame(
        {
            "open": [100, 101, 99, 102, 98],
            "high": [102, 103, 101, 104, 100],
            "low": [98, 99, 97, 100, 96],
            "close": [101, 100, 102, 99, 101],
            "volume": [1000] * 5,
        }
    )

    extractor = VolatilityFeatures()
    features = extractor.extract(ohlcv)

    assert "atr_ratio" in features
    assert "volatility_percentile" in features
    assert "volatility_trend" in features
    assert "range_ratio" in features
    assert "gap_size" in features
    assert "intraday_volatility" in features
    assert extractor.validate_output(features)


def test_signal_features():
    """Test signal metadata extraction."""
    signal_data = {
        "technical_score": 8.0,
        "conviction": "HIGH",
        "entry_price": 100,
        "target_price": 110,
        "stop_price": 95,
    }

    extractor = SignalMetadataFeatures()
    features = extractor.extract(signal_data)

    assert features["technical_score"] == 0.8  # Normalized
    assert features["conviction_high"] == 1.0
    assert features["conviction_medium"] == 0.0
    assert features["conviction_low"] == 0.0
    assert features["risk_reward_ratio"] == 2.0  # 10/5
    assert extractor.validate_output(features)


def test_signal_features_missing_values():
    """Test signal features handle missing values."""
    signal_data = {
        "technical_score": 5.0,
        # Missing other fields
    }

    extractor = SignalMetadataFeatures()
    features = extractor.extract(signal_data)

    # Should handle missing values gracefully
    assert features["news_score"] == 0.0
    assert features["combined_score"] == 0.0
    assert features["risk_reward_ratio"] == 0.0
    assert features["conviction_high"] == 0.0
    assert features["is_equity"] == 0.0
    assert extractor.validate_output(features)


def test_market_regime_features():
    """Test market regime feature extraction."""
    ohlcv = pd.DataFrame(
        {
            "open": [100] * 100,
            "high": [101] * 100,
            "low": [99] * 100,
            "close": list(range(100, 200)),  # Uptrend
            "volume": [1000] * 100,
        }
    )

    extractor = MarketRegimeFeatures()
    features = extractor.extract(ohlcv)

    assert "market_trend" in features
    assert -1 <= features["market_trend"] <= 1
    assert "market_breadth" in features
    assert 0 <= features["market_breadth"] <= 1
    assert "correlation_regime" in features
    assert "volatility_regime" in features
    assert 0 <= features["volatility_regime"] <= 1
    assert "drawdown_depth" in features
    assert "days_from_high" in features
    assert "rally_strength" in features
    assert extractor.validate_output(features)


def test_market_regime_with_benchmark():
    """Test market features work with benchmark data."""
    ohlcv = pd.DataFrame(
        {
            "open": [100] * 100,
            "high": [101] * 100,
            "low": [99] * 100,
            "close": list(range(100, 200)),
            "volume": [1000] * 100,
        }
    )

    benchmark = pd.DataFrame(
        {
            "open": [200] * 100,
            "high": [201] * 100,
            "low": [199] * 100,
            "close": list(range(200, 300)),
            "volume": [2000] * 100,
        }
    )

    extractor = MarketRegimeFeatures()
    features = extractor.extract(ohlcv, benchmark_data=benchmark)

    # Should use benchmark for market features
    assert "market_trend" in features
    assert "correlation_regime" in features
    assert extractor.validate_output(features)


def test_base_extractor_interface():
    """Test that all extractors implement base interface."""
    extractors = [
        TrendFeatures(),
        MomentumFeatures(),
        VolatilityFeatures(),
        MarketRegimeFeatures(),
        SignalMetadataFeatures(),
    ]

    for extractor in extractors:
        assert hasattr(extractor, "name")
        assert hasattr(extractor, "feature_names")
        assert hasattr(extractor, "category")
        assert hasattr(extractor, "extract")
        assert hasattr(extractor, "validate_output")

        # Test properties
        name = extractor.name
        assert isinstance(name, str)
        assert len(name) > 0

        feature_names = extractor.feature_names
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0

        category = extractor.category
        assert isinstance(category, str)


def test_ohlcv_extractor_safe_get():
    """Test safe_get method handles edge cases."""
    extractor = TrendFeatures()
    series = pd.Series([1.0, 2.0, 3.0])

    # Normal access
    assert extractor._safe_get(series, 0) == 1.0
    assert extractor._safe_get(series, -1) == 3.0

    # Out of bounds
    assert extractor._safe_get(series, 10) == 0.0
    assert extractor._safe_get(series, -10) == 0.0

    # With default
    assert extractor._safe_get(series, 10, default=5.0) == 5.0


def test_trend_features_normalized():
    """Test that trend features are normalized appropriately."""
    ohlcv = pd.DataFrame(
        {
            "open": [100] * 250,
            "high": [101] * 250,
            "low": [99] * 250,
            "close": list(range(100, 350)),
            "volume": [1000] * 250,
        }
    )

    extractor = TrendFeatures()
    features = extractor.extract(ohlcv)

    # Price vs MA should be normalized (percentage)
    assert isinstance(features["price_vs_ma20"], float)
    # Higher highs should be 0-1
    assert 0 <= features["higher_highs"] <= 1
    assert 0 <= features["lower_lows"] <= 1
    # Trend strength should be 0-1
    assert 0 <= features["trend_strength"] <= 1


def test_momentum_features_normalized():
    """Test that momentum features are normalized."""
    ohlcv = pd.DataFrame(
        {
            "open": [100] * 50,
            "high": [102] * 50,
            "low": [99] * 50,
            "close": list(range(100, 150)),
            "volume": [1000] * 50,
        }
    )

    extractor = MomentumFeatures()
    features = extractor.extract(ohlcv)

    # RSI should be normalized to 0-1
    assert 0 <= features["rsi_14"] <= 1
    # RSI deviation should be -1 to 1
    assert -1 <= features["rsi_deviation"] <= 1
    # ROC should be percentage
    assert isinstance(features["roc_5"], float)


def test_signal_features_one_hot():
    """Test one-hot encoding in signal features."""
    # Test HIGH conviction
    signal_data = {"conviction": "HIGH", "asset_class": "equity", "signal_type": "breakout"}
    extractor = SignalMetadataFeatures()
    features = extractor.extract(signal_data)
    assert features["conviction_high"] == 1.0
    assert features["conviction_medium"] == 0.0
    assert features["conviction_low"] == 0.0
    assert features["is_equity"] == 1.0
    assert features["is_crypto"] == 0.0
    assert features["is_breakout"] == 1.0
    assert features["is_momentum"] == 0.0

    # Test MEDIUM conviction
    signal_data = {"conviction": "MEDIUM", "asset_class": "crypto", "signal_type": "momentum"}
    features = extractor.extract(signal_data)
    assert features["conviction_high"] == 0.0
    assert features["conviction_medium"] == 1.0
    assert features["conviction_low"] == 0.0
    assert features["is_equity"] == 0.0
    assert features["is_crypto"] == 1.0
    assert features["is_breakout"] == 0.0
    assert features["is_momentum"] == 1.0


def test_volatility_features_edge_cases():
    """Test volatility features handle edge cases."""
    # Very small dataset
    ohlcv = pd.DataFrame(
        {
            "open": [100, 101],
            "high": [102, 103],
            "low": [98, 99],
            "close": [101, 102],
            "volume": [1000, 1000],
        }
    )

    extractor = VolatilityFeatures()
    features = extractor.extract(ohlcv)

    # Should return all features even with limited data
    assert "atr_ratio" in features
    assert "volatility_percentile" in features
    assert extractor.validate_output(features)


def test_market_regime_insufficient_data():
    """Test market regime with insufficient data."""
    ohlcv = pd.DataFrame(
        {
            "open": [100] * 10,
            "high": [101] * 10,
            "low": [99] * 10,
            "close": [100] * 10,
            "volume": [1000] * 10,
        }
    )

    extractor = MarketRegimeFeatures()
    features = extractor.extract(ohlcv)

    # Should handle insufficient data gracefully
    assert "market_trend" in features
    assert "market_breadth" in features
    # With insufficient data, should return defaults
    assert features["market_breadth"] == 0.5  # Default when insufficient data
    assert extractor.validate_output(features)

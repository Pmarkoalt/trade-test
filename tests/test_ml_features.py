"""Tests for enhanced ML feature engineering."""

import numpy as np
import pandas as pd
import pytest

from trading_system.ml.feature_engineering import MLFeatureEngineer
from trading_system.models.features import FeatureRow


@pytest.fixture
def sample_feature_rows():
    """Create sample FeatureRow objects for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    rows = []

    for i, date in enumerate(dates):
        close = 100 + i * 0.1
        row = FeatureRow(
            date=date,
            symbol="AAPL",
            asset_class="equity",
            close=close,
            open=close - 0.5,
            high=close + 1.0,
            low=close - 1.0,
            ma20=close - 0.5 if i >= 20 else None,
            ma50=close - 1.0 if i >= 50 else None,
            ma200=close - 2.0 if i >= 200 else None,
            ma50_slope=0.001 if i >= 70 else None,
            atr14=1.5 if i >= 14 else None,
            roc60=0.05 if i >= 60 else None,
            highest_close_20d=close + 0.5 if i >= 20 else None,
            highest_close_55d=close + 1.0 if i >= 55 else None,
            adv20=1e6 if i >= 20 else None,
            returns_1d=0.001,
            benchmark_roc60=0.03,
            benchmark_returns_1d=0.0005,
        )
        rows.append(row)

    return rows


def test_feature_engineer_basic(sample_feature_rows):
    """Test basic feature engineering."""
    engineer = MLFeatureEngineer(
        include_raw_features=True,
        include_derived_features=True,
        include_technical_indicators=False,
    )

    # Fit and transform
    df = engineer.fit_transform(sample_feature_rows)

    assert len(df) == len(sample_feature_rows)
    assert len(df.columns) > 0


def test_feature_engineer_technical_indicators(sample_feature_rows):
    """Test technical indicators (RSI, MACD, Bollinger Bands)."""
    engineer = MLFeatureEngineer(
        include_technical_indicators=True,
        rsi_period=14,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        bb_period=20,
        bb_std=2.0,
    )

    # Fit and transform
    df = engineer.fit_transform(sample_feature_rows)

    # Check that technical indicators are present
    assert "rsi" in df.columns
    assert "macd" in df.columns
    assert "macd_signal" in df.columns
    assert "macd_hist" in df.columns
    assert "bb_upper" in df.columns
    assert "bb_middle" in df.columns
    assert "bb_lower" in df.columns
    assert "bb_position" in df.columns


def test_feature_engineer_volatility_features(sample_feature_rows):
    """Test volatility features."""
    engineer = MLFeatureEngineer(
        include_volatility_features=True,
    )

    df = engineer.fit_transform(sample_feature_rows)

    assert "realized_vol" in df.columns
    assert "atr_pct" in df.columns


def test_feature_engineer_market_regime_features(sample_feature_rows):
    """Test market regime features."""
    engineer = MLFeatureEngineer(
        include_market_regime_features=True,
    )

    df = engineer.fit_transform(sample_feature_rows)

    assert "trend_bullish" in df.columns
    assert "trend_bearish" in df.columns
    assert "volatility_regime_high" in df.columns
    assert "volatility_regime_low" in df.columns


def test_feature_engineer_cross_asset_features(sample_feature_rows):
    """Test cross-asset features."""
    engineer = MLFeatureEngineer(
        include_cross_asset_features=True,
    )

    df = engineer.fit_transform(sample_feature_rows)

    assert "relative_strength_vs_benchmark" in df.columns
    assert "returns_correlation_sign" in df.columns


def test_feature_engineer_normalization(sample_feature_rows):
    """Test feature normalization."""
    engineer = MLFeatureEngineer(
        normalize_features=True,
    )

    df = engineer.fit_transform(sample_feature_rows)

    # Check that features are normalized (values between 0 and 1)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            assert col_data.min() >= 0.0
            assert col_data.max() <= 1.0


def test_feature_engineer_batch_transform(sample_feature_rows):
    """Test batch transformation."""
    engineer = MLFeatureEngineer()
    engineer.fit(sample_feature_rows)

    # Transform batch
    df_batch = engineer.transform_batch(sample_feature_rows)

    assert len(df_batch) == len(sample_feature_rows)

    # Transform single
    df_single = engineer.transform(sample_feature_rows[0])

    assert len(df_single) > 0


def test_feature_engineer_feature_names(sample_feature_rows):
    """Test getting feature names."""
    engineer = MLFeatureEngineer()
    engineer.fit(sample_feature_rows)

    feature_names = engineer.get_feature_names()

    assert len(feature_names) > 0
    assert isinstance(feature_names, list)

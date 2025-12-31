"""Test utilities for trading system tests."""

from .test_helpers import (
    create_sample_bar,
    create_sample_feature_row,
    create_sample_signal,
    create_sample_order,
    create_sample_fill,
    create_sample_portfolio,
    create_sample_position,
    create_mock_market_data,
    generate_ohlcv_data,
)
from .assertions import (
    assert_no_lookahead,
    assert_valid_bar,
    assert_valid_signal,
    assert_valid_order,
    assert_valid_fill,
    assert_valid_portfolio,
)
from .data_generator import (
    SyntheticDataGenerator,
    DataPattern,
    TrendType,
    BreakoutType,
    generate_trend_data,
    generate_breakout_data,
    generate_edge_case_data,
)

__all__ = [
    # Helpers
    "create_sample_bar",
    "create_sample_feature_row",
    "create_sample_signal",
    "create_sample_order",
    "create_sample_fill",
    "create_sample_portfolio",
    "create_sample_position",
    "create_mock_market_data",
    "generate_ohlcv_data",
    # Assertions
    "assert_no_lookahead",
    "assert_valid_bar",
    "assert_valid_signal",
    "assert_valid_order",
    "assert_valid_fill",
    "assert_valid_portfolio",
    # Data Generator
    "SyntheticDataGenerator",
    "DataPattern",
    "TrendType",
    "BreakoutType",
    "generate_trend_data",
    "generate_breakout_data",
    "generate_edge_case_data",
]


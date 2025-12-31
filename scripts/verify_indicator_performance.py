#!/usr/bin/env python3
"""Verify indicator computation performance requirements.

This script verifies:
1. Full feature computation: < 50ms per symbol (10K bars)
2. Scales linearly with symbol count
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path to import trading_system
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.indicators.feature_computer import compute_features


def create_test_data(n_bars: int = 10000) -> pd.DataFrame:
    """Create test OHLC data with specified number of bars."""
    dates = pd.date_range("2020-01-01", periods=n_bars, freq="D")
    base_price = 100.0
    returns = np.random.randn(n_bars) * 0.02
    
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_bars) * 0.001),
            "high": prices * (1 + abs(np.random.randn(n_bars)) * 0.005),
            "low": prices * (1 - abs(np.random.randn(n_bars)) * 0.005),
            "close": prices,
            "volume": np.random.randint(1000000, 10000000, n_bars),
        },
        index=dates,
    )
    
    df["dollar_volume"] = df["close"] * df["volume"]
    return df


def test_single_symbol_performance(n_bars: int = 10000, n_iterations: int = 10) -> dict:
    """Test performance for a single symbol with 10K bars.
    
    Returns:
        dict with 'avg_time_ms', 'min_time_ms', 'max_time_ms', 'meets_requirement'
    """
    print(f"\n{'='*60}")
    print(f"Testing single symbol performance ({n_bars:,} bars)")
    print(f"{'='*60}")
    
    times_ms = []
    
    for i in range(n_iterations):
        df = create_test_data(n_bars)
        
        start = time.perf_counter()
        result = compute_features(df, symbol="TEST", asset_class="equity", use_cache=False)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        times_ms.append(elapsed_ms)
        print(f"  Iteration {i+1}/{n_iterations}: {elapsed_ms:.2f} ms")
        
        assert len(result) == len(df), "Result length mismatch"
    
    avg_time = np.mean(times_ms)
    min_time = np.min(times_ms)
    max_time = np.max(times_ms)
    std_time = np.std(times_ms)
    
    meets_requirement = avg_time < 50.0
    
    print(f"\nResults:")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  Min:     {min_time:.2f} ms")
    print(f"  Max:     {max_time:.2f} ms")
    print(f"  Std:     {std_time:.2f} ms")
    print(f"  Target:  < 50.0 ms")
    print(f"  Status:  {'✅ PASS' if meets_requirement else '❌ FAIL'}")
    
    return {
        "avg_time_ms": avg_time,
        "min_time_ms": min_time,
        "max_time_ms": max_time,
        "std_time_ms": std_time,
        "meets_requirement": meets_requirement,
    }


def test_multi_symbol_scaling(n_symbols_list: list = [1, 5, 10, 20], n_bars: int = 10000) -> dict:
    """Test that computation scales linearly with symbol count.
    
    Returns:
        dict with 'times_per_symbol', 'scaling_factor', 'is_linear', 'meets_requirement'
    """
    print(f"\n{'='*60}")
    print(f"Testing multi-symbol scaling ({n_bars:,} bars per symbol)")
    print(f"{'='*60}")
    
    times_per_symbol = {}
    total_times = {}
    
    for n_symbols in n_symbols_list:
        print(f"\nTesting with {n_symbols} symbol(s)...")
        
        # Create data for all symbols
        symbols_data = {}
        for i in range(n_symbols):
            symbol = f"SYM{i:02d}"
            symbols_data[symbol] = create_test_data(n_bars)
        
        # Time the computation
        start = time.perf_counter()
        for symbol, df in symbols_data.items():
            compute_features(df, symbol=symbol, asset_class="equity", use_cache=False)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        total_times[n_symbols] = elapsed_ms
        times_per_symbol[n_symbols] = elapsed_ms / n_symbols
        
        print(f"  Total time: {elapsed_ms:.2f} ms")
        print(f"  Time per symbol: {times_per_symbol[n_symbols]:.2f} ms")
    
    # Check linearity: time per symbol should be roughly constant
    # (within 20% variation is acceptable)
    time_per_symbol_values = list(times_per_symbol.values())
    avg_time_per_symbol = np.mean(time_per_symbol_values)
    std_time_per_symbol = np.std(time_per_symbol_values)
    cv = std_time_per_symbol / avg_time_per_symbol if avg_time_per_symbol > 0 else 0
    
    # Also check that total time scales roughly linearly
    # Fit a linear model: total_time = a * n_symbols + b
    n_syms = np.array(list(total_times.keys()))
    total_times_arr = np.array(list(total_times.values()))
    
    # Simple linear fit: y = a*x
    # For perfect linearity, intercept should be near zero
    # We'll use least squares: a = sum(x*y) / sum(x^2)
    a = np.sum(n_syms * total_times_arr) / np.sum(n_syms ** 2)
    
    # Calculate R-squared to measure linearity
    y_pred = a * n_syms
    ss_res = np.sum((total_times_arr - y_pred) ** 2)
    ss_tot = np.sum((total_times_arr - np.mean(total_times_arr)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Check if scaling is linear (R² > 0.95 and CV < 0.20)
    is_linear = r_squared > 0.95 and cv < 0.20
    
    print(f"\nScaling Analysis:")
    print(f"  Average time per symbol: {avg_time_per_symbol:.2f} ms")
    print(f"  Coefficient of variation: {cv:.3f} (target: < 0.20)")
    print(f"  R-squared (linearity):   {r_squared:.3f} (target: > 0.95)")
    print(f"  Status: {'✅ PASS (linear scaling)' if is_linear else '❌ FAIL (non-linear scaling)'}")
    
    return {
        "times_per_symbol": times_per_symbol,
        "total_times": total_times,
        "avg_time_per_symbol": avg_time_per_symbol,
        "coefficient_of_variation": cv,
        "r_squared": r_squared,
        "is_linear": is_linear,
        "meets_requirement": is_linear,
    }


def main():
    """Run all performance verification tests."""
    print("="*60)
    print("Indicator Computation Performance Verification")
    print("="*60)
    print("\nRequirements:")
    print("  1. Full feature computation: < 50ms per symbol (10K bars)")
    print("  2. Scales linearly with symbol count")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Test 1: Single symbol performance
    single_result = test_single_symbol_performance(n_bars=10000, n_iterations=10)
    
    # Test 2: Multi-symbol scaling
    scaling_result = test_multi_symbol_scaling(n_symbols_list=[1, 5, 10, 20], n_bars=10000)
    
    # Summary
    print(f"\n{'='*60}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"\n1. Single Symbol Performance (< 50ms for 10K bars):")
    print(f"   Average: {single_result['avg_time_ms']:.2f} ms")
    print(f"   Status:  {'✅ PASS' if single_result['meets_requirement'] else '❌ FAIL'}")
    
    print(f"\n2. Linear Scaling with Symbol Count:")
    print(f"   R-squared: {scaling_result['r_squared']:.3f}")
    print(f"   CV: {scaling_result['coefficient_of_variation']:.3f}")
    print(f"   Status:  {'✅ PASS' if scaling_result['meets_requirement'] else '❌ FAIL'}")
    
    all_passed = single_result['meets_requirement'] and scaling_result['meets_requirement']
    
    print(f"\n{'='*60}")
    if all_passed:
        print("✅ ALL REQUIREMENTS MET")
    else:
        print("❌ SOME REQUIREMENTS NOT MET")
    print(f"{'='*60}\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())


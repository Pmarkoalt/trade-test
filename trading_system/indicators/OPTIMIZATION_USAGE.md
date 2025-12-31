# Indicator Optimization Usage Guide

This document explains how to use the optimization features added to the indicator calculations.

## Features

1. **Caching**: Avoid recomputing indicators for the same data
2. **Vectorization**: Optimized pandas operations
3. **Profiling**: Measure performance of indicator calculations
4. **Parallel Processing**: Compute features for multiple symbols in parallel

## Caching

Enable caching to avoid recomputing indicators when the same data is processed:

```python
from trading_system.indicators import enable_caching, compute_features
import pandas as pd

# Enable caching with default max size (128)
cache = enable_caching()

# Or with custom max size
cache = enable_caching(max_size=256)

# Compute features (caching is automatic)
df_ohlc = pd.DataFrame(...)  # Your OHLC data
features = compute_features(df_ohlc, 'AAPL', 'equity')

# Check cache statistics
stats = cache.get_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")

# Disable caching if needed
from trading_system.indicators import disable_caching
disable_caching()
```

## Profiling

Profile indicator calculations to identify bottlenecks:

```python
from trading_system.indicators import enable_profiling, compute_features

# Enable profiling
profiler = enable_profiling()

# Start detailed profiling (cProfile)
profiler.start_profiling()

# Run your computations
features = compute_features(df_ohlc, 'AAPL', 'equity')

# Stop and print detailed profile
print(profiler.stop_profiling())

# Get timing statistics
stats = profiler.get_stats()
profiler.print_stats()

# Reset profiler
profiler.reset()
```

## Parallel Processing

Compute features for multiple symbols in parallel:

```python
from trading_system.indicators import compute_features_parallel, compute_features

# Prepare data
symbols_data = {
    'AAPL': df_aapl,
    'MSFT': df_msft,
    'GOOGL': df_googl,
    # ... more symbols
}

asset_classes = {
    'AAPL': 'equity',
    'MSFT': 'equity',
    'GOOGL': 'equity',
}

# Compute features in parallel (using threads)
features_dict = compute_features_parallel(
    symbols_data,
    compute_features,
    asset_classes,
    max_workers=4,  # Use 4 threads
    use_threads=True
)

# Or use processes (better for CPU-bound tasks)
features_dict = compute_features_parallel(
    symbols_data,
    compute_features,
    asset_classes,
    max_workers=4,
    use_threads=False  # Use processes instead
)
```

## Batch Processing

Process symbols in batches (useful for memory management):

```python
from trading_system.indicators import batch_compute_features, compute_features

features_dict = batch_compute_features(
    symbols_data,
    compute_features,
    asset_classes,
    batch_size=10  # Process 10 symbols at a time
)
```

## Combined Usage

Use all optimizations together:

```python
from trading_system.indicators import (
    enable_caching,
    enable_profiling,
    compute_features_parallel,
    compute_features
)

# Enable optimizations
cache = enable_caching(max_size=256)
profiler = enable_profiling()

# Profile parallel computation
profiler.start_profiling()

features_dict = compute_features_parallel(
    symbols_data,
    compute_features,
    asset_classes,
    max_workers=4,
    use_cache=True  # Enable caching in parallel computation
)

profiler.stop_profiling()

# Check results
print("Cache stats:", cache.get_stats())
profiler.print_stats()
```

## Performance Tips

1. **Enable caching** when processing the same symbols repeatedly (e.g., in walk-forward backtests)
2. **Use parallel processing** when computing features for many symbols (>10)
3. **Profile first** to identify bottlenecks before optimizing
4. **Use threads** for I/O-bound operations, **processes** for CPU-bound operations
5. **Adjust batch size** based on available memory

## Disabling Optimizations

To disable optimizations (e.g., for debugging):

```python
from trading_system.indicators import disable_caching, disable_profiling

disable_caching()
disable_profiling()

# Or disable caching per call
features = compute_features(df_ohlc, 'AAPL', 'equity', use_cache=False)
```


"""Profiling utilities for indicator calculations."""

import cProfile
import io
import pstats
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional

import pandas as pd


class IndicatorProfiler:
    """Profiler for measuring indicator calculation performance."""

    def __init__(self):
        """Initialize profiler."""
        self.timings: Dict[str, list] = {}
        self.call_counts: Dict[str, int] = {}
        self.profiler: Optional[cProfile.Profile] = None

    def time_function(self, func_name: str):
        """Decorator to time a function.

        Args:
            func_name: Name to use for this function in profiling
        """

        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start

                if func_name not in self.timings:
                    self.timings[func_name] = []
                    self.call_counts[func_name] = 0

                self.timings[func_name].append(elapsed)
                self.call_counts[func_name] += 1

                return result

            return wrapper

        return decorator

    def start_profiling(self) -> None:
        """Start cProfile profiling."""
        self.profiler = cProfile.Profile()
        self.profiler.enable()

    def stop_profiling(self) -> str:
        """Stop cProfile profiling and return stats.

        Returns:
            Formatted profiling statistics
        """
        if self.profiler is None:
            return "Profiler not started"

        self.profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats("cumulative")
        ps.print_stats(20)  # Top 20 functions
        return s.getvalue()

    def get_stats(self) -> Dict[str, Any]:
        """Get timing statistics.

        Returns:
            Dictionary with timing stats per function
        """
        stats = {}
        for func_name, timings in self.timings.items():
            if timings:
                stats[func_name] = {
                    "calls": self.call_counts[func_name],
                    "total_time": sum(timings),
                    "avg_time": sum(timings) / len(timings),
                    "min_time": min(timings),
                    "max_time": max(timings),
                    "median_time": sorted(timings)[len(timings) // 2],
                }
        return stats

    def print_stats(self) -> None:
        """Print timing statistics."""
        stats = self.get_stats()
        print("\n=== Indicator Performance Stats ===")
        for func_name, func_stats in sorted(stats.items(), key=lambda x: x[1]["total_time"], reverse=True):
            print(f"\n{func_name}:")
            print(f"  Calls: {func_stats['calls']}")
            print(f"  Total time: {func_stats['total_time']:.4f}s")
            print(f"  Avg time: {func_stats['avg_time']:.4f}s")
            print(f"  Min time: {func_stats['min_time']:.4f}s")
            print(f"  Max time: {func_stats['max_time']:.4f}s")
            print(f"  Median time: {func_stats['median_time']:.4f}s")
        print("=" * 35)

    def reset(self) -> None:
        """Reset all profiling data."""
        self.timings.clear()
        self.call_counts.clear()
        self.profiler = None


# Global profiler instance
_global_profiler: Optional[IndicatorProfiler] = None


def get_profiler() -> Optional[IndicatorProfiler]:
    """Get global profiler instance.

    Returns:
        IndicatorProfiler instance or None
    """
    return _global_profiler


def set_profiler(profiler: Optional[IndicatorProfiler]) -> None:
    """Set global profiler instance.

    Args:
        profiler: IndicatorProfiler instance or None
    """
    global _global_profiler
    _global_profiler = profiler


def enable_profiling() -> IndicatorProfiler:
    """Enable profiling.

    Returns:
        IndicatorProfiler instance
    """
    profiler = IndicatorProfiler()
    set_profiler(profiler)
    return profiler


def disable_profiling() -> None:
    """Disable profiling."""
    set_profiler(None)

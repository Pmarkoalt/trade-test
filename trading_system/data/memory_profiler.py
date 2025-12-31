"""Memory profiling utilities for tracking memory usage in data loading and processing."""

import sys
import logging
from typing import Dict, Optional, Callable, Any
import psutil
import os

logger = logging.getLogger(__name__)


class MemoryProfiler:
    """Utility class for profiling memory usage."""
    
    def __init__(self, process: Optional[psutil.Process] = None):
        """Initialize memory profiler.
        
        Args:
            process: Optional psutil.Process instance (defaults to current process)
        """
        self.process = process or psutil.Process(os.getpid())
        self.snapshots: Dict[str, Dict[str, float]] = {}
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics.
        
        Returns:
            Dictionary with memory metrics:
            - rss: Resident Set Size (physical memory) in MB
            - vms: Virtual Memory Size in MB
            - percent: Memory usage as percentage of available
        """
        mem_info = self.process.memory_info()
        mem_percent = self.process.memory_percent()
        
        return {
            'rss_mb': mem_info.rss / (1024 * 1024),  # Convert to MB
            'vms_mb': mem_info.vms / (1024 * 1024),
            'percent': mem_percent
        }
    
    def snapshot(self, label: str) -> Dict[str, float]:
        """Take a memory snapshot with a label.
        
        Args:
            label: Label for this snapshot
        
        Returns:
            Current memory usage dictionary
        """
        usage = self.get_memory_usage()
        self.snapshots[label] = usage.copy()
        return usage
    
    def get_diff(self, label1: str, label2: str) -> Dict[str, float]:
        """Get memory difference between two snapshots.
        
        Args:
            label1: First snapshot label
            label2: Second snapshot label
        
        Returns:
            Dictionary with memory differences
        """
        if label1 not in self.snapshots:
            raise ValueError(f"Snapshot '{label1}' not found")
        if label2 not in self.snapshots:
            raise ValueError(f"Snapshot '{label2}' not found")
        
        s1 = self.snapshots[label1]
        s2 = self.snapshots[label2]
        
        return {
            'rss_mb_diff': s2['rss_mb'] - s1['rss_mb'],
            'vms_mb_diff': s2['vms_mb'] - s1['vms_mb'],
            'percent_diff': s2['percent'] - s1['percent']
        }
    
    def log_snapshot(self, label: str, logger_instance: Optional[logging.Logger] = None) -> None:
        """Take a snapshot and log it.
        
        Args:
            label: Label for this snapshot
            logger_instance: Optional logger (defaults to module logger)
        """
        usage = self.snapshot(label)
        log = logger_instance or logger
        log.info(
            f"Memory snapshot '{label}': "
            f"RSS={usage['rss_mb']:.1f}MB, "
            f"VMS={usage['vms_mb']:.1f}MB, "
            f"Percent={usage['percent']:.1f}%"
        )
    
    def log_diff(self, label1: str, label2: str, logger_instance: Optional[logging.Logger] = None) -> None:
        """Log memory difference between two snapshots.
        
        Args:
            label1: First snapshot label
            label2: Second snapshot label
            logger_instance: Optional logger (defaults to module logger)
        """
        diff = self.get_diff(label1, label2)
        log = logger_instance or logger
        log.info(
            f"Memory change '{label1}' -> '{label2}': "
            f"RSS={diff['rss_mb_diff']:+.1f}MB, "
            f"VMS={diff['vms_mb_diff']:+.1f}MB, "
            f"Percent={diff['percent_diff']:+.1f}%"
        )
    
    def profile_function(self, func: Callable, *args, **kwargs) -> tuple[Any, Dict[str, float]]:
        """Profile memory usage of a function call.
        
        Args:
            func: Function to profile
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
        
        Returns:
            Tuple of (function result, memory usage dict)
        """
        before = self.get_memory_usage()
        result = func(*args, **kwargs)
        after = self.get_memory_usage()
        
        usage = {
            'before_rss_mb': before['rss_mb'],
            'after_rss_mb': after['rss_mb'],
            'rss_mb_diff': after['rss_mb'] - before['rss_mb'],
            'before_vms_mb': before['vms_mb'],
            'after_vms_mb': after['vms_mb'],
            'vms_mb_diff': after['vms_mb'] - before['vms_mb'],
            'percent_diff': after['percent'] - before['percent']
        }
        
        return result, usage
    
    def get_summary(self) -> str:
        """Get summary of all snapshots.
        
        Returns:
            Formatted string with all snapshots
        """
        lines = ["Memory Profiling Summary:"]
        lines.append("-" * 60)
        
        for label, usage in sorted(self.snapshots.items()):
            lines.append(
                f"{label:30s} RSS={usage['rss_mb']:8.1f}MB  "
                f"VMS={usage['vms_mb']:8.1f}MB  "
                f"Percent={usage['percent']:5.1f}%"
            )
        
        return "\n".join(lines)


def optimize_dataframe_dtypes(df, price_cols=None, volume_cols=None):
    """Optimize DataFrame dtypes to reduce memory usage.
    
    Converts:
    - float64 -> float32 for price/volume columns (sufficient precision)
    - object -> category for string columns with limited unique values
    
    Args:
        df: DataFrame to optimize
        price_cols: List of price column names (defaults to ['open', 'high', 'low', 'close'])
        volume_cols: List of volume column names (defaults to ['volume', 'dollar_volume'])
    
    Returns:
        DataFrame with optimized dtypes
    """
    if price_cols is None:
        price_cols = ['open', 'high', 'low', 'close']
    if volume_cols is None:
        volume_cols = ['volume', 'dollar_volume']
    
    df = df.copy()
    
    # Convert price columns to float32 (sufficient precision for prices)
    for col in price_cols:
        if col in df.columns and df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    
    # Convert volume columns to float32
    for col in volume_cols:
        if col in df.columns and df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    
    # Convert string columns to category if they have limited unique values
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_count = df[col].nunique()
            total_count = len(df)
            # Convert to category if unique values < 50% of total (heuristic)
            if unique_count < total_count * 0.5 and unique_count < 1000:
                df[col] = df[col].astype('category')
    
    return df


def estimate_dataframe_memory(df) -> Dict[str, float]:
    """Estimate memory usage of a DataFrame.
    
    Args:
        df: DataFrame to estimate
    
    Returns:
        Dictionary with memory estimates in MB:
        - total_mb: Total memory usage
        - data_mb: Data memory usage
        - index_mb: Index memory usage
    """
    total_bytes = df.memory_usage(deep=True).sum()
    data_bytes = df.memory_usage(deep=True, index=False).sum()
    index_bytes = df.index.memory_usage(deep=True)
    
    return {
        'total_mb': total_bytes / (1024 * 1024),
        'data_mb': data_bytes / (1024 * 1024),
        'index_mb': index_bytes / (1024 * 1024)
    }


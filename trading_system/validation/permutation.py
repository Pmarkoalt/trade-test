"""Permutation test for entry timing significance."""

import logging
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

logger = logging.getLogger(__name__)


class PermutationTest:
    """Permutation test: Randomize entry dates while preserving exit logic."""

    # Default timeout values
    DEFAULT_TIMEOUT_SECONDS = 300  # 5 minutes
    TIMEOUT_CHECK_INTERVAL = 100  # Check every N iterations

    def __init__(
        self,
        actual_trades: List[Dict],
        period: Tuple[pd.Timestamp, pd.Timestamp],
        compute_sharpe_func: Callable[[List[Dict]], float],
        n_iterations: int = 1000,
        random_seed: Optional[int] = None,
        max_time_seconds: Optional[int] = None,
    ):
        """Initialize permutation test.

        Args:
            actual_trades: List of actual trades with entry_date, exit_date, symbol, r_multiple
            period: (start_date, end_date) for test period
            compute_sharpe_func: Function that takes trades and returns Sharpe ratio
            n_iterations: Number of randomized runs
            random_seed: Random seed for reproducibility
            max_time_seconds: Maximum time allowed for test (default: 300 = 5 min).
                              Set to None to disable timeout.
        """
        self.actual_trades = actual_trades
        self.period = period
        self.compute_sharpe_func = compute_sharpe_func
        self.n_iterations = n_iterations
        self.max_time_seconds = max_time_seconds if max_time_seconds is not None else self.DEFAULT_TIMEOUT_SECONDS

        if random_seed is not None:
            np.random.seed(random_seed)

    def run(self) -> Dict:
        """Run permutation test.

        Returns:
            Dictionary with results and percentile rank
        """
        # Track start time for timeout checking
        start_time = time.time()

        # Compute actual Sharpe
        actual_sharpe = self.compute_sharpe_func(self.actual_trades)

        # Ensure actual_sharpe is finite
        if not np.isfinite(actual_sharpe):
            actual_sharpe = 0.0

        # Extract actual entry/exit pairs
        actual_entries = [
            {
                "entry_date": t["entry_date"],
                "exit_date": t["exit_date"],
                "symbol": t["symbol"],
                "hold_days": (t["exit_date"] - t["entry_date"]).days,
            }
            for t in self.actual_trades
        ]

        # Store randomized Sharpe ratios
        random_sharpes = []
        iterations_completed = 0

        for i in range(self.n_iterations):
            # Randomize entry dates
            randomized_entries = self._randomize_entry_dates(actual_entries)

            # Create randomized trades (preserve R-multiples if available)
            randomized_trades = self._create_randomized_trades(randomized_entries, self.actual_trades)

            # Compute Sharpe on randomized trades
            try:
                random_sharpe = self.compute_sharpe_func(randomized_trades)
                # Only add finite values
                if np.isfinite(random_sharpe):
                    random_sharpes.append(random_sharpe)
            except Exception:
                # Skip invalid randomizations
                continue

            iterations_completed += 1

            # Check timeout periodically (every N iterations to avoid overhead)
            if self.max_time_seconds and (i + 1) % self.TIMEOUT_CHECK_INTERVAL == 0:
                elapsed_seconds = time.time() - start_time
                if elapsed_seconds > self.max_time_seconds:
                    logger.warning(
                        f"Permutation test timeout: exceeded {self.max_time_seconds}s "
                        f"(elapsed: {elapsed_seconds:.1f}s, completed {iterations_completed}/{self.n_iterations} iterations)"
                    )
                    # Return partial results instead of raising error for graceful degradation
                    break

        # Log completion
        total_time = time.time() - start_time
        logger.info(f"Permutation test completed: {iterations_completed}/{self.n_iterations} iterations in {total_time:.1f}s")

        if not random_sharpes:
            # Return default structure if no valid iterations
            return {
                "actual_sharpe": actual_sharpe,
                "random_sharpe_5th": 0.0,
                "random_sharpe_25th": 0.0,
                "random_sharpe_50th": 0.0,
                "random_sharpe_75th": 0.0,
                "random_sharpe_95th": 0.0,
                "percentile_rank": 0.0,
                "passed": False,
                "n_valid_iterations": 0,
            }

        # Compute percentile rank
        try:
            percentile_rank = percentileofscore(random_sharpes, actual_sharpe)
        except Exception:
            # If percentile calculation fails, default to 0.0
            percentile_rank = 0.0

        # Ensure percentile_rank is finite and in valid range [0.0, 100.0]
        if not np.isfinite(percentile_rank):
            percentile_rank = 0.0
        percentile_rank = max(0.0, min(100.0, percentile_rank))

        # Compute percentiles, ensuring they are finite
        random_sharpes_array = np.array(random_sharpes)

        # Helper to safely compute percentile
        def safe_percentile(arr, p):
            val = np.percentile(arr, p)
            return float(val) if np.isfinite(val) else 0.0

        results = {
            "actual_sharpe": actual_sharpe if np.isfinite(actual_sharpe) else 0.0,
            "random_sharpe_5th": safe_percentile(random_sharpes_array, 5),
            "random_sharpe_25th": safe_percentile(random_sharpes_array, 25),
            "random_sharpe_50th": safe_percentile(random_sharpes_array, 50),
            "random_sharpe_75th": safe_percentile(random_sharpes_array, 75),
            "random_sharpe_95th": safe_percentile(random_sharpes_array, 95),
            "percentile_rank": percentile_rank,
            "passed": percentile_rank >= 95,  # Must be >= 95th percentile
            "n_valid_iterations": len(random_sharpes),
        }

        # Final validation: ensure all values are finite
        for key, value in results.items():
            if isinstance(value, (int, float)) and not np.isfinite(value):
                results[key] = 0.0 if key != "passed" else False

        return results

    def _randomize_entry_dates(self, actual_entries: List[Dict]) -> List[Dict]:
        """Randomize entry dates while preserving holding periods.

        Args:
            actual_entries: List of entry/exit date pairs

        Returns:
            List of randomized entry/exit pairs
        """
        start_date, end_date = self.period
        randomized = []

        for entry_info in actual_entries:
            hold_days = entry_info["hold_days"]

            # Randomize entry date (within period, ensuring exit is also in period)
            max_entry_date = end_date - pd.Timedelta(days=hold_days)

            if max_entry_date < start_date:
                # Cannot fit: skip this trade
                continue

            # Random entry date
            days_available = (max_entry_date - start_date).days + 1
            if days_available <= 0:
                continue

            random_entry = start_date + pd.Timedelta(days=np.random.randint(0, days_available))

            # Preserve exit date relative to entry
            random_exit = random_entry + pd.Timedelta(days=hold_days)

            randomized.append(
                {"entry_date": random_entry, "exit_date": random_exit, "symbol": entry_info["symbol"], "hold_days": hold_days}
            )

        return randomized

    def _create_randomized_trades(self, randomized_entries: List[Dict], actual_trades: List[Dict]) -> List[Dict]:
        """Create randomized trades preserving R-multiples if available.

        Args:
            randomized_entries: Randomized entry/exit dates
            actual_trades: Original trades (for R-multiple preservation)

        Returns:
            List of randomized trades
        """
        randomized_trades = []

        # If we have R-multiples, preserve them
        # Otherwise, we'd need price data to recompute
        for i, entry_info in enumerate(randomized_entries):
            if i < len(actual_trades):
                # Preserve R-multiple from original trade
                original_trade = actual_trades[i]
                trade = {
                    "entry_date": entry_info["entry_date"],
                    "exit_date": entry_info["exit_date"],
                    "symbol": entry_info["symbol"],
                    "r_multiple": original_trade.get("r_multiple", 0.0),
                }
            else:
                # No corresponding original trade, use zero R-multiple
                trade = {
                    "entry_date": entry_info["entry_date"],
                    "exit_date": entry_info["exit_date"],
                    "symbol": entry_info["symbol"],
                    "r_multiple": 0.0,
                }

            randomized_trades.append(trade)

        return randomized_trades


def run_permutation_test(
    actual_trades: List[Dict],
    period: Tuple[pd.Timestamp, pd.Timestamp],
    compute_sharpe_func: Callable[[List[Dict]], float],
    n_iterations: int = 1000,
    random_seed: Optional[int] = None,
    max_time_seconds: Optional[int] = None,
) -> Dict:
    """Convenience function to run permutation test.

    Args:
        actual_trades: List of actual trades
        period: (start_date, end_date) for test period
        compute_sharpe_func: Function that takes trades and returns Sharpe
        n_iterations: Number of randomized runs
        random_seed: Random seed for reproducibility
        max_time_seconds: Maximum time allowed (default: 300 = 5 min)

    Returns:
        Permutation test results
    """
    test = PermutationTest(
        actual_trades, period, compute_sharpe_func, n_iterations, random_seed, max_time_seconds=max_time_seconds
    )
    return test.run()


def check_permutation_results(results: Dict) -> Tuple[bool, List[str]]:
    """Check permutation test results.

    Returns:
        (passed, warnings)
    """
    warnings = []

    if not results["passed"]:
        return (
            False,
            [
                "REJECT: Permutation test failed "
                f"(actual Sharpe {results['actual_sharpe']:.2f} < "
                f"95th percentile random {results['random_sharpe_95th']:.2f})"
            ],
        )

    # Warning if percentile rank is close to threshold
    if results["percentile_rank"] < 97:
        warnings.append(f"WARNING: Permutation percentile rank {results['percentile_rank']:.1f} " "is close to threshold (95)")

    return (True, warnings)

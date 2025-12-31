"""Walk-forward validation for time-series ML."""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
from loguru import logger

from trading_system.ml_refinement.config import TrainingConfig


@dataclass
class WalkForwardSplit:
    """A single train/validation split."""

    # Split indices
    train_start: int
    train_end: int
    val_start: int
    val_end: int

    # Date boundaries
    train_start_date: str = ""
    train_end_date: str = ""
    val_start_date: str = ""
    val_end_date: str = ""

    # Sizes
    train_size: int = 0
    val_size: int = 0


@dataclass
class WalkForwardResults:
    """Results from walk-forward validation."""

    # Per-fold results
    fold_results: List[Dict] = field(default_factory=list)

    # Aggregated metrics
    avg_metrics: Dict[str, float] = field(default_factory=dict)
    std_metrics: Dict[str, float] = field(default_factory=dict)

    # Predictions
    all_predictions: List[Tuple[int, float, float]] = field(default_factory=list)
    # List of (sample_idx, predicted, actual)

    # Summary
    n_folds: int = 0
    total_train_samples: int = 0
    total_val_samples: int = 0


class WalkForwardValidator:
    """
    Walk-forward cross-validation for time-series data.

    Walk-forward validation works by:
    1. Training on historical data window
    2. Validating on subsequent data
    3. Rolling forward and repeating

    Example:
        validator = WalkForwardValidator(
            train_window=252,  # ~1 year
            val_window=63,     # ~3 months
            step_size=21,      # ~1 month
        )

        for split in validator.generate_splits(n_samples=500, dates=date_list):
            # Train on split.train_start:split.train_end
            # Validate on split.val_start:split.val_end
            pass
    """

    def __init__(
        self,
        train_window: int = 252,
        val_window: int = 63,
        step_size: int = 21,
        min_train_samples: int = 100,
        min_val_samples: int = 20,
    ):
        """
        Initialize validator.

        Args:
            train_window: Number of samples in training window.
            val_window: Number of samples in validation window.
            step_size: How far to step forward each fold.
            min_train_samples: Minimum training samples required.
            min_val_samples: Minimum validation samples required.
        """
        self.train_window = train_window
        self.val_window = val_window
        self.step_size = step_size
        self.min_train_samples = min_train_samples
        self.min_val_samples = min_val_samples

    def generate_splits(
        self,
        n_samples: int,
        dates: Optional[List[str]] = None,
    ) -> Generator[WalkForwardSplit, None, None]:
        """
        Generate train/validation splits.

        Args:
            n_samples: Total number of samples.
            dates: Optional list of dates for each sample.

        Yields:
            WalkForwardSplit objects.
        """
        if n_samples < self.min_train_samples + self.min_val_samples:
            logger.warning(f"Insufficient samples: {n_samples}")
            return

        # Start after we have enough training data
        train_start = 0
        train_end = self.train_window

        fold_idx = 0
        while train_end + self.val_window <= n_samples:
            val_start = train_end
            val_end = min(val_start + self.val_window, n_samples)

            # Check minimum sizes
            actual_train_size = train_end - train_start
            actual_val_size = val_end - val_start

            if actual_train_size >= self.min_train_samples and actual_val_size >= self.min_val_samples:

                split = WalkForwardSplit(
                    train_start=train_start,
                    train_end=train_end,
                    val_start=val_start,
                    val_end=val_end,
                    train_size=actual_train_size,
                    val_size=actual_val_size,
                )

                # Add dates if available
                if dates:
                    split.train_start_date = dates[train_start]
                    split.train_end_date = dates[train_end - 1]
                    split.val_start_date = dates[val_start]
                    split.val_end_date = dates[val_end - 1]

                yield split
                fold_idx += 1

            # Step forward
            train_start += self.step_size
            train_end = train_start + self.train_window

    def count_folds(self, n_samples: int) -> int:
        """Count the number of folds for given sample size."""
        count = 0
        for _ in self.generate_splits(n_samples):
            count += 1
        return count

    @classmethod
    def from_config(cls, config: TrainingConfig) -> "WalkForwardValidator":
        """Create validator from config."""
        return cls(
            train_window=config.train_window_days,
            val_window=config.validation_window_days,
            step_size=config.step_size_days,
            min_train_samples=config.min_training_samples,
            min_val_samples=config.min_validation_samples,
        )


class ExpandingWindowValidator:
    """
    Expanding window validation.

    Unlike walk-forward which uses a fixed training window,
    expanding window uses all available historical data.
    """

    def __init__(
        self,
        initial_train_size: int = 252,
        val_window: int = 63,
        step_size: int = 21,
        min_val_samples: int = 20,
    ):
        """
        Initialize validator.

        Args:
            initial_train_size: Initial training window size.
            val_window: Validation window size.
            step_size: Step size between folds.
            min_val_samples: Minimum validation samples.
        """
        self.initial_train_size = initial_train_size
        self.val_window = val_window
        self.step_size = step_size
        self.min_val_samples = min_val_samples

    def generate_splits(
        self,
        n_samples: int,
        dates: Optional[List[str]] = None,
    ) -> Generator[WalkForwardSplit, None, None]:
        """Generate expanding window splits."""
        train_end = self.initial_train_size

        while train_end + self.val_window <= n_samples:
            val_start = train_end
            val_end = min(val_start + self.val_window, n_samples)

            if val_end - val_start >= self.min_val_samples:
                split = WalkForwardSplit(
                    train_start=0,  # Always start from beginning
                    train_end=train_end,
                    val_start=val_start,
                    val_end=val_end,
                    train_size=train_end,
                    val_size=val_end - val_start,
                )

                if dates:
                    split.train_start_date = dates[0]
                    split.train_end_date = dates[train_end - 1]
                    split.val_start_date = dates[val_start]
                    split.val_end_date = dates[val_end - 1]

                yield split

            train_end += self.step_size


class PurgedKFold:
    """
    K-Fold with purging and embargo for overlapping labels.

    Used when labels span multiple time periods to prevent leakage.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_window: int = 5,
        embargo_pct: float = 0.01,
    ):
        """
        Initialize purged k-fold.

        Args:
            n_splits: Number of folds.
            purge_window: Samples to purge around test set.
            embargo_pct: Percentage of training data to embargo after test.
        """
        self.n_splits = n_splits
        self.purge_window = purge_window
        self.embargo_pct = embargo_pct

    def split(
        self,
        n_samples: int,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices with purging.

        Yields:
            Tuple of (train_indices, test_indices).
        """
        fold_size = n_samples // self.n_splits
        embargo_size = int(n_samples * self.embargo_pct)

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples

            # Purge before test
            train_end_before = max(0, test_start - self.purge_window)

            # Embargo after test
            train_start_after = min(n_samples, test_end + embargo_size)

            # Build train indices
            train_before = np.arange(0, train_end_before)
            train_after = np.arange(train_start_after, n_samples)
            train_idx = np.concatenate([train_before, train_after])

            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx

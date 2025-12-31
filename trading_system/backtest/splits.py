"""Walk-forward split management for train/validation/holdout periods."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardSplit:
    """Walk-forward split configuration.
    
    Defines train, validation, and holdout date ranges for backtesting.
    """
    name: str  # Split name (e.g., "split_1")
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp
    
    def get_period_dates(self, period: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Get start and end dates for a specific period.
        
        Args:
            period: One of "train", "validation", "holdout"
        
        Returns:
            Tuple of (start_date, end_date)
        
        Raises:
            ValueError: If period is invalid
        """
        if period == "train":
            return (self.train_start, self.train_end)
        elif period == "validation":
            return (self.validation_start, self.validation_end)
        elif period == "holdout":
            return (self.holdout_start, self.holdout_end)
        else:
            raise ValueError(f"Invalid period: {period}, must be 'train', 'validation', or 'holdout'")
    
    def contains_date(self, date: pd.Timestamp, period: str) -> bool:
        """Check if a date falls within a specific period.
        
        Args:
            date: Date to check
            period: One of "train", "validation", "holdout"
        
        Returns:
            True if date is within the period (inclusive)
        """
        start, end = self.get_period_dates(period)
        return start <= date <= end
    
    def validate(self) -> bool:
        """Validate split configuration.
        
        Checks:
        - All dates are in order (train -> validation -> holdout)
        - No overlaps between periods
        - All dates are valid
        
        Returns:
            True if valid, False otherwise
        """
        # Check date ordering
        if not (self.train_start <= self.train_end < self.validation_start <= self.validation_end < self.holdout_start <= self.holdout_end):
            logger.error(f"Invalid date ordering in split {self.name}")
            return False
        
        # Check for gaps (optional - could allow gaps)
        # For now, we allow gaps between periods
        
        return True
    
    def __post_init__(self):
        """Validate split after initialization."""
        if not self.validate():
            raise ValueError(f"Invalid split configuration: {self.name}")


def load_splits_from_config(config_path: str) -> list[WalkForwardSplit]:
    """Load walk-forward splits from YAML config file.
    
    Expected config format:
    ```yaml
    splits:
      - name: split_1
        train:
          start: "2020-01-01"
          end: "2021-03-31"
        validation:
          start: "2021-04-01"
          end: "2021-06-30"
        holdout:
          start: "2021-07-01"
          end: "2021-12-31"
      - name: split_2
        ...
    ```
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        List of WalkForwardSplit objects
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'splits' not in config:
        raise ValueError("Config file must contain 'splits' key")
    
    splits = []
    for split_config in config['splits']:
        if 'name' not in split_config:
            raise ValueError("Each split must have a 'name' field")
        
        # Parse dates
        train_start = pd.Timestamp(split_config['train']['start'])
        train_end = pd.Timestamp(split_config['train']['end'])
        validation_start = pd.Timestamp(split_config['validation']['start'])
        validation_end = pd.Timestamp(split_config['validation']['end'])
        holdout_start = pd.Timestamp(split_config['holdout']['start'])
        holdout_end = pd.Timestamp(split_config['holdout']['end'])
        
        split = WalkForwardSplit(
            name=split_config['name'],
            train_start=train_start,
            train_end=train_end,
            validation_start=validation_start,
            validation_end=validation_end,
            holdout_start=holdout_start,
            holdout_end=holdout_end
        )
        
        splits.append(split)
    
    logger.info(f"Loaded {len(splits)} walk-forward splits from {config_path}")
    return splits


def create_default_split(
    start_date: pd.Timestamp,
    months: int = 24
) -> WalkForwardSplit:
    """Create a default 24-month split (15/3/6 months).
    
    Preferred split:
    - Train: months 1-15
    - Validation: months 16-18
    - Holdout: months 19-24
    
    Minimum split (18 months):
    - Train: months 1-12
    - Validation: months 13-15
    - Holdout: months 16-18
    
    Args:
        start_date: Start date for the split
        months: Total number of months (24 preferred, 18 minimum)
    
    Returns:
        WalkForwardSplit object
    """
    if months == 24:
        # Preferred: 15/3/6 months
        train_end = start_date + pd.DateOffset(months=15) - pd.Timedelta(days=1)
        validation_start = train_end + pd.Timedelta(days=1)
        validation_end = validation_start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
        holdout_start = validation_end + pd.Timedelta(days=1)
        holdout_end = holdout_start + pd.DateOffset(months=6) - pd.Timedelta(days=1)
    elif months == 18:
        # Minimum: 12/3/3 months
        train_end = start_date + pd.DateOffset(months=12) - pd.Timedelta(days=1)
        validation_start = train_end + pd.Timedelta(days=1)
        validation_end = validation_start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
        holdout_start = validation_end + pd.Timedelta(days=1)
        holdout_end = holdout_start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
    else:
        raise ValueError(f"Unsupported months: {months}, must be 18 or 24")
    
    return WalkForwardSplit(
        name="default_split",
        train_start=start_date,
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end,
        holdout_start=holdout_start,
        holdout_end=holdout_end
    )


"""Configuration for scheduler module."""

from pydantic import BaseModel


class SchedulerConfig(BaseModel):
    """Configuration for the scheduler module.

    Attributes:
        enabled: Whether the scheduler is enabled
        timezone: Default timezone for scheduled jobs
    """

    enabled: bool = True
    timezone: str = "UTC"


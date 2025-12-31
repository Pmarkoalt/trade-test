"""Cron scheduler runner for automated daily signal generation."""

from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from ..logging.logger import get_logger
from .config import SchedulerConfig
from .jobs.daily_signals_job import daily_signals_job

logger = get_logger(__name__)


class CronRunner:
    """Cron scheduler runner for automated daily signal generation."""

    def __init__(self, config: Optional[SchedulerConfig] = None):
        """Initialize cron runner.

        Args:
            config: Optional scheduler configuration. If None, uses default config.
        """
        self.config = config or SchedulerConfig()
        self.scheduler = AsyncIOScheduler()

    def register_jobs(self) -> None:
        """Register all scheduled jobs."""
        if not self.config.enabled:
            logger.info("Scheduler is disabled, skipping job registration")
            return

        # Daily equity signals - 4:30 PM ET
        self.scheduler.add_job(
            daily_signals_job,
            CronTrigger(
                hour=16,
                minute=30,
                timezone="America/New_York"
            ),
            id="daily_equity_signals",
            kwargs={"asset_class": "equity"},
            replace_existing=True,
        )

        # Daily crypto signals - midnight UTC
        self.scheduler.add_job(
            daily_signals_job,
            CronTrigger(
                hour=0,
                minute=0,
                timezone="UTC"
            ),
            id="daily_crypto_signals",
            kwargs={"asset_class": "crypto"},
            replace_existing=True,
        )

        logger.info("Registered scheduled jobs: daily_equity_signals (4:30 PM ET), daily_crypto_signals (midnight UTC)")

    def start(self) -> None:
        """Start the scheduler."""
        if not self.config.enabled:
            logger.info("Scheduler is disabled, not starting")
            return

        self.scheduler.start()
        logger.info("Scheduler started")

    def stop(self) -> None:
        """Stop the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=True)
            logger.info("Scheduler stopped")
        else:
            logger.info("Scheduler was not running")

    def is_running(self) -> bool:
        """Check if scheduler is running.

        Returns:
            True if scheduler is running, False otherwise
        """
        return self.scheduler.running


"""Storage module for backtest results."""

from .database import ResultsDatabase, get_default_db_path

__all__ = ['ResultsDatabase', 'get_default_db_path']


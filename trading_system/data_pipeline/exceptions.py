"""Custom exceptions for the data pipeline module."""


class DataPipelineError(Exception):
    """Base exception for data pipeline errors."""


class DataFetchError(DataPipelineError):
    """Raised when data fetching fails."""


class APIRateLimitError(DataPipelineError):
    """Raised when API rate limit is exceeded."""


class DataValidationError(DataPipelineError):
    """Raised when fetched data fails validation."""

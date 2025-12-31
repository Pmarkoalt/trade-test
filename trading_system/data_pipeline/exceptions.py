"""Custom exceptions for the data pipeline module."""


class DataPipelineError(Exception):
    """Base exception for data pipeline errors."""

    pass


class DataFetchError(DataPipelineError):
    """Raised when data fetching fails."""

    pass


class APIRateLimitError(DataPipelineError):
    """Raised when API rate limit is exceeded."""

    pass


class DataValidationError(DataPipelineError):
    """Raised when fetched data fails validation."""

    pass


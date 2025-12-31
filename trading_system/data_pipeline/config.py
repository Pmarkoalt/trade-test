"""Configuration models for the data pipeline module."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class DataPipelineConfig(BaseModel):
    """Configuration for the data pipeline module.

    Attributes:
        polygon_api_key: API key for Polygon.io (optional)
        alpha_vantage_api_key: API key for Alpha Vantage (optional)
        cache_path: Path to the cache directory for storing fetched data
        cache_ttl_hours: Time-to-live for cached data in hours
    """

    polygon_api_key: Optional[str] = None
    alpha_vantage_api_key: Optional[str] = None
    cache_path: Path = Path("data/cache")
    cache_ttl_hours: int = 24

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True


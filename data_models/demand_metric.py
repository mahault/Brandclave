"""DemandMetric schema — geo-resolvable demand time series points."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DemandMetricCreate(BaseModel):
    """One metric observation for one city on one day."""

    source: str = Field(..., description="Registry source name (e.g. wikimedia_pageviews)")
    city: str = Field(..., description="City the metric describes")
    country: str | None = Field(None, description="Country of the city")
    metric: str = Field(..., description="Metric name (e.g. wikipedia_pageviews)")
    date: datetime = Field(..., description="Day the observation covers")
    value: float = Field(..., description="Observed value")
    metadata: dict[str, Any] = Field(default_factory=dict)

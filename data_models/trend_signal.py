"""TrendSignal schema for Social Pulse trend cards."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AudienceSegment(str, Enum):
    """Target audience segments."""

    LUXURY = "luxury"
    BOUTIQUE = "boutique"
    BUDGET = "budget"
    BUSINESS = "business"
    FAMILY = "family"
    WELLNESS = "wellness"
    ADVENTURE = "adventure"
    ECO = "eco"
    GENERAL = "general"


class TrendSignal(BaseModel):
    """Schema for processed trend signals (Social Pulse cards)."""

    id: str | None = None
    name: str = Field(..., description="Trend name/title")
    description: str = Field(..., description="What this trend is about")
    why_it_matters: str = Field(..., description="Strategic relevance explanation")

    # Strength metrics
    strength_score: float = Field(..., ge=0, le=1, description="Overall trend strength (0-1)")
    volume: int = Field(default=0, description="Number of mentions/posts")
    engagement_score: float = Field(default=0, ge=0, description="Engagement metric")
    sentiment_delta: float = Field(default=0, description="Sentiment change over time")

    # White-space analysis
    white_space_score: float = Field(default=0, ge=0, le=1, description="Demand vs supply gap (0-1)")
    demand_signals: int = Field(default=0, description="Count of demand indicators")
    supply_signals: int = Field(default=0, description="Count of supply indicators")

    # Categorization
    region: str | None = Field(None, description="Geographic region")
    audience_segment: AudienceSegment = Field(default=AudienceSegment.GENERAL)
    topics: list[str] = Field(default_factory=list, description="Associated topic tags")

    # Temporal
    first_seen: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    time_window_days: int = Field(default=7, description="Analysis time window")

    # Source tracking
    source_content_ids: list[str] = Field(default_factory=list, description="RawContent IDs contributing to this trend")
    sample_quotes: list[str] = Field(default_factory=list, description="Representative quotes")

    # Clustering reference
    cluster_id: str | None = Field(None, description="Reference to embedding cluster")

    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class TrendSignalCreate(BaseModel):
    """Schema for creating new TrendSignal items."""

    name: str
    description: str
    why_it_matters: str
    strength_score: float
    region: str | None = None
    audience_segment: AudienceSegment = AudienceSegment.GENERAL
    topics: list[str] = Field(default_factory=list)
    source_content_ids: list[str] = Field(default_factory=list)

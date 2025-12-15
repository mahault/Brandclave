"""PropertyFeatures schema for Demand Scan property analysis."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PropertyType(str, Enum):
    """Types of hospitality properties."""

    HOTEL = "hotel"
    RESORT = "resort"
    BOUTIQUE = "boutique"
    HOSTEL = "hostel"
    VACATION_RENTAL = "vacation_rental"
    BED_AND_BREAKFAST = "bed_and_breakfast"
    MOTEL = "motel"
    OTHER = "other"


class PriceSegment(str, Enum):
    """Price positioning segments."""

    BUDGET = "budget"
    MIDSCALE = "midscale"
    UPSCALE = "upscale"
    LUXURY = "luxury"
    ULTRA_LUXURY = "ultra_luxury"
    UNKNOWN = "unknown"


class PropertyFeatures(BaseModel):
    """Schema for extracted property features from Demand Scan."""

    id: str | None = None
    url: str = Field(..., description="Property website URL")
    name: str | None = Field(None, description="Property name")
    property_type: PropertyType = Field(default=PropertyType.HOTEL)

    # Positioning
    brand_positioning: str | None = Field(None, description="Extracted brand positioning statement")
    tagline: str | None = Field(None, description="Property tagline if found")
    tone: str | None = Field(None, description="Brand voice/tone analysis")
    themes: list[str] = Field(default_factory=list, description="Key themes (e.g., 'wellness', 'adventure')")

    # Features
    amenities: list[str] = Field(default_factory=list, description="Listed amenities")
    room_types: list[str] = Field(default_factory=list, description="Room categories")
    dining_options: list[str] = Field(default_factory=list, description="F&B offerings")
    experiences: list[str] = Field(default_factory=list, description="Activities and experiences")

    # Location
    location: str | None = Field(None, description="Property location")
    region: str | None = Field(None, description="Geographic region")

    # Pricing
    price_segment: PriceSegment = Field(default=PriceSegment.UNKNOWN)
    price_indicators: list[str] = Field(default_factory=list, description="Price-related text found")

    # Demand Fit Analysis
    demand_fit_score: float | None = Field(None, ge=0, le=1, description="Fit with regional demand (0-1)")
    experience_gaps: list[str] = Field(default_factory=list, description="Missing high-demand experiences")
    opportunity_lanes: list[str] = Field(default_factory=list, description="Recommended positioning opportunities")
    competitive_advantages: list[str] = Field(default_factory=list, description="Current strengths")
    recommendations: list[str] = Field(default_factory=list, description="Strategic recommendations")

    # Matching trends
    matching_trend_ids: list[str] = Field(default_factory=list, description="TrendSignal IDs that match")

    # Extraction metadata
    scraped_at: datetime = Field(default_factory=datetime.utcnow)
    source_content_id: str | None = Field(None, description="RawContent ID")

    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class PropertyFeaturesCreate(BaseModel):
    """Schema for creating new PropertyFeatures from URL scan."""

    url: str
    name: str | None = None
    property_type: PropertyType = PropertyType.HOTEL
    brand_positioning: str | None = None
    themes: list[str] = Field(default_factory=list)
    amenities: list[str] = Field(default_factory=list)
    location: str | None = None

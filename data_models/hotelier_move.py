"""HotelierMove schema for Hotelier Bets intelligence cards."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MoveType(str, Enum):
    """Types of strategic moves by hoteliers."""

    LAUNCH = "launch"  # New property or brand launch
    ACQUISITION = "acquisition"  # M&A activity
    REPOSITIONING = "repositioning"  # Brand/market repositioning
    REFLAG = "reflag"  # Changing brand affiliation
    CONCEPT = "concept"  # New concept or experience
    EXPANSION = "expansion"  # Geographic or portfolio expansion
    RENOVATION = "renovation"  # Major property renovation
    PARTNERSHIP = "partnership"  # Strategic partnership
    TECHNOLOGY = "technology"  # Tech implementation
    SUSTAINABILITY = "sustainability"  # Green/sustainability initiative
    OTHER = "other"


class HotelierMove(BaseModel):
    """Schema for extracted hotelier strategic moves (Hotelier Bets cards)."""

    id: str | None = None
    title: str = Field(..., description="Action title/headline")
    summary: str = Field(..., description="Brief summary of the move")
    why_it_matters: str = Field(..., description="Strategic implications")

    # Company info
    company: str = Field(..., description="Company/brand making the move")
    company_type: str | None = Field(None, description="e.g., 'chain', 'independent', 'management company'")

    # Move details
    move_type: MoveType = Field(..., description="Category of strategic move")
    market: str | None = Field(None, description="Target market/location")
    investment_amount: str | None = Field(None, description="Investment size if mentioned")

    # Strategic context
    strategic_implications: list[str] = Field(default_factory=list, description="Key strategic takeaways")
    competitive_impact: str | None = Field(None, description="Impact on competitive landscape")
    related_trends: list[str] = Field(default_factory=list, description="Related trend signals")

    # Source info
    source_url: str = Field(..., description="Original article URL")
    source_name: str = Field(..., description="Source publication")
    published_at: datetime | None = Field(None, description="Article publication date")
    extracted_at: datetime = Field(default_factory=datetime.utcnow)

    # References
    source_content_id: str | None = Field(None, description="RawContent ID")

    # Confidence
    confidence_score: float = Field(default=0.5, ge=0, le=1, description="Extraction confidence")

    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class HotelierMoveCreate(BaseModel):
    """Schema for creating new HotelierMove items."""

    title: str
    summary: str
    why_it_matters: str
    company: str
    move_type: MoveType
    source_url: str
    source_name: str
    market: str | None = None
    published_at: datetime | None = None
    source_content_id: str | None = None

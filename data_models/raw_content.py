"""RawContent schema for scraped content items."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Type of content source."""

    NEWS = "news"
    SOCIAL = "social"
    REVIEW = "review"
    PROPERTY = "property"


class RawContent(BaseModel):
    """Schema for raw scraped content before processing."""

    id: str | None = None
    source: str = Field(..., description="Source identifier (e.g., 'hospitalitynet', 'reddit')")
    source_type: SourceType = Field(..., description="Category of source")
    url: str = Field(..., description="Original URL of the content")
    title: str | None = Field(None, description="Title or headline")
    content: str = Field(..., description="Main text content")
    author: str | None = Field(None, description="Author or username")
    published_at: datetime | None = Field(None, description="Original publication date")
    scraped_at: datetime = Field(default_factory=datetime.utcnow, description="When content was scraped")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional source-specific data")
    embedding_id: str | None = Field(None, description="Reference to embedding in vector store")

    # Processing status
    is_processed: bool = Field(default=False, description="Whether NLP pipeline has processed this")
    language: str | None = Field(None, description="Detected language code")
    sentiment_score: float | None = Field(None, description="Sentiment score (-1 to 1)")

    class Config:
        use_enum_values = True


class RawContentCreate(BaseModel):
    """Schema for creating new RawContent items."""

    source: str
    source_type: SourceType
    url: str
    title: str | None = None
    content: str
    author: str | None = None
    published_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

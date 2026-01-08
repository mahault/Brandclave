"""SQLAlchemy ORM models for BrandClave Aggregator."""

import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Enum, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid.uuid4())


class RawContentModel(Base):
    """SQLAlchemy model for raw scraped content."""

    __tablename__ = "raw_content"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    source: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    source_type: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    url: Mapped[str] = mapped_column(String(2048), nullable=False, unique=True)
    title: Mapped[str | None] = mapped_column(String(500))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    author: Mapped[str | None] = mapped_column(String(200))
    published_at: Mapped[datetime | None] = mapped_column(DateTime)
    scraped_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, default=dict)
    embedding_id: Mapped[str | None] = mapped_column(String(36))

    # Processing status
    is_processed: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    language: Mapped[str | None] = mapped_column(String(10))
    sentiment_score: Mapped[float | None] = mapped_column(Float)


class TrendSignalModel(Base):
    """SQLAlchemy model for trend signals."""

    __tablename__ = "trend_signals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    why_it_matters: Mapped[str] = mapped_column(Text, nullable=False)

    # Metrics
    strength_score: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int] = mapped_column(Integer, default=0)
    engagement_score: Mapped[float] = mapped_column(Float, default=0)
    sentiment_delta: Mapped[float] = mapped_column(Float, default=0)

    # White-space
    white_space_score: Mapped[float] = mapped_column(Float, default=0)
    demand_signals: Mapped[int] = mapped_column(Integer, default=0)
    supply_signals: Mapped[int] = mapped_column(Integer, default=0)

    # Categorization
    region: Mapped[str | None] = mapped_column(String(100), index=True)
    audience_segment: Mapped[str] = mapped_column(String(50), default="general")
    topics: Mapped[list | None] = mapped_column(JSON, default=list)

    # Temporal
    first_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    time_window_days: Mapped[int] = mapped_column(Integer, default=7)

    # References
    source_content_ids: Mapped[list | None] = mapped_column(JSON, default=list)
    sample_quotes: Mapped[list | None] = mapped_column(JSON, default=list)
    cluster_id: Mapped[str | None] = mapped_column(String(36))

    metadata_json: Mapped[dict | None] = mapped_column(JSON, default=dict)


class HotelierMoveModel(Base):
    """SQLAlchemy model for hotelier strategic moves."""

    __tablename__ = "hotelier_moves"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    why_it_matters: Mapped[str] = mapped_column(Text, nullable=False)

    # Company
    company: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    company_type: Mapped[str | None] = mapped_column(String(50))

    # Move details
    move_type: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    market: Mapped[str | None] = mapped_column(String(100), index=True)
    investment_amount: Mapped[str | None] = mapped_column(String(50))

    # Strategic
    strategic_implications: Mapped[list | None] = mapped_column(JSON, default=list)
    competitive_impact: Mapped[str | None] = mapped_column(Text)
    related_trends: Mapped[list | None] = mapped_column(JSON, default=list)

    # Source
    source_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    source_name: Mapped[str] = mapped_column(String(100), nullable=False)
    published_at: Mapped[datetime | None] = mapped_column(DateTime)
    extracted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    source_content_id: Mapped[str | None] = mapped_column(String(36))
    confidence_score: Mapped[float] = mapped_column(Float, default=0.5)

    metadata_json: Mapped[dict | None] = mapped_column(JSON, default=dict)


class PropertyFeaturesModel(Base):
    """SQLAlchemy model for property features from Demand Scan."""

    __tablename__ = "property_features"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    url: Mapped[str] = mapped_column(String(2048), nullable=False, unique=True, index=True)
    name: Mapped[str | None] = mapped_column(String(300))
    property_type: Mapped[str] = mapped_column(String(30), default="hotel", index=True)

    # Positioning
    brand_positioning: Mapped[str | None] = mapped_column(Text)
    tagline: Mapped[str | None] = mapped_column(String(500))
    tone: Mapped[str | None] = mapped_column(String(200))
    themes: Mapped[list | None] = mapped_column(JSON, default=list)

    # Features
    amenities: Mapped[list | None] = mapped_column(JSON, default=list)
    room_types: Mapped[list | None] = mapped_column(JSON, default=list)
    dining_options: Mapped[list | None] = mapped_column(JSON, default=list)
    experiences: Mapped[list | None] = mapped_column(JSON, default=list)

    # Location
    location: Mapped[str | None] = mapped_column(String(300))
    region: Mapped[str | None] = mapped_column(String(100), index=True)

    # Pricing
    price_segment: Mapped[str] = mapped_column(String(30), default="unknown")
    price_indicators: Mapped[list | None] = mapped_column(JSON, default=list)

    # Demand Fit Analysis
    demand_fit_score: Mapped[float | None] = mapped_column(Float)
    experience_gaps: Mapped[list | None] = mapped_column(JSON, default=list)
    opportunity_lanes: Mapped[list | None] = mapped_column(JSON, default=list)
    competitive_advantages: Mapped[list | None] = mapped_column(JSON, default=list)
    positioning_misalignment_flags: Mapped[list | None] = mapped_column(JSON, default=list)
    recommendations: Mapped[list | None] = mapped_column(JSON, default=list)

    # Matching trends
    matching_trend_ids: Mapped[list | None] = mapped_column(JSON, default=list)

    # Metadata
    scraped_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    source_content_id: Mapped[str | None] = mapped_column(String(36))
    metadata_json: Mapped[dict | None] = mapped_column(JSON, default=dict)


class ProcessingJobModel(Base):
    """Track scraping and processing jobs."""

    __tablename__ = "processing_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    job_type: Mapped[str] = mapped_column(String(50), nullable=False)  # 'scrape', 'process', 'cluster'
    source: Mapped[str | None] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending, running, completed, failed
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
    items_processed: Mapped[int] = mapped_column(Integer, default=0)
    items_failed: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, default=dict)


class BrandBlueprintModel(Base):
    """SQLAlchemy model for generated brand blueprints."""

    __tablename__ = "brand_blueprints"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)

    # Input data
    location: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    segment: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    adr: Mapped[float] = mapped_column(Float, nullable=False)
    rooms: Mapped[int] = mapped_column(Integer, nullable=False)
    developer_goal: Mapped[str] = mapped_column(Text, nullable=False)
    source_trend_id: Mapped[str | None] = mapped_column(String(36))
    profile_data_json: Mapped[dict | None] = mapped_column(JSON)

    # Stage 1: Foundation
    brand_name_primary: Mapped[str] = mapped_column(String(200), nullable=False)
    brand_name_alt_1: Mapped[str | None] = mapped_column(String(200))
    brand_name_alt_2: Mapped[str | None] = mapped_column(String(200))
    one_liner: Mapped[str] = mapped_column(String(500), nullable=False)
    thesis: Mapped[str] = mapped_column(Text, nullable=False)

    # Stage 2: Strategic
    pillars: Mapped[list] = mapped_column(JSON, default=list)
    positioning_statement: Mapped[str] = mapped_column(Text, nullable=False)
    unmet_desires_solved: Mapped[list] = mapped_column(JSON, default=list)

    # Stage 3: Experience
    guest_personas: Mapped[list] = mapped_column(JSON, default=list)
    signature_experiences: Mapped[list] = mapped_column(JSON, default=list)
    guest_journey: Mapped[dict | None] = mapped_column(JSON)

    # Stage 4: Atmosphere & Revenue
    design_direction: Mapped[str] = mapped_column(Text, nullable=False)
    fnb_concepts: Mapped[list] = mapped_column(JSON, default=list)
    revenue_logic: Mapped[str] = mapped_column(Text, nullable=False)

    # Stage 5: Summary
    investor_summary: Mapped[str] = mapped_column(Text, nullable=False)

    # Metadata
    status: Mapped[str] = mapped_column(String(20), default="completed")
    confidence: Mapped[float] = mapped_column(Float, default=0.8)
    warnings: Mapped[list | None] = mapped_column(JSON, default=list)

    # Token tracking
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # User association (for future auth)
    user_id: Mapped[str | None] = mapped_column(String(36), index=True)

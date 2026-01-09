"""Brand Blueprint Pydantic schemas.

Defines the data structures for the multi-step brand generation pipeline.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# Re-use existing schemas from chat module
from services.chat.schemas import (
    SignatureExperience,
    GuestJourney,
    GuestPersona,
    FnBConcept,
)


class StageStatus(str, Enum):
    """Status of a pipeline stage."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class BlueprintInputs(BaseModel):
    """Input form data for blueprint generation."""
    location: str = Field(..., min_length=2, max_length=100, description="Target city/region")
    segment: str = Field(..., description="Hotel segment: lifestyle|luxury|boutique|wellness|eco|business|family|adventure")
    adr: float = Field(..., ge=1, le=5000, description="Target ADR in USD")
    rooms: int = Field(..., ge=10, le=1000, description="Room count")
    developer_goal: str = Field(..., min_length=10, max_length=2000, description="Developer's vision/goal")
    source_trend_id: str | None = Field(None, description="Optional trend ID to build from")
    profile_data: dict[str, Any] | None = Field(None, description="Saved research profile")


class AlternateBrandNames(BaseModel):
    """Brand name with alternatives."""
    primary: str = Field(..., description="Recommended brand name")
    alternate_1: str = Field(..., description="First alternative")
    alternate_2: str = Field(..., description="Second alternative")


class UnmetDesireSolved(BaseModel):
    """An unmet desire solved by the brand."""
    desire: str = Field(..., description="The unmet guest desire")
    how_solved: str = Field(..., description="How the brand solves it")
    linked_trend_id: str | None = Field(None, description="Related trend ID")
    demand_strength: float = Field(0.5, ge=0, le=1, description="Strength of demand signal")


class StageProgress(BaseModel):
    """Progress update for a single pipeline stage."""
    stage: str = Field(..., description="Stage name")
    status: StageStatus = Field(..., description="Current status")
    progress_pct: int = Field(0, ge=0, le=100, description="Progress percentage")
    output: dict[str, Any] | None = Field(None, description="Stage output if completed")
    error: str | None = Field(None, description="Error message if failed")
    tokens_used: int = Field(0, description="Tokens used in this stage")


class TokenUsage(BaseModel):
    """Token usage tracking."""
    input_tokens: int = Field(0, description="Total input tokens")
    output_tokens: int = Field(0, description="Total output tokens")
    total_tokens: int = Field(0, description="Total tokens used")
    estimated_cost_usd: float = Field(0.0, description="Estimated cost in USD")


# Stage-specific output schemas

class FoundationOutput(BaseModel):
    """Output from Stage 1: Foundation."""
    brand_names: AlternateBrandNames
    one_liner: str
    thesis: str


class StrategicOutput(BaseModel):
    """Output from Stage 2: Strategic."""
    pillars: list[str] = Field(..., min_length=3, max_length=5)
    positioning_statement: str
    unmet_desires_solved: list[UnmetDesireSolved]


class ExperienceOutput(BaseModel):
    """Output from Stage 3: Experience."""
    guest_personas: list[GuestPersona] = Field(..., min_length=2, max_length=3)
    signature_experiences: list[SignatureExperience] = Field(..., min_length=3, max_length=5)
    guest_journey: GuestJourney


class AtmosphereOutput(BaseModel):
    """Output from Stage 4: Atmosphere & Revenue."""
    design_direction: str
    fnb_concepts: list[FnBConcept]
    revenue_logic: str


class SummaryOutput(BaseModel):
    """Output from Stage 5: Summary."""
    investor_summary: str


class BrandBlueprintFull(BaseModel):
    """Complete brand blueprint output.

    The core product - an AI-generated brand blueprint with all 13 components.
    """
    type: str = "brand_blueprint_full_v1"
    id: str | None = Field(None, description="Blueprint ID from database")

    # Inputs captured
    inputs: BlueprintInputs

    # Stage 1: Foundation
    brand_names: AlternateBrandNames
    one_liner: str
    thesis: str

    # Stage 2: Strategic
    pillars: list[str] = Field(default_factory=list)
    positioning_statement: str = ""
    unmet_desires_solved: list[UnmetDesireSolved] = Field(default_factory=list)

    # Stage 3: Experience
    guest_personas: list[GuestPersona] = Field(default_factory=list)
    signature_experiences: list[SignatureExperience] = Field(default_factory=list)
    guest_journey: GuestJourney | None = None

    # Stage 4: Atmosphere & Revenue
    design_direction: str = ""
    fnb_concepts: list[FnBConcept] = Field(default_factory=list)
    revenue_logic: str = ""

    # Stage 5: Summary
    investor_summary: str = ""

    # Metadata
    status: str = Field("completed", description="completed|partial|failed")
    confidence: float = Field(0.8, ge=0, le=1, description="Overall confidence")
    warnings: list[str] = Field(default_factory=list, description="Generation warnings")

    # Token tracking
    token_usage: TokenUsage = Field(default_factory=TokenUsage)

    # Timestamps
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class BlueprintGenerateRequest(BaseModel):
    """Request to generate a new blueprint."""
    inputs: BlueprintInputs


class BlueprintGenerateResponse(BaseModel):
    """Response with generated blueprint."""
    blueprint_id: str
    status: str = Field(..., description="completed|partial|failed")
    blueprint: BrandBlueprintFull | None = None
    stages: list[StageProgress] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)


class BlueprintListResponse(BaseModel):
    """Response for listing blueprints."""
    blueprints: list[BrandBlueprintFull]
    total: int
    offset: int = 0
    limit: int = 20

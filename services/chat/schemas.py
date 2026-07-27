"""Chat module data schemas - structured artifact outputs."""

from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class ChatMode(str, Enum):
    """Chat interaction modes."""
    INSIGHT = "insight"
    BRAND_BUILD = "brand_build"
    DEMAND_SCAN = "demand_scan"


class Evidence(BaseModel):
    """Evidence reference for a claim."""
    chunk_id: str
    text_snippet: str
    source_type: str | None = None
    confidence: float = 0.5


class KeySignal(BaseModel):
    """A key signal/finding from analysis."""
    signal: str
    why_it_matters: str
    confidence: float = 0.5
    evidence: list[Evidence] = Field(default_factory=list)


class WhiteSpaceOpportunity(BaseModel):
    """A white space opportunity identified."""
    opportunity: str
    who_it_serves: str
    why_now: str
    risk: str | None = None
    confidence: float = 0.5


class NextStepAction(str, Enum):
    """Recommended next actions."""
    SEND_TO_BUILD_A_BRAND = "send_to_build_a_brand"
    ASK_MORE = "ask_more"
    SAVE = "save"


class RecommendedNextStep(BaseModel):
    """Recommended next action."""
    action: NextStepAction
    reason: str
    prefill: dict[str, Any] | None = None


# ============================================================================
# INSIGHT MODE SCHEMA
# ============================================================================

class InsightBrief(BaseModel):
    """Structured output for insight mode queries.

    Generated when user asks about trends, markets, or opportunities.
    """
    type: str = "insight_brief_v1"
    topic: str
    location: str | None = None
    key_signals: list[KeySignal] = Field(default_factory=list)
    white_space_opportunities: list[WhiteSpaceOpportunity] = Field(default_factory=list)
    recommended_next_step: RecommendedNextStep | None = None
    confidence: float = 0.5
    sources_used: int = 0


# ============================================================================
# DEMAND SCAN SCHEMA
# ============================================================================

class ExperienceGap(BaseModel):
    """An experience gap identified in property analysis."""
    theme: str
    what_guests_want: str
    what_is_missing: str


class OpportunityLane(BaseModel):
    """A strategic trajectory for the property."""
    trajectory: str
    what_to_build: str
    why_it_wins: str


class DemandScanLite(BaseModel):
    """Structured output for demand scan analysis.

    Generated when analyzing a property URL.
    """
    type: str = "demand_scan_lite_v1"
    property_url: str
    property_name: str | None = None
    location: str | None = None
    segment: str | None = None
    target_adr: float | None = None
    demand_fit_score: int = Field(ge=0, le=100, default=50)
    positioning_misalignment_flags: list[str] = Field(default_factory=list)
    experience_gap_snapshot: list[ExperienceGap] = Field(default_factory=list)
    opportunity_lanes: list[OpportunityLane] = Field(default_factory=list)
    recommended_next_step: RecommendedNextStep | None = None
    confidence: float = 0.5


# ============================================================================
# BRAND BUILD SCHEMA
# ============================================================================

class SignatureExperience(BaseModel):
    """A signature brand experience."""
    name: str
    description: str
    why_it_matters: str


class GuestJourney(BaseModel):
    """Guest journey phases."""
    arrival: str
    stay: str
    departure: str


class GuestPersona(BaseModel):
    """Target guest persona."""
    name: str
    description: str
    spend_behavior: str


class FnBConcept(BaseModel):
    """F&B micro-concept."""
    name: str
    concept: str
    vibe: str


class BrandBlueprintLite(BaseModel):
    """Structured output for brand building.

    The core product output - an AI-generated brand blueprint.
    """
    type: str = "brand_blueprint_lite_v1"

    # Inputs captured
    inputs: dict[str, Any] = Field(default_factory=dict)

    # Core brand elements
    brand_name: str
    one_liner: str
    thesis: str
    pillars: list[str] = Field(default_factory=list)
    positioning_statement: str

    # Experience
    signature_experiences: list[SignatureExperience] = Field(default_factory=list)
    guest_journey: GuestJourney | None = None

    # Direction
    design_direction: str
    revenue_logic: str

    # Audience
    guest_personas: list[GuestPersona] = Field(default_factory=list)
    unmet_desires_solved: list[str] = Field(default_factory=list)

    # F&B
    fnb_concepts: list[FnBConcept] = Field(default_factory=list)

    # Summary
    investor_summary: str

    confidence: float = 0.5


# ============================================================================
# BELIEF STATE (POMDP-lite)
# ============================================================================

class SlotValues(BaseModel):
    """Slots to track for context."""
    location: str | None = None
    segment: str | None = None
    adr: float | None = None
    url: str | None = None
    developer_goal: str | None = None


class RetrievalState(BaseModel):
    """State of last retrieval."""
    top_posterior: float = 0.0
    entropy: float = 1.0
    chunks_retrieved: int = 0


class BeliefState(BaseModel):
    """POMDP-lite belief state for dialogue control.

    Tracks:
    - Mode probabilities
    - Slot fill status
    - Retrieval confidence
    - User stage (exploring vs committing)
    """
    mode_probs: dict[str, float] = Field(
        default_factory=lambda: {"insight": 0.5, "brand_build": 0.3, "demand_scan": 0.2}
    )
    slots: SlotValues = Field(default_factory=SlotValues)
    retrieval: RetrievalState = Field(default_factory=RetrievalState)
    stage: dict[str, float] = Field(
        default_factory=lambda: {"explore": 0.7, "commit": 0.3}
    )
    conversation_turns: int = 0

    def get_dominant_mode(self) -> ChatMode:
        """Get the most likely mode."""
        mode_name = max(self.mode_probs, key=self.mode_probs.get)
        return ChatMode(mode_name)

    def get_confidence(self) -> float:
        """Get confidence in current mode prediction."""
        probs = list(self.mode_probs.values())
        max_prob = max(probs)
        return max_prob

    def is_high_entropy(self, threshold: float = 0.7) -> bool:
        """Check if retrieval has high entropy (uncertainty)."""
        return self.retrieval.entropy > threshold


# ============================================================================
# CHAT MESSAGES & ARTIFACTS
# ============================================================================

class ChatMessage(BaseModel):
    """A chat message."""
    id: str | None = None
    project_id: str | None = None
    role: str  # "user" or "assistant"
    content: str
    mode: ChatMode | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ChatArtifact(BaseModel):
    """A saved artifact from chat."""
    id: str | None = None
    project_id: str | None = None
    artifact_type: str  # "insight_brief_v1", "demand_scan_lite_v1", etc.
    data: dict[str, Any]  # The actual artifact JSON
    sources: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# ROUTER OUTPUT
# ============================================================================

class RouterOutput(BaseModel):
    """Output from the mode router."""
    p_insight: float
    p_brand_build: float
    p_demand_scan: float
    confidence: float
    slots_detected: SlotValues
    slots_needed: list[str] = Field(default_factory=list)

    def get_mode(self) -> ChatMode:
        """Get highest probability mode."""
        probs = {
            ChatMode.INSIGHT: self.p_insight,
            ChatMode.BRAND_BUILD: self.p_brand_build,
            ChatMode.DEMAND_SCAN: self.p_demand_scan,
        }
        return max(probs, key=probs.get)

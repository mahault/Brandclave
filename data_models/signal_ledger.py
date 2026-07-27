"""Signal Ledger schemas — BrandClave's longitudinal prediction record.

Every demand hypothesis becomes a timestamped, append-only record: the signal,
the prediction, the evidence, the decision and the eventual outcome. The ledger
exists so BrandClave can later prove a prediction was captured before the
outcome was known, and measure its own accuracy as a KPI.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EvidenceStage(str, Enum):
    """Progression from cultural signal toward financeable demand."""

    AWARENESS = "awareness"
    ENGAGEMENT = "engagement"
    DECLARED_INTENT = "declared_intent"
    WILLINGNESS_TO_PAY = "willingness_to_pay"
    DEPOSIT = "deposit"
    CONTRACT = "contract"
    OPERATING_REVENUE = "operating_revenue"


class PredictionStatus(str, Enum):
    """Lifecycle of a ledger prediction."""

    OPEN = "open"
    RESOLVED_HIT = "resolved_hit"
    RESOLVED_MISS = "resolved_miss"
    FALSIFIED = "falsified"
    WITHDRAWN = "withdrawn"


class LedgerEventType(str, Enum):
    """Kinds of append-only events attached to a prediction."""

    EVIDENCE = "evidence"
    OUTCOME = "outcome"
    DECISION = "decision"
    NOTE = "note"


class ForecastItem(BaseModel):
    """One measurable prediction with a time horizon and uncertainty range.

    A vague trend statement that can never be disproven does not qualify;
    every forecast needs a metric, a range and a horizon date.
    """

    metric: str = Field(..., description="What is predicted (e.g. ADR, occupancy, memberships)")
    unit: str = Field(default="", description="Unit of the metric (USD, %, count)")
    predicted_low: float = Field(..., description="Lower bound of the predicted range")
    predicted_high: float = Field(..., description="Upper bound of the predicted range")
    horizon_date: datetime = Field(..., description="When the forecast should be evaluated")
    confidence: float = Field(..., ge=0, le=1, description="Stated probability the outcome lands in range")
    falsifier: str | None = Field(None, description="Observation that would falsify this forecast")


class PredictionRecordCreate(BaseModel):
    """Schema for capturing a new prediction. Core fields are immutable once written."""

    title: str = Field(..., description="Short name for the hypothesis")
    signal_date: datetime = Field(..., description="When the signal was first identified")
    signal_source: str = Field(..., description="Where the signal came from (search, social, interviews, mobility...)")
    hypothesis: str = Field(..., description="The societal or behavioral change believed to be occurring")
    product_implication: str = Field(..., description="What physical concept should exist because of the change")
    location_thesis: str | None = Field(None, description="Where the audience and economics make it most plausible")
    forecasts: list[ForecastItem] = Field(default_factory=list, description="Measurable predictions")
    uncertainty_notes: str | None = Field(None, description="What is known, inferred, and what would falsify the thesis")
    methodology_version: str = Field(default="v1", description="Version of the methodology that produced the signal")
    project: str | None = Field(None, description="Client project, internal concept, SENTIENT experiment, Living Lab...")
    source_trend_ids: list[str] = Field(default_factory=list, description="TrendSignal IDs behind this prediction")
    source_content_ids: list[str] = Field(default_factory=list, description="RawContent IDs preserved as source data")
    metadata: dict[str, Any] = Field(default_factory=dict)


class LedgerEventCreate(BaseModel):
    """Schema for appending an event to an existing prediction."""

    event_type: LedgerEventType
    description: str = Field(..., description="What happened")
    stage: EvidenceStage | None = Field(None, description="Evidence stage reached, for evidence events")
    metric: str | None = Field(None, description="Metric being reported, for outcome events")
    actual_value: float | None = Field(None, description="Realized value, for outcome events")
    evidence_refs: list[str] = Field(default_factory=list, description="URLs, document IDs or file refs preserving the evidence")
    metadata: dict[str, Any] = Field(default_factory=dict)


class OutcomeResult(BaseModel):
    """Computed comparison between a forecast and a reported outcome."""

    metric: str
    predicted_low: float
    predicted_high: float
    actual_value: float
    hit: bool = Field(..., description="Whether the actual value landed inside the predicted range")
    error_from_midpoint: float = Field(..., description="Actual minus range midpoint")
    error_pct: float | None = Field(None, description="Error as a fraction of the midpoint, when defined")


class LedgerMetrics(BaseModel):
    """Prediction-accuracy KPIs across the ledger."""

    total_predictions: int = 0
    open_predictions: int = 0
    resolved_predictions: int = 0
    hit_rate: float | None = Field(None, description="Share of resolved forecasts that landed in range")
    mean_abs_error_pct: float | None = Field(None, description="Mean absolute forecast error vs range midpoint")
    calibration_gap: float | None = Field(
        None, description="Mean stated confidence minus realized hit rate (positive = overconfident)"
    )
    predictions_by_stage: dict[str, int] = Field(default_factory=dict, description="Highest evidence stage reached per prediction")

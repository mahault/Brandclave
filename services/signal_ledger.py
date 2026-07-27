"""Signal Ledger service — capture predictions, append evidence, measure accuracy.

Core rule: a prediction's content is sealed at creation with a SHA-256 hash and
never edited. Corrections, evidence, outcomes and decisions are appended as
events, so the ledger remains auditable end to end.
"""

import hashlib
import json
import logging
from datetime import datetime

from sqlalchemy.orm import Session

from data_models.signal_ledger import (
    EvidenceStage,
    LedgerEventCreate,
    LedgerEventType,
    LedgerMetrics,
    OutcomeResult,
    PredictionRecordCreate,
    PredictionStatus,
)
from db.models import LedgerEventModel, PredictionRecordModel

logger = logging.getLogger(__name__)

# Ordered progression used to track the highest stage a prediction has reached
STAGE_ORDER = [stage.value for stage in EvidenceStage]


def _seal_hash(record: PredictionRecordCreate, recorded_at: datetime) -> str:
    """SHA-256 over the canonical prediction payload, proving it existed as-of recorded_at."""
    payload = record.model_dump(mode="json")
    payload["recorded_at"] = recorded_at.isoformat()
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def create_prediction(db: Session, record: PredictionRecordCreate) -> PredictionRecordModel:
    """Capture a new prediction and seal it with a content hash."""
    recorded_at = datetime.utcnow()
    model = PredictionRecordModel(
        title=record.title,
        signal_date=record.signal_date,
        signal_source=record.signal_source,
        hypothesis=record.hypothesis,
        product_implication=record.product_implication,
        location_thesis=record.location_thesis,
        forecasts=[f.model_dump(mode="json") for f in record.forecasts],
        uncertainty_notes=record.uncertainty_notes,
        methodology_version=record.methodology_version,
        project=record.project,
        source_trend_ids=record.source_trend_ids,
        source_content_ids=record.source_content_ids,
        recorded_at=recorded_at,
        content_hash=_seal_hash(record, recorded_at),
        status=PredictionStatus.OPEN.value,
        metadata_json=record.metadata,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    logger.info("Ledger prediction recorded: %s (%s)", model.title, model.id)
    return model


def verify_prediction(db: Session, prediction_id: str) -> bool:
    """Re-derive the content hash and confirm the sealed fields are untouched."""
    model = get_prediction(db, prediction_id)
    if model is None:
        raise ValueError(f"Prediction not found: {prediction_id}")
    record = PredictionRecordCreate(
        title=model.title,
        signal_date=model.signal_date,
        signal_source=model.signal_source,
        hypothesis=model.hypothesis,
        product_implication=model.product_implication,
        location_thesis=model.location_thesis,
        forecasts=model.forecasts or [],
        uncertainty_notes=model.uncertainty_notes,
        methodology_version=model.methodology_version,
        project=model.project,
        source_trend_ids=model.source_trend_ids or [],
        source_content_ids=model.source_content_ids or [],
        metadata=model.metadata_json or {},
    )
    return _seal_hash(record, model.recorded_at) == model.content_hash


def append_event(db: Session, prediction_id: str, event: LedgerEventCreate) -> LedgerEventModel:
    """Append an evidence/outcome/decision/note event to a prediction."""
    prediction = get_prediction(db, prediction_id)
    if prediction is None:
        raise ValueError(f"Prediction not found: {prediction_id}")

    model = LedgerEventModel(
        prediction_id=prediction_id,
        event_type=event.event_type.value,
        description=event.description,
        stage=event.stage.value if event.stage else None,
        metric=event.metric,
        actual_value=event.actual_value,
        evidence_refs=event.evidence_refs,
        metadata_json=event.metadata,
    )

    # Outcome events are scored against the sealed forecast at write time
    if event.event_type == LedgerEventType.OUTCOME and event.metric is not None and event.actual_value is not None:
        result = _score_outcome(prediction, event.metric, event.actual_value)
        if result is not None:
            model.outcome_json = result.model_dump(mode="json")

    # Track the highest evidence stage the prediction has reached
    if event.stage is not None:
        current = prediction.highest_evidence_stage
        if current is None or STAGE_ORDER.index(event.stage.value) > STAGE_ORDER.index(current):
            prediction.highest_evidence_stage = event.stage.value

    db.add(model)
    db.commit()
    db.refresh(model)
    return model


def _score_outcome(prediction: PredictionRecordModel, metric: str, actual_value: float) -> OutcomeResult | None:
    """Compare a reported outcome against the sealed forecast for that metric."""
    for forecast in prediction.forecasts or []:
        if forecast.get("metric") != metric:
            continue
        low, high = forecast["predicted_low"], forecast["predicted_high"]
        midpoint = (low + high) / 2
        return OutcomeResult(
            metric=metric,
            predicted_low=low,
            predicted_high=high,
            actual_value=actual_value,
            hit=low <= actual_value <= high,
            error_from_midpoint=actual_value - midpoint,
            error_pct=(actual_value - midpoint) / midpoint if midpoint != 0 else None,
        )
    logger.warning("Outcome reported for metric with no sealed forecast: %s", metric)
    return None


def resolve_prediction(
    db: Session, prediction_id: str, status: PredictionStatus, summary: str
) -> PredictionRecordModel:
    """Close a prediction. Resolution requires at least one outcome event unless withdrawn."""
    prediction = get_prediction(db, prediction_id)
    if prediction is None:
        raise ValueError(f"Prediction not found: {prediction_id}")
    if status == PredictionStatus.OPEN:
        raise ValueError("Cannot resolve a prediction to 'open'")

    if status in (PredictionStatus.RESOLVED_HIT, PredictionStatus.RESOLVED_MISS):
        outcomes = (
            db.query(LedgerEventModel)
            .filter(
                LedgerEventModel.prediction_id == prediction_id,
                LedgerEventModel.event_type == LedgerEventType.OUTCOME.value,
            )
            .count()
        )
        if outcomes == 0:
            raise ValueError("Record at least one outcome event before resolving hit/miss")

    prediction.status = status.value
    prediction.resolved_at = datetime.utcnow()
    prediction.resolution_summary = summary
    db.commit()
    db.refresh(prediction)
    return prediction


def get_prediction(db: Session, prediction_id: str) -> PredictionRecordModel | None:
    return db.query(PredictionRecordModel).filter(PredictionRecordModel.id == prediction_id).first()


def list_predictions(
    db: Session,
    status: str | None = None,
    project: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[PredictionRecordModel]:
    query = db.query(PredictionRecordModel)
    if status:
        query = query.filter(PredictionRecordModel.status == status)
    if project:
        query = query.filter(PredictionRecordModel.project == project)
    return query.order_by(PredictionRecordModel.recorded_at.desc()).offset(offset).limit(limit).all()


def list_events(db: Session, prediction_id: str) -> list[LedgerEventModel]:
    return (
        db.query(LedgerEventModel)
        .filter(LedgerEventModel.prediction_id == prediction_id)
        .order_by(LedgerEventModel.recorded_at.asc())
        .all()
    )


def compute_metrics(db: Session) -> LedgerMetrics:
    """Prediction-accuracy KPIs: hit rate, forecast error, calibration, stage funnel."""
    predictions = db.query(PredictionRecordModel).all()
    metrics = LedgerMetrics(total_predictions=len(predictions))
    metrics.open_predictions = sum(1 for p in predictions if p.status == PredictionStatus.OPEN.value)
    metrics.resolved_predictions = sum(
        1
        for p in predictions
        if p.status in (PredictionStatus.RESOLVED_HIT.value, PredictionStatus.RESOLVED_MISS.value)
    )

    for prediction in predictions:
        if prediction.highest_evidence_stage:
            stage = prediction.highest_evidence_stage
            metrics.predictions_by_stage[stage] = metrics.predictions_by_stage.get(stage, 0) + 1

    # Scored outcomes across all predictions, paired with the stated confidence
    hits: list[bool] = []
    abs_errors_pct: list[float] = []
    confidences: list[float] = []
    scored = (
        db.query(LedgerEventModel)
        .filter(
            LedgerEventModel.event_type == LedgerEventType.OUTCOME.value,
            LedgerEventModel.outcome_json.isnot(None),
        )
        .all()
    )
    predictions_by_id = {p.id: p for p in predictions}
    for event in scored:
        outcome = event.outcome_json
        hits.append(bool(outcome["hit"]))
        if outcome.get("error_pct") is not None:
            abs_errors_pct.append(abs(outcome["error_pct"]))
        prediction = predictions_by_id.get(event.prediction_id)
        if prediction is not None:
            for forecast in prediction.forecasts or []:
                if forecast.get("metric") == outcome["metric"]:
                    confidences.append(forecast["confidence"])
                    break

    if hits:
        metrics.hit_rate = sum(hits) / len(hits)
    if abs_errors_pct:
        metrics.mean_abs_error_pct = sum(abs_errors_pct) / len(abs_errors_pct)
    if confidences and metrics.hit_rate is not None:
        metrics.calibration_gap = sum(confidences) / len(confidences) - metrics.hit_rate

    return metrics

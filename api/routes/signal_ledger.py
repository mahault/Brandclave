"""Signal Ledger API routes.

Capture demand predictions before outcomes are known, append evidence as it
accumulates, and report prediction-accuracy KPIs.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from data_models.signal_ledger import (
    LedgerEventCreate,
    LedgerMetrics,
    PredictionRecordCreate,
    PredictionStatus,
)
from db.database import get_db
from services import signal_ledger

logger = logging.getLogger(__name__)

router = APIRouter()


def _prediction_dict(model) -> dict:
    return {
        "id": model.id,
        "title": model.title,
        "signal_date": model.signal_date,
        "signal_source": model.signal_source,
        "hypothesis": model.hypothesis,
        "product_implication": model.product_implication,
        "location_thesis": model.location_thesis,
        "forecasts": model.forecasts or [],
        "uncertainty_notes": model.uncertainty_notes,
        "methodology_version": model.methodology_version,
        "project": model.project,
        "source_trend_ids": model.source_trend_ids or [],
        "source_content_ids": model.source_content_ids or [],
        "recorded_at": model.recorded_at,
        "content_hash": model.content_hash,
        "status": model.status,
        "resolved_at": model.resolved_at,
        "resolution_summary": model.resolution_summary,
        "highest_evidence_stage": model.highest_evidence_stage,
    }


def _event_dict(model) -> dict:
    return {
        "id": model.id,
        "prediction_id": model.prediction_id,
        "event_type": model.event_type,
        "description": model.description,
        "stage": model.stage,
        "metric": model.metric,
        "actual_value": model.actual_value,
        "outcome": model.outcome_json,
        "evidence_refs": model.evidence_refs or [],
        "recorded_at": model.recorded_at,
    }


@router.post("/signal-ledger/predictions")
async def create_prediction(record: PredictionRecordCreate, db: Session = Depends(get_db)):
    """Record a new prediction. Its content is sealed with a hash and cannot be edited."""
    model = signal_ledger.create_prediction(db, record)
    return _prediction_dict(model)


@router.get("/signal-ledger/predictions")
async def list_predictions(
    status: str | None = Query(None, description="Filter by status"),
    project: str | None = Query(None, description="Filter by project"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List predictions, newest first."""
    models = signal_ledger.list_predictions(db, status=status, project=project, limit=limit, offset=offset)
    return {"predictions": [_prediction_dict(m) for m in models], "count": len(models)}


@router.get("/signal-ledger/predictions/{prediction_id}")
async def get_prediction(prediction_id: str, db: Session = Depends(get_db)):
    """Get a prediction with its full event history and hash verification."""
    model = signal_ledger.get_prediction(db, prediction_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    events = signal_ledger.list_events(db, prediction_id)
    return {
        **_prediction_dict(model),
        "hash_verified": signal_ledger.verify_prediction(db, prediction_id),
        "events": [_event_dict(e) for e in events],
    }


@router.post("/signal-ledger/predictions/{prediction_id}/events")
async def append_event(prediction_id: str, event: LedgerEventCreate, db: Session = Depends(get_db)):
    """Append an evidence, outcome, decision or note event to a prediction."""
    try:
        model = signal_ledger.append_event(db, prediction_id, event)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return _event_dict(model)


@router.post("/signal-ledger/predictions/{prediction_id}/resolve")
async def resolve_prediction(
    prediction_id: str,
    status: PredictionStatus,
    summary: str,
    db: Session = Depends(get_db),
):
    """Close a prediction as hit, miss, falsified or withdrawn."""
    try:
        model = signal_ledger.resolve_prediction(db, prediction_id, status, summary)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _prediction_dict(model)


@router.get("/signal-ledger/metrics", response_model=LedgerMetrics)
async def get_metrics(db: Session = Depends(get_db)):
    """Prediction-accuracy KPIs: hit rate, forecast error, calibration, stage funnel."""
    return signal_ledger.compute_metrics(db)

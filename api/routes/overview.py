"""Overview API — one call that feeds the dashboard's Signal Room.

Aggregates what the front page needs into a single response so the first paint
does not fan out into six requests: headline KPIs with week-over-week deltas, a
30-day intake series, per-city demand curves with movers, source freshness and
the Signal Ledger's accuracy metrics.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import case, func
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import (
    DemandMetricModel,
    HotelierMoveModel,
    RawContentModel,
    TrendSignalModel,
)
from ingestion.registry import get_registry, retired_sources
from services import signal_ledger

logger = logging.getLogger(__name__)

router = APIRouter()

WINDOW_DAYS = 7


def _pct_change(current: float, previous: float) -> float | None:
    if previous <= 0:
        return None
    return (current - previous) / previous


def _count_between(db: Session, column, start: datetime, end: datetime) -> int:
    return db.query(func.count()).filter(column >= start, column < end).scalar() or 0


def _kpis(db: Session, now: datetime) -> dict:
    week_ago = now - timedelta(days=WINDOW_DAYS)
    two_weeks_ago = now - timedelta(days=2 * WINDOW_DAYS)

    def window(column, extra=None):
        base = db.query(func.count()).filter(column >= week_ago, column < now)
        prev_q = db.query(func.count()).filter(column >= two_weeks_ago, column < week_ago)
        if extra is not None:
            base = base.filter(extra)
            prev_q = prev_q.filter(extra)
        cur = base.scalar() or 0
        prev = prev_q.scalar() or 0
        return {"last_7d": cur, "prior_7d": prev, "change_pct": _pct_change(cur, prev)}

    retired = retired_sources()
    live_rows = RawContentModel.source.notin_(retired) if retired else True
    content_total = db.query(func.count(RawContentModel.id)).filter(live_rows).scalar() or 0
    archived = db.query(func.count(RawContentModel.id)).filter(RawContentModel.source.in_(retired)).scalar() or 0 if retired else 0
    processed = (
        db.query(func.count(RawContentModel.id))
        .filter(RawContentModel.is_processed.is_(True), live_rows)
        .scalar()
        or 0
    )

    return {
        "content": {
            "total": content_total,
            "archived": archived,
            "processed": processed,
            **window(RawContentModel.scraped_at, live_rows),
        },
        "trends": {
            "total": db.query(func.count(TrendSignalModel.id)).scalar() or 0,
            **window(TrendSignalModel.last_updated),
        },
        "moves": {
            "total": db.query(func.count(HotelierMoveModel.id)).scalar() or 0,
            **window(HotelierMoveModel.extracted_at),
        },
    }


def _intake_series(db: Session, now: datetime, days: int = 30) -> list[dict]:
    """Items scraped per day for the sparkline, zero-filled."""
    start = (now - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    retired = retired_sources()
    rows = (
        db.query(func.date(RawContentModel.scraped_at), func.count(RawContentModel.id))
        .filter(RawContentModel.scraped_at >= start)
        .filter(RawContentModel.source.notin_(retired) if retired else True)
        .group_by(func.date(RawContentModel.scraped_at))
        .all()
    )
    counts = {str(d): n for d, n in rows}
    return [
        {"date": (start + timedelta(days=i)).strftime("%Y-%m-%d"), "count": counts.get((start + timedelta(days=i)).strftime("%Y-%m-%d"), 0)}
        for i in range(days)
    ]


def _demand_cities(db: Session, metric: str, limit: int) -> dict:
    """Per-city demand series plus week-over-week movers.

    Each city's series is indexed to its own 30-day mean (=100) so cities of
    very different absolute traffic share one axis honestly — New York's raw
    pageviews would otherwise flatten Lisbon into the baseline.
    """
    rows = (
        db.query(DemandMetricModel)
        .filter(DemandMetricModel.metric == metric)
        .order_by(DemandMetricModel.city, DemandMetricModel.date)
        .all()
    )
    by_city: dict[str, list[DemandMetricModel]] = defaultdict(list)
    for row in rows:
        by_city[row.city].append(row)

    cities = []
    for city, points in by_city.items():
        if not points:
            continue
        values = [p.value for p in points]
        mean = sum(values) / len(values)
        # Daily series compare the last 7 days with the 7 before; coarser
        # series (monthly, quarterly) compare the last point with the previous;
        # a single snapshot has no change yet and is reported as a level.
        window = WINDOW_DAYS if len(points) >= 2 * WINDOW_DAYS else 1
        recent = values[-window:]
        prior = values[-2 * window : -window] if len(values) > window else []
        recent_avg = sum(recent) / len(recent)
        prior_avg = sum(prior) / len(prior) if prior else 0.0
        cities.append(
            {
                "city": city,
                "country": points[0].country,
                "latest_date": points[-1].date.strftime("%Y-%m-%d"),
                "mean_daily": round(mean),
                "recent_7d_avg": round(recent_avg),
                "prior_7d_avg": round(prior_avg),
                "change_pct": _pct_change(recent_avg, prior_avg),
                "series": [
                    {"date": p.date.strftime("%Y-%m-%d"), "value": p.value, "index": round(100 * p.value / mean, 1) if mean else None}
                    for p in points
                ],
            }
        )

    snapshot = all(len(c["series"]) == 1 for c in cities) if cities else False
    if snapshot:
        # No change to rank by: order by level so the "movers" read as a league table.
        cities.sort(key=lambda c: -c["recent_7d_avg"])
        movers_up = [c["city"] for c in cities[:limit]]
        movers_down = [c["city"] for c in cities[-limit:][::-1]]
    else:
        cities.sort(key=lambda c: (c["change_pct"] is None, -(c["change_pct"] or 0)))
        movers_up = [c["city"] for c in cities if c["change_pct"] is not None][:limit]
        movers_down = [
            c["city"]
            for c in sorted([c for c in cities if c["change_pct"] is not None], key=lambda c: c["change_pct"])[:limit]
        ]
    return {
        "metric": metric,
        "snapshot": snapshot,
        "city_count": len(cities),
        "cities": cities,
        "movers_up": movers_up,
        "movers_down": movers_down,
    }


def _source_freshness(db: Session, now: datetime) -> list[dict]:
    registry = get_registry()
    rows = (
        db.query(
            RawContentModel.source,
            func.count(RawContentModel.id),
            func.max(RawContentModel.scraped_at),
            func.sum(case((RawContentModel.scraped_at >= now - timedelta(days=WINDOW_DAYS), 1), else_=0)),
        )
        .group_by(RawContentModel.source)
        .all()
    )
    metric_rows = (
        db.query(
            DemandMetricModel.source,
            func.count(DemandMetricModel.id),
            func.max(DemandMetricModel.scraped_at),
            func.sum(case((DemandMetricModel.scraped_at >= now - timedelta(days=WINDOW_DAYS), 1), else_=0)),
        )
        .group_by(DemandMetricModel.source)
        .all()
    )
    seen = {}
    for source, total, last, recent in list(rows) + list(metric_rows):
        spec = registry.get(source)
        seen[source] = {
            "source": source,
            "kind": spec.kind if spec else "content",
            "type": spec.type if spec else None,
            "status": spec.status if spec else "unknown",
            "total_items": int(total or 0),
            "items_7d": int(recent or 0),
            "last_scraped": last.isoformat() if last else None,
            "hours_since": round((now - last).total_seconds() / 3600, 1) if last else None,
        }
    # Registered-but-silent sources still belong on the coverage strip.
    for name, spec in registry.items():
        if name not in seen and spec.status == "active":
            seen[name] = {
                "source": name,
                "kind": spec.kind,
                "type": spec.type,
                "status": spec.status,
                "total_items": 0,
                "items_7d": 0,
                "last_scraped": None,
                "hours_since": None,
            }
    ordered = sorted(seen.values(), key=lambda s: (s["hours_since"] is None, s["hours_since"] or 0))
    return ordered


@router.get("/overview")
async def get_overview(
    demand_metric: str = Query("wikipedia_pageviews", description="Demand metric to chart"),
    movers: int = Query(5, ge=1, le=15, description="How many movers to name each way"),
    db: Session = Depends(get_db),
):
    """Everything the Signal Room needs for its first paint."""
    now = datetime.utcnow()
    top_trends = (
        db.query(TrendSignalModel).order_by(TrendSignalModel.last_updated.desc(), TrendSignalModel.strength_score.desc()).limit(6).all()
    )
    latest_moves = db.query(HotelierMoveModel).order_by(HotelierMoveModel.extracted_at.desc()).limit(5).all()

    registry = get_registry()
    status_counts = defaultdict(int)
    for spec in registry.values():
        status_counts[spec.status] += 1

    return {
        "generated_at": now.isoformat(),
        "window_days": WINDOW_DAYS,
        "kpis": _kpis(db, now),
        "intake": _intake_series(db, now),
        "demand": _demand_cities(db, demand_metric, movers),
        "sources": {
            "registry": dict(status_counts),
            "freshness": _source_freshness(db, now),
        },
        "ledger": signal_ledger.compute_metrics(db).model_dump(),
        "trends": [
            {
                "id": t.id,
                "name": t.name,
                "strength_score": t.strength_score,
                "white_space_score": t.white_space_score,
                "volume": t.volume,
                "region": t.region,
                "last_updated": t.last_updated.isoformat() if t.last_updated else None,
                "why_it_matters": t.why_it_matters,
            }
            for t in top_trends
        ],
        "moves": [
            {
                "id": m.id,
                "title": m.title,
                "company": m.company,
                "move_type": m.move_type,
                "market": m.market,
                "investment_amount": m.investment_amount,
                "published_at": m.published_at.isoformat() if m.published_at else None,
                "source_name": m.source_name,
                "confidence_score": m.confidence_score,
            }
            for m in latest_moves
        ],
    }

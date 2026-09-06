"""Overview API — one call that feeds the dashboard's Signal Room.

Aggregates what the front page needs into a single response so the first paint
does not fan out into six requests: headline KPIs with week-over-week deltas, a
30-day intake series, per-city demand curves with movers, source freshness and
the Signal Ledger's accuracy metrics.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
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


# Region slots for categorical colour: fixed order, never cycled (dataviz rule).
REGION_SLOTS = ["europe", "asia", "north_america", "other"]
COUNTRY_REGION = {
    "Portugal": "europe", "Spain": "europe", "France": "europe", "Italy": "europe", "Greece": "europe",
    "UK": "europe", "Iceland": "europe", "Turkey": "europe", "Georgia": "europe",
    "Japan": "asia", "Indonesia": "asia", "India": "asia", "Thailand": "asia", "South Korea": "asia",
    "Singapore": "asia", "UAE": "asia",
    "USA": "north_america", "Mexico": "north_america",
}

MOVE_GROUPS = {
    "acquisition": "deals", "expansion": "deals", "reflag": "deals", "partnership": "deals",
    "launch": "product", "concept": "product", "renovation": "product", "repositioning": "product",
    "technology": "technology",
}
MOVE_GROUP_ORDER = ["deals", "product", "technology", "other"]


def _trend_map(db: Session) -> list[dict]:
    """Every trend as a point: strength x white space, sized by volume."""
    rows = db.query(TrendSignalModel).order_by(TrendSignalModel.strength_score.desc()).limit(200).all()
    return [
        {
            "id": t.id,
            "name": t.name,
            "strength": round(t.strength_score or 0, 3),
            "white_space": round(t.white_space_score or 0, 3),
            "volume": t.volume or 0,
            "region": (t.region if t.region in REGION_SLOTS else "other"),
            "region_raw": t.region,
            "last_updated": t.last_updated.isoformat() if t.last_updated else None,
        }
        for t in rows
    ]


def _latest_metric(db: Session, metric: str) -> dict[str, float]:
    """city -> most recent value for a metric."""
    rows = (
        db.query(DemandMetricModel.city, DemandMetricModel.value, DemandMetricModel.date)
        .filter(DemandMetricModel.metric == metric)
        .order_by(DemandMetricModel.city, DemandMetricModel.date.desc())
        .all()
    )
    out: dict[str, float] = {}
    for city, value, _date in rows:
        out.setdefault(city, value)
    return out


def _city_matrix(db: Session, demand: dict) -> list[dict]:
    """Attention momentum (Wikipedia, week over week) against hotel supply
    (OSM), bubble by Airbnb listings. Only cities with both axes plot."""
    hotels = _latest_metric(db, "osm_hotels")
    listings = _latest_metric(db, "airbnb_listings")
    velocity = _latest_metric(db, "airbnb_reviews_per_month")
    restaurants = _latest_metric(db, "osm_restaurants")
    points = []
    for c in demand.get("cities", []):
        if c["city"] not in hotels or c["change_pct"] is None:
            continue
        points.append(
            {
                "city": c["city"],
                "country": c["country"],
                "region": COUNTRY_REGION.get(c["country"] or "", "other"),
                "attention_change_pct": round(c["change_pct"], 4),
                "attention_daily": c["recent_7d_avg"],
                "hotels": hotels[c["city"]],
                "restaurants": restaurants.get(c["city"]),
                "airbnb_listings": listings.get(c["city"]),
                "airbnb_reviews_per_month": velocity.get(c["city"]),
            }
        )
    return points


def _moves_by_week(db: Session, now: datetime, weeks: int = 12) -> dict:
    start = (now - timedelta(weeks=weeks)).replace(hour=0, minute=0, second=0, microsecond=0)
    start -= timedelta(days=start.weekday())  # Monday
    rows = (
        db.query(HotelierMoveModel.move_type, HotelierMoveModel.published_at, HotelierMoveModel.extracted_at, HotelierMoveModel.source_name)
        .filter(func.coalesce(HotelierMoveModel.published_at, HotelierMoveModel.extracted_at) >= start)
        .all()
    )
    buckets: dict[str, dict[str, int]] = {}
    filings = 0
    for move_type, published, extracted, source in rows:
        when = published or extracted
        week_start = (when - timedelta(days=when.weekday())).strftime("%Y-%m-%d")
        group = MOVE_GROUPS.get((move_type or "other").lower(), "other")
        buckets.setdefault(week_start, {g: 0 for g in MOVE_GROUP_ORDER})[group] += 1
        if source == "sec_edgar":
            filings += 1
    series = []
    cursor = start
    while cursor <= now:
        key = cursor.strftime("%Y-%m-%d")
        series.append({"week": key, **buckets.get(key, {g: 0 for g in MOVE_GROUP_ORDER})})
        cursor += timedelta(weeks=1)
    return {"groups": MOVE_GROUP_ORDER, "weeks": series, "total": len(rows), "from_filings": filings}


def _intake_by_type(db: Session, now: datetime, days: int = 30) -> list[dict]:
    start = (now - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    retired = retired_sources()
    rows = (
        db.query(func.date(RawContentModel.scraped_at), RawContentModel.source_type, func.count(RawContentModel.id))
        .filter(RawContentModel.scraped_at >= start)
        .filter(RawContentModel.source.notin_(retired) if retired else True)
        .group_by(func.date(RawContentModel.scraped_at), RawContentModel.source_type)
        .all()
    )
    table: dict[str, dict[str, int]] = {}
    for day, source_type, n in rows:
        table.setdefault(str(day), {})[source_type or "other"] = n
    return [
        {"date": (start + timedelta(days=i)).strftime("%Y-%m-%d"), **table.get((start + timedelta(days=i)).strftime("%Y-%m-%d"), {})}
        for i in range(days)
    ]


def _companies(db: Session, limit: int = 12) -> list[dict]:
    """Who is moving most, with the mix of what they are doing."""
    rows = db.query(HotelierMoveModel).order_by(HotelierMoveModel.extracted_at.desc()).limit(600).all()
    by: dict[str, dict] = {}
    for m in rows:
        name = (m.company or "Unknown").strip()
        if name.lower() in {"unknown", "n/a", ""}:
            continue
        entry = by.setdefault(name, {"company": name, "moves": 0, "filings": 0, "groups": {g: 0 for g in MOVE_GROUP_ORDER}, "markets": set(), "latest": None, "latest_at": None})
        entry["moves"] += 1
        if m.source_name == "sec_edgar":
            entry["filings"] += 1
        entry["groups"][MOVE_GROUPS.get((m.move_type or "other").lower(), "other")] += 1
        if m.market:
            entry["markets"].add(m.market)
        when = m.published_at or m.extracted_at
        if when and (entry["latest_at"] is None or when > entry["latest_at"]):
            entry["latest_at"], entry["latest"] = when, m.title
    ranked = sorted(by.values(), key=lambda e: (-e["moves"], -e["filings"]))[:limit]
    for e in ranked:
        e["markets"] = sorted(e["markets"])[:3]
        e["latest_at"] = e["latest_at"].isoformat() if e["latest_at"] else None
    return ranked


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

    demand = _demand_cities(db, demand_metric, movers)
    attention = demand if demand_metric == "wikipedia_pageviews" else _demand_cities(db, "wikipedia_pageviews", movers)

    return {
        "generated_at": now.isoformat(),
        "window_days": WINDOW_DAYS,
        "kpis": _kpis(db, now),
        "intake": _intake_series(db, now),
        "intake_by_type": _intake_by_type(db, now),
        "demand": demand,
        "trend_map": _trend_map(db),
        "city_matrix": _city_matrix(db, attention),
        "moves_by_week": _moves_by_week(db, now),
        "companies": _companies(db),
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


COUNTRY_TO_EUROSTAT = {
    "Portugal": "Portugal", "Spain": "Spain", "France": "France", "Italy": "Italy", "Greece": "Greece",
    "Iceland": "Iceland", "Turkey": "Türkiye", "Germany": "Germany", "Netherlands": "Netherlands",
    "Austria": "Austria", "Croatia": "Croatia", "Ireland": "Ireland", "Denmark": "Denmark", "Sweden": "Sweden",
    "Poland": "Poland", "Czechia": "Czechia", "Hungary": "Hungary",
}


@router.get("/overview/city/{city}")
async def get_city_facts(city: str, db: Session = Depends(get_db)):
    """Everything the metric sources know about one city, for the Cities view."""
    name = city.strip()
    rows = (
        db.query(DemandMetricModel)
        .filter(func.lower(DemandMetricModel.city) == name.lower())
        .order_by(DemandMetricModel.metric, DemandMetricModel.date)
        .all()
    )
    series: dict[str, list[dict]] = defaultdict(list)
    country = None
    for r in rows:
        series[r.metric].append({"date": r.date.strftime("%Y-%m-%d"), "value": r.value})
        country = country or r.country
    attention = series.get("wikipedia_pageviews", [])
    change = None
    if len(attention) >= 14:
        recent = sum(p["value"] for p in attention[-7:]) / 7
        prior = sum(p["value"] for p in attention[-14:-7]) / 7
        change = _pct_change(recent, prior)
    latest = lambda m: (series[m][-1]["value"] if series.get(m) else None)  # noqa: E731

    nights = []
    eu_name = COUNTRY_TO_EUROSTAT.get(country or "")
    if eu_name:
        nights = [
            {"date": r.date.strftime("%Y-%m"), "value": r.value}
            for r in db.query(DemandMetricModel)
            .filter(DemandMetricModel.metric == "eurostat_nights_spent", DemandMetricModel.city == eu_name)
            .order_by(DemandMetricModel.date)
            .all()
        ]

    trend_hits = (
        db.query(TrendSignalModel)
        .filter(func.lower(TrendSignalModel.name).like(f"%{name.lower()}%") | func.lower(TrendSignalModel.description).like(f"%{name.lower()}%"))
        .order_by(TrendSignalModel.strength_score.desc())
        .limit(5)
        .all()
    )

    if not rows and not trend_hits:
        raise HTTPException(status_code=404, detail=f"No metrics on file for {name}")

    return {
        "city": name,
        "country": country,
        "attention": {"series": attention[-30:], "change_pct": change},
        "airbnb": {
            "listings": latest("airbnb_listings"),
            "entire_home_share": latest("airbnb_entire_home_share"),
            "median_price_local": latest("airbnb_median_price"),
            "reviews_per_month": latest("airbnb_reviews_per_month"),
        },
        "supply": {
            "hotels": latest("osm_hotels"),
            "restaurants": latest("osm_restaurants"),
            "nightlife": latest("osm_nightlife"),
            "attractions": latest("osm_attractions"),
        },
        "country_nights": {"country": eu_name, "series": nights[-24:]},
        "trends": [{"id": t.id, "name": t.name, "strength": t.strength_score, "white_space": t.white_space_score} for t in trend_hits],
    }

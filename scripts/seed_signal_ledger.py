#!/usr/bin/env python
"""Seed the Signal Ledger with open predictions derived from live trend signals.

The ledger only proves anything once predictions exist *before* outcomes, so an
empty ledger on a fresh deployment shows the mechanism but no track record.
This script stakes one measurable, falsifiable forecast per strong trend, sealed
with the ledger's content hash, and leaves every one of them OPEN. It never
invents outcomes: hit rate stays blank until real horizons pass and real
evidence is appended.

Forecast ranges are derived from the trend's own strength and white-space
scores, which is exactly the methodology the ledger is meant to audit.

Usage:
    python scripts/seed_signal_ledger.py            # top 8 trends
    python scripts/seed_signal_ledger.py --limit 12
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv()

from data_models.signal_ledger import ForecastItem, PredictionRecordCreate
from processing.trend_names import normalize_trend_title
from db.database import SessionLocal
from db.models import PredictionRecordModel, TrendSignalModel
from services import signal_ledger

logger = logging.getLogger("seed_signal_ledger")

METHODOLOGY_VERSION = "v1.1-trend-derived"
PROJECT = "BrandClave platform"


def _forecasts_for(trend: TrendSignalModel, now: datetime) -> list[ForecastItem]:
    """Two forecasts per trend: one on the signal itself, one on the market.

    - Signal persistence: will the trend's source volume still be growing in 90
      days? Falsifiable from our own corpus.
    - Market response: how many named operator moves will cite this theme in
      180 days? Falsifiable from Hotelier Bets extraction.
    """
    strength = max(0.05, min(0.95, trend.strength_score or 0.5))
    white_space = max(0.0, min(1.0, trend.white_space_score or 0.0))
    volume = max(1, trend.volume or 1)

    growth_low = round(volume * (1.0 + 0.15 * strength))
    growth_high = round(volume * (1.0 + 0.9 * strength))
    moves_low = round(1 + 3 * strength)
    moves_high = round(3 + 9 * strength + 3 * white_space)

    return [
        ForecastItem(
            metric="source_volume",
            unit="items",
            predicted_low=float(growth_low),
            predicted_high=float(growth_high),
            horizon_date=now + timedelta(days=90),
            confidence=round(0.45 + 0.4 * strength, 2),
            falsifier=f"Fewer than {growth_low} corpus items match this cluster at the horizon date",
        ),
        ForecastItem(
            metric="operator_moves_citing_theme",
            unit="count",
            predicted_low=float(moves_low),
            predicted_high=float(moves_high),
            horizon_date=now + timedelta(days=180),
            confidence=round(0.35 + 0.35 * strength, 2),
            falsifier=f"Fewer than {moves_low} extracted hotelier moves reference this theme by the horizon",
        ),
    ]


def seed(limit: int) -> int:
    db = SessionLocal()
    created = 0
    try:
        existing = {
            tid
            for (ids,) in db.query(PredictionRecordModel.source_trend_ids).all()
            for tid in (ids or [])
        }
        trends = (
            db.query(TrendSignalModel)
            .order_by(TrendSignalModel.strength_score.desc(), TrendSignalModel.white_space_score.desc())
            .limit(limit * 2)
            .all()
        )
        now = datetime.utcnow()
        # Clustering re-discovers the same theme under slightly different names
        # ("Connected Revenue Revolution" with and without quotes); one sealed
        # prediction per theme is the honest count.
        staked_titles: set[str] = set()
        for trend in trends:
            if created >= limit:
                break
            if trend.id in existing:
                continue
            title_key = normalize_trend_title(trend.name)
            if title_key in staked_titles:
                continue
            staked_titles.add(title_key)

            topics = ", ".join((trend.topics or [])[:4]) or "hospitality demand"
            record = PredictionRecordCreate(
                title=trend.name,
                signal_date=trend.first_seen or now,
                signal_source=f"social + trade press clustering ({trend.volume or 0} items)",
                hypothesis=trend.description,
                product_implication=trend.why_it_matters,
                location_thesis=trend.region,
                forecasts=_forecasts_for(trend, now),
                uncertainty_notes=(
                    f"Derived from cluster strength {trend.strength_score:.2f} and white-space "
                    f"{(trend.white_space_score or 0):.2f} over a {trend.time_window_days}-day window; topics: {topics}. "
                    "Falsified if source volume contracts or no operator acts on the theme by the horizon."
                ),
                methodology_version=METHODOLOGY_VERSION,
                project=PROJECT,
                source_trend_ids=[trend.id],
                source_content_ids=list((trend.source_content_ids or [])[:25]),
                metadata={"seeded_by": "scripts/seed_signal_ledger.py", "audience_segment": trend.audience_segment},
            )
            model = signal_ledger.create_prediction(db, record)
            created += 1
            logger.info("Sealed %s  %s", model.content_hash[:12], trend.name)
    finally:
        db.close()
    return created


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=8, help="Maximum predictions to create")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    n = seed(args.limit)
    print(f"Seeded {n} open prediction(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Scheduler management API routes."""

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from scheduler.scheduler import get_scheduler

logger = logging.getLogger(__name__)

router = APIRouter()


class JobResponse(BaseModel):
    """Response model for a scheduled job."""

    id: str
    name: str
    next_run: Optional[str]
    trigger: str
    pending: bool


class JobListResponse(BaseModel):
    """Response model for job list."""

    jobs: list[JobResponse]
    total: int
    scheduler_running: bool
    scheduler_available: bool


class AddJobRequest(BaseModel):
    """Request model for adding a job."""

    source_name: str
    interval_minutes: Optional[int] = 60
    cron_expression: Optional[str] = None


class JobActionResponse(BaseModel):
    """Response for job actions."""

    success: bool
    message: str
    job_id: str


@router.get("/scheduler/status")
def get_scheduler_status():
    """Get scheduler status."""
    scheduler = get_scheduler()
    return {
        "available": scheduler.is_available,
        "running": scheduler.is_running,
        "total_jobs": len(scheduler.get_jobs()) if scheduler.is_available else 0,
    }


@router.get("/scheduler/jobs", response_model=JobListResponse)
def get_scheduled_jobs():
    """Get all scheduled jobs."""
    scheduler = get_scheduler()
    jobs = scheduler.get_jobs()

    return JobListResponse(
        jobs=[JobResponse(**j) for j in jobs],
        total=len(jobs),
        scheduler_running=scheduler.is_running,
        scheduler_available=scheduler.is_available,
    )


@router.get("/scheduler/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str):
    """Get a specific job by ID."""
    scheduler = get_scheduler()

    if not scheduler.is_available:
        raise HTTPException(status_code=503, detail="Scheduler not available")

    job = scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse(**job)


@router.post("/scheduler/jobs", response_model=JobActionResponse)
def add_job(request: AddJobRequest):
    """Add a new scheduled scraper job."""
    scheduler = get_scheduler()

    if not scheduler.is_available:
        raise HTTPException(status_code=503, detail="Scheduler not available")

    # Check if source is registered
    if request.source_name not in scheduler._scraper_registry:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown source: {request.source_name}. Available: {list(scheduler._scraper_registry.keys())}",
        )

    job_id = scheduler.add_scraper_job(
        source_name=request.source_name,
        interval_minutes=request.interval_minutes,
        cron_expression=request.cron_expression,
    )

    if not job_id:
        raise HTTPException(status_code=500, detail="Failed to add job")

    return JobActionResponse(
        success=True,
        message=f"Job {job_id} added successfully",
        job_id=job_id,
    )


@router.delete("/scheduler/jobs/{job_id}", response_model=JobActionResponse)
def remove_job(job_id: str):
    """Remove a scheduled job."""
    scheduler = get_scheduler()

    if not scheduler.is_available:
        raise HTTPException(status_code=503, detail="Scheduler not available")

    if scheduler.remove_job(job_id):
        return JobActionResponse(
            success=True,
            message=f"Job {job_id} removed",
            job_id=job_id,
        )

    raise HTTPException(status_code=404, detail="Job not found")


@router.post("/scheduler/jobs/{job_id}/pause", response_model=JobActionResponse)
def pause_job(job_id: str):
    """Pause a scheduled job."""
    scheduler = get_scheduler()

    if not scheduler.is_available:
        raise HTTPException(status_code=503, detail="Scheduler not available")

    if scheduler.pause_job(job_id):
        return JobActionResponse(
            success=True,
            message=f"Job {job_id} paused",
            job_id=job_id,
        )

    raise HTTPException(status_code=404, detail="Job not found")


@router.post("/scheduler/jobs/{job_id}/resume", response_model=JobActionResponse)
def resume_job(job_id: str):
    """Resume a paused job."""
    scheduler = get_scheduler()

    if not scheduler.is_available:
        raise HTTPException(status_code=503, detail="Scheduler not available")

    if scheduler.resume_job(job_id):
        return JobActionResponse(
            success=True,
            message=f"Job {job_id} resumed",
            job_id=job_id,
        )

    raise HTTPException(status_code=404, detail="Job not found")


@router.post("/scheduler/jobs/{job_id}/run", response_model=JobActionResponse)
def run_job_now(job_id: str):
    """Trigger immediate execution of a job."""
    scheduler = get_scheduler()

    if not scheduler.is_available:
        raise HTTPException(status_code=503, detail="Scheduler not available")

    if scheduler.run_job_now(job_id):
        return JobActionResponse(
            success=True,
            message=f"Job {job_id} triggered for immediate execution",
            job_id=job_id,
        )

    raise HTTPException(status_code=404, detail="Job not found")


@router.post("/scheduler/start")
def start_scheduler():
    """Start the scheduler."""
    scheduler = get_scheduler()

    if not scheduler.is_available:
        raise HTTPException(status_code=503, detail="Scheduler not available")

    if scheduler.start():
        return {"status": "started", "message": "Scheduler started successfully"}

    raise HTTPException(status_code=500, detail="Failed to start scheduler")


@router.post("/scheduler/stop")
def stop_scheduler():
    """Stop the scheduler."""
    scheduler = get_scheduler()

    if not scheduler.is_available:
        raise HTTPException(status_code=503, detail="Scheduler not available")

    scheduler.shutdown(wait=False)
    return {"status": "stopped", "message": "Scheduler stopped"}


# The POMDP status is expensive: the schedule recommendation alone takes ~8 s
# of JAX work on a laptop and minutes on a small instance. Beliefs only move
# when the scheduler runs, so the payload is computed in a background thread,
# persisted to data/pomdp_status.json and served from there. A request never
# waits on JAX.
_POMDP_CACHE: dict[str, tuple[float, dict]] = {}
_POMDP_TTL_SECONDS = 3600.0
_POMDP_SNAPSHOT_MAX_AGE_SECONDS = 24 * 3600.0
_POMDP_SNAPSHOT = Path(__file__).resolve().parents[2] / "data" / "pomdp_status.json"
_POMDP_BUILDING = threading.Lock()


def compute_pomdp_payload() -> dict:
    """Run the full status computation (slow). Called off the request path."""
    scheduler = get_scheduler()
    if not scheduler.use_pomdp or scheduler.scraping_pomdp is None:
        return {"enabled": False, "reason": "Scraping POMDP not available or disabled"}
    payload = {
        "enabled": True,
        "status": scheduler.get_pomdp_status(),
        "next_recommended_source": scheduler.get_next_source_pomdp(),
        "recommended_schedule": scheduler.get_scraping_schedule_pomdp(budget_minutes=60),
        "computed_at": datetime.utcnow().isoformat() + "Z",
    }
    return payload


def write_pomdp_snapshot(payload: dict) -> None:
    _POMDP_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    tmp = _POMDP_SNAPSHOT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, default=str, indent=1), encoding="utf-8")
    tmp.replace(_POMDP_SNAPSHOT)


def _refresh_pomdp_snapshot_in_background() -> bool:
    """Start one background computation; returns False if one is already running."""
    if not _POMDP_BUILDING.acquire(blocking=False):
        return False

    def run():
        try:
            payload = compute_pomdp_payload()
            if payload.get("enabled"):
                write_pomdp_snapshot(payload)
            _POMDP_CACHE["status"] = (time.time(), payload)
        except Exception as exc:  # pragma: no cover - logged, never raised to a request
            logger.warning(f"POMDP snapshot refresh failed: {exc}")
        finally:
            _POMDP_BUILDING.release()

    threading.Thread(target=run, name="pomdp-snapshot", daemon=True).start()
    return True


def _snapshot_age_seconds(payload: dict) -> Optional[float]:
    stamp = payload.get("computed_at")
    if not stamp:
        return None
    try:
        computed = datetime.fromisoformat(stamp.rstrip("Z"))
    except ValueError:
        return None
    return (datetime.utcnow() - computed).total_seconds()


@router.get("/scheduler/pomdp")
def get_pomdp_status(refresh: bool = Query(False, description="Recompute in the background")):
    """Get POMDP (Active Inference) status and beliefs.

    Served from the last computed snapshot (see computed_at). The computation
    runs in the background when the snapshot is missing, older than a day, or
    refresh=true is passed; the response never waits on it.
    """
    hit = _POMDP_CACHE.get("status")
    if hit and time.time() - hit[0] < _POMDP_TTL_SECONDS and not refresh:
        return hit[1]

    payload: Optional[dict] = None
    if _POMDP_SNAPSHOT.exists():
        try:
            payload = json.loads(_POMDP_SNAPSHOT.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning(f"POMDP snapshot unreadable: {exc}")

    if payload is None:
        started = _refresh_pomdp_snapshot_in_background()
        return {
            "enabled": True,
            "warming": True,
            "building": started or _POMDP_BUILDING.locked(),
            "reason": "First status computation on this instance is running in the background.",
        }

    age = _snapshot_age_seconds(payload)
    if refresh or age is None or age > _POMDP_SNAPSHOT_MAX_AGE_SECONDS:
        _refresh_pomdp_snapshot_in_background()
    payload["snapshot_age_seconds"] = round(age) if age is not None else None
    _POMDP_CACHE["status"] = (time.time(), payload)
    return payload


@router.post("/scheduler/pomdp/recommend")
def get_pomdp_recommendation():
    """Get POMDP recommendation for next source to scrape.

    Uses Expected Free Energy minimization to balance:
    - Pragmatic value: sources likely to yield good content
    - Epistemic value: sources with uncertain state (exploration)
    """
    scheduler = get_scheduler()

    if not scheduler.use_pomdp or scheduler.scraping_pomdp is None:
        raise HTTPException(
            status_code=503,
            detail="Scraping POMDP not available",
        )

    return scheduler.get_next_source_pomdp()


@router.get("/scheduler/pomdp/beliefs")
def get_pomdp_beliefs():
    """Get detailed POMDP beliefs about each source.

    Shows:
    - Productivity belief (expected yield)
    - Freshness belief (how stale the source is)
    - Time since last scrape
    - Observation count

    Sources with low freshness have high epistemic value (info gain).
    """
    scheduler = get_scheduler()

    if not scheduler.use_pomdp or scheduler.scraping_pomdp is None:
        return {
            "enabled": False,
            "reason": "Scraping POMDP not available",
        }

    try:
        # Trigger staleness update
        scheduler.scraping_pomdp._update_staleness_beliefs()

        status = scheduler.scraping_pomdp.get_status()
        sources = status.get("sources", {})

        # Sort by freshness (stalest first)
        sorted_sources = sorted(
            sources.items(),
            key=lambda x: x[1].get("freshness", 1.0)
        )

        return {
            "enabled": True,
            "total_observations": status.get("total_observations", 0),
            "free_energy": status.get("free_energy", 0),
            "sources": [
                {
                    "name": name,
                    "productivity": round(info.get("productivity", 0.5), 3),
                    "freshness": round(info.get("freshness", 0.5), 3),
                    "error_rate": round(info.get("error_rate", 0), 3),
                    "observations": info.get("observations", 0),
                    "last_scraped": info.get("last_scraped"),
                    "needs_attention": info.get("freshness", 1.0) < 0.3,
                }
                for name, info in sorted_sources
            ],
        }
    except Exception as e:
        return {"enabled": True, "error": str(e)}


@router.post("/scheduler/scrape-all")
def scrape_all_sources(
    sources: Optional[list[str]] = Query(None, description="Specific sources to scrape (default: all hospitality)")
):
    """Run all scrapers in sequence for initial data collection.

    This is useful for:
    - Initial setup when database is empty
    - Manual refresh of all sources
    - Testing that all scrapers work correctly

    Note: This runs synchronously and may take several minutes.
    """
    from scripts.run_crawlers import get_scraper_class, SCRAPERS

    # Default to every registry-active source
    if sources is None:
        from ingestion.registry import active_sources
        sources = active_sources()

    results = []
    for source in sources:
        if source not in SCRAPERS:
            results.append({"source": source, "status": "skipped", "error": "Unknown source"})
            continue

        try:
            logger.info(f"Running scraper: {source}")
            scraper_class = get_scraper_class(source)
            with scraper_class() as scraper:
                result = scraper.run()
            results.append({
                "source": source,
                "status": "completed",
                "items": result.get("items_scraped", 0),
            })
        except Exception as e:
            logger.error(f"Scraper {source} failed: {e}")
            results.append({"source": source, "status": "failed", "error": str(e)})

    return {
        "total_sources": len(sources),
        "completed": sum(1 for r in results if r["status"] == "completed"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "results": results,
    }

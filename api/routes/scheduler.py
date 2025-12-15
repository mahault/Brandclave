"""Scheduler management API routes."""

import logging
from datetime import datetime
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
async def get_scheduler_status():
    """Get scheduler status."""
    scheduler = get_scheduler()
    return {
        "available": scheduler.is_available,
        "running": scheduler.is_running,
        "total_jobs": len(scheduler.get_jobs()) if scheduler.is_available else 0,
    }


@router.get("/scheduler/jobs", response_model=JobListResponse)
async def get_scheduled_jobs():
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
async def get_job(job_id: str):
    """Get a specific job by ID."""
    scheduler = get_scheduler()

    if not scheduler.is_available:
        raise HTTPException(status_code=503, detail="Scheduler not available")

    job = scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse(**job)


@router.post("/scheduler/jobs", response_model=JobActionResponse)
async def add_job(request: AddJobRequest):
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
async def remove_job(job_id: str):
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
async def pause_job(job_id: str):
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
async def resume_job(job_id: str):
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
async def run_job_now(job_id: str):
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
async def start_scheduler():
    """Start the scheduler."""
    scheduler = get_scheduler()

    if not scheduler.is_available:
        raise HTTPException(status_code=503, detail="Scheduler not available")

    if scheduler.start():
        return {"status": "started", "message": "Scheduler started successfully"}

    raise HTTPException(status_code=500, detail="Failed to start scheduler")


@router.post("/scheduler/stop")
async def stop_scheduler():
    """Stop the scheduler."""
    scheduler = get_scheduler()

    if not scheduler.is_available:
        raise HTTPException(status_code=503, detail="Scheduler not available")

    scheduler.shutdown(wait=False)
    return {"status": "stopped", "message": "Scheduler stopped"}

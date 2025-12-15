"""Social Pulse API routes."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from services.social_pulse import SocialPulseService
from db.database import SessionLocal
from db.models import TrendSignalModel
from db.vector_store import get_vector_store
from data_models.embeddings import get_embedding_provider

logger = logging.getLogger(__name__)

router = APIRouter()


class TrendResponse(BaseModel):
    """Response model for a single trend."""

    id: str
    name: str
    description: str
    why_it_matters: str
    strength_score: float
    white_space_score: float
    volume: int
    engagement_score: float
    sentiment_delta: float
    region: Optional[str]
    audience_segment: str
    topics: list[str]
    sample_quotes: list[str]
    first_seen: Optional[str]
    last_updated: Optional[str]


class TrendListResponse(BaseModel):
    """Response model for trend list."""

    trends: list[TrendResponse]
    total: int
    filters: dict


class GenerateResponse(BaseModel):
    """Response model for trend generation."""

    generated: int
    saved: int
    message: str


@router.get("/social-pulse", response_model=TrendListResponse)
async def get_trends(
    limit: int = Query(20, ge=1, le=100, description="Maximum trends to return"),
    region: Optional[str] = Query(None, description="Filter by region"),
    audience: Optional[str] = Query(None, description="Filter by audience segment"),
    min_strength: float = Query(0, ge=0, le=1, description="Minimum strength score"),
):
    """Get Social Pulse trend signals.

    Returns a list of trend signals with filtering options.
    """
    service = SocialPulseService()
    trends = service.get_trends(
        limit=limit,
        region=region,
        audience_segment=audience,
        min_strength=min_strength,
    )

    return TrendListResponse(
        trends=[TrendResponse(**t) for t in trends],
        total=len(trends),
        filters={
            "region": region,
            "audience": audience,
            "min_strength": min_strength,
        },
    )


@router.get("/social-pulse/{trend_id}", response_model=TrendResponse)
async def get_trend(trend_id: str):
    """Get a single trend by ID."""
    db = SessionLocal()
    try:
        trend = db.query(TrendSignalModel).filter(
            TrendSignalModel.id == trend_id
        ).first()

        if not trend:
            raise HTTPException(status_code=404, detail="Trend not found")

        return TrendResponse(
            id=trend.id,
            name=trend.name,
            description=trend.description,
            why_it_matters=trend.why_it_matters,
            strength_score=trend.strength_score,
            white_space_score=trend.white_space_score,
            volume=trend.volume,
            engagement_score=trend.engagement_score,
            sentiment_delta=trend.sentiment_delta,
            region=trend.region,
            audience_segment=trend.audience_segment,
            topics=trend.topics or [],
            sample_quotes=trend.sample_quotes or [],
            first_seen=trend.first_seen.isoformat() if trend.first_seen else None,
            last_updated=trend.last_updated.isoformat() if trend.last_updated else None,
        )
    finally:
        db.close()


@router.get("/social-pulse/search/semantic")
async def search_trends(
    query: str = Query(..., min_length=3, description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results"),
):
    """Semantic search for trends.

    Searches trends using embedding similarity.
    """
    try:
        # Get embedding for query
        provider = get_embedding_provider()
        query_embedding = provider.embed(query)

        # Search in vector store
        vector_store = get_vector_store()

        # Note: This searches content, not trends directly
        # For a full implementation, you'd index trend embeddings too
        results = vector_store.search_similar(
            query_embedding=query_embedding,
            n_results=limit,
        )

        # Get matching content IDs
        content_ids = results.get("ids", [[]])[0]

        # Find trends that include this content
        db = SessionLocal()
        try:
            # Search for trends containing these content IDs
            # This is a simplified approach - production would use a dedicated trend index
            trends = db.query(TrendSignalModel).order_by(
                TrendSignalModel.strength_score.desc()
            ).limit(limit).all()

            return {
                "query": query,
                "results": [
                    {
                        "id": t.id,
                        "name": t.name,
                        "description": t.description,
                        "strength_score": t.strength_score,
                    }
                    for t in trends
                ],
                "total": len(trends),
            }
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/social-pulse/generate", response_model=GenerateResponse)
async def generate_trends(
    days_back: int = Query(30, ge=1, le=90, description="Days of content to analyze"),
    source_types: Optional[str] = Query(None, description="Comma-separated source types"),
    save: bool = Query(True, description="Save trends to database"),
):
    """Generate new trend signals from content.

    Runs the clustering and trend generation pipeline.
    """
    try:
        sources = source_types.split(",") if source_types else None

        service = SocialPulseService(days_back=days_back)
        trends = service.generate_trends(source_types=sources)

        saved_count = 0
        if save and trends:
            saved_count = service.save_trends(trends)

        return GenerateResponse(
            generated=len(trends),
            saved=saved_count,
            message=f"Generated {len(trends)} trends, saved {saved_count}",
        )

    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/social-pulse/regions")
async def get_regions():
    """Get available regions with trend counts."""
    db = SessionLocal()
    try:
        from sqlalchemy import func

        results = db.query(
            TrendSignalModel.region,
            func.count(TrendSignalModel.id).label("count"),
        ).filter(
            TrendSignalModel.region.isnot(None)
        ).group_by(
            TrendSignalModel.region
        ).all()

        return {
            "regions": [
                {"region": r.region, "count": r.count}
                for r in results
            ]
        }
    finally:
        db.close()


@router.get("/social-pulse/audiences")
async def get_audiences():
    """Get available audience segments with trend counts."""
    db = SessionLocal()
    try:
        from sqlalchemy import func

        results = db.query(
            TrendSignalModel.audience_segment,
            func.count(TrendSignalModel.id).label("count"),
        ).group_by(
            TrendSignalModel.audience_segment
        ).all()

        return {
            "audiences": [
                {"segment": r.audience_segment, "count": r.count}
                for r in results
            ]
        }
    finally:
        db.close()

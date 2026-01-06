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
    source_content_ids: list[str] = []
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


# =============================================================================
# STATIC ROUTES (must come before parameterized routes)
# =============================================================================

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


@router.get("/social-pulse/debug")
async def debug_rag():
    """Debug endpoint to check RAG components."""
    from db.models import RawContentModel

    result = {
        "vector_store": {},
        "database": {},
        "embedding": {},
        "test_search": {},
    }

    # Check vector store
    try:
        vector_store = get_vector_store()
        stats = vector_store.get_collection_stats()
        result["vector_store"] = {
            "status": "ok",
            "content_count": stats.get("content_count", 0),
            "trends_count": stats.get("trends_count", 0),
        }
    except Exception as e:
        result["vector_store"] = {"status": "error", "error": str(e)}

    # Check database content
    db = SessionLocal()
    try:
        total = db.query(RawContentModel).count()
        processed = db.query(RawContentModel).filter(RawContentModel.is_processed == True).count()
        result["database"] = {
            "status": "ok",
            "total_content": total,
            "processed_content": processed,
        }
    except Exception as e:
        result["database"] = {"status": "error", "error": str(e)}
    finally:
        db.close()

    # Check embedding function
    try:
        provider = get_embedding_provider()
        test_embedding = provider.embed("test wellness hotel trends")
        result["embedding"] = {
            "status": "ok",
            "provider": type(provider).__name__,
            "dimension": len(test_embedding),
        }
    except Exception as e:
        result["embedding"] = {"status": "error", "error": str(e)}

    # Test vector search
    try:
        if result["embedding"].get("status") == "ok" and result["vector_store"].get("content_count", 0) > 0:
            provider = get_embedding_provider()
            query_embedding = provider.embed("wellness hotel trends")
            vector_store = get_vector_store()
            search_results = vector_store.search_similar(query_embedding, n_results=5)
            result["test_search"] = {
                "status": "ok",
                "results_found": len(search_results.get("ids", [[]])[0]),
                "sample_ids": search_results.get("ids", [[]])[0][:3],
            }
        else:
            result["test_search"] = {"status": "skipped", "reason": "prerequisites not met"}
    except Exception as e:
        result["test_search"] = {"status": "error", "error": str(e)}

    return result


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


@router.post("/social-pulse/regenerate", response_model=GenerateResponse)
async def regenerate_trends(
    days_back: int = Query(30, ge=1, le=90, description="Days of content to analyze"),
    max_trends: int = Query(20, ge=5, le=50, description="Maximum trends to generate"),
):
    """Clear all existing trends and regenerate from scratch with LLM.

    Use this to fix bad trend names/descriptions by regenerating all trends
    using the current LLM pipeline.
    """
    try:
        # Clear all existing trends
        db = SessionLocal()
        try:
            deleted_count = db.query(TrendSignalModel).delete()
            db.commit()
            logger.info(f"Cleared {deleted_count} existing trends")
        finally:
            db.close()

        # Generate new trends with LLM enabled
        service = SocialPulseService(
            days_back=days_back,
            use_llm=True,
            use_adaptive=True,
        )
        trends = service.generate_trends(max_trends=max_trends)

        saved_count = 0
        if trends:
            saved_count = service.save_trends(trends)

        return GenerateResponse(
            generated=len(trends),
            saved=saved_count,
            message=f"Cleared {deleted_count} old trends, generated {len(trends)} new trends with LLM",
        )

    except Exception as e:
        logger.error(f"Regeneration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# PARAMETERIZED ROUTES (must come after static routes)
# =============================================================================

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


@router.get("/social-pulse/{trend_id}/sources")
async def get_trend_sources(trend_id: str, limit: int = Query(20, ge=1, le=100)):
    """Get the source content items that formed this trend.

    Returns list of content items with URLs that can be clicked.
    """
    from db.models import RawContentModel

    db = SessionLocal()
    try:
        # Get the trend
        trend = db.query(TrendSignalModel).filter(
            TrendSignalModel.id == trend_id
        ).first()

        if not trend:
            raise HTTPException(status_code=404, detail="Trend not found")

        # Get source content IDs
        content_ids = trend.source_content_ids or []

        if not content_ids:
            return {"sources": [], "total": 0, "trend_name": trend.name}

        # Fetch the content items
        content_items = db.query(RawContentModel).filter(
            RawContentModel.id.in_(content_ids)
        ).limit(limit).all()

        return {
            "sources": [
                {
                    "id": item.id,
                    "title": item.title,
                    "url": item.url,
                    "source": item.source,
                    "source_type": item.source_type,
                    "author": item.author,
                    "published_at": item.published_at.isoformat() if item.published_at else None,
                    "content_preview": item.content[:300] if item.content else None,
                }
                for item in content_items
            ],
            "total": len(content_items),
            "trend_name": trend.name,
        }

    finally:
        db.close()

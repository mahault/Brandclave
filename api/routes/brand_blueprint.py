"""Brand Blueprint API routes."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from services.brand_blueprint.schemas import (
    BlueprintInputs,
    BrandBlueprintFull,
    BlueprintGenerateRequest,
    BlueprintGenerateResponse,
    BlueprintListResponse,
    StageProgress,
    StageStatus,
    TokenUsage,
)
from services.brand_blueprint.pipeline import BlueprintPipeline
from services.brand_blueprint.repository import BlueprintRepository
from services.chat.llm_client import get_llm_client
from services.chat.rag import BayesianRAG

logger = logging.getLogger(__name__)

router = APIRouter()


def get_pipeline() -> BlueprintPipeline:
    """Get or create the blueprint pipeline."""
    llm_client = get_llm_client()
    if llm_client is None:
        raise HTTPException(
            status_code=503,
            detail="LLM client not available. Check MISTRAL_API_KEY.",
        )

    # Try to get RAG instance
    try:
        rag = BayesianRAG()
    except Exception as e:
        logger.warning(f"RAG not available: {e}")
        rag = None

    return BlueprintPipeline(llm_client, rag)


@router.post("/brand-blueprint/generate", response_model=BlueprintGenerateResponse)
async def generate_blueprint(request: BlueprintGenerateRequest):
    """Generate a complete brand blueprint.

    This endpoint runs the 5-stage pipeline:
    1. Foundation: brand names, one-liner, thesis
    2. Strategic: pillars, positioning, unmet desires
    3. Experience: personas, experiences, guest journey
    4. Atmosphere: design direction, F&B, revenue logic
    5. Summary: investor summary

    The process takes 30-60 seconds depending on LLM response times.
    """
    try:
        pipeline = get_pipeline()
        repository = BlueprintRepository()

        # Track progress
        stages_progress: list[StageProgress] = []

        def progress_callback(progress: StageProgress):
            # Update or append progress
            for i, s in enumerate(stages_progress):
                if s.stage == progress.stage:
                    stages_progress[i] = progress
                    return
            stages_progress.append(progress)

        # Generate blueprint
        blueprint = await pipeline.generate(
            request.inputs,
            progress_callback=progress_callback,
        )

        # Save to database
        blueprint_id = repository.save(blueprint)
        blueprint.id = blueprint_id

        return BlueprintGenerateResponse(
            blueprint_id=blueprint_id,
            status=blueprint.status,
            blueprint=blueprint,
            stages=stages_progress,
            warnings=blueprint.warnings,
            token_usage=blueprint.token_usage,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Blueprint generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Blueprint generation failed: {str(e)}",
        )


@router.get("/brand-blueprint/{blueprint_id}", response_model=BrandBlueprintFull)
async def get_blueprint(blueprint_id: str):
    """Get a saved blueprint by ID."""
    repository = BlueprintRepository()
    blueprint = repository.get(blueprint_id)

    if blueprint is None:
        raise HTTPException(status_code=404, detail="Blueprint not found")

    return blueprint


@router.get("/brand-blueprint", response_model=BlueprintListResponse)
async def list_blueprints(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    location: Optional[str] = Query(None, description="Filter by location"),
    segment: Optional[str] = Query(None, description="Filter by segment"),
):
    """List saved blueprints with optional filters."""
    repository = BlueprintRepository()
    blueprints, total = repository.list(
        limit=limit,
        offset=offset,
        location=location,
        segment=segment,
    )

    return BlueprintListResponse(
        blueprints=blueprints,
        total=total,
        offset=offset,
        limit=limit,
    )


@router.delete("/brand-blueprint/{blueprint_id}")
async def delete_blueprint(blueprint_id: str):
    """Delete a blueprint by ID."""
    repository = BlueprintRepository()

    if not repository.delete(blueprint_id):
        raise HTTPException(status_code=404, detail="Blueprint not found")

    return {"status": "deleted", "blueprint_id": blueprint_id}


# Simple form endpoint for testing
class SimpleGenerateRequest(BaseModel):
    """Simplified request for form submission."""
    location: str
    segment: str
    adr: float
    rooms: int = 100
    developer_goal: str
    source_trend_id: Optional[str] = None


@router.post("/brand-blueprint/generate-simple")
async def generate_blueprint_simple(request: SimpleGenerateRequest):
    """Simplified endpoint for form submission.

    Converts simple form data to full request and generates blueprint.
    """
    full_request = BlueprintGenerateRequest(
        inputs=BlueprintInputs(
            location=request.location,
            segment=request.segment,
            adr=request.adr,
            rooms=request.rooms,
            developer_goal=request.developer_goal,
            source_trend_id=request.source_trend_id,
        )
    )

    return await generate_blueprint(full_request)

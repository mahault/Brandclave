"""City Desires API routes.

Type a city → See what travelers want but can't get.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class CityDesireRequest(BaseModel):
    """Request model for city desire analysis."""

    city: str
    country: Optional[str] = ""


class DesireThemeResponse(BaseModel):
    """Response model for a desire theme."""

    theme_name: str
    description: str
    intensity_score: float
    frustration_score: float
    frequency: int
    category: str
    segments: list[str]
    keywords: list[str]
    example_snippets: list[str]
    opportunity_score: float


class ConceptLaneResponse(BaseModel):
    """Response model for a concept lane recommendation."""

    concept: str
    target_segments: list[str]
    key_features: list[str]
    opportunity_score: float
    rationale: str


class CityDesireResponse(BaseModel):
    """Response model for city desire profile."""

    city: str
    country: str
    total_signals: int
    total_sources: int
    avg_frustration: float
    top_desires: list[dict]
    underserved_segments: list[str]
    white_space_opportunities: list[str]
    concept_lanes: list[dict]
    generated_at: str


@router.post("/city-desires", response_model=CityDesireResponse)
async def analyze_city(request: CityDesireRequest):
    """Analyze what travelers want in a city but can't find.

    This endpoint scrapes Reddit, YouTube, and travel forums to identify:
    - Top desires and unmet needs
    - Underserved traveler segments
    - White space opportunities
    - Recommended hotel concept lanes

    Takes 30-60 seconds to gather and analyze data.
    """
    from services.city_desires import CityDesireEngine

    if not request.city:
        raise HTTPException(status_code=400, detail="City name is required")

    logger.info(f"Analyzing city desires for: {request.city}, {request.country}")

    try:
        with CityDesireEngine() as engine:
            profile = engine.analyze_city(request.city, request.country or "")

        return CityDesireResponse(
            city=profile.city,
            country=profile.country,
            total_signals=profile.total_signals,
            total_sources=profile.total_sources,
            avg_frustration=profile.avg_frustration,
            top_desires=[d.to_dict() for d in profile.top_desires],
            underserved_segments=profile.underserved_segments,
            white_space_opportunities=profile.white_space_opportunities,
            concept_lanes=profile.concept_lanes,
            generated_at=profile.generated_at.isoformat(),
        )

    except Exception as e:
        logger.error(f"City desire analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/city-desires/quick")
async def quick_city_search(
    city: str = Query(..., description="City name to analyze"),
    country: str = Query("", description="Country (optional)"),
):
    """Quick city desire analysis (same as POST but via GET for easy testing)."""
    from services.city_desires import CityDesireEngine

    if not city:
        raise HTTPException(status_code=400, detail="City name is required")

    logger.info(f"Quick city analysis for: {city}, {country}")

    try:
        with CityDesireEngine() as engine:
            profile = engine.analyze_city(city, country)

        return profile.to_dict()

    except Exception as e:
        logger.error(f"City desire analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/city-desires/popular")
async def get_popular_cities():
    """Get list of popular cities for quick analysis."""
    return {
        "popular_cities": [
            {"city": "Lisbon", "country": "Portugal"},
            {"city": "Barcelona", "country": "Spain"},
            {"city": "Paris", "country": "France"},
            {"city": "Tokyo", "country": "Japan"},
            {"city": "Bali", "country": "Indonesia"},
            {"city": "New York", "country": "USA"},
            {"city": "London", "country": "UK"},
            {"city": "Amsterdam", "country": "Netherlands"},
            {"city": "Dubai", "country": "UAE"},
            {"city": "Mexico City", "country": "Mexico"},
            {"city": "Bangkok", "country": "Thailand"},
            {"city": "Miami", "country": "USA"},
        ],
        "hint": "Use POST /api/city-desires or GET /api/city-desires/quick?city=Lisbon to analyze",
    }

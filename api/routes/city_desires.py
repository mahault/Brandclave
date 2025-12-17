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


@router.post("/city-desires/adaptive")
async def analyze_city_adaptive(request: CityDesireRequest):
    """Analyze city using active inference and structure learning.

    This endpoint uses an adaptive approach that:
    - Learns categories from data instead of using fixed ones
    - Actively decides what to search for next
    - Expands/merges categories as evidence accumulates
    - Returns confidence scores and model metrics

    Takes 60-120 seconds due to iterative exploration.
    """
    from services.active_inference.adaptive_city_analyzer import AdaptiveCityAnalyzer

    if not request.city:
        raise HTTPException(status_code=400, detail="City name is required")

    logger.info(f"Adaptive analysis for: {request.city}, {request.country}")

    try:
        with AdaptiveCityAnalyzer(
            alpha=1.0,
            fit_threshold=0.3,
            max_iterations=8,
            confidence_threshold=0.7,
        ) as analyzer:
            result = analyzer.analyze_city(request.city, request.country or "")

        return result

    except Exception as e:
        logger.error(f"Adaptive analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/city-desires/genius")
async def analyze_city_genius(request: CityDesireRequest):
    """Analyze city using VERSES Genius Active Inference API.

    This endpoint uses the VERSES Genius service for proper Bayesian
    active inference with:
    - POMDP-based action selection (Expected Free Energy minimization)
    - Online parameter learning from observations
    - Proper variational free energy computation
    - Bayesian category inference

    Requires GENIUS_API_KEY to be configured.
    Takes 60-120 seconds due to iterative exploration.
    """
    from services.active_inference.genius_city_analyzer import GeniusCityAnalyzer

    if not request.city:
        raise HTTPException(status_code=400, detail="City name is required")

    logger.info(f"Genius analysis for: {request.city}, {request.country}")

    try:
        with GeniusCityAnalyzer(
            max_iterations=8,
            confidence_threshold=0.7,
        ) as analyzer:
            result = analyzer.analyze_city(request.city, request.country or "")

        return result

    except ValueError as e:
        # API key not configured
        logger.error(f"Genius API configuration error: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Genius API not configured: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Genius analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/city-desires/genius/status")
async def get_genius_status():
    """Check VERSES Genius API connection status."""
    from services.active_inference.genius_client import test_genius_connection

    try:
        status = test_genius_connection()
        return {
            "genius_api": status,
            "available": status.get("healthy", False),
        }
    except Exception as e:
        return {
            "genius_api": {"error": str(e)},
            "available": False,
        }


@router.post("/city-desires/pymdp")
async def analyze_city_pymdp(request: CityDesireRequest):
    """Analyze city using PyMDP active inference (JAX-based).

    This endpoint uses the open-source pymdp library for local
    active inference with:
    - Expected Free Energy (EFE) minimization for query selection
    - Variational inference for belief updates
    - Online parameter learning
    - No external API required

    Takes 60-120 seconds due to iterative exploration.
    """
    from services.active_inference.pymdp_city_analyzer import PyMDPCityAnalyzer

    if not request.city:
        raise HTTPException(status_code=400, detail="City name is required")

    logger.info(f"PyMDP analysis for: {request.city}, {request.country}")

    try:
        with PyMDPCityAnalyzer(
            max_iterations=8,
            confidence_threshold=0.7,
            use_learning=True,
        ) as analyzer:
            result = analyzer.analyze_city(request.city, request.country or "")

        return result

    except Exception as e:
        logger.error(f"PyMDP analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/city-desires/pymdp/status")
async def get_pymdp_status():
    """Check PyMDP availability status."""
    from services.active_inference.pymdp_learner import PYMDP_AVAILABLE

    return {
        "pymdp_available": PYMDP_AVAILABLE,
        "description": "JAX-based active inference library",
        "install_command": "pip install inferactively-pymdp jax jaxlib" if not PYMDP_AVAILABLE else None,
    }


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
        "endpoints": {
            "standard": "POST /api/city-desires - Fixed category analysis",
            "adaptive": "POST /api/city-desires/adaptive - Local structure learning",
            "genius": "POST /api/city-desires/genius - VERSES Genius active inference",
            "pymdp": "POST /api/city-desires/pymdp - PyMDP JAX-based active inference",
        }
    }

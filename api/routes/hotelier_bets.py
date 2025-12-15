"""Hotelier Bets API routes."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from services.hotelier_bets import HotelierBetsService
from data_models.hotelier_move import MoveType

logger = logging.getLogger(__name__)

router = APIRouter()


class MoveResponse(BaseModel):
    """Response model for a single hotelier move."""

    id: str
    title: str
    summary: str
    why_it_matters: str
    company: str
    company_type: Optional[str]
    move_type: str
    market: Optional[str]
    investment_amount: Optional[str]
    strategic_implications: list[str]
    competitive_impact: Optional[str]
    source_url: str
    source_name: str
    published_at: Optional[str]
    extracted_at: Optional[str]
    confidence_score: float


class MoveListResponse(BaseModel):
    """Response model for move list."""

    moves: list[MoveResponse]
    total: int
    filters: dict


class GenerateResponse(BaseModel):
    """Response model for move extraction."""

    extracted: int
    saved: int
    message: str


@router.get("/hotelier-bets", response_model=MoveListResponse)
async def get_moves(
    limit: int = Query(20, ge=1, le=100, description="Maximum moves to return"),
    company: Optional[str] = Query(None, description="Filter by company name"),
    move_type: Optional[str] = Query(None, description="Filter by move type"),
    market: Optional[str] = Query(None, description="Filter by market"),
    min_confidence: float = Query(0, ge=0, le=1, description="Minimum confidence score"),
):
    """Get Hotelier Bets strategic moves.

    Returns a list of extracted strategic moves with filtering options.
    """
    service = HotelierBetsService()
    moves = service.get_moves(
        limit=limit,
        company=company,
        move_type=move_type,
        market=market,
        min_confidence=min_confidence,
    )

    return MoveListResponse(
        moves=[MoveResponse(**m) for m in moves],
        total=len(moves),
        filters={
            "company": company,
            "move_type": move_type,
            "market": market,
            "min_confidence": min_confidence,
        },
    )


@router.get("/hotelier-bets/companies")
async def get_companies():
    """Get list of companies with extracted moves."""
    service = HotelierBetsService()
    companies = service.get_companies()

    return {
        "companies": companies,
        "total": len(companies),
    }


@router.get("/hotelier-bets/move-types")
async def get_move_types():
    """Get list of move types with descriptions."""
    # Return all possible move types with descriptions
    all_types = [
        {"type": mt.value, "label": mt.name.replace("_", " ").title()}
        for mt in MoveType
    ]

    # Also get types currently in use
    service = HotelierBetsService()
    in_use = service.get_move_types()

    return {
        "move_types": all_types,
        "in_use": in_use,
    }


@router.get("/hotelier-bets/markets")
async def get_markets():
    """Get list of markets with extracted moves."""
    service = HotelierBetsService()
    markets = service.get_markets()

    return {
        "markets": markets,
        "total": len(markets),
    }


@router.get("/hotelier-bets/{move_id}", response_model=MoveResponse)
async def get_move(move_id: str):
    """Get a single move by ID."""
    service = HotelierBetsService()
    move = service.get_move_by_id(move_id)

    if not move:
        raise HTTPException(status_code=404, detail="Move not found")

    return MoveResponse(**move)


@router.post("/hotelier-bets/generate", response_model=GenerateResponse)
async def generate_moves(
    days_back: int = Query(30, ge=1, le=90, description="Days of content to analyze"),
    limit: int = Query(100, ge=1, le=500, description="Maximum articles to process"),
    save: bool = Query(True, description="Save moves to database"),
):
    """Extract strategic moves from news content.

    Runs the LLM-powered move extraction pipeline on news articles.
    """
    try:
        service = HotelierBetsService()
        moves = service.extract_moves(days_back=days_back, limit=limit)

        saved_count = 0
        if save and moves:
            saved_count = service.save_moves(moves)

        return GenerateResponse(
            extracted=len(moves),
            saved=saved_count,
            message=f"Extracted {len(moves)} moves, saved {saved_count}",
        )

    except Exception as e:
        logger.error(f"Extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

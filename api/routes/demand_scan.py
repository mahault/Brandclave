"""Demand Scan API routes."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, HttpUrl

from services.demand_scan import DemandScanService
from data_models.property_features import PropertyType, PriceSegment

logger = logging.getLogger(__name__)

router = APIRouter()


class ScanRequest(BaseModel):
    """Request model for property scan."""

    url: str


class PropertyResponse(BaseModel):
    """Response model for property features."""

    id: Optional[str]
    url: str
    name: Optional[str]
    property_type: str
    brand_positioning: Optional[str]
    tagline: Optional[str]
    tone: Optional[str]
    themes: list[str]
    amenities: list[str]
    room_types: list[str]
    dining_options: list[str]
    experiences: list[str]
    location: Optional[str]
    region: Optional[str]
    price_segment: str
    price_indicators: list[str]
    demand_fit_score: Optional[float]
    experience_gaps: list[str]
    opportunity_lanes: list[str]
    competitive_advantages: list[str]
    recommendations: list[str]
    matching_trend_ids: list[str]
    scraped_at: Optional[str]


class PropertyListResponse(BaseModel):
    """Response model for property list."""

    properties: list[PropertyResponse]
    total: int
    filters: dict


class ScanStatusResponse(BaseModel):
    """Response model for scan status."""

    status: str
    message: str
    property: Optional[PropertyResponse]


@router.post("/demand-scan", response_model=ScanStatusResponse)
async def scan_property(request: ScanRequest):
    """Scan a property URL and analyze against demand trends.

    Scrapes the property website, extracts features, and compares
    against current demand trends to identify gaps and opportunities.
    """
    try:
        service = DemandScanService()

        # Check if already scanned
        existing = service.get_property_by_url(request.url)
        if existing:
            return ScanStatusResponse(
                status="exists",
                message="Property already scanned. Use refresh=true to rescan.",
                property=PropertyResponse(**existing),
            )

        # Scan property
        result = service.scan_property(request.url)

        if not result:
            raise HTTPException(
                status_code=400,
                detail="Failed to scan property. Check if the URL is valid and accessible.",
            )

        # Save to database
        property_id = service.save_property(result)
        result["id"] = property_id

        return ScanStatusResponse(
            status="success",
            message="Property scanned and analyzed successfully",
            property=PropertyResponse(**result),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Scan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/demand-scan/refresh", response_model=ScanStatusResponse)
async def rescan_property(request: ScanRequest):
    """Rescan a property URL with fresh data.

    Forces a new scan even if the property was previously analyzed.
    """
    try:
        service = DemandScanService()
        result = service.scan_property(request.url)

        if not result:
            raise HTTPException(
                status_code=400,
                detail="Failed to scan property. Check if the URL is valid and accessible.",
            )

        # Save (will update if exists)
        property_id = service.save_property(result)
        result["id"] = property_id

        return ScanStatusResponse(
            status="success",
            message="Property rescanned and updated",
            property=PropertyResponse(**result),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rescan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/demand-scan", response_model=PropertyListResponse)
async def get_properties(
    limit: int = Query(20, ge=1, le=100, description="Maximum properties to return"),
    region: Optional[str] = Query(None, description="Filter by region"),
    property_type: Optional[str] = Query(None, description="Filter by property type"),
    min_demand_fit: float = Query(0, ge=0, le=1, description="Minimum demand fit score"),
):
    """Get list of scanned properties.

    Returns properties with optional filtering.
    """
    service = DemandScanService()
    properties = service.get_properties(
        limit=limit,
        region=region,
        property_type=property_type,
        min_demand_fit=min_demand_fit,
    )

    return PropertyListResponse(
        properties=[PropertyResponse(**p) for p in properties],
        total=len(properties),
        filters={
            "region": region,
            "property_type": property_type,
            "min_demand_fit": min_demand_fit,
        },
    )


@router.get("/demand-scan/property-types")
async def get_property_types():
    """Get list of property types."""
    return {
        "property_types": [
            {"type": pt.value, "label": pt.name.replace("_", " ").title()}
            for pt in PropertyType
        ]
    }


@router.get("/demand-scan/price-segments")
async def get_price_segments():
    """Get list of price segments."""
    return {
        "price_segments": [
            {"segment": ps.value, "label": ps.name.replace("_", " ").title()}
            for ps in PriceSegment
        ]
    }


@router.get("/demand-scan/{property_id}", response_model=PropertyResponse)
async def get_property(property_id: str):
    """Get a single property by ID."""
    service = DemandScanService()
    prop = service.get_property(property_id)

    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")

    return PropertyResponse(**prop)

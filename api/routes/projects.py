"""Projects API routes: per-user saved research items (trends and hotelier moves).

All endpoints require authentication; each user only ever sees their own
saved items. A snapshot of the item is stored at save time so the saved copy
survives re-clustering or deletion of the underlying signal.
"""

import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import SavedItemModel, UserModel
from services.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Schemas ---


class SavedItemCreate(BaseModel):
    """Payload for saving a research item."""

    item_type: Literal["trend", "move"]
    item_id: str = Field(..., min_length=1, max_length=36)
    title: str | None = Field(None, max_length=300)
    snapshot: dict | None = None


class SavedItemResponse(BaseModel):
    """A saved research item."""

    id: str
    item_type: str
    item_id: str
    title: str | None = None
    snapshot: dict | None = None
    created_at: str | None = None


class SavedItemListResponse(BaseModel):
    """List of saved items for the authenticated user."""

    items: list[SavedItemResponse]
    total: int


def _to_response(model: SavedItemModel) -> SavedItemResponse:
    return SavedItemResponse(
        id=model.id,
        item_type=model.item_type,
        item_id=model.item_id,
        title=model.title,
        snapshot=model.snapshot_json,
        created_at=model.created_at.isoformat() if model.created_at else None,
    )


# --- Routes ---


@router.get("/projects/saved", response_model=SavedItemListResponse)
async def list_saved_items(
    item_type: Literal["trend", "move"] | None = Query(None, description="Filter by item type"),
    user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List the authenticated user's saved items, newest first."""
    query = db.query(SavedItemModel).filter(SavedItemModel.user_id == user.id)
    if item_type:
        query = query.filter(SavedItemModel.item_type == item_type)

    models = query.order_by(SavedItemModel.created_at.desc()).all()
    return SavedItemListResponse(items=[_to_response(m) for m in models], total=len(models))


@router.post("/projects/saved", response_model=SavedItemResponse, status_code=201)
async def save_item(
    request: SavedItemCreate,
    user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Save a research item (trend or move) to the user's workspace."""
    existing = (
        db.query(SavedItemModel)
        .filter(
            SavedItemModel.user_id == user.id,
            SavedItemModel.item_type == request.item_type,
            SavedItemModel.item_id == request.item_id,
        )
        .first()
    )
    if existing is not None:
        raise HTTPException(status_code=409, detail="Item is already saved")

    model = SavedItemModel(
        user_id=user.id,
        item_type=request.item_type,
        item_id=request.item_id,
        title=request.title,
        snapshot_json=request.snapshot or {},
    )
    db.add(model)
    db.commit()
    db.refresh(model)

    logger.info(f"User {user.id} saved {request.item_type} {request.item_id}")
    return _to_response(model)


@router.delete("/projects/saved/{saved_item_id}")
async def delete_saved_item(
    saved_item_id: str,
    user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Remove a saved item. Only the owner can delete it."""
    model = (
        db.query(SavedItemModel)
        .filter(
            SavedItemModel.id == saved_item_id,
            SavedItemModel.user_id == user.id,
        )
        .first()
    )
    if model is None:
        raise HTTPException(status_code=404, detail="Saved item not found")

    db.delete(model)
    db.commit()
    return {"status": "deleted", "saved_item_id": saved_item_id}

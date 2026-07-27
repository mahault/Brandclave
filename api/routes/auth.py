"""Authentication API routes: register, login, and current-user profile."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import UserModel
from services.auth import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Schemas ---


class RegisterRequest(BaseModel):
    """Payload for creating a new account."""

    email: str = Field(..., min_length=3, max_length=320)
    password: str = Field(..., min_length=8, max_length=200)
    display_name: str | None = Field(None, max_length=200)

    @field_validator("email")
    @classmethod
    def _normalize_email(cls, value: str) -> str:
        """Lowercase-normalize and sanity-check the email address."""
        email = value.strip().lower()
        if "@" not in email or email.startswith("@") or email.endswith("@"):
            raise ValueError("Invalid email address")
        return email


class LoginRequest(BaseModel):
    """Payload for logging in."""

    email: str
    password: str

    @field_validator("email")
    @classmethod
    def _normalize_email(cls, value: str) -> str:
        return value.strip().lower()


class UserProfile(BaseModel):
    """Public view of a user account."""

    id: str
    email: str
    display_name: str | None = None
    created_at: str | None = None
    is_active: bool = True


class AuthResponse(BaseModel):
    """Token + profile returned on register/login."""

    token: str
    token_type: str = "bearer"
    user: UserProfile


def _to_profile(user: UserModel) -> UserProfile:
    return UserProfile(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        created_at=user.created_at.isoformat() if user.created_at else None,
        is_active=user.is_active,
    )


# --- Routes ---


@router.post("/auth/register", response_model=AuthResponse, status_code=201)
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """Create a new account and return an access token."""
    existing = db.query(UserModel).filter(UserModel.email == request.email).first()
    if existing is not None:
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    user = UserModel(
        email=request.email,
        password_hash=hash_password(request.password),
        display_name=request.display_name,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    logger.info(f"Registered user {user.id}")
    return AuthResponse(token=create_access_token(user.id), user=_to_profile(user))


@router.post("/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Exchange email + password for an access token."""
    user = db.query(UserModel).filter(UserModel.email == request.email).first()
    if user is None or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not user.is_active:
        raise HTTPException(status_code=401, detail="Account is disabled")

    return AuthResponse(token=create_access_token(user.id), user=_to_profile(user))


@router.get("/auth/me", response_model=UserProfile)
async def me(user: UserModel = Depends(get_current_user)):
    """Return the authenticated user's profile."""
    return _to_profile(user)

"""Authentication service: password hashing, JWT tokens, and FastAPI user dependencies.

Password hashing uses stdlib PBKDF2-HMAC-SHA256 with a per-user random salt
and constant-time comparison — no external hashing dependency. Access tokens
are HS256 JWTs signed with ``Settings.jwt_secret`` (env ``JWT_SECRET``); when
that is unset an ephemeral secret is generated at boot, which means every
issued token is invalidated on restart.
"""

import hashlib
import hmac
import logging
import secrets
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from config.settings import get_settings
from db.database import get_db
from db.models import UserModel

logger = logging.getLogger(__name__)

# --- Password hashing (stdlib PBKDF2) ---

PBKDF2_ALGORITHM = "sha256"
PBKDF2_ITERATIONS = 260_000
_SALT_BYTES = 16

JWT_ALGORITHM = "HS256"

# Ephemeral fallback secret, generated lazily when JWT_SECRET is not configured
_ephemeral_secret: str | None = None

# Extracts the Bearer token from the Authorization header without auto-raising,
# so get_optional_user can degrade gracefully for anonymous requests.
_bearer_scheme = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    """Hash a password with PBKDF2-HMAC-SHA256 and a per-user random salt.

    Returns:
        Encoded string ``pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>``.
    """
    salt = secrets.token_bytes(_SALT_BYTES)
    derived = hashlib.pbkdf2_hmac(
        PBKDF2_ALGORITHM, password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return f"pbkdf2_{PBKDF2_ALGORITHM}${PBKDF2_ITERATIONS}${salt.hex()}${derived.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against a stored hash using constant-time comparison."""
    try:
        scheme, iterations_str, salt_hex, hash_hex = password_hash.split("$")
        if scheme != f"pbkdf2_{PBKDF2_ALGORITHM}":
            return False
        iterations = int(iterations_str)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except (ValueError, AttributeError):
        return False

    derived = hashlib.pbkdf2_hmac(
        PBKDF2_ALGORITHM, password.encode("utf-8"), salt, iterations
    )
    return hmac.compare_digest(derived, expected)


# --- JWT access tokens ---


def get_jwt_secret() -> str:
    """Return the JWT signing secret, generating an ephemeral one if unconfigured."""
    settings = get_settings()
    if settings.jwt_secret and settings.jwt_secret.strip():
        return settings.jwt_secret

    global _ephemeral_secret
    if _ephemeral_secret is None:
        _ephemeral_secret = secrets.token_urlsafe(48)
        logger.warning(
            "JWT_SECRET is not set - generated an ephemeral signing secret. "
            "Sessions will NOT survive a restart; set JWT_SECRET in .env for "
            "persistent sessions."
        )
    return _ephemeral_secret


def create_access_token(user_id: str, expires_hours: int = 72) -> str:
    """Create a signed JWT access token for a user.

    Args:
        user_id: The user's ID, stored in the ``sub`` claim.
        expires_hours: Token lifetime in hours (default 72).

    Returns:
        The encoded JWT string.
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + timedelta(hours=expires_hours),
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> str | None:
    """Decode a JWT access token and return the user ID, or None if invalid/expired."""
    try:
        payload = jwt.decode(token, get_jwt_secret(), algorithms=[JWT_ALGORITHM])
    except jwt.InvalidTokenError:
        return None
    user_id = payload.get("sub")
    return user_id if isinstance(user_id, str) else None


# --- FastAPI dependencies ---


def _resolve_user(token: str, db: Session) -> UserModel | None:
    """Resolve a bearer token to an active user, or None."""
    user_id = decode_access_token(token)
    if user_id is None:
        return None
    user = db.query(UserModel).filter(UserModel.id == user_id).first()
    if user is None or not user.is_active:
        return None
    return user


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> UserModel:
    """FastAPI dependency: return the authenticated user or raise 401."""
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = _resolve_user(credentials.credentials, db)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> UserModel | None:
    """FastAPI dependency: return the authenticated user, or None if missing/invalid."""
    if credentials is None:
        return None
    return _resolve_user(credentials.credentials, db)

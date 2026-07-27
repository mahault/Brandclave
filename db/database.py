"""Database connection and session management.

Works against local SQLite (default) or managed Postgres: set DATABASE_URL
to the connection string a provider hands out (postgres://, postgresql:// and
postgresql+psycopg:// all accepted). Schema changes are managed with Alembic
(`python -m alembic upgrade head`); `init_db()` remains for local bootstrap.
"""

from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Populate os.environ from .env for modules that still call os.getenv directly
load_dotenv()

from config.settings import get_settings


def normalize_database_url(url: str) -> str:
    """Route Postgres URLs through the psycopg3 driver.

    Managed providers (Neon, Render, Supabase, Railway) hand out postgres://
    or postgresql:// strings; SQLAlchemy 2 + psycopg3 wants postgresql+psycopg://.
    """
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


def _engine_kwargs(url: str) -> dict:
    """Engine options per backend: SQLite needs the thread flag; Postgres
    gets connection health checks and an explicit pool for concurrent users."""
    if url.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False}}
    return {
        "pool_pre_ping": True,
        "pool_size": 5,
        "max_overflow": 10,
        "pool_recycle": 300,
    }


# Database URL from typed settings (env DATABASE_URL, default local SQLite)
DATABASE_URL = normalize_database_url(get_settings().database_url)

# Ensure data directory exists for the SQLite default
if DATABASE_URL.startswith("sqlite"):
    Path("./data").mkdir(exist_ok=True)

engine = create_engine(DATABASE_URL, echo=False, **_engine_kwargs(DATABASE_URL))

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    """Get database session. Use as FastAPI dependency (generator)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session() -> Session:
    """Get database session directly (not a generator).

    Use this when you need a session outside of FastAPI dependency injection.
    Caller is responsible for closing the session or using it as context manager.
    """
    return SessionLocal()


def init_db():
    """Create all tables directly (local bootstrap; deploys use Alembic)."""
    from db.models import Base

    Base.metadata.create_all(bind=engine)
    print(f"Database initialized at: {DATABASE_URL}")

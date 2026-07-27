"""Database connection and session management."""

from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Populate os.environ from .env for modules that still call os.getenv directly
load_dotenv()

from config.settings import get_settings

# Database URL from typed settings (env DATABASE_URL, default local SQLite)
DATABASE_URL = get_settings().database_url

# Ensure data directory exists
data_dir = Path("./data")
data_dir.mkdir(exist_ok=True)

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False,
)

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
    """Initialize database tables."""
    from db.models import Base

    Base.metadata.create_all(bind=engine)
    print(f"Database initialized at: {DATABASE_URL}")

"""Database connection and session management."""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()

# Database URL from environment or default
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/brandclave.db")

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
    """Get database session. Use as context manager or dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    from db.models import Base

    Base.metadata.create_all(bind=engine)
    print(f"Database initialized at: {DATABASE_URL}")

#!/usr/bin/env python
"""Initialize the BrandClave database and vector store."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv()


def main():
    """Initialize database and vector store."""
    print("Initializing BrandClave database...")

    # Ensure data directory exists
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    print(f"Data directory: {data_dir}")

    # Initialize SQLite database
    print("\nInitializing SQLite database...")
    from db.database import init_db

    init_db()

    # Initialize ChromaDB vector store
    print("\nInitializing ChromaDB vector store...")
    from db.vector_store import init_vector_store

    init_vector_store()

    print("\n✓ Database initialization complete!")
    print("\nNext steps:")
    print("  1. Copy .env.example to .env and configure your settings")
    print("  2. Run scrapers: python scripts/run_crawlers.py --source hospitalitynet")
    print("  3. Process content: python scripts/run_crawlers.py --process")


if __name__ == "__main__":
    main()

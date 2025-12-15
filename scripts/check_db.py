#!/usr/bin/env python
"""Quick script to verify database contents."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from db.database import SessionLocal
from db.models import RawContentModel
from db.vector_store import get_vector_store

db = SessionLocal()

# Check raw content
count = db.query(RawContentModel).count()
processed = db.query(RawContentModel).filter(RawContentModel.is_processed == True).count()
print(f"Raw content: {count} total, {processed} processed")

# Sample one item
item = db.query(RawContentModel).first()
if item:
    print(f"\nSample item:")
    print(f"  Title: {item.title[:60] if item.title else 'N/A'}...")
    print(f"  Source: {item.source}")
    print(f"  Language: {item.language}")
    print(f"  Sentiment: {item.sentiment_score:.3f}" if item.sentiment_score else "  Sentiment: N/A")
    print(f"  Has embedding: {item.embedding_id is not None}")

# Check vector store
vs = get_vector_store()
stats = vs.get_collection_stats()
print(f"\nVector store: {stats}")

db.close()

#!/usr/bin/env python
"""Rebuild the content vector store under a different embedding provider.

A Chroma collection fixes its dimension on the first insert, so a store built
with Mistral (1024-wide) rejects every vector from the sentence-transformers
fallback (384-wide) and vice versa. Switching providers therefore means
re-embedding the corpus, which is what this script does.

Usage:
    python scripts/rebuild_vector_store.py --provider local
    python scripts/rebuild_vector_store.py --provider mistral --yes

The old collection is dropped and every processed row in raw_content is
re-embedded, so this costs one API call per batch when the target is Mistral.
Nothing in the relational database is deleted: embedding_id values are rewritten
in place as each item is re-added.
"""

import argparse
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv()

from data_models.embeddings import get_embedding_provider
from db.database import SessionLocal
from db.models import RawContent as RawContentModel
from db.vector_store import get_vector_store

logger = logging.getLogger("rebuild_vector_store")

BATCH_SIZE = 64


def rebuild(provider_name: str, batch_size: int = BATCH_SIZE) -> dict:
    """Drop and repopulate the content collection using `provider_name`."""
    provider = get_embedding_provider(provider_name)
    store = get_vector_store()

    existing = store.content_dimension()
    logger.info(
        "Rebuilding vector store: %s -> %s (%d-wide)",
        f"{existing}-wide" if existing else "empty",
        provider_name,
        provider.dimension,
    )

    # Drop the collection so its dimension is re-derived from the new provider.
    store.client.delete_collection("raw_content")
    store.content_collection = store.client.get_or_create_collection(
        name="raw_content",
        metadata={"description": "Embeddings for scraped content"},
    )

    db = SessionLocal()
    embedded = skipped = 0
    try:
        rows = db.query(RawContentModel).filter(RawContentModel.is_processed.is_(True)).all()
        logger.info("Re-embedding %d processed items", len(rows))

        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            texts, keep = [], []
            for row in batch:
                text = (row.content or "").strip()
                if not text:
                    skipped += 1
                    continue
                texts.append(text)
                keep.append(row)

            if not texts:
                continue

            vectors = provider.embed_batch(texts)
            store.add_content_embeddings_batch(
                ids=[row.id for row in keep],
                embeddings=vectors,
                texts=texts,
                metadatas=[
                    {"source": row.source or "", "source_type": str(row.source_type or "")}
                    for row in keep
                ],
            )
            for row in keep:
                row.embedding_id = row.id
            db.commit()
            embedded += len(keep)
            logger.info("  %d/%d", embedded, len(rows))
    finally:
        db.close()

    result = {"provider": provider_name, "embedded": embedded, "skipped_empty": skipped}
    logger.info("Rebuild complete: %s", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        required=True,
        choices=["mistral", "local"],
        help="Embedding provider to rebuild the store with",
    )
    parser.add_argument(
        "--yes", action="store_true", help="Skip the confirmation prompt (for CI/scripts)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if not args.yes:
        print(
            "This drops the 'raw_content' collection and re-embeds the corpus "
            f"with the '{args.provider}' provider."
        )
        if input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Aborted.")
            return 1

    rebuild(args.provider)
    print("Set EMBEDDING_PROVIDER=%s in .env so the app matches the store." % args.provider)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

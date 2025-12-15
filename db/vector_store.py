"""ChromaDB vector store wrapper for embeddings."""

import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")


class VectorStore:
    """ChromaDB wrapper for managing embeddings."""

    def __init__(self, persist_dir: str | None = None):
        """Initialize ChromaDB client.

        Args:
            persist_dir: Directory for persistent storage. Defaults to env var or ./data/chroma
        """
        self.persist_dir = persist_dir or CHROMA_PERSIST_DIR
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # Main collection for content embeddings
        self.content_collection = self.client.get_or_create_collection(
            name="raw_content",
            metadata={"description": "Embeddings for scraped content"},
        )

        # Collection for trend clusters
        self.trends_collection = self.client.get_or_create_collection(
            name="trend_clusters",
            metadata={"description": "Trend cluster centroids"},
        )

    def add_content_embedding(
        self,
        id: str,
        embedding: list[float],
        text: str,
        metadata: dict | None = None,
    ) -> str:
        """Add content embedding to the store.

        Args:
            id: Unique identifier (usually RawContent.id)
            embedding: Vector embedding
            text: Original text content
            metadata: Additional metadata (source, source_type, etc.)

        Returns:
            The embedding ID
        """
        self.content_collection.add(
            ids=[id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata or {}],
        )
        return id

    def add_content_embeddings_batch(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict] | None = None,
    ) -> list[str]:
        """Add multiple content embeddings in batch.

        Args:
            ids: List of unique identifiers
            embeddings: List of vector embeddings
            texts: List of original text contents
            metadatas: List of metadata dicts

        Returns:
            List of embedding IDs
        """
        self.content_collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas or [{} for _ in ids],
        )
        return ids

    def search_similar(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict | None = None,
    ) -> dict:
        """Search for similar content by embedding.

        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            where: Optional filter conditions

        Returns:
            ChromaDB query results with ids, distances, documents, metadatas
        """
        return self.content_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
        )

    def search_by_text(
        self,
        query_text: str,
        embedding_fn: callable,
        n_results: int = 10,
        where: dict | None = None,
    ) -> dict:
        """Search by text using an embedding function.

        Args:
            query_text: Text to search for
            embedding_fn: Function to convert text to embedding
            n_results: Number of results
            where: Optional filters

        Returns:
            ChromaDB query results
        """
        query_embedding = embedding_fn(query_text)
        return self.search_similar(query_embedding, n_results, where)

    def get_embedding(self, id: str) -> dict | None:
        """Get a specific embedding by ID.

        Args:
            id: Embedding ID

        Returns:
            Embedding data or None if not found
        """
        result = self.content_collection.get(ids=[id], include=["embeddings", "documents", "metadatas"])
        if result["ids"]:
            embeddings = result.get("embeddings")
            documents = result.get("documents")
            metadatas = result.get("metadatas")
            return {
                "id": result["ids"][0],
                "embedding": embeddings[0] if embeddings is not None and len(embeddings) > 0 else None,
                "document": documents[0] if documents is not None and len(documents) > 0 else None,
                "metadata": metadatas[0] if metadatas is not None and len(metadatas) > 0 else None,
            }
        return None

    def delete_embedding(self, id: str) -> None:
        """Delete an embedding by ID."""
        self.content_collection.delete(ids=[id])

    def get_collection_stats(self) -> dict:
        """Get statistics about the collections."""
        return {
            "content_count": self.content_collection.count(),
            "trends_count": self.trends_collection.count(),
        }


# Singleton instance
_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Get the singleton VectorStore instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def init_vector_store() -> VectorStore:
    """Initialize and return the vector store."""
    store = get_vector_store()
    print(f"Vector store initialized at: {store.persist_dir}")
    print(f"Collections: {store.get_collection_stats()}")
    return store

"""Bayesian RAG - Retrieval with uncertainty-aware fusion."""

import logging
import math
from dataclasses import dataclass
from typing import Any

from db.vector_store import get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved chunk with scoring."""
    id: str
    text: str
    metadata: dict
    vector_score: float      # Similarity from vector search
    keyword_score: float     # BM25/keyword match score
    metadata_score: float    # Location/segment tag match
    posterior: float         # P(relevant | scores)
    source_type: str


@dataclass
class RAGResult:
    """Result from RAG retrieval."""
    chunks: list[RetrievedChunk]
    top_posterior: float     # Highest posterior
    entropy: float           # Uncertainty in retrieval
    sources_used: int


class BayesianRAG:
    """Retrieval-Augmented Generation with Bayesian fusion scoring.

    Combines:
    - Dense retrieval (vector similarity)
    - Sparse retrieval (keyword matching)
    - Metadata matching (location, segment tags)

    Uses logistic regression-style fusion to compute posterior relevance.
    """

    # Fusion weights (can be tuned from feedback)
    ALPHA = -1.0        # Prior (negative = skeptical by default)
    BETA_VECTOR = 3.0   # Weight for vector similarity
    BETA_KEYWORD = 2.0  # Weight for keyword score
    BETA_META = 1.5     # Weight for metadata match

    # Thresholds
    POSTERIOR_THRESHOLD = 0.4   # Min posterior to include
    TOP_K_VECTOR = 20           # Candidates from vector search
    TOP_K_FINAL = 10            # Final chunks to return

    def __init__(self, embedding_fn: callable = None):
        """Initialize RAG.

        Args:
            embedding_fn: Function to convert text to embeddings
        """
        self.vector_store = get_vector_store()
        self.embedding_fn = embedding_fn

    async def retrieve(
        self,
        query: str,
        location: str | None = None,
        segment: str | None = None,
        source_types: list[str] | None = None,
        top_k: int | None = None,
    ) -> RAGResult:
        """Retrieve relevant chunks with Bayesian scoring.

        Args:
            query: Search query
            location: Optional location filter/boost
            segment: Optional segment filter/boost
            source_types: Optional source type filter
            top_k: Max results to return

        Returns:
            RAGResult with scored chunks and uncertainty metrics
        """
        top_k = top_k or self.TOP_K_FINAL

        # Step 1: Dense retrieval (vector search)
        vector_candidates = await self._vector_search(query, source_types)

        if not vector_candidates:
            logger.warning("No vector candidates found")
            return RAGResult(chunks=[], top_posterior=0.0, entropy=1.0, sources_used=0)

        # Step 2: Compute keyword scores
        query_terms = set(query.lower().split())
        for chunk in vector_candidates:
            chunk.keyword_score = self._compute_keyword_score(chunk.text, query_terms)

        # Step 3: Compute metadata match scores
        for chunk in vector_candidates:
            chunk.metadata_score = self._compute_metadata_score(
                chunk.metadata, location, segment
            )

        # Step 4: Bayesian fusion - compute posteriors
        for chunk in vector_candidates:
            chunk.posterior = self._compute_posterior(
                chunk.vector_score,
                chunk.keyword_score,
                chunk.metadata_score,
            )

        # Step 5: Filter and sort by posterior
        relevant_chunks = [
            c for c in vector_candidates
            if c.posterior >= self.POSTERIOR_THRESHOLD
        ]
        relevant_chunks.sort(key=lambda c: c.posterior, reverse=True)
        final_chunks = relevant_chunks[:top_k]

        # Compute entropy (uncertainty)
        entropy = self._compute_entropy([c.posterior for c in final_chunks])

        # Top posterior
        top_posterior = final_chunks[0].posterior if final_chunks else 0.0

        return RAGResult(
            chunks=final_chunks,
            top_posterior=top_posterior,
            entropy=entropy,
            sources_used=len(final_chunks),
        )

    async def _vector_search(
        self,
        query: str,
        source_types: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Perform vector similarity search.

        Args:
            query: Search query
            source_types: Optional filter

        Returns:
            List of RetrievedChunk with vector scores
        """
        if not self.embedding_fn:
            logger.error("No embedding function provided")
            return []

        try:
            # Get query embedding
            query_embedding = self.embedding_fn(query)

            # Build filter
            where = None
            if source_types:
                where = {"source_type": {"$in": source_types}}

            # Search
            results = self.vector_store.search_similar(
                query_embedding=query_embedding,
                n_results=self.TOP_K_VECTOR,
                where=where,
            )

            # Convert to RetrievedChunk
            chunks = []
            ids = results.get("ids", [[]])[0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for i, doc_id in enumerate(ids):
                # Convert distance to similarity (assuming L2 distance)
                # Smaller distance = more similar
                distance = distances[i] if i < len(distances) else 1.0
                similarity = 1.0 / (1.0 + distance)  # Convert to 0-1 range

                chunks.append(RetrievedChunk(
                    id=doc_id,
                    text=documents[i] if i < len(documents) else "",
                    metadata=metadatas[i] if i < len(metadatas) else {},
                    vector_score=similarity,
                    keyword_score=0.0,  # Computed later
                    metadata_score=0.0,  # Computed later
                    posterior=0.0,       # Computed later
                    source_type=metadatas[i].get("source_type", "unknown") if i < len(metadatas) else "unknown",
                ))

            return chunks

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def _compute_keyword_score(self, text: str, query_terms: set[str]) -> float:
        """Compute keyword/BM25-style score.

        Simple term frequency based scoring.

        Args:
            text: Document text
            query_terms: Set of query terms

        Returns:
            Score 0-1
        """
        if not text or not query_terms:
            return 0.0

        text_lower = text.lower()
        text_terms = set(text_lower.split())

        # Term overlap
        overlap = len(query_terms & text_terms)

        # Also check for substring matches (important for phrases)
        substring_matches = sum(1 for term in query_terms if term in text_lower)

        # Combine
        total_matches = overlap + substring_matches * 0.5
        max_possible = len(query_terms) * 1.5

        return min(total_matches / max_possible, 1.0) if max_possible > 0 else 0.0

    def _compute_metadata_score(
        self,
        metadata: dict,
        location: str | None,
        segment: str | None,
    ) -> float:
        """Compute metadata match score.

        Args:
            metadata: Chunk metadata
            location: Target location
            segment: Target segment

        Returns:
            Score 0-1
        """
        score = 0.0

        if location:
            chunk_location = metadata.get("location", "").lower()
            chunk_region = metadata.get("region", "").lower()
            target_loc = location.lower()

            if target_loc in chunk_location or target_loc in chunk_region:
                score += 0.5
            elif chunk_region and any(
                loc in target_loc for loc in chunk_region.split()
            ):
                score += 0.25

        if segment:
            chunk_segment = metadata.get("segment", "").lower()
            chunk_type = metadata.get("source_type", "").lower()
            target_seg = segment.lower()

            if target_seg in chunk_segment:
                score += 0.5
            elif target_seg in chunk_type:
                score += 0.25

        return min(score, 1.0)

    def _compute_posterior(
        self,
        vector_score: float,
        keyword_score: float,
        metadata_score: float,
    ) -> float:
        """Compute posterior P(relevant | scores) using logistic model.

        logit(P) = alpha + beta_v * s_v + beta_k * s_k + beta_m * s_m

        Args:
            vector_score: Vector similarity (0-1)
            keyword_score: Keyword match (0-1)
            metadata_score: Metadata match (0-1)

        Returns:
            Posterior probability (0-1)
        """
        logit = (
            self.ALPHA +
            self.BETA_VECTOR * vector_score +
            self.BETA_KEYWORD * keyword_score +
            self.BETA_META * metadata_score
        )

        # Sigmoid
        posterior = 1.0 / (1.0 + math.exp(-logit))

        return posterior

    def _compute_entropy(self, posteriors: list[float]) -> float:
        """Compute entropy of posterior distribution.

        Higher entropy = more uncertainty in retrieval quality.

        Args:
            posteriors: List of posterior probabilities

        Returns:
            Entropy value (0 = certain, higher = uncertain)
        """
        if not posteriors:
            return 1.0  # Maximum uncertainty

        # Normalize to distribution
        total = sum(posteriors)
        if total == 0:
            return 1.0

        probs = [p / total for p in posteriors]

        # Shannon entropy
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize by max entropy (uniform distribution)
        max_entropy = math.log2(len(probs)) if len(probs) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return normalized_entropy

    def format_context(self, chunks: list[RetrievedChunk], max_tokens: int = 4000) -> str:
        """Format retrieved chunks as context for LLM.

        Args:
            chunks: Retrieved chunks
            max_tokens: Approximate max tokens

        Returns:
            Formatted context string
        """
        if not chunks:
            return ""

        context_parts = []
        char_count = 0
        char_limit = max_tokens * 4  # Rough chars per token

        for i, chunk in enumerate(chunks):
            chunk_text = f"[Source {i+1}] ({chunk.source_type})\n{chunk.text}\n"

            if char_count + len(chunk_text) > char_limit:
                break

            context_parts.append(chunk_text)
            char_count += len(chunk_text)

        return "\n---\n".join(context_parts)

"""Clustering module for trend detection using HDBSCAN."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import hdbscan
import numpy as np

from db.database import SessionLocal
from db.models import RawContentModel
from db.vector_store import get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class Cluster:
    """Represents a content cluster (potential trend)."""

    cluster_id: int
    content_ids: list[str]
    centroid: np.ndarray | None
    size: int


class ContentClusterer:
    """Cluster content embeddings to identify trends."""

    def __init__(
        self,
        min_cluster_size: int = 3,
        min_samples: int = 2,
        metric: str = "euclidean",
    ):
        """Initialize clusterer.

        Args:
            min_cluster_size: Minimum cluster size for HDBSCAN
            min_samples: Min samples for core point
            metric: Distance metric
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric

    def get_embeddings_from_db(
        self,
        source_types: list[str] | None = None,
        days_back: int = 30,
        limit: int = 1000,
    ) -> tuple[list[str], np.ndarray]:
        """Fetch embeddings from vector store for clustering.

        Args:
            source_types: Filter by source types (e.g., ['social', 'news'])
            days_back: Only include content from last N days
            limit: Maximum number of items

        Returns:
            Tuple of (content_ids, embeddings_array)
        """
        db = SessionLocal()
        vector_store = get_vector_store()

        try:
            # Get processed content IDs from database
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)

            query = db.query(RawContentModel).filter(
                RawContentModel.is_processed == True,
                RawContentModel.scraped_at >= cutoff_date,
            )

            if source_types:
                query = query.filter(RawContentModel.source_type.in_(source_types))

            content_records = query.order_by(
                RawContentModel.scraped_at.desc()
            ).limit(limit).all()

            if not content_records:
                logger.warning("No processed content found for clustering")
                return [], np.array([])

            # Fetch embeddings from ChromaDB
            content_ids = [r.id for r in content_records]

            # Get embeddings in batches
            embeddings = []
            valid_ids = []

            for content_id in content_ids:
                result = vector_store.get_embedding(content_id)
                if result and result.get("embedding"):
                    embeddings.append(result["embedding"])
                    valid_ids.append(content_id)

            if not embeddings:
                logger.warning("No embeddings found in vector store")
                return [], np.array([])

            return valid_ids, np.array(embeddings)

        finally:
            db.close()

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """Run HDBSCAN clustering on embeddings.

        Args:
            embeddings: Array of embeddings (n_samples, n_features)

        Returns:
            Array of cluster labels (-1 for noise)
        """
        if len(embeddings) < self.min_cluster_size:
            logger.warning(f"Not enough samples ({len(embeddings)}) for clustering")
            return np.array([-1] * len(embeddings))

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method="eom",
        )

        labels = clusterer.fit_predict(embeddings)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        logger.info(f"Clustering: {n_clusters} clusters, {n_noise} noise points")
        return labels

    def extract_clusters(
        self,
        content_ids: list[str],
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> list[Cluster]:
        """Extract cluster information.

        Args:
            content_ids: List of content IDs
            embeddings: Embeddings array
            labels: Cluster labels from HDBSCAN

        Returns:
            List of Cluster objects (excluding noise)
        """
        clusters = []
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:  # Skip noise
                continue

            # Get indices for this cluster
            mask = labels == label
            cluster_ids = [content_ids[i] for i, m in enumerate(mask) if m]
            cluster_embeddings = embeddings[mask]

            # Calculate centroid
            centroid = np.mean(cluster_embeddings, axis=0)

            clusters.append(Cluster(
                cluster_id=int(label),
                content_ids=cluster_ids,
                centroid=centroid,
                size=len(cluster_ids),
            ))

        # Sort by size (largest first)
        clusters.sort(key=lambda c: c.size, reverse=True)
        return clusters

    def cluster_content(
        self,
        source_types: list[str] | None = None,
        days_back: int = 30,
        limit: int = 1000,
    ) -> list[Cluster]:
        """Full clustering pipeline: fetch, cluster, extract.

        Args:
            source_types: Filter by source types
            days_back: Days of content to include
            limit: Max items to cluster

        Returns:
            List of Cluster objects
        """
        logger.info(f"Starting clustering (days_back={days_back}, limit={limit})")

        # Get embeddings
        content_ids, embeddings = self.get_embeddings_from_db(
            source_types=source_types,
            days_back=days_back,
            limit=limit,
        )

        if len(embeddings) == 0:
            return []

        # Cluster
        labels = self.cluster(embeddings)

        # Extract clusters
        clusters = self.extract_clusters(content_ids, embeddings, labels)

        logger.info(f"Found {len(clusters)} clusters")
        return clusters


def get_content_for_cluster(cluster: Cluster) -> list[dict]:
    """Fetch full content records for a cluster.

    Args:
        cluster: Cluster object

    Returns:
        List of content dicts with text and metadata
    """
    db = SessionLocal()
    try:
        records = db.query(RawContentModel).filter(
            RawContentModel.id.in_(cluster.content_ids)
        ).all()

        return [
            {
                "id": r.id,
                "title": r.title,
                "content": r.content,
                "source": r.source,
                "source_type": r.source_type,
                "sentiment_score": r.sentiment_score,
                "metadata": r.metadata_json or {},
                "scraped_at": r.scraped_at,
            }
            for r in records
        ]
    finally:
        db.close()


def run_clustering(
    source_types: list[str] | None = None,
    days_back: int = 30,
    min_cluster_size: int = 3,
) -> list[Cluster]:
    """Convenience function to run clustering.

    Args:
        source_types: Filter by source types
        days_back: Days of content
        min_cluster_size: Min cluster size

    Returns:
        List of clusters
    """
    clusterer = ContentClusterer(min_cluster_size=min_cluster_size)
    return clusterer.cluster_content(
        source_types=source_types,
        days_back=days_back,
    )

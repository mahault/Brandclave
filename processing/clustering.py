"""Clustering module for trend detection using HDBSCAN.

Integrates with Clustering POMDP for adaptive parameter selection.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import hdbscan
import numpy as np

from db.database import SessionLocal
from db.models import RawContentModel
from db.vector_store import get_vector_store

logger = logging.getLogger(__name__)

# Import Clustering POMDP
try:
    from services.active_inference.clustering_pomdp import get_clustering_pomdp, ClusteringPOMDP
    CLUSTERING_POMDP_AVAILABLE = True
except ImportError:
    CLUSTERING_POMDP_AVAILABLE = False
    logger.info("Clustering POMDP not available")


@dataclass
class Cluster:
    """Represents a content cluster (potential trend)."""

    cluster_id: int
    content_ids: list[str]
    centroid: np.ndarray | None
    size: int


class ContentClusterer:
    """Cluster content embeddings to identify trends.

    Supports adaptive parameter selection via Clustering POMDP.
    """

    def __init__(
        self,
        min_cluster_size: int = 3,
        min_samples: int = 2,
        metric: str = "euclidean",
        use_adaptive: bool = True,
    ):
        """Initialize clusterer.

        Args:
            min_cluster_size: Minimum cluster size for HDBSCAN (used as fallback)
            min_samples: Min samples for core point (used as fallback)
            metric: Distance metric
            use_adaptive: Whether to use POMDP for adaptive parameter selection
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric

        # Initialize POMDP for adaptive clustering
        self.use_adaptive = use_adaptive and CLUSTERING_POMDP_AVAILABLE
        self.clustering_pomdp: Optional[ClusteringPOMDP] = None
        if self.use_adaptive:
            try:
                self.clustering_pomdp = get_clustering_pomdp()
                logger.info("Clustering POMDP enabled for adaptive parameter selection")
            except Exception as e:
                logger.warning(f"Failed to initialize Clustering POMDP: {e}")
                self.use_adaptive = False

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
                if result:
                    embedding = result.get("embedding")
                    if embedding is not None and len(embedding) > 0:
                        embeddings.append(embedding)
                        valid_ids.append(content_id)

            if not embeddings:
                logger.warning("No embeddings found in vector store")
                return [], np.array([])

            return valid_ids, np.array(embeddings)

        finally:
            db.close()

    def cluster(self, embeddings: np.ndarray) -> tuple[np.ndarray, dict]:
        """Run HDBSCAN clustering on embeddings.

        Uses POMDP for adaptive parameter selection when available.

        Args:
            embeddings: Array of embeddings (n_samples, n_features)

        Returns:
            Tuple of (cluster labels array, params dict)
        """
        # Determine parameters
        if self.clustering_pomdp is not None:
            param_decision = self.clustering_pomdp.select_parameters(embeddings)
            mcs = param_decision["min_cluster_size"]
            ms = param_decision["min_samples"]
            logger.info(f"POMDP selected params: min_cluster_size={mcs}, min_samples={ms}, "
                       f"confidence={param_decision.get('confidence', 0):.2f}")
        else:
            mcs = self.min_cluster_size
            ms = self.min_samples
            param_decision = {"method": "fixed", "min_cluster_size": mcs, "min_samples": ms}

        if len(embeddings) < mcs:
            logger.warning(f"Not enough samples ({len(embeddings)}) for clustering")
            return np.array([-1] * len(embeddings)), param_decision

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            metric=self.metric,
            cluster_selection_method="eom",
        )

        labels = clusterer.fit_predict(embeddings)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        logger.info(f"Clustering: {n_clusters} clusters, {n_noise} noise points")

        # Update POMDP with result
        if self.clustering_pomdp is not None:
            result = self.clustering_pomdp.observe_clustering_result(
                params={"min_cluster_size": mcs, "min_samples": ms},
                labels=labels,
                embeddings=embeddings,
            )
            logger.debug(f"Clustering POMDP updated: silhouette={result.get('silhouette', 0):.3f}")

        return labels, param_decision

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
    ) -> tuple[list[Cluster], dict]:
        """Full clustering pipeline: fetch, cluster, extract.

        Args:
            source_types: Filter by source types
            days_back: Days of content to include
            limit: Max items to cluster

        Returns:
            Tuple of (List of Cluster objects, params dict)
        """
        logger.info(f"Starting clustering (days_back={days_back}, limit={limit})")

        # Get embeddings
        content_ids, embeddings = self.get_embeddings_from_db(
            source_types=source_types,
            days_back=days_back,
            limit=limit,
        )

        if len(embeddings) == 0:
            return [], {}

        # Cluster (now returns labels and params)
        labels, params = self.cluster(embeddings)

        # Extract clusters
        clusters = self.extract_clusters(content_ids, embeddings, labels)

        logger.info(f"Found {len(clusters)} clusters")
        return clusters, params

    def get_pomdp_status(self) -> dict:
        """Get Clustering POMDP status.

        Returns:
            Dict with POMDP state information
        """
        if self.clustering_pomdp is not None:
            return self.clustering_pomdp.get_status()
        return {"enabled": False, "reason": "POMDP not available"}


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
    use_adaptive: bool = True,
) -> tuple[list[Cluster], dict]:
    """Convenience function to run clustering.

    Args:
        source_types: Filter by source types
        days_back: Days of content
        min_cluster_size: Min cluster size (fallback if POMDP disabled)
        use_adaptive: Whether to use POMDP for adaptive parameter selection

    Returns:
        Tuple of (List of clusters, params dict)
    """
    clusterer = ContentClusterer(min_cluster_size=min_cluster_size, use_adaptive=use_adaptive)
    return clusterer.cluster_content(
        source_types=source_types,
        days_back=days_back,
    )

"""
Adaptive Clustering POMDP using JAX/PyMDP.

Adaptively selects HDBSCAN parameters based on data characteristics
and observed clustering quality. Uses EFE minimization to balance
exploration of parameter space with exploitation of good parameters.
All computations are JIT-compiled for performance.
"""

import logging
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# JAX imports with graceful fallback
try:
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    from pymdp.agent import Agent
    JAX_AVAILABLE = True
    logger.info("JAX/PyMDP loaded for Clustering POMDP")
except ImportError as e:
    JAX_AVAILABLE = False
    logger.warning(f"JAX/PyMDP not available for Clustering POMDP: {e}")

# Scikit-learn for silhouette score
try:
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available for silhouette score")


@dataclass
class ClusteringResult:
    """Result from a clustering operation."""
    params: dict
    num_clusters: int
    noise_ratio: float
    silhouette: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ClusteringPOMDP:
    """
    POMDP controller for adaptive HDBSCAN parameter selection.

    Hidden States: Cluster quality levels (poor, fair, good, very_good, excellent)
    Observations: [silhouette_bucket, num_clusters_bucket, noise_bucket]
    Actions: Parameter combinations from grid
    Reward: silhouette_score - noise_penalty + stability_bonus

    All inference is JIT-compiled via PyMDP/JAX.
    """

    # Parameter grid
    MIN_CLUSTER_SIZES = [2, 3, 4, 5]
    MIN_SAMPLES = [1, 2, 3]

    # Quality levels (hidden states)
    QUALITY_LEVELS = ["poor", "fair", "good", "very_good", "excellent"]

    # Observation buckets
    SILHOUETTE_BUCKETS = ["negative", "low", "medium", "high"]  # <0, 0-0.25, 0.25-0.5, >0.5
    CLUSTER_COUNT_BUCKETS = ["few", "moderate", "many", "too_many"]  # 1-3, 4-10, 11-20, >20
    NOISE_BUCKETS = ["low", "moderate", "high"]  # <10%, 10-30%, >30%

    def __init__(
        self,
        learning_rate: float = 0.1,
        quality_threshold: float = 0.3,
        batch_size: int = 1,
        policy_len: int = 1,
        rng_seed: int = 42,
    ):
        """
        Initialize the Clustering POMDP.

        Args:
            learning_rate: Rate for updating beliefs
            quality_threshold: Minimum acceptable silhouette score
            batch_size: Batch size for JAX inference
            policy_len: Planning horizon
            rng_seed: Random seed for JAX
        """
        self.learning_rate = learning_rate
        self.quality_threshold = quality_threshold
        self.batch_size = batch_size
        self.policy_len = policy_len

        # Build action space (all parameter combinations)
        self.actions = []
        for mcs in self.MIN_CLUSTER_SIZES:
            for ms in self.MIN_SAMPLES:
                self.actions.append({
                    "min_cluster_size": mcs,
                    "min_samples": ms,
                })

        # Model dimensions
        self.num_states = [len(self.QUALITY_LEVELS)]  # 5 quality levels
        self.num_obs = [
            len(self.SILHOUETTE_BUCKETS),  # 4
            len(self.CLUSTER_COUNT_BUCKETS),  # 4
            len(self.NOISE_BUCKETS),  # 3
        ]
        self.num_actions = len(self.actions)
        self.num_controls = [self.num_actions]

        # Track parameter effectiveness
        self.param_beliefs: dict[str, float] = {}
        for action in self.actions:
            key = f"{action['min_cluster_size']}_{action['min_samples']}"
            self.param_beliefs[key] = 0.5  # Prior belief

        # History
        self.clustering_history: list[ClusteringResult] = []

        # Initialize JAX agent
        self.agent: Optional[Agent] = None
        self.rng_key = None
        self.qs = None
        self.empirical_prior = None

        if JAX_AVAILABLE:
            self.rng_key = jr.PRNGKey(rng_seed)
            self._initialize_agent()
        else:
            logger.warning("Running Clustering POMDP in fallback mode (no JAX)")

    def _initialize_agent(self):
        """Initialize the PyMDP Agent with JAX arrays."""
        if not JAX_AVAILABLE:
            return

        self.rng_key, key = jr.split(self.rng_key)

        # Build A matrix (likelihood)
        A = self._build_likelihood_model()

        # Build B matrix (transition)
        B = self._build_transition_model()

        # D: Prior over initial states (slightly favor good)
        D_vals = np.array([0.1, 0.2, 0.4, 0.2, 0.1])  # Prior favoring "good"
        D = [jnp.array(D_vals)[None, :].repeat(self.batch_size, axis=0)]

        # pA: Dirichlet priors for learning
        pA = [
            jnp.ones((self.batch_size, no, self.num_states[0])) * 1.0
            for no in self.num_obs
        ]

        try:
            self.agent = Agent(
                A=A,
                B=B,
                D=D,
                pA=pA,
                num_controls=self.num_controls,
                policy_len=self.policy_len,
                batch_size=self.batch_size,
                inference_algo="fpi",
                use_utility=True,
                use_states_info_gain=True,
                use_param_info_gain=True,
                action_selection="stochastic",
                learn_A=True,
                learn_B=False,
            )

            # Initialize beliefs with a dummy observation to get proper qs shape
            # This is required for infer_policies to work correctly
            dummy_obs = []
            for no in self.num_obs:
                one_hot = jnp.zeros((self.batch_size, no))
                one_hot = one_hot.at[:, 0].set(1.0)
                dummy_obs.append(one_hot)

            initial_qs = [jnp.array(D_vals)[None, :].repeat(self.batch_size, axis=0)]
            self.qs = self.agent.infer_states(
                observations=dummy_obs,
                empirical_prior=initial_qs,
            )
            self.empirical_prior = self.qs

            logger.info(f"Initialized Clustering POMDP with {self.num_actions} parameter combos (JAX/JIT enabled)")

        except Exception as e:
            logger.error(f"Failed to initialize Clustering POMDP agent: {e}")
            self.agent = None

    def _build_likelihood_model(self) -> list:
        """Build A matrix: P(observation | quality_state)."""
        A = []
        ns = self.num_states[0]  # 5 quality levels

        # Modality 1: Silhouette observation
        # excellent -> likely high silhouette, poor -> likely negative
        A_sil = np.array([
            [0.7, 0.3, 0.1, 0.05, 0.02],  # negative
            [0.2, 0.4, 0.3, 0.15, 0.08],  # low
            [0.08, 0.2, 0.4, 0.4, 0.3],  # medium
            [0.02, 0.1, 0.2, 0.4, 0.6],  # high
        ])
        A.append(jnp.array(A_sil)[None, ...].repeat(self.batch_size, axis=0))

        # Modality 2: Cluster count observation
        A_cnt = np.array([
            [0.5, 0.3, 0.2, 0.15, 0.1],  # few (often indicates poor params)
            [0.3, 0.4, 0.5, 0.5, 0.4],  # moderate (ideal)
            [0.15, 0.2, 0.25, 0.3, 0.4],  # many
            [0.05, 0.1, 0.05, 0.05, 0.1],  # too many (fragmented)
        ])
        A.append(jnp.array(A_cnt)[None, ...].repeat(self.batch_size, axis=0))

        # Modality 3: Noise ratio observation
        A_noise = np.array([
            [0.1, 0.3, 0.5, 0.6, 0.7],  # low noise
            [0.3, 0.4, 0.35, 0.3, 0.25],  # moderate noise
            [0.6, 0.3, 0.15, 0.1, 0.05],  # high noise
        ])
        A.append(jnp.array(A_noise)[None, ...].repeat(self.batch_size, axis=0))

        return A

    def _build_transition_model(self) -> list:
        """Build B matrix: P(quality' | quality, params)."""
        ns = self.num_states[0]
        na = self.num_actions

        # B[state_t+1, state_t, action]
        B = np.zeros((ns, ns, na))

        for a_idx, action in enumerate(self.actions):
            mcs = action["min_cluster_size"]
            ms = action["min_samples"]

            # Parameters affect transition probabilities
            # Larger min_cluster_size tends to be more stable but might miss small clusters
            # Smaller min_samples is more sensitive to noise

            stability = mcs / 5  # 0.4 - 1.0
            sensitivity = 1 - (ms / 3)  # 0.33 - 1.0

            # More stable params: higher chance of maintaining/improving quality
            for s in range(ns):
                # Probability of staying in current state
                B[s, s, a_idx] = 0.3 + 0.2 * stability

                # Probability of improving (if not already excellent)
                if s < ns - 1:
                    B[s + 1, s, a_idx] = 0.2 * (1 - sensitivity)

                # Probability of degrading (if not already poor)
                if s > 0:
                    B[s - 1, s, a_idx] = 0.1 * sensitivity

        # Normalize
        B = B / B.sum(axis=0, keepdims=True)

        return [jnp.array(B)[None, ...].repeat(self.batch_size, axis=0)]

    def select_parameters(self, embeddings: np.ndarray) -> dict:
        """
        Select clustering parameters using EFE minimization.

        Args:
            embeddings: The embeddings to be clustered

        Returns:
            Dict with selected parameters and confidence
        """
        # Estimate data characteristics
        data_density = self._estimate_density(embeddings)

        if self.agent is not None and JAX_AVAILABLE:
            try:
                # Run policy inference (JIT compiled)
                q_pi, G = self.agent.infer_policies(self.qs)

                # Sample action
                self.rng_key, *subkeys = jr.split(self.rng_key, self.batch_size + 1)
                batch_keys = jnp.stack(subkeys)
                action = self.agent.sample_action(q_pi, rng_key=batch_keys)

                # Extract action index
                if hasattr(action, 'ndim') and action.ndim > 1:
                    action_idx = int(action[0, 0])
                elif hasattr(action, '__iter__'):
                    action_idx = int(action[0])
                else:
                    action_idx = int(action)

                action_idx = min(action_idx, len(self.actions) - 1)

                # Get selected parameters
                params = self.actions[action_idx]

                # Get EFE values
                efe_values = np.array(G)
                if efe_values.ndim > 1:
                    efe_values = efe_values[0]

                # Compute confidence from belief entropy
                beliefs = np.array(self.qs[0])
                if beliefs.ndim == 3:
                    beliefs = beliefs[0, -1]
                elif beliefs.ndim == 2:
                    beliefs = beliefs[0]
                confidence = 1 - (-np.sum(beliefs * np.log(beliefs + 1e-10)) / np.log(len(beliefs)))

                return {
                    "min_cluster_size": params["min_cluster_size"],
                    "min_samples": params["min_samples"],
                    "action_index": action_idx,
                    "confidence": float(confidence),
                    "efe": float(efe_values[action_idx]) if action_idx < len(efe_values) else 0.0,
                    "data_density": data_density,
                    "method": "pymdp_efe",
                }

            except Exception as e:
                logger.warning(f"JAX policy inference failed: {e}")

        # Fallback: use data density heuristic
        return self._fallback_select_parameters(data_density)

    def _estimate_density(self, embeddings: np.ndarray) -> float:
        """Estimate data density from embeddings."""
        if len(embeddings) < 2:
            return 0.5

        # Simple heuristic: average pairwise distance normalized
        n_samples = min(100, len(embeddings))
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        sample = embeddings[indices]

        # Compute mean pairwise distance
        diffs = sample[:, None, :] - sample[None, :, :]
        distances = np.sqrt(np.sum(diffs ** 2, axis=-1))
        mean_dist = np.mean(distances)

        # Normalize to [0, 1] - higher means more spread out (lower density)
        density = 1.0 / (1.0 + mean_dist)
        return float(density)

    def _fallback_select_parameters(self, data_density: float) -> dict:
        """Fallback parameter selection based on heuristics."""
        # Higher density -> smaller min_cluster_size
        # Lower density -> larger min_cluster_size
        if data_density > 0.6:
            mcs = 2
            ms = 1
        elif data_density > 0.4:
            mcs = 3
            ms = 2
        elif data_density > 0.2:
            mcs = 4
            ms = 2
        else:
            mcs = 5
            ms = 3

        return {
            "min_cluster_size": mcs,
            "min_samples": ms,
            "confidence": 0.5,
            "data_density": data_density,
            "method": "fallback_heuristic",
        }

    def observe_clustering_result(
        self,
        params: dict,
        labels: np.ndarray,
        embeddings: np.ndarray,
    ) -> dict:
        """
        Update beliefs after observing clustering result.

        Args:
            params: Parameters used
            labels: Cluster labels (-1 for noise)
            embeddings: The clustered embeddings

        Returns:
            Updated beliefs and quality metrics
        """
        # Compute metrics
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = np.sum(labels == -1) / len(labels)

        # Compute silhouette score
        if num_clusters >= 2 and SKLEARN_AVAILABLE:
            try:
                # Filter out noise points for silhouette
                mask = labels != -1
                if np.sum(mask) > num_clusters:
                    sil_score = silhouette_score(embeddings[mask], labels[mask])
                else:
                    sil_score = -1.0
            except Exception:
                sil_score = -1.0
        else:
            sil_score = 0.0 if num_clusters == 1 else -1.0

        # Store result
        result = ClusteringResult(
            params=params,
            num_clusters=num_clusters,
            noise_ratio=noise_ratio,
            silhouette=sil_score,
        )
        self.clustering_history.append(result)

        # Update parameter beliefs
        key = f"{params['min_cluster_size']}_{params['min_samples']}"
        old_belief = self.param_beliefs.get(key, 0.5)
        quality = (sil_score + 1) / 2  # Normalize to [0, 1]
        self.param_beliefs[key] = old_belief * (1 - self.learning_rate) + quality * self.learning_rate

        # Encode observation for POMDP
        obs_indices = self._encode_observation(sil_score, num_clusters, noise_ratio)

        if self.agent is not None and JAX_AVAILABLE:
            try:
                # Convert to one-hot
                obs_one_hot = []
                for m, (obs_idx, no) in enumerate(zip(obs_indices, self.num_obs)):
                    one_hot = jnp.zeros((self.batch_size, no))
                    one_hot = one_hot.at[:, obs_idx].set(1.0)
                    obs_one_hot.append(one_hot)

                # Run inference
                self.qs = self.agent.infer_states(
                    observations=obs_one_hot,
                    empirical_prior=self.empirical_prior,
                )

                # Update empirical prior for next iteration
                # (Assuming last action was taken)
                # self.empirical_prior = self.qs

            except Exception as e:
                logger.warning(f"JAX inference failed during observation: {e}")

        return {
            "silhouette": sil_score,
            "num_clusters": num_clusters,
            "noise_ratio": noise_ratio,
            "quality_bucket": self._quality_from_silhouette(sil_score),
            "param_belief": self.param_beliefs[key],
        }

    def _encode_observation(
        self,
        silhouette: float,
        num_clusters: int,
        noise_ratio: float,
    ) -> list[int]:
        """Encode metrics into discrete observation indices."""
        # Silhouette bucket
        if silhouette < 0:
            sil_obs = 0
        elif silhouette < 0.25:
            sil_obs = 1
        elif silhouette < 0.5:
            sil_obs = 2
        else:
            sil_obs = 3

        # Cluster count bucket
        if num_clusters <= 3:
            cnt_obs = 0
        elif num_clusters <= 10:
            cnt_obs = 1
        elif num_clusters <= 20:
            cnt_obs = 2
        else:
            cnt_obs = 3

        # Noise bucket
        if noise_ratio < 0.1:
            noise_obs = 0
        elif noise_ratio < 0.3:
            noise_obs = 1
        else:
            noise_obs = 2

        return [sil_obs, cnt_obs, noise_obs]

    def _quality_from_silhouette(self, silhouette: float) -> str:
        """Map silhouette score to quality level."""
        if silhouette < 0:
            return "poor"
        elif silhouette < 0.2:
            return "fair"
        elif silhouette < 0.4:
            return "good"
        elif silhouette < 0.6:
            return "very_good"
        else:
            return "excellent"

    def get_free_energy(self) -> float:
        """Compute current free energy estimate."""
        if self.agent is not None and self.qs is not None and JAX_AVAILABLE:
            beliefs = np.array(self.qs[0])
            if beliefs.ndim == 3:
                beliefs = beliefs[0, -1]
            elif beliefs.ndim == 2:
                beliefs = beliefs[0]

            entropy = -np.sum(beliefs * np.log(beliefs + 1e-10))
            return float(entropy - 1.0)

        return 0.0

    def get_status(self) -> dict:
        """Get current POMDP status."""
        return {
            "enabled": True,
            "jax_available": JAX_AVAILABLE,
            "agent_initialized": self.agent is not None,
            "num_actions": self.num_actions,
            "total_clusterings": len(self.clustering_history),
            "free_energy": self.get_free_energy(),
            "param_beliefs": self.param_beliefs,
            "recent_results": [
                {
                    "params": r.params,
                    "silhouette": r.silhouette,
                    "num_clusters": r.num_clusters,
                    "noise_ratio": r.noise_ratio,
                }
                for r in self.clustering_history[-5:]
            ],
        }

    def reset(self):
        """Reset POMDP to initial state."""
        for key in self.param_beliefs:
            self.param_beliefs[key] = 0.5

        self.clustering_history = []

        if JAX_AVAILABLE:
            self._initialize_agent()

        logger.info("Clustering POMDP reset to initial state")


# Singleton instance
_clustering_pomdp: Optional[ClusteringPOMDP] = None


def get_clustering_pomdp() -> ClusteringPOMDP:
    """Get or create the clustering POMDP instance."""
    global _clustering_pomdp
    if _clustering_pomdp is None:
        _clustering_pomdp = ClusteringPOMDP()
    return _clustering_pomdp

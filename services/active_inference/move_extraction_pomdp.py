"""
Move Extraction POMDP using JAX/PyMDP.

Decides extraction strategy per article based on expected quality vs cost.
Uses Expected Free Energy (EFE) minimization for action selection.
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
    logger.info("JAX/PyMDP loaded for Move Extraction POMDP")
except ImportError as e:
    JAX_AVAILABLE = False
    logger.warning(f"JAX/PyMDP not available for Move Extraction POMDP: {e}")


@dataclass
class ExtractionResult:
    """Result from a move extraction operation."""
    article_id: str
    method: str
    quality: float
    cost: float
    success: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MoveExtractionPOMDP:
    """
    POMDP controller for adaptive move extraction strategy.

    Hidden States: Article quality levels (noise, weak_signal, clear_signal, strong_signal)
    Observations: [title_keywords, content_length, source_reputation, recency]
    Actions: skip, regex_extract, llm_single, llm_multi
    Reward: extraction_quality - llm_cost

    All inference is JIT-compiled via PyMDP/JAX.
    """

    # Extraction methods (actions)
    METHODS = ["skip", "regex", "llm_single", "llm_multi"]

    # Action costs (relative)
    ACTION_COSTS = {
        "skip": 0.0,
        "regex": 0.01,  # Very cheap
        "llm_single": 1.0,  # Baseline
        "llm_multi": 3.0,  # 3x cost for multi-sample
    }

    # Article quality levels (hidden states)
    QUALITY_LEVELS = ["noise", "weak_signal", "clear_signal", "strong_signal"]

    # Observation modalities
    OBS_TITLE_KEYWORDS = ["none", "weak", "strong"]  # 3 levels
    OBS_CONTENT_LENGTH = ["short", "medium", "long"]  # 3 levels
    OBS_SOURCE_REP = ["low", "medium", "high"]  # 3 levels

    # Source reputation scores (normalized)
    SOURCE_REPUTATION = {
        # High reputation news sources
        "skift": 0.9, "hospitalitynet": 0.85, "hoteldive": 0.85,
        "hotelmanagement": 0.8, "phocuswire": 0.8, "costar": 0.9,
        # Medium reputation
        "reddit": 0.5, "youtube": 0.5, "quora": 0.4,
        # Default
        "default": 0.5,
    }

    # Keywords that indicate strategic moves
    MOVE_KEYWORDS = [
        "acquisition", "merger", "partnership", "expansion", "investment",
        "launch", "rebrand", "renovation", "opening", "closing",
        "acquisition", "deal", "agreement", "joint venture", "stake",
        "million", "billion", "property", "portfolio", "brand",
        "hilton", "marriott", "ihg", "hyatt", "accor", "wyndham",
    ]

    def __init__(
        self,
        learning_rate: float = 0.1,
        quality_threshold: float = 0.5,
        batch_size: int = 1,
        policy_len: int = 1,
        rng_seed: int = 42,
    ):
        """
        Initialize the Move Extraction POMDP.

        Args:
            learning_rate: Rate for updating beliefs
            quality_threshold: Minimum quality to consider extraction successful
            batch_size: Batch size for JAX inference
            policy_len: Planning horizon
            rng_seed: Random seed for JAX
        """
        self.learning_rate = learning_rate
        self.quality_threshold = quality_threshold
        self.batch_size = batch_size
        self.policy_len = policy_len

        # Model dimensions
        self.num_states = [len(self.QUALITY_LEVELS)]  # 4 quality levels
        self.num_obs = [
            len(self.OBS_TITLE_KEYWORDS),  # 3
            len(self.OBS_CONTENT_LENGTH),  # 3
            len(self.OBS_SOURCE_REP),  # 3
        ]
        self.num_actions = len(self.METHODS)  # 4
        self.num_controls = [self.num_actions]

        # Track method effectiveness
        self.method_beliefs: dict[str, dict] = {}
        for method in self.METHODS:
            self.method_beliefs[method] = {
                "success_rate": 0.5,
                "avg_quality": 0.5,
                "total_uses": 0,
            }

        # History
        self.extraction_history: list[ExtractionResult] = []
        self.llm_calls_saved = 0
        self.total_extractions = 0

        # Initialize JAX agent
        self.agent: Optional[Agent] = None
        self.rng_key = None
        self.qs = None
        self.empirical_prior = None

        if JAX_AVAILABLE:
            self.rng_key = jr.PRNGKey(rng_seed)
            self._initialize_agent()
        else:
            logger.warning("Running Move Extraction POMDP in fallback mode (no JAX)")

    def _initialize_agent(self):
        """Initialize the PyMDP Agent with JAX arrays."""
        if not JAX_AVAILABLE:
            return

        self.rng_key, key = jr.split(self.rng_key)

        # Build A matrix (likelihood)
        A = self._build_likelihood_model()

        # Build B matrix (transition)
        B = self._build_transition_model()

        # D: Prior over initial states (uniform with slight bias toward weak signal)
        D_vals = np.array([0.3, 0.3, 0.25, 0.15])  # Most articles are noise/weak
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

            logger.info(f"Initialized Move Extraction POMDP with {self.num_actions} methods (JAX/JIT enabled)")

        except Exception as e:
            logger.error(f"Failed to initialize Move Extraction POMDP agent: {e}")
            self.agent = None

    def _build_likelihood_model(self) -> list:
        """Build A matrix: P(observation | article_quality)."""
        A = []
        ns = self.num_states[0]  # 4 quality levels

        # Modality 1: Title keywords observation
        # Strong signals have strong keywords, noise has none
        A_title = np.array([
            [0.7, 0.4, 0.2, 0.1],  # none
            [0.2, 0.4, 0.4, 0.3],  # weak
            [0.1, 0.2, 0.4, 0.6],  # strong
        ])
        A.append(jnp.array(A_title)[None, ...].repeat(self.batch_size, axis=0))

        # Modality 2: Content length observation
        # Strong signals tend to be medium-long
        A_length = np.array([
            [0.5, 0.3, 0.2, 0.1],  # short
            [0.3, 0.4, 0.5, 0.4],  # medium
            [0.2, 0.3, 0.3, 0.5],  # long
        ])
        A.append(jnp.array(A_length)[None, ...].repeat(self.batch_size, axis=0))

        # Modality 3: Source reputation observation
        # High rep sources more likely to have strong signals
        A_source = np.array([
            [0.4, 0.3, 0.2, 0.1],  # low rep
            [0.4, 0.4, 0.4, 0.3],  # medium rep
            [0.2, 0.3, 0.4, 0.6],  # high rep
        ])
        A.append(jnp.array(A_source)[None, ...].repeat(self.batch_size, axis=0))

        return A

    def _build_transition_model(self) -> list:
        """Build B matrix: P(quality' | quality, action)."""
        ns = self.num_states[0]
        na = self.num_actions

        # B[state_t+1, state_t, action]
        B = np.zeros((ns, ns, na))

        for a_idx, method in enumerate(self.METHODS):
            if method == "skip":
                # Skip doesn't reveal information, state stays same
                B[:, :, a_idx] = np.eye(ns)
            elif method == "regex":
                # Regex reveals a bit but not much
                for s in range(ns):
                    B[s, s, a_idx] = 0.7  # Mostly stays same
                    if s > 0:
                        B[s - 1, s, a_idx] = 0.15  # Might reveal as lower
                    if s < ns - 1:
                        B[s + 1, s, a_idx] = 0.15  # Might reveal as higher
            elif method == "llm_single":
                # LLM reveals true quality more accurately
                for s in range(ns):
                    B[s, s, a_idx] = 0.8
                    if s > 0:
                        B[s - 1, s, a_idx] = 0.1
                    if s < ns - 1:
                        B[s + 1, s, a_idx] = 0.1
            else:  # llm_multi
                # Multi-sample LLM is most accurate
                B[:, :, a_idx] = np.eye(ns) * 0.9  # 90% accurate
                for s in range(ns):
                    if s > 0:
                        B[s - 1, s, a_idx] = 0.05
                    if s < ns - 1:
                        B[s + 1, s, a_idx] = 0.05

        # Normalize
        B = B / B.sum(axis=0, keepdims=True)

        return [jnp.array(B)[None, ...].repeat(self.batch_size, axis=0)]

    def select_extraction_method(self, article: dict) -> dict:
        """
        Select extraction method using EFE minimization.

        Args:
            article: Article dict with title, content, source

        Returns:
            Dict with method, expected quality, expected cost, and reason
        """
        # Encode article features
        obs_indices = self._encode_article(article)

        if self.agent is not None and JAX_AVAILABLE:
            try:
                # Convert to one-hot
                obs_one_hot = []
                for m, (obs_idx, no) in enumerate(zip(obs_indices, self.num_obs)):
                    one_hot = jnp.zeros((self.batch_size, no))
                    one_hot = one_hot.at[:, obs_idx].set(1.0)
                    obs_one_hot.append(one_hot)

                # Update beliefs based on observation
                self.qs = self.agent.infer_states(
                    observations=obs_one_hot,
                    empirical_prior=self.empirical_prior,
                )

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

                action_idx = min(action_idx, len(self.METHODS) - 1)

                # Note: We don't update empirical_prior here because each article
                # is an independent decision. Learning happens through observe_extraction_result
                # which updates method_beliefs based on actual outcomes.

                # Get method and compute expected values
                method = self.METHODS[action_idx]
                cost = self.ACTION_COSTS[method]

                # Estimate expected quality from beliefs
                beliefs = np.array(self.qs[0])
                if beliefs.ndim == 3:
                    beliefs = beliefs[0, -1]
                elif beliefs.ndim == 2:
                    beliefs = beliefs[0]

                # Quality is weighted by state index
                quality_weights = np.array([0.1, 0.3, 0.6, 0.9])
                expected_quality = float(np.dot(beliefs, quality_weights))

                # Get EFE values
                efe_values = np.array(G)
                if efe_values.ndim > 1:
                    efe_values = efe_values[0]

                return {
                    "method": method,
                    "action_index": action_idx,
                    "expected_quality": expected_quality,
                    "expected_cost": cost,
                    "beliefs": {
                        self.QUALITY_LEVELS[i]: float(beliefs[i])
                        for i in range(len(self.QUALITY_LEVELS))
                    },
                    "efe": float(efe_values[action_idx]) if action_idx < len(efe_values) else 0.0,
                    "reason": f"EFE minimization (G={float(efe_values[action_idx]):.3f})" if action_idx < len(efe_values) else "EFE",
                    "method_detail": "pymdp_efe",
                }

            except Exception as e:
                logger.warning(f"JAX policy inference failed: {e}")

        # Fallback: heuristic-based selection
        return self._fallback_select_method(article, obs_indices)

    def _encode_article(self, article: dict) -> list[int]:
        """Encode article features into observation indices."""
        title = (article.get("title") or "").lower()
        content = article.get("content") or ""
        source = (article.get("source") or "default").lower()

        # Title keywords
        keyword_count = sum(1 for kw in self.MOVE_KEYWORDS if kw in title)
        if keyword_count == 0:
            title_obs = 0  # none
        elif keyword_count <= 2:
            title_obs = 1  # weak
        else:
            title_obs = 2  # strong

        # Content length
        content_len = len(content)
        if content_len < 500:
            length_obs = 0  # short
        elif content_len < 2000:
            length_obs = 1  # medium
        else:
            length_obs = 2  # long

        # Source reputation
        rep = self.SOURCE_REPUTATION.get(source, self.SOURCE_REPUTATION["default"])
        if rep < 0.4:
            source_obs = 0  # low
        elif rep < 0.7:
            source_obs = 1  # medium
        else:
            source_obs = 2  # high

        return [title_obs, length_obs, source_obs]

    def _fallback_select_method(self, article: dict, obs_indices: list[int]) -> dict:
        """Fallback method selection without JAX."""
        title_obs, length_obs, source_obs = obs_indices

        # Simple heuristic: score based on observations
        score = (title_obs * 0.5) + (length_obs * 0.3) + (source_obs * 0.2)
        score /= 2.0  # Normalize to 0-1

        # Select method based on score
        if score < 0.2:
            method = "skip"
        elif score < 0.4:
            method = "regex"
        elif score < 0.7:
            method = "llm_single"
        else:
            method = "llm_multi"

        return {
            "method": method,
            "expected_quality": score,
            "expected_cost": self.ACTION_COSTS[method],
            "reason": f"Fallback heuristic (score={score:.2f})",
            "method_detail": "fallback",
        }

    def observe_extraction_result(
        self,
        article: dict,
        method: str,
        result: dict | None,
    ) -> dict:
        """
        Update beliefs after observing extraction result.

        Args:
            article: The processed article
            method: Method used (skip, regex, llm_single, llm_multi)
            result: Extraction result dict (None if skipped)

        Returns:
            Updated statistics
        """
        self.total_extractions += 1

        # Determine quality and success
        if result is None:
            quality = 0.0
            success = False
        else:
            quality = result.get("confidence_score", 0.5)
            success = quality >= self.quality_threshold

        cost = self.ACTION_COSTS.get(method, 0)

        # Track LLM savings
        if method in ["skip", "regex"]:
            self.llm_calls_saved += 1

        # Update method beliefs
        if method in self.method_beliefs:
            mb = self.method_beliefs[method]
            mb["total_uses"] += 1
            n = mb["total_uses"]

            # Update running averages
            mb["success_rate"] = (mb["success_rate"] * (n - 1) + (1.0 if success else 0.0)) / n
            mb["avg_quality"] = (mb["avg_quality"] * (n - 1) + quality) / n

        # Store result
        self.extraction_history.append(ExtractionResult(
            article_id=article.get("id", "unknown"),
            method=method,
            quality=quality,
            cost=cost,
            success=success,
        ))

        return {
            "method": method,
            "quality": quality,
            "success": success,
            "cost": cost,
            "llm_calls_saved": self.llm_calls_saved,
            "total_extractions": self.total_extractions,
        }

    def get_confidence_threshold(self) -> float:
        """Get adaptive confidence threshold based on learned quality distribution."""
        if not self.extraction_history:
            return self.quality_threshold

        # Compute recent quality distribution
        recent = self.extraction_history[-100:]
        qualities = [r.quality for r in recent if r.quality > 0]

        if not qualities:
            return self.quality_threshold

        # Adaptive threshold: aim for top 30% of results
        sorted_q = sorted(qualities)
        idx = int(len(sorted_q) * 0.7)
        return sorted_q[idx] if idx < len(sorted_q) else self.quality_threshold

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
            "num_methods": len(self.METHODS),
            "total_extractions": self.total_extractions,
            "llm_calls_saved": self.llm_calls_saved,
            "savings_rate": self.llm_calls_saved / max(self.total_extractions, 1),
            "free_energy": self.get_free_energy(),
            "method_beliefs": self.method_beliefs,
            "adaptive_threshold": self.get_confidence_threshold(),
            "recent_results": [
                {
                    "article_id": r.article_id,
                    "method": r.method,
                    "quality": r.quality,
                    "success": r.success,
                }
                for r in self.extraction_history[-5:]
            ],
        }

    def reset(self):
        """Reset POMDP to initial state."""
        for method in self.METHODS:
            self.method_beliefs[method] = {
                "success_rate": 0.5,
                "avg_quality": 0.5,
                "total_uses": 0,
            }

        self.extraction_history = []
        self.llm_calls_saved = 0
        self.total_extractions = 0

        if JAX_AVAILABLE:
            self._initialize_agent()

        logger.info("Move Extraction POMDP reset to initial state")


# Singleton instance
_extraction_pomdp: Optional[MoveExtractionPOMDP] = None


def get_extraction_pomdp() -> MoveExtractionPOMDP:
    """Get or create the move extraction POMDP instance."""
    global _extraction_pomdp
    if _extraction_pomdp is None:
        _extraction_pomdp = MoveExtractionPOMDP()
    return _extraction_pomdp

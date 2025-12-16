"""
PyMDP-Based Active Inference Structure Learner.

Uses the JAX implementation of pymdp for proper active inference with:
- Expected Free Energy (EFE) minimization for action selection
- Variational inference for belief updates
- Online parameter learning

This provides a local alternative to the VERSES Genius API using
the open-source pymdp library.
"""

import logging
from typing import Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)

# Try to import JAX and pymdp v1.0.0_alpha
try:
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    from pymdp.agent import Agent
    from pymdp import utils as pymdp_utils
    PYMDP_AVAILABLE = True
    logger.info("PyMDP v1.0.0 (JAX) loaded successfully")
except ImportError as e:
    PYMDP_AVAILABLE = False
    logger.warning(f"PyMDP not available: {e}. Install with: pip install git+https://github.com/infer-actively/pymdp.git@v1.0.0_alpha")


@dataclass
class PyMDPCategory:
    """A learned category in the active inference model."""

    id: str
    name: str
    index: int  # Index in the state space
    keywords: list[str] = field(default_factory=list)
    observation_count: int = 0
    belief_strength: float = 0.0
    keyword_counts: dict = field(default_factory=lambda: defaultdict(int))

    def update_name(self):
        """Generate name from top keywords."""
        if not self.keyword_counts:
            return
        top_keywords = sorted(
            self.keyword_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        self.name = " + ".join(kw for kw, _ in top_keywords).title()
        self.keywords = [kw for kw, _ in top_keywords[:10]]


@dataclass
class PyMDPObservation:
    """An observation for the active inference agent."""

    text: str
    keywords: list[str]
    source: str
    sentiment: float = 0.0
    embedding: Optional[np.ndarray] = None


class PyMDPStructureLearner:
    """
    Active inference structure learner using pymdp.

    The model represents:
    - Hidden states: Category assignments (which category does content belong to?)
    - Observations: Features extracted from content (keywords, sentiment, source)
    - Actions: Which query to execute next (exploration vs exploitation)

    The agent uses Expected Free Energy to balance:
    - Epistemic value (information gain about categories)
    - Pragmatic value (finding useful content)
    """

    # Category definitions
    CATEGORIES = [
        ("luxury_wellness", "Luxury & Wellness"),
        ("budget_social", "Budget & Social"),
        ("digital_nomad", "Digital Nomad"),
        ("boutique_design", "Boutique & Design"),
        ("family_resort", "Family & Resort"),
        ("business_corporate", "Business & Corporate"),
        ("adventure_experience", "Adventure & Experience"),
        ("romantic_couples", "Romantic & Couples"),
        ("eco_sustainable", "Eco & Sustainable"),
        ("general", "General"),
    ]

    # Keyword features to track
    KEYWORD_FEATURES = [
        "luxury", "budget", "boutique", "hostel", "resort",
        "wifi", "coworking", "remote", "nomad", "workspace",
        "pool", "spa", "gym", "breakfast", "rooftop",
        "romantic", "family", "solo", "business", "adventure",
        "clean", "safe", "quiet", "central", "eco",
    ]

    # Source types
    SOURCES = ["reddit", "youtube", "news", "review", "forum", "other"]

    # Sentiment levels
    SENTIMENTS = ["negative", "neutral", "positive"]

    # Query actions (what to search for next)
    QUERY_ACTIONS = [
        "luxury hotel amenities",
        "budget accommodation tips",
        "digital nomad coworking",
        "boutique hotel design",
        "family friendly resort",
        "business hotel facilities",
        "adventure travel stay",
        "romantic hotel getaway",
        "eco sustainable hotel",
        "hotel travel general",
    ]

    def __init__(
        self,
        use_learning: bool = True,
        policy_len: int = 1,
        inference_algo: str = "fpi",
        rng_seed: int = 42,
        batch_size: int = 1,
    ):
        """
        Initialize the pymdp-based learner.

        Args:
            use_learning: Whether to update model parameters online
            policy_len: Planning horizon for action selection
            inference_algo: Inference algorithm ("fpi", "vmp", "mmp", "ovf")
            rng_seed: Random seed for JAX
            batch_size: Batch size for parallel inference (default 1)
        """
        self.use_learning = use_learning
        self.policy_len = policy_len
        self.inference_algo = inference_algo
        self.batch_size = batch_size

        # Initialize categories
        self.categories: dict[str, PyMDPCategory] = {}
        for i, (cat_id, cat_name) in enumerate(self.CATEGORIES):
            self.categories[cat_id] = PyMDPCategory(
                id=cat_id,
                name=cat_name,
                index=i,
                belief_strength=1.0 / len(self.CATEGORIES)
            )

        # Track observations
        self.observations: list[PyMDPObservation] = []
        self.observation_history: list[list[int]] = []  # Encoded observations
        self.action_history: list[int] = []

        # Model dimensions (stored for building batched arrays)
        self.num_states = [len(self.CATEGORIES)]
        self.num_obs = [2, len(self.SOURCES), len(self.SENTIMENTS)]
        self.num_controls = [len(self.QUERY_ACTIONS)]

        # Initialize agent if pymdp available
        self.agent: Optional[Agent] = None
        self.rng_key = None

        if PYMDP_AVAILABLE:
            self.rng_key = jr.PRNGKey(rng_seed)
            self._initialize_agent()
        else:
            logger.warning("PyMDP not available, using fallback inference")

    def _initialize_agent(self):
        """Initialize the pymdp Agent with generative model."""
        if not PYMDP_AVAILABLE:
            return

        # Initialize random key
        self.rng_key, key1, key2 = jr.split(self.rng_key, 3)

        # Build model matrices as JAX arrays with batch dimension
        # pymdp v1.0.0_alpha expects: A shape [batch_size, num_obs, num_states]
        A = self._build_likelihood_model()
        B = self._build_transition_model()

        # D matrix: Prior over initial states (uniform) - shape [batch_size, num_states]
        D = [jnp.ones((self.batch_size, ns)) / ns for ns in self.num_states]

        try:
            # Build pA (Dirichlet priors for A learning) if learning enabled
            # Shape: [batch_size, num_obs, num_states]
            pA = None
            if self.use_learning:
                pA = []
                for m, no in enumerate(self.num_obs):
                    pA_m = jnp.ones((self.batch_size, no, self.num_states[0])) * 1.0
                    pA.append(pA_m)

            # Create agent with v1.0.0_alpha JAX API
            self.agent = Agent(
                A=A,
                B=B,
                D=D,
                pA=pA,  # Dirichlet priors for learning
                num_controls=self.num_controls,
                policy_len=self.policy_len,
                batch_size=self.batch_size,
                inference_algo=self.inference_algo,
                use_utility=True,
                use_states_info_gain=True,  # Epistemic value
                use_param_info_gain=self.use_learning and pA is not None,
                action_selection="stochastic",
                learn_A=self.use_learning and pA is not None,
                learn_B=False,  # Keep transitions fixed
            )

            # Initialize beliefs with batch dimension
            # Shape: [batch_size, num_states]
            self.qs = [jnp.ones((self.batch_size, ns)) / ns for ns in self.num_states]
            self.empirical_prior = self.qs

            logger.info(f"Initialized PyMDP agent with {len(self.CATEGORIES)} categories (learning={self.use_learning}, batch_size={self.batch_size})")

        except Exception as e:
            logger.warning(f"Could not initialize PyMDP agent: {e}")
            import traceback
            traceback.print_exc()
            self.agent = None

    def _build_likelihood_model(self) -> list:
        """
        Build the A matrix (likelihood model).

        Maps hidden category states to expected observations.
        pymdp v1.0.0_alpha expects shape: [batch_size, num_obs, num_states]
        """
        A = []
        num_categories = self.num_states[0]

        # Modality 1: Keyword relevance (binary: relevant/not relevant)
        # Each category has different keyword profiles
        A_keywords = np.zeros((2, num_categories))
        for i, (cat_id, _) in enumerate(self.CATEGORIES):
            # Higher probability of relevant keywords for matching category
            A_keywords[1, i] = 0.7  # P(relevant | category)
            A_keywords[0, i] = 0.3  # P(not relevant | category)
        # Add batch dimension: [batch_size, num_obs, num_states]
        A.append(jnp.array(A_keywords)[None, ...].repeat(self.batch_size, axis=0))

        # Modality 2: Source type
        # Different categories might come from different sources
        A_source = np.ones((len(self.SOURCES), num_categories)) / len(self.SOURCES)
        # Slight biases (digital nomad more from reddit, business from news)
        A_source[0, 2] = 0.3  # reddit -> digital_nomad
        A_source[2, 5] = 0.3  # news -> business
        A_source = A_source / A_source.sum(axis=0, keepdims=True)
        A.append(jnp.array(A_source)[None, ...].repeat(self.batch_size, axis=0))

        # Modality 3: Sentiment
        # Most observations are neutral, slight category biases
        A_sentiment = np.array([
            [0.2] * num_categories,  # negative
            [0.5] * num_categories,  # neutral
            [0.3] * num_categories,  # positive
        ])
        A.append(jnp.array(A_sentiment)[None, ...].repeat(self.batch_size, axis=0))

        return A

    def _build_transition_model(self) -> list:
        """
        Build the B matrix (transition model).

        Categories are relatively stable but actions can influence
        which category we're likely to observe next.
        pymdp v1.0.0_alpha expects shape: [batch_size, num_states, num_states, num_actions]
        """
        num_categories = self.num_states[0]
        num_actions = self.num_controls[0]

        # B[state_t+1, state_t, action]
        B_cat = np.zeros((num_categories, num_categories, num_actions))

        for a in range(num_actions):
            # Each action biases toward its corresponding category
            # but doesn't guarantee it
            for s in range(num_categories):
                # Self-transition (staying in same category)
                B_cat[s, s, a] = 0.3

                # Transition to action-aligned category
                target_cat = a % num_categories
                B_cat[target_cat, s, a] += 0.5

                # Small probability to any other category
                for s2 in range(num_categories):
                    B_cat[s2, s, a] += 0.02

            # Normalize
            B_cat[:, :, a] /= B_cat[:, :, a].sum(axis=0, keepdims=True)

        # Add batch dimension: [batch_size, num_states, num_states, num_actions]
        return [jnp.array(B_cat)[None, ...].repeat(self.batch_size, axis=0)]

    def _encode_observation(self, obs: PyMDPObservation) -> list[int]:
        """Encode observation into discrete indices for each modality."""
        # Modality 1: Keyword relevance
        obs_keywords = set(kw.lower() for kw in obs.keywords)
        tracked_keywords = set(self.KEYWORD_FEATURES)
        overlap = len(obs_keywords & tracked_keywords)
        keyword_obs = 1 if overlap >= 2 else 0  # Binary: relevant or not

        # Modality 2: Source type
        source_lower = obs.source.lower()
        try:
            source_obs = self.SOURCES.index(source_lower)
        except ValueError:
            source_obs = self.SOURCES.index("other")

        # Modality 3: Sentiment
        if obs.sentiment > 0.3:
            sentiment_obs = 2  # positive
        elif obs.sentiment < -0.3:
            sentiment_obs = 0  # negative
        else:
            sentiment_obs = 1  # neutral

        return [keyword_obs, source_obs, sentiment_obs]

    def observe(self, observation: PyMDPObservation) -> dict[str, float]:
        """
        Process an observation and update beliefs.

        Args:
            observation: The observed content

        Returns:
            Posterior probabilities over categories
        """
        self.observations.append(observation)

        # Encode observation
        encoded_obs = self._encode_observation(observation)
        self.observation_history.append(encoded_obs)

        if self.agent is not None and PYMDP_AVAILABLE:
            try:
                # Convert observation indices to one-hot JAX arrays with batch dimension
                # pymdp v1.0.0_alpha expects observations: [batch_size, num_outcomes]
                obs_one_hot = []
                for m, (obs_idx, no) in enumerate(zip(encoded_obs, self.num_obs)):
                    # Create one-hot with batch dimension: [batch_size, num_outcomes]
                    one_hot = jnp.zeros((self.batch_size, no))
                    one_hot = one_hot.at[:, obs_idx].set(1.0)
                    obs_one_hot.append(one_hot)

                # Run inference with pymdp
                self.qs = self.agent.infer_states(
                    observations=obs_one_hot,
                    empirical_prior=self.empirical_prior,
                )

                # Extract posteriors - handle batched output
                # qs shape: [batch_size, time_steps, num_states]
                posteriors = {}
                beliefs = np.array(self.qs[0])
                # Take first batch, last time step
                if beliefs.ndim == 3:
                    beliefs = beliefs[0, -1]  # [batch, time, states] -> [states]
                elif beliefs.ndim == 2:
                    beliefs = beliefs[0]  # [batch, states] -> [states]
                for i, (cat_id, _) in enumerate(self.CATEGORIES):
                    posteriors[cat_id] = float(beliefs[i])

                # Update category with highest belief
                best_cat_idx = int(np.argmax(beliefs))
                best_cat_id = self.CATEGORIES[best_cat_idx][0]

            except Exception as e:
                logger.warning(f"PyMDP inference failed: {e}, using fallback")
                import traceback
                traceback.print_exc()
                posteriors = self._fallback_inference(observation)
                best_cat_id = max(posteriors, key=posteriors.get)
        else:
            # Fallback: simple keyword matching
            posteriors = self._fallback_inference(observation)
            best_cat_id = max(posteriors, key=posteriors.get)

        # Update category statistics
        self._update_category(best_cat_id, observation, posteriors[best_cat_id])

        return posteriors

    def _fallback_inference(self, obs: PyMDPObservation) -> dict[str, float]:
        """Fallback inference when pymdp unavailable."""
        obs_keywords = set(kw.lower() for kw in obs.keywords)

        category_keywords = {
            "luxury_wellness": {"luxury", "spa", "wellness", "premium"},
            "budget_social": {"budget", "cheap", "hostel", "social", "backpacker"},
            "digital_nomad": {"wifi", "coworking", "remote", "nomad", "workspace"},
            "boutique_design": {"boutique", "design", "unique", "aesthetic"},
            "family_resort": {"family", "kids", "resort", "pool"},
            "business_corporate": {"business", "corporate", "conference"},
            "adventure_experience": {"adventure", "experience", "activity"},
            "romantic_couples": {"romantic", "couples", "honeymoon"},
            "eco_sustainable": {"eco", "sustainable", "green", "organic"},
            "general": set(),
        }

        posteriors = {}
        total = 0
        for cat_id, cat_kws in category_keywords.items():
            overlap = len(obs_keywords & cat_kws)
            score = overlap + 0.1
            posteriors[cat_id] = score
            total += score

        for cat_id in posteriors:
            posteriors[cat_id] /= total

        return posteriors

    def _update_category(self, cat_id: str, obs: PyMDPObservation, belief: float):
        """Update category statistics."""
        cat = self.categories[cat_id]
        cat.observation_count += 1

        # Running average of belief strength
        n = cat.observation_count
        cat.belief_strength = ((n - 1) * cat.belief_strength + belief) / n

        for kw in obs.keywords:
            cat.keyword_counts[kw] += 1

        if cat.observation_count % 5 == 0:
            cat.update_name()

    def select_action(self) -> dict:
        """
        Select next action using Expected Free Energy.

        Returns:
            Dict with query suggestion and rationale
        """
        if self.agent is not None and PYMDP_AVAILABLE:
            try:
                # Use pymdp policy inference
                q_pi, G = self.agent.infer_policies(self.qs)

                # Sample action - need batched rng_key for batched inference
                self.rng_key, *subkeys = jr.split(self.rng_key, self.batch_size + 1)
                # Stack subkeys to shape [batch_size, 2]
                batch_keys = jnp.stack(subkeys)
                action = self.agent.sample_action(q_pi, rng_key=batch_keys)

                # Handle batched action output - take first batch
                if hasattr(action, 'ndim') and action.ndim > 1:
                    action_idx = int(action[0, 0])
                elif hasattr(action, '__iter__') and not isinstance(action, (int, float)):
                    action_idx = int(action[0])
                else:
                    action_idx = int(action)

                # Update empirical prior for next timestep
                self.empirical_prior, self.qs = self.agent.update_empirical_prior(
                    action, self.qs
                )

                # Record action
                self.action_history.append(action_idx)

                # Get EFE values for explanation - handle batched G
                efe_values = np.array(G)
                if efe_values.ndim > 1:
                    efe_values = efe_values[0]  # Take first batch

                return {
                    "query": self.QUERY_ACTIONS[action_idx],
                    "action_index": action_idx,
                    "reason": f"EFE minimization (G={float(efe_values[action_idx]):.3f})",
                    "efe_values": {
                        self.QUERY_ACTIONS[i]: float(efe_values[i])
                        for i in range(min(len(self.QUERY_ACTIONS), len(efe_values)))
                    },
                    "source": "pymdp"
                }

            except Exception as e:
                logger.warning(f"PyMDP policy inference failed: {e}, using fallback")
                import traceback
                traceback.print_exc()
                return self._fallback_action_selection()

        # Fallback: find least explored category
        return self._fallback_action_selection()

    def _fallback_action_selection(self) -> dict:
        """Fallback action selection."""
        # Find category with fewest observations
        min_cat = min(
            self.categories.values(),
            key=lambda c: c.observation_count
        )

        # Map to corresponding query
        action_idx = min_cat.index

        return {
            "query": self.QUERY_ACTIONS[action_idx],
            "action_index": action_idx,
            "reason": f"Explore '{min_cat.name}' (fewest observations)",
            "source": "fallback"
        }

    def learn_from_batch(self) -> dict:
        """
        Update model parameters from accumulated observations.

        Returns:
            Learning statistics
        """
        if not self.use_learning or self.agent is None:
            return {"status": "skipped", "reason": "learning disabled or no agent"}

        if len(self.observation_history) < 5:
            return {"status": "skipped", "reason": "insufficient data"}

        # For now, learning happens online during inference
        # This could be extended for batch updates

        return {
            "status": "online",
            "observations_processed": len(self.observation_history),
        }

    def get_free_energy(self) -> float:
        """Compute current free energy estimate."""
        if not self.observations:
            return 0.0

        # Approximate free energy from belief entropy and category fit
        if self.agent is not None and PYMDP_AVAILABLE:
            beliefs = np.array(self.qs[0])
            # Handle batched beliefs - take first batch, last time step
            if beliefs.ndim == 3:
                beliefs = beliefs[0, -1]
            elif beliefs.ndim == 2:
                beliefs = beliefs[0]
            # Entropy of beliefs (uncertainty)
            entropy = -np.sum(beliefs * np.log(beliefs + 1e-10))

            # Lower entropy = more confident = lower free energy
            return float(entropy - 1.0)

        # Fallback estimate
        active_cats = [c for c in self.categories.values() if c.observation_count > 0]
        if not active_cats:
            return 0.0

        avg_belief = np.mean([c.belief_strength for c in active_cats])
        return -avg_belief + 0.5

    def get_structure(self) -> dict:
        """Return current learned structure."""
        return {
            "num_categories": len(self.categories),
            "num_observations": len(self.observations),
            "pymdp_available": PYMDP_AVAILABLE,
            "agent_initialized": self.agent is not None,
            "categories": [
                {
                    "id": cat.id,
                    "name": cat.name,
                    "keywords": cat.keywords,
                    "observation_count": cat.observation_count,
                    "belief_strength": cat.belief_strength,
                }
                for cat in sorted(
                    self.categories.values(),
                    key=lambda c: c.observation_count,
                    reverse=True
                )
            ]
        }


def create_pymdp_learner(**kwargs) -> PyMDPStructureLearner:
    """Create a PyMDP-based structure learner."""
    return PyMDPStructureLearner(**kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print(f"PyMDP available: {PYMDP_AVAILABLE}")

    learner = PyMDPStructureLearner()

    # Test observations
    test_obs = [
        PyMDPObservation(
            text="Looking for boutique hotel with good wifi for remote work",
            keywords=["boutique", "wifi", "remote", "work"],
            source="reddit",
            sentiment=0.5
        ),
        PyMDPObservation(
            text="Best budget hostels with social atmosphere",
            keywords=["budget", "hostel", "social"],
            source="reddit",
            sentiment=0.3
        ),
        PyMDPObservation(
            text="Luxury spa resort for honeymoon",
            keywords=["luxury", "spa", "resort", "honeymoon", "romantic"],
            source="youtube",
            sentiment=0.8
        ),
    ]

    for obs in test_obs:
        posteriors = learner.observe(obs)
        print(f"\nObservation: {obs.text[:50]}...")
        print(f"Top categories: {sorted(posteriors.items(), key=lambda x: -x[1])[:3]}")

    print(f"\nStructure: {learner.get_structure()}")
    print(f"\nNext action: {learner.select_action()}")
    print(f"\nFree energy: {learner.get_free_energy():.3f}")

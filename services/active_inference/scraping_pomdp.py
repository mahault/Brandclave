"""
Adaptive Scraping POMDP using JAX/PyMDP.

Decides which sources to scrape based on expected information gain vs cost.
Uses Expected Free Energy (EFE) minimization for action selection.
All computations are JIT-compiled for performance.
"""

import logging
from datetime import datetime, timedelta
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
    logger.info("JAX/PyMDP loaded for Scraping POMDP")
except ImportError as e:
    JAX_AVAILABLE = False
    logger.warning(f"JAX/PyMDP not available for Scraping POMDP: {e}")


@dataclass
class SourceState:
    """State information for a single scraping source."""
    name: str
    index: int
    productivity_belief: float = 0.5  # Expected items per scrape
    freshness_belief: float = 0.5  # How fresh the content is
    error_rate_belief: float = 0.1  # Expected error rate
    last_scraped: Optional[datetime] = None
    total_items: int = 0
    total_errors: int = 0
    observation_count: int = 0


class ScrapingPOMDP:
    """
    POMDP controller for adaptive scraping decisions.

    Hidden States: Source productivity levels (high, medium, low, stale)
    Observations: [productivity_level, freshness, error_rate, novelty]
    Actions: Scrape source_i (one per source) + wait
    Reward: information_gain - time_cost - error_penalty

    All inference is JIT-compiled via PyMDP/JAX.
    """

    # All available scraping sources
    SOURCES = [
        # News sources
        "skift", "hospitalitynet", "hoteldive", "hotelmanagement",
        "phocuswire", "travelweekly", "hotelnewsresource", "traveldailynews",
        "businesstravelnews", "boutiquehotelier", "hotelonline", "hoteltechreport",
        "tophotelnews", "siteminder", "ehlinsights", "cbrehotels",
        "cushmanwakefield", "costar", "traveldaily",
        # Social sources
        "reddit", "youtube", "quora",
        # Review sources
        "tripadvisor", "booking",
    ]

    # Productivity levels (hidden states)
    PRODUCTIVITY_LEVELS = ["high", "medium", "low", "stale"]

    # Observation modalities
    OBS_PRODUCTIVITY = ["high", "medium", "low"]  # 3 levels
    OBS_FRESHNESS = ["fresh", "moderate", "stale"]  # 3 levels
    OBS_ERROR_RATE = ["low", "medium", "high"]  # 3 levels

    def __init__(
        self,
        learning_rate: float = 0.1,
        exploration_bonus: float = 0.2,
        batch_size: int = 1,
        policy_len: int = 1,
        rng_seed: int = 42,
    ):
        """
        Initialize the Scraping POMDP.

        Args:
            learning_rate: Rate for updating beliefs
            exploration_bonus: Bonus for exploring uncertain sources
            batch_size: Batch size for JAX inference
            policy_len: Planning horizon
            rng_seed: Random seed for JAX
        """
        self.learning_rate = learning_rate
        self.exploration_bonus = exploration_bonus
        self.batch_size = batch_size
        self.policy_len = policy_len

        # Initialize source states
        self.sources: dict[str, SourceState] = {}
        for i, name in enumerate(self.SOURCES):
            self.sources[name] = SourceState(name=name, index=i)

        # Model dimensions
        self.num_states = [len(self.PRODUCTIVITY_LEVELS)]  # 4 productivity levels
        self.num_obs = [
            len(self.OBS_PRODUCTIVITY),  # 3
            len(self.OBS_FRESHNESS),  # 3
            len(self.OBS_ERROR_RATE),  # 3
        ]
        self.num_actions = len(self.SOURCES) + 1  # One per source + wait
        self.num_controls = [self.num_actions]

        # Initialize JAX agent
        self.agent: Optional[Agent] = None
        self.rng_key = None
        self.qs = None
        self.empirical_prior = None

        if JAX_AVAILABLE:
            self.rng_key = jr.PRNGKey(rng_seed)
            self._initialize_agent()
        else:
            logger.warning("Running Scraping POMDP in fallback mode (no JAX)")

        # Action history for logging
        self.action_history: list[dict] = []

    def _initialize_agent(self):
        """Initialize the PyMDP Agent with JAX arrays."""
        if not JAX_AVAILABLE:
            return

        self.rng_key, key = jr.split(self.rng_key)

        # Build A matrix (likelihood): P(observation | state)
        # Shape: [batch_size, num_obs, num_states]
        A = self._build_likelihood_model()

        # Build B matrix (transition): P(state' | state, action)
        # Shape: [batch_size, num_states, num_states, num_actions]
        B = self._build_transition_model()

        # D: Prior over initial states (uniform)
        D = [jnp.ones((self.batch_size, ns)) / ns for ns in self.num_states]

        # pA: Dirichlet priors for A learning
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

            initial_qs = [jnp.ones((self.batch_size, ns)) / ns for ns in self.num_states]
            self.qs = self.agent.infer_states(
                observations=dummy_obs,
                empirical_prior=initial_qs,
            )
            self.empirical_prior = self.qs

            logger.info(f"Initialized Scraping POMDP with {len(self.SOURCES)} sources (JAX/JIT enabled)")

        except Exception as e:
            logger.error(f"Failed to initialize Scraping POMDP agent: {e}")
            self.agent = None

    def _build_likelihood_model(self) -> list:
        """
        Build A matrix mapping states to observations.
        High productivity sources more likely to yield high observations.
        """
        A = []
        ns = self.num_states[0]  # 4 productivity levels

        # Modality 1: Productivity observation
        # High state -> likely high obs, Low state -> likely low obs
        A_prod = np.array([
            [0.7, 0.5, 0.2, 0.1],  # P(high_obs | state)
            [0.2, 0.4, 0.5, 0.3],  # P(med_obs | state)
            [0.1, 0.1, 0.3, 0.6],  # P(low_obs | state)
        ])
        A.append(jnp.array(A_prod)[None, ...].repeat(self.batch_size, axis=0))

        # Modality 2: Freshness observation
        A_fresh = np.array([
            [0.8, 0.5, 0.3, 0.1],  # P(fresh | state)
            [0.15, 0.4, 0.4, 0.3],  # P(moderate | state)
            [0.05, 0.1, 0.3, 0.6],  # P(stale | state)
        ])
        A.append(jnp.array(A_fresh)[None, ...].repeat(self.batch_size, axis=0))

        # Modality 3: Error rate observation
        A_error = np.array([
            [0.8, 0.6, 0.4, 0.2],  # P(low_error | state)
            [0.15, 0.3, 0.4, 0.4],  # P(med_error | state)
            [0.05, 0.1, 0.2, 0.4],  # P(high_error | state)
        ])
        A.append(jnp.array(A_error)[None, ...].repeat(self.batch_size, axis=0))

        return A

    def _build_transition_model(self) -> list:
        """
        Build B matrix for state transitions.
        States decay toward stale without scraping, improve with scraping.
        """
        ns = self.num_states[0]  # 4 states
        na = self.num_actions  # 25 actions (24 sources + wait)

        # B[state_t+1, state_t, action]
        B = np.zeros((ns, ns, na))

        for a in range(na):
            if a == na - 1:  # Wait action
                # States decay toward stale
                B[0, 0, a] = 0.7  # high stays high
                B[1, 0, a] = 0.3  # high -> medium
                B[1, 1, a] = 0.6  # medium stays medium
                B[2, 1, a] = 0.4  # medium -> low
                B[2, 2, a] = 0.5  # low stays low
                B[3, 2, a] = 0.5  # low -> stale
                B[3, 3, a] = 1.0  # stale stays stale
            else:
                # Scraping action - can refresh state
                B[0, :, a] = [0.3, 0.2, 0.1, 0.1]  # Chance to become high
                B[1, :, a] = [0.4, 0.4, 0.3, 0.2]  # Chance to become medium
                B[2, :, a] = [0.2, 0.3, 0.4, 0.3]  # Chance to become low
                B[3, :, a] = [0.1, 0.1, 0.2, 0.4]  # Chance to stay stale

        # Normalize
        B = B / B.sum(axis=0, keepdims=True)

        # Add batch dimension
        return [jnp.array(B)[None, ...].repeat(self.batch_size, axis=0)]

    def observe_scrape_result(
        self,
        source: str,
        items_scraped: int,
        errors: int,
        novelty_ratio: float = 0.5,
    ) -> dict:
        """
        Update beliefs after observing a scrape result.

        Args:
            source: Name of the scraped source
            items_scraped: Number of items obtained
            errors: Number of errors encountered
            novelty_ratio: Ratio of new/unique items (0-1)

        Returns:
            Updated beliefs about the source
        """
        if source not in self.sources:
            logger.warning(f"Unknown source: {source}")
            return {}

        state = self.sources[source]
        state.total_items += items_scraped
        state.total_errors += errors
        state.observation_count += 1
        state.last_scraped = datetime.utcnow()

        # Encode observation
        obs_indices = self._encode_observation(items_scraped, errors, novelty_ratio)

        if self.agent is not None and JAX_AVAILABLE:
            # Convert to one-hot with batch dimension
            obs_one_hot = []
            for m, (obs_idx, no) in enumerate(zip(obs_indices, self.num_obs)):
                one_hot = jnp.zeros((self.batch_size, no))
                one_hot = one_hot.at[:, obs_idx].set(1.0)
                obs_one_hot.append(one_hot)

            try:
                # Run inference
                self.qs = self.agent.infer_states(
                    observations=obs_one_hot,
                    empirical_prior=self.empirical_prior,
                )

                # Extract beliefs
                beliefs = np.array(self.qs[0])
                if beliefs.ndim == 3:
                    beliefs = beliefs[0, -1]
                elif beliefs.ndim == 2:
                    beliefs = beliefs[0]

                # Update source state
                state.productivity_belief = float(beliefs[0] + beliefs[1] * 0.5)
                state.freshness_belief = float(beliefs[0] * 0.9 + beliefs[1] * 0.6)
                state.error_rate_belief = float(errors / max(items_scraped, 1))

                return {
                    "source": source,
                    "beliefs": {
                        self.PRODUCTIVITY_LEVELS[i]: float(beliefs[i])
                        for i in range(len(self.PRODUCTIVITY_LEVELS))
                    },
                    "productivity": state.productivity_belief,
                    "freshness": state.freshness_belief,
                }

            except Exception as e:
                logger.warning(f"JAX inference failed: {e}")

        # Fallback: simple belief update
        state.productivity_belief = 0.8 * state.productivity_belief + 0.2 * (items_scraped / 50)
        state.freshness_belief = 1.0  # Just scraped
        state.error_rate_belief = errors / max(items_scraped + errors, 1)

        return {
            "source": source,
            "productivity": state.productivity_belief,
            "freshness": state.freshness_belief,
            "method": "fallback",
        }

    def _encode_observation(
        self,
        items: int,
        errors: int,
        novelty: float,
    ) -> list[int]:
        """Encode raw observations into discrete indices."""
        # Productivity: based on items
        if items > 30:
            prod_obs = 0  # high
        elif items > 10:
            prod_obs = 1  # medium
        else:
            prod_obs = 2  # low

        # Freshness: based on novelty ratio
        if novelty > 0.7:
            fresh_obs = 0  # fresh
        elif novelty > 0.3:
            fresh_obs = 1  # moderate
        else:
            fresh_obs = 2  # stale

        # Error rate
        error_rate = errors / max(items + errors, 1)
        if error_rate < 0.1:
            error_obs = 0  # low
        elif error_rate < 0.3:
            error_obs = 1  # medium
        else:
            error_obs = 2  # high

        return [prod_obs, fresh_obs, error_obs]

    def select_next_source(self) -> dict:
        """
        Select the next source to scrape using EFE minimization.

        Returns:
            Dict with source name, priority, and reasoning
        """
        if self.agent is not None and JAX_AVAILABLE:
            try:
                # Run policy inference (JIT compiled)
                q_pi, G = self.agent.infer_policies(self.qs)

                # Sample action with batched keys
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

                # Update empirical prior
                self.empirical_prior, self.qs = self.agent.update_empirical_prior(
                    action, self.qs
                )

                # Get EFE values
                efe_values = np.array(G)
                if efe_values.ndim > 1:
                    efe_values = efe_values[0]

                # Determine source name
                if action_idx >= len(self.SOURCES):
                    source_name = "wait"
                    reason = "Low expected information gain from all sources"
                else:
                    source_name = self.SOURCES[action_idx]
                    reason = f"EFE minimization (G={float(efe_values[action_idx]):.3f})"

                result = {
                    "source": source_name,
                    "action_index": action_idx,
                    "priority": 1.0 - float(efe_values[action_idx]) if action_idx < len(efe_values) else 0.0,
                    "reason": reason,
                    "efe_values": {
                        self.SOURCES[i]: float(efe_values[i])
                        for i in range(min(len(self.SOURCES), len(efe_values)))
                    },
                    "method": "pymdp_efe",
                }

                self.action_history.append({
                    "action": source_name,
                    "timestamp": datetime.utcnow().isoformat(),
                    **result,
                })

                return result

            except Exception as e:
                logger.warning(f"JAX policy inference failed: {e}")

        # Fallback: select least recently scraped source with high belief
        return self._fallback_select_source()

    def _fallback_select_source(self) -> dict:
        """Fallback source selection without JAX."""
        now = datetime.utcnow()
        best_source = None
        best_score = -float('inf')

        for name, state in self.sources.items():
            # Calculate staleness
            if state.last_scraped is None:
                staleness = 1.0
            else:
                hours_since = (now - state.last_scraped).total_seconds() / 3600
                staleness = min(1.0, hours_since / 24)

            # Score: productivity * staleness - error_rate
            score = (
                state.productivity_belief * staleness
                - state.error_rate_belief * 0.5
                + self.exploration_bonus * (1.0 / (state.observation_count + 1))
            )

            if score > best_score:
                best_score = score
                best_source = name

        return {
            "source": best_source or self.SOURCES[0],
            "priority": best_score,
            "reason": "Fallback: staleness + productivity heuristic",
            "method": "fallback",
        }

    def get_scraping_schedule(self, budget_minutes: int = 60) -> list[dict]:
        """
        Generate an optimal scraping schedule for a time budget.

        Args:
            budget_minutes: Total time budget in minutes

        Returns:
            Ordered list of sources to scrape
        """
        schedule = []
        time_per_scrape = 2  # Estimated minutes per scrape

        num_scrapes = budget_minutes // time_per_scrape

        for _ in range(num_scrapes):
            next_source = self.select_next_source()
            if next_source["source"] == "wait":
                break
            schedule.append(next_source)

        return schedule

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
            "num_sources": len(self.SOURCES),
            "total_observations": sum(s.observation_count for s in self.sources.values()),
            "free_energy": self.get_free_energy(),
            "sources": {
                name: {
                    "productivity": state.productivity_belief,
                    "freshness": state.freshness_belief,
                    "error_rate": state.error_rate_belief,
                    "observations": state.observation_count,
                    "last_scraped": state.last_scraped.isoformat() if state.last_scraped else None,
                }
                for name, state in self.sources.items()
            },
        }

    def reset(self):
        """Reset POMDP to initial state."""
        for state in self.sources.values():
            state.productivity_belief = 0.5
            state.freshness_belief = 0.5
            state.error_rate_belief = 0.1
            state.last_scraped = None
            state.total_items = 0
            state.total_errors = 0
            state.observation_count = 0

        if JAX_AVAILABLE:
            self._initialize_agent()

        self.action_history = []
        logger.info("Scraping POMDP reset to initial state")


# Singleton instance
_scraping_pomdp: Optional[ScrapingPOMDP] = None


def get_scraping_pomdp() -> ScrapingPOMDP:
    """Get or create the scraping POMDP instance."""
    global _scraping_pomdp
    if _scraping_pomdp is None:
        _scraping_pomdp = ScrapingPOMDP()
    return _scraping_pomdp

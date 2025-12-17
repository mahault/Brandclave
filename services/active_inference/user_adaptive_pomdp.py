"""
User-Adaptive POMDP using JAX/PyMDP.

Learns user preferences and personalizes responses based on interaction history.
Uses active inference to balance exploitation (serving known preferences)
with exploration (discovering new preferences).

All computations are JIT-compiled for performance.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)

# JAX imports with graceful fallback
try:
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    from pymdp.agent import Agent
    JAX_AVAILABLE = True
    logger.info("JAX/PyMDP loaded for User-Adaptive POMDP")
except ImportError as e:
    JAX_AVAILABLE = False
    logger.warning(f"JAX/PyMDP not available for User-Adaptive POMDP: {e}")


@dataclass
class UserInteraction:
    """A single user interaction event."""
    user_id: str
    component: str  # social_pulse, demand_scan, hotelier_bets, city_desires
    action_type: str  # view, click, filter, bookmark, search, dismiss
    item_id: Optional[str] = None
    item_type: Optional[str] = None  # trend, move, property, desire
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class UserPreferenceState:
    """Tracked state for a single user."""
    user_id: str
    interaction_count: int = 0
    component_preferences: dict = field(default_factory=lambda: {
        "social_pulse": 0.25,
        "demand_scan": 0.25,
        "hotelier_bets": 0.25,
        "city_desires": 0.25,
    })
    content_preferences: dict = field(default_factory=lambda: {
        "luxury": 0.25,
        "budget": 0.25,
        "boutique": 0.25,
        "business": 0.25,
    })
    action_patterns: dict = field(default_factory=lambda: {
        "engagement_rate": 0.5,
        "exploration_tendency": 0.5,
        "session_depth": 0.5,
    })
    last_interaction: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


class UserAdaptivePOMDP:
    """
    POMDP controller for user-adaptive personalization.

    Hidden States: User preference profile (explorer, focused, casual, power_user)
    Observations: [component_usage, engagement_level, content_preference, recency]
    Actions: recommend_exploration, recommend_depth, recommend_popular, personalize_feed
    Goal: Maximize user engagement while learning preferences

    All inference is JIT-compiled via PyMDP/JAX.
    """

    # User profile types (hidden states)
    USER_PROFILES = ["explorer", "focused", "casual", "power_user"]

    # Observation modalities
    OBS_COMPONENT = ["single_focus", "multi_component", "broad_usage"]
    OBS_ENGAGEMENT = ["low", "medium", "high"]
    OBS_CONTENT = ["luxury_leaning", "budget_leaning", "balanced", "business_leaning"]
    OBS_RECENCY = ["new_user", "returning", "regular", "power_user"]

    # Personalization actions
    ACTIONS = [
        "recommend_exploration",  # Show diverse content
        "recommend_depth",        # Deep dive on preferred topics
        "recommend_popular",      # Show trending/popular items
        "personalize_feed",       # Strongly personalized based on history
        "introduce_new",          # Introduce new features/content types
    ]

    # Content segments for preference learning
    CONTENT_SEGMENTS = ["luxury", "budget", "boutique", "business"]

    # Components
    COMPONENTS = ["social_pulse", "demand_scan", "hotelier_bets", "city_desires"]

    def __init__(
        self,
        learning_rate: float = 0.1,
        belief_decay_rate: float = 0.01,
        min_interactions_before_personalization: int = 5,
        batch_size: int = 1,
        policy_len: int = 1,
        rng_seed: int = 42,
    ):
        """
        Initialize the User-Adaptive POMDP.

        Args:
            learning_rate: Rate for updating preference beliefs
            belief_decay_rate: How fast old beliefs fade
            min_interactions_before_personalization: Min interactions before personalizing
            batch_size: Batch size for JAX inference
            policy_len: Planning horizon
            rng_seed: Random seed for JAX
        """
        self.learning_rate = learning_rate
        self.belief_decay_rate = belief_decay_rate
        self.min_interactions = min_interactions_before_personalization
        self.batch_size = batch_size
        self.policy_len = policy_len

        # Model dimensions
        self.num_states = [len(self.USER_PROFILES)]  # 4 profiles
        self.num_obs = [
            len(self.OBS_COMPONENT),   # 3
            len(self.OBS_ENGAGEMENT),  # 3
            len(self.OBS_CONTENT),     # 4
            len(self.OBS_RECENCY),     # 4
        ]
        self.num_actions = len(self.ACTIONS)  # 5
        self.num_controls = [self.num_actions]

        # Per-user state tracking
        self.user_states: dict[str, UserPreferenceState] = {}
        self.user_interactions: dict[str, list[UserInteraction]] = defaultdict(list)

        # Global statistics
        self.total_interactions = 0
        self.total_users = 0

        # Initialize JAX agent
        self.agent: Optional[Agent] = None
        self.rng_key = None
        self.qs = None
        self.empirical_prior = None

        if JAX_AVAILABLE:
            self.rng_key = jr.PRNGKey(rng_seed)
            self._initialize_agent()
        else:
            logger.warning("Running User-Adaptive POMDP in fallback mode (no JAX)")

    def _initialize_agent(self):
        """Initialize the PyMDP Agent with JAX arrays."""
        if not JAX_AVAILABLE:
            return

        self.rng_key, key = jr.split(self.rng_key)

        # Build A matrix (likelihood)
        A = self._build_likelihood_model()

        # Build B matrix (transition)
        B = self._build_transition_model()

        # D: Prior over initial states (favor casual)
        D_vals = np.array([0.2, 0.2, 0.4, 0.2])  # Most users start casual
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

            # Initialize beliefs
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

            logger.info(f"Initialized User-Adaptive POMDP with {self.num_actions} actions (JAX/JIT enabled)")

        except Exception as e:
            logger.error(f"Failed to initialize User-Adaptive POMDP agent: {e}")
            self.agent = None

    def _build_likelihood_model(self) -> list:
        """Build A matrix: P(observation | user_profile)."""
        A = []
        ns = self.num_states[0]  # 4 profiles

        # Modality 1: Component usage pattern
        # explorer -> multi, focused -> single, casual -> broad, power -> multi
        A_comp = np.array([
            [0.2, 0.6, 0.4, 0.2],   # single_focus
            [0.5, 0.3, 0.3, 0.5],   # multi_component
            [0.3, 0.1, 0.3, 0.3],   # broad_usage
        ])
        A.append(jnp.array(A_comp)[None, ...].repeat(self.batch_size, axis=0))

        # Modality 2: Engagement level
        # explorer -> medium, focused -> high, casual -> low, power -> high
        A_engage = np.array([
            [0.2, 0.1, 0.5, 0.1],   # low
            [0.5, 0.3, 0.35, 0.3],  # medium
            [0.3, 0.6, 0.15, 0.6],  # high
        ])
        A.append(jnp.array(A_engage)[None, ...].repeat(self.batch_size, axis=0))

        # Modality 3: Content preference
        A_content = np.array([
            [0.25, 0.3, 0.25, 0.2],  # luxury_leaning
            [0.25, 0.2, 0.35, 0.2],  # budget_leaning
            [0.3, 0.2, 0.25, 0.3],   # balanced
            [0.2, 0.3, 0.15, 0.3],   # business_leaning
        ])
        A.append(jnp.array(A_content)[None, ...].repeat(self.batch_size, axis=0))

        # Modality 4: User recency/tenure
        A_recency = np.array([
            [0.3, 0.2, 0.4, 0.1],   # new_user
            [0.35, 0.3, 0.35, 0.2], # returning
            [0.25, 0.35, 0.2, 0.4], # regular
            [0.1, 0.15, 0.05, 0.3], # power_user
        ])
        A.append(jnp.array(A_recency)[None, ...].repeat(self.batch_size, axis=0))

        return A

    def _build_transition_model(self) -> list:
        """Build B matrix: P(profile' | profile, action)."""
        ns = self.num_states[0]
        na = self.num_actions

        # B[state_t+1, state_t, action]
        B = np.zeros((ns, ns, na))

        for a_idx, action in enumerate(self.ACTIONS):
            if action == "recommend_exploration":
                # Encourages explorer behavior
                B[:, :, a_idx] = np.array([
                    [0.5, 0.2, 0.15, 0.2],   # explorer
                    [0.25, 0.5, 0.15, 0.25], # focused
                    [0.15, 0.2, 0.55, 0.2],  # casual
                    [0.1, 0.1, 0.15, 0.35],  # power_user
                ])
            elif action == "recommend_depth":
                # Encourages focused behavior
                B[:, :, a_idx] = np.array([
                    [0.35, 0.15, 0.15, 0.15],
                    [0.35, 0.55, 0.2, 0.3],
                    [0.2, 0.2, 0.5, 0.2],
                    [0.1, 0.1, 0.15, 0.35],
                ])
            elif action == "recommend_popular":
                # Moderate effect, keeps users engaged
                B[:, :, a_idx] = np.array([
                    [0.3, 0.2, 0.2, 0.2],
                    [0.25, 0.4, 0.2, 0.25],
                    [0.3, 0.3, 0.45, 0.3],
                    [0.15, 0.1, 0.15, 0.25],
                ])
            elif action == "personalize_feed":
                # Increases power user tendency for engaged users
                B[:, :, a_idx] = np.array([
                    [0.3, 0.15, 0.15, 0.1],
                    [0.3, 0.45, 0.15, 0.2],
                    [0.2, 0.2, 0.5, 0.2],
                    [0.2, 0.2, 0.2, 0.5],
                ])
            else:  # introduce_new
                # Encourages exploration, might convert casual to explorer
                B[:, :, a_idx] = np.array([
                    [0.4, 0.25, 0.25, 0.2],
                    [0.2, 0.4, 0.15, 0.2],
                    [0.25, 0.25, 0.45, 0.25],
                    [0.15, 0.1, 0.15, 0.35],
                ])

        return [jnp.array(B)[None, ...].repeat(self.batch_size, axis=0)]

    def _get_or_create_user_state(self, user_id: str) -> UserPreferenceState:
        """Get or create user state."""
        if user_id not in self.user_states:
            self.user_states[user_id] = UserPreferenceState(user_id=user_id)
            self.total_users += 1
        return self.user_states[user_id]

    def record_interaction(self, interaction: UserInteraction) -> dict:
        """
        Record a user interaction and update beliefs.

        Args:
            interaction: UserInteraction event

        Returns:
            Updated user state summary
        """
        user_state = self._get_or_create_user_state(interaction.user_id)
        user_state.interaction_count += 1
        user_state.last_interaction = interaction.timestamp
        self.total_interactions += 1

        # Store interaction
        self.user_interactions[interaction.user_id].append(interaction)

        # Keep only recent interactions (last 1000)
        if len(self.user_interactions[interaction.user_id]) > 1000:
            self.user_interactions[interaction.user_id] = \
                self.user_interactions[interaction.user_id][-1000:]

        # Update component preferences
        if interaction.component in user_state.component_preferences:
            for comp in user_state.component_preferences:
                if comp == interaction.component:
                    user_state.component_preferences[comp] = min(
                        1.0,
                        user_state.component_preferences[comp] + self.learning_rate * 0.5
                    )
                else:
                    user_state.component_preferences[comp] *= (1 - self.learning_rate * 0.1)

            # Normalize
            total = sum(user_state.component_preferences.values())
            for comp in user_state.component_preferences:
                user_state.component_preferences[comp] /= total

        # Update engagement patterns based on action type
        engagement_weights = {
            "view": 0.3,
            "click": 0.5,
            "bookmark": 0.8,
            "search": 0.6,
            "filter": 0.7,
            "dismiss": -0.2,
        }
        weight = engagement_weights.get(interaction.action_type, 0.4)
        user_state.action_patterns["engagement_rate"] = (
            user_state.action_patterns["engagement_rate"] * 0.95 +
            (0.5 + weight * 0.5) * 0.05
        )

        # Update content preferences from metadata
        if interaction.metadata.get("content_segment") in self.CONTENT_SEGMENTS:
            segment = interaction.metadata["content_segment"]
            for seg in user_state.content_preferences:
                if seg == segment:
                    user_state.content_preferences[seg] = min(
                        1.0,
                        user_state.content_preferences[seg] + self.learning_rate * 0.3
                    )
                else:
                    user_state.content_preferences[seg] *= (1 - self.learning_rate * 0.05)

            # Normalize
            total = sum(user_state.content_preferences.values())
            for seg in user_state.content_preferences:
                user_state.content_preferences[seg] /= total

        return {
            "user_id": interaction.user_id,
            "interaction_count": user_state.interaction_count,
            "component_preferences": user_state.component_preferences,
            "content_preferences": user_state.content_preferences,
            "engagement_rate": user_state.action_patterns["engagement_rate"],
        }

    def _encode_user_observations(self, user_id: str) -> list[int]:
        """Encode user state into observation indices."""
        user_state = self._get_or_create_user_state(user_id)
        interactions = self.user_interactions.get(user_id, [])
        recent_interactions = interactions[-50:] if interactions else []

        obs_indices = []

        # Component usage pattern
        if recent_interactions:
            components_used = set(i.component for i in recent_interactions)
            if len(components_used) == 1:
                obs_indices.append(0)  # single_focus
            elif len(components_used) >= 3:
                obs_indices.append(1)  # multi_component
            else:
                obs_indices.append(2)  # broad_usage
        else:
            obs_indices.append(2)  # default broad

        # Engagement level
        engagement = user_state.action_patterns["engagement_rate"]
        if engagement < 0.35:
            obs_indices.append(0)  # low
        elif engagement < 0.65:
            obs_indices.append(1)  # medium
        else:
            obs_indices.append(2)  # high

        # Content preference
        prefs = user_state.content_preferences
        max_pref = max(prefs.values())
        if max_pref < 0.35:  # All similar
            obs_indices.append(2)  # balanced
        elif prefs["luxury"] == max_pref:
            obs_indices.append(0)  # luxury_leaning
        elif prefs["budget"] == max_pref:
            obs_indices.append(1)  # budget_leaning
        elif prefs["business"] == max_pref:
            obs_indices.append(3)  # business_leaning
        else:
            obs_indices.append(2)  # balanced

        # User recency/tenure
        count = user_state.interaction_count
        if count < 5:
            obs_indices.append(0)  # new_user
        elif count < 20:
            obs_indices.append(1)  # returning
        elif count < 100:
            obs_indices.append(2)  # regular
        else:
            obs_indices.append(3)  # power_user

        return obs_indices

    def get_recommendation(self, user_id: str) -> dict:
        """
        Get personalized recommendation action for a user.

        Args:
            user_id: User identifier

        Returns:
            Dict with recommendation action and metadata
        """
        user_state = self._get_or_create_user_state(user_id)

        # Check if we have enough data for personalization
        if user_state.interaction_count < self.min_interactions:
            return {
                "action": "recommend_exploration",
                "reason": "Not enough interactions for personalization",
                "personalized": False,
                "interaction_count": user_state.interaction_count,
                "min_required": self.min_interactions,
            }

        # Encode observations
        obs_indices = self._encode_user_observations(user_id)

        if self.agent is not None and JAX_AVAILABLE:
            try:
                # Convert to one-hot
                obs_one_hot = []
                for obs_idx, no in zip(obs_indices, self.num_obs):
                    one_hot = jnp.zeros((self.batch_size, no))
                    one_hot = one_hot.at[:, obs_idx].set(1.0)
                    obs_one_hot.append(one_hot)

                # Update beliefs
                self.qs = self.agent.infer_states(
                    observations=obs_one_hot,
                    empirical_prior=self.empirical_prior,
                )

                # Run policy inference
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

                action_idx = min(action_idx, len(self.ACTIONS) - 1)
                selected_action = self.ACTIONS[action_idx]

                # Get beliefs for reporting
                beliefs = np.array(self.qs[0])
                if beliefs.ndim == 3:
                    beliefs = beliefs[0, -1]
                elif beliefs.ndim == 2:
                    beliefs = beliefs[0]

                # Get EFE values
                efe_values = np.array(G)
                if efe_values.ndim > 1:
                    efe_values = efe_values[0]

                return {
                    "action": selected_action,
                    "action_index": action_idx,
                    "personalized": True,
                    "user_profile_beliefs": {
                        self.USER_PROFILES[i]: float(beliefs[i])
                        for i in range(len(self.USER_PROFILES))
                    },
                    "efe": float(efe_values[action_idx]) if action_idx < len(efe_values) else 0.0,
                    "component_preferences": user_state.component_preferences,
                    "content_preferences": user_state.content_preferences,
                    "interaction_count": user_state.interaction_count,
                    "method": "pymdp_efe",
                }

            except Exception as e:
                logger.warning(f"JAX recommendation failed: {e}")

        # Fallback recommendation
        return self._fallback_recommendation(user_id)

    def _fallback_recommendation(self, user_id: str) -> dict:
        """Fallback recommendation without JAX."""
        user_state = self._get_or_create_user_state(user_id)

        # Simple heuristic based on engagement
        engagement = user_state.action_patterns["engagement_rate"]

        if engagement < 0.3:
            action = "recommend_popular"  # Hook low-engagement users
        elif engagement > 0.7:
            action = "personalize_feed"   # Serve high-engagement users
        elif user_state.interaction_count < 20:
            action = "recommend_exploration"  # Help new users explore
        else:
            action = "recommend_depth"    # Medium users get depth

        return {
            "action": action,
            "personalized": True,
            "component_preferences": user_state.component_preferences,
            "content_preferences": user_state.content_preferences,
            "interaction_count": user_state.interaction_count,
            "method": "fallback",
        }

    def get_personalized_content_weights(self, user_id: str) -> dict:
        """
        Get content ranking weights for a user.

        Args:
            user_id: User identifier

        Returns:
            Dict with weights for different content types
        """
        user_state = self._get_or_create_user_state(user_id)

        if user_state.interaction_count < self.min_interactions:
            # Return uniform weights for new users
            return {
                "component_weights": {c: 0.25 for c in self.COMPONENTS},
                "content_weights": {s: 0.25 for s in self.CONTENT_SEGMENTS},
                "personalized": False,
            }

        return {
            "component_weights": user_state.component_preferences.copy(),
            "content_weights": user_state.content_preferences.copy(),
            "engagement_rate": user_state.action_patterns["engagement_rate"],
            "personalized": True,
        }

    def decay_beliefs(self) -> int:
        """
        Apply belief decay to all users (run periodically).

        Returns:
            Number of users processed
        """
        now = datetime.utcnow()
        decayed = 0

        for user_id, user_state in self.user_states.items():
            if user_state.last_interaction is None:
                continue

            hours_since = (now - user_state.last_interaction).total_seconds() / 3600

            if hours_since > 24:  # Only decay if inactive > 24h
                decay = self.belief_decay_rate * (hours_since / 24)

                # Decay preferences toward uniform
                for prefs in [user_state.component_preferences, user_state.content_preferences]:
                    uniform = 1.0 / len(prefs)
                    for key in prefs:
                        prefs[key] = prefs[key] * (1 - decay) + uniform * decay

                decayed += 1

        return decayed

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

    def get_user_status(self, user_id: str) -> dict:
        """Get status for a specific user."""
        if user_id not in self.user_states:
            return {"exists": False}

        user_state = self.user_states[user_id]
        return {
            "exists": True,
            "user_id": user_id,
            "interaction_count": user_state.interaction_count,
            "component_preferences": user_state.component_preferences,
            "content_preferences": user_state.content_preferences,
            "action_patterns": user_state.action_patterns,
            "last_interaction": user_state.last_interaction.isoformat() if user_state.last_interaction else None,
            "created_at": user_state.created_at.isoformat(),
        }

    def get_status(self) -> dict:
        """Get current POMDP status."""
        return {
            "enabled": True,
            "jax_available": JAX_AVAILABLE,
            "agent_initialized": self.agent is not None,
            "num_actions": self.num_actions,
            "total_users": self.total_users,
            "total_interactions": self.total_interactions,
            "min_interactions_for_personalization": self.min_interactions,
            "free_energy": self.get_free_energy(),
        }

    def reset(self):
        """Reset POMDP to initial state."""
        self.user_states = {}
        self.user_interactions = defaultdict(list)
        self.total_interactions = 0
        self.total_users = 0

        if JAX_AVAILABLE:
            self._initialize_agent()

        logger.info("User-Adaptive POMDP reset to initial state")


# Singleton instance
_user_adaptive_pomdp: Optional[UserAdaptivePOMDP] = None


def get_user_adaptive_pomdp() -> UserAdaptivePOMDP:
    """Get or create the user-adaptive POMDP instance."""
    global _user_adaptive_pomdp
    if _user_adaptive_pomdp is None:
        _user_adaptive_pomdp = UserAdaptivePOMDP()
    return _user_adaptive_pomdp

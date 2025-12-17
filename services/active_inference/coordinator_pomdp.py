"""
Cross-Component Coordinator POMDP using JAX/PyMDP.

Orchestrates information flow across Scraping, Clustering, and Extraction POMDPs.
Detects cross-component correlations and surfaces optimization opportunities.
All computations are JIT-compiled for performance.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

# JAX imports with graceful fallback
try:
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    from pymdp.agent import Agent
    JAX_AVAILABLE = True
    logger.info("JAX/PyMDP loaded for Coordinator POMDP")
except ImportError as e:
    JAX_AVAILABLE = False
    logger.warning(f"JAX/PyMDP not available for Coordinator POMDP: {e}")


@dataclass
class ComponentSignal:
    """A signal from one of the component POMDPs."""
    component: str  # scraping, clustering, extraction
    signal_type: str  # productivity_spike, quality_change, cost_savings, etc.
    value: float
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CrossComponentOpportunity:
    """An optimization opportunity detected across components."""
    opportunity_type: str
    source_component: str
    target_component: str
    description: str
    priority: float
    action_suggestion: str
    detected_at: datetime = field(default_factory=datetime.utcnow)


class CoordinatorPOMDP:
    """
    POMDP controller for cross-component coordination.

    Hidden States: System optimization level (suboptimal, baseline, good, optimal)
    Observations: [scraping_signal, clustering_signal, extraction_signal, correlation_strength]
    Actions: focus_scraping, focus_clustering, focus_extraction, balance_all, investigate
    Goal: Maximize overall system efficiency by coordinating component behaviors

    All inference is JIT-compiled via PyMDP/JAX.
    """

    # System optimization levels (hidden states)
    OPTIMIZATION_LEVELS = ["suboptimal", "baseline", "good", "optimal"]

    # Observation modalities
    OBS_SCRAPING = ["low", "medium", "high"]  # Scraping productivity
    OBS_CLUSTERING = ["poor", "fair", "good"]  # Clustering quality
    OBS_EXTRACTION = ["low", "medium", "high"]  # Extraction success rate
    OBS_CORRELATION = ["none", "weak", "strong"]  # Cross-component correlation

    # Coordination actions
    ACTIONS = [
        "focus_scraping",  # Prioritize scraping optimization
        "focus_clustering",  # Prioritize clustering parameter tuning
        "focus_extraction",  # Prioritize extraction method selection
        "balance_all",  # Equal attention to all components
        "investigate",  # Gather more information before acting
    ]

    def __init__(
        self,
        learning_rate: float = 0.05,
        cross_component_threshold: float = 0.7,
        update_interval_seconds: int = 300,
        signal_window_size: int = 100,
        batch_size: int = 1,
        policy_len: int = 1,
        rng_seed: int = 42,
    ):
        """
        Initialize the Coordinator POMDP.

        Args:
            learning_rate: Rate for updating beliefs
            cross_component_threshold: Min correlation to surface as opportunity
            update_interval_seconds: How often to run coordination
            signal_window_size: Number of signals to keep in history
            batch_size: Batch size for JAX inference
            policy_len: Planning horizon
            rng_seed: Random seed for JAX
        """
        self.learning_rate = learning_rate
        self.cross_component_threshold = cross_component_threshold
        self.update_interval_seconds = update_interval_seconds
        self.signal_window_size = signal_window_size
        self.batch_size = batch_size
        self.policy_len = policy_len

        # Model dimensions
        self.num_states = [len(self.OPTIMIZATION_LEVELS)]  # 4 levels
        self.num_obs = [
            len(self.OBS_SCRAPING),  # 3
            len(self.OBS_CLUSTERING),  # 3
            len(self.OBS_EXTRACTION),  # 3
            len(self.OBS_CORRELATION),  # 3
        ]
        self.num_actions = len(self.ACTIONS)  # 5
        self.num_controls = [self.num_actions]

        # Signal history per component
        self.signal_history: dict[str, deque] = {
            "scraping": deque(maxlen=signal_window_size),
            "clustering": deque(maxlen=signal_window_size),
            "extraction": deque(maxlen=signal_window_size),
        }

        # Cross-component correlation cache
        self.correlations: dict[tuple[str, str], float] = {}

        # Detected opportunities
        self.opportunities: list[CrossComponentOpportunity] = []
        self.opportunities_surfaced = 0

        # Component focus weights (learned)
        self.focus_weights = {
            "scraping": 0.33,
            "clustering": 0.33,
            "extraction": 0.34,
        }

        # Last coordination time
        self.last_coordination: Optional[datetime] = None

        # Initialize JAX agent
        self.agent: Optional[Agent] = None
        self.rng_key = None
        self.qs = None
        self.empirical_prior = None

        if JAX_AVAILABLE:
            self.rng_key = jr.PRNGKey(rng_seed)
            self._initialize_agent()
        else:
            logger.warning("Running Coordinator POMDP in fallback mode (no JAX)")

    def _initialize_agent(self):
        """Initialize the PyMDP Agent with JAX arrays."""
        if not JAX_AVAILABLE:
            return

        self.rng_key, key = jr.split(self.rng_key)

        # Build A matrix (likelihood)
        A = self._build_likelihood_model()

        # Build B matrix (transition)
        B = self._build_transition_model()

        # D: Prior over initial states (favor baseline)
        D_vals = np.array([0.2, 0.4, 0.3, 0.1])
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
                one_hot = one_hot.at[:, 1].set(1.0)  # Start with medium observations
                dummy_obs.append(one_hot)

            initial_qs = [jnp.array(D_vals)[None, :].repeat(self.batch_size, axis=0)]
            self.qs = self.agent.infer_states(
                observations=dummy_obs,
                empirical_prior=initial_qs,
            )
            self.empirical_prior = self.qs

            logger.info(f"Initialized Coordinator POMDP with {self.num_actions} actions (JAX/JIT enabled)")

        except Exception as e:
            logger.error(f"Failed to initialize Coordinator POMDP agent: {e}")
            self.agent = None

    def _build_likelihood_model(self) -> list:
        """Build A matrix: P(observation | optimization_level)."""
        A = []
        ns = self.num_states[0]  # 4 optimization levels

        # Modality 1: Scraping productivity observation
        # Optimal system -> likely high scraping productivity
        A_scraping = np.array([
            [0.6, 0.3, 0.15, 0.05],  # low
            [0.3, 0.4, 0.4, 0.3],    # medium
            [0.1, 0.3, 0.45, 0.65],  # high
        ])
        A.append(jnp.array(A_scraping)[None, ...].repeat(self.batch_size, axis=0))

        # Modality 2: Clustering quality observation
        A_clustering = np.array([
            [0.6, 0.3, 0.15, 0.05],  # poor
            [0.3, 0.4, 0.4, 0.3],    # fair
            [0.1, 0.3, 0.45, 0.65],  # good
        ])
        A.append(jnp.array(A_clustering)[None, ...].repeat(self.batch_size, axis=0))

        # Modality 3: Extraction success observation
        A_extraction = np.array([
            [0.6, 0.3, 0.15, 0.05],  # low
            [0.3, 0.4, 0.4, 0.3],    # medium
            [0.1, 0.3, 0.45, 0.65],  # high
        ])
        A.append(jnp.array(A_extraction)[None, ...].repeat(self.batch_size, axis=0))

        # Modality 4: Cross-component correlation observation
        # Optimal systems have strong positive correlations
        A_correlation = np.array([
            [0.5, 0.3, 0.2, 0.1],   # none
            [0.35, 0.4, 0.35, 0.3],  # weak
            [0.15, 0.3, 0.45, 0.6],  # strong
        ])
        A.append(jnp.array(A_correlation)[None, ...].repeat(self.batch_size, axis=0))

        return A

    def _build_transition_model(self) -> list:
        """Build B matrix: P(optimization' | optimization, action)."""
        ns = self.num_states[0]
        na = self.num_actions

        # B[state_t+1, state_t, action]
        B = np.zeros((ns, ns, na))

        for a_idx, action in enumerate(self.ACTIONS):
            if action == "focus_scraping":
                # Can improve if scraping was the bottleneck
                B[:, :, a_idx] = np.array([
                    [0.5, 0.1, 0.05, 0.05],
                    [0.3, 0.5, 0.2, 0.1],
                    [0.15, 0.3, 0.5, 0.3],
                    [0.05, 0.1, 0.25, 0.55],
                ])
            elif action == "focus_clustering":
                # Similar pattern for clustering
                B[:, :, a_idx] = np.array([
                    [0.5, 0.1, 0.05, 0.05],
                    [0.3, 0.5, 0.2, 0.1],
                    [0.15, 0.3, 0.5, 0.3],
                    [0.05, 0.1, 0.25, 0.55],
                ])
            elif action == "focus_extraction":
                # Similar pattern for extraction
                B[:, :, a_idx] = np.array([
                    [0.5, 0.1, 0.05, 0.05],
                    [0.3, 0.5, 0.2, 0.1],
                    [0.15, 0.3, 0.5, 0.3],
                    [0.05, 0.1, 0.25, 0.55],
                ])
            elif action == "balance_all":
                # Balanced approach - steady improvement
                B[:, :, a_idx] = np.array([
                    [0.4, 0.1, 0.05, 0.02],
                    [0.35, 0.45, 0.15, 0.08],
                    [0.2, 0.35, 0.5, 0.3],
                    [0.05, 0.1, 0.3, 0.6],
                ])
            else:  # investigate
                # Mostly stays same but might reveal better state
                B[:, :, a_idx] = np.array([
                    [0.6, 0.15, 0.05, 0.02],
                    [0.25, 0.55, 0.2, 0.08],
                    [0.1, 0.2, 0.55, 0.25],
                    [0.05, 0.1, 0.2, 0.65],
                ])

        return [jnp.array(B)[None, ...].repeat(self.batch_size, axis=0)]

    def receive_signal(self, signal: ComponentSignal) -> None:
        """
        Receive a signal from a component POMDP.

        Args:
            signal: ComponentSignal from scraping, clustering, or extraction
        """
        if signal.component in self.signal_history:
            self.signal_history[signal.component].append(signal)
            logger.debug(f"Received {signal.signal_type} signal from {signal.component}: {signal.value:.3f}")

    def record_scraping_result(
        self,
        source: str,
        items: int,
        errors: int,
        novelty: float,
    ) -> None:
        """Record a scraping result as a signal."""
        productivity = items / max(items + errors * 5, 1)  # Penalize errors
        signal = ComponentSignal(
            component="scraping",
            signal_type="productivity",
            value=productivity,
            metadata={"source": source, "items": items, "errors": errors, "novelty": novelty},
        )
        self.receive_signal(signal)

    def record_clustering_result(
        self,
        silhouette: float,
        num_clusters: int,
        noise_ratio: float,
    ) -> None:
        """Record a clustering result as a signal."""
        # Quality score: high silhouette, low noise
        quality = max(0, (silhouette + 1) / 2) * (1 - noise_ratio)
        signal = ComponentSignal(
            component="clustering",
            signal_type="quality",
            value=quality,
            metadata={"silhouette": silhouette, "clusters": num_clusters, "noise": noise_ratio},
        )
        self.receive_signal(signal)

    def record_extraction_result(
        self,
        method: str,
        quality: float,
        cost: float,
        success: bool,
    ) -> None:
        """Record an extraction result as a signal."""
        # Efficiency: quality vs cost
        efficiency = quality / max(cost, 0.1) if success else 0
        signal = ComponentSignal(
            component="extraction",
            signal_type="efficiency",
            value=efficiency,
            metadata={"method": method, "quality": quality, "cost": cost, "success": success},
        )
        self.receive_signal(signal)

    def _compute_correlations(self) -> dict[tuple[str, str], float]:
        """Compute pairwise correlations between component signals."""
        correlations = {}
        components = list(self.signal_history.keys())

        for i, comp1 in enumerate(components):
            for comp2 in components[i + 1:]:
                signals1 = list(self.signal_history[comp1])
                signals2 = list(self.signal_history[comp2])

                if len(signals1) < 10 or len(signals2) < 10:
                    correlations[(comp1, comp2)] = 0.0
                    continue

                # Get recent values
                values1 = np.array([s.value for s in signals1[-50:]])
                values2 = np.array([s.value for s in signals2[-50:]])

                # Align lengths
                min_len = min(len(values1), len(values2))
                if min_len < 5:
                    correlations[(comp1, comp2)] = 0.0
                    continue

                values1 = values1[-min_len:]
                values2 = values2[-min_len:]

                # Compute correlation
                if np.std(values1) > 0 and np.std(values2) > 0:
                    corr = np.corrcoef(values1, values2)[0, 1]
                    correlations[(comp1, comp2)] = float(corr) if not np.isnan(corr) else 0.0
                else:
                    correlations[(comp1, comp2)] = 0.0

        self.correlations = correlations
        return correlations

    def _encode_observations(self) -> list[int]:
        """Encode current state into observation indices."""
        obs_indices = []

        # Scraping productivity (recent average)
        scraping_signals = list(self.signal_history["scraping"])[-20:]
        if scraping_signals:
            avg_scraping = np.mean([s.value for s in scraping_signals])
            if avg_scraping < 0.3:
                obs_indices.append(0)  # low
            elif avg_scraping < 0.6:
                obs_indices.append(1)  # medium
            else:
                obs_indices.append(2)  # high
        else:
            obs_indices.append(1)  # default medium

        # Clustering quality
        clustering_signals = list(self.signal_history["clustering"])[-20:]
        if clustering_signals:
            avg_clustering = np.mean([s.value for s in clustering_signals])
            if avg_clustering < 0.3:
                obs_indices.append(0)  # poor
            elif avg_clustering < 0.6:
                obs_indices.append(1)  # fair
            else:
                obs_indices.append(2)  # good
        else:
            obs_indices.append(1)  # default fair

        # Extraction success
        extraction_signals = list(self.signal_history["extraction"])[-20:]
        if extraction_signals:
            avg_extraction = np.mean([s.value for s in extraction_signals])
            if avg_extraction < 0.3:
                obs_indices.append(0)  # low
            elif avg_extraction < 0.6:
                obs_indices.append(1)  # medium
            else:
                obs_indices.append(2)  # high
        else:
            obs_indices.append(1)  # default medium

        # Correlation strength
        self._compute_correlations()
        if self.correlations:
            max_corr = max(abs(c) for c in self.correlations.values())
            if max_corr < 0.3:
                obs_indices.append(0)  # none
            elif max_corr < 0.6:
                obs_indices.append(1)  # weak
            else:
                obs_indices.append(2)  # strong
        else:
            obs_indices.append(0)  # default none

        return obs_indices

    def coordinate(self) -> dict:
        """
        Run coordination logic to determine focus and detect opportunities.

        Returns:
            Dict with coordination decision and any detected opportunities
        """
        now = datetime.utcnow()

        # Check if enough time has passed
        if self.last_coordination is not None:
            elapsed = (now - self.last_coordination).total_seconds()
            if elapsed < self.update_interval_seconds:
                return {
                    "action": "wait",
                    "reason": f"Last coordination was {elapsed:.0f}s ago",
                    "next_in": self.update_interval_seconds - elapsed,
                }

        self.last_coordination = now

        # Encode current observations
        obs_indices = self._encode_observations()

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

                # Update focus weights based on action
                self._update_focus_weights(selected_action)

                # Detect opportunities
                opportunities = self._detect_opportunities()

                # Get beliefs for reporting
                beliefs = np.array(self.qs[0])
                if beliefs.ndim == 3:
                    beliefs = beliefs[0, -1]
                elif beliefs.ndim == 2:
                    beliefs = beliefs[0]

                return {
                    "action": selected_action,
                    "action_index": action_idx,
                    "beliefs": {
                        self.OPTIMIZATION_LEVELS[i]: float(beliefs[i])
                        for i in range(len(self.OPTIMIZATION_LEVELS))
                    },
                    "focus_weights": self.focus_weights.copy(),
                    "correlations": self.correlations.copy(),
                    "opportunities": opportunities,
                    "method": "pymdp_efe",
                }

            except Exception as e:
                logger.warning(f"JAX coordination failed: {e}")

        # Fallback coordination
        return self._fallback_coordinate()

    def _fallback_coordinate(self) -> dict:
        """Fallback coordination without JAX."""
        # Simple heuristic: focus on component with lowest recent performance
        component_scores = {}

        for component, signals in self.signal_history.items():
            recent = list(signals)[-20:]
            if recent:
                component_scores[component] = np.mean([s.value for s in recent])
            else:
                component_scores[component] = 0.5

        # Focus on weakest component
        weakest = min(component_scores, key=component_scores.get)
        action = f"focus_{weakest}"

        self._update_focus_weights(action)
        opportunities = self._detect_opportunities()

        return {
            "action": action,
            "component_scores": component_scores,
            "focus_weights": self.focus_weights.copy(),
            "opportunities": opportunities,
            "method": "fallback",
        }

    def _update_focus_weights(self, action: str) -> None:
        """Update focus weights based on selected action."""
        decay = 0.9

        if action == "focus_scraping":
            self.focus_weights["scraping"] = min(0.5, self.focus_weights["scraping"] * decay + 0.1)
            self.focus_weights["clustering"] *= decay
            self.focus_weights["extraction"] *= decay
        elif action == "focus_clustering":
            self.focus_weights["clustering"] = min(0.5, self.focus_weights["clustering"] * decay + 0.1)
            self.focus_weights["scraping"] *= decay
            self.focus_weights["extraction"] *= decay
        elif action == "focus_extraction":
            self.focus_weights["extraction"] = min(0.5, self.focus_weights["extraction"] * decay + 0.1)
            self.focus_weights["scraping"] *= decay
            self.focus_weights["clustering"] *= decay
        elif action == "balance_all":
            self.focus_weights["scraping"] = 0.33
            self.focus_weights["clustering"] = 0.33
            self.focus_weights["extraction"] = 0.34

        # Normalize
        total = sum(self.focus_weights.values())
        for k in self.focus_weights:
            self.focus_weights[k] /= total

    def _detect_opportunities(self) -> list[dict]:
        """Detect cross-component optimization opportunities."""
        opportunities = []

        # Check for strong correlations
        for (comp1, comp2), corr in self.correlations.items():
            if abs(corr) >= self.cross_component_threshold:
                if corr > 0:
                    opp = CrossComponentOpportunity(
                        opportunity_type="positive_correlation",
                        source_component=comp1,
                        target_component=comp2,
                        description=f"Strong positive correlation ({corr:.2f}) between {comp1} and {comp2}",
                        priority=abs(corr),
                        action_suggestion=f"Improvements in {comp1} likely benefit {comp2}",
                    )
                else:
                    opp = CrossComponentOpportunity(
                        opportunity_type="negative_correlation",
                        source_component=comp1,
                        target_component=comp2,
                        description=f"Negative correlation ({corr:.2f}) between {comp1} and {comp2}",
                        priority=abs(corr),
                        action_suggestion=f"Check for resource contention between {comp1} and {comp2}",
                    )

                self.opportunities.append(opp)
                self.opportunities_surfaced += 1
                opportunities.append({
                    "type": opp.opportunity_type,
                    "source": opp.source_component,
                    "target": opp.target_component,
                    "description": opp.description,
                    "priority": opp.priority,
                    "suggestion": opp.action_suggestion,
                })

        # Check for performance anomalies
        for component, signals in self.signal_history.items():
            recent = list(signals)[-10:]
            older = list(signals)[-30:-10]

            if len(recent) >= 5 and len(older) >= 10:
                recent_avg = np.mean([s.value for s in recent])
                older_avg = np.mean([s.value for s in older])

                # Significant change detection
                if older_avg > 0 and recent_avg > older_avg * 1.3:  # 30% improvement
                    opp = CrossComponentOpportunity(
                        opportunity_type="performance_spike",
                        source_component=component,
                        target_component="all",
                        description=f"{component} showing {(recent_avg/older_avg - 1)*100:.0f}% improvement",
                        priority=0.8,
                        action_suggestion=f"Investigate and propagate {component} improvements",
                    )
                    self.opportunities.append(opp)
                    self.opportunities_surfaced += 1
                    opportunities.append({
                        "type": opp.opportunity_type,
                        "component": opp.source_component,
                        "description": opp.description,
                        "priority": opp.priority,
                        "suggestion": opp.action_suggestion,
                    })
                elif older_avg > 0 and recent_avg < older_avg * 0.7:  # 30% degradation
                    opp = CrossComponentOpportunity(
                        opportunity_type="performance_drop",
                        source_component=component,
                        target_component="all",
                        description=f"{component} showing {(1 - recent_avg/older_avg)*100:.0f}% degradation",
                        priority=0.9,
                        action_suggestion=f"Investigate {component} for issues",
                    )
                    self.opportunities.append(opp)
                    self.opportunities_surfaced += 1
                    opportunities.append({
                        "type": opp.opportunity_type,
                        "component": opp.source_component,
                        "description": opp.description,
                        "priority": opp.priority,
                        "suggestion": opp.action_suggestion,
                    })

        return opportunities

    def get_focus_recommendation(self, component: str) -> float:
        """
        Get the recommended focus weight for a component.

        Args:
            component: scraping, clustering, or extraction

        Returns:
            Focus weight (0-1)
        """
        return self.focus_weights.get(component, 0.33)

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
            "signal_counts": {
                comp: len(signals)
                for comp, signals in self.signal_history.items()
            },
            "focus_weights": self.focus_weights.copy(),
            "correlations": self.correlations.copy(),
            "opportunities_surfaced": self.opportunities_surfaced,
            "free_energy": self.get_free_energy(),
            "last_coordination": self.last_coordination.isoformat() if self.last_coordination else None,
        }

    def reset(self):
        """Reset POMDP to initial state."""
        for signals in self.signal_history.values():
            signals.clear()

        self.correlations = {}
        self.opportunities = []
        self.opportunities_surfaced = 0
        self.focus_weights = {
            "scraping": 0.33,
            "clustering": 0.33,
            "extraction": 0.34,
        }
        self.last_coordination = None

        if JAX_AVAILABLE:
            self._initialize_agent()

        logger.info("Coordinator POMDP reset to initial state")


# Singleton instance
_coordinator_pomdp: Optional[CoordinatorPOMDP] = None


def get_coordinator_pomdp() -> CoordinatorPOMDP:
    """Get or create the coordinator POMDP instance."""
    global _coordinator_pomdp
    if _coordinator_pomdp is None:
        _coordinator_pomdp = CoordinatorPOMDP()
    return _coordinator_pomdp

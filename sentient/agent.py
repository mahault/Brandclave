"""SentientAgent: infer the state of a space, recommend one action or nothing, and say why.

Copilot, not autopilot. Every recommendation carries the inferred state, a reason, a
confidence, and a prediction about a specific observation at a fixed horizon. The caller
decides what was actually applied and tells the agent (commit), and whether the person
accepted or dismissed it (feedback). Dismissals raise the bar for that action.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from .model import (
    A_DEPENDENCIES,
    ACTION_TARGET,
    ACTIONS,
    AROUSAL,
    DO_NOTHING,
    INTENT_KEY_MODALITY,
    MODALITIES,
    MODALITY_NAMES,
    OCCUPANCY,
    Reading,
    build_A,
    build_B,
    build_C,
    build_D,
    build_E,
    discretize,
    expected_observation,
    propagate,
)

logger = logging.getLogger(__name__)

try:
    import jax.numpy as jnp
    from pymdp.agent import Agent

    PYMDP_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - environment without pymdp
    PYMDP_AVAILABLE = False
    logger.warning("pymdp unavailable (%s); SentientAgent will not run", exc)


@dataclass
class Prediction:
    modality: str
    predicted: int
    horizon: int
    probability: float = 0.0    # model's probability of the predicted category

    @property
    def predicted_label(self) -> str:
        return MODALITIES[self.modality][self.predicted]


@dataclass
class Recommendation:
    action: str                 # one of ACTIONS, or "setback"
    reason: str
    confidence: float           # posterior probability of the chosen policy
    beliefs: dict[str, dict[str, float]]
    observation: dict[str, int]
    prediction: Prediction
    q_pi: list[float] = field(default_factory=list)
    gated: bool = False         # True when a better-scoring action was withheld
    masked: list[str] = field(default_factory=list)
    # optional: a forecast for every channel, {modality: (predicted, probability)}
    forecasts: dict[str, tuple[int, float]] = field(default_factory=dict)

    @property
    def is_action(self) -> bool:
        return self.action != "do_nothing"


class SentientAgent:
    def __init__(self, intent: str, margin: float = 0.08, horizon: int = 3, seed: int = 0, gamma: float = 4.0):
        if not PYMDP_AVAILABLE:
            raise RuntimeError("pymdp is required for SentientAgent")
        self.intent = intent
        self.horizon = horizon
        self.gamma = gamma
        self.margins = {a: margin for a in ACTIONS}
        self.A = build_A()
        self.B = build_B()
        self.C = build_C(intent)
        self.D = build_D()
        self.E = build_E()

        n_a = len(ACTIONS)
        policies = jnp.array([[[a, a]] for a in range(n_a)])
        self.agent = Agent(
            A=[jnp.array(a)[None, ...] for a in self.A],
            B=[jnp.array(b)[None, ...] for b in self.B],
            C=[jnp.array(c)[None, :] for c in self.C],
            D=[jnp.array(d)[None, :] for d in self.D],
            E=jnp.array(self.E),
            A_dependencies=A_DEPENDENCIES,
            num_controls=[n_a, n_a],
            policies=policies,
            policy_len=1,
            batch_size=1,
            inference_algo="fpi",
            use_utility=True,
            use_states_info_gain=True,
            use_param_info_gain=False,
            action_selection="deterministic",
            gamma=gamma,
        )
        self.empirical_prior = [jnp.array(d)[None, :] for d in self.D]
        self._qs = None
        self._empty_streak = 0
        self.setback_streak_required = 2  # rises when a setback is dismissed: this radar misses people
        self.last: Recommendation | None = None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _beliefs_from_qs(qs) -> list[np.ndarray]:
        out = []
        for q in qs:
            arr = np.asarray(q)
            while arr.ndim > 1:
                arr = arr[-1] if arr.shape[0] > 1 else arr[0]
            out.append(arr / arr.sum())
        return out

    def _target_modality(self, action: str, reading: Reading) -> str:
        target = ACTION_TARGET.get(action)
        if action == "cool_air" and reading.hr is None:
            target = "motion"
        return target or INTENT_KEY_MODALITY[self.intent]

    def _reason(self, action: str, beliefs: list[np.ndarray], pred: Prediction, gated: bool, masked: list[str]) -> str:
        occ = OCCUPANCY[int(np.argmax(beliefs[0]))]
        ar = AROUSAL[int(np.argmax(beliefs[1]))]
        state = f"occupancy likely {occ} ({beliefs[0].max():.2f}), arousal likely {ar} ({beliefs[1].max():.2f})"
        if action == "setback":
            return f"{state}; space confidently empty, set lights and cooling back"
        if action == "do_nothing":
            why = "state uncertain, holding" if gated else "current state matches intent"
            if masked and len(masked) >= 3:
                why = "no environmental lever left to pull" + (", state uncertain" if gated else "")
            return f"{state}; intent {self.intent}; {why}"
        return (
            f"{state}; intent {self.intent}; {action} expected to move {pred.modality} "
            f"to {pred.predicted_label} within {pred.horizon} steps"
        )

    # --------------------------------------------------------------------- loop
    def step(self, reading: Reading, levers: dict[str, bool] | None = None) -> Recommendation:
        obs_idx = discretize(reading)
        # pymdp (categorical_obs=False) takes integer indices per modality, batch-shaped.
        observations = [jnp.array([obs_idx[m]]) for m in MODALITY_NAMES]

        qs = self.agent.infer_states(observations=observations, empirical_prior=self.empirical_prior)
        self._qs = qs
        beliefs = self._beliefs_from_qs(qs)

        q_pi, _G = self.agent.infer_policies(qs)
        q = np.asarray(q_pi)[0].astype(float).copy()

        masked = []
        if levers:
            for a, name in enumerate(ACTIONS):
                if name != "do_nothing" and not levers.get(name, True):
                    q[a] = 0.0
                    masked.append(name)
        if q.sum() > 0:
            q = q / q.sum()

        # Walk actions from most to least preferred; take the first that clears its margin.
        # Margins rise when a person dismisses that action, so a dismissed lever gives way to
        # the next one, and when every lever has been dismissed the agent holds.
        gated = False
        chosen = DO_NOTHING
        for a in np.argsort(-q):
            a = int(a)
            if a == DO_NOTHING or q[a] <= 0.0:
                break
            if (q[a] - q[DO_NOTHING]) >= self.margins[ACTIONS[a]]:
                chosen = a
                break
            gated = True

        action = ACTIONS[chosen]
        # Rule on top of the model, labelled as such: confidently and persistently empty -> setback.
        # Two consecutive steps so a single radar miscount cannot switch a room off on a guest.
        self._empty_streak = self._empty_streak + 1 if beliefs[0][0] > 0.9 else 0
        if action == "do_nothing" and self._empty_streak >= self.setback_streak_required:
            action = "setback"

        pred_action = chosen  # setback keeps the do_nothing dynamics
        target = self._target_modality(action if action != "setback" else "do_nothing", reading)
        future = propagate(self.B, beliefs, pred_action, self.horizon)
        p_obs = expected_observation(self.A, future, target)
        pred = Prediction(
            modality=target, predicted=int(np.argmax(p_obs)), horizon=self.horizon,
            probability=float(np.max(p_obs)),
        )

        rec = Recommendation(
            action=action,
            reason=self._reason(action, beliefs, pred, gated, masked),
            confidence=float(q[chosen]),
            beliefs={
                "occupancy": {s: float(p) for s, p in zip(OCCUPANCY, beliefs[0])},
                "arousal": {s: float(p) for s, p in zip(AROUSAL, beliefs[1])},
            },
            observation=obs_idx,
            prediction=pred,
            q_pi=[float(x) for x in q],
            gated=gated,
            masked=masked,
        )
        self.last = rec
        return rec

    def commit(self, applied_action: str) -> None:
        """Tell the agent what was actually applied so its beliefs roll forward correctly."""
        idx = ACTIONS.index(applied_action) if applied_action in ACTIONS else DO_NOTHING
        act = jnp.array([[idx, idx]])
        self.empirical_prior = self.agent.update_empirical_prior(act, self._qs)

    def feedback(self, rec: Recommendation, accepted: bool) -> None:
        """Learn from accept / dismiss: dismissals raise the bar for that action."""
        if rec.action == "setback":
            if not accepted:
                self.setback_streak_required = min(8, self.setback_streak_required + 1)
            return
        if rec.action not in ACTIONS or rec.action == "do_nothing":
            return
        m = self.margins[rec.action]
        self.margins[rec.action] = min(0.6, m + 0.06) if not accepted else max(0.02, m - 0.01)

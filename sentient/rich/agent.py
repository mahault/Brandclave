"""RichAgent: the same copilot loop as the first prototype, over the seven-factor model.

Infer the joint state of the space from ten sensor channels, recommend one action or
nothing, say why, attach a sealed forecast. Two labelled rules sit on top of the model:
a persistently empty room gets a setback, and a room the model believes has a degraded
HVAC unit gets a maintenance ticket instead of another cooling command.
"""

import logging
from collections import deque

import numpy as np

from sentient.agent import Prediction, Recommendation

from .spec import (
    A_DEPENDENCIES,
    ACT,
    ACTION_TARGET,
    ACTIONS,
    B_DEPENDENCIES,
    DO_NOTHING,
    FACTOR_NAMES,
    FACTORS,
    INTENT_KEY_MODALITY,
    MODALITIES,
    MODALITY_NAMES,
    NUM_STATES,
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
except ImportError as exc:  # pragma: no cover
    PYMDP_AVAILABLE = False
    logger.warning("pymdp unavailable (%s)", exc)


class RichAgent:
    def __init__(self, intent: str, margin: float = 0.06, horizon: int = 3, gamma: float = 8.0,
                 hvac_threshold: float = 0.6):
        if not PYMDP_AVAILABLE:
            raise RuntimeError("pymdp is required")
        self.intent = intent
        self.horizon = horizon
        self.hvac_threshold = hvac_threshold
        self.margins = {a: margin for a in ACTIONS}
        self.A = build_A()
        self.B = build_B(intent)
        self.C = build_C(intent)
        self.D = build_D(intent)
        self.E = build_E()

        n_a = len(ACTIONS)
        n_f = len(FACTORS)
        policies = jnp.array([[[a] * n_f] for a in range(n_a)])
        self.agent = Agent(
            A=[jnp.array(a)[None, ...] for a in self.A],
            B=[jnp.array(b)[None, ...] for b in self.B],
            C=[jnp.array(c)[None, :] for c in self.C],
            D=[jnp.array(d)[None, :] for d in self.D],
            E=jnp.array(self.E),
            A_dependencies=A_DEPENDENCIES,
            B_dependencies=B_DEPENDENCIES,
            num_controls=[n_a] * n_f,
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
        # a transit space empties and refills every few minutes: wait longer before a setback
        self.setback_streak_required = 4 if intent == "service" else 2
        # channel health, online. Two signatures of a stuck sensor: the same raw value four
        # reads running (real sensors are noisy), or forecasts that keep failing while "same as
        # now" is right almost every time and the other channels hold. A flagged channel still
        # feeds inference; no action is taken that only that channel could confirm.
        self._t = 0
        self._pending: list[dict] = []
        self._hist = {m: deque(maxlen=15) for m in MODALITY_NAMES}
        self.flagged: set[str] = set()
        self._last_beliefs: list[np.ndarray] | None = None
        self._raw_prev: dict[str, float] = {}
        self._raw_same: dict[str, int] = {}
        self.last: Recommendation | None = None
        self.hvac_belief_trace: list[float] = []
        # Explicit filter for the HVAC factor. pymdp's one-step filter does not pass the
        # evidence "we commanded cooling and the temperature did not move" back from the
        # thermal factor to the hvac factor, so that one message is computed here, from the
        # same A and B, and fed back in as the hvac prior.
        self._hvac_post = self.D[FACTOR_NAMES.index("hvac")].copy()
        self._thermal_prev: np.ndarray | None = None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _beliefs_from_qs(qs) -> list[np.ndarray]:
        out = []
        for f, q in enumerate(qs):
            arr = np.asarray(q).reshape(-1, NUM_STATES[f])[-1]
            out.append(arr / arr.sum())
        return out

    def _target_modality(self, action: str, reading: Reading) -> str:
        target = ACTION_TARGET.get(action)
        return target or INTENT_KEY_MODALITY[self.intent]

    def _state_summary(self, beliefs: list[np.ndarray]) -> str:
        parts = []
        for f, name in enumerate(FACTOR_NAMES):
            if name in ("occupancy", "arousal", "valence", "thermal", "air", "hvac", "music"):
                i = int(np.argmax(beliefs[f]))
                parts.append(f"{name} {FACTORS[name][i]} ({beliefs[f][i]:.2f})")
        return ", ".join(parts)

    def _reason(self, action: str, beliefs: list[np.ndarray], pred: Prediction, gated: bool,
                masked: list[str], rule: str | None) -> str:
        state = self._state_summary(beliefs)
        if rule == "setback":
            return f"{state}; space confidently empty, set lights and HVAC back"
        if rule == "maintenance":
            return f"{state}; cooling commanded but temperature not responding, HVAC likely degraded, ticket instead of another command"
        if action == "do_nothing":
            why = "state uncertain, holding" if gated else "current state matches intent"
            if len(masked) >= 4:
                why = "no lever left to pull" + (", state uncertain" if gated else "")
            return f"{state}; intent {self.intent}; {why}"
        return (f"{state}; intent {self.intent}; {action} expected to move {pred.modality} "
                f"to {MODALITIES[pred.modality][pred.predicted]} within {pred.horizon} steps")

    # --------------------------------------------------------------------- loop
    def _score_channels(self, obs_idx: dict[str, int]) -> None:
        still = []
        for p in self._pending:
            if p["due"] > self._t:
                still.append(p)
                continue
            if not p["scored"]:
                continue
            for m, pred in p["pred"].items():
                self._hist[m].append((pred == obs_idx[m], p["persist"][m] == obs_idx[m]))
        self._pending = still
        acc = {m: (sum(h for h, _ in d) / len(d), sum(q for _, q in d) / len(d))
               for m, d in self._hist.items() if len(d) >= 12}
        for m, (a, q) in acc.items():
            others = [x for k, (x, _) in acc.items() if k != m and k not in self.flagged]
            if others and a < 0.25 and q >= 0.95 and sum(others) / len(others) >= 0.6:
                self.flagged.add(m)

    def _flatline(self, reading: Reading) -> None:
        raw = {"sound_level": reading.db, "temp": reading.temp_c, "co2": reading.co2_ppm,
               "motion": reading.motion_level, "lux": reading.lux}
        for m, v in raw.items():
            if v is None:
                continue
            self._raw_same[m] = self._raw_same.get(m, 0) + 1 if self._raw_prev.get(m) == v else 0
            self._raw_prev[m] = v
            if self._raw_same[m] >= 3:
                self.flagged.add(m)

    def step(self, reading: Reading, levers: dict[str, bool] | None = None) -> Recommendation:
        raw_obs = discretize(reading)
        self._score_channels(raw_obs)
        self._flatline(reading)
        obs_idx = dict(raw_obs)
        observations = [jnp.array([obs_idx[m]]) for m in MODALITY_NAMES]
        self._update_hvac(obs_idx["temp"])
        qs = self.agent.infer_states(observations=observations, empirical_prior=self.empirical_prior)
        self._qs = qs
        beliefs = self._beliefs_from_qs(qs)
        beliefs[FACTOR_NAMES.index("hvac")] = self._hvac_post.copy()
        self._thermal_prev = beliefs[FACTOR_NAMES.index("thermal")].copy()
        p_degraded = float(self._hvac_post[1])
        self.hvac_belief_trace.append(p_degraded)

        q_pi, _G = self.agent.infer_policies(qs)
        q = np.asarray(q_pi)[0].astype(float).copy()

        masked = []
        for a, name in enumerate(ACTIONS):
            if name == "do_nothing":
                continue
            if (levers and not levers.get(name, True)) or ACTION_TARGET.get(name) in self.flagged:
                q[a] = 0.0
                masked.append(name)
        if q.sum() > 0:
            q = q / q.sum()

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
        rule = None

        # In a service space a staff call is for a queue that is there, not one that might form.
        occ = beliefs[FACTOR_NAMES.index("occupancy")]
        if self.intent == "service" and action == "staff_alert" and occ[2] + occ[3] < 0.6:
            action, chosen = "do_nothing", DO_NOTHING

        # Rule 1: confidently and persistently empty -> setback.
        # empty only when the radar and the door counter agree
        self._empty_streak = (self._empty_streak + 1
                              if (occ[0] > 0.9 and raw_obs["entries"] == 0 and raw_obs["count"] == 0) else 0)
        if action == "do_nothing" and self._empty_streak >= self.setback_streak_required:
            action, rule = "setback", "setback"

        # Rule 2: HVAC believed degraded while the room is uncomfortable -> maintenance ticket,
        # instead of commanding a unit the model believes is not responding.
        thermal = beliefs[FACTOR_NAMES.index("thermal")]
        if (p_degraded > self.hvac_threshold and int(np.argmax(thermal)) != 1
                and action != "setback" and (not levers or levers.get("maintenance_ticket", True))):
            action, chosen, rule = "maintenance_ticket", ACT["maintenance_ticket"], "maintenance"

        target = self._target_modality(action if action != "setback" else "do_nothing", reading)
        future = propagate(self.B, beliefs, chosen if action != "setback" else DO_NOTHING, self.horizon)
        p_obs = expected_observation(self.A, future, target)
        pred = Prediction(modality=target, predicted=int(np.argmax(p_obs)), horizon=self.horizon,
                          probability=float(np.max(p_obs)))
        # a forecast for every channel, so every sensor is scored every step
        forecasts = {}
        for m in MODALITY_NAMES:
            if m == target:
                continue
            pm = expected_observation(self.A, future, m)
            forecasts[m] = (int(np.argmax(pm)), float(np.max(pm)))

        rec = Recommendation(
            action=action,
            reason=self._reason(action, beliefs, pred, gated, masked, rule),
            confidence=float(q[chosen]) if action != "setback" else float(q[DO_NOTHING]),
            beliefs={name: {s: float(p) for s, p in zip(FACTORS[name], beliefs[f])} for f, name in enumerate(FACTOR_NAMES)},
            observation=obs_idx,
            prediction=pred,
            q_pi=[float(x) for x in q],
            gated=gated,
            masked=masked,
            forecasts=forecasts,
        )
        self.last = rec
        self._last_beliefs = beliefs
        preds = {m: v[0] for m, v in forecasts.items()}
        preds[pred.modality] = pred.predicted
        self._pending.append({"due": self._t + self.horizon, "pred": preds, "persist": dict(raw_obs),
                              "scored": True})
        self._t += 1
        return rec

    def commit(self, applied_action: str, hvac_mode: str | None = None) -> None:
        """Roll beliefs forward. `hvac_mode` is the sustained control state of the unit
        ('cool_air', 'warm_air' or None): a known input, like intent, used by the HVAC filter."""
        idx = ACTIONS.index(applied_action) if applied_action in ACTIONS else DO_NOTHING
        act = jnp.array([[idx] * len(FACTORS)])
        self.empirical_prior = self.agent.update_empirical_prior(act, self._qs)
        if hvac_mode in ACT:
            self._pending_hvac_action = ACT[hvac_mode]
        elif applied_action in ("cool_air", "warm_air"):
            self._pending_hvac_action = idx
        else:
            self._pending_hvac_action = DO_NOTHING
        hv = FACTOR_NAMES.index("hvac")
        self.empirical_prior[hv] = jnp.array(self._hvac_post)[None, :]

    def _update_hvac(self, temp_obs: int) -> None:
        """P(hvac | thermal before, command, temperature seen), one exact message."""
        if self._thermal_prev is None:
            return
        a = getattr(self, "_pending_hvac_action", DO_NOTHING)
        th, hv = FACTOR_NAMES.index("thermal"), FACTOR_NAMES.index("hvac")
        B_th = self.B[th][..., a]                 # [next, prev, hvac]
        A_temp = self.A[MODALITY_NAMES.index("temp")]   # [obs, thermal]
        prior = self.B[hv][..., a] @ self._hvac_post    # [hvac]
        like = np.zeros(2)
        for h in range(2):
            p_next = B_th[:, :, h] @ self._thermal_prev
            like[h] = float(A_temp[temp_obs] @ p_next)
        post = prior * like
        self._hvac_post = post / post.sum()

    def feedback(self, rec: Recommendation, accepted: bool) -> None:
        if not accepted and self._pending:
            self._pending[-1]["scored"] = False   # its forecast assumed an action that never happened
        if rec.action == "setback":
            if not accepted:
                # someone pressed dismiss: the space is not empty
                self.setback_streak_required = min(8, self.setback_streak_required + 1)
                self._empty_streak = 0
            return
        if rec.action not in ACTIONS or rec.action == "do_nothing":
            return
        m = self.margins[rec.action]
        self.margins[rec.action] = min(0.95, m + 0.15) if not accepted else max(0.02, m - 0.01)

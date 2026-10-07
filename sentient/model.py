"""The generative model: what the space can be in, what the sensors say about it,
what actions do to it, and what each intent prefers.

Hidden state is inferred. Intent and purpose are set by humans and select preferences.
All matrices are plain numpy here; the agent wraps them for pymdp.
"""

from dataclasses import dataclass

import numpy as np

OCCUPANCY = ["empty", "some", "crowded"]
AROUSAL = ["calm", "elevated", "stressed"]
FACTORS = ["occupancy", "arousal"]

ACTIONS = ["do_nothing", "dim_lights", "lower_sound", "cool_air", "staff_alert"]
DO_NOTHING = 0

MODALITIES = {
    "count": ["0", "1-3", "4+"],
    "motion": ["still", "moving", "agitated"],
    "sound": ["quiet", "moderate", "loud"],
    "wearable": ["unknown", "normal", "elevated"],
}
MODALITY_NAMES = list(MODALITIES)
# Which hidden factors each modality depends on (indices into FACTORS)
A_DEPENDENCIES = [[0], [0, 1], [0, 1], [0, 1]]

INTENTS = ["restore", "work", "elevated", "service"]

# The observation an action is trying to move; used for the sealed prediction.
ACTION_TARGET = {
    "do_nothing": None,  # falls back to the intent's key modality
    "dim_lights": "motion",
    "lower_sound": "sound",
    "cool_air": "wearable",
    "staff_alert": "count",
}
INTENT_KEY_MODALITY = {"restore": "sound", "work": "sound", "elevated": "sound", "service": "count"}


@dataclass
class Reading:
    """One raw sensor reading. hr is None when no guest has opted a wearable in."""

    count: int
    motion_level: float  # 0..1
    db: float
    temp_c: float
    light_lux: float
    hr: float | None = None


def discretize(r: Reading) -> dict[str, int]:
    """Map a raw reading onto the model's observation categories."""
    count = 0 if r.count <= 0 else (1 if r.count <= 3 else 2)
    motion = 0 if r.motion_level < 0.33 else (1 if r.motion_level < 0.66 else 2)
    sound = 0 if r.db < 45 else (1 if r.db < 60 else 2)
    if r.hr is None:
        wearable = 0
    else:
        wearable = 1 if r.hr < 85 else 2
    return {"count": count, "motion": motion, "sound": sound, "wearable": wearable}


def _norm_cols(a: np.ndarray) -> np.ndarray:
    return a / a.sum(axis=0, keepdims=True)


def build_A() -> list[np.ndarray]:
    """Observation likelihoods P(obs | hidden states), one per modality.
    Shapes follow A_DEPENDENCIES: [obs, *dep_factor_states]."""
    # count | occupancy
    A_count = _norm_cols(np.array([
        [0.90, 0.08, 0.02],
        [0.08, 0.80, 0.15],
        [0.02, 0.12, 0.83],
    ]))

    # motion | occupancy, arousal  -> shape (3, 3, 3) = [obs, occ, arousal]
    A_motion = np.zeros((3, 3, 3))
    for ar in range(3):
        A_motion[:, 0, ar] = [0.90, 0.08, 0.02]  # empty: still
    A_motion[:, 1, :] = np.array([[0.65, 0.30, 0.05], [0.25, 0.55, 0.20], [0.10, 0.40, 0.50]]).T
    A_motion[:, 2, :] = np.array([[0.40, 0.50, 0.10], [0.15, 0.55, 0.30], [0.05, 0.35, 0.60]]).T

    # sound | occupancy, arousal
    A_sound = np.zeros((3, 3, 3))
    for ar in range(3):
        A_sound[:, 0, ar] = [0.90, 0.08, 0.02]
    A_sound[:, 1, :] = np.array([[0.60, 0.35, 0.05], [0.30, 0.50, 0.20], [0.15, 0.45, 0.40]]).T
    A_sound[:, 2, :] = np.array([[0.20, 0.50, 0.30], [0.10, 0.40, 0.50], [0.05, 0.25, 0.70]]).T

    # wearable | occupancy, arousal. 'unknown' carries mass in every occupied state: most
    # guests don't opt in and wearables drop sync. An empty room never produces a reading,
    # so a live heart rate is also evidence that someone is there.
    A_wear = np.zeros((3, 3, 3))
    for ar in range(3):
        A_wear[:, 0, ar] = [0.98, 0.01, 0.01]
    by_arousal = np.array([[0.50, 0.45, 0.05], [0.50, 0.25, 0.25], [0.50, 0.10, 0.40]]).T
    A_wear[:, 1, :] = by_arousal
    A_wear[:, 2, :] = by_arousal
    return [A_count, A_motion, A_sound, A_wear]


def build_B() -> list[np.ndarray]:
    """Transition models P(next | prev, action), shape [next, prev, action]."""
    n_a = len(ACTIONS)

    # occupancy: sticky; a staff alert relieves crowding (open the second desk).
    B_occ = np.zeros((3, 3, n_a))
    sticky = np.full((3, 3), 0.05)
    np.fill_diagonal(sticky, 0.90)
    for a in range(n_a):
        B_occ[:, :, a] = sticky
    B_occ[:, 2, ACTIONS.index("staff_alert")] = [0.05, 0.50, 0.45]

    # arousal: drifts up on its own; calming levers push it down.
    B_ar = np.zeros((3, 3, n_a))
    default = np.array([
        [0.88, 0.10, 0.02],   # from calm
        [0.10, 0.75, 0.15],   # from elevated
        [0.03, 0.15, 0.82],   # from stressed
    ]).T

    def calming(strength: float) -> np.ndarray:
        # strength: extra probability mass moved one step toward calm per tick
        return np.array([
            [0.92, 0.06, 0.02],
            [0.35 + strength, 0.55 - strength, 0.10],
            [0.10, 0.35 + strength, 0.55 - strength],
        ]).T

    # Sound is the biggest driver of arousal in a shared space, light next, temperature slowest.
    strengths = {"lower_sound": 0.20, "dim_lights": 0.15, "cool_air": 0.10}
    for a, name in enumerate(ACTIONS):
        B_ar[:, :, a] = calming(strengths[name]) if name in strengths else default
    return [B_occ, B_ar]


def build_C(intent: str) -> list[np.ndarray]:
    """Preferences over observations, selected by the intent a human set for the space."""
    if intent not in INTENTS:
        raise ValueError(f"unknown intent {intent!r}; expected one of {INTENTS}")
    z = np.zeros(3)
    table = {
        "restore": [z, np.array([1.0, 0.0, -1.5]), np.array([1.0, 0.0, -1.5]), np.array([0.0, 0.5, -1.0])],
        "work": [z, np.array([0.8, 0.0, -1.0]), np.array([1.0, -0.3, -1.5]), z],
        # A bar at 9pm: loud and moving are the target, not friction.
        "elevated": [np.array([-0.3, 0.3, 0.3]), np.array([-0.5, 1.0, 0.3]), np.array([-0.5, 0.5, 1.0]), z],
        "service": [np.array([0.0, 0.5, -1.0]), np.array([0.3, 0.0, -0.8]), np.array([0.5, 0.0, -1.0]), z],
    }
    return table[intent]


def build_D() -> list[np.ndarray]:
    return [np.array([0.4, 0.4, 0.2]), np.array([0.5, 0.3, 0.2])]


def build_E() -> np.ndarray:
    """Prior over policies: doing nothing is the default."""
    e = np.full(len(ACTIONS), 0.5 / (len(ACTIONS) - 1))
    e[DO_NOTHING] = 0.5
    return e


def expected_observation(A: list[np.ndarray], beliefs: list[np.ndarray], modality: str) -> np.ndarray:
    """P(obs) for one modality given factor beliefs, honouring A_DEPENDENCIES."""
    m = MODALITY_NAMES.index(modality)
    deps = A_DEPENDENCIES[m]
    if len(deps) == 1:
        return A[m] @ beliefs[deps[0]]
    return np.einsum("oij,i,j->o", A[m], beliefs[deps[0]], beliefs[deps[1]])


def propagate(B: list[np.ndarray], beliefs: list[np.ndarray], action: int, steps: int) -> list[np.ndarray]:
    """Roll factor beliefs forward under a repeated action."""
    out = [b.copy() for b in beliefs]
    for _ in range(steps):
        out = [B[f][:, :, action] @ out[f] for f in range(len(out))]
    return out

"""The rich generative model, as data.

Factors (hidden), modalities (observed, each depending on a subset of factors), actions,
and the builders for A (likelihoods), B (transitions, some depending on other factors),
C (preferences selected by intent), D (priors) and E (policy prior).

All matrices are plain numpy. The agent wraps them for pymdp.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np

# ------------------------------------------------------------------ factors
FACTORS: dict[str, list[str]] = {
    "occupancy": ["empty", "few", "some", "crowded"],
    "arousal": ["calm", "elevated", "stressed"],
    "valence": ["negative", "neutral", "positive"],
    "activity": ["resting", "working", "socializing", "transiting"],
    "thermal": ["cold", "comfortable", "warm"],
    "air": ["fresh", "stale"],
    "hvac": ["ok", "degraded"],
    "music": ["off", "low", "high"],   # the room's own sound system: controlled, not inferred from people
    "lighting": ["dim", "mid", "bright"],   # the room's own lights: controlled, seen through the lux sensor
}
FACTOR_NAMES = list(FACTORS)
F = {name: i for i, name in enumerate(FACTOR_NAMES)}
NUM_STATES = [len(v) for v in FACTORS.values()]

# --------------------------------------------------------------- modalities
MODALITIES: dict[str, list[str]] = {
    "count": ["0", "1-2", "3-6", "7+"],
    "entries": ["none", "few", "many"],
    "motion": ["still", "moving", "agitated"],
    "sound_level": ["quiet", "moderate", "loud"],
    "sound_character": ["flat", "steady", "bursty"],
    "temp": ["cold", "ok", "warm"],
    "co2": ["fresh", "ok", "stale"],
    "hr": ["unknown", "normal", "elevated"],
    "hrv": ["unknown", "low", "high"],
    "dwell": ["short", "medium", "long"],
    "lux": ["unknown", "dim", "mid", "bright"],
}
MODALITY_NAMES = list(MODALITIES)
NUM_OBS = [len(v) for v in MODALITIES.values()]
A_DEPENDENCIES: list[list[int]] = [
    [F["occupancy"]],                                   # count
    [F["occupancy"]],                                   # entries
    [F["occupancy"], F["arousal"], F["activity"]],      # motion
    [F["occupancy"], F["arousal"], F["activity"], F["music"]],   # sound_level
    [F["valence"], F["activity"]],                      # sound_character
    [F["thermal"]],                                     # temp
    [F["occupancy"], F["air"]],                         # co2
    [F["occupancy"], F["arousal"]],                     # hr
    [F["occupancy"], F["arousal"], F["valence"]],       # hrv
    [F["activity"]],                                    # dwell
    [F["lighting"]],                                    # lux
]

# ------------------------------------------------------------------ actions
ACTIONS = [
    "do_nothing", "dim_lights", "raise_lights", "lower_sound", "raise_sound",
    "cool_air", "warm_air", "ventilate", "staff_alert", "maintenance_ticket",
]
DO_NOTHING = 0
ACT = {a: i for i, a in enumerate(ACTIONS)}

# Transition dependencies: first entry is the factor itself.
B_DEPENDENCIES: list[list[int]] = [
    [F["occupancy"]],
    [F["arousal"], F["valence"]],
    [F["valence"], F["arousal"], F["thermal"]],
    [F["activity"]],
    [F["thermal"], F["hvac"]],
    [F["air"], F["occupancy"]],
    [F["hvac"]],
    [F["music"]],
    [F["lighting"]],
]

INTENTS = ["restore", "work", "elevated", "service"]

# The observation an action is trying to move, for the sealed prediction.
ACTION_TARGET = {
    "do_nothing": None,
    "dim_lights": "lux",
    "raise_lights": "lux",
    "lower_sound": "sound_level",
    "raise_sound": "sound_level",
    "cool_air": "temp",
    "warm_air": "temp",
    "ventilate": "co2",
    "staff_alert": "count",
    "maintenance_ticket": "temp",
}
INTENT_KEY_MODALITY = {"restore": "sound_level", "work": "co2", "elevated": "sound_level", "service": "count"}


@dataclass
class Reading:
    """One raw sensor frame. Wearable fields are None when nobody has opted in or the
    device is out of sync."""

    count: int
    entries: int
    motion_level: float      # 0..1
    db: float
    burstiness: float        # 0..1, how spiky the sound is (laughter vs hum)
    temp_c: float
    co2_ppm: float
    hr: float | None = None
    hrv: float | None = None
    mean_dwell_min: float = 0.0
    lux: float | None = None     # ambient light level; None when no light sensor reports


def discretize(r: Reading) -> dict[str, int]:
    def bin3(x, a, b):
        return 0 if x < a else (1 if x < b else 2)

    count = 0 if r.count <= 0 else (1 if r.count <= 2 else (2 if r.count <= 6 else 3))
    entries = 0 if r.entries <= 0 else (1 if r.entries <= 2 else 2)
    return {
        "count": count,
        "entries": entries,
        "motion": bin3(r.motion_level, 0.33, 0.66),
        "sound_level": bin3(r.db, 45.0, 58.0),
        "sound_character": bin3(r.burstiness, 0.2, 0.5),
        "temp": bin3(r.temp_c, 20.5, 24.0),     # the comfort band a building engineer would set
        "co2": bin3(r.co2_ppm, 800.0, 1000.0),   # under 800 fresh, over 1,000 stale
        "hr": 0 if r.hr is None else (1 if r.hr < 85 else 2),
        "hrv": 0 if r.hrv is None else (1 if r.hrv < 45 else 2),
        "dwell": bin3(r.mean_dwell_min, 10.0, 40.0),
        "lux": 0 if r.lux is None else 1 + bin3(r.lux, 150.0, 400.0),
    }


# ------------------------------------------------------------------ helpers
def _n(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    v = np.clip(v, 1e-4, None)
    return v / v.sum()


def _shift_up(p: np.ndarray, amount: float) -> np.ndarray:
    """Move probability mass one category upward."""
    q = p.copy()
    for i in range(len(p) - 1):
        m = p[i] * amount
        q[i] -= m
        q[i + 1] += m
    return _n(q)


def _shift_down(p: np.ndarray, amount: float) -> np.ndarray:
    q = p.copy()
    for i in range(1, len(p)):
        m = p[i] * amount
        q[i] -= m
        q[i - 1] += m
    return _n(q)


def _fill_A(m: int, fn) -> np.ndarray:
    """Build A[m] of shape [obs, *dep_states] from fn(*state_indices) -> obs distribution."""
    deps = A_DEPENDENCIES[m]
    shape = [NUM_OBS[m]] + [NUM_STATES[f] for f in deps]
    A = np.zeros(shape)
    for idx in product(*[range(NUM_STATES[f]) for f in deps]):
        A[(slice(None),) + idx] = _n(fn(*idx))
    return A


def _fill_B(f: int, fn) -> np.ndarray:
    """Build B[f] of shape [next, self, *other_deps, action] from
    fn(self_state, *other_states, action) -> distribution over next."""
    deps = B_DEPENDENCIES[f]
    others = deps[1:]
    n = NUM_STATES[f]
    shape = [n, n] + [NUM_STATES[o] for o in others] + [len(ACTIONS)]
    B = np.zeros(shape)
    for idx in product(range(n), *[range(NUM_STATES[o]) for o in others], range(len(ACTIONS))):
        B[(slice(None),) + idx] = _n(fn(*idx))
    return B


# --------------------------------------------------------------- A builders
def build_A() -> list[np.ndarray]:
    def count(occ):
        return [[0.92, 0.06, 0.02, 0.0], [0.08, 0.82, 0.10, 0.0],
                [0.01, 0.12, 0.80, 0.07], [0.0, 0.02, 0.15, 0.83]][occ]

    def entries(occ):
        # arrivals in any two-minute window are rare; a fuller room sees slightly more
        return [[0.92, 0.07, 0.01], [0.78, 0.19, 0.03], [0.68, 0.27, 0.05], [0.58, 0.33, 0.09]][occ]

    def motion(occ, ar, act):
        if occ == 0:
            return [0.92, 0.06, 0.02]
        p = np.array([[0.50, 0.42, 0.08], [0.18, 0.57, 0.25], [0.06, 0.38, 0.56]][ar])
        if act == 0:
            p = _shift_down(p, 0.15)    # resting: a little stiller
        elif act == 3:
            p = _shift_up(p, 0.25)      # transiting: moving
        return p

    def sound_level(occ, ar, act, music):
        # sound is people plus the room's own music
        if occ == 0:
            p = np.array([0.92, 0.07, 0.01])
        else:
            p = np.array([[0.60, 0.33, 0.07], [0.30, 0.50, 0.20], [0.10, 0.40, 0.50]][occ - 1])
            p = _shift_up(p, [0.0, 0.30, 0.50][ar])
            if act == 0:
                p = _shift_down(p, 0.15)
            elif act == 2:
                p = _shift_up(p, 0.15)
            elif act == 1:
                p = _shift_down(p, 0.10)
        if music == 1:
            p = _shift_up(p, 0.45)
        elif music == 2:
            p = _shift_up(_shift_up(p, 0.70), 0.60)
        return p

    def sound_character(val, act):
        if act == 0:
            base = [0.80, 0.15, 0.05]
        elif act == 1:
            base = [0.20, 0.65, 0.15]
        elif act == 3:
            base = [0.40, 0.45, 0.15]
        else:  # socializing: character follows valence
            base = [[0.25, 0.55, 0.20], [0.15, 0.50, 0.35], [0.05, 0.30, 0.65]][val]
        p = np.array(base)
        if act != 2 and val == 2:
            p = _shift_up(p, 0.15)
        return p

    def temp(th):
        return [[0.85, 0.13, 0.02], [0.10, 0.80, 0.10], [0.02, 0.13, 0.85]][th]

    def co2(occ, air):
        fresh = [[0.90, 0.09, 0.01], [0.78, 0.19, 0.03], [0.62, 0.32, 0.06], [0.45, 0.40, 0.15]]
        stale = [[0.50, 0.40, 0.10], [0.30, 0.50, 0.20], [0.12, 0.43, 0.45], [0.05, 0.30, 0.65]]
        return (fresh if air == 0 else stale)[occ]

    def hr(occ, ar):
        if occ == 0:
            return [0.98, 0.01, 0.01]
        return [[0.5, 0.45, 0.05], [0.5, 0.28, 0.22], [0.5, 0.10, 0.40]][ar]

    def hrv(occ, ar, val):
        if occ == 0:
            return [0.98, 0.01, 0.01]
        low, high = [(0.10, 0.40), (0.25, 0.25), (0.40, 0.10)][ar]
        d = [-0.08, 0.0, 0.08][val]
        return [0.5, low - d, high + d]

    def dwell(act):
        return [[0.10, 0.40, 0.50], [0.05, 0.35, 0.60], [0.15, 0.50, 0.35], [0.75, 0.20, 0.05]][act]

    def lux(li):
        # unknown carries a little mass everywhere: not every room has a light sensor
        return [[0.05, 0.85, 0.08, 0.02], [0.05, 0.08, 0.80, 0.07], [0.05, 0.02, 0.08, 0.85]][li]

    fns = [count, entries, motion, sound_level, sound_character, temp, co2, hr, hrv, dwell, lux]
    return [_fill_A(m, fn) for m, fn in enumerate(fns)]


# --------------------------------------------------------------- B builders
def build_B(intent: str = "service") -> list[np.ndarray]:
    def occupancy(s, a):
        n = 4
        p = np.full(n, 0.03)
        p[s] = 0.88
        if s > 0:
            p[s - 1] += 0.03
        if s < n - 1:
            p[s + 1] += 0.03
        if intent == "service" and a == ACT["staff_alert"] and s >= 2:
            p = np.array([0.02, 0.18, 0.55, 0.25]) if s == 3 else np.array([0.02, 0.30, 0.60, 0.08])
        return p

    def arousal(s, val, a):
        p = np.array([[0.92, 0.07, 0.01], [0.08, 0.84, 0.08], [0.02, 0.10, 0.88]][s])
        calming = {"lower_sound": 0.35, "dim_lights": 0.25, "cool_air": 0.15, "ventilate": 0.10}
        arousing = {"raise_sound": 0.45, "raise_lights": 0.30}
        name = ACTIONS[a]
        if name in calming:
            p = _shift_down(p, calming[name])
        elif name in arousing:
            p = _shift_up(p, arousing[name])
        elif name == "staff_alert" and s == 2 and val < 2:
            p = _shift_down(p, 0.40)      # a person calms someone stressed and not enjoying it
        return p

    def valence(s, ar, th, a):
        p = np.array([[0.85, 0.13, 0.02], [0.07, 0.86, 0.07], [0.02, 0.13, 0.85]][s])
        if ar == 2:
            p = _shift_down(p, 0.20)      # stress sours
        elif ar == 0:
            p = _shift_up(p, 0.05)
        if th != 1:
            p = _shift_down(p, 0.12)      # discomfort sours
        if ACTIONS[a] == "staff_alert" and ar == 2 and s < 2:
            p = _shift_up(p, 0.45 if s == 0 else 0.10)   # someone came: matters when stressed and unhappy
        return p

    def activity(s, a):
        p = np.full(4, 0.02)
        p[s] = 0.94
        return p

    def thermal(s, hv, a):
        # natural drift: bodies warm a room slowly; cold rooms drift toward ok
        p = np.array([[0.85, 0.15, 0.0], [0.02, 0.90, 0.08], [0.0, 0.05, 0.95]][s])
        name = ACTIONS[a]
        if hv == 0:  # hvac works
            if name == "cool_air":
                p = np.array([[0.92, 0.08, 0.0], [0.12, 0.85, 0.03], [0.02, 0.58, 0.40]][s])
            elif name == "warm_air":
                p = np.array([[0.40, 0.58, 0.02], [0.02, 0.85, 0.13], [0.0, 0.08, 0.92]][s])
        return p

    def air(s, occ, a):
        stale_rate = [0.0, 0.03, 0.06, 0.12][occ]
        if s == 0:
            p = np.array([1.0 - stale_rate, stale_rate])
        else:
            recover = 0.20 if occ == 0 else 0.05
            p = np.array([recover, 1.0 - recover])
        if ACTIONS[a] == "ventilate":
            p = np.array([0.95, 0.05]) if s == 0 else np.array([0.35, 0.65])
        return p

    def hvac(s, a):
        p = np.array([[0.997, 0.003], [0.01, 0.99]][s])
        if ACTIONS[a] == "maintenance_ticket" and s == 1:
            p = np.array([0.35, 0.65])
        return p

    def music(s, a):
        p = np.full(3, 0.01)
        p[s] = 0.98
        name = ACTIONS[a]
        if name == "lower_sound":
            p = np.full(3, 0.02)
            p[max(0, s - 1)] = 0.96
        elif name == "raise_sound":
            p = np.full(3, 0.02)
            p[min(2, s + 1)] = 0.96
        return p

    def lighting(s, a):
        p = np.full(3, 0.01)
        p[s] = 0.98
        name = ACTIONS[a]
        if name == "dim_lights":
            p = np.full(3, 0.02)
            p[max(0, s - 1)] = 0.96
        elif name == "raise_lights":
            p = np.full(3, 0.02)
            p[min(2, s + 1)] = 0.96
        return p

    fns = [occupancy, arousal, valence, activity, thermal, air, hvac, music, lighting]
    return [_fill_B(f, fn) for f, fn in enumerate(fns)]


# ---------------------------------------------------------------- C, D, E
def build_C(intent: str) -> list[np.ndarray]:
    if intent not in INTENTS:
        raise ValueError(f"unknown intent {intent!r}")
    z3 = np.zeros(3)
    z4 = np.zeros(4)
    # universal comfort preferences; "ok" air is fine, only stale is a problem
    temp = np.array([-0.8, 1.0, -0.8])
    co2 = np.array([0.0, 0.0, -1.2])
    # lighting that fits the purpose of the space (lux: unknown, dim, mid, bright)
    lux = {
        "restore": np.array([0.0, 0.8, 0.1, -0.6]),
        "work": np.array([0.0, -0.6, 0.3, 0.3]),
        "elevated": np.array([0.0, -0.3, 0.1, 0.1]),
        "service": np.array([0.0, -0.4, 0.2, 0.2]),
    }[intent]
    if intent == "restore":
        return [z4, z3, np.array([1.0, 0.0, -1.2]), np.array([1.0, 0.0, -1.0]),
                np.array([0.2, 0.2, 0.2]), temp, co2, np.array([0.0, 0.5, -1.0]),
                np.array([0.0, -0.8, 0.6]), z3, lux]
    if intent == "work":
        return [z4, z3, np.array([0.8, 0.0, -1.0]), np.array([0.8, 0.2, -1.5]),
                np.array([0.0, 0.5, -0.7]), temp, co2 * 1.5, z3, np.array([0.0, -0.5, 0.4]), z3, lux]
    if intent == "elevated":
        return [np.array([-0.5, -0.2, 0.3, 0.4]), z3, np.array([-0.6, 0.8, 0.5]),
                np.array([-1.0, 0.2, 1.2]), np.array([-0.6, 0.0, 1.0]), temp, co2 * 0.6, z3, z3, z3, lux]
    # service: a lobby that should not be crowded
    return [np.array([0.0, 0.5, 0.2, -1.2]), z3, np.array([0.2, 0.0, -0.6]),
            np.array([0.4, 0.0, -0.8]), z3, temp, co2, z3, z3, z3, lux]


def build_D(intent: str) -> list[np.ndarray]:
    activity = {
        "restore": [0.6, 0.15, 0.15, 0.1],
        "work": [0.1, 0.65, 0.15, 0.1],
        "elevated": [0.05, 0.05, 0.8, 0.1],
        "service": [0.05, 0.1, 0.15, 0.7],
    }[intent]
    return [np.array([0.4, 0.3, 0.2, 0.1]), np.array([0.5, 0.3, 0.2]), np.array([0.2, 0.6, 0.2]),
            np.array(activity), np.array([0.15, 0.6, 0.25]), np.array([0.8, 0.2]), np.array([0.95, 0.05]),
            np.array([0.2, 0.4, 0.4]), np.array([0.3, 0.4, 0.3])]


def build_E() -> np.ndarray:
    e = np.full(len(ACTIONS), 0.72 / (len(ACTIONS) - 2))
    e[DO_NOTHING] = 0.25
    e[ACT["maintenance_ticket"]] = 0.03
    return e / e.sum()


# ------------------------------------------------- belief propagation helpers
def expected_observation(A: list[np.ndarray], beliefs: list[np.ndarray], modality: str) -> np.ndarray:
    m = MODALITY_NAMES.index(modality)
    out = A[m]
    for f in reversed(A_DEPENDENCIES[m]):
        out = np.tensordot(out, beliefs[f], axes=([out.ndim - 1], [0]))
    return out


def propagate(B: list[np.ndarray], beliefs: list[np.ndarray], action: int, steps: int) -> list[np.ndarray]:
    out = [b.copy() for b in beliefs]
    for _ in range(steps):
        nxt = []
        for f, deps in enumerate(B_DEPENDENCIES):
            Bf = B[f][..., action]          # [next, self, *others]
            for d in reversed(deps):
                Bf = np.tensordot(Bf, out[d], axes=([Bf.ndim - 1], [0]))
            nxt.append(Bf / Bf.sum())
        out = nxt
    return out

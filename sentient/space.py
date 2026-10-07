"""A simulated space, built to be richer than the agent's model of it.

Individual occupants with their own purpose, sensitivities, valence and wearable; groups
arrive and leave; temperature has inertia so cooling is felt minutes later; staff take a
variable time to respond; radar misses and ghosts people; the dB meter spikes on doors and
HVAC; wearables drop out of sync; daylight moves with the clock. The agent never sees any of
this directly. It sees `read()`, and it has to cope with a world that does not share its
structure. One step is two minutes.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .model import Reading

STEP_MINUTES = 2.0

# purpose -> (baseline arousal, sociability, valence)
PURPOSE_PROFILE = {
    "restore": (0.30, 0.20, 0.55),
    "work": (0.45, 0.25, 0.50),
    "social": (0.70, 0.85, 0.75),
    "transit": (0.50, 0.45, 0.45),
}


@dataclass
class Occupant:
    purpose: str
    arousal: float
    baseline: float
    sociability: float
    valence: float               # 0 unhappy .. 1 delighted; the model does not know this exists
    sens_sound: float
    sens_light: float
    sens_crowd: float
    sens_warm: float
    motion_scale: float
    wearable: bool
    hr_offset: float
    arrived_at: int
    dwell_steps: float           # expected stay; per-step leave hazard is 1/dwell


@dataclass
class SpaceConfig:
    name: str
    intent: str
    purpose: str = "restore"           # purpose of new arrivals
    n_occupants: int = 1               # present at t=0
    capacity: int = 12
    arrival_rate: Callable[[float], float] | None = None  # expected arrivals per step, by hour
    dwell_steps: float = 30.0
    group_sizes: tuple = ((1, 0.6), (2, 0.25), (3, 0.1), (4, 0.05))
    wearable_rate: float = 0.0         # probability a new arrival has opted a wearable in
    light: float = 0.8
    sound_setting: float = 0.5
    cooling: bool = False
    arousal0: float = 0.5              # initial arousal of those present at t=0
    motion_scale: float = 0.8          # default; a resting guest is lower
    social_drive: float = 0.0          # arousal pressure from the setting itself
    contagion: float = 0.10            # pull toward the group mean
    start_hour: float = 14.0
    daylight: bool = True
    # sensor artifacts
    radar_miss: float = 0.05
    radar_ghost: float = 0.03
    dropout: float = 0.02              # stale reading
    db_spike: float = 0.04             # door / HVAC transient
    wearable_gap: float = 0.15
    stuck_db: float | None = None      # a silently failed dB meter
    seed: int = 0


@dataclass
class Space:
    cfg: SpaceConfig
    rng: np.random.Generator = field(init=False)
    occupants: list[Occupant] = field(default_factory=list)
    t: int = 0
    light: float = 0.8
    sound_setting: float = 0.5
    cooling: bool = False
    temp: float = 23.0
    relief_timer: int = 0
    alert_cooldown: int = 0
    _last_reading: Reading | None = None
    arrivals_total: int = 0
    departures_total: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.cfg.seed)
        self.light = self.cfg.light
        self.sound_setting = self.cfg.sound_setting
        self.cooling = self.cfg.cooling
        self.temp = 21.5 if self.cooling else 24.0
        for _ in range(self.cfg.n_occupants):
            self.occupants.append(self._new_occupant(self.cfg.purpose, self.cfg.arousal0))

    # ---------------------------------------------------------------- population
    @property
    def n(self) -> int:
        return len(self.occupants)

    @property
    def hour(self) -> float:
        return (self.cfg.start_hour + self.t * STEP_MINUTES / 60.0) % 24.0

    @property
    def arousal(self) -> float:
        return float(np.mean([o.arousal for o in self.occupants])) if self.occupants else 0.0

    def mean_valence(self) -> float:
        return float(np.mean([o.valence for o in self.occupants])) if self.occupants else 0.5

    def _new_occupant(self, purpose: str, arousal: float | None = None) -> Occupant:
        rng = self.rng
        base, soc, val = PURPOSE_PROFILE[purpose]
        base = float(np.clip(base + rng.normal(0, 0.08), 0.05, 0.95))
        return Occupant(
            purpose=purpose,
            arousal=float(np.clip(arousal if arousal is not None else base + rng.normal(0, 0.1), 0, 1)),
            baseline=base,
            sociability=float(np.clip(soc + rng.normal(0, 0.15), 0, 1)),
            valence=float(np.clip(val + rng.normal(0, 0.15), 0, 1)),
            sens_sound=float(np.clip(rng.normal(1.0, 0.3), 0.2, 2.0)),
            sens_light=float(np.clip(rng.normal(1.0, 0.3), 0.2, 2.0)),
            sens_crowd=float(np.clip(rng.normal(1.0, 0.3), 0.2, 2.0)),
            sens_warm=float(np.clip(rng.normal(1.0, 0.3), 0.2, 2.0)),
            motion_scale=float(np.clip(self.cfg.motion_scale + rng.normal(0, 0.15), 0.05, 1.2)),
            wearable=bool(rng.random() < self.cfg.wearable_rate),
            hr_offset=float(rng.normal(0, 6)),
            arrived_at=self.t,
            dwell_steps=float(max(3.0, rng.normal(self.cfg.dwell_steps, self.cfg.dwell_steps * 0.4))),
        )

    def _arrivals_and_departures(self) -> None:
        cfg, rng = self.cfg, self.rng
        # departures: per-occupant hazard
        keep = []
        for o in self.occupants:
            if rng.random() < 1.0 / o.dwell_steps:
                self.departures_total += 1
            else:
                keep.append(o)
        self.occupants = keep
        # arrivals: Poisson groups
        if cfg.arrival_rate is not None:
            lam = max(0.0, cfg.arrival_rate(self.hour))
            for _ in range(rng.poisson(lam)):
                sizes, probs = zip(*cfg.group_sizes)
                size = int(rng.choice(sizes, p=np.array(probs) / sum(probs)))
                for _ in range(size):
                    if self.n < cfg.capacity:
                        self.occupants.append(self._new_occupant(cfg.purpose))
                        self.arrivals_total += 1

    # ------------------------------------------------------------------ control
    def levers(self) -> dict[str, bool]:
        return {
            "dim_lights": self.light > 0.25,
            "lower_sound": self.sound_setting > 0.15,
            "cool_air": not self.cooling,
            "staff_alert": self.relief_timer == 0 and self.alert_cooldown == 0,
        }

    def apply(self, action: str) -> None:
        if action == "dim_lights":
            self.light = max(0.2, self.light - 0.3)
        elif action == "lower_sound":
            self.sound_setting = max(0.1, self.sound_setting - 0.3)
        elif action == "cool_air":
            self.cooling = True  # felt over the next ~10 steps, see temperature inertia
        elif action == "staff_alert":
            self.relief_timer = int(self.rng.integers(2, 6))  # staff take 4-10 minutes
            self.alert_cooldown = 8
        elif action == "setback":
            self.light = 0.1
            self.cooling = False
        elif action == "do_nothing":
            pass
        else:
            raise ValueError(f"unknown action {action!r}")

    def preset(self, light: float, sound_setting: float, cooling: bool) -> None:
        self.light, self.sound_setting, self.cooling = light, sound_setting, cooling

    # ----------------------------------------------------------------- dynamics
    def step(self) -> None:
        cfg, rng = self.cfg, self.rng
        self._arrivals_and_departures()

        if self.alert_cooldown > 0:
            self.alert_cooldown -= 1
        if self.relief_timer > 0:
            self.relief_timer -= 1
            if self.relief_timer == 0 and self.n > 3:
                # second desk opens / staff route people on: most of the queue clears
                leaving = int(self.n - rng.integers(2, 4))
                order = rng.permutation(self.n)[:leaving]
                self.occupants = [o for i, o in enumerate(self.occupants) if i not in set(order)]
                self.departures_total += leaving

        # temperature with inertia: setpoint depends on cooling, bodies add heat
        setpoint = (21.0 if self.cooling else 24.5) + 0.25 * self.n
        self.temp += 0.15 * (setpoint - self.temp) + rng.normal(0, 0.1)

        if self.occupants:
            crowd = min(1.0, self.n / 8.0)
            warm_felt = float(np.clip((self.temp - 21.0) / 5.0, 0.0, 1.0))
            group_mean = self.arousal
            for o in self.occupants:
                pressure = (
                    o.baseline
                    + o.sens_sound * 0.35 * self.sound_setting
                    + o.sens_light * 0.20 * self.light
                    + o.sens_crowd * 0.25 * crowd
                    + o.sens_warm * 0.20 * warm_felt
                    + cfg.social_drive
                    - 0.35  # centre so a calm setting brings people back toward baseline
                )
                target = float(np.clip(pressure, 0.0, 1.0))
                o.arousal += 0.25 * (target - o.arousal) + cfg.contagion * (group_mean - o.arousal) + rng.normal(0, 0.04)
                o.arousal = float(np.clip(o.arousal, 0.0, 1.0))
        self.t += 1

    # ------------------------------------------------------------------ sensing
    def _daylight_lux(self) -> float:
        if not self.cfg.daylight:
            return 0.0
        h = self.hour
        return 300.0 * max(0.0, float(np.sin(np.pi * (h - 6.0) / 14.0))) if 6.0 <= h <= 20.0 else 0.0

    def read(self) -> Reading:
        cfg, rng = self.cfg, self.rng
        if self._last_reading is not None and rng.random() < cfg.dropout:
            return self._last_reading  # stale frame: the gateway repeated the last packet

        # radar: misses and ghosts
        detected = sum(1 for _ in self.occupants if rng.random() > cfg.radar_miss)
        if rng.random() < cfg.radar_ghost:
            detected += 1
        if self.occupants:
            motion = float(np.mean([o.motion_scale * o.arousal + 0.12 for o in self.occupants])) + rng.normal(0, 0.05)
        else:
            motion = abs(rng.normal(0, 0.03))

        # sound: power sum of floor, music, HVAC hum, conversation, transients
        sources = [32.0, 32.0 + 42.0 * self.sound_setting]
        if self.cooling:
            sources.append(38.0)
        if self.n >= 2:
            for o in self.occupants:
                talk = rng.random() < 0.3 + 0.5 * o.sociability
                if talk:
                    sources.append(40.0 + 15.0 * o.arousal + 8.0 * o.sociability)
        elif self.n == 1 and rng.random() < 0.08:
            sources.append(50.0)  # a phone call
        if rng.random() < cfg.db_spike:
            sources.append(float(rng.uniform(62, 72)))  # door, cart, HVAC kick
        db = 10.0 * np.log10(np.sum(10.0 ** (np.array(sources) / 10.0))) + rng.normal(0, 1.5)
        if cfg.stuck_db is not None:
            db = cfg.stuck_db

        lux = 50.0 + 450.0 * self.light + self._daylight_lux() + rng.normal(0, 10)

        hr = None
        wearers = [o for o in self.occupants if o.wearable]
        if wearers and rng.random() > cfg.wearable_gap:
            hr = float(np.mean([60.0 + 40.0 * o.arousal + o.hr_offset for o in wearers]) + rng.normal(0, 2.5))

        reading = Reading(
            count=int(detected),
            motion_level=float(np.clip(motion, 0.0, 1.0)),
            db=float(db),
            temp_c=float(self.temp + rng.normal(0, 0.2)),
            light_lux=float(lux),
            hr=hr,
        )
        self._last_reading = reading
        return reading

    def energy(self) -> float:
        """Energy proxy per step: lighting plus cooling compressor plus base load."""
        return 0.2 + 1.0 * self.light + (1.0 if self.cooling else 0.0)

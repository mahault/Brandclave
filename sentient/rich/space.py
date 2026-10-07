"""The rich simulated space. Richer than the model, on purpose.

Everything from the first world (individual occupants, arrivals, artifacts, lag) plus:
occupants have an activity and a valence that responds to comfort and fit; CO2 builds
with people and clears with ventilation; the HVAC unit can degrade mid-run so cooling
stops working; the microphone reports how spiky the sound is; a door counter reports
entries; wearables also report HRV. One step is two minutes.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .spec import Reading

STEP_MINUTES = 2.0

# purpose -> (baseline arousal, sociability, valence, activity index)
PURPOSE_PROFILE = {
    "restore": (0.30, 0.20, 0.55, 0),
    "work": (0.45, 0.25, 0.50, 1),
    "social": (0.70, 0.85, 0.75, 2),
    "transit": (0.50, 0.45, 0.45, 3),
}


@dataclass
class Occupant:
    purpose: str
    activity: int
    arousal: float
    baseline: float
    sociability: float
    valence: float
    sens_sound: float
    sens_light: float
    sens_crowd: float
    sens_warm: float
    sens_air: float
    motion_scale: float
    wearable: bool
    hr_offset: float
    arrived_at: int
    dwell_steps: float
    mood_anchor: float | None = None   # a mood the room cannot fix; a person can


@dataclass
class SpaceConfig:
    name: str
    intent: str
    purpose: str = "restore"
    n_occupants: int = 1
    capacity: int = 12
    arrival_rate: Callable[[float], float] | None = None
    dwell_steps: float = 30.0
    group_sizes: tuple = ((1, 0.6), (2, 0.25), (3, 0.1), (4, 0.05))
    wearable_rate: float = 0.0
    light: float = 0.8
    sound_setting: float = 0.5
    cooling: bool = False
    heating: bool = False
    arousal0: float = 0.5
    valence0: float | None = None
    motion_scale: float = 0.8
    social_drive: float = 0.0
    contagion: float = 0.10
    start_hour: float = 14.0
    outdoor_temp: float = 26.0
    co2_0: float = 500.0
    ventilation_rate: float = 0.03       # fraction of excess CO2 cleared per step, unventilated
    hvac_degrade_at: int | None = None   # step at which cooling silently stops working
    # sensor artifacts
    radar_miss: float = 0.05
    radar_ghost: float = 0.03
    dropout: float = 0.02
    db_spike: float = 0.04
    wearable_gap: float = 0.15
    stuck_db: float | None = None
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
    heating: bool = False
    ventilating: int = 0            # steps of boosted ventilation remaining
    temp: float = 23.0
    co2: float = 500.0
    hvac_degraded: bool = False
    relief_timer: int = 0
    alert_cooldown: int = 0
    comfort_timer: int = 0          # a staff member is with a distressed guest
    ticket_cooldown: int = 0
    repair_timer: int = 0
    _last_reading: Reading | None = None
    _entries_since_read: int = 0
    arrivals_total: int = 0
    departures_total: int = 0
    tickets: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.cfg.seed)
        self._lux_rng = np.random.default_rng(self.cfg.seed + 55_555)   # own stream: the world is unchanged
        self.light = self.cfg.light
        self.sound_setting = self.cfg.sound_setting
        self.cooling = self.cfg.cooling
        self.heating = self.cfg.heating
        self.temp = 21.5 if self.cooling else (23.5 if self.heating else 24.0)
        self.co2 = self.cfg.co2_0
        for _ in range(self.cfg.n_occupants):
            self.occupants.append(self._new_occupant(self.cfg.purpose, self.cfg.arousal0, self.cfg.valence0))

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

    def _new_occupant(self, purpose: str, arousal: float | None = None, valence: float | None = None) -> Occupant:
        rng = self.rng
        base, soc, val, act = PURPOSE_PROFILE[purpose]
        base = float(np.clip(base + rng.normal(0, 0.08), 0.05, 0.95))

        def s():
            return float(np.clip(rng.normal(1.0, 0.3), 0.2, 2.0))

        return Occupant(
            purpose=purpose, activity=act,
            arousal=float(np.clip(arousal if arousal is not None else base + rng.normal(0, 0.1), 0, 1)),
            baseline=base,
            sociability=float(np.clip(soc + rng.normal(0, 0.15), 0, 1)),
            valence=float(np.clip((valence if valence is not None else val) + rng.normal(0, 0.1), 0, 1)),
            sens_sound=s(), sens_light=s(), sens_crowd=s(), sens_warm=s(), sens_air=s(),
            motion_scale=float(np.clip(self.cfg.motion_scale + rng.normal(0, 0.15), 0.05, 1.2)),
            wearable=bool(rng.random() < self.cfg.wearable_rate),
            hr_offset=float(rng.normal(0, 6)),
            arrived_at=self.t if self.t > 0 else -int(rng.integers(2, max(3, int(min(self.cfg.dwell_steps, 40))))),
            dwell_steps=float(max(3.0, rng.normal(self.cfg.dwell_steps, self.cfg.dwell_steps * 0.4))),
        )

    def _arrivals_and_departures(self) -> None:
        cfg, rng = self.cfg, self.rng
        keep = []
        for o in self.occupants:
            if rng.random() < 1.0 / o.dwell_steps:
                self.departures_total += 1
            else:
                keep.append(o)
        self.occupants = keep
        if cfg.arrival_rate is not None:
            lam = max(0.0, cfg.arrival_rate(self.hour))
            for _ in range(rng.poisson(lam)):
                sizes, probs = zip(*cfg.group_sizes)
                size = int(rng.choice(sizes, p=np.array(probs) / sum(probs)))
                for _ in range(size):
                    if self.n < cfg.capacity:
                        self.occupants.append(self._new_occupant(cfg.purpose))
                        self.arrivals_total += 1
                        self._entries_since_read += 1

    # ------------------------------------------------------------------ control
    def levers(self) -> dict[str, bool]:
        return {
            "dim_lights": self.light > 0.25,
            "raise_lights": self.light < 0.85,
            "lower_sound": self.sound_setting > 0.15,
            "raise_sound": self.sound_setting < 0.85,
            "cool_air": not self.cooling,
            "warm_air": not self.heating,
            "ventilate": self.ventilating == 0,
            "staff_alert": self.relief_timer == 0 and self.alert_cooldown == 0,
            "maintenance_ticket": self.ticket_cooldown == 0,
        }

    def apply(self, action: str) -> None:
        rng = self.rng
        if action == "dim_lights":
            self.light = max(0.2, self.light - 0.3)
        elif action == "raise_lights":
            self.light = min(1.0, self.light + 0.3)
        elif action == "lower_sound":
            self.sound_setting = max(0.1, self.sound_setting - 0.3)
        elif action == "raise_sound":
            self.sound_setting = min(1.0, self.sound_setting + 0.3)
        elif action == "cool_air":
            self.cooling, self.heating = True, False
        elif action == "warm_air":
            self.heating, self.cooling = True, False
        elif action == "ventilate":
            self.ventilating = 10
        elif action == "staff_alert":
            self.alert_cooldown = 8
            if self.n > 3:
                self.relief_timer = int(rng.integers(2, 6))
            else:
                self.comfort_timer = int(rng.integers(4, 8))   # someone comes over for 8-16 minutes
        elif action == "maintenance_ticket":
            self.tickets += 1
            self.ticket_cooldown = 30
            if self.hvac_degraded:
                self.repair_timer = int(rng.integers(6, 15))   # engineer takes 12-30 minutes
        elif action == "setback":
            self.light = 0.1
            self.cooling = self.heating = False
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

        for attr in ("alert_cooldown", "ticket_cooldown", "ventilating", "comfort_timer"):
            if getattr(self, attr) > 0:
                setattr(self, attr, getattr(self, attr) - 1)
        if self.relief_timer > 0:
            self.relief_timer -= 1
            if self.relief_timer == 0 and self.n > 3:
                leaving = int(self.n - rng.integers(2, 4))
                order = set(rng.permutation(self.n)[:leaving].tolist())
                self.occupants = [o for i, o in enumerate(self.occupants) if i not in order]
                self.departures_total += leaving
        if cfg.hvac_degrade_at is not None and self.t == cfg.hvac_degrade_at:
            self.hvac_degraded = True
        if self.repair_timer > 0:
            self.repair_timer -= 1
            if self.repair_timer == 0:
                self.hvac_degraded = False

        # temperature with inertia; a degraded unit cannot cool
        if self.cooling and not self.hvac_degraded:
            setpoint = 21.0
        elif self.heating:
            setpoint = 23.0
        else:
            setpoint = 20.0 + 0.35 * (cfg.outdoor_temp - 20.0) + 1.5
        setpoint += 0.25 * self.n
        self.temp += 0.15 * (setpoint - self.temp) + rng.normal(0, 0.1)

        # CO2: people add, ventilation clears
        gen = 4.0 * self.n                                   # ppm per 2-min step per person
        rate = cfg.ventilation_rate * (6.0 if self.ventilating > 0 else 1.0)
        self.co2 += gen - rate * (self.co2 - 420.0) + rng.normal(0, 4)
        self.co2 = float(max(420.0, self.co2))

        if self.occupants:
            crowd = min(1.0, self.n / 8.0)
            warm_felt = float(np.clip((self.temp - 21.0) / 5.0, 0.0, 1.0))
            cold_felt = float(np.clip((20.5 - self.temp) / 3.0, 0.0, 1.0))
            stale_felt = float(np.clip((self.co2 - 700.0) / 800.0, 0.0, 1.0))
            group_mean = self.arousal
            for o in self.occupants:
                pressure = (
                    o.baseline
                    + o.sens_sound * 0.35 * self.sound_setting
                    + o.sens_light * 0.20 * self.light
                    + o.sens_crowd * 0.25 * crowd
                    + o.sens_warm * 0.20 * (warm_felt + cold_felt)
                    + o.sens_air * 0.15 * stale_felt
                    + cfg.social_drive
                    - 0.35
                )
                target = float(np.clip(pressure, 0.0, 1.0))
                if o.mood_anchor is not None and self.comfort_timer == 0:
                    target = max(target, 0.8)                  # distress the room cannot reach
                if self.comfort_timer > 0 and o.arousal > 0.6:
                    target = min(target, 0.35)                 # being attended to
                o.arousal += 0.25 * (target - o.arousal) + cfg.contagion * (group_mean - o.arousal) + rng.normal(0, 0.04)
                o.arousal = float(np.clip(o.arousal, 0.0, 1.0))

                # valence: discomfort sours, fit between setting and activity sweetens
                social = o.activity == 2
                discomfort = 0.35 * (warm_felt + cold_felt) + 0.30 * stale_felt
                if not social:
                    discomfort += 0.15 * max(0.0, crowd - 0.6)
                fit = 0.0
                if o.activity == 2:                      # socializing likes energy and company
                    fit = 0.4 * self.sound_setting + 0.3 * crowd
                elif o.activity == 0:                    # resting likes quiet and dim
                    fit = 0.4 * (1 - self.sound_setting) + 0.3 * (1 - self.light)
                elif o.activity == 1:                    # working likes moderate quiet and fresh air
                    fit = 0.3 * (1 - self.sound_setting) + 0.3 * (1 - stale_felt)
                over = max(0.0, o.arousal - (0.92 if social else 0.7))
                vt = float(np.clip(0.5 + 0.35 * fit - 0.5 * discomfort - 0.25 * over + (0.25 if self.comfort_timer > 0 else 0.0), 0, 1))
                if o.mood_anchor is not None:
                    if self.comfort_timer > 0:
                        o.mood_anchor = None            # they were helped
                    else:
                        vt = 0.3 * vt + 0.7 * o.mood_anchor
                o.valence += 0.15 * (vt - o.valence) + rng.normal(0, 0.03)
                o.valence = float(np.clip(o.valence, 0.0, 1.0))
        self.t += 1

    # ------------------------------------------------------------------ sensing
    def read(self) -> Reading:
        cfg, rng = self.cfg, self.rng
        if self._last_reading is not None and rng.random() < cfg.dropout:
            return self._last_reading

        detected = sum(1 for _ in self.occupants if rng.random() > cfg.radar_miss)
        if rng.random() < cfg.radar_ghost:
            detected += 1
        entries = self._entries_since_read
        self._entries_since_read = 0

        if self.occupants:
            motion = float(np.mean([0.85 * o.motion_scale * o.arousal + 0.06 for o in self.occupants])) + rng.normal(0, 0.05)
        else:
            motion = abs(rng.normal(0, 0.03))

        sources = [32.0, 32.0 + 42.0 * self.sound_setting]
        if self.cooling or self.heating:
            sources.append(38.0)
        bursts, steady = 0.0, 0.0
        if self.n >= 2:
            for o in self.occupants:
                if rng.random() < 0.3 + 0.5 * o.sociability:
                    sources.append(37.0 + 14.0 * o.arousal + 7.0 * o.sociability)
                    if o.activity == 2 and o.valence > 0.55 and rng.random() < o.valence:
                        bursts += 1.0                    # laughter
                    else:
                        steady += 1.0
        elif self.n == 1:
            o = self.occupants[0]
            if rng.random() < (0.35 if (o.arousal > 0.7 and o.valence < 0.35) else 0.08):
                sources.append(50.0 + 6.0 * o.arousal)     # a phone call, louder when upset
                steady += 0.5
            if self.occupants[0].activity == 1:
                steady += 0.4                            # keyboard, page turns
        if rng.random() < cfg.db_spike:
            sources.append(float(rng.uniform(62, 72)))
            bursts += 0.5
        db = 10.0 * np.log10(np.sum(10.0 ** (np.array(sources) / 10.0))) + rng.normal(0, 1.5)
        if cfg.stuck_db is not None:
            db = cfg.stuck_db
        total = bursts + steady
        burstiness = 0.0 if total == 0 else float(np.clip(bursts / total + rng.normal(0, 0.05), 0, 1))
        if total > 0 and self.n >= 2:
            burstiness = max(burstiness, 0.15)          # any conversation has some texture
        if cfg.stuck_db is not None:
            burstiness = 0.0

        hr = hrv = None
        wearers = [o for o in self.occupants if o.wearable]
        if wearers and rng.random() > cfg.wearable_gap:
            hr = float(np.mean([60.0 + 40.0 * o.arousal + o.hr_offset for o in wearers]) + rng.normal(0, 2.5))
            hrv = float(np.mean([75.0 - 45.0 * o.arousal + 15.0 * (o.valence - 0.5) for o in wearers]) + rng.normal(0, 4))

        dwell = float(np.mean([(self.t - o.arrived_at) * STEP_MINUTES for o in self.occupants])) if self.occupants else 0.0

        reading = Reading(
            count=int(detected), entries=int(entries),
            motion_level=float(np.clip(motion, 0.0, 1.0)), db=float(db), burstiness=burstiness,
            temp_c=float(self.temp + rng.normal(0, 0.2)), co2_ppm=float(self.co2 + rng.normal(0, 15)),
            hr=hr, hrv=hrv, mean_dwell_min=dwell,
            lux=float(max(0.0, 600.0 * self.light + self._lux_rng.normal(0, 20))),
        )
        self._last_reading = reading
        return reading

    def energy(self) -> float:
        return 0.2 + 1.0 * self.light + (1.0 if self.cooling else 0.0) + (0.8 if self.heating else 0.0) + (0.3 if self.ventilating else 0.0)

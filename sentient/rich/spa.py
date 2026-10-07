"""An outdoor thermal spa on a cold evening: the setting an operator asked about.

Guests walk a circuit (hot pool, sauna, cold plunge, rest on the quiet terrace, sometimes the
fire pit) and leave. Over the evening the air cools, a coach group arrives and the sauna queue
builds, a terrace heater fails, a loud group drifts into the quiet zone, and one guest who paired
a watch at check-in is distressed.

Three copies of the evening run from the same guests and the same plans:

  sentient  the RichAgent on the terrace (intent restore) and on the sauna door (intent service),
            unchanged from the room evaluations; nothing here is tuned for the spa
  rules     RuleController on the same sensors, plus a heater-fault detector that mirrors its
            cooling one (cold while heating, eight reads)
  team      today's spa run by a good team by eye: a second sauna opened when six are waiting,
            terrace heaters on at 18:30, a noisy group asked to move after a few minutes of
            noise, a heater fault reported once guests complain, a staff member passing a
            distressed guest about once every half hour

One step is two minutes. Every number here is simulated.
"""

import copy
from dataclasses import dataclass, field

import numpy as np

from .agent import RichAgent
from .rules import RuleController
from .scenarios import simulated_override
from .space import STEP_MINUTES, Occupant, Space, SpaceConfig
from .spec import Reading

START_HOUR = 17.0
STEPS = 90                      # 17:00 to 20:00
SAUNA_SEATS = 8
SECOND_SAUNA_STEPS = 15
COACH_AT = 15                   # 17:30
HEATER_FAILS_AT = 66            # 19:12
LOUD_GROUP_AT = 54              # 18:48
DISTRESSED_AT = 36              # 18:12
ZONES = ("arriving", "hot_pool", "sauna", "queue", "cold_plunge", "terrace", "firepit", "gone")


def outdoor_temp(t: int) -> float:
    """14 C at five, falling after sunset (about 18:15) to 6 C by eight."""
    h = START_HOUR + t * STEP_MINUTES / 60.0
    return 14.0 - 2.0 * max(0.0, h - 17.0) - 2.0 * max(0.0, h - 18.25)


class Terrace(Space):
    """The quiet terrace: loungers under patio heaters. A heater that has failed does not heat.
    A staff call goes to one paired guest when aimed at them, otherwise to whoever is making the
    terrace loud (staff can see who it is, and invite them to the fire pit), otherwise to whoever
    looks worst off."""

    def __init__(self, cfg: SpaceConfig):
        super().__init__(cfg)
        self.outbox: list[Occupant] = []

    def apply(self, action: str) -> None:
        if action == "staff_alert" and self.target_present() is None:
            self.alert_cooldown = 8
            loud = [o for o in self.occupants if o.purpose == "social"]
            if len(loud) >= 2:
                for o in loud:
                    self.occupants.remove(o)
                    self.outbox.append(o)
            elif self.occupants:
                self.attending = max(self.occupants, key=lambda o: o.arousal - o.valence)
                self.attend_timer = int(self.rng.integers(4, 8))
            return
        super().apply(action)

    def step(self) -> None:
        failed = self.hvac_degraded and self.heating
        if failed:
            self.heating = False
        n = self.n
        super().step()
        if self.heating:
            self.temp -= 0.15 * 0.25 * n      # outdoors, bodies do not warm the air under the heaters either
        if failed:
            self.heating = True


@dataclass
class World:
    key: str
    terrace: Terrace
    firepit: Space
    guests: list[Occupant] = field(default_factory=list)
    queue: list[Occupant] = field(default_factory=list)
    second_sauna: int = 0
    second_sauna_steps: int = 0
    calls: int = 0
    dismissed: int = 0
    tickets: int = 0
    ticket_step: int | None = None
    energy: float = 0.0
    mood: list[float] = field(default_factory=list)
    queue_minutes: float = 0.0
    queue_max: int = 0
    cold_minutes: float = 0.0
    noise_minutes: float = 0.0
    distress_minutes: float = 0.0
    team_noise: int = 0
    quiet_called: int = -100
    team_cold: int = 0
    last_db: float = 40.0
    actions: list[tuple[int, str, str]] = field(default_factory=list)

    def zone(self, name: str) -> list[Occupant]:
        if name == "terrace":
            return self.terrace.occupants
        if name == "firepit":
            return self.firepit.occupants
        if name == "queue":
            return self.queue
        return [g for g in self.guests if g.zone == name]

    @property
    def present(self) -> list[Occupant]:
        return [g for g in self.guests if g.zone != "gone"]

    def sauna_seats(self) -> int:
        return SAUNA_SEATS * (2 if self.second_sauna > 0 else 1)


def _terrace(seed: int) -> Terrace:
    return Terrace(SpaceConfig(
        name="spa_terrace", intent="restore", purpose="restore", n_occupants=0, capacity=40,
        light=0.4, sound_setting=0.2, heating=False, cooling=False, outdoor_temp=outdoor_temp(0),
        start_hour=START_HOUR, seed=seed, wearable_gap=0.1,
    ))


def _firepit(seed: int) -> Space:
    return Space(SpaceConfig(
        name="spa_firepit", intent="elevated", purpose="social", n_occupants=0, capacity=40,
        light=0.5, sound_setting=0.4, heating=True, outdoor_temp=outdoor_temp(0),
        start_hour=START_HOUR, seed=seed + 1,
    ))


class SpaEvening:
    def __init__(self, seed: int, distressed: bool = True, staff: str = "auto"):
        self.seed, self.t, self.staff = seed, 0, staff
        self._maker = Space(SpaceConfig(name="maker", intent="restore", n_occupants=0, seed=seed + 777,
                                        wearable_rate=0.15))
        self.rng = np.random.default_rng(seed + 31)
        self.distressed = distressed
        self.next_gid = 0
        self.worlds = {k: World(k, _terrace(seed), _firepit(seed)) for k in ("sentient", "rules", "team")}
        self.agents = {"terrace": RichAgent(intent="restore"), "sauna": RichAgent(intent="service")}
        self.rules = {"terrace": SpaRules("restore"), "sauna": RuleController("service")}
        self.staff_rng = {k: np.random.default_rng(seed + 1000) for k in self.worlds}
        self.events: list[tuple[int, str]] = []
        self.last_rec = None
        self._entered: list[int] = []

    # ----------------------------------------------------------------- guests
    def _guest(self, purpose: str = "restore", **kw) -> Occupant:
        m = self._maker
        m.t = self.t
        o = m._new_occupant(purpose, **kw)
        o.dwell_steps = 1e9                 # zones never drop people; the plan moves them
        o.gid = self.next_gid
        self.next_gid += 1
        r = self.rng
        laps = int(r.integers(2, 4))
        plan = [("arriving", 1)]
        for _ in range(laps):
            plan += [("hot_pool", int(r.integers(3, 7))), ("sauna", int(r.integers(4, 7))),
                     ("cold_plunge", 1), ("terrace", int(r.integers(5, 11)))]
        if r.random() < 0.5:
            plan.append(("firepit", int(r.integers(4, 9))))
        o.plan, o.leg, o.zone, o.until = plan, 0, "arriving", self.t + 1
        return o

    def _arrive(self, people: list[Occupant]) -> None:
        if self._entered and len(self._entered) == self.t + 1:
            self._entered[-1] += len(people)
        for w in self.worlds.values():
            for o in people:
                c = copy.deepcopy(o)
                w.guests.append(c)

    def _arrivals(self) -> None:
        r, t = self.rng, self.t
        self._entered.append(0)
        for _ in range(r.poisson(0.55)):
            self._arrive([self._guest()])
        if t == COACH_AT:
            self._arrive([self._guest() for _ in range(12)])
            self.events.append((t, "A coach group of twelve arrives and heads for the circuit."))
        if self.distressed and t == DISTRESSED_AT:
            o = self._guest(arousal=0.9, valence=0.15)
            o.baseline, o.mood_anchor, o.motion_scale = 0.8, 0.15, 0.7
            o.plan = [("arriving", 1), ("terrace", 60)]
            o.wearable = True
            if o.pid is None:
                o.pid, self._maker._next_pid = self._maker._next_pid, self._maker._next_pid + 1
            self._arrive([o])
            self.events.append((t, "A guest who paired a watch at check-in sits alone on the terrace, upset."))
        if t == LOUD_GROUP_AT:
            group = []
            for _ in range(4):
                o = self._guest("social", valence=0.8)
                o.plan = [("arriving", 1), ("terrace", 25), ("firepit", 8)]
                group.append(o)
            self._arrive(group)
            self.events.append((t, "Four friends settle on the quiet terrace, laughing and talking loudly."))

    def _move(self, w: World) -> None:
        for o in list(w.terrace.outbox):
            w.terrace.outbox.remove(o)
            self._place(w, o, "firepit", self.t + 8)
            o.leg = next((i for i, (z, _) in enumerate(o.plan) if z == "firepit"), len(o.plan) - 1)
        for o in w.present:
            if o.zone == "queue" or self.t < o.until:
                continue
            o.leg += 1
            if o.leg >= len(o.plan):
                self._place(w, o, "gone", 1e9)
                continue
            zone, dur = o.plan[o.leg]
            if zone == "sauna" and len([g for g in w.guests if g.zone == "sauna"]) >= w.sauna_seats():
                self._place(w, o, "queue", 1e9)
                o.queued_at = self.t
                continue
            self._place(w, o, zone, self.t + dur)
        # the queue moves into the sauna as seats free up
        while w.queue and len([g for g in w.guests if g.zone == "sauna"]) < w.sauna_seats():
            o = w.queue[0]
            self._place(w, o, "sauna", self.t + o.plan[o.leg][1])

    @staticmethod
    def _place(w: World, o: Occupant, zone: str, until: float) -> None:
        for lst in (w.terrace.occupants, w.firepit.occupants, w.queue):
            if o in lst:
                lst.remove(o)
        if zone == "terrace":
            w.terrace.occupants.append(o)
            w.terrace._entries_since_read += 1
        elif zone == "firepit":
            w.firepit.occupants.append(o)
        elif zone == "queue":
            w.queue.append(o)
        o.zone, o.until = zone, until

    # ------------------------------------------------------------------ world
    def _drift(self, w: World) -> None:
        """Affect outside the two modelled zones: pools and sauna are pleasant, queues are not."""
        for o in w.present:
            if o.zone in ("hot_pool", "sauna"):
                o.valence += 0.06 * (0.75 - o.valence)
                o.arousal += 0.10 * (0.30 - o.arousal)
            elif o.zone == "cold_plunge":
                o.arousal = min(1.0, o.arousal + 0.15)
            elif o.zone == "queue":
                o.valence = max(0.0, o.valence - 0.02)
                o.arousal = min(1.0, o.arousal + 0.03)

    def _quiet_zone(self, w: World, db: float, loud: float, action: str) -> bool:
        """The operator declares the terrace a quiet zone. Loud two reads running, with the soundscape
        already down, means people: ask a member of staff to invite them to the fire pit. Each
        controller uses its own definition of loud (the model's loud bin, the rulebook's 62 dB)."""
        w.team_noise = w.team_noise + 1 if db > loud else 0
        if (w.team_noise >= 2 and w.terrace.sound_setting <= 0.2 and action not in ("maintenance_ticket",)
                and self.t - w.quiet_called >= 15 and w.terrace.levers()["staff_alert"]):
            w.quiet_called = self.t
            w.team_noise = 0
            return True
        return False

    def _expected_at_sauna(self, w: World) -> float:
        """Sentient only: the entrance door count. About half of new arrivals reach the sauna within
        ten minutes (hot pool first). Expected queue = waiting now + those on their way - seats free."""
        free = w.sauna_seats() - len(w.zone("sauna"))
        return len(w.queue) + 0.5 * sum(self._entered[-5:]) - free

    def _sauna_reading(self, w: World) -> Reading:
        n = len(w.queue)
        return Reading(count=n + len(w.zone("sauna")) // 3, entries=0, motion_level=0.3 + 0.03 * n,
                       db=45.0 + 1.5 * n, burstiness=0.2, temp_c=22.0, co2_ppm=500.0, queue=n)

    def _sauna_levers(self, w: World) -> dict[str, bool]:
        lv = {k: False for k in ("dim_lights", "raise_lights", "lower_sound", "raise_sound", "cool_air",
                                 "warm_air", "ventilate", "maintenance_ticket")}
        lv["staff_alert"] = w.second_sauna == 0
        return lv

    def _terrace_levers(self, w: World) -> dict[str, bool]:
        lv = w.terrace.levers()
        lv["cool_air"] = False                       # outdoors in October: heaters only
        lv["ventilate"] = False                      # it is already outside
        return lv

    def _open_second(self, w: World) -> None:
        w.second_sauna = SECOND_SAUNA_STEPS

    def _decide(self, w: World, where: str, action: str, target: int | None, reason: str) -> bool:
        """Simulated staff accept or dismiss, the same way for every world. Returns applied."""
        if action == "do_nothing":
            return False
        if where == "sauna":
            dismiss = len(w.queue) < 3 and self.staff_rng[w.key].random() < 0.7
        else:
            w.terrace.staff_target = target if action == "staff_alert" else None
            dismiss = simulated_override(action, w.terrace, self.staff_rng[w.key])
        w.actions.append((self.t, action if where == "terrace" else "open_second_sauna", "dismissed" if dismiss else reason))
        if action in ("staff_alert", "maintenance_ticket"):
            w.calls += int(not dismiss and action == "staff_alert")
            w.dismissed += int(dismiss)
        if dismiss:
            return False
        if where == "sauna":
            self._open_second(w)
        else:
            if action == "maintenance_ticket":
                w.tickets += 1
                if w.terrace.hvac_degraded and w.ticket_step is None:
                    w.ticket_step = self.t
            w.terrace.apply(action)
        return True

    def _team(self, w: World) -> None:
        """Today's spa, run well by eye."""
        r = self.staff_rng["team"]
        if len(w.queue) >= 6 and w.second_sauna == 0:
            self._open_second(w)
            w.actions.append((self.t, "open_second_sauna", "staff see six waiting"))
        hour = START_HOUR + self.t * STEP_MINUTES / 60.0
        if hour >= 18.5 and not w.terrace.heating:
            w.terrace.apply("warm_air")
            w.actions.append((self.t, "warm_air", "heaters on, as every evening at 18:30"))
        w.team_noise = w.team_noise + 1 if w.last_db > 60 else 0
        if w.team_noise >= 4 and w.terrace.alert_cooldown == 0:
            w.terrace.apply("staff_alert")
            w.calls += 1
            w.team_noise = 0
            w.actions.append((self.t, "staff_alert", "staff notice the noise and invite the group to the fire pit"))
        on_terrace = w.terrace.n > 0
        w.team_cold = w.team_cold + 1 if (on_terrace and w.terrace.temp < 19.0) else 0
        if w.team_cold >= 10 and w.terrace.ticket_cooldown == 0:
            w.terrace.apply("maintenance_ticket")
            w.tickets += 1
            if w.terrace.hvac_degraded and w.ticket_step is None:
                w.ticket_step = self.t
            w.team_cold = 0
            w.actions.append((self.t, "maintenance_ticket", "guests complain of the cold"))
        upset = [o for o in w.terrace.occupants if o.mood_anchor is not None]
        if upset and w.terrace.attend_timer == 0 and r.random() < 1 / 15:
            w.terrace.attending = upset[0]
            w.terrace.attend_timer = int(r.integers(4, 8))
            w.calls += 1
            w.actions.append((self.t, "staff_alert", "a staff member passing by notices the guest"))

    def step(self) -> None:
        t = self.t
        self._arrivals()
        if t == HEATER_FAILS_AT:
            for w in self.worlds.values():
                w.terrace.hvac_degraded = True
            self.events.append((t, "One terrace heater fails. It stays switched on, and gives no heat."))
        for w in self.worlds.values():
            # outdoors, bodies do not warm the air: unheated, the terrace feels about 4 C above the air
            # outside (shelter, warm stone); the base room formula is solved for that
            w.terrace.cfg.outdoor_temp = 20.0 + (outdoor_temp(t) + 4.0 - 21.5 - 0.25 * w.terrace.n) / 0.35
            w.firepit.cfg.outdoor_temp = outdoor_temp(t) + 6.0
            self._move(w)
            if w.second_sauna > 0:
                w.second_sauna -= 1
                w.second_sauna_steps += 1

        # Sentient
        w = self.worlds["sentient"]
        reading = w.terrace.read()
        w.last_db = reading.db
        rec = self.agents["terrace"].step(reading, self._terrace_levers(w))
        if self._quiet_zone(w, reading.db, 58.0, rec.action):
            rec.action, rec.target = "staff_alert", None
            rec.reason = f"quiet zone loud for 4 min ({reading.db:.0f} dB)"
        applied = self._decide(w, "terrace", rec.action, getattr(rec, "target", None), _plain(rec))
        if rec.action != "do_nothing":
            self.agents["terrace"].feedback(rec, accepted=applied)
        mode = "warm_air" if w.terrace.heating else None
        self.agents["terrace"].commit(rec.action if applied else "do_nothing", hvac_mode=mode)
        self.last_rec = rec
        srec = self.agents["sauna"].step(self._sauna_reading(w), self._sauna_levers(w))
        if srec.action != "staff_alert":
            srec.action = "do_nothing"           # the only lever at the sauna door is a second sauna
        why = (f"the sauna queue will reach {(self.agents['sauna'].queue_forecast or 0):.0f} "
               "in six minutes at this rate")
        coming = self._expected_at_sauna(w)
        if srec.action == "do_nothing" and w.second_sauna == 0 and coming >= 4:
            srec.action = "staff_alert"
            why = f"{sum(self._entered[-5:])} guests came in through the door in ten minutes; about {coming:.0f} will be waiting for the sauna soon"
        sapplied = self._decide(w, "sauna", srec.action, None, why)
        if srec.action != "do_nothing":
            self.agents["sauna"].feedback(srec, accepted=sapplied)
        self.agents["sauna"].commit(srec.action if sapplied else "do_nothing")

        # Rulebook
        w = self.worlds["rules"]
        reading = w.terrace.read()
        w.last_db = reading.db
        ctrl = self.rules["terrace"]
        a = ctrl.step(reading, self._terrace_levers(w), w.terrace.cooling, w.terrace.light, w.terrace.heating)
        if self._quiet_zone(w, reading.db, 62.0, a):
            a, ctrl.target = "staff_alert", None
        self._decide(w, "terrace", a, ctrl.target, "rule")
        sa = self.rules["sauna"].step(self._sauna_reading(w), self._sauna_levers(w), False, 0.5)
        if sa == "staff_alert":
            self._decide(w, "sauna", sa, None, "four or more waiting")

        # Team
        w = self.worlds["team"]
        w.last_db = w.terrace.read().db
        self._team(w)

        for w in self.worlds.values():
            self._drift(w)
            w.terrace.step()
            w.firepit.step()
            self._score(w)
        self.t += 1

    def _score(self, w: World) -> None:
        present = w.present
        if present:
            w.mood.append(float(np.mean([o.valence for o in present if o.zone != "arriving"] or [0.5])))
        w.queue_minutes += len(w.queue) * STEP_MINUTES
        w.queue_max = max(w.queue_max, len(w.queue))
        if w.terrace.n and w.terrace.temp < 20.5:
            w.cold_minutes += w.terrace.n * STEP_MINUTES
        resting = [o for o in w.terrace.occupants if o.purpose != "social"]
        if resting and w.last_db > 58:
            w.noise_minutes += len(resting) * STEP_MINUTES
        if any(o.mood_anchor is not None for o in w.terrace.occupants):
            w.distress_minutes += STEP_MINUTES
        w.energy += (0.2 + 1.0 * w.terrace.light + (1.2 if w.terrace.heating else 0.0)
                     + (1.5 if w.second_sauna > 0 else 0.0))

    def summary(self) -> dict:
        out = {}
        for k, w in self.worlds.items():
            out[k] = {
                "mood": round(float(np.mean(w.mood)), 3) if w.mood else None,
                "queue_minutes": round(w.queue_minutes, 1), "queue_max": w.queue_max,
                "cold_guest_minutes": round(w.cold_minutes, 1), "noise_guest_minutes": round(w.noise_minutes, 1),
                "distress_minutes": round(w.distress_minutes, 1), "energy": round(w.energy, 1),
                "staff_calls": w.calls, "dismissed": w.dismissed, "tickets": w.tickets,
                "heater_ticket_minutes": None if w.ticket_step is None else (w.ticket_step - HEATER_FAILS_AT) * STEP_MINUTES,
                "second_sauna_minutes": w.second_sauna_steps * STEP_MINUTES,
            }
        return out

    def clock(self, t: int | None = None) -> str:
        h = START_HOUR + (self.t if t is None else t) * STEP_MINUTES / 60.0
        return f"{int(h):02d}:{round((h % 1) * 60) % 60:02d}"

    def view(self, k: str) -> dict:
        """What the page draws: people by zone, the terrace, the saunas. No model internals."""
        w = self.worlds[k]
        return {
            "people": [{"id": o.gid, "zone": o.zone, "mood": round(o.valence, 2), "stress": round(o.arousal, 2),
                        "social": o.purpose == "social", "wearable": o.pid is not None and o.wearable,
                        "distressed": o.mood_anchor is not None,
                        "q": w.queue.index(o) if o in w.queue else None}
                       for o in w.present if o.zone != "arriving"],
            "temp": round(w.terrace.temp, 1), "outdoor": round(outdoor_temp(self.t), 1),
            "heaters": w.terrace.heating, "heater_broken": w.terrace.hvac_degraded,
            "fixing": w.terrace.repair_timer > 0, "second_sauna": w.second_sauna > 0,
            "staff": w.terrace.attend_timer > 0, "light": w.terrace.light, "sound": w.terrace.sound_setting,
            "db": round(w.last_db, 1), "queue": len(w.queue),
        }


class SpaRules(RuleController):
    """The rulebook, with a heater-fault detector that mirrors its cooling one."""

    def __init__(self, intent: str):
        super().__init__(intent)
        self.cold_while_heating = 0

    def step(self, r: Reading, levers: dict[str, bool], cooling: bool, light: float, heating: bool = False) -> str:
        self.cold_while_heating = self.cold_while_heating + 1 if (heating and r.temp_c < 20.0) else 0
        if self.cold_while_heating >= 8 and levers["maintenance_ticket"]:
            self.cold_while_heating = 0
            self.t += 1
            self.target = None
            return "maintenance_ticket"
        return super().step(r, levers, cooling, light)


PLAIN = {
    "dim_lights": "lower the terrace lights", "raise_lights": "raise the terrace lights",
    "lower_sound": "lower the soundscape", "raise_sound": "raise the soundscape",
    "warm_air": "turn the terrace heaters on", "staff_alert": "send a member of staff",
    "maintenance_ticket": "call maintenance", "setback": "power down the empty terrace",
}


def _plain(rec) -> str:
    r = rec.reason
    if "quiet zone" in r:
        return "the quiet terrace has been loud for four minutes; someone can invite the group to the fire pit"
    if "paired guest" in r:
        return "one guest's own watch reads stressed and low for several minutes; someone should go to them"
    if "not responding" in r:
        return "the heaters are on and the terrace keeps getting colder; a heater has likely failed"
    if rec.action == "warm_air":
        return "the terrace is getting cold for guests resting there"
    if rec.action == "staff_alert":
        return "the terrace reads tense; a member of staff can check on the guests there"
    if rec.action == "setback":
        return "nobody has been on the terrace for a while; heaters and lights go down until someone comes back"
    if rec.action == "dim_lights":
        return "the terrace is for rest and the lights are up; lower light helps people settle"
    if rec.action == "lower_sound":
        return "the terrace is louder than a quiet zone should be; the soundscape goes down first"
    return PLAIN.get(rec.action, rec.action)


def run(seed: int, distressed: bool = True, record: bool = False) -> dict:
    ev = SpaEvening(seed, distressed=distressed)
    frames = []
    for _ in range(STEPS):
        ev.step()
        if record:
            frames.append({"clock": ev.clock(ev.t - 1), "w": {k: ev.view(k) for k in ev.worlds}})
    out = {"seed": seed, "summary": ev.summary(),
           "actions": {k: [(ev.clock(t), a, why) for t, a, why in w.actions] for k, w in ev.worlds.items()},
           "events": [(ev.clock(t), e) for t, e in ev.events]}
    if record:
        out["frames"] = frames
    return out


if __name__ == "__main__":
    import json
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    seeds = range(int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else range(1)
    for s in seeds:
        res = run(s)
        print(json.dumps({"seed": s, **res["summary"]}), flush=True)

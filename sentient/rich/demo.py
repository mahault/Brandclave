"""Live demo: the real engine on a simulated room, driven from a browser.

    python -m sentient.rich.demo            # then open http://127.0.0.1:8765

Three copies of the same room run in lockstep from the same seed: Sentient, the rulebook
(sentient/rich/rules.py) and the fixed preset. Faults and arrivals injected from the page
reach all three, so the scoreboard stays a fair comparison. In "you decide" mode the viewer
plays the staff and accepts or dismisses each Sentient recommendation; the rulebook's
staff stay simulated. Runs locally: nothing about the model leaves this machine.
"""

import argparse
import copy
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from sentient.baseline import FixedPresetController
from sentient.ledger import RecommendationLedger

from .agent import RichAgent
from .rules import RuleController
from .run import Outcomes
from .scenarios import SCENARIOS, simulated_override
from .space import STEP_MINUTES, Space, SpaceConfig
from .spec import FACTORS, INTENTS, MODALITIES, discretize

logging.getLogger("jax").setLevel(logging.ERROR)

PAGE = Path(__file__).with_name("demo.html")


class LoungeSpace(Space):
    """A lounge, not a queue. A staff call sends someone to the guest who looks worst off, and that
    guest is the one helped. (In the base world a visit lifts everyone in the room at once, which is
    right for one guest alone and too generous for a lounge of ten; and a call to a busy room thins
    it, which models a desk.)"""

    attending: object = None
    attend_timer: int = 0

    def apply(self, action: str) -> None:
        if action == "staff_alert":
            self.alert_cooldown = 8
            if self.occupants:
                self.attending = max(self.occupants, key=lambda o: o.arousal - o.valence)
                self.attend_timer = int(self.rng.integers(4, 8))
            return
        super().apply(action)

    def step(self) -> None:
        super().step()
        o = self.attending
        if self.attend_timer > 0 and o in self.occupants:
            o.mood_anchor = None                                  # they were helped
            o.valence = float(min(1.0, o.valence + 0.15 * 0.25))  # the same lift a visit gives, to them
            if o.arousal > 0.6:
                o.arousal += 0.25 * (0.35 - o.arousal)
            self.attend_timer -= 1


def lounge_afternoon(seed: int = 0) -> Space:
    """A hotel lounge from four in the afternoon: two guests resting, one working, small groups
    coming and going. The script below adds a distressed guest, a failing air conditioner and a
    group of four at set times, identically in all three rooms. Demo only: not in the evaluated set."""
    space = LoungeSpace(SpaceConfig(
        name="lounge_afternoon", intent="restore", purpose="social", n_occupants=0, capacity=12,
        arrival_rate=None, dwell_steps=1e6, wearable_rate=0.4, light=0.85, sound_setting=0.55,
        cooling=True, outdoor_temp=31.0, start_hour=16.0, seed=seed,
    ))
    for purpose in ("restore", "restore", "work"):
        o = space._new_occupant(purpose)
        o.dwell_steps = 1e6
        space.occupants.append(o)
    return space


class LobbySpace(Space):
    """A hotel lobby. Guests checking in queue at the desk and are served one every two steps (about
    four minutes each); each step in the queue makes them a little more tense. A staff call opens a
    second desk for eight steps, so the queue moves twice as fast. Served guests head for the lifts."""

    desk2: int = 0

    def apply(self, action: str) -> None:
        if action == "staff_alert":
            self.alert_cooldown = 8
            self.desk2 = 8
            return
        super().apply(action)

    def read(self):
        r = super().read()
        seen = [o for o in self.queue() if self._lux_rng.random() > self.cfg.radar_miss]
        r.queue = len(seen)          # the radar's queue zone (own random stream: the world is unchanged)
        return r

    def queue(self) -> list:
        q = [o for o in self.occupants if getattr(o, "role", "") == "checkin" and not getattr(o, "served", False)]
        return sorted(q, key=lambda o: (o.arrived_at, getattr(o, "order", 0)))

    def step(self) -> None:
        q = self.queue()
        serve = (1 if self.t % 2 == 0 else 0) + (1 if self.desk2 > 0 and self.t % 2 == 1 else 0)
        for o in q[:serve]:
            o.served = True
            o.leave_at = self.t + 1
        for o in q[serve:]:
            o.baseline = min(0.9, o.baseline + 0.02)   # waiting wears on people
        if self.desk2 > 0:
            self.desk2 -= 1
        super().step()


def hotel_lobby(seed: int = 0) -> Space:
    """A hotel lobby from three in the afternoon, with a check-in rush around 15:20 to 15:50.
    Demo only: not in the evaluated set."""
    space = LobbySpace(SpaceConfig(
        name="hotel_lobby", intent="service", purpose="transit", n_occupants=0, capacity=40,
        arrival_rate=None, dwell_steps=1e6, wearable_rate=0.25, light=0.8, sound_setting=0.35,
        cooling=True, outdoor_temp=29.0, start_hour=15.0, seed=seed,
    ))
    for role, purpose in (("waiting", "restore"), ("waiting", "restore"), ("bar", "social")):
        o = space._new_occupant(purpose)
        o.role, o.leave_at = role, 12 + int(space.rng.integers(0, 8))
        space.occupants.append(o)
    return space


DEMO_SCENARIOS = {"lounge_afternoon": lounge_afternoon, "hotel_lobby": hotel_lobby}
ROLE_PURPOSE = {"checkin": "transit", "pass": "transit", "waiting": "restore", "meetup": "social", "bar": "social"}
# step -> injection, applied to all three rooms at the same moment
SCRIPTS = {"lounge_afternoon": {10: "ac_fail", 26: "group"}}
# walk-in visitors: (groups per step, mean stay in steps). Drawn once and copied into all three
# rooms, so the same people come and go whoever is in charge.
VISITORS = {"lounge_afternoon": (0.14, 18)}
SHOWN_BELIEFS = ("arousal", "valence", "occupancy", "thermal", "air", "hvac", "lighting")
HISTORY = 90


class Session:
    def __init__(self, scenario: str, seed: int, staff: str):
        build = DEMO_SCENARIOS.get(scenario) or SCENARIOS.get(scenario)
        if build is None:
            raise HTTPException(400, f"unknown scenario {scenario}")
        self.scenario, self.seed, self.staff = scenario, seed, staff
        self.script = dict(SCRIPTS.get(scenario, {}))
        self.visitors = VISITORS.get(scenario)
        # people are created once, from their own stream, and copied into every room
        self._maker = build(seed)
        self._maker.rng = np.random.default_rng(seed + 777)
        self._meet_id = 0
        self._partners: list[tuple[int, int]] = []
        self.world = {k: build(seed) for k in ("sentient", "rules", "preset")}
        FixedPresetController().install(self.world["preset"])
        intent = self.world["sentient"].cfg.intent
        self.agent = RichAgent(intent=intent)
        self.rules = RuleController(intent)
        self.staff_rng = {"sentient": np.random.default_rng(seed + 1000), "rules": np.random.default_rng(seed + 1000)}
        self.ledger = RecommendationLedger(scenario=scenario, source="live-demo")
        self.pending: list[tuple[str, int, dict]] = []
        self.score = {k: Outcomes() for k in self.world}
        self.t = 0
        self.awaiting = False
        self.rec = None
        self.reading = None
        self.history: list[dict] = []
        self.events: list[str] = []
        self.tampered: set[str] = set()
        self.traces: dict[str, dict] = {}   # one complete example per decision, for the "follow one decision" panel

    # ------------------------------------------------------------------ stepping
    def step(self, decision: str | None) -> None:
        if self.awaiting:
            if decision not in ("accept", "dismiss"):
                raise HTTPException(400, "a recommendation is waiting: accept or dismiss it")
            self._finish(overridden=(decision == "dismiss"))
            return
        self._comings_and_goings()
        if self.t in self.script:
            self.inject(self.script.pop(self.t), scripted=True)
        space = self.world["sentient"]
        self.reading = space.read()
        obs = discretize(self.reading)
        still = []
        for rid, due, persist in self.pending:
            if self.t >= due:
                self.ledger.resolve_all(rid, obs, persist)
            else:
                still.append((rid, due, persist))
        self.pending = still
        self._obs = obs
        self.rec = self.agent.step(self.reading, space.levers())
        if self.rec.action != "do_nothing" and self.staff == "you":
            self.awaiting = True
            return
        self._finish(overridden=simulated_override(self.rec.action, space, self.staff_rng["sentient"]))

    def _finish(self, overridden: bool) -> None:
        space, rec = self.world["sentient"], self.rec
        self.awaiting = False
        applied = "do_nothing" if overridden else rec.action
        rid = self.ledger.record(self.t, rec, applied, overridden)
        if rec.action != "do_nothing":
            r, b = self.reading, rec.beliefs
            def top(f):
                return max(b[f].items(), key=lambda kv: kv[1])
            self.traces[rid] = {
                "id": rid, "time": self._clock(), "sealed_at": datetime.now().astimezone().strftime("%H:%M:%S"),
                "hash": self.ledger._records[rid]["hash"][:16],
                "inputs": {"radar count": r.count, "door entries": r.entries, "movement": round(r.motion_level, 2),
                           "sound": f"{r.db:.0f} dB", "temperature": f"{r.temp_c:.1f} C", "CO2": f"{r.co2_ppm:.0f} ppm",
                           "heart rate": "no wearable" if r.hr is None else f"{r.hr:.0f}",
                           **({"queue zone": r.queue} if r.queue is not None else {})},
                "beliefs": {f: [top(f)[0], round(top(f)[1], 2)] for f in ("occupancy", "arousal", "valence", "thermal", "hvac")},
                "action": rec.action, "reason": rec.reason,
                "forecast": {"channel": rec.prediction.modality, "predicted": MODALITIES[rec.prediction.modality][rec.prediction.predicted],
                             "probability": round(rec.prediction.probability, 2), "due": self._clock_at(self.t + rec.prediction.horizon)},
                "also_sealed": {m: MODALITIES[m][rec.forecasts[m][0]] for m in ("hr", "motion", "sound_level") if m in rec.forecasts},
                "decision": "dismissed" if overridden else "accepted",
                "decided_by": "you" if self.staff == "you" else "simulated staff",
            }
        self.pending.append((rid, self.t + rec.prediction.horizon, dict(self._obs)))
        self.score["sentient"].record(space, self.t, rec.action, overridden)
        if rec.action != "do_nothing":
            self.agent.feedback(rec, accepted=not overridden)
            self.events.append(f"{self._clock()}  Sentient: {_label(rec.action)}"
                               + ("  (dismissed)" if overridden else ""))
        space.apply(applied)
        hvac_mode = "cool_air" if space.cooling else ("warm_air" if space.heating else None)
        self.agent.commit(applied, hvac_mode=hvac_mode)
        space.step()
        self.score["sentient"].after_step(space)

        rw = self.world["rules"]
        action = self.rules.step(rw.read(), rw.levers(), rw.cooling, rw.light)
        ov = simulated_override(action, rw, self.staff_rng["rules"])
        self.score["rules"].record(rw, self.t, action, ov)
        rw.apply("do_nothing" if ov else action)
        rw.step()
        self.score["rules"].after_step(rw)

        pw = self.world["preset"]
        pw.step()
        self.score["preset"].after_step(pw)

        b = rec.beliefs
        self.history.append({
            "t": self.t,
            "p_stressed": round(b["arousal"]["stressed"], 3),
            # expected stress level on the world's 0-1 scale: calm ~0.2, tense ~0.5, stressed ~0.82
            "est_arousal": round(0.2 * b["arousal"]["calm"] + 0.5 * b["arousal"]["elevated"]
                                 + 0.82 * b["arousal"]["stressed"], 3),
            "true_arousal": round(space.arousal, 3) if space.n else None,
            "p_positive": round(b["valence"]["positive"], 3),
            "true_mood": round(space.mean_valence(), 3) if space.n else None,
            "p_ac_broken": round(b["hvac"]["degraded"], 3),
            "ac_broken": space.hvac_degraded,
            "queues": {k: (len(w.queue()) if hasattr(w, "queue") else None) for k, w in self.world.items()},
            "action": rec.action if rec.action != "do_nothing" else None,
            "dismissed": overridden,
        })
        self.history = self.history[-HISTORY:]
        self.t += 1

    def _comings_and_goings(self) -> None:
        for w in self.world.values():
            leaving = [o for o in w.occupants if getattr(o, "leave_at", 1e9) <= self.t]
            if leaving:
                w.occupants = [o for o in w.occupants if o not in leaving]
                w.departures_total += len(leaving)
        if self.scenario == "hotel_lobby":
            self._lobby_arrivals()
            return
        if not self.visitors:
            return
        rate, stay = self.visitors
        rng = self._maker.rng
        busier = 0.35 + 0.9 * min(1.0, self.t / 40)   # quiet at four, filling up toward the evening
        for _ in range(rng.poisson(rate * busier)):
            size = int(rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15]))
            leave = self.t + max(4, int(rng.normal(stay, stay * 0.4)))
            group = [self._person("social", leave_at=leave) for _ in range(size)]
            self._arrive_everywhere(group)

    def _lobby_arrivals(self) -> None:
        rng, t = self._maker.rng, self.t
        rush = max(0.0, 1.0 - abs(t - 18) / 10)           # check-in rush around 15:20-15:50
        for _ in range(rng.poisson(0.22 + 0.75 * rush)):  # arriving guests, alone or as a couple
            size = 1 if rng.random() < 0.6 else 2
            group = [self._role("checkin") for _ in range(size)]
            for i, o in enumerate(group):
                o.order = i
            self._arrive_everywhere(group)
        for _ in range(rng.poisson(0.3)):                  # walking through to the lifts
            self._arrive_everywhere([self._role("pass", leave_at=t + 1)])
        if rng.random() < 0.08:                            # waiting for a taxi or a friend
            self._arrive_everywhere([self._role("waiting", leave_at=t + int(rng.integers(5, 13)))])
        if rng.random() < 0.06:                            # a small group for a drink
            leave = t + int(rng.integers(10, 18))
            self._arrive_everywhere([self._role("bar", leave_at=leave) for _ in range(int(rng.integers(2, 4)))])
        if rng.random() < 0.06:                            # meeting someone: they wait, the friend comes later
            self._meet_id += 1
            first = self._role("meetup")
            first.meet = self._meet_id
            self._arrive_everywhere([first])
            self._partners.append((t + int(rng.integers(3, 8)), self._meet_id))
        for when, mid in [x for x in self._partners if x[0] <= t]:
            self._partners.remove((when, mid))
            leave = t + int(rng.integers(8, 15))
            second = self._role("meetup", leave_at=leave)
            second.meet = mid
            self._arrive_everywhere([second])
            for w in self.world.values():
                for o in w.occupants:
                    if getattr(o, "meet", None) == mid:
                        o.leave_at = leave

    def _role(self, role: str, leave_at: float = 1e9):
        o = self._person(ROLE_PURPOSE[role], leave_at=leave_at)
        o.role = role
        return o

    def _person(self, purpose: str, leave_at: float = 1e9, **kw):
        self._maker.t = self.t
        o = self._maker._new_occupant(purpose, **kw)
        o.leave_at = leave_at
        return o

    def _arrive_everywhere(self, people) -> None:
        for w in self.world.values():
            self._arrive(w, [copy.deepcopy(o) for o in people])

    # ---------------------------------------------------------------- injections
    def inject(self, kind: str, scripted: bool = False) -> None:
        if kind == "stressed_guest":
            o = self._person("restore", arousal=0.9, valence=0.15)
            o.baseline, o.mood_anchor, o.wearable, o.motion_scale = 0.8, 0.15, True, 0.7
            self._arrive_everywhere([o])
        elif kind == "group":
            leave = self.t + 20
            self._arrive_everywhere([self._person("social", leave_at=leave, valence=0.8) for _ in range(4)])
        for w in self.world.values():
            if kind == "ac_fail":
                w.hvac_degraded = True
            elif kind == "ac_fix":
                w.hvac_degraded = False
            elif kind == "mic_dead":
                w.cfg.stuck_db = 41.0
            elif kind == "mic_fix":
                w.cfg.stuck_db = None
            elif kind in ("stressed_guest", "group"):
                pass
            elif kind == "everyone_leaves":
                w.departures_total += w.n
                w.occupants = []
            else:
                raise HTTPException(400, f"unknown injection {kind}")
        self.events.append(f"{self._clock()}  {'Meanwhile' if scripted else 'You'}: {INJECT_LABEL[kind]}")

    @staticmethod
    def _arrive(w, people) -> None:
        for o in people:
            o.arrived_at = w.t
            w.occupants.append(o)
            w.arrivals_total += 1
            w._entries_since_read += 1

    def set_intent(self, intent: str) -> None:
        if intent not in INTENTS:
            raise HTTPException(400, f"unknown intent {intent}")
        old = self.agent
        new = RichAgent(intent=intent)
        keep = {"intent", "A", "B", "C", "D", "E", "agent"}
        for k, v in old.__dict__.items():
            if k not in keep:
                setattr(new, k, v)
        self.agent = new
        self.rules = RuleController(intent)
        for w in self.world.values():
            w.cfg.intent = intent
        self.events.append(f"{self._clock()}  You: the room is now for {INTENT_LABEL[intent]}")

    # -------------------------------------------------------------------- views
    def _clock(self) -> str:
        h = self.world["sentient"].hour
        return f"{int(h):02d}:{round((h % 1) * 60) % 60:02d}"

    def state(self) -> dict:
        w = self.world["sentient"]
        rec = self.rec
        recs = [e for e in self.ledger.events if e["type"] == "recommendation"]
        outs = [e for e in self.ledger.events if e["type"] == "outcome" and e.get("scored", True)]
        tail = []
        for e in recs[-8:][::-1]:
            out = next((o for o in outs if o["id"] == e["id"]), None)
            pred = e["prediction"]
            tail.append({
                "id": e["id"], "time": self._clock_at(e["t"]), "action": _label(e["action"]),
                "applied": e["applied"] != "do_nothing" or e["action"] == "do_nothing",
                "forecast": f"{pred['modality'].replace('_', ' ')} {MODALITIES[pred['modality']][pred['predicted']]}",
                "hash": e["hash"][:16],
                "result": None if out is None else ("hit" if out["hit"] else "miss"),
                "tampered": e["id"] in self.tampered,
            })
        trace = None
        for rid in sorted(self.traces, reverse=True):
            tr = dict(self.traces[rid])
            out = next((o for o in outs if o["id"] == rid), None)
            if out is not None or trace is None:
                mod = tr["forecast"]["channel"]
                tr["outcome"] = None if out is None else {
                    "actual": MODALITIES[mod][out["actual"]], "hit": out["hit"],
                    "same_as_now": MODALITIES[mod][out["persistence_predicted"]], "same_as_now_hit": out["persistence_hit"],
                    "scored": out.get("scored", True),
                    "also": {o["modality"]: [MODALITIES[o["modality"]][o["actual"]], o["hit"]]
                             for o in self.ledger.events if o["type"] == "channel_outcome" and o["id"] == rid
                             and o["modality"] in ("hr", "motion", "sound_level")}}
                trace = tr
            if out is not None:
                break
        scores = {}
        for k, o in self.score.items():
            s = o.summary()
            scores[k] = {"mood": s["mood_mean"], "energy": s["energy"], "wasted": s["dismissed"],
                         "recommendations": s["recommendations"], "staff_calls": s["staff_alerts"],
                         "tickets": s["tickets"]}
        r = self.reading
        return {
            "t": self.t, "clock": self._clock(), "minutes": int(self.t * STEP_MINUTES),
            "scenario": self.scenario, "seed": self.seed, "staff": self.staff,
            "intent": self.agent.intent, "awaiting": self.awaiting,
            "room": {
                "people": w.n, "temp": round(w.temp, 1), "co2": round(w.co2), "sound": round(w.sound_setting, 1),
                "light": round(w.light, 1), "cooling": w.cooling, "heating": w.heating,
                "ac_broken": w.hvac_degraded, "mic_dead": w.cfg.stuck_db is not None,
                "true_arousal": round(w.arousal, 2) if w.n else None,
                "true_mood": round(w.mean_valence(), 2) if w.n else None,
            },
            "reading": None if r is None else {
                "count": r.count, "db": round(r.db, 1), "temp": round(r.temp_c, 1), "co2": round(r.co2_ppm),
                "hr": None if r.hr is None else round(r.hr), "hrv": None if r.hrv is None else round(r.hrv),
                "motion": round(r.motion_level, 2), "lux": None if r.lux is None else round(r.lux),
            },
            "beliefs": None if rec is None else {f: rec.beliefs[f] for f in SHOWN_BELIEFS},
            "states": {f: FACTORS[f] for f in SHOWN_BELIEFS},
            "rec": None if rec is None else {
                "action": _label(rec.action), "raw": rec.action, "reason": rec.reason,
                "confidence": round(rec.confidence, 2),
                "forecast": f"{rec.prediction.modality.replace('_', ' ')} will read "
                            f"{MODALITIES[rec.prediction.modality][rec.prediction.predicted]} in "
                            f"{int(rec.prediction.horizon * STEP_MINUTES)} min "
                            f"({rec.prediction.probability:.0%})",
            },
            "flags": sorted(self.agent.flagged),
            "queue_forecast": None if self.agent.queue_forecast is None else round(self.agent.queue_forecast, 1),
            "worlds": {k: self._world_view(k) for k in self.world},
            "purpose": w.cfg.purpose, "layout": "lobby" if self.scenario == "hotel_lobby" else "lounge", "gated": None if rec is None else rec.gated,
            "scores": scores,
            "forecasts": {"scored": len(outs), "hits": sum(o["hit"] for o in outs),
                          "same_as_now": sum(o["persistence_hit"] for o in outs)},
            "history": self.history,
            "ledger": tail,
            "trace": trace,
            "events": self.events[-8:][::-1],
        }

    def _world_view(self, key: str) -> dict:
        w, s = self.world[key], self.score[key].summary()
        return {
            "people": [{"id": id(o), "mood": round(o.valence, 2), "stress": round(o.arousal, 2),
                        "activity": ["rest", "work", "social", "transit"][o.activity],
                        "wearable": o.wearable, "distressed": o.mood_anchor is not None,
                        "role": getattr(o, "role", None), "served": getattr(o, "served", False),
                        "meet": getattr(o, "meet", None),
                        "queue": next((i for i, x in enumerate(w.queue()) if x is o), None)
                        if hasattr(w, "queue") else None}
                       for o in w.occupants],
            "light": round(w.light, 2), "sound": round(w.sound_setting, 2), "temp": round(w.temp, 1),
            "co2": round(w.co2), "cooling": w.cooling, "heating": w.heating,
            "ventilating": w.ventilating > 0, "ac_broken": w.hvac_degraded, "mic_dead": w.cfg.stuck_db is not None,
            "staff_here": w.comfort_timer > 0 or w.relief_timer > 0 or getattr(w, "attend_timer", 0) > 0,
            "repair_here": w.repair_timer > 0,
            "mood": s["mood_mean"], "energy": s["energy"], "wasted": s["dismissed"],
            "staff_calls": s["staff_alerts"], "tickets": s["tickets"],
            "desk2": getattr(w, "desk2", 0) > 0,
            "queue": len(w.queue()) if hasattr(w, "queue") else None,
        }

    def _clock_at(self, t: int) -> str:
        h = (self.world["sentient"].cfg.start_hour + t * STEP_MINUTES / 60.0) % 24.0
        return f"{int(h):02d}:{round((h % 1) * 60) % 60:02d}"


LABEL = {
    "do_nothing": "Hold", "dim_lights": "Dim the lights", "raise_lights": "Raise the lights",
    "lower_sound": "Lower the music", "raise_sound": "Raise the music", "cool_air": "Cool the room",
    "warm_air": "Warm the room", "ventilate": "Bring in fresh air", "staff_alert": "Send a person",
    "maintenance_ticket": "Call maintenance", "setback": "Power down the empty room",
}
INJECT_LABEL = {
    "ac_fail": "air conditioner fails silently", "ac_fix": "air conditioner repaired",
    "mic_dead": "microphone dies", "mic_fix": "microphone fixed",
    "stressed_guest": "a distressed guest walks in", "group": "a group of four friends arrives",
    "everyone_leaves": "everyone leaves",
}
INTENT_LABEL = {"restore": "rest", "work": "work", "elevated": "a lively evening", "service": "service"}


def _label(action: str) -> str:
    return LABEL.get(action, action)


# ------------------------------------------------------------------------- app
app = FastAPI(title="Sentient live demo")
SESSION: Session | None = None


class ResetBody(BaseModel):
    scenario: str = "restore_pod"
    seed: int = 11
    staff: str = "auto"


class StepBody(BaseModel):
    decision: str | None = None
    n: int = 1


def _session() -> Session:
    if SESSION is None:
        raise HTTPException(409, "start a session first")
    return SESSION


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return PAGE.read_text(encoding="utf-8")


@app.get("/api/scenarios")
def scenarios() -> dict:
    return {"scenarios": list(DEMO_SCENARIOS) + list(SCENARIOS), "intents": INTENTS}


@app.post("/api/reset")
def reset(body: ResetBody) -> dict:
    global SESSION
    SESSION = Session(body.scenario, body.seed, body.staff)
    return SESSION.state()


@app.get("/api/state")
def current() -> dict:
    return _session().state()


@app.post("/api/step")
def step(body: StepBody) -> dict:
    s = _session()
    for _ in range(max(1, min(body.n, 30))):
        s.step(body.decision)
        body.decision = None
        if s.awaiting:
            break
    return s.state()


@app.post("/api/inject/{kind}")
def inject(kind: str) -> dict:
    s = _session()
    s.inject(kind)
    return s.state()


@app.post("/api/intent/{intent}")
def intent(intent: str) -> dict:
    s = _session()
    s.set_intent(intent)
    return s.state()


@app.post("/api/staff/{mode}")
def staff(mode: str) -> dict:
    s = _session()
    if mode not in ("auto", "you"):
        raise HTTPException(400, "auto or you")
    s.staff = mode
    return s.state()


@app.get("/api/verify/{rid}")
def verify(rid: str) -> dict:
    s = _session()
    if rid not in s.ledger._records:
        raise HTTPException(404, rid)
    return {"id": rid, "intact": s.ledger.verify(rid)}


@app.post("/api/tamper/{rid}")
def tamper(rid: str) -> dict:
    """Edit a sealed record after the fact, to show that verification catches it."""
    s = _session()
    if rid not in s.ledger._records:
        raise HTTPException(404, rid)
    rec = s.ledger._records[rid]
    rec["confidence"] = round(min(1.0, rec["confidence"] + 0.25), 4)
    s.tampered.add(rid)
    return {"id": rid, "intact": s.ledger.verify(rid)}


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description="Sentient live demo")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    print(f"Sentient live demo: open http://127.0.0.1:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

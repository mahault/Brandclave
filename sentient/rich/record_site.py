"""Record simulated runs for the public Sentient page.

    python -m sentient.rich.record_site            # writes web/sentient/data/*.json

Every run uses seed 12, the same seed the live demo opens on; nothing is picked for looking good.
The files hold only what the page draws (people, rooms, outcomes) and plain-language reasons.
No beliefs, likelihoods or model structure leave the engine.
"""

import json
import warnings
from pathlib import Path

from . import spa
from .demo import Session

OUT = Path("web/sentient/data")
SEED = 12


def _reason(rec, intent: str) -> str:
    r, a = rec.reason, rec.action
    if "paired guest" in r:
        return "One guest's own watch has read stressed and low for several minutes, while the room as a whole looks fine. A person should go to them."
    if "not responding" in r:
        return "Cooling is on and the room keeps warming. The unit has most likely failed, so it calls a technician instead of sending more commands."
    if "in the queue" in r:
        return "At the rate guests are joining, the queue will pass four within six minutes. A second desk now keeps it from forming."
    if "empty" in r:
        return "Radar and door sensor agree nobody has been here for a while."
    return {
        "dim_lights": "Set for rest, and bright. Lower light should help people settle.",
        "raise_lights": "Too dark for what the room is for.",
        "lower_sound": "Louder than a room for rest should be.",
        "raise_sound": "Meant to be lively, and it has gone flat.",
        "cool_air": "Getting warm for the people here.",
        "warm_air": "Getting cold for the people here.",
        "ventilate": "The air is going stale as the room fills.",
        "staff_alert": "People here read as stressed, and the room is already set right. A person can help.",
        "maintenance_ticket": "The equipment is not doing what it was told.",
    }.get(a, "")


LABEL = {"dim_lights": "Dim the lights", "raise_lights": "Raise the lights", "lower_sound": "Lower the music",
         "raise_sound": "Raise the music", "cool_air": "Cool the room", "warm_air": "Warm the room",
         "ventilate": "Bring in fresh air", "staff_alert": "Send a member of staff", "maintenance_ticket": "Call maintenance",
         "setback": "Power down the empty room"}


def record_session(scenario: str, steps: int, script: dict | None = None, title: str = "", intro: str = "") -> dict:
    s = Session(scenario, SEED, "auto")
    if script is not None:
        s.script = dict(script)
    frames = []
    seen_events = 0
    for _ in range(steps):
        before = len(s.events)
        s.step(None)
        st = s.state()
        new = s.events[before:]
        rec = s.rec
        decision = None
        if rec is not None and rec.action != "do_nothing":
            label = LABEL.get(rec.action, rec.action)
            if s.scenario == "hotel_lobby" and rec.action == "staff_alert":
                label = "Open a second check-in desk"
            if getattr(rec, "target", None) is not None:
                label = "Send a member of staff to one guest"
            dismissed = any("(dismissed)" in e for e in new)
            decision = {"action": label, "why": _reason(rec, s.agent.intent), "dismissed": dismissed}
        worlds = {}
        for k, w in st["worlds"].items():
            worlds[k] = {
                "people": [{key: p[key] for key in ("id", "mood", "stress", "activity", "wearable", "distressed",
                                                     "role", "served", "meet", "queue")} for p in w["people"]],
                **{key: w[key] for key in ("light", "sound", "temp", "co2", "cooling", "heating", "ventilating",
                                           "ac_broken", "staff_here", "repair_here", "mood", "energy", "wasted",
                                           "staff_calls", "tickets", "desk2", "queue", "ticket_step")},
            }
        frames.append({"clock": st["clock"], "worlds": worlds, "decision": decision,
                       "events": [e.split("  ", 1)[1] for e in new if "Meanwhile" in e or "You:" in e]})
        seen_events = len(s.events)
    return {"kind": "room", "layout": st["layout"], "title": title, "intro": intro, "seed": SEED,
            "labels": {"sentient": "With Sentient", "rules": "A good rulebook", "preset": "Today's fixed setup"},
            "ac_failed_at": s.ac_failed_at, "frames": frames}


def record_spa(title: str, intro: str) -> dict:
    ev = spa.SpaEvening(SEED)
    frames = []
    for _ in range(spa.STEPS):
        n_events = len(ev.events)
        acts_before = {k: len(w.actions) for k, w in ev.worlds.items()}
        ev.step()
        t = ev.t - 1
        decision = None
        new = ev.worlds["sentient"].actions[acts_before["sentient"]:]
        if new:
            _, a, why = new[-1]
            label = {"open_second_sauna": "Open the second sauna", "warm_air": "Turn the terrace heaters on",
                     "staff_alert": "Send a member of staff", "maintenance_ticket": "Call maintenance",
                     "dim_lights": "Lower the terrace lights", "lower_sound": "Lower the soundscape",
                     "setback": "Power down the empty terrace"}.get(a, a)
            decision = {"action": label, "why": "" if why == "dismissed" else why[0].upper() + why[1:] + ".",
                        "dismissed": why == "dismissed"}
        others = {k: [{"action": a, "why": why} for _, a, why in ev.worlds[k].actions[acts_before[k]:]]
                  for k in ("rules", "team")}
        worlds = {k: {**ev.view(k), **_score(ev.worlds[k])} for k in ev.worlds}
        frames.append({"clock": ev.clock(t), "worlds": worlds, "decision": decision, "others": others,
                       "events": [e for _, e in ev.events[n_events:]]})
    return {"kind": "spa", "layout": "spa", "title": title, "intro": intro, "seed": SEED,
            "labels": {"sentient": "With Sentient", "rules": "A good rulebook", "team": "A good team, by eye"},
            "heater_failed_at": spa.HEATER_FAILS_AT, "frames": frames, "summary": ev.summary()}


def _score(w) -> dict:
    import numpy as np
    return {"mood": round(float(np.mean(w.mood)), 3) if w.mood else None, "energy": round(w.energy, 1),
            "queue_minutes": w.queue_minutes, "cold_minutes": w.cold_minutes, "noise_minutes": w.noise_minutes,
            "distress_minutes": w.distress_minutes, "staff_calls": w.calls, "wasted": w.dismissed,
            "ticket_step": w.ticket_step}


def main() -> None:
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {
        "spa_evening": lambda: record_spa(
            "A thermal spa, one cold evening",
            "Guests move between the hot pool, the sauna, the cold plunge and a quiet terrace with heaters. "
            "The air cools after sunset, a coach group arrives, a heater fails, a loud group drifts into the "
            "quiet zone, and one guest is having a hard evening."),
        "lounge_ac": lambda: record_session(
            "lounge_afternoon", 50, None, "A hotel lounge, one hot afternoon",
            "At 16:20 the air conditioning fails silently: it keeps running, and stops cooling. "
            "There is no sensor on the unit."),
        "lounge_distress": lambda: record_session(
            "lounge_afternoon", 58, {16: "group", 28: "stressed_guest"}, "One guest in a busy lounge",
            "The lounge fills up through the afternoon, and a group of friends arrives at 16:34. At 16:58 a guest "
            "who paired a watch at check-in sits down among them, upset. The room as a whole looks fine."),
        "lobby_rush": lambda: record_session(
            "hotel_lobby", 45, None, "A hotel lobby, the check-in rush",
            "Between 15:20 and 15:50 arrivals pile up at the desk. A second desk can be opened."),
    }
    for name, make in runs.items():
        data = make()
        (OUT / f"{name}.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        print(name, len(data["frames"]), "frames", flush=True)


if __name__ == "__main__":
    main()

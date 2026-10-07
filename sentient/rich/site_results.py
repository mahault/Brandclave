"""Build web/sentient/data/results.json from held-out evaluation summaries.

    python -m sentient.rich.site_results --rooms data/sentient_eval_heldout_v3 --spa data/sentient_spa_eval_heldout

The verdict rule is the one used everywhere else: a difference counts only when it is larger than
the larger of the two 95% intervals and a floor (0.015 for mood, 0.5 for energy and dismissed calls).
Otherwise it is "level". Every number on the page comes from these files.
"""

import argparse
import json
from pathlib import Path

FLOORS = {"mood_mean": 0.015, "energy": 0.5, "dismissed": 0.5}
HIGHER_BETTER = {"mood_mean": True, "energy": False, "dismissed": False}
NAMES = {
    "restore_pod": "Restore pod", "bar_9pm": "Bar at 9pm", "flat_bar": "Flat bar", "ambiguous_hr": "Calm guest, high heart rate",
    "lobby_peak": "Lobby rush", "happy_group": "Happy group", "stale_meeting": "Stuffy meeting room",
    "hvac_failure": "Air conditioner fails", "cold_morning": "Cold morning", "stressed_guest": "Distressed guest alone",
    "sensor_failure": "Dead sound sensor", "busy_lounge_distress": "Upset guest in a busy lounge",
}


def verdict(a: dict | None, b: dict | None, metric: str) -> str | None:
    if not a or not b:
        return None
    gap = a["mean"] - b["mean"]
    if abs(gap) <= max(a["ci"], b["ci"], FLOORS[metric]):
        return "level"
    better = gap > 0 if HIGHER_BETTER[metric] else gap < 0
    return "ahead" if better else "behind"


def rooms(path: Path) -> tuple[dict, dict]:
    s = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    out = {"rules": {"ahead": 0, "level": 0, "behind": 0}, "preset": {"ahead": 0, "level": 0, "behind": 0}}
    behind = []
    for name, r in s.items():
        for m in ("mood_mean", "energy", "dismissed"):
            v = verdict(r["agent"][m], r["rules"][m], m)
            if v:
                out["rules"][v] += 1
                if v == "behind":
                    behind.append(f"{NAMES.get(name, name)}: {'more dismissed calls' if m == 'dismissed' else m.replace('_mean', '')}")
        for m in ("mood_mean", "energy"):
            v = verdict(r["agent"][m], r["preset"][m], m)
            if v:
                out["preset"][v] += 1
    mood_vs_preset = {v: sum(verdict(r["agent"]["mood_mean"], r["preset"]["mood_mean"], "mood_mean") == v for r in s.values())
                      for v in ("ahead", "level", "behind")}
    bl = s.get("busy_lounge_distress")
    table = {
        "note": f"{len(s)} kinds of room, ten unseen runs each. Compared on simulated mood, energy and staff calls that "
                "staff turned down. \"Level\" means the gap is inside the 95% range.",
        "rows": [
            {"against": "A good rulebook, same sensors", **out["rules"]},
            {"against": "Today's fixed setup", **out["preset"]},
        ],
        "notes": ([f"Where the rulebook comes out ahead: {'; '.join(behind)}."] if behind else [])
                 + [f"Simulated mood against today's setup: higher in {mood_vs_preset['ahead']} rooms, level in "
                    f"{mood_vs_preset['level']}, lower in {mood_vs_preset['behind']}."],
    }
    extra = {"mood_vs_preset": mood_vs_preset, "n_rooms": len(s)}
    if bl:
        extra["busy"] = {k: bl[k]["distress_steps"] for k in ("agent", "rules", "preset")}
        extra["busy_dismissed"] = {k: bl[k]["dismissed"] for k in ("agent", "rules")}
    hv = s.get("hvac_failure")
    if hv:
        extra["hvac"] = {k: hv[k]["ticket_step"] for k in ("agent", "rules")}
    return table, extra


def spa(path: Path) -> dict:
    s = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    rows = [
        ("Waiting for the sauna", "guest-minutes in the queue", "queue_minutes"),
        ("Cold on the terrace", "guest-minutes under 20.5 °C", "cold_guest_minutes"),
        ("Rest spoiled by noise", "resting guest-minutes in a loud quiet zone", "noise_guest_minutes"),
        ("Upset guest left alone", "minutes before someone came", "distress_minutes"),
        ("Failed heater reported", "minutes after it failed", "heater_ticket_minutes"),
        ("Energy", "heaters, lights, second sauna (units)", "energy"),
        ("Simulated mood", "0–1, higher is better", "mood"),
    ]
    out = []
    for label, unit, key in rows:
        v = {k: s[k][key] for k in ("sentient", "rules", "team") if s[k].get(key)}
        out.append({"label": label, "unit": unit, "key": key, "v": v})
    return {"n": s["n_seeds"], "note": f"{s['n_seeds']} unseen evenings. Lower is better on every row except mood. "
                                      "Bars are averages; the thin line is the 95% range of the average.", "rows": out, "raw": s}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", default="data/sentient_eval_heldout_v3")
    ap.add_argument("--spa", default="data/sentient_spa_eval_heldout")
    ap.add_argument("--out", default="web/sentient/data/results.json")
    args = ap.parse_args()
    table, extra = rooms(Path(args.rooms))
    sp = spa(Path(args.spa))
    raw = sp.pop("raw")
    m = lambda who, k: raw[who][k]["mean"] if raw[who].get(k) else None  # noqa: E731
    hero = []
    hs, ht = m("sentient", "heater_ticket_minutes"), m("team", "heater_ticket_minutes")
    if hs is not None and ht is not None:
        hero.append({"v": f"{hs:.0f}<small> min vs {ht:.0f}</small>",
                     "t": "Failed spa heater reported, Sentient against a good team by eye (unseen evenings)."})
    qs, qt = m("sentient", "queue_minutes"), m("team", "queue_minutes")
    if qs is not None and qt:
        hero.append({"v": f"{100 * (1 - qs / qt):.0f}%<small> less</small>",
                     "t": "Waiting for the sauna on busy evenings, against a good team opening the second one by eye."})
    if "busy" in extra:
        b = extra["busy"]
        hero.append({"v": f"{2 * b['agent']['mean']:.0f}<small> min vs {2 * b['preset']['mean']:.0f}</small>",
                     "t": "An upset guest in a busy lounge left alone, Sentient against today's setup."})
    mv = extra["mood_vs_preset"]
    hero.append({"v": f"{mv['ahead']}<small> of {extra['n_rooms']}</small>",
                 "t": f"Kinds of room where simulated guests did better than with today's setup. Lower in {mv['behind']}."})
    res = {"hero": hero, "spa": sp, "rooms": table}
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1)[:3000])


if __name__ == "__main__":
    main()

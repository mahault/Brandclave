"""Multi-seed evaluation: the model against a rule-based controller and a fixed preset.

    python -m sentient.rich.evaluate --seeds 0-9 --workers 6          # all scenarios
    python -m sentient.rich.evaluate --summarize                        # aggregate only

Each (scenario, seed) is written to data/sentient_eval/<scenario>_s<seed>.json as soon as it
finishes, so an interrupted run resumes where it stopped. Every controller sees the same
world, the same seed, and the same simulated staff accepting or dismissing.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from sentient.baseline import FixedPresetController

from .rules import RuleController
from .run import Outcomes, run_agent
from .scenarios import SCENARIOS, simulated_override
from .spec import discretize  # noqa: F401  (keeps import order identical to run.py)

OUT = Path("data/sentient_eval")


def run_rules(name: str, steps: int, seed: int) -> dict:
    space = SCENARIOS[name](seed)
    ctrl = RuleController(space.cfg.intent)
    rng = np.random.default_rng(seed + 1000)
    o = Outcomes()
    for t in range(steps):
        reading = space.read()
        action = ctrl.step(reading, space.levers(), space.cooling, space.light)
        overridden = simulated_override(action, space, rng)
        o.record(space, t, action, overridden)
        space.apply("do_nothing" if overridden else action)
        space.step()
        o.after_step(space)
    return o.summary()


def run_preset(name: str, steps: int, seed: int) -> dict:
    space = SCENARIOS[name](seed)
    FixedPresetController().install(space)
    o = Outcomes()
    for _ in range(steps):
        space.step()
        o.after_step(space)
    return o.summary()


def run_one(name: str, seed: int, steps: int) -> dict:
    t0 = time.perf_counter()
    agent = run_agent(name, steps, seed, None)
    agent.get("affect", {}).pop("true_arousal", None)
    for k in ("true_valence", "p_stressed", "p_positive"):
        agent.get("affect", {}).pop(k, None)
    return {
        "scenario": name, "seed": seed, "steps": steps,
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "agent": agent,
        "rules": run_rules(name, steps, seed),
        "preset": run_preset(name, steps, seed),
    }


def _seeds(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def _mean_ci(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    a = np.asarray(xs, dtype=float)
    half = 1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    return {"mean": round(float(a.mean()), 3), "ci": round(float(half), 3), "n": len(a),
            "min": round(float(a.min()), 3), "max": round(float(a.max()), 3)}


def _pluck(runs: list[dict], *keys: str) -> list:
    out = []
    for r in runs:
        for k in keys:
            r = r[k]
        out.append(r)
    return out


def summarize(out: Path = OUT) -> dict:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("*_s*.json"))]
    by = {}
    for r in runs:
        by.setdefault(r["scenario"], []).append(r)
    summary = {}
    for name, rs in by.items():
        summary[name] = {
            "n_seeds": len(rs),
            "forecast_acc": _mean_ci(_pluck(rs, "agent", "prediction_accuracy")),
            "persistence_acc": _mean_ci(_pluck(rs, "agent", "persistence_accuracy")),
            "forecast_beats_persistence": sum(
                (r["agent"]["prediction_accuracy"] or 0) >= (r["agent"]["persistence_accuracy"] or 0) for r in rs),
            "ece": _mean_ci(_pluck(rs, "agent", "calibration", "ece")),
            "r_arousal": _mean_ci(_pluck(rs, "agent", "affect", "r_arousal")),
            "auc_stressed": _mean_ci(_pluck(rs, "agent", "affect", "auc_stressed")),
            "auc_positive": _mean_ci(_pluck(rs, "agent", "affect", "auc_positive")),
            "auc_negative": _mean_ci(_pluck(rs, "agent", "affect", "auc_negative")),
            "sensor_flagged": sum(bool(r["agent"]["sensor_flags"]) for r in rs),
            "hvac_flag_step": _mean_ci(_pluck(rs, "agent", "hvac_flag_step")),
            "step_median_s": _mean_ci(_pluck(rs, "agent", "step_seconds", "median")),
            "step_p95_s": _mean_ci(_pluck(rs, "agent", "step_seconds", "p95")),
            "dismissed_by_third": [
                _mean_ci([r["agent"]["by_time_third"][i]["dismissed"] for r in rs]) for i in range(3)],
        }
        for who, path in (("agent", ("agent", "outcomes")), ("rules", ("rules",)), ("preset", ("preset",))):
            summary[name][who] = {k: _mean_ci(_pluck(rs, *path, k)) for k in (
                "energy", "mood_mean", "arousal_mean", "comfort_frac", "stale_frac",
                "recommendations", "dismissed", "staff_alerts", "tickets", "false_tickets", "ticket_step")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--one", action="store_true", help="run a single (scenario, seed) and write it")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        print(json.dumps(summarize(out), indent=1)[:4000])
        return
    if args.one:
        seed = int(args.seeds)
        res = run_one(args.scenario, seed, args.steps)
        (out / f"{args.scenario}_s{seed}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"done {args.scenario} s{seed} {res['wall_seconds']}s", flush=True)
        return

    names = list(SCENARIOS) if args.scenario == "all" else args.scenario.split(",")
    jobs = [(n, s) for s in _seeds(args.seeds) for n in names
            if not (out / f"{n}_s{s}.json").exists()]
    print(f"{len(jobs)} jobs, {args.workers} workers", flush=True)
    running: list[subprocess.Popen] = []
    while jobs or running:
        while jobs and len(running) < args.workers:
            n, s = jobs.pop(0)
            running.append(subprocess.Popen([sys.executable, "-m", "sentient.rich.evaluate", "--one",
                                             "--scenario", n, "--seeds", str(s), "--steps", str(args.steps),
                                             "--out", str(out)]))
        time.sleep(2)
        running = [p for p in running if p.poll() is None]
    summarize(out)
    print("summary written", flush=True)


if __name__ == "__main__":
    main()

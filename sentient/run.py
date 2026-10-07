"""Run scenarios for the agent and the fixed-preset baseline and print the three numbers.

    python -m sentient.run --scenario all --steps 60 --seed 0
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from .agent import SentientAgent
from .baseline import FixedPresetController, PersistencePredictor
from .ledger import RecommendationLedger
from .model import discretize
from .scenarios import SCENARIOS, simulated_override

logging.getLogger("jax").setLevel(logging.ERROR)


def run_agent(name: str, steps: int, seed: int, out_dir: Path | None) -> dict:
    space = SCENARIOS[name](seed)
    agent = SentientAgent(intent=space.cfg.intent, seed=seed)
    rng = np.random.default_rng(seed + 1000)
    ledger = RecommendationLedger(
        path=(out_dir / f"{name}.jsonl") if out_dir else None, scenario=name, source="simulated",
    )
    pending: list[tuple[str, str, int, int]] = []  # (rid, modality, due_t, persistence_pred)
    energy = 0.0
    arousal_trace = []

    for t in range(steps):
        reading = space.read()
        obs = discretize(reading)
        # settle predictions that are due
        still = []
        for rid, modality, due, persist in pending:
            if t >= due:
                ledger.resolve(rid, obs[modality], persist)
            else:
                still.append((rid, modality, due, persist))
        pending = still

        rec = agent.step(reading, space.levers())
        overridden = simulated_override(rec.action, space, rng)
        applied = "do_nothing" if overridden else rec.action
        rid = ledger.record(t, rec, applied, overridden)
        pending.append((rid, rec.prediction.modality, t + rec.prediction.horizon,
                        PersistencePredictor.predict(obs, rec.prediction.modality)))
        if rec.is_action:
            agent.feedback(rec, accepted=not overridden)
        space.apply(applied)
        agent.commit(applied)
        space.step()
        energy += space.energy()
        arousal_trace.append(space.arousal)

    m = ledger.metrics()
    m.update({
        "energy_total": round(energy, 2),
        "arousal_start": round(arousal_trace[0], 3),
        "arousal_end": round(float(np.mean(arousal_trace[-5:])), 3),
        "final_margins": {k: round(v, 3) for k, v in agent.margins.items()},
        "arrivals": space.arrivals_total,
        "departures": space.departures_total,
        "occupancy_end": space.n,
        "mean_valence_end": round(space.mean_valence(), 2),
        "setback_streak_required": agent.setback_streak_required,
    })
    return m


def run_baseline(name: str, steps: int, seed: int) -> dict:
    space = SCENARIOS[name](seed)
    ctrl = FixedPresetController()
    ctrl.install(space)
    energy = 0.0
    arousal_trace = []
    for _ in range(steps):
        space.apply(ctrl.step(space))
        space.step()
        energy += space.energy()
        arousal_trace.append(space.arousal)
    return {
        "energy_total": round(energy, 2),
        "arousal_start": round(arousal_trace[0], 3),
        "arousal_end": round(float(np.mean(arousal_trace[-5:])), 3),
    }


def run_scenario(name: str, steps: int = 60, seed: int = 0, out_dir: Path | None = None) -> dict:
    return {"scenario": name, "agent": run_agent(name, steps, seed, out_dir), "baseline": run_baseline(name, steps, seed)}


def _fmt(x) -> str:
    return "-" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))


def print_report(results: list[dict]) -> None:
    for r in results:
        a, b = r["agent"], r["baseline"]
        print(f"\n== {r['scenario']} ==")
        print(f"  actions            {a['action_counts']}")
        print(f"  prediction acc     agent {_fmt(a['prediction_accuracy'])}  vs persistence {_fmt(a['persistence_accuracy'])}  (n={a['predictions_scored']})")
        print(f"  override rate      first third {_fmt(a['override_rate_first_third'])}  ->  last third {_fmt(a['override_rate_last_third'])}  (overall {_fmt(a['override_rate'])}, {a['recommendations']} recs)")
        saving = (1 - a["energy_total"] / b["energy_total"]) * 100
        print(f"  energy             agent {a['energy_total']}  vs preset {b['energy_total']}  ({saving:.0f}% less)")
        print(f"  arousal end        agent {a['arousal_end']}  vs preset {b['arousal_end']}  (start {a['arousal_start']})")
        print(f"  gated steps        {a['gated_steps']}")
        thirds = "  ->  ".join(f"{x['dismissed']}/{x['recommendations']}" for x in a["by_time_third"])
        print(f"  dismissed/recs     by time third: {thirds}   (margins {a['final_margins']}, setback streak {a['setback_streak_required']})")
        if a["sensor_flags"]:
            pm = {m: (a["per_modality"][m]["accuracy"], a["per_modality"][m]["persistence_accuracy"]) for m in a["sensor_flags"]}
            print(f"  SENSOR FLAG        {a['sensor_flags']}  model vs persistence accuracy {pm}")
        cal = a["calibration"]
        bins = "  ".join(f"p~{b['confidence']:.2f}->{b['hit_rate']:.2f}(n={b['n']})" for b in cal["bins"])
        print(f"  calibration        ECE {_fmt(cal['ece'])}   {bins}")
        print(f"  population         arrivals {a['arrivals']}  departures {a['departures']}  present at end {a['occupancy_end']}  (valence {a['mean_valence_end']})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sentient simulation runner")
    ap.add_argument("--scenario", default="all", help=f"one of {list(SCENARIOS)} or 'all'")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/sentient", help="directory for ledgers and results")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    results = [run_scenario(n, args.steps, args.seed, out_dir) for n in names]
    print_report(results)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nledgers and results written to {out_dir}/")


if __name__ == "__main__":
    main()

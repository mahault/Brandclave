"""Run the rich scenarios and print the numbers.

    python -m sentient.rich.run --scenario all --steps 60 --seed 0
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from sentient.baseline import FixedPresetController
from sentient.ledger import RecommendationLedger

from .agent import RichAgent
from .scenarios import SCENARIOS, simulated_override
from .spec import discretize

logging.getLogger("jax").setLevel(logging.ERROR)


class Outcomes:
    """What happened to the people and the building, scored the same way for every controller."""

    def __init__(self):
        self.energy = 0.0
        self.occupied = 0
        self.valence, self.arousal = [], []
        self.comfort = 0      # occupied steps with temperature in 20.5-24.5
        self.stale = 0        # occupied steps with CO2 over 1,000
        self.recs = self.dismissed = self.staff_alerts = self.tickets = 0
        self.ticket_step = None
        self.false_tickets = 0

    def record(self, space, t: int, action: str, overridden: bool) -> None:
        if action != "do_nothing":
            self.recs += 1
            self.dismissed += int(overridden)
        if not overridden:
            if action == "staff_alert":
                self.staff_alerts += 1
            if action == "maintenance_ticket":
                self.tickets += 1
                if space.hvac_degraded and self.ticket_step is None:
                    self.ticket_step = t
                if not space.hvac_degraded:
                    self.false_tickets += 1

    def after_step(self, space) -> None:
        self.energy += space.energy()
        if space.n > 0:
            self.occupied += 1
            self.valence.append(space.mean_valence())
            self.arousal.append(space.arousal)
            self.comfort += int(20.5 <= space.temp <= 24.5)
            self.stale += int(space.co2 > 1000)

    def summary(self) -> dict:
        occ = max(1, self.occupied)
        return {
            "energy": round(self.energy, 2),
            "mood_mean": round(float(np.mean(self.valence)), 3) if self.valence else None,
            "arousal_mean": round(float(np.mean(self.arousal)), 3) if self.arousal else None,
            "comfort_frac": round(self.comfort / occ, 3),
            "stale_frac": round(self.stale / occ, 3),
            "recommendations": self.recs,
            "dismissed": self.dismissed,
            "staff_alerts": self.staff_alerts,
            "tickets": self.tickets,
            "false_tickets": self.false_tickets,
            "ticket_step": self.ticket_step,
        }


def run_agent(name: str, steps: int, seed: int, out_dir: Path | None) -> dict:
    space = SCENARIOS[name](seed)
    agent = RichAgent(intent=space.cfg.intent)
    rng = np.random.default_rng(seed + 1000)
    ledger = RecommendationLedger(path=(out_dir / f"{name}.jsonl") if out_dir else None,
                                  scenario=name, source="simulated-rich")
    pending: list[tuple[str, int, dict]] = []   # (rid, due_t, persistence obs at issue)
    energy = 0.0
    arousal_trace, valence_trace, co2_trace, temp_trace = [], [], [], []
    ticket_step = None
    hvac_flag_step = None
    # affect inference: what the model believed vs what the world knew
    outcomes = Outcomes()
    step_seconds = []
    true_arousal, true_valence, p_stressed, p_positive, p_negative = [], [], [], [], []

    for t in range(steps):
        reading = space.read()
        obs = discretize(reading)
        still = []
        for rid, due, persist in pending:
            if t >= due:
                ledger.resolve_all(rid, obs, persist)
            else:
                still.append((rid, due, persist))
        pending = still

        t0 = time.perf_counter()
        rec = agent.step(reading, space.levers())
        step_seconds.append(time.perf_counter() - t0)
        if space.n > 0:
            true_arousal.append(space.arousal)
            true_valence.append(space.mean_valence())
            p_stressed.append(rec.beliefs["arousal"]["stressed"])
            p_positive.append(rec.beliefs["valence"]["positive"])
            p_negative.append(rec.beliefs["valence"]["negative"])
        if hvac_flag_step is None and agent.hvac_belief_trace[-1] > agent.hvac_threshold:
            hvac_flag_step = t
        overridden = simulated_override(rec.action, space, rng)
        applied = "do_nothing" if overridden else rec.action
        if applied == "maintenance_ticket" and ticket_step is None:
            ticket_step = t
        rid = ledger.record(t, rec, applied, overridden)
        outcomes.record(space, t, rec.action, overridden)
        pending.append((rid, t + rec.prediction.horizon, dict(obs)))
        if rec.is_action:
            agent.feedback(rec, accepted=not overridden)
        space.apply(applied)
        hvac_mode = "cool_air" if space.cooling else ("warm_air" if space.heating else None)
        agent.commit(applied, hvac_mode=hvac_mode)
        space.step()
        outcomes.after_step(space)
        energy += space.energy()
        arousal_trace.append(space.arousal)
        valence_trace.append(space.mean_valence())
        co2_trace.append(space.co2)
        temp_trace.append(space.temp)

    m = ledger.metrics()
    m.update({
        "energy_total": round(energy, 2),
        "arousal_start": round(arousal_trace[0], 3),
        "arousal_end": round(float(np.mean(arousal_trace[-5:])), 3),
        "valence_start": round(valence_trace[0], 3),
        "valence_end": round(float(np.mean(valence_trace[-5:])), 3),
        "co2_max": round(max(co2_trace), 0),
        "co2_end": round(float(np.mean(co2_trace[-5:])), 0),
        "temp_end": round(float(np.mean(temp_trace[-5:])), 2),
        "hvac_degraded_true": space.hvac_degraded,
        "hvac_degrade_at": space.cfg.hvac_degrade_at,
        "hvac_belief_end": round(agent.hvac_belief_trace[-1], 3),
        "hvac_belief_max": round(max(agent.hvac_belief_trace), 3),
        "hvac_flag_step": hvac_flag_step,
        "maintenance_ticket_step": ticket_step,
        "final_margins": {k: round(v, 3) for k, v in agent.margins.items()},
        "affect": affect_scores(true_arousal, true_valence, p_stressed, p_positive, p_negative),
        "arrivals": space.arrivals_total,
        "departures": space.departures_total,
        "occupancy_end": space.n,
        "outcomes": outcomes.summary(),
        "online_flags": sorted(agent.flagged),
        "step_seconds": {"median": round(float(np.median(step_seconds)), 3),
                         "p95": round(float(np.percentile(step_seconds, 95)), 3),
                         "first": round(step_seconds[0], 3)},
    })
    return m


def _auc(scores: list[float], labels: list[bool]) -> float | None:
    """Rank-based AUC: P(score of a positive > score of a negative)."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def _corr(a: list[float], b: list[float]) -> float | None:
    if len(a) < 5:
        return None
    a, b = np.asarray(a), np.asarray(b)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return None
    return round(float(np.corrcoef(a, b)[0, 1]), 3)


def affect_scores(true_arousal, true_valence, p_stressed, p_positive, p_negative) -> dict:
    """How well the inferred human state tracked the world's hidden truth."""
    return {
        "n": len(true_arousal),
        "r_arousal": _corr(p_stressed, true_arousal),
        "auc_stressed": _auc(p_stressed, [a > 0.65 for a in true_arousal]),
        "r_valence": _corr(p_positive, true_valence),
        "auc_positive": _auc(p_positive, [v > 0.6 for v in true_valence]),
        "auc_negative": _auc(p_negative, [v < 0.4 for v in true_valence]),
        "true_arousal": [round(x, 3) for x in true_arousal],
        "true_valence": [round(x, 3) for x in true_valence],
        "p_stressed": [round(x, 3) for x in p_stressed],
        "p_positive": [round(x, 3) for x in p_positive],
    }


def run_baseline(name: str, steps: int, seed: int) -> dict:
    space = SCENARIOS[name](seed)
    ctrl = FixedPresetController()
    ctrl.install(space)
    energy = 0.0
    arousal_trace, valence_trace, co2_trace = [], [], []
    for _ in range(steps):
        space.apply(ctrl.step(space))
        space.step()
        energy += space.energy()
        arousal_trace.append(space.arousal)
        valence_trace.append(space.mean_valence())
        co2_trace.append(space.co2)
    return {
        "energy_total": round(energy, 2),
        "arousal_end": round(float(np.mean(arousal_trace[-5:])), 3),
        "valence_end": round(float(np.mean(valence_trace[-5:])), 3),
        "co2_end": round(float(np.mean(co2_trace[-5:])), 0),
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
        thirds = "  ->  ".join(f"{x['dismissed']}/{x['recommendations']}" for x in a["by_time_third"])
        print(f"  dismissed/recs     by time third: {thirds}")
        saving = (1 - a["energy_total"] / b["energy_total"]) * 100
        print(f"  energy             agent {a['energy_total']}  vs preset {b['energy_total']}  ({saving:.0f}% less)")
        print(f"  arousal            {a['arousal_start']} -> {a['arousal_end']}  (preset {b['arousal_end']})")
        print(f"  valence            {a['valence_start']} -> {a['valence_end']}  (preset {b['valence_end']})")
        print(f"  co2                max {a['co2_max']:.0f}  end {a['co2_end']:.0f}  (preset end {b['co2_end']:.0f})   temp end {a['temp_end']}")
        print(f"  hvac               truly degraded {a['hvac_degraded_true']} (at {a['hvac_degrade_at']})  belief end {a['hvac_belief_end']}  max {a['hvac_belief_max']}  flagged step {a['hvac_flag_step']}  ticket step {a['maintenance_ticket_step']}")
        if a["sensor_flags"]:
            print(f"  SENSOR FLAG        {a['sensor_flags']}")
        af = a["affect"]
        print(f"  affect             r(arousal) {_fmt(af['r_arousal'])}  AUC stressed {_fmt(af['auc_stressed'])}  r(valence) {_fmt(af['r_valence'])}  AUC positive {_fmt(af['auc_positive'])}  AUC negative {_fmt(af['auc_negative'])}  (n={af['n']})")
        cal = a["calibration"]
        bins = "  ".join(f"p~{x['confidence']:.2f}->{x['hit_rate']:.2f}(n={x['n']})" for x in cal["bins"])
        print(f"  calibration        ECE {_fmt(cal['ece'])}   {bins}")
        print(f"  gated steps        {a['gated_steps']}   population arrivals {a['arrivals']} departures {a['departures']} end {a['occupancy_end']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sentient rich simulation runner")
    ap.add_argument("--scenario", default="all")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/sentient_rich")
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

"""Many spa evenings, three ways. Each seed is written as it finishes, so a run resumes.

    python -m sentient.rich.spa_eval --seeds 0-9 --out data/sentient_spa_eval --workers 2
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

KEYS = ("mood", "queue_minutes", "queue_max", "cold_guest_minutes", "noise_guest_minutes", "distress_minutes",
        "energy", "staff_calls", "dismissed", "tickets", "heater_ticket_minutes", "second_sauna_minutes")


def summarize(out: Path) -> dict:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("spa_s*.json"))]
    summary = {"n_seeds": len(runs)}
    for who in ("sentient", "rules", "team"):
        summary[who] = {}
        for k in KEYS:
            xs = [r["summary"][who][k] for r in runs if r["summary"][who][k] is not None]
            if not xs:
                summary[who][k] = None
                continue
            a = np.asarray(xs, dtype=float)
            half = 1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
            summary[who][k] = {"mean": round(float(a.mean()), 3), "ci": round(float(half), 3), "n": len(a)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--out", default="data/sentient_spa_eval")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--one", type=int, default=None)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        print(json.dumps(summarize(out), indent=1))
        return
    if args.one is not None:
        import warnings
        warnings.filterwarnings("ignore")
        from .spa import run
        res = run(args.one)
        (out / f"spa_s{args.one}.json").write_text(json.dumps(res), encoding="utf-8")
        print(f"done spa s{args.one}", flush=True)
        return
    a, b = args.seeds.split("-")
    jobs = [s for s in range(int(a), int(b) + 1) if not (out / f"spa_s{s}.json").exists()]
    running: list[subprocess.Popen] = []
    while jobs or running:
        while jobs and len(running) < args.workers:
            running.append(subprocess.Popen([sys.executable, "-m", "sentient.rich.spa_eval", "--one", str(jobs.pop(0)),
                                             "--out", str(out)]))
        time.sleep(2)
        running = [p for p in running if p.poll() is None]
    print(json.dumps(summarize(out), indent=1))


if __name__ == "__main__":
    main()

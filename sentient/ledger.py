"""Append-only recommendation ledger.

Same rule as the platform's Signal Ledger: a recommendation and its prediction are sealed
with a SHA-256 hash when written and never edited. Outcomes, overrides and notes are
appended as events. `metrics()` derives the three numbers from the log.
"""

import hashlib
import json
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

from .agent import Recommendation


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class RecommendationLedger:
    def __init__(self, path: Path | None = None, scenario: str = "unnamed", source: str = "simulated"):
        self.path = Path(path) if path else None
        self.scenario = scenario
        self.source = source
        self.events: list[dict] = []
        self._records: dict[str, dict] = {}
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def _append(self, event: dict) -> None:
        self.events.append(event)
        if self.path:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(_canonical(event) + "\n")

    def record(self, t: int, rec: Recommendation, applied: str, overridden: bool) -> str:
        content = {
            "t": t,
            "scenario": self.scenario,
            "source": self.source,
            "action": rec.action,
            "applied": applied,
            "overridden": overridden,
            "reason": rec.reason,
            "confidence": round(rec.confidence, 4),
            "beliefs": rec.beliefs,
            "observation": rec.observation,
            "prediction": asdict(rec.prediction),
            "gated": rec.gated,
            "masked": rec.masked,
            "forecasts": {m: [int(p), round(float(q), 4)] for m, (p, q) in rec.forecasts.items()},
        }
        rid = f"{self.scenario}-{t:05d}"
        content_hash = hashlib.sha256(_canonical(content).encode("utf-8")).hexdigest()
        event = {"type": "recommendation", "id": rid, "hash": content_hash, **content}
        self._records[rid] = event
        self._append(event)
        return rid

    def resolve(self, rid: str, actual: int, persistence_predicted: int) -> None:
        rec = self._records[rid]
        pred = rec["prediction"]
        self._append({
            "type": "outcome",
            "id": rid,
            "t_due": rec["t"] + pred["horizon"],
            "modality": pred["modality"],
            "predicted": pred["predicted"],
            "probability": pred.get("probability", 0.0),
            "actual": actual,
            "hit": pred["predicted"] == actual,
            "persistence_predicted": persistence_predicted,
            "persistence_hit": persistence_predicted == actual,
        })

    def resolve_all(self, rid: str, actual: dict[str, int], persistence: dict[str, int]) -> None:
        """Score the primary prediction and every per-channel forecast. A recommendation that
        was dismissed is recorded as unscored: its forecast assumed an action that never
        happened, so it says nothing about the model or the sensors."""
        rec = self._records[rid]
        pred = rec["prediction"]
        scored = not rec["overridden"]
        self._append({
            "type": "outcome",
            "id": rid,
            "t_due": rec["t"] + pred["horizon"],
            "modality": pred["modality"],
            "predicted": pred["predicted"],
            "probability": pred.get("probability", 0.0),
            "actual": actual[pred["modality"]],
            "hit": pred["predicted"] == actual[pred["modality"]],
            "persistence_predicted": persistence[pred["modality"]],
            "persistence_hit": persistence[pred["modality"]] == actual[pred["modality"]],
            "scored": scored,
        })
        for m, (p, q) in rec.get("forecasts", {}).items():
            self._append({
                "type": "channel_outcome",
                "id": rid,
                "modality": m,
                "predicted": p,
                "probability": q,
                "actual": actual[m],
                "hit": p == actual[m],
                "persistence_hit": persistence[m] == actual[m],
                "scored": scored,
            })

    def verify(self, rid: str) -> bool:
        rec = dict(self._records[rid])
        stored = rec.pop("hash")
        rec.pop("type")
        rec.pop("id")
        return hashlib.sha256(_canonical(rec).encode("utf-8")).hexdigest() == stored

    # ----------------------------------------------------------------- metrics
    @staticmethod
    def calibration(outs: list[dict], bins: int = 5) -> dict:
        """Reliability of the sealed predictions: when the model said p, how often was it right?
        Returns per-bin (mean confidence, hit rate, n) and the expected calibration error."""
        if not outs:
            return {"bins": [], "ece": None}
        probs = [o.get("probability", 0.0) for o in outs]
        hits = [1.0 if o["hit"] else 0.0 for o in outs]
        edges = [i / bins for i in range(bins + 1)]
        rows, ece = [], 0.0
        for lo, hi in pairwise(edges):
            idx = [i for i, pr in enumerate(probs) if (lo <= pr < hi) or (hi == 1.0 and pr == 1.0)]
            if not idx:
                continue
            conf = sum(probs[i] for i in idx) / len(idx)
            acc = sum(hits[i] for i in idx) / len(idx)
            rows.append({"confidence": round(conf, 3), "hit_rate": round(acc, 3), "n": len(idx)})
            ece += abs(conf - acc) * len(idx) / len(outs)
        return {"bins": rows, "ece": round(ece, 3)}

    def metrics(self) -> dict:
        recs = [e for e in self.events if e["type"] == "recommendation"]
        outs = [e for e in self.events if e["type"] == "outcome" and e.get("scored", True)]
        channel_outs = [e for e in self.events if e["type"] == "channel_outcome" and e.get("scored", True)]
        actions = [r for r in recs if r["action"] != "do_nothing"]

        def rate(rs: list[dict]) -> float | None:
            return None if not rs else sum(1 for r in rs if r["overridden"]) / len(rs)

        third = max(1, len(actions) // 3)
        # learning shows up over time: dismissals per third of the run, by step, not by rec index
        t_max = max((r["t"] for r in recs), default=0) + 1
        span = max(1, t_max // 3)
        by_time = []
        for k in range(3):
            lo, hi = k * span, (k + 1) * span if k < 2 else t_max
            rs = [r for r in actions if lo <= r["t"] < hi]
            by_time.append({"recommendations": len(rs), "dismissed": sum(1 for r in rs if r["overridden"])})
        # per-modality accuracy, and a flag when one modality's predictions collapse while
        # persistence stays right: the signature of a sensor that is lying
        per_mod: dict[str, dict] = {}
        for o in outs + channel_outs:
            d = per_mod.setdefault(o["modality"], {"n": 0, "hit": 0, "persist_hit": 0})
            d["n"] += 1
            d["hit"] += int(o["hit"])
            d["persist_hit"] += int(o["persistence_hit"])
        sensor_flags = []
        for m, d in per_mod.items():
            d["accuracy"] = round(d["hit"] / d["n"], 3)
            d["persistence_accuracy"] = round(d["persist_hit"] / d["n"], 3)
        for m, d in per_mod.items():
            others = [x["accuracy"] for k, x in per_mod.items() if k != m and x["n"] >= 10]
            ref = sum(others) / len(others) if others else 1.0
            if (d["n"] >= 10 and d["accuracy"] < 0.30 and d["persistence_accuracy"] > 0.8
                    and d["accuracy"] < 0.5 * ref):
                sensor_flags.append(m)
        counts: dict[str, int] = {}
        for r in recs:
            counts[r["action"]] = counts.get(r["action"], 0) + 1
        return {
            "steps": len(recs),
            "recommendations": len(actions),
            "action_counts": counts,
            "prediction_accuracy": None if not outs else sum(1 for o in outs if o["hit"]) / len(outs),
            "persistence_accuracy": None if not outs else sum(1 for o in outs if o["persistence_hit"]) / len(outs),
            "predictions_scored": len(outs),
            "override_rate": rate(actions),
            "override_rate_first_third": rate(actions[:third]),
            "override_rate_last_third": rate(actions[-third:]) if actions else None,
            "gated_steps": sum(1 for r in recs if r["gated"]),
            "mean_confidence": None if not outs else round(sum(o.get("probability", 0.0) for o in outs) / len(outs), 3),
            "calibration": self.calibration(outs),
            "by_time_third": by_time,
            "per_modality": per_mod,
            "sensor_flags": sensor_flags,
        }

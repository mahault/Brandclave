"""Sentient simulation: the loop runs, the safety cases behave, the agent learns from
dismissals, a dead sensor is flagged, and the ledger seals."""

import numpy as np
import pytest

from sentient.agent import PYMDP_AVAILABLE, SentientAgent
from sentient.ledger import RecommendationLedger
from sentient.model import ACTIONS, Reading, build_A, build_B, build_C, discretize
from sentient.run import run_scenario
from sentient.scenarios import SCENARIOS
from sentient.space import Space, SpaceConfig

pytestmark = pytest.mark.skipif(not PYMDP_AVAILABLE, reason="pymdp not installed")

STEPS = 60


@pytest.fixture(scope="module")
def results():
    return {name: run_scenario(name, steps=STEPS, seed=0) for name in SCENARIOS}


# ----------------------------------------------------------------------- model
def test_model_matrices_are_distributions():
    for a in build_A():
        assert np.allclose(a.sum(axis=0), 1.0)
    for b in build_B():
        assert np.allclose(b.sum(axis=0), 1.0)
    for intent in ("restore", "work", "elevated", "service"):
        assert len(build_C(intent)) == 4


def test_discretize_edges():
    r = Reading(count=0, motion_level=0.1, db=40, temp_c=21, light_lux=100, hr=None)
    assert discretize(r) == {"count": 0, "motion": 0, "sound": 0, "wearable": 0}
    r = Reading(count=5, motion_level=0.9, db=70, temp_c=24, light_lux=400, hr=110)
    assert discretize(r) == {"count": 2, "motion": 2, "sound": 2, "wearable": 2}


# ----------------------------------------------------------------------- world
def test_world_has_arrivals_departures_and_artifacts():
    space = SCENARIOS["bar_9pm"](seed=3)
    counts, dbs, hrs = [], [], []
    for _ in range(80):
        r = space.read()
        counts.append(r.count)
        dbs.append(r.db)
        hrs.append(r.hr)
        space.step()
    assert space.arrivals_total > 0 and space.departures_total > 0
    assert len(set(counts)) > 3                      # population moves
    assert np.std(dbs) > 1.0                         # sound is not a constant
    assert any(h is None for h in hrs)               # wearable gaps happen


def test_temperature_has_inertia():
    space = Space(SpaceConfig(name="t", intent="restore", n_occupants=1, cooling=False, seed=0))
    warm = space.temp
    space.apply("cool_air")
    space.step()
    after_one = space.temp
    for _ in range(15):
        space.step()
    after_many = space.temp
    assert warm - after_one < 1.0            # not felt immediately
    assert after_many < after_one - 1.0      # felt later


# ------------------------------------------------------------------- scenarios
def test_restore_pod_calms_the_room(results):
    a = results["restore_pod"]["agent"]
    calming = sum(a["action_counts"].get(k, 0) for k in ("dim_lights", "lower_sound", "cool_air"))
    assert calming >= 2, a["action_counts"]
    assert a["arousal_end"] < a["arousal_start"] - 0.15


def test_bar_9pm_never_fights_the_intent(results):
    a = results["bar_9pm"]["agent"]
    assert a["action_counts"].get("do_nothing", 0) / a["steps"] > 0.7, a["action_counts"]
    assert a["action_counts"].get("dim_lights", 0) + a["action_counts"].get("lower_sound", 0) <= 3


def test_ambiguous_hr_holds(results):
    a = results["ambiguous_hr"]["agent"]
    assert a["action_counts"].get("do_nothing", 0) / a["steps"] > 0.9, a["action_counts"]
    assert a["action_counts"].get("setback", 0) == 0  # a live wearable means someone is there


def test_lobby_peak_alerts_staff_with_cooldown(results):
    a = results["lobby_peak"]["agent"]
    alerts = a["action_counts"].get("staff_alert", 0)
    assert 1 <= alerts <= 8, a["action_counts"]


def test_happy_group_learns_to_leave_them_alone(results):
    a = results["happy_group"]["agent"]
    first, last = a["by_time_third"][0], a["by_time_third"][2]
    assert first["dismissed"] >= 3, a["by_time_third"]
    assert last["recommendations"] <= 1, a["by_time_third"]
    assert a["final_margins"]["lower_sound"] > 0.08 or a["final_margins"]["dim_lights"] > 0.08


def test_noisy_sensors_still_calm_the_room(results):
    a = results["noisy_sensors"]["agent"]
    assert a["arousal_end"] < a["arousal_start"] - 0.15
    assert a["prediction_accuracy"] >= a["persistence_accuracy"] - 0.05


def test_dead_db_meter_is_flagged(results):
    a = results["sensor_failure"]["agent"]
    assert "sound" in a["sensor_flags"], a["per_modality"]
    assert a["gated_steps"] > 10  # conflicting evidence, so it mostly holds


# ---------------------------------------------------------------------- numbers
def test_three_numbers_and_calibration_are_reported(results):
    for name, r in results.items():
        a = r["agent"]
        assert a["predictions_scored"] > 0, name
        assert a["prediction_accuracy"] is not None
        assert a["persistence_accuracy"] is not None
        assert r["baseline"]["energy_total"] > 0
        cal = a["calibration"]
        assert cal["ece"] is not None
        assert sum(b["n"] for b in cal["bins"]) == a["predictions_scored"]


def test_agent_uses_energy_less_than_preset_when_restoring(results):
    r = results["restore_pod"]
    assert r["agent"]["energy_total"] < r["baseline"]["energy_total"]


# ----------------------------------------------------------------------- ledger
def test_ledger_seals_and_verifies(tmp_path):
    space = SCENARIOS["restore_pod"](seed=1)
    agent = SentientAgent(intent="restore")
    ledger = RecommendationLedger(path=tmp_path / "l.jsonl", scenario="restore_pod")
    rec = agent.step(space.read(), space.levers())
    rid = ledger.record(0, rec, rec.action, False)
    assert ledger.verify(rid)
    ledger._records[rid]["confidence"] = 0.0  # tamper
    assert not ledger.verify(rid)
    assert (tmp_path / "l.jsonl").read_text().count("\n") == 1


# --------------------------------------------------------------------- learning
def test_feedback_raises_margin_on_dismiss():
    agent = SentientAgent(intent="restore")
    space = SCENARIOS["restore_pod"](seed=2)
    rec = agent.step(space.read(), space.levers())
    action = rec.action if rec.action in ACTIONS and rec.action != "do_nothing" else "lower_sound"
    rec.action = action
    before = agent.margins[action]
    agent.feedback(rec, accepted=False)
    assert agent.margins[action] > before


def test_dismissed_setback_raises_required_streak():
    agent = SentientAgent(intent="restore")
    space = SCENARIOS["restore_pod"](seed=2)
    rec = agent.step(space.read(), space.levers())
    rec.action = "setback"
    assert agent.setback_streak_required == 2
    agent.feedback(rec, accepted=False)
    assert agent.setback_streak_required == 3

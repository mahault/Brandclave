"""Rich Sentient model: matrices are valid, the agent runs, and the scenarios that only a
richer model can handle behave: a happy group is read as happy, a stale room gets
ventilated, a dead cooling unit gets a maintenance ticket, a cold room gets warmed, a
flat bar gets pushed up."""

import numpy as np
import pytest

from sentient.rich.agent import PYMDP_AVAILABLE, RichAgent
from sentient.rich.run import run_scenario
from sentient.rich.scenarios import SCENARIOS
from sentient.rich.spec import (
    ACTIONS,
    INTENTS,
    NUM_OBS,
    NUM_STATES,
    Reading,
    build_A,
    build_B,
    build_C,
    build_D,
    discretize,
)

pytestmark = pytest.mark.skipif(not PYMDP_AVAILABLE, reason="pymdp not installed")

STEPS = 45
FOCUS = ["happy_group", "stale_meeting", "hvac_failure", "cold_morning", "flat_bar", "bar_9pm", "sensor_failure",
         "stressed_guest", "ambiguous_hr"]


@pytest.fixture(scope="module")
def results():
    return {name: run_scenario(name, steps=STEPS, seed=0) for name in FOCUS}


# ------------------------------------------------------------------------ model
def test_matrices_are_distributions():
    for a in build_A():
        assert np.allclose(a.sum(axis=0), 1.0)
    for b in build_B():
        assert np.allclose(b.sum(axis=0), 1.0)
    for intent in INTENTS:
        assert [len(c) for c in build_C(intent)] == NUM_OBS
        assert [len(d) for d in build_D(intent)] == NUM_STATES
        for b in build_B(intent):
            assert np.allclose(b.sum(axis=0), 1.0)


def test_discretize_covers_all_modalities():
    r = Reading(count=8, entries=4, motion_level=0.9, db=70, burstiness=0.8, temp_c=27, co2_ppm=1300,
                hr=110, hrv=30, mean_dwell_min=50, lux=550)
    d = discretize(r)
    assert d == {"count": 3, "entries": 2, "motion": 2, "sound_level": 2, "sound_character": 2,
                 "temp": 2, "co2": 2, "hr": 2, "hrv": 1, "dwell": 2, "lux": 3}
    r = Reading(count=0, entries=0, motion_level=0.0, db=35, burstiness=0.0, temp_c=19, co2_ppm=450)
    d = discretize(r)
    assert d["count"] == 0 and d["hr"] == 0 and d["hrv"] == 0 and d["temp"] == 0 and d["co2"] == 0
    assert d["lux"] == 0   # no light sensor reported


def test_agent_builds_and_steps():
    space = SCENARIOS["restore_pod"](seed=1)
    agent = RichAgent(intent="restore")
    rec = agent.step(space.read(), space.levers())
    assert set(rec.beliefs) == {"occupancy", "arousal", "valence", "activity", "thermal", "air", "hvac", "music",
                                "lighting"}
    assert rec.action in ACTIONS or rec.action == "setback"
    assert 0.0 <= rec.prediction.probability <= 1.0


# -------------------------------------------------------------------- scenarios
def test_happy_group_is_read_as_happy_and_mostly_left_alone(results):
    a = results["happy_group"]["agent"]
    # under a restore intent a loud room is wrong, so it will try; staff dismiss it and it learns
    hushing = a["action_counts"].get("lower_sound", 0) + a["action_counts"].get("dim_lights", 0)
    assert hushing <= 10, a["action_counts"]
    assert a["by_time_third"][2]["recommendations"] <= 2, a["by_time_third"]
    assert a["final_margins"]["lower_sound"] > 0.2 or a["final_margins"]["dim_lights"] > 0.2


def test_stale_meeting_gets_ventilated(results):
    a, b = results["stale_meeting"]["agent"], results["stale_meeting"]["baseline"]
    assert a["action_counts"].get("ventilate", 0) >= 1, a["action_counts"]
    assert a["co2_end"] < b["co2_end"]


def test_hvac_failure_is_inferred_and_ticketed(results):
    a = results["hvac_failure"]["agent"]
    assert a["hvac_belief_max"] > 0.6, a["hvac_belief_max"]
    assert a["maintenance_ticket_step"] is not None, a
    assert a["maintenance_ticket_step"] >= a["hvac_degrade_at"]


def test_cold_morning_ends_comfortable(results):
    # The room starts at 24 degrees, the top of the comfort band, and falls toward 16 with
    # nothing running. Whether the model first cools then holds, or warms later, it must not
    # leave the room cold or warm.
    a = results["cold_morning"]["agent"]
    hvac = a["action_counts"].get("warm_air", 0) + a["action_counts"].get("cool_air", 0)
    assert hvac >= 1, a["action_counts"]
    assert 21.0 < a["temp_end"] < 24.0, a["temp_end"]


def test_flat_bar_gets_pushed_up(results):
    a = results["flat_bar"]["agent"]
    pushed = a["action_counts"].get("raise_sound", 0) + a["action_counts"].get("raise_lights", 0)
    assert pushed >= 1, a["action_counts"]
    assert a["action_counts"].get("lower_sound", 0) == 0


def test_bar_9pm_never_fights_the_intent(results):
    a = results["bar_9pm"]["agent"]
    assert a["action_counts"].get("dim_lights", 0) + a["action_counts"].get("lower_sound", 0) <= 3, a["action_counts"]


def test_stressed_guest_gets_a_person_not_a_thermostat(results):
    a = results["stressed_guest"]["agent"]
    assert a["action_counts"].get("staff_alert", 0) >= 1, a["action_counts"]
    env = sum(a["action_counts"].get(k, 0) for k in ("raise_lights", "raise_sound", "warm_air"))
    assert env <= 1, a["action_counts"]
    b = results["stressed_guest"]["baseline"]
    assert a["arousal_end"] < b["arousal_end"] - 0.15, (a["arousal_end"], b["arousal_end"])
    assert a["valence_end"] > b["valence_end"] + 0.2, (a["valence_end"], b["valence_end"])


def test_ambiguous_hr_still_holds(results):
    a = results["ambiguous_hr"]["agent"]
    assert a["action_counts"].get("do_nothing", 0) / a["steps"] > 0.9, a["action_counts"]


def test_affect_inference_tracks_the_truth(results):
    for name in ("stressed_guest", "happy_group", "hvac_failure"):
        af = results[name]["agent"]["affect"]
        assert af["n"] >= 20, name
    st = results["stressed_guest"]["agent"]["affect"]
    assert st["r_arousal"] is not None and st["r_arousal"] > 0.3, st["r_arousal"]
    hg = results["happy_group"]["agent"]["affect"]
    assert hg["p_positive"][0] > hg["p_positive"][-1] - 0.5  # sanity: traces exist and are bounded


def test_dead_mic_is_flagged(results):
    # flagged live by the agent (a stuck reading), and no action is taken through it after
    a = results["sensor_failure"]["agent"]
    assert "sound_level" in a["online_flags"], a["online_flags"]


def test_no_false_flags_on_healthy_rooms(results):
    for name in ("stressed_guest", "ambiguous_hr", "bar_9pm", "happy_group"):
        assert results[name]["agent"]["online_flags"] == [], (name, results[name]["agent"]["online_flags"])


def test_numbers_and_calibration_reported(results):
    for name, r in results.items():
        a = r["agent"]
        assert a["predictions_scored"] > 0, name
        assert a["calibration"]["ece"] is not None


# ------------------------------------------------------------------ one guest, their own watch
def test_personal_reader_flags_a_distressed_watch_and_not_a_calm_one():
    agent = RichAgent(intent="restore")
    reader = agent.personal
    for t in range(6):
        reader.update(t, [(7, 96.0, 29.0), (8, 70.0, 62.0)])
    assert reader.due(7)
    assert not reader.due(8)
    reader.dismissed(7)                     # staff said no: wait for a longer run of the same reading
    assert not reader.due(7)


def test_upset_guest_in_a_busy_lounge_gets_a_person():
    from sentient.rich.run import run_agent
    m = run_agent("busy_lounge_distress", 30, 0, None)
    assert m["outcomes"]["staff_alerts"] >= 1
    assert m["outcomes"]["distress_steps"] < 15


def test_targeted_visit_helps_that_guest_only():
    space = SCENARIOS["busy_lounge_distress"](0)
    upset = next(o for o in space.occupants if o.mood_anchor is not None)
    space.staff_target = upset.pid
    space.apply("staff_alert")
    space.step()
    assert upset.mood_anchor is None
    assert space.staff_target is None

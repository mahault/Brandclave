"""Scenarios for the rich model. The first-prototype set, plus the ones only a richer
model can handle: a happy group that should now be read as happy, a meeting room going
stale, a cooling unit that dies mid-run, a cold morning, a bar that has gone flat.
"""

import numpy as np

from .space import Space, SpaceConfig


def restore_pod(seed: int = 0) -> Space:
    return Space(SpaceConfig(
        name="restore_pod", intent="restore", purpose="restore", n_occupants=1, wearable_rate=1.0,
        dwell_steps=1e6, light=0.9, sound_setting=0.7, cooling=False, arousal0=0.72,
        start_hour=15.0, seed=seed,
    ))


def bar_9pm(seed: int = 0) -> Space:
    def arrivals(hour):
        return 0.25 if hour < 22.5 else 0.05

    return Space(SpaceConfig(
        name="bar_9pm", intent="elevated", purpose="social", n_occupants=6, capacity=14,
        arrival_rate=arrivals, dwell_steps=35, wearable_rate=0.1,
        light=0.6, sound_setting=0.8, cooling=True, arousal0=0.7, social_drive=0.2,
        ventilation_rate=0.06, start_hour=21.0, seed=seed,
    ))


def flat_bar(seed: int = 0) -> Space:
    """A bar at nine with the music off and the lights up. Intent elevated. The room is
    quiet and flat and the people are drifting toward neutral. The model should push it up."""
    def arrivals(hour):
        return 0.08

    return Space(SpaceConfig(
        name="flat_bar", intent="elevated", purpose="social", n_occupants=3, capacity=10,
        arrival_rate=arrivals, dwell_steps=40, light=0.9, sound_setting=0.15, cooling=True,
        arousal0=0.35, valence0=0.5, social_drive=0.0, start_hour=21.0, seed=seed,
    ))


def ambiguous_hr(seed: int = 0) -> Space:
    return Space(SpaceConfig(
        name="ambiguous_hr", intent="restore", purpose="restore", n_occupants=1, wearable_rate=1.0,
        dwell_steps=1e6, light=0.2, sound_setting=0.1, cooling=True, arousal0=0.75,
        motion_scale=0.15, start_hour=19.0, seed=seed,
    ))


def lobby_peak(seed: int = 0) -> Space:
    def arrivals(hour):
        return 0.15 + 0.45 * max(0.0, 1.0 - abs(hour - 16.3) / 0.8)

    return Space(SpaceConfig(
        name="lobby_peak", intent="service", purpose="transit", n_occupants=2, capacity=14,
        arrival_rate=arrivals, dwell_steps=8, light=0.7, sound_setting=0.4, cooling=True,
        arousal0=0.45, start_hour=15.4, seed=seed,
    ))


def happy_group(seed: int = 0) -> Space:
    """Five delighted friends in a lounge set to restore. With valence and sound character
    in the model, the laughter should read as happy, and the model should mostly hold."""
    return Space(SpaceConfig(
        name="happy_group", intent="restore", purpose="social", n_occupants=5, capacity=8,
        dwell_steps=1e6, wearable_rate=0.2, light=0.6, sound_setting=0.5, cooling=True,
        arousal0=0.75, valence0=0.88, social_drive=0.25, start_hour=17.0, seed=seed,
    ))


def stale_meeting(seed: int = 0) -> Space:
    """Six people working in a closed room. CO2 climbs past a thousand. Ventilate."""
    return Space(SpaceConfig(
        name="stale_meeting", intent="work", purpose="work", n_occupants=6, capacity=8,
        dwell_steps=1e6, light=0.8, sound_setting=0.2, cooling=True, arousal0=0.45,
        co2_0=650.0, ventilation_rate=0.012, start_hour=10.0, seed=seed,
    ))


def hvac_failure(seed: int = 0) -> Space:
    """A warm restore room. The cooling unit dies at step 8. The model should command
    cooling, watch the temperature not respond, come to believe the unit is degraded, and
    raise a maintenance ticket instead of commanding again."""
    return Space(SpaceConfig(
        name="hvac_failure", intent="restore", purpose="restore", n_occupants=2, wearable_rate=0.5,
        dwell_steps=1e6, light=0.5, sound_setting=0.3, cooling=False, arousal0=0.55,
        outdoor_temp=31.0, hvac_degrade_at=8, start_hour=14.0, seed=seed,
    ))


def cold_morning(seed: int = 0) -> Space:
    """A work room at eight in the morning, cold. Warm it."""
    return Space(SpaceConfig(
        name="cold_morning", intent="work", purpose="work", n_occupants=3, capacity=8,
        dwell_steps=1e6, light=0.7, sound_setting=0.2, cooling=False, arousal0=0.45,
        outdoor_temp=4.0, start_hour=8.0, seed=seed,
    ))


def stressed_guest(seed: int = 0) -> Space:
    """One guest, purpose restore, wearable on, in a room that is already dim, quiet, cool
    and fresh. They are distressed: fidgeting, on the phone, heart rate up, HRV down,
    unhappy. No environmental lever would help. The right action is a person."""
    space = Space(SpaceConfig(
        name="stressed_guest", intent="restore", purpose="restore", n_occupants=1, wearable_rate=1.0,
        dwell_steps=1e6, light=0.2, sound_setting=0.1, cooling=True, arousal0=0.85, valence0=0.15,
        motion_scale=0.7, start_hour=22.0, seed=seed,
    ))
    space.occupants[0].baseline = 0.8      # the distress is theirs, not the room's
    space.occupants[0].mood_anchor = 0.15
    return space


def sensor_failure(seed: int = 0) -> Space:
    return Space(SpaceConfig(
        name="sensor_failure", intent="work", purpose="social", n_occupants=5, capacity=8,
        dwell_steps=200, wearable_rate=0.4, light=0.8, sound_setting=0.5, cooling=True,
        arousal0=0.65, stuck_db=41.0, start_hour=10.0, seed=seed,
    ))


SCENARIOS = {
    "restore_pod": restore_pod,
    "bar_9pm": bar_9pm,
    "flat_bar": flat_bar,
    "ambiguous_hr": ambiguous_hr,
    "lobby_peak": lobby_peak,
    "happy_group": happy_group,
    "stale_meeting": stale_meeting,
    "hvac_failure": hvac_failure,
    "cold_morning": cold_morning,
    "stressed_guest": stressed_guest,
    "sensor_failure": sensor_failure,
}


def simulated_override(action: str, space: Space, rng: np.random.Generator) -> bool:
    """Staff or guest pressing accept or dismiss. Knows the room."""
    if action == "do_nothing":
        return False
    calming = action in ("dim_lights", "lower_sound", "cool_air")
    arousing = action in ("raise_lights", "raise_sound")
    intent = space.cfg.intent
    if action == "setback":
        p = 0.9 if space.n > 0 else 0.05
    elif intent == "elevated" and calming and action != "cool_air":
        p = 0.9
    elif intent in ("restore", "work") and arousing or calming and space.n >= 3 and space.mean_valence() > 0.7 and action != "cool_air":
        p = 0.85
    elif action == "staff_alert" and space.n < 4:
        distressed = any(o.arousal > 0.7 and o.valence < 0.4 for o in space.occupants)
        p = 0.05 if distressed else 0.8
    elif action == "maintenance_ticket" and not space.hvac_degraded:
        p = 0.9
    elif action == "ventilate" and space.co2 < 750:
        p = 0.7
    elif action == "warm_air" and space.temp > 22.5 or action == "cool_air" and space.temp < 22.0:
        p = 0.8
    elif calming and space.arousal < 0.3:
        p = 0.6
    else:
        p = 0.05
    return bool(rng.random() < p)

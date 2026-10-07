"""Seven scenarios. Four from the call, three where the agent has to cope with being wrong.

bar_9pm and ambiguous_hr are Carlos's objections. happy_group, noisy_sensors and
sensor_failure exist because the demo that convinces engineers is the one where the
confidence drops and the agent holds, or where a person corrects it and it learns.
"""

import numpy as np

from .space import Space, SpaceConfig


def restore_pod(seed: int = 0) -> Space:
    """One guest, purpose restore, wearable on. Room is bright, noisy, warm. Nobody else comes."""
    return Space(SpaceConfig(
        name="restore_pod", intent="restore", purpose="restore", n_occupants=1, wearable_rate=1.0,
        dwell_steps=1e6,light=0.9, sound_setting=0.7, cooling=False, arousal0=0.72,
        start_hour=15.0, seed=seed,
    ))


def bar_9pm(seed: int = 0) -> Space:
    """Operator intent elevated. Groups arrive through the evening; loud and moving are the target."""
    def arrivals(hour: float) -> float:
        return 0.25 if hour < 22.5 else 0.05

    return Space(SpaceConfig(
        name="bar_9pm", intent="elevated", purpose="social", n_occupants=6, capacity=14,
        arrival_rate=arrivals, dwell_steps=35, wearable_rate=0.1,
        light=0.6, sound_setting=0.8, cooling=True, arousal0=0.7, social_drive=0.2,
        start_hour=21.0, daylight=False, seed=seed,
    ))


def ambiguous_hr(seed: int = 0) -> Space:
    """One resting guest, elevated heart rate after a workout, room already dim, quiet, cool.
    Nothing else says stressed and no lever is available. The right recommendation is none."""
    return Space(SpaceConfig(
        name="ambiguous_hr", intent="restore", purpose="restore", n_occupants=1, wearable_rate=1.0,
        dwell_steps=1e6,light=0.2, sound_setting=0.1, cooling=True, arousal0=0.75,
        motion_scale=0.15, start_hour=19.0, seed=seed,
    ))


def lobby_peak(seed: int = 0) -> Space:
    """Check-in peak: arrivals climb toward four o'clock, sound rises. Staff should be alerted."""
    def arrivals(hour: float) -> float:
        return 0.15 + 0.45 * max(0.0, 1.0 - abs(hour - 16.3) / 0.8)

    return Space(SpaceConfig(
        name="lobby_peak", intent="service", purpose="transit", n_occupants=2, capacity=14,
        arrival_rate=arrivals, dwell_steps=8, light=0.7, sound_setting=0.4, cooling=True,
        arousal0=0.45, start_hour=15.4, seed=seed,
    ))


def happy_group(seed: int = 0) -> Space:
    """A lounge set to restore. A loud, delighted group of friends arrives. The sensors say
    stressed; the people are having the time of their lives. Staff will dismiss any attempt to
    hush them, and the agent has to learn that from the dismissals."""
    def arrivals(hour: float) -> float:
        return 0.0

    space = Space(SpaceConfig(
        name="happy_group", intent="restore", purpose="social", n_occupants=5, capacity=8,
        arrival_rate=arrivals, dwell_steps=400, wearable_rate=0.2,
        light=0.6, sound_setting=0.5, cooling=True, arousal0=0.75, social_drive=0.25,
        start_hour=17.0, seed=seed,
    ))
    for o in space.occupants:
        o.valence = float(np.clip(0.85 + space.rng.normal(0, 0.05), 0, 1))
    return space


def noisy_sensors(seed: int = 0) -> Space:
    """The restore pod with a bad install: radar misses a quarter of the time and ghosts often,
    the gateway drops frames, the wearable is out of sync half the time, doors slam."""
    return Space(SpaceConfig(
        name="noisy_sensors", intent="restore", purpose="restore", n_occupants=1, wearable_rate=1.0,
        dwell_steps=1e6,light=0.9, sound_setting=0.7, cooling=False, arousal0=0.72,
        radar_miss=0.25, radar_ghost=0.15, dropout=0.10, db_spike=0.15, wearable_gap=0.5,
        start_hour=15.0, seed=seed,
    ))


def sensor_failure(seed: int = 0) -> Space:
    """The dB meter has died and reports a constant quiet room while five people talk.
    Motion and count disagree with sound; the model should become less sure, not more."""
    return Space(SpaceConfig(
        name="sensor_failure", intent="work", purpose="social", n_occupants=5, capacity=8,
        dwell_steps=200, light=0.8, sound_setting=0.5, cooling=True, arousal0=0.65,
        stuck_db=41.0, start_hour=10.0, seed=seed,
    ))


SCENARIOS = {
    "restore_pod": restore_pod,
    "bar_9pm": bar_9pm,
    "ambiguous_hr": ambiguous_hr,
    "lobby_peak": lobby_peak,
    "happy_group": happy_group,
    "noisy_sensors": noisy_sensors,
    "sensor_failure": sensor_failure,
}


def simulated_override(action: str, space: Space, rng: np.random.Generator) -> bool:
    """A stand-in for the staff member or guest pressing accept or dismiss.

    Dismisses actions that fight the intent, hush people who are visibly happy, page staff to an
    empty lobby, switch a room off on a guest, or fix a problem nobody has. Knows things the
    model does not: who is in the room and how they feel."""
    if action == "do_nothing":
        return False
    calming = action in ("dim_lights", "lower_sound", "cool_air")
    if action == "setback":
        p = 0.9 if space.n > 0 else 0.05
    elif space.cfg.intent == "elevated" and action in ("dim_lights", "lower_sound"):
        p = 0.9
    elif calming and space.n >= 3 and space.mean_valence() > 0.7:
        p = 0.85  # they're happy, leave them alone
    elif action == "staff_alert" and space.n < 4:
        p = 0.8
    elif calming and space.arousal < 0.35:
        p = 0.7
    else:
        p = 0.05
    return bool(rng.random() < p)

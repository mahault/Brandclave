"""Sensor feeds. The agent only ever sees a Reading; where it came from is this layer's
business. SimulatedFeed reads the simulator. HomeAssistantFeed reads a real room through
Home Assistant's REST API with the same shape, so the model does not change between them.
"""

from typing import Protocol

from .model import Reading
from .space import Space


class SensorFeed(Protocol):
    def read(self) -> Reading: ...
    def levers(self) -> dict[str, bool]: ...


class SimulatedFeed:
    def __init__(self, space: Space):
        self.space = space

    def read(self) -> Reading:
        return self.space.read()

    def levers(self) -> dict[str, bool]:
        return self.space.levers()


class HomeAssistantFeed:
    """Reads a real room. Entity ids map to the four sensors in the MVP kit:
    mmWave presence (count + motion), ambient unit (dB, temperature, lux), optional wearable.

    Not exercised by tests; it needs a running Home Assistant. Kept here so the
    interface is fixed before the hardware arrives.
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        entities: dict[str, str],
        light_entity: str | None = None,
        cooling_entity: str | None = None,
        sound_entity: str | None = None,
        timeout: float = 5.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.entities = entities  # keys: count, motion, db, temp, lux, hr (optional)
        self.light_entity = light_entity
        self.cooling_entity = cooling_entity
        self.sound_entity = sound_entity
        self.timeout = timeout

    def _state(self, entity_id: str) -> str | None:
        import requests  # local import: optional dependency path

        resp = requests.get(
            f"{self.base_url}/api/states/{entity_id}",
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json().get("state")

    def _num(self, key: str, default: float) -> float:
        ent = self.entities.get(key)
        if not ent:
            return default
        try:
            return float(self._state(ent))
        except (TypeError, ValueError):
            return default

    def read(self) -> Reading:
        hr_ent = self.entities.get("hr")
        hr = None
        if hr_ent:
            try:
                hr = float(self._state(hr_ent))
            except (TypeError, ValueError):
                hr = None
        return Reading(
            count=int(self._num("count", 0)),
            motion_level=float(self._num("motion", 0.0)),
            db=self._num("db", 40.0),
            temp_c=self._num("temp", 22.0),
            light_lux=self._num("lux", 200.0),
            hr=hr,
        )

    def levers(self) -> dict[str, bool]:
        # Without the control entities we assume every lever is available.
        light_on = True
        cooling_on = False
        sound_up = True
        if self.light_entity:
            light_on = (self._state(self.light_entity) or "").lower() != "off"
        if self.cooling_entity:
            cooling_on = (self._state(self.cooling_entity) or "").lower() in ("on", "cool", "cooling")
        if self.sound_entity:
            try:
                sound_up = float(self._state(self.sound_entity)) > 0.15
            except (TypeError, ValueError):
                sound_up = True
        return {"dim_lights": light_on, "lower_sound": sound_up, "cool_air": not cooling_on, "staff_alert": True}

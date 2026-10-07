"""Baselines the three numbers are compared against.

FixedPresetController: what a hotel does today. Lights on, cooling on, music at a fixed
level, nobody watching the room. PersistencePredictor: "it will be what it is now".
"""

from .space import Space


class FixedPresetController:
    def __init__(self, light: float = 0.8, sound_setting: float = 0.5, cooling: bool = True):
        self.light, self.sound_setting, self.cooling = light, sound_setting, cooling

    def install(self, space: Space) -> None:
        space.preset(self.light, self.sound_setting, self.cooling)

    def step(self, space: Space) -> str:
        return "do_nothing"


class PersistencePredictor:
    """Predicts the target observation will be unchanged at the horizon."""

    @staticmethod
    def predict(current: dict[str, int], modality: str) -> int:
        return current[modality]

"""A rule-based controller on the same sensors: the strongest fair comparator.

What a competent integrator would program today with the same hardware. It sees the same
readings, knows the operator intent, and gets an explicit fault detector the model does not
have. It does not learn from dismissals, because rules do not. Thresholds are the ones a
building engineer would pick, set once and not tuned on the scenarios.
"""

from .spec import Reading


class RuleController:
    def __init__(self, intent: str):
        self.intent = intent
        self.empty_reads = 0
        self.hot_while_cooling = 0

    def step(self, r: Reading, levers: dict[str, bool], cooling: bool, light: float) -> str:
        self.empty_reads = self.empty_reads + 1 if r.count == 0 else 0
        self.hot_while_cooling = self.hot_while_cooling + 1 if (cooling and r.temp_c > 24.5) else 0

        if self.empty_reads >= 3 and light > 0.15:
            return "setback"
        if self.hot_while_cooling >= 8 and levers["maintenance_ticket"]:
            return "maintenance_ticket"
        if r.hr is not None and r.hr > 90 and levers["staff_alert"]:
            return "staff_alert"
        if self.intent == "service" and levers["staff_alert"] and (
                (r.queue is not None and r.queue >= 4) or (r.queue is None and r.count >= 8)):
            return "staff_alert"
        if r.co2_ppm > 1000 and levers["ventilate"]:
            return "ventilate"
        if r.temp_c > 24.0 and levers["cool_air"]:
            return "cool_air"
        if r.temp_c < 20.5 and levers["warm_air"]:
            return "warm_air"
        if self.intent in ("restore", "work") and r.db > 62 and levers["lower_sound"]:
            return "lower_sound"
        if self.intent == "restore" and light > 0.5 and levers["dim_lights"]:
            return "dim_lights"
        if self.intent == "elevated" and r.db < 55 and levers["raise_sound"]:
            return "raise_sound"
        return "do_nothing"

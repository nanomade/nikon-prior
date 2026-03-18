import json
import os
from dataclasses import dataclass


@dataclass
class _MockPosition:
    Position: int


class MockMotorManager:
    """Software-only motor manager for environments without hardware.

    Mirrors the full public API of MotorManager so all panels work
    without changes when no Standa controllers are connected.
    """

    def __init__(self, config_file: str = "motors/step_config.json"):
        self.devices = {}
        self.step_config = {}
        self._positions_steps: dict[str, int] = {}
        self.load_step_config(config_file)

    def load_step_config(self, filename: str = "motors/step_config.json"):
        if not os.path.exists(filename):
            return
        with open(filename, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for axis, cfg in data.items():
            if not isinstance(cfg, dict):
                continue
            self.step_config[axis] = {
                "step": float(cfg["step"]),
                "invert": int(cfg.get("invert", 1)),
            }
            self._positions_steps.setdefault(axis, 0)

    # ---- motion ----

    def move(self, axis: str, steps: float):
        factor = self.step_config.get(axis, {}).get("invert", 1)
        self._positions_steps[axis] = (
            self._positions_steps.get(axis, 0) + int(round(steps * factor))
        )

    def move_absolute(self, axis: str, position: float):
        factor = self.step_config.get(axis, {}).get("invert", 1)
        self._positions_steps[axis] = int(round(position * factor))

    def move_units(self, axis: str, delta_physical: float):
        step_size = self.step_config.get(axis, {}).get("step", 1.0)
        self.move(axis, delta_physical / step_size)

    def move_absolute_units(self, axis: str, target_physical: float, wait: bool = True):
        step_size = self.step_config.get(axis, {}).get("step", 1.0)
        factor = self.step_config.get(axis, {}).get("invert", 1)
        self._positions_steps[axis] = int(round(target_physical / step_size * factor))

    def move_absolute_xy_units(self, x_mm: float, y_mm: float, wait: bool = False):
        self.move_absolute_units("X", x_mm)
        self.move_absolute_units("Y", y_mm)

    def home(self, axis: str):
        self._positions_steps[axis] = 0

    def home_all_axes(self):
        for axis in list(self._positions_steps):
            self._positions_steps[axis] = 0

    # ---- readback ----

    def get_position(self, axis: str):
        return _MockPosition(self._positions_steps.get(axis, 0))

    def get_position_units(self, axis: str) -> float:
        step_size = self.step_config.get(axis, {}).get("step", 1.0)
        factor = self.step_config.get(axis, {}).get("invert", 1)
        return self._positions_steps.get(axis, 0) * step_size * factor

    def get_speed(self, axis: str):
        return 0

    def get_speed_units(self, axis: str) -> float:
        return 0.0

    def close_all(self):
        pass

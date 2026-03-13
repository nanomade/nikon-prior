"""Gamepad control panel for the Nikon / Prior stage.

Polls an Xbox controller at 20 Hz and maps sticks/triggers/hat to XYZ.
Speed scales with the current magnification so that full deflection always
traverses ~1 field-of-view per second.
"""

import json
import os

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QProgressBar, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

from gamepad.xbox_controller import XboxController

_SETTINGS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               "gamepad_settings.json")

_DEFAULT_SETTINGS = {
    "deadzone": 0.15,
    "turbo_multiplier": 5,
    # XY step per tick (µm) per objective
    "xy_speeds": {"5x": 500, "10x": 250, "20x": 100, "50x": 40, "100x": 20},
    # Z step per tick (µm) per objective
    "z_speeds":  {"5x": 50,  "10x": 25,  "20x": 10,  "50x": 4,  "100x": 2},
}

_MAG_ORDER = ["5x", "10x", "20x", "50x", "100x"]


class GamepadPanel(QWidget):
    """Settings and live-status panel for Xbox controller stage control."""

    _OBJECTIVES = _MAG_ORDER

    def __init__(self, stage_controls, controller, autofocus_panel=None,
                 preview=None, controls=None):
        super().__init__()
        self.setWindowTitle("Gamepad Control")
        self.stage_controls = stage_controls
        self.controller = controller
        self.autofocus_panel = autofocus_panel
        self.preview = preview
        self.controls = controls

        self.xbox = XboxController()
        self._settings = dict(_DEFAULT_SETTINGS)
        self._load_settings()
        self._build_ui()

        self._prev_active = {"lx": False, "ly": False, "hat_z": False}
        self._prev_a = False
        self._prev_x = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start(50)  # 20 Hz

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout()

        # Status
        status_box = QGroupBox("Controller Status")
        status_row = QHBoxLayout()
        self._status_label = QLabel("No controller found")
        reconnect_btn = QPushButton("Reconnect")
        reconnect_btn.clicked.connect(self._try_connect)
        status_row.addWidget(self._status_label)
        status_row.addWidget(reconnect_btn)
        status_box.setLayout(status_row)
        layout.addWidget(status_box)

        self._enable_cb = QCheckBox("Enable gamepad control")
        self._enable_cb.setChecked(True)
        layout.addWidget(self._enable_cb)

        dz_row = QHBoxLayout()
        dz_row.addWidget(QLabel("Deadzone:"))
        self._dz_spin = QDoubleSpinBox()
        self._dz_spin.setRange(0.0, 0.5)
        self._dz_spin.setSingleStep(0.01)
        self._dz_spin.setDecimals(2)
        self._dz_spin.setValue(self._settings["deadzone"])
        self._dz_spin.valueChanged.connect(lambda v: setattr(self.xbox, "deadzone", v))
        dz_row.addWidget(self._dz_spin)
        dz_row.addStretch()
        layout.addLayout(dz_row)

        # Speed settings
        speed_box = QGroupBox("Step size per tick @ 20 Hz (LB = turbo)")
        form = QFormLayout()

        self._xy_speed_spins = {}
        for mag in _MAG_ORDER:
            spin = QSpinBox()
            spin.setRange(1, 50000)
            spin.setValue(self._settings["xy_speeds"].get(mag, 100))
            self._xy_speed_spins[mag] = spin
            form.addRow(f"XY {mag} (µm/tick):", spin)

        self._z_speed_spins = {}
        for mag in _MAG_ORDER:
            spin = QSpinBox()
            spin.setRange(1, 5000)
            spin.setValue(self._settings["z_speeds"].get(mag, 10))
            self._z_speed_spins[mag] = spin
            form.addRow(f"Z {mag} (µm/tick):", spin)

        self._turbo_spin = QSpinBox()
        self._turbo_spin.setRange(1, 20)
        self._turbo_spin.setValue(self._settings["turbo_multiplier"])
        form.addRow("Turbo multiplier (LB):", self._turbo_spin)

        speed_box.setLayout(form)
        layout.addWidget(speed_box)

        # Axis monitor
        monitor_box = QGroupBox("Live Axes")
        monitor_layout = QFormLayout()
        self._axis_bars = {}
        for name in ("left_x", "left_y", "right_x", "right_y", "trigger_l", "trigger_r"):
            bar = QProgressBar()
            bar.setRange(-100, 100)
            bar.setValue(0)
            bar.setFormat(f"{name}: %v%")
            self._axis_bars[name] = bar
            monitor_layout.addRow(bar)
        monitor_box.setLayout(monitor_layout)
        layout.addWidget(monitor_box)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self._save_settings)
        load_btn = QPushButton("Load Settings")
        load_btn.clicked.connect(self._load_settings_and_refresh)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(load_btn)
        layout.addLayout(btn_row)

        self.setLayout(layout)
        self._try_connect()

    # ------------------------------------------------------------------
    # Timer — 20 Hz
    # ------------------------------------------------------------------

    def _on_timer(self):
        self.xbox.poll()
        self._tick_count = getattr(self, "_tick_count", 0) + 1
        self._update_connection_status()
        if self._tick_count % 5 == 0:
            self._update_axis_monitor()

        if not self._enable_cb.isChecked() or not self.xbox.connected():
            return

        mag = getattr(self.controller, "magnification", "10x")

        # X button — cycle magnification
        x = self.xbox.get_button("x")
        if x and not self._prev_x:
            self._cycle_magnification()
        self._prev_x = x

        # A button — autofocus toggle
        a = self.xbox.get_button("a")
        if a and not self._prev_a and self.autofocus_panel is not None:
            if self.autofocus_panel.worker is not None:
                self.autofocus_panel._stop_worker()
            else:
                self.autofocus_panel._start()
        self._prev_a = a

        lb = self.xbox.get_button("lb")
        rb = self.xbox.get_button("rb")
        turbo = self._turbo_spin.value() if lb else 1
        bumper = lb

        def _sign(v):
            return 1 if v > 0 else -1

        def _should_fire(key, active):
            prev = self._prev_active[key]
            self._prev_active[key] = active
            return active and (bumper or not prev)

        # Left stick — XY stage
        xy_step = self._xy_speed_for_mag(mag) * turbo
        lx = self.xbox.get_axis("left_x")
        ly = self.xbox.get_axis("left_y")
        if _should_fire("lx", bool(lx)):
            self.stage_controls.jog_axis("X", -_sign(lx) * (1 if rb else xy_step))
        if _should_fire("ly", bool(ly)):
            self.stage_controls.jog_axis("Y", _sign(ly) * (1 if rb else xy_step))

        # D-pad — Z focus
        _, hat_y = self.xbox.get_hat()
        z_step = self._z_speed_for_mag(mag) * turbo
        if hat_y:
            self.stage_controls.jog_axis("Z", hat_y * (1 if rb else z_step))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cycle_magnification(self):
        current = getattr(self.preview, "magnification", None) or self.controller.magnification
        try:
            idx = self._OBJECTIVES.index(current)
        except ValueError:
            idx = 0
        next_mag = self._OBJECTIVES[(idx + 1) % len(self._OBJECTIVES)]
        if self.controls is not None:
            self.controls._mag_selector.setCurrentText(next_mag)
        else:
            self.controller.magnification_changed.emit(next_mag)
        if self.preview is not None and hasattr(self.preview, "flash_mag"):
            self.preview.flash_mag(next_mag)

    def _xy_speed_for_mag(self, mag: str) -> int:
        return self._xy_speed_spins.get(mag, self._xy_speed_spins["10x"]).value()

    def _z_speed_for_mag(self, mag: str) -> int:
        return self._z_speed_spins.get(mag, self._z_speed_spins["10x"]).value()

    def _try_connect(self):
        if self.xbox.connect():
            self._status_label.setText(f"Connected: {self.xbox.name()}")
        else:
            self._status_label.setText("No controller found")

    def _update_connection_status(self):
        if self.xbox.connected():
            name = self.xbox.name()
            text = f"Connected: {name}" if name else "Connected"
            if self._status_label.text() != text:
                self._status_label.setText(text)
        else:
            if not self._status_label.text().startswith("No"):
                self._status_label.setText("No controller found")
                self._try_connect()

    def _update_axis_monitor(self):
        if not self.xbox.connected():
            for bar in self._axis_bars.values():
                bar.setValue(0)
            return
        for name, bar in self._axis_bars.items():
            bar.setValue(int(self.xbox.get_axis(name) * 100))

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _save_settings(self):
        data = {
            "deadzone": self._dz_spin.value(),
            "turbo_multiplier": self._turbo_spin.value(),
            "xy_speeds": {m: self._xy_speed_spins[m].value() for m in _MAG_ORDER},
            "z_speeds":  {m: self._z_speed_spins[m].value()  for m in _MAG_ORDER},
        }
        try:
            with open(_SETTINGS_PATH, "w") as fh:
                json.dump(data, fh, indent=2)
        except Exception as exc:
            print(f"[GamepadPanel] Could not save: {exc}")

    def _load_settings(self):
        if not os.path.exists(_SETTINGS_PATH):
            return
        try:
            with open(_SETTINGS_PATH) as fh:
                self._settings.update(json.load(fh))
        except Exception as exc:
            print(f"[GamepadPanel] Could not load: {exc}")

    def _load_settings_and_refresh(self):
        self._load_settings()
        self._dz_spin.setValue(self._settings["deadzone"])
        self.xbox.deadzone = self._settings["deadzone"]
        self._turbo_spin.setValue(self._settings["turbo_multiplier"])
        for mag in _MAG_ORDER:
            self._xy_speed_spins[mag].setValue(self._settings["xy_speeds"].get(mag, 100))
            self._z_speed_spins[mag].setValue(self._settings["z_speeds"].get(mag, 10))

    def showEvent(self, event):
        self._timer.start(50)
        super().showEvent(event)

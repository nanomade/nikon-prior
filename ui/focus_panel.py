# focus_panel.py
#
# Focus presets store Z motor position in mm (relative to last zero).
# Since Z is a relative axis (coarse focus can be rotated independently),
# presets are only meaningful when the coarse focus is in the same position
# as when the preset was saved.

import json
import os

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QCheckBox,
    QPushButton, QLabel, QMessageBox,
)

_PRESETS_FILE = "focus_presets.json"


class FocusPanel(QWidget):
    def __init__(self, stage_controls, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Focus Panel")
        self.stage_controls = stage_controls

        self.objectives = ["5x", "10x", "20x", "50x", "100x"]
        # Focus Z positions per objective in mm (relative from last zero)
        self.focus_presets = {name: None for name in self.objectives}

        layout = QVBoxLayout()
        layout.addWidget(QLabel("<b>Objective Focus Presets</b>"))
        layout.addWidget(QLabel("<i>Positions are relative — re-zero if coarse focus changes.</i>"))

        self.lock_delta_checkbox = QCheckBox("Lock Delta")
        layout.addWidget(self.lock_delta_checkbox)

        self.locks = {}
        for obj in self.objectives:
            hl = QHBoxLayout()
            goto_btn = QPushButton(f"Goto {obj}")
            set_btn = QPushButton(f"Set {obj}")
            lock_box = QCheckBox("Lock")
            self.locks[obj] = lock_box
            goto_btn.clicked.connect(lambda _, o=obj: self.goto_focus(o))
            set_btn.clicked.connect(lambda _, o=obj: self.set_focus(o))
            hl.addWidget(goto_btn)
            hl.addWidget(set_btn)
            hl.addWidget(lock_box)
            lbl = QLabel("Unset")
            self.__dict__[f"label_{obj}"] = lbl
            hl.addWidget(lbl)
            layout.addLayout(hl)

        self.setLayout(layout)
        self._load_presets()
        self.update_labels()

    def _current_z_mm(self) -> float | None:
        try:
            return self.stage_controls.motor_manager.get_position_units("Z")
        except Exception:
            return None

    def goto_focus(self, obj):
        z_mm = self.focus_presets.get(obj)
        if z_mm is None:
            QMessageBox.information(self, "Not set", f"Preset for {obj} is not set.")
            return
        try:
            self.stage_controls.motor_manager.move_absolute_units("Z", z_mm, wait=False)
        except Exception as exc:
            print(f"FocusPanel: goto {obj} error: {exc}")
        self.update_labels()

    def set_focus(self, obj):
        if self.locks[obj].isChecked():
            return
        z_new = self._current_z_mm()
        if z_new is None:
            return
        z_old = self.focus_presets.get(obj)
        if not self.lock_delta_checkbox.isChecked() or \
                None in [self.focus_presets[o] for o in self.objectives if o != obj]:
            self.focus_presets[obj] = z_new
        else:
            delta = (z_new - z_old) if z_old is not None else 0.0
            for o in self.objectives:
                if self.locks[o].isChecked():
                    continue
                if o == obj:
                    self.focus_presets[o] = z_new
                else:
                    z_other = self.focus_presets.get(o)
                    if z_other is not None:
                        self.focus_presets[o] = z_other + delta
        self.update_labels()
        self._save_presets()

    def _save_presets(self):
        try:
            try:
                with open(_PRESETS_FILE) as fh:
                    data = json.load(fh)
            except Exception:
                data = {}
            data.update({obj: self.focus_presets[obj] for obj in self.objectives})
            with open(_PRESETS_FILE, "w") as fh:
                json.dump(data, fh, indent=2)
        except Exception as exc:
            print(f"FocusPanel: could not save presets: {exc}")

    def _load_presets(self):
        if not os.path.exists(_PRESETS_FILE):
            return
        try:
            with open(_PRESETS_FILE) as fh:
                data = json.load(fh)
            for obj in self.objectives:
                if obj in data and data[obj] is not None:
                    self.focus_presets[obj] = float(data[obj])
        except Exception as exc:
            print(f"FocusPanel: could not load presets: {exc}")

    def update_labels(self):
        for obj in self.objectives:
            label = getattr(self, f"label_{obj}")
            val = self.focus_presets[obj]
            label.setText(f"{val:.4f} mm" if val is not None else "Unset")

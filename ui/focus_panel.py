# focus_panel.py

import json
import os

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QPushButton, QLabel, QMessageBox

_PRESETS_FILE = "focus_presets.json"

class FocusPanel(QWidget):
    def __init__(self, stage_controls, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Focus Panel")
        self.stage_controls = stage_controls

        # Objective list (customize as needed)
        self.objectives = ["5x", "10x", "20x", "50x", "100x"]
        # Focus Z positions per objective (init to None)
        self.focus_presets = {name: None for name in self.objectives}

        layout = QVBoxLayout()
        layout.addWidget(QLabel("<b>Objective Focus Presets</b>"))

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
            self.__dict__[f"label_{obj}"] = QLabel("Unset")
            hl.addWidget(self.__dict__[f"label_{obj}"])
            layout.addLayout(hl)

        self.setLayout(layout)
        self._load_presets()
        self.update_labels()

        # Snap focus slider to nearest preset on release
        self.stage_controls.focus_z_slider.sliderReleased.connect(self._snap_to_nearest_preset)

    def goto_focus(self, obj):
        z = self.focus_presets.get(obj)
        if z is None:
            QMessageBox.information(self, "Not set", f"Preset for {obj} is not set.")
            return
        # Move slider and hardware
        self.stage_controls.focus_z_slider.setValue(z)
        # Optionally move hardware directly if needed
        if hasattr(self.stage_controls, "motor_manager"):
            self.stage_controls.motor_manager.move_absolute("Z", z)
        self.update_labels()

    def set_focus(self, obj):
        if self.locks[obj].isChecked():
            return  # Skip if locked!
        z_new = self.stage_controls.focus_z_slider.value()
        z_old = self.focus_presets.get(obj)
        # If lock delta is off, or only one preset is set, just update this one
        if not self.lock_delta_checkbox.isChecked() or None in [self.focus_presets[o] for o in self.objectives if o != obj]:
            self.focus_presets[obj] = z_new
        else:
            # Lock delta mode: shift all other objectives by the same delta
            delta = z_new - z_old if z_old is not None else 0
            for o in self.objectives:
                if self.locks[o].isChecked():
                    continue  # Skip locked objectives!
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
                    self.focus_presets[obj] = int(data[obj])
        except Exception as exc:
            print(f"FocusPanel: could not load presets: {exc}")

    def update_labels(self):
        for obj in self.objectives:
            label = getattr(self, f"label_{obj}")
            val = self.focus_presets[obj]
            label.setText(f"{val}" if val is not None else "Unset")
        self._update_detents()

    def _update_detents(self):
        detents = {v: obj for obj, v in self.focus_presets.items() if v is not None}
        slider = self.stage_controls.focus_z_slider
        if hasattr(slider, 'set_detents'):
            slider.set_detents(detents)

    def _snap_to_nearest_preset(self):
        """On slider release, snap to the nearest set preset if within 50 steps."""
        slider = self.stage_controls.focus_z_slider
        current = slider.value()
        snap_threshold = 50
        best_dist, best_z = snap_threshold + 1, None
        for obj in self.objectives:
            preset = self.focus_presets.get(obj)
            if preset is not None:
                dist = abs(current - preset)
                if dist < best_dist:
                    best_dist, best_z = dist, preset
        if best_z is not None and best_z != current:
            slider.setValue(best_z)
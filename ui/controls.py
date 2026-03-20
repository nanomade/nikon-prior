# camera_stage_refactor/ui/controls.py

import datetime
import json
import math
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QSlider, QComboBox,
    QCheckBox, QPushButton, QFileDialog, QSpinBox,
)
from PyQt5.QtCore import Qt, QTimer

_PRESETS_FILE    = "focus_presets.json"
_SETTINGS_FILE   = "ui_settings.json"
_EXP_MIN_US = 10         # 10 µs minimum (0.01 ms)
_EXP_MAX_US = 1_000_000  # 1 s maximum
_OBJECTIVES = ["5x", "10x", "20x", "50x", "100x"]


def _exp_from_pos(pos):
    """Log-scale slider position (0–1000) → exposure in µs."""
    return round(_EXP_MIN_US * (_EXP_MAX_US / _EXP_MIN_US) ** (pos / 1000.0))


def _pos_from_exp(exp_us):
    """Exposure in µs → log-scale slider position (0–1000)."""
    ratio = math.log(_EXP_MAX_US / _EXP_MIN_US)
    return round(math.log(max(_EXP_MIN_US, exp_us) / _EXP_MIN_US) / ratio * 1000)


class ControlWindow(QWidget):
    def __init__(self, controller, preview, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.controller = controller
        self.preview = preview
        self.setWindowTitle("Controls")

        self._current_mag = "10x"
        self._exp_presets = {obj: None for obj in _OBJECTIVES}
        self._load_exposure_presets()

        self.setMinimumWidth(320)
        layout = QVBoxLayout()
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 0)

        # --- Exposure (log scale, 0.01–1000 ms) ---
        grid.addWidget(QLabel("Exposure:"), 0, 0)
        self._exp_slider = QSlider(Qt.Horizontal)
        self._exp_slider.setRange(0, 1000)
        self._exp_text = QLineEdit()
        self._exp_text.setFixedWidth(72)
        self._exp_text.setToolTip("Exposure in ms — edit and press Enter to set")

        def _on_exp_slider(pos):
            exp_us = _exp_from_pos(pos)
            self._exp_text.setText(f"{exp_us / 1000:.3f}")
            controller.exposure_changed.emit(float(exp_us))

        def _on_exp_text_entered():
            try:
                exp_us = max(_EXP_MIN_US, round(float(self._exp_text.text()) * 1000))
                self._exp_slider.setValue(_pos_from_exp(exp_us))
            except ValueError:
                pass

        self._exp_slider.valueChanged.connect(_on_exp_slider)
        self._exp_text.returnPressed.connect(_on_exp_text_entered)
        self._exp_slider.setValue(_pos_from_exp(7000))  # 7000 µs = 7 ms default at 10x
        grid.addWidget(self._exp_slider, 0, 1)
        grid.addWidget(self._exp_text, 0, 2)

        # --- Gain ---
        _, self._gain_slider = self.add_slider(grid, "Gain", 0, 100, 0, 1, controller.gain_changed)

        # --- White Balance Temperature ---
        grid.addWidget(QLabel("WB Temp (K):"), 2, 0)
        self._wb_slider = QSlider(Qt.Horizontal)
        self._wb_slider.setRange(2800, 7500)
        self._wb_slider.setValue(5300)
        self._wb_slider.setTickInterval(100)
        wb_label = QLabel("5300 K")
        wb_label.setFixedWidth(55)
        def _on_wb(v):
            wb_label.setText(f"{v} K")
            controller.wb_temperature_changed.emit(v)
        self._wb_slider.valueChanged.connect(_on_wb)
        grid.addWidget(self._wb_slider, 2, 1)
        grid.addWidget(wb_label, 2, 2)

        # --- Auto Exposure ---
        self._auto_exp_check = auto_exp_check = QCheckBox("Auto Exposure")
        auto_exp_check.setChecked(False)

        def _on_auto_exposure(state):
            enabled = (state == Qt.Checked)
            self._exp_slider.setEnabled(not enabled)
            self._exp_text.setEnabled(not enabled)
            controller.auto_exposure_changed.emit(enabled)
            if not enabled and preview.cap:
                # Read back actual exposure in µs; convert to 100µs units for slider
                try:
                    current_us = preview.cap.get_exposure_us()
                    if current_us > 0:
                        self._exp_slider.setValue(_pos_from_exp(round(current_us)))
                except Exception:
                    pass

        auto_exp_check.stateChanged.connect(_on_auto_exposure)
        grid.addWidget(auto_exp_check, 3, 0, 1, 2)

        # --- Magnification ---
        grid.addWidget(QLabel("Magnification:"), 4, 0)
        self._mag_selector = QComboBox()
        self._mag_selector.addItems(_OBJECTIVES)
        self._mag_selector.currentTextChanged.connect(controller.magnification_changed.emit)
        grid.addWidget(self._mag_selector, 4, 1, 1, 2)

        # --- Mag-based exposure preset ---
        self._mag_exp_check = QCheckBox("Mag-based exposure")
        self._mag_exp_check.setChecked(True)
        grid.addWidget(self._mag_exp_check, 5, 0, 1, 2)

        # Connect _on_mag_changed only after _mag_exp_check exists, then set initial value
        self._mag_selector.currentTextChanged.connect(self._on_mag_changed)
        self._mag_selector.setCurrentText("10x")

        # --- Scale Bar Color ---
        grid.addWidget(QLabel("Scale Bar Color:"), 6, 0)
        color_selector = QComboBox()
        color_selector.addItems(["White", "Black"])
        color_selector.setCurrentText("White")
        color_selector.currentTextChanged.connect(controller.color_changed.emit)
        grid.addWidget(color_selector, 6, 1, 1, 2)

        scale_check = QCheckBox("Show Scale Bar")
        scale_check.setChecked(True)
        scale_check.stateChanged.connect(lambda state: controller.show_scale_bar_changed.emit(state == Qt.Checked))
        grid.addWidget(scale_check, 7, 0, 1, 2)

        measure_check = QCheckBox("Measure Tool")
        measure_check.setChecked(False)
        measure_check.stateChanged.connect(lambda state: controller.measure_mode_changed.emit(state == Qt.Checked))
        grid.addWidget(measure_check, 8, 0, 1, 2)

        self.crosshair_check = QCheckBox("Show Centre Zoom")
        self.crosshair_check.setChecked(False)
        self.crosshair_check.stateChanged.connect(lambda state: controller.crosshair_visible_changed.emit(state == Qt.Checked))
        grid.addWidget(self.crosshair_check, 9, 0, 1, 2)

        zoom_cursor_check = QCheckBox("Zoom Under Cursor")
        zoom_cursor_check.setChecked(False)
        zoom_cursor_check.setToolTip(
            "Show a live zoom window that follows the mouse cursor\n"
            "over the camera preview.")
        zoom_cursor_check.stateChanged.connect(lambda state: controller.zoom_under_cursor_changed.emit(state == Qt.Checked))
        grid.addWidget(zoom_cursor_check, 10, 0, 1, 2)

        full_xhair_check = QCheckBox("Full-Screen Crosshair + Ticks")
        full_xhair_check.setChecked(False)
        full_xhair_check.stateChanged.connect(lambda state: controller.full_crosshair_changed.emit(state == Qt.Checked))
        grid.addWidget(full_xhair_check, 11, 0, 1, 2)

        hud_check = QCheckBox("Show Info")
        hud_check.setChecked(False)
        hud_check.stateChanged.connect(lambda state: controller.hud_changed.emit(state == Qt.Checked))
        grid.addWidget(hud_check, 12, 0, 1, 2)

        # --- Temporal averaging (flicker suppression) ---
        grid.addWidget(QLabel("Flicker averaging:"), 13, 0)
        self._avg_spin = avg_spin = QSpinBox()
        avg_spin.setRange(1, 8)
        avg_spin.setValue(1)
        avg_spin.setSuffix(" frames")
        avg_spin.setToolTip(
            "Average this many consecutive frames before display.\n"
            "Suppresses 50/60 Hz AC lighting flicker and reduces noise.\n"
            "1 = off.  Try 3–4 for fluorescent/LED ambient light.")
        avg_spin.valueChanged.connect(preview.set_temporal_average)
        grid.addWidget(avg_spin, 13, 1, 1, 2)

        # --- Binning ---
        grid.addWidget(QLabel("Binning:"), 14, 0)
        self._binning_selector = binning_selector = QComboBox()
        binning_selector.addItems(["1x (full)", "2x", "4x"])
        # Connect before setCurrentText so the initial "4x" selection fires the signal
        binning_selector.currentTextChanged.connect(
            lambda t: controller.binning_changed.emit(int(t.split("x")[0]))
        )
        binning_selector.setCurrentText("4x")
        grid.addWidget(binning_selector, 14, 1, 1, 2)

        native_zoom_check = QCheckBox("Native Zoom (1:1)")
        native_zoom_check.setChecked(False)
        native_zoom_check.stateChanged.connect(lambda state: controller.native_zoom_toggled.emit(state == Qt.Checked))
        grid.addWidget(native_zoom_check, 15, 0, 1, 2)

        self.status_label = QLabel("Resolution: ? x ?, FPS: ?")
        grid.addWidget(self.status_label, 16, 0, 1, 2)

        save_view_button = QPushButton("Save View")
        save_view_button.clicked.connect(self.save_view)
        grid.addWidget(save_view_button, 17, 0, 1, 2)

        capture_frame_button = QPushButton("Capture Frame")
        capture_frame_button.clicked.connect(self.capture_frame)
        grid.addWidget(capture_frame_button, 18, 0, 1, 2)

        layout.addLayout(grid)
        self.setLayout(layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_status)
        self.timer.start(1000)

        self._load_settings()

    # ------------------------------------------------------------------
    # Magnification-based exposure
    # ------------------------------------------------------------------

    def _on_mag_changed(self, mag):
        self._current_mag = mag
        if not self._mag_exp_check.isChecked():
            return
        self._load_exposure_presets()  # re-read file so edits take effect without restart
        exp = self._exp_presets.get(mag)
        if exp is not None:
            self._exp_slider.setValue(_pos_from_exp(exp))  # fires exposure signal

    def _load_exposure_presets(self):
        try:
            with open(_PRESETS_FILE) as fh:
                data = json.load(fh)
            # Values stored as µs; legacy files used 100µs units (<1000 → multiply by 100)
            for mag, val in data.get("exposure_defaults", {}).items():
                if mag in self._exp_presets and val is not None:
                    us = int(val)
                    if us < 1000:       # likely old 100µs-unit value
                        us *= 100
                    self._exp_presets[mag] = us
        except Exception:
            pass

    # ------------------------------------------------------------------

    def add_slider(self, layout, label, min_val, max_val, init_val, row, signal, scale=1):
        lab = QLabel(f"{label}: {init_val}")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val * scale, max_val * scale)
        slider.setValue(init_val * scale)
        slider.valueChanged.connect(lambda val: (
            lab.setText(f"{label}: {val / scale:.1f}" if scale > 1 else f"{label}: {val}"),
            signal.emit(val / scale if scale > 1 else val)
        ))
        layout.addWidget(lab, row, 0)
        layout.addWidget(slider, row, 1)
        return lab, slider

    def save_view(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename, _ = QFileDialog.getSaveFileName(self, "Save View", f"view_{timestamp}.png", "PNG Files (*.png)")
        if filename:
            pixmap = self.preview.image_label.pixmap()
            if pixmap:
                pixmap.save(filename, "PNG")

    def capture_frame(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename, _ = QFileDialog.getSaveFileName(self, "Capture Frame", f"frame_{timestamp}.png", "PNG Files (*.png)")
        if filename:
            import cv2
            frame = self.preview.get_frame()
            if frame is not None:
                cv2.imwrite(filename, frame)

    def get_exposure_ms(self) -> float:
        """Return the currently displayed exposure in milliseconds."""
        return _exp_from_pos(self._exp_slider.value()) / 1000.0

    def update_status(self):
        width = self.preview.native_width if self.preview.cap else 0
        height = self.preview.native_height if self.preview.cap else 0
        fps = getattr(self.preview, "measured_fps", 0.0)
        bin_text = self._binning_selector.currentText().split()[0]  # "4x"
        self.status_label.setText(
            f"{width} x {height}  |  {bin_text}  |  {fps:.1f} fps")

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def _load_settings(self):
        try:
            with open(_SETTINGS_FILE) as fh:
                s = json.load(fh)
        except Exception:
            return

        # Restore magnification first so mag-based preset doesn't overwrite exp
        if "magnification" in s:
            self._mag_selector.setCurrentText(s["magnification"])
        if "mag_exp_preset" in s:
            self._mag_exp_check.setChecked(s["mag_exp_preset"])

        # Exposure — restore only when mag-based presets are off
        if "exposure_us" in s and not self._mag_exp_check.isChecked():
            self._exp_slider.setValue(_pos_from_exp(int(s["exposure_us"])))

        if "gain" in s:
            self._gain_slider.setValue(int(s["gain"]))
        if "wb_kelvin" in s:
            self._wb_slider.setValue(int(s["wb_kelvin"]))
        if "auto_exposure" in s:
            self._auto_exp_check.setChecked(bool(s["auto_exposure"]))
        if "binning" in s:
            self._binning_selector.setCurrentText(s["binning"])
        if "flicker_avg" in s:
            self._avg_spin.setValue(int(s["flicker_avg"]))

    def _save_settings(self):
        try:
            s = {
                "exposure_us":     _exp_from_pos(self._exp_slider.value()),
                "gain":            self._gain_slider.value(),
                "wb_kelvin":       self._wb_slider.value(),
                "auto_exposure":   self._auto_exp_check.isChecked(),
                "magnification":   self._mag_selector.currentText(),
                "mag_exp_preset":  self._mag_exp_check.isChecked(),
                "binning":         self._binning_selector.currentText(),
                "flicker_avg":     self._avg_spin.value(),
            }
            with open(_SETTINGS_FILE, "w") as fh:
                json.dump(s, fh, indent=2)
        except Exception:
            pass

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)

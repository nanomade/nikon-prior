# camera_stage_refactor/ui/controls.py

import datetime
import json
import math
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QLabel, QSlider, QComboBox,
    QCheckBox, QPushButton, QFileDialog, QSpinBox,
)
from PyQt5.QtCore import Qt, QTimer

_PRESETS_FILE = "focus_presets.json"
_EXP_LOG_MAX = 5000  # max exposure in 100µs units (= 500 ms)
_OBJECTIVES = ["5x", "10x", "20x", "50x", "100x"]


def _exp_from_pos(pos):
    """Log-scale slider position (0–1000) → exposure in 100µs units."""
    return round(10 ** (pos / 1000.0 * math.log10(_EXP_LOG_MAX)))


def _pos_from_exp(exp):
    """Exposure in 100µs units → log-scale slider position (0–1000)."""
    return round(math.log10(max(1, exp)) / math.log10(_EXP_LOG_MAX) * 1000)


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

        layout = QVBoxLayout()
        grid = QGridLayout()

        # --- Exposure (log scale, 0.1–500 ms) ---
        self._exp_label = QLabel()
        self._exp_slider = QSlider(Qt.Horizontal)
        self._exp_slider.setRange(0, 1000)

        def _on_exp_slider(pos):
            exp = _exp_from_pos(pos)
            self._exp_label.setText(f"Exposure: {exp * 0.1:.1f} ms")
            controller.exposure_changed.emit(float(exp))

        self._exp_slider.valueChanged.connect(_on_exp_slider)
        self._exp_slider.setValue(_pos_from_exp(500))  # 50 ms default
        grid.addWidget(self._exp_label, 0, 0)
        grid.addWidget(self._exp_slider, 0, 1)

        # --- Gain ---
        self.add_slider(grid, "Gain", 0, 100, 0, 1, controller.gain_changed)

        # --- White Balance Temperature ---
        grid.addWidget(QLabel("WB Temp (K):"), 2, 0)
        wb_slider = QSlider(Qt.Horizontal)
        wb_slider.setRange(2800, 7500)
        wb_slider.setValue(4600)
        wb_slider.setTickInterval(100)
        wb_label = QLabel("4600 K")
        wb_label.setFixedWidth(55)
        def _on_wb(v):
            wb_label.setText(f"{v} K")
            controller.wb_temperature_changed.emit(v)
        wb_slider.valueChanged.connect(_on_wb)
        grid.addWidget(wb_slider, 2, 1)
        grid.addWidget(wb_label, 2, 2)

        # --- Auto Exposure ---
        auto_exp_check = QCheckBox("Auto Exposure")
        auto_exp_check.setChecked(False)

        def _on_auto_exposure(state):
            enabled = (state == Qt.Checked)
            self._exp_label.setEnabled(not enabled)
            self._exp_slider.setEnabled(not enabled)
            controller.auto_exposure_changed.emit(enabled)
            if not enabled and preview.cap:
                import cv2
                current = preview.cap.get(cv2.CAP_PROP_EXPOSURE)
                if current > 0:
                    self._exp_slider.setValue(_pos_from_exp(round(current)))

        auto_exp_check.stateChanged.connect(_on_auto_exposure)
        grid.addWidget(auto_exp_check, 3, 0, 1, 2)

        # --- Magnification ---
        grid.addWidget(QLabel("Magnification:"), 4, 0)
        self._mag_selector = QComboBox()
        self._mag_selector.addItems(_OBJECTIVES)
        self._mag_selector.currentTextChanged.connect(controller.magnification_changed.emit)
        grid.addWidget(self._mag_selector, 4, 1)

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
        grid.addWidget(color_selector, 6, 1)

        scale_check = QCheckBox("Show Scale Bar")
        scale_check.setChecked(True)
        scale_check.stateChanged.connect(lambda state: controller.show_scale_bar_changed.emit(state == Qt.Checked))
        grid.addWidget(scale_check, 7, 0, 1, 2)

        measure_check = QCheckBox("Measure Tool")
        measure_check.setChecked(False)
        measure_check.stateChanged.connect(lambda state: controller.measure_mode_changed.emit(state == Qt.Checked))
        grid.addWidget(measure_check, 8, 0, 1, 2)

        self.crosshair_check = QCheckBox("Show Center Crosshair")
        self.crosshair_check.setChecked(False)
        self.crosshair_check.stateChanged.connect(lambda state: controller.crosshair_visible_changed.emit(state == Qt.Checked))
        grid.addWidget(self.crosshair_check, 9, 0, 1, 2)

        full_xhair_check = QCheckBox("Full-Screen Crosshair + Ticks")
        full_xhair_check.setChecked(False)
        full_xhair_check.stateChanged.connect(lambda state: controller.full_crosshair_changed.emit(state == Qt.Checked))
        grid.addWidget(full_xhair_check, 10, 0, 1, 2)

        hud_check = QCheckBox("Show HUD")
        hud_check.setChecked(False)
        hud_check.stateChanged.connect(lambda state: controller.hud_changed.emit(state == Qt.Checked))
        grid.addWidget(hud_check, 11, 0, 1, 2)

        # --- Temporal averaging (flicker suppression) ---
        grid.addWidget(QLabel("Flicker averaging:"), 12, 0)
        avg_spin = QSpinBox()
        avg_spin.setRange(1, 8)
        avg_spin.setValue(1)
        avg_spin.setSuffix(" frames")
        avg_spin.setToolTip(
            "Average this many consecutive frames before display.\n"
            "Suppresses 50/60 Hz AC lighting flicker and reduces noise.\n"
            "1 = off.  Try 3–4 for fluorescent/LED ambient light.")
        avg_spin.valueChanged.connect(preview.set_temporal_average)
        grid.addWidget(avg_spin, 12, 1)

        grid.addWidget(QLabel("Resolution:"), 13, 0)
        res_selector = QComboBox()
        resolutions = [
            (4032, 3040), (3840, 2160), (2592, 1944), (2560, 1440),
            (1920, 1080), (1600, 1200), (1280, 960), (1280, 760), (640, 480)
        ]
        for res in resolutions:
            res_selector.addItem(f"{res[0]} x {res[1]}", res)
        res_selector.currentIndexChanged.connect(lambda i: controller.resolution_changed.emit(res_selector.itemData(i)))
        res_selector.setCurrentText("1600 x 1200")
        grid.addWidget(res_selector, 13, 1)

        native_zoom_check = QCheckBox("Native Zoom (1:1)")
        native_zoom_check.setChecked(False)
        native_zoom_check.stateChanged.connect(lambda state: controller.native_zoom_toggled.emit(state == Qt.Checked))
        grid.addWidget(native_zoom_check, 14, 0, 1, 2)

        save_view_button = QPushButton("Save View")
        save_view_button.clicked.connect(self.save_view)
        grid.addWidget(save_view_button, 15, 0, 1, 2)

        capture_frame_button = QPushButton("Capture Frame")
        capture_frame_button.clicked.connect(self.capture_frame)
        grid.addWidget(capture_frame_button, 16, 0, 1, 2)

        self.status_label = QLabel("Resolution: ? x ?, FPS: ?")
        grid.addWidget(self.status_label, 15, 0, 1, 2)

        layout.addLayout(grid)
        self.setLayout(layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_status)
        self.timer.start(1000)

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
            for mag, val in data.get("exposure_defaults", {}).items():
                if mag in self._exp_presets and val is not None:
                    self._exp_presets[mag] = int(val)
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
            ret, frame = self.preview.cap.read()
            if ret:
                import cv2
                cv2.imwrite(filename, frame)

    def update_status(self):
        width = int(self.preview.cap.get(3)) if self.preview.cap else 0
        height = int(self.preview.cap.get(4)) if self.preview.cap else 0
        fps = getattr(self.preview, "measured_fps", 0.0)
        self.status_label.setText(f"Resolution: {width} x {height}, FPS: {fps:.1f}")

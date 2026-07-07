# camera_stage_refactor/ui/controls.py

import json
import math
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit, QSlider,
    QComboBox, QCheckBox, QPushButton, QSpinBox, QStyle, QStyleOptionSlider,
)
from PyQt5.QtCore import Qt, QTimer, QEvent, QObject
from PyQt5.QtGui import QColor, QPainter, QPen

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


class _WheelGuard(QObject):
    """Swallow wheel events on value widgets unless they have keyboard focus.

    Scrolling over the panel must never silently drag exposure/gain/WB/binning
    — the operator is navigating, not editing. Click a widget first to make
    the wheel adjust it deliberately.
    """

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel and not obj.hasFocus():
            event.ignore()
            return True   # consume: don't change the value, don't scroll-adjust
        return super().eventFilter(obj, event)


class _DetentSlider(QSlider):
    """Horizontal slider with a painted notch at a neutral detent value.
    Snaps to the detent on mouse release if within ±2 ticks."""

    def __init__(self, detent=0, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self._detent = detent

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if abs(self.value() - self._detent) <= 2:
            self.setValue(self._detent)

    def paintEvent(self, event):
        super().paintEvent(event)
        lo, hi = self.minimum(), self.maximum()
        if lo == hi:
            return
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        groove = self.style().subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)
        frac = (self._detent - lo) / (hi - lo)
        x = int(groove.left() + frac * groove.width())
        p = QPainter(self)
        p.setPen(QPen(QColor(60, 120, 255), 2))
        p.drawLine(x, groove.top() - 1, x, groove.bottom() + 1)
        p.end()


class ControlWindow(QWidget):
    def __init__(self, controller, preview, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.controller = controller
        self.preview = preview
        self.setWindowTitle("Controls")

        self._current_mag = "10x"
        self._exp_presets = {obj: None for obj in _OBJECTIVES}
        self._wb_defaults = {"red": 1.0, "blue": 1.0}
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
        self._exp_slider.setValue(_pos_from_exp(7000))  # 7 ms until presets/settings load
        grid.addWidget(self._exp_slider, 0, 1)
        grid.addWidget(self._exp_text, 0, 2)

        # --- Gain (0–48 dB in 0.1 dB steps, with text entry) ---
        grid.addWidget(QLabel("Gain (dB):"), 1, 0)
        self._gain_slider = QSlider(Qt.Horizontal)
        self._gain_slider.setRange(0, 480)   # ticks = dB × 10
        self._gain_text = QLineEdit()
        self._gain_text.setFixedWidth(50)
        self._gain_text.setToolTip("Gain in dB (0–48) — edit and press Enter to set")

        def _on_gain_slider(v):
            db = v / 10.0
            self._gain_text.setText(f"{db:.1f}")
            controller.gain_changed.emit(db)

        def _on_gain_text():
            try:
                db = max(0.0, min(48.0, float(self._gain_text.text())))
                self._gain_slider.setValue(int(round(db * 10)))
            except ValueError:
                pass

        self._gain_slider.valueChanged.connect(_on_gain_slider)
        self._gain_text.returnPressed.connect(_on_gain_text)
        self._gain_slider.setValue(0)
        self._gain_text.setText("0.0")
        grid.addWidget(self._gain_slider, 1, 1)
        grid.addWidget(self._gain_text, 1, 2)

        # --- White Balance (manual Red/Blue gains; no auto-WB on this rig) ---
        grid.addWidget(QLabel("White Bal:"), 2, 0)
        wb_row = QHBoxLayout()

        # Debounce timer for persisting WB to focus_presets.json
        self._wb_save_timer = QTimer(self)
        self._wb_save_timer.setSingleShot(True)
        self._wb_save_timer.setInterval(500)
        self._wb_save_timer.timeout.connect(self._save_wb)

        wb_row.addWidget(QLabel("R"))
        self._wb_red_slider = QSlider(Qt.Horizontal)
        self._wb_red_slider.setRange(5, 160)      # value/20 → 0.25–8.00
        self._wb_red_text = QLineEdit("1.00")
        self._wb_red_text.setFixedWidth(44)
        self._wb_red_text.setToolTip("Red channel gain (0.25–8.00) — edit and press Enter")

        def _on_wb_red(v):
            self._wb_red_text.setText(f"{v / 20:.2f}")
            controller.wb_red_changed.emit(v / 20.0)
            self._wb_save_timer.start()

        def _on_wb_red_text():
            try:
                v = max(5, min(160, int(round(float(self._wb_red_text.text()) * 20))))
                self._wb_red_slider.setValue(v)
            except ValueError:
                pass

        self._wb_red_slider.valueChanged.connect(_on_wb_red)
        self._wb_red_text.returnPressed.connect(_on_wb_red_text)
        self._wb_red_slider.setValue(20)
        wb_row.addWidget(self._wb_red_slider)
        wb_row.addWidget(self._wb_red_text)

        wb_row.addWidget(QLabel("B"))
        self._wb_blue_slider = QSlider(Qt.Horizontal)
        self._wb_blue_slider.setRange(5, 160)
        self._wb_blue_text = QLineEdit("1.00")
        self._wb_blue_text.setFixedWidth(44)
        self._wb_blue_text.setToolTip("Blue channel gain (0.25–8.00) — edit and press Enter")

        def _on_wb_blue(v):
            self._wb_blue_text.setText(f"{v / 20:.2f}")
            controller.wb_blue_changed.emit(v / 20.0)
            self._wb_save_timer.start()

        def _on_wb_blue_text():
            try:
                v = max(5, min(160, int(round(float(self._wb_blue_text.text()) * 20))))
                self._wb_blue_slider.setValue(v)
            except ValueError:
                pass

        self._wb_blue_slider.valueChanged.connect(_on_wb_blue)
        self._wb_blue_text.returnPressed.connect(_on_wb_blue_text)
        self._wb_blue_slider.setValue(20)
        wb_row.addWidget(self._wb_blue_slider)
        wb_row.addWidget(self._wb_blue_text)

        grid.addLayout(wb_row, 2, 1, 1, 2)

        # Apply the rig's WB defaults from focus_presets.json (tungsten R/B
        # gains, kept up to date by the debounced save). setValue fires
        # valueChanged, which pushes the ratios to the camera.
        self._wb_red_slider.setValue(max(5, min(160, round(self._wb_defaults["red"] * 20))))
        self._wb_blue_slider.setValue(max(5, min(160, round(self._wb_defaults["blue"] * 20))))

        # --- Auto Exposure ---
        self._auto_exp_check = auto_exp_check = QCheckBox("Auto Exposure")
        auto_exp_check.setChecked(False)

        def _on_auto_exposure(state):
            enabled = (state == Qt.Checked)
            self._exp_slider.setEnabled(not enabled)
            self._exp_text.setEnabled(not enabled)
            controller.auto_exposure_changed.emit(enabled)
            if not enabled and preview.cap:
                # Capture what the AE algorithm settled on (exposure AND gain)
                # so switching to manual doesn't jump brightness.
                try:
                    current_us = preview.cap.get_exposure_us()
                    if current_us > 0:
                        self._exp_slider.setValue(_pos_from_exp(round(current_us)))
                    db = preview.cap.get_gain_db()
                    if db >= 0:
                        self._gain_slider.setValue(int(round(db * 10)))
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

        # --- Mag-based exposure preset + Set Preset ---
        preset_row = QHBoxLayout()
        self._mag_exp_check = QCheckBox("Mag-based exposure")
        self._mag_exp_check.setChecked(True)
        preset_row.addWidget(self._mag_exp_check)
        save_preset_btn = QPushButton("Set Preset")
        save_preset_btn.setToolTip(
            "Save the current exposure as the default for the current\n"
            "magnification (focus_presets.json → exposure_defaults).")
        save_preset_btn.clicked.connect(self._save_exposure_preset)
        preset_row.addWidget(save_preset_btn)
        grid.addLayout(preset_row, 5, 0, 1, 3)

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

        # ── Checkboxes: Display | Tools (two columns) ─────────────────────
        scale_check = QCheckBox("Show Scale Bar")
        scale_check.setChecked(True)
        scale_check.stateChanged.connect(lambda state: controller.show_scale_bar_changed.emit(state == Qt.Checked))

        hud_check = QCheckBox("Show Info")
        hud_check.setChecked(False)
        hud_check.stateChanged.connect(lambda state: controller.hud_changed.emit(state == Qt.Checked))

        self.crosshair_check = QCheckBox("Zoom Target")
        self.crosshair_check.setChecked(False)
        self.crosshair_check.stateChanged.connect(lambda state: controller.crosshair_visible_changed.emit(state == Qt.Checked))

        center_xhair_check = QCheckBox("Centre Cross")
        center_xhair_check.setChecked(True)
        center_xhair_check.stateChanged.connect(lambda state: controller.center_crosshair_changed.emit(state == Qt.Checked))

        full_xhair_check = QCheckBox("Full Crosshair")
        full_xhair_check.setChecked(False)
        full_xhair_check.stateChanged.connect(lambda state: controller.full_crosshair_changed.emit(state == Qt.Checked))

        self.measure_check = QCheckBox("Measure")
        self.measure_check.setChecked(False)
        self.measure_check.stateChanged.connect(lambda state: controller.measure_mode_changed.emit(state == Qt.Checked))

        self.measure_area_check = QCheckBox("Measure Area")
        self.measure_area_check.setChecked(False)
        self.measure_area_check.setToolTip(
            "Left-click to place polygon vertices; right-click to close.\n"
            "Shows the enclosed area in px² and µm².")
        self.measure_area_check.stateChanged.connect(lambda state: controller.measure_area_mode_changed.emit(state == Qt.Checked))

        native_zoom_check = QCheckBox("Native Zoom (1:1)")
        native_zoom_check.setChecked(False)
        native_zoom_check.stateChanged.connect(lambda state: controller.native_zoom_toggled.emit(state == Qt.Checked))

        zoom_cursor_check = QCheckBox("Zoom Under Cursor")
        zoom_cursor_check.setChecked(False)
        zoom_cursor_check.setToolTip(
            "Show a live zoom window that follows the mouse cursor\n"
            "over the camera preview.")
        zoom_cursor_check.stateChanged.connect(lambda state: controller.zoom_under_cursor_changed.emit(state == Qt.Checked))

        # Back-sync: the preview can programmatically clear these modes; keep
        # the checkboxes honest without re-emitting.
        def _uncheck(cb, v):
            cb.blockSignals(True)
            cb.setChecked(v)
            cb.blockSignals(False)

        controller.measure_mode_changed.connect(lambda v: _uncheck(self.measure_check, v))
        controller.measure_area_mode_changed.connect(lambda v: _uncheck(self.measure_area_check, v))
        controller.crosshair_visible_changed.connect(lambda v: _uncheck(self.crosshair_check, v))

        # Two-column layout: left=Display, right=Tools
        grid.addWidget(scale_check,             7, 0)
        grid.addWidget(hud_check,               7, 1, 1, 2)
        grid.addWidget(self.crosshair_check,    8, 0)
        grid.addWidget(center_xhair_check,      8, 1, 1, 2)
        grid.addWidget(full_xhair_check,        9, 0)
        grid.addWidget(self.measure_check,      9, 1, 1, 2)
        grid.addWidget(native_zoom_check,      10, 0)
        grid.addWidget(self.measure_area_check, 10, 1, 1, 2)
        grid.addWidget(zoom_cursor_check,      11, 0)

        # --- Temporal averaging (flicker suppression) ---
        grid.addWidget(QLabel("Flicker averaging:"), 12, 0)
        self._avg_spin = avg_spin = QSpinBox()
        avg_spin.setRange(1, 8)
        avg_spin.setValue(1)
        avg_spin.setSuffix(" frames")
        avg_spin.setToolTip(
            "Average this many consecutive frames before display.\n"
            "Suppresses 50/60 Hz AC lighting flicker and reduces noise.\n"
            "1 = off.  Try 3–4 for fluorescent/LED ambient light.")
        avg_spin.valueChanged.connect(preview.set_temporal_average)
        grid.addWidget(avg_spin, 12, 1, 1, 2)

        # --- Binning (this rig runs 1x always; 2x/4x remain available) ---
        grid.addWidget(QLabel("Binning:"), 13, 0)
        self._binning_selector = binning_selector = QComboBox()
        binning_selector.addItems(["1x (full)", "2x", "4x"])
        # Connect before setCurrentText so the initial selection fires the signal
        binning_selector.currentTextChanged.connect(
            lambda t: controller.binning_changed.emit(int(t.split("x")[0]))
        )
        binning_selector.setCurrentText("1x (full)")
        grid.addWidget(binning_selector, 13, 1, 1, 2)

        self.status_label = QLabel("Resolution: ? x ?, FPS: ?")
        grid.addWidget(self.status_label, 14, 0, 1, 3)

        layout.addLayout(grid)

        # ── Image Processing (display-only LUT; collapsed by default) ─────
        ip_content = QWidget()
        ip_grid = QGridLayout(ip_content)
        ip_grid.setColumnStretch(1, 1)

        def _ip_row(row, label, lo, hi, detent, signal):
            ip_grid.addWidget(QLabel(label), row, 0)
            sl = _DetentSlider(detent=detent)
            sl.setRange(lo, hi)
            sl.setValue(detent)
            sp = QSpinBox()
            sp.setRange(lo, hi)
            sp.setValue(detent)
            sp.setFixedWidth(52)
            sl.valueChanged.connect(lambda v, s=sp, sig=signal: (
                s.blockSignals(True), s.setValue(v), s.blockSignals(False),
                sig.emit(v)
            ))
            sp.valueChanged.connect(lambda v, s=sl, sig=signal: (
                s.blockSignals(True), s.setValue(v), s.blockSignals(False),
                sig.emit(v)
            ))
            ip_grid.addWidget(sl, row, 1)
            ip_grid.addWidget(sp, row, 2)
            return sl, sp

        self._brightness_slider, self._brightness_spin = _ip_row(
            0, "Brightness:", -64, 64, 0,   controller.brightness_changed)
        self._contrast_slider,   self._contrast_spin   = _ip_row(
            1, "Contrast:",   0,   64, 32,  controller.contrast_changed)
        self._gamma_slider,      self._gamma_spin      = _ip_row(
            2, "Gamma:",      72,  500, 100, controller.gamma_changed)
        self._sharpness_slider,  self._sharpness_spin  = _ip_row(
            3, "Sharpness:",  0,   6,   0,  controller.sharpness_changed)
        # Sharpness is not implemented on the Alvium — hide until wired
        for w in (self._sharpness_slider, self._sharpness_spin):
            w.setVisible(False)
        ip_grid.itemAtPosition(3, 0).widget().setVisible(False)

        ip_btn = QPushButton("  ▶  Image Processing")
        ip_btn.setCheckable(True)
        ip_btn.setChecked(False)
        ip_btn.setStyleSheet("text-align: left; padding-left: 14px;")

        def _toggle_ip(checked):
            ip_content.setVisible(checked)
            ip_btn.setText(f"  {'▼' if checked else '▶'}  Image Processing")

        ip_btn.clicked.connect(_toggle_ip)
        ip_content.setVisible(False)
        layout.addWidget(ip_btn)
        layout.addWidget(ip_content)

        # ── Live Histogram (collapsed by default; ported from standa_stacker) ──
        from ui.histogram_widget import LiveHistogram
        self._histogram = LiveHistogram(preview)
        hist_btn = QPushButton("  ▶  Live Histogram")
        hist_btn.setCheckable(True)
        hist_btn.setChecked(False)
        hist_btn.setStyleSheet("text-align: left; padding-left: 14px;")

        def _toggle_hist(checked):
            self._histogram.setVisible(checked)
            hist_btn.setText(f"  {'▼' if checked else '▶'}  Live Histogram")

        hist_btn.clicked.connect(_toggle_hist)
        self._histogram.setVisible(False)
        layout.addWidget(hist_btn)
        layout.addWidget(self._histogram)

        self.setLayout(layout)

        # Guard every value widget against accidental scroll-wheel edits:
        # wheel only adjusts a widget that was clicked (has focus) first.
        self._wheel_guard = _WheelGuard(self)
        for w in self.findChildren((QSlider, QSpinBox, QComboBox)):
            w.setFocusPolicy(Qt.StrongFocus)
            w.installEventFilter(self._wheel_guard)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_status)
        self.timer.start(1000)

        self._load_settings()

        # Explicit ordered push of all UI values to the camera once the event
        # loop starts (construction-time emits can race camera setup).
        QTimer.singleShot(0, self.apply_to_camera)

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
            wb = data.get("wb", {})
            for ch in ("red", "blue"):
                if wb.get(ch) is not None:
                    self._wb_defaults[ch] = float(wb[ch])
        except Exception:
            pass

    def _save_exposure_preset(self):
        """Save the current exposure as the preset for the current magnification."""
        mag = self._mag_selector.currentText()
        exp_us = _exp_from_pos(self._exp_slider.value())
        self._exp_presets[mag] = exp_us
        try:
            try:
                with open(_PRESETS_FILE) as fh:
                    data = json.load(fh)
            except Exception:
                data = {}
            data.setdefault("exposure_defaults", {})[mag] = int(exp_us)
            with open(_PRESETS_FILE, "w") as fh:
                json.dump(data, fh, indent=2)
            print(f"[exposure] saved preset: {mag} = {exp_us / 1000:.3f} ms")
        except Exception as exc:
            print(f"[exposure] could not save preset: {exc}")

    # ------------------------------------------------------------------
    # White balance persistence (focus_presets.json is the single WB store)
    # ------------------------------------------------------------------

    def _save_wb(self):
        """Persist current R/B WB ratios to focus_presets.json (debounced)."""
        try:
            try:
                with open(_PRESETS_FILE) as fh:
                    data = json.load(fh)
            except Exception:
                data = {}
            data["wb"] = {
                "red":  round(self._wb_red_slider.value()  / 20.0, 4),
                "blue": round(self._wb_blue_slider.value() / 20.0, 4),
            }
            with open(_PRESETS_FILE, "w") as fh:
                json.dump(data, fh, indent=2)
        except Exception as exc:
            print(f"[wb] save failed: {exc}")

    # ------------------------------------------------------------------

    def apply_to_camera(self):
        """Push all current UI values to the camera in a defined order."""
        auto = self._auto_exp_check.isChecked()
        self.controller.auto_exposure_changed.emit(auto)
        if not auto:
            self.controller.exposure_changed.emit(float(_exp_from_pos(self._exp_slider.value())))
        self.controller.gain_changed.emit(self._gain_slider.value() / 10.0)
        self.controller.wb_red_changed.emit(self._wb_red_slider.value() / 20.0)
        self.controller.wb_blue_changed.emit(self._wb_blue_slider.value() / 20.0)
        self.controller.brightness_changed.emit(self._brightness_spin.value())
        self.controller.contrast_changed.emit(self._contrast_spin.value())
        self.controller.gamma_changed.emit(self._gamma_spin.value())
        QTimer.singleShot(400, self._log_camera_state)

    def _log_camera_state(self):
        """Read back camera state after applying settings; warn on mismatch."""
        info = self.preview.get_camera_info() if self.preview else None
        if info is None:
            return
        exp_ms = info['exposure_ms']
        exp_ui_ms = _exp_from_pos(self._exp_slider.value()) / 1000.0
        mode = "auto" if info['auto_exposure'] else "manual"
        print(f"[camera] after-init: exp={exp_ms:.3f}ms (UI={exp_ui_ms:.3f}ms)  "
              f"gain={info['gain_db']:.1f}dB  auto-exp={mode}")
        if not info['auto_exposure'] and exp_ui_ms > 0 and abs(exp_ms - exp_ui_ms) / exp_ui_ms > 0.25:
            print(f"[camera] WARNING: exposure mismatch  "
                  f"camera={exp_ms:.3f}ms  UI={exp_ui_ms:.3f}ms")

    def get_exposure_ms(self) -> float:
        """Return the currently displayed exposure in milliseconds."""
        return _exp_from_pos(self._exp_slider.value()) / 1000.0

    def get_imaging_metadata(self) -> dict:
        """Return a dict of all current imaging settings for metadata logging."""
        data = {
            'magnification': self._mag_selector.currentText(),
            'exposure_ms':   self.get_exposure_ms(),
            'auto_exposure': self._auto_exp_check.isChecked(),
            'gain':          self._gain_slider.value() / 10.0,
            'wb_red':        self._wb_red_slider.value() / 20.0,
            'wb_blue':       self._wb_blue_slider.value() / 20.0,
            'brightness':    self._brightness_spin.value(),
            'contrast':      self._contrast_spin.value(),
            'gamma':         self._gamma_spin.value(),
            'sharpness':     self._sharpness_spin.value(),
        }
        if self.preview:
            info = self.preview.get_camera_info()
            if info:
                data['camera_exposure_ms']   = round(info['exposure_ms'], 3)
                data['camera_gain_db']       = round(info['gain_db'], 2)
                data['camera_auto_exposure'] = info['auto_exposure']
                data['frame_width']          = info['width']
                data['frame_height']         = info['height']
                data['camera_model']         = 'Alvium 1800 U-508c (IMX250, BayerRG8)'
            data['measured_fps'] = round(getattr(self.preview, 'measured_fps', 0.0), 1)
        return data

    def update_status(self):
        width = self.preview.native_width if self.preview.cap else 0
        height = self.preview.native_height if self.preview.cap else 0
        fps = getattr(self.preview, "measured_fps", 0.0)
        bin_text = self._binning_selector.currentText().split()[0]  # "1x"
        cam_str = ""
        info = self.preview.get_camera_info() if self.preview else None
        if info:
            exp_ms = info['exposure_ms']
            gain_db = info['gain_db']
            demanded_ms = _exp_from_pos(self._exp_slider.value()) / 1000.0
            if info['auto_exposure']:
                cam_str = f"  |  {exp_ms:.2f} ms  {gain_db:.1f} dB [AUTO]"
            else:
                flag = " !" if abs(exp_ms - demanded_ms) > max(0.05, demanded_ms * 0.15) else ""
                cam_str = f"  |  {exp_ms:.2f} ms{flag}  {gain_db:.1f} dB"
        self.status_label.setText(
            f"{width} x {height}  |  {bin_text}  |  {fps:.1f} fps{cam_str}")

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

        if "gain_db" in s:
            self._gain_slider.setValue(int(round(float(s["gain_db"]) * 10)))
        elif "gain" in s:   # legacy: whole dB int
            self._gain_slider.setValue(int(s["gain"]) * 10)
        # WB is NOT restored from ui_settings — focus_presets.json is the
        # single WB store (loaded at construction, saved on every change).
        if "auto_exposure" in s:
            self._auto_exp_check.setChecked(bool(s["auto_exposure"]))
        if "binning" in s:
            self._binning_selector.setCurrentText(s["binning"])
        if "flicker_avg" in s:
            self._avg_spin.setValue(int(s["flicker_avg"]))
        if "brightness" in s:
            self._brightness_spin.setValue(int(s["brightness"]))
        if "contrast" in s:
            self._contrast_spin.setValue(int(s["contrast"]))
        if "gamma" in s:
            self._gamma_spin.setValue(int(s["gamma"]))

    def _save_settings(self):
        try:
            s = {
                "exposure_us":     _exp_from_pos(self._exp_slider.value()),
                "gain_db":         self._gain_slider.value() / 10.0,
                "auto_exposure":   self._auto_exp_check.isChecked(),
                "magnification":   self._mag_selector.currentText(),
                "mag_exp_preset":  self._mag_exp_check.isChecked(),
                "binning":         self._binning_selector.currentText(),
                "flicker_avg":     self._avg_spin.value(),
                "brightness":      self._brightness_spin.value(),
                "contrast":        self._contrast_spin.value(),
                "gamma":           self._gamma_spin.value(),
            }
            with open(_SETTINGS_FILE, "w") as fh:
                json.dump(s, fh, indent=2)
        except Exception:
            pass

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)

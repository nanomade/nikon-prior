"""Stage control panel for the Prior ProScan III XYZ stage.

Slider values are in µm (0.001 mm per unit) throughout the UI.
Motor commands are converted to mm via move_absolute_units / move_units.

Axes: X, Y (stage translation), Z (focus).  No rotation axis.
"""

import json
import os

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox, QFileDialog, QHBoxLayout, QLabel, QPushButton,
    QSlider, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

# Slider units are µm.  Ranges match typical Prior H117 / MFC stage.
_X_RANGE_UM = 65_000   # ±65 mm
_Y_RANGE_UM = 42_500   # ±42.5 mm
_Z_RANGE_UM = 25_000   # 0 – 25 mm

_UM_PER_MM = 1000.0    # slider value × this factor = mm when dividing


class DetentSlider(QSlider):
    """Vertical slider that draws small labelled arrows at preset detent positions."""

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self._detent_groups = {}
        self.setMinimumWidth(72)

    @property
    def _detents(self):
        merged = {}
        for g in self._detent_groups.values():
            merged.update(g)
        return merged

    def set_detents(self, detents: dict, group: str = "default"):
        self._detent_groups[group] = dict(detents)
        self.update()

    def clear_detents(self, group: str = "default"):
        self._detent_groups.pop(group, None)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        detents = self._detents
        if not detents:
            return
        rng = self.maximum() - self.minimum()
        if rng == 0:
            return
        painter = QPainter(self)
        font = QFont("Arial", 10)
        painter.setFont(font)
        fm = painter.fontMetrics()
        h, w = self.height(), self.width()
        for val, label in detents.items():
            frac = 1.0 - (val - self.minimum()) / rng
            yp = int(frac * (h - 4) + 2)
            painter.setPen(QColor(30, 100, 210))
            painter.drawLine(w - 58, yp, w - 46, yp)
            painter.drawText(w - 45, yp + fm.ascent() // 2, f"\u25b6 {label}")
        painter.end()


class InteractiveStageDisplay(QLabel):
    """Click/drag XY stage map."""

    def __init__(self, stage_controls):
        super().__init__()
        self.sc = stage_controls
        self.dragging = False
        self.setMouseTracking(True)

    def _pixel_to_slider(self, pos):
        w, h = self.width(), self.height()
        if not w or not h:
            return None, None
        x_val = int(pos.x() / w * 2 * _X_RANGE_UM - _X_RANGE_UM)
        y_val = int((1.0 - pos.y() / h) * 2 * _Y_RANGE_UM - _Y_RANGE_UM)
        return x_val, y_val

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self._apply(event)

    def mouseMoveEvent(self, event):
        if self.dragging and event.buttons() == Qt.LeftButton:
            self._apply(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False

    def mouseDoubleClickEvent(self, event):
        self._apply(event)

    def _apply(self, event):
        x_val, y_val = self._pixel_to_slider(event.pos())
        if x_val is None:
            return
        self.sc.stage_x_slider.setValue(x_val)
        self.sc.stage_y_slider.setValue(y_val)


# ---------------------------------------------------------------------------
# StageControlWindow
# ---------------------------------------------------------------------------

class StageControlWindow(QWidget):
    """XYZ stage control: sliders, jog buttons, position readout."""

    def __init__(self, preview, motor_manager=None):
        super().__init__()
        self.preview = preview
        self.setWindowTitle("Stage Controls")
        self.setFocusPolicy(Qt.StrongFocus)

        if motor_manager is not None:
            self.motor_manager = motor_manager
        else:
            from motors.factory import create_motor_manager
            self.motor_manager = create_motor_manager()

        self.position_manager = None  # injected after construction
        self.focus_panel = None       # injected after construction

        self._build_ui()
        self._sync_sliders_to_motors()
        self.update_all_displays()

        self._mini_update_timer = QTimer(self)
        self._mini_update_timer.timeout.connect(self.update_all_displays)
        self._mini_update_timer.start(200)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        main_layout = QHBoxLayout()

        # ---- XY panel ----
        xy_panel = QVBoxLayout()
        xy_panel.addWidget(QLabel("Stage X / Y  (µm)"))

        display_row = QHBoxLayout()
        self.stage_y_slider = QSlider(Qt.Vertical)
        self.stage_y_slider.setRange(-_Y_RANGE_UM, _Y_RANGE_UM)
        display_row.addWidget(self.stage_y_slider)

        self.stage_display = InteractiveStageDisplay(self)
        self.stage_display.setFixedSize(500, 500)
        display_row.addWidget(self.stage_display)
        xy_panel.addLayout(display_row)

        x_row = QHBoxLayout()
        x_row.addWidget(QLabel("X"))
        self.stage_x_slider = QSlider(Qt.Horizontal)
        self.stage_x_slider.setRange(-_X_RANGE_UM, _X_RANGE_UM)
        x_row.addWidget(self.stage_x_slider)
        xy_panel.addLayout(x_row)

        jog_row = QHBoxLayout()
        for label, axis, delta in [
            ("←←←", "X", -1000), ("←←", "X", -100), ("←", "X", -10),
            ("→", "X", 10), ("→→", "X", 100), ("→→→", "X", 1000),
            ("↑↑↑", "Y", 1000), ("↑↑", "Y", 100), ("↑", "Y", 10),
            ("↓", "Y", -10), ("↓↓", "Y", -100), ("↓↓↓", "Y", -1000),
        ]:
            btn = QPushButton(label)
            btn.setFixedWidth(40)
            btn.clicked.connect(lambda _, a=axis, d=delta: self.jog_axis(a, d))
            jog_row.addWidget(btn)
        xy_panel.addLayout(jog_row)

        main_layout.addLayout(xy_panel)

        # ---- Z panel ----
        z_panel = QVBoxLayout()
        z_panel.setAlignment(Qt.AlignHCenter)
        z_panel.addWidget(QLabel("Z focus (µm)"))

        z_plus_btn = QPushButton("Z+")
        z_plus_btn.clicked.connect(lambda: self.jog_axis("Z", 10))
        z_panel.addWidget(z_plus_btn)

        self.focus_z_slider = DetentSlider(Qt.Vertical)
        self.focus_z_slider.setRange(0, _Z_RANGE_UM)
        self.focus_z_slider.setTickPosition(QSlider.TicksLeft)
        self.focus_z_slider.setTickInterval(1000)
        z_panel.addWidget(self.focus_z_slider)

        z_minus_btn = QPushButton("Z-")
        z_minus_btn.clicked.connect(lambda: self.jog_axis("Z", -10))
        z_panel.addWidget(z_minus_btn)
        main_layout.addLayout(z_panel)

        # ---- Full layout ----
        full_layout = QVBoxLayout()
        full_layout.addLayout(main_layout)

        self.value_display = QLabel("Setpoint: X: 0 µm,  Y: 0 µm,  Z: 0 µm")
        self.value_display.setAlignment(Qt.AlignCenter)
        self.value_display.setFont(QFont("Arial", 11, QFont.Bold))
        full_layout.addWidget(self.value_display)

        self.unit_display = QLabel("Motor position: X: N/A,  Y: N/A,  Z: N/A")
        self.unit_display.setAlignment(Qt.AlignCenter)
        self.unit_display.setFont(QFont("Arial", 11))
        full_layout.addWidget(self.unit_display)

        btn_row = QHBoxLayout()
        zero_btn = QPushButton("Zero position here")
        zero_btn.setToolTip("Set current stage position as the 0,0,0 origin")
        zero_btn.clicked.connect(self._zero_here)
        goto_zero_btn = QPushButton("Go to origin")
        goto_zero_btn.clicked.connect(self.goto_zero)
        btn_row.addWidget(zero_btn)
        btn_row.addWidget(goto_zero_btn)
        full_layout.addLayout(btn_row)

        self.setLayout(full_layout)

        # Wire sliders
        for slider in (self.stage_x_slider, self.stage_y_slider):
            slider.valueChanged.connect(self.update_all_displays)
        self.stage_x_slider.valueChanged.connect(lambda v: self._slider_moved("X", v))
        self.stage_y_slider.valueChanged.connect(lambda v: self._slider_moved("Y", v))
        self.focus_z_slider.valueChanged.connect(lambda v: self._slider_moved("Z", v))

    # ------------------------------------------------------------------
    # Sync motor → sliders
    # ------------------------------------------------------------------

    def _sync_sliders_to_motors(self):
        for axis, slider in [("X", self.stage_x_slider),
                              ("Y", self.stage_y_slider),
                              ("Z", self.focus_z_slider)]:
            try:
                pos_mm = self.motor_manager.get_position_units(axis)
                if pos_mm is not None:
                    slider.blockSignals(True)
                    slider.setValue(int(round(pos_mm * _UM_PER_MM)))
                    slider.blockSignals(False)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Display update
    # ------------------------------------------------------------------

    def update_all_displays(self):
        # Readout labels
        x_um = self.stage_x_slider.value()
        y_um = self.stage_y_slider.value()
        z_um = self.focus_z_slider.value()
        self.value_display.setText(
            f"Setpoint:  X: {x_um} µm    Y: {y_um} µm    Z: {z_um} µm"
        )

        def _fmt(val, unit):
            try:
                return f"{val:.4f} {unit}" if val is not None else "N/A"
            except Exception:
                return "N/A"

        try:
            mx = self.motor_manager.get_position_units("X")
            my = self.motor_manager.get_position_units("Y")
            mz = self.motor_manager.get_position_units("Z")
            self.unit_display.setText(
                f"Motor:  X: {_fmt(mx, 'mm')}    Y: {_fmt(my, 'mm')}    Z: {_fmt(mz, 'mm')}"
            )
        except Exception:
            pass

        # XY stage map
        w, h = self.stage_display.width(), self.stage_display.height()
        img = np.full((h, w, 3), 255, dtype=np.uint8)

        # Grid lines
        for i in range(0, w, w // 10):
            img[:, i] = [200, 200, 200]
        for j in range(0, h, h // 10):
            img[j, :] = [200, 200, 200]

        # Breadcrumb thumbnails + labels
        if self.position_manager:
            for pos in self.position_manager.positions:
                thumb = pos.get("Thumbnail")
                if thumb is None:
                    continue
                # Slider value (µm) → pixel on display
                px_x = int((pos["X"] + _X_RANGE_UM) / (2 * _X_RANGE_UM) * w)
                px_y = int((1.0 - (pos["Y"] + _Y_RANGE_UM) / (2 * _Y_RANGE_UM)) * h)
                hm, wm = thumb.shape[:2]
                x0 = int(np.clip(px_x - wm // 2, 0, w - wm))
                y0 = int(np.clip(px_y - hm // 2, 0, h - hm))
                img[y0:y0 + hm, x0:x0 + wm] = thumb
                label = pos.get("Name", "")
                if label:
                    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
                    lx = int(np.clip(px_x - tw // 2, 0, w - tw - 1))
                    ly = int(np.clip(y0 + hm + th + 1, th + 1, h - 1))
                    cv2.putText(img, label, (lx + 1, ly + 1), font, scale, (0, 0, 0), thick + 1)
                    cv2.putText(img, label, (lx, ly), font, scale, (255, 255, 255), thick)

        # Current position dot
        cx = int((x_um + _X_RANGE_UM) / (2 * _X_RANGE_UM) * w)
        cy = int((1.0 - (y_um + _Y_RANGE_UM) / (2 * _Y_RANGE_UM)) * h)
        cv2.circle(img, (cx, cy), 6, (0, 180, 0), -1)
        cv2.circle(img, (cx, cy), 6, (0, 0, 0), 1)

        qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
        self.stage_display.setPixmap(QPixmap.fromImage(qimg))

        # Push Z detents from focus panel
        if self.focus_panel is not None and hasattr(self.focus_panel, "get_detents"):
            self.focus_z_slider.set_detents(
                self.focus_panel.get_detents(), group="objectives"
            )

    # ------------------------------------------------------------------
    # Motor control
    # ------------------------------------------------------------------

    def _slider_moved(self, axis: str, value_um: int):
        """Called when a slider moves — drives motor to the new position."""
        try:
            self.motor_manager.move_absolute_units(axis, value_um / _UM_PER_MM,
                                                    wait=False)
        except Exception as exc:
            print(f"[StageControls] {axis} move error: {exc}")

    def jog_axis(self, axis: str, steps_um: int):
        """Jog axis by steps_um µm (positive = positive direction)."""
        sliders = {"X": self.stage_x_slider, "Y": self.stage_y_slider,
                   "Z": self.focus_z_slider}
        if axis in sliders:
            sliders[axis].setValue(sliders[axis].value() + steps_um)

    def _zero_here(self):
        """Tell the ProScan III that the current position is the origin."""
        try:
            self.motor_manager.home()
        except Exception as exc:
            print(f"[StageControls] zero error: {exc}")
        for slider in (self.stage_x_slider, self.stage_y_slider, self.focus_z_slider):
            slider.blockSignals(True)
            slider.setValue(0)
            slider.blockSignals(False)
        self.update_all_displays()

    def goto_zero(self):
        for slider in (self.stage_x_slider, self.stage_y_slider, self.focus_z_slider):
            slider.setValue(0)

    def _snap_to_nearest_preset(self):
        """Snap focus_z_slider to the nearest objective preset (called by focus_panel)."""
        if self.focus_panel is None:
            return
        detents = self.focus_panel.get_detents() if hasattr(self.focus_panel, "get_detents") else {}
        if not detents:
            return
        cur = self.focus_z_slider.value()
        nearest = min(detents, key=lambda v: abs(v - cur))
        self.focus_z_slider.setValue(nearest)

    # ------------------------------------------------------------------
    # Keyboard jog
    # ------------------------------------------------------------------

    def keyPressEvent(self, event):
        step = 10 if (event.modifiers() & Qt.ShiftModifier) else 100
        key = event.key()
        if key == Qt.Key_Left:
            self.jog_axis("X", -step)
        elif key == Qt.Key_Right:
            self.jog_axis("X", step)
        elif key == Qt.Key_Up:
            self.jog_axis("Y", step)
        elif key == Qt.Key_Down:
            self.jog_axis("Y", -step)
        elif key == Qt.Key_PageUp:
            self.jog_axis("Z", step)
        elif key == Qt.Key_PageDown:
            self.jog_axis("Z", -step)
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if hasattr(self, "_mini_update_timer"):
            self._mini_update_timer.stop()
        try:
            self.motor_manager.close()
        except Exception:
            pass
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# PositionManagerWindow
# ---------------------------------------------------------------------------

class PositionManagerWindow(QWidget):
    """Save, recall and manage named stage positions."""

    def __init__(self, stage_controls: StageControlWindow):
        super().__init__()
        self.setWindowTitle("Stage Positions")
        self.stage_controls = stage_controls
        self.positions = [
            {"Name": f"Pos {i+1}", "Locked": False,
             "X": 0, "Y": 0, "Z": 0, "Thumbnail": None, "Mag": None}
            for i in range(10)
        ]

        layout = QVBoxLayout()
        self.table = QTableWidget(10, 5)
        self.table.setHorizontalHeaderLabels(["Name", "X (µm)", "Y (µm)", "Z (µm)", "Locked"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(
            QTableWidget.DoubleClicked | QTableWidget.SelectedClicked
        )
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        for label, slot in [
            ("Save Current", self.save_current_position),
            ("Recall Selected", self.recall_selected_position),
            ("Save to File", self.save_to_file),
            ("Load from File", self.load_from_file),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            btn_row.addWidget(btn)
        layout.addLayout(btn_row)
        self.setLayout(layout)
        self.refresh_table()

    # ------------------------------------------------------------------

    def _get_stage_values(self) -> dict:
        sc = self.stage_controls
        return {
            "X": sc.stage_x_slider.value(),
            "Y": sc.stage_y_slider.value(),
            "Z": sc.focus_z_slider.value(),
        }

    def _set_stage_values(self, pos: dict):
        sc = self.stage_controls
        sc.stage_x_slider.setValue(pos["X"])
        sc.stage_y_slider.setValue(pos["Y"])
        sc.focus_z_slider.setValue(pos["Z"])

    def save_current_position(self):
        row = self.table.currentRow()
        if row < 0 or self.positions[row]["Locked"]:
            return
        self.positions[row].update(self._get_stage_values())
        name_item = self.table.item(row, 0)
        if name_item:
            self.positions[row]["Name"] = name_item.text()

        # Thumbnail
        sc = self.stage_controls
        if getattr(sc, "preview", None) is not None:
            frame = sc.preview.get_latest_frame()
            mag = getattr(sc.preview, "magnification", "10x")
            if frame is not None:
                mag_num = {"5x": 5, "10x": 10, "20x": 20, "50x": 50, "100x": 100}.get(mag, 10)
                fov_mm = 0.835 * 10.0 / mag_num
                sz = max(1, round(fov_mm * 750.0 / 75.0))
                import cv2 as _cv
                thumb = _cv.resize(
                    _cv.cvtColor(frame, _cv.COLOR_BGR2RGB) if frame.shape[2] == 3 else frame,
                    (sz, sz), interpolation=_cv.INTER_AREA,
                )
                self.positions[row]["Thumbnail"] = thumb
                self.positions[row]["Mag"] = mag
            else:
                self.positions[row]["Thumbnail"] = None
                self.positions[row]["Mag"] = None

        self.positions[row]["Locked"] = True
        self.refresh_table()

    def recall_selected_position(self):
        row = self.table.currentRow()
        if row < 0:
            return
        self._set_stage_values(self.positions[row])
        self.stage_controls.update_all_displays()

    def refresh_table(self):
        self.table.blockSignals(True)
        for row, pos in enumerate(self.positions):
            name_item = QTableWidgetItem(pos["Name"])
            name_item.setFlags(name_item.flags() | Qt.ItemIsEditable)
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, QTableWidgetItem(str(pos["X"])))
            self.table.setItem(row, 2, QTableWidgetItem(str(pos["Y"])))
            self.table.setItem(row, 3, QTableWidgetItem(str(pos["Z"])))
            chk = QCheckBox()
            chk.setChecked(pos["Locked"])
            chk.stateChanged.connect(
                lambda state, r=row: self._set_lock(r, state)
            )
            self.table.setCellWidget(row, 4, chk)
        self.table.blockSignals(False)

    def _set_lock(self, row: int, state):
        self.positions[row]["Locked"] = (state == Qt.Checked)

    def save_to_file(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Positions", "positions.json", "JSON Files (*.json)"
        )
        if filename:
            data = [{k: v for k, v in p.items() if k != "Thumbnail"}
                    for p in self.positions]
            with open(filename, "w") as fh:
                json.dump(data, fh, indent=2)

    def load_from_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Positions", "", "JSON Files (*.json)"
        )
        if not filename:
            return
        with open(filename) as fh:
            loaded = json.load(fh)
        if isinstance(loaded, list):
            self.positions = loaded[:10]
            while len(self.positions) < 10:
                n = len(self.positions) + 1
                self.positions.append(
                    {"Name": f"Pos {n}", "Locked": False,
                     "X": 0, "Y": 0, "Z": 0, "Thumbnail": None, "Mag": None}
                )
            for p in self.positions:
                p.setdefault("Thumbnail", None)
                p.setdefault("Mag", None)
            self.refresh_table()

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
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox, QFileDialog, QHBoxLayout, QLabel, QPushButton,
    QSlider, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

_UM_PER_MM = 1000.0    # slider value × this factor = mm when dividing

# Fallback limits (mm) used if step_config lacks min_mm/max_mm.
_X_MIN_MM_DEFAULT = -127.5
_X_MAX_MM_DEFAULT =  127.5
_Y_MIN_MM_DEFAULT = -107.5
_Y_MAX_MM_DEFAULT =  107.5

# Z jog button step size (µm) per objective magnification.
# Higher mag → shallower depth of focus → smaller step.
_Z_JOG_UM = {"5x": 40, "10x": 20, "20x": 10, "50x": 4, "100x": 2}



class ZVelocitySlider(QSlider):
    """Vertical slider that springs back to 0 on mouse release.

    Used for Z fine-focus velocity control: position encodes speed, not
    absolute position.  The stage_controls timer reads the value every 50 ms
    and issues a proportional relative Z move.
    """

    def __init__(self, parent=None):
        super().__init__(Qt.Vertical, parent)
        self.setRange(-100, 100)
        self.setValue(0)
        self.setTickPosition(QSlider.TicksBothSides)
        self.setTickInterval(25)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.setValue(0)


class InteractiveStageDisplay(QLabel):
    """Click/drag XY stage map."""

    def __init__(self, stage_controls):
        super().__init__()
        self.sc = stage_controls
        self.dragging = False
        self.setMouseTracking(True)

    def _pixel_to_mm(self, pos):
        w, h = self.width(), self.height()
        if not w or not h:
            return None, None
        x_travel = self.sc._x_max_um - self.sc._x_min_um
        y_travel = self.sc._y_max_um - self.sc._y_min_um
        x_mm = (pos.x() / w * x_travel + self.sc._x_min_um) / _UM_PER_MM
        y_mm = ((1.0 - pos.y() / h) * y_travel + self.sc._y_min_um) / _UM_PER_MM
        return x_mm, y_mm

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
        x_mm, y_mm = self._pixel_to_mm(event.pos())
        if x_mm is None:
            return
        # Clamp to stage travel limits
        x_mm = max(self.sc._x_min_um / _UM_PER_MM, min(self.sc._x_max_um / _UM_PER_MM, x_mm))
        y_mm = max(self.sc._y_min_um / _UM_PER_MM, min(self.sc._y_max_um / _UM_PER_MM, y_mm))
        # Update slider display without triggering individual motor moves
        self.sc.stage_x_slider.blockSignals(True)
        self.sc.stage_y_slider.blockSignals(True)
        self.sc.stage_x_slider.setValue(int(round(x_mm * _UM_PER_MM)))
        self.sc.stage_y_slider.setValue(int(round(y_mm * _UM_PER_MM)))
        self.sc.stage_x_slider.blockSignals(False)
        self.sc.stage_y_slider.blockSignals(False)
        # Single combined XY move — avoids axis desync from two separate commands
        self.sc._move_xy(x_mm, y_mm)


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

        self._z_jog_step_um = 20   # updated by magnification signal; default = 10x value
        self._init_axis_limits()
        self._build_ui()
        self._sync_sliders_to_motors()
        self.update_all_displays()
        # Connect magnification changes so Z jog step scales with objective
        if hasattr(preview, 'controller'):
            preview.controller.magnification_changed.connect(self._on_mag_changed)

        self._mini_update_timer = QTimer(self)
        self._mini_update_timer.timeout.connect(self.update_all_displays)
        self._mini_update_timer.start(200)

        self._z_vel_timer = QTimer(self)
        self._z_vel_timer.timeout.connect(self._z_velocity_step)
        self._z_vel_timer.start(50)

    # ------------------------------------------------------------------
    # Axis limits (read from step_config via motor_manager)
    # ------------------------------------------------------------------

    def _init_axis_limits(self):
        cfg = getattr(self.motor_manager, "step_config", {})
        x = cfg.get("X", {})
        y = cfg.get("Y", {})
        self._x_min_um = int(x.get("min_mm", _X_MIN_MM_DEFAULT) * _UM_PER_MM)
        self._x_max_um = int(x.get("max_mm", _X_MAX_MM_DEFAULT) * _UM_PER_MM)
        self._y_min_um = int(y.get("min_mm", _Y_MIN_MM_DEFAULT) * _UM_PER_MM)
        self._y_max_um = int(y.get("max_mm", _Y_MAX_MM_DEFAULT) * _UM_PER_MM)

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
        self.stage_y_slider.setRange(self._y_min_um, self._y_max_um)
        display_row.addWidget(self.stage_y_slider)

        # Size map to correct stage aspect ratio
        x_travel = self._x_max_um - self._x_min_um
        y_travel = self._y_max_um - self._y_min_um
        map_w = 500
        map_h = max(100, int(round(map_w * y_travel / x_travel)))
        self.stage_display = InteractiveStageDisplay(self)
        self.stage_display.setFixedSize(map_w, map_h)
        display_row.addWidget(self.stage_display)
        xy_panel.addLayout(display_row)

        x_row = QHBoxLayout()
        x_row.addWidget(QLabel("X"))
        self.stage_x_slider = QSlider(Qt.Horizontal)
        self.stage_x_slider.setRange(self._x_min_um, self._x_max_um)
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
        z_panel.addWidget(QLabel("Z fine focus"))

        self._z_plus_btn = QPushButton("Z+")
        self._z_plus_btn.clicked.connect(lambda: self.jog_axis("Z", self._z_jog_step_um))
        z_panel.addWidget(self._z_plus_btn)

        self.z_velocity_slider = ZVelocitySlider()
        self.z_velocity_slider.setToolTip(
            "Hold to move fine focus.  Drag toward + to focus up, - to focus down.\n"
            "Releases back to centre on mouse-up."
        )
        z_panel.addWidget(self.z_velocity_slider)

        self._z_minus_btn = QPushButton("Z-")
        self._z_minus_btn.clicked.connect(lambda: self.jog_axis("Z", -self._z_jog_step_um))
        z_panel.addWidget(self._z_minus_btn)

        self._z_step_label = QLabel(f"step: {self._z_jog_step_um} µm")
        self._z_step_label.setAlignment(Qt.AlignCenter)
        z_panel.addWidget(self._z_step_label)

        self.z_pos_label = QLabel("Z: 0.0000 mm")
        self.z_pos_label.setAlignment(Qt.AlignCenter)
        z_panel.addWidget(self.z_pos_label)

        main_layout.addLayout(z_panel)

        # ---- Full layout ----
        full_layout = QVBoxLayout()
        full_layout.addLayout(main_layout)

        self.value_display = QLabel("Setpoint  X:   0.0000 mm   Y:   0.0000 mm")
        self.value_display.setAlignment(Qt.AlignCenter)
        self.value_display.setFont(QFont("Courier New", 11, QFont.Bold))
        full_layout.addWidget(self.value_display)

        self.unit_display = QLabel("Motor     X:        N/A   Y:        N/A   Z:        N/A (rel)")
        self.unit_display.setAlignment(Qt.AlignCenter)
        self.unit_display.setFont(QFont("Courier New", 11))
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
        self.stage_x_slider.valueChanged.connect(self.update_all_displays)
        self.stage_y_slider.valueChanged.connect(self.update_all_displays)
        self.stage_x_slider.valueChanged.connect(lambda v: self._slider_moved("X", v))
        self.stage_y_slider.valueChanged.connect(lambda v: self._slider_moved("Y", v))

    # ------------------------------------------------------------------
    # Sync motor → sliders
    # ------------------------------------------------------------------

    def _sync_sliders_to_motors(self):
        for axis, slider in [("X", self.stage_x_slider), ("Y", self.stage_y_slider)]:
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
        # Readout labels — both in mm, fixed-width so columns align
        x_mm = self.stage_x_slider.value() / _UM_PER_MM
        y_mm = self.stage_y_slider.value() / _UM_PER_MM
        self.value_display.setText(
            f"Setpoint  X: {x_mm:+9.4f} mm   Y: {y_mm:+9.4f} mm"
        )

        def _fmt(val):
            try:
                return f"{val:+9.4f} mm" if val is not None else "      N/A   "
            except Exception:
                return "      N/A   "

        try:
            mx = self.motor_manager.get_position_units("X")
            my = self.motor_manager.get_position_units("Y")
            mz = self.motor_manager.get_position_units("Z")
            self.unit_display.setText(
                f"Motor     X: {_fmt(mx)}   Y: {_fmt(my)}   Z: {_fmt(mz)} (rel)"
            )
            self.z_pos_label.setText(f"Z: {_fmt(mz)} (rel)")
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

        x_travel = self._x_max_um - self._x_min_um
        y_travel = self._y_max_um - self._y_min_um

        def _um_to_px(x_um_, y_um_):
            px = int((x_um_ - self._x_min_um) / x_travel * w)
            py = int((1.0 - (y_um_ - self._y_min_um) / y_travel) * h)
            return px, py

        # Breadcrumb thumbnails + labels
        if self.position_manager:
            for pos in self.position_manager.positions:
                thumb = pos.get("Thumbnail")
                if thumb is None:
                    continue
                px_x, px_y = _um_to_px(pos["X"], pos["Y"])
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
        cx, cy = _um_to_px(int(x_mm * _UM_PER_MM), int(y_mm * _UM_PER_MM))
        cv2.circle(img, (cx, cy), 6, (0, 180, 0), -1)
        cv2.circle(img, (cx, cy), 6, (0, 0, 0), 1)

        qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
        self.stage_display.setPixmap(QPixmap.fromImage(qimg))


    # ------------------------------------------------------------------
    # Motor control
    # ------------------------------------------------------------------

    def _slider_moved(self, axis: str, value_um: int):
        """Called when an XY slider moves — drives motor to the new position."""
        try:
            self.motor_manager.move_absolute_units(axis, value_um / _UM_PER_MM,
                                                    wait=False)
        except Exception as exc:
            print(f"[StageControls] {axis} move error: {exc}")

    def _move_xy(self, x_mm: float, y_mm: float):
        """Send a single combined XY move — prevents axis desync."""
        try:
            self.motor_manager.move_absolute_xy_units(x_mm, y_mm, wait=False)
        except Exception as exc:
            print(f"[StageControls] XY move error: {exc}")

    def _z_velocity_step(self):
        """Fired every 50 ms by the Z velocity timer; moves Z relative to slider value."""
        vel = self.z_velocity_slider.value()
        if vel == 0:
            return
        # vel ∈ [-100, 100]: at max → 0.05 mm per 50 ms tick = 1.0 mm/s
        step_mm = vel * 0.0005
        try:
            self.motor_manager.move_units("Z", step_mm, wait=False)
        except Exception as exc:
            print(f"[StageControls] Z velocity error: {exc}")

    def jog_axis(self, axis: str, steps_um: int):
        """Jog axis by steps_um µm. XY via slider; Z via direct relative move."""
        if axis == "Z":
            try:
                self.motor_manager.move_units("Z", steps_um / _UM_PER_MM, wait=False)
            except Exception as exc:
                print(f"[StageControls] Z jog error: {exc}")
        else:
            sliders = {"X": self.stage_x_slider, "Y": self.stage_y_slider}
            if axis in sliders:
                sliders[axis].setValue(sliders[axis].value() + steps_um)

    def _on_mag_changed(self, mag: str):
        """Scale Z jog button step to match the objective depth of focus."""
        self._z_jog_step_um = _Z_JOG_UM.get(mag, 10)
        self._z_step_label.setText(f"step: {self._z_jog_step_um} µm")

    def _zero_here(self):
        """Tell the ProScan III that the current position is the origin."""
        try:
            self.motor_manager.home()
        except Exception as exc:
            print(f"[StageControls] zero error: {exc}")
        for slider in (self.stage_x_slider, self.stage_y_slider):
            slider.blockSignals(True)
            slider.setValue(0)
            slider.blockSignals(False)
        self.update_all_displays()

    def goto_zero(self):
        """Move XY to origin; Z returns to its last-zeroed position."""
        self.stage_x_slider.blockSignals(True)
        self.stage_y_slider.blockSignals(True)
        self.stage_x_slider.setValue(0)
        self.stage_y_slider.setValue(0)
        self.stage_x_slider.blockSignals(False)
        self.stage_y_slider.blockSignals(False)
        self._move_xy(0.0, 0.0)
        try:
            self.motor_manager.move_absolute_units("Z", 0.0, wait=False)
        except Exception as exc:
            print(f"[StageControls] goto_zero Z error: {exc}")

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
        for attr in ("_mini_update_timer", "_z_vel_timer"):
            timer = getattr(self, attr, None)
            if timer:
                timer.stop()
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
        try:
            z_mm = sc.motor_manager.get_position_units("Z") or 0.0
        except Exception:
            z_mm = 0.0
        return {
            "X": sc.stage_x_slider.value(),
            "Y": sc.stage_y_slider.value(),
            "Z": int(round(z_mm * 1000)),  # store as µm for table display
        }

    def _set_stage_values(self, pos: dict):
        sc = self.stage_controls
        x_mm = pos["X"] / _UM_PER_MM
        y_mm = pos["Y"] / _UM_PER_MM
        # Update sliders without triggering individual axis moves
        sc.stage_x_slider.blockSignals(True)
        sc.stage_y_slider.blockSignals(True)
        sc.stage_x_slider.setValue(pos["X"])
        sc.stage_y_slider.setValue(pos["Y"])
        sc.stage_x_slider.blockSignals(False)
        sc.stage_y_slider.blockSignals(False)
        # Single combined XY move
        sc._move_xy(x_mm, y_mm)
        # Z is relative; recalling a saved position means going to that Z
        try:
            sc.motor_manager.move_absolute_units("Z", pos.get("Z", 0) / 1000.0, wait=False)
        except Exception:
            pass

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

# camera_stage_refactor/ui/preview.py

import collections
import cv2
import numpy as np
import time
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QProgressBar, QSizePolicy
from PyQt5.QtCore import QTimer, Qt, QPoint, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QMouseEvent

from camera_manager import create_camera_manager

class PreviewWindow(QWidget):
    _warned_calibrations = set()  # suppress repeated per-frame warnings
    color_sampled = pyqtSignal(int, int, int)  # r, g, b — emitted after eyedropper pick

    def __init__(self, controller):
        super().__init__()
        self.setWindowTitle("Camera Preview")
        self.controller = controller
    
        # Calibration table: magnification string → pixels per µm (ppm).
        # The Alvium 1800 U-508c always runs at full resolution (2464×2056,
        # decimation not supported), so resolution is not a calibration axis.
        #
        # Theoretical starting point: Sony IMX250 pixel pitch = 3.45 µm,
        #   ppm_theory = objective_mag / 3.45
        # These must be verified with a stage micrometer and updated below.
        # Set a magnification to None to mark it as uncalibrated — the measure
        # tool will then show distances in pixels only.
        self.calibration_table = {
            "5x":   1.015,       # implied from 10x (linear scaling)
            "10x":  2.03,        # measured 2026-03-18 on nikon-257 with stage micrometer
            "20x":  4.06,        # implied from 10x
            "50x":  10.15,       # implied from 10x
            "100x": 20.30,       # implied from 10x
        }

        self.cap = create_camera_manager()
        self.native_width  = self.cap.native_width
        self.native_height = self.cap.native_height
        print(f"Camera ready: {self.native_width}×{self.native_height}")

        self.image_label = QLabel()
        self.image_label.setMinimumSize(320, 240)
        self.image_label.setScaledContents(False)
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self.handle_click
        self.image_label.mouseDoubleClickEvent = self.handle_double_click
        self.image_label.mouseMoveEvent = self.track_mouse

        self.motor_manager = None    # injected by stagecontrol after construction
        self.stage_controls = None   # injected by stagecontrol after construction
        self.manip_controls = None   # injected by stagecontrol after construction
        self.layer_panel      = None    # injected by LayerContrastPanel on construction
        self.mark_panel       = None    # injected by IndexMarkPanel on construction
        self.flat_field_panel = None    # injected by FlatFieldPanel on construction

        self._pick_mode = False
        self._last_raw_frame = None
        self._avg_n = 1                  # 1 = off; set via set_temporal_average()
        self._avg_buf = collections.deque()

        self.display_offset = (0, 0)
        self.display_scale = (1.0, 1.0)
        self.last_mouse_pos = QPoint(self.native_width // 2, self.native_height // 2)

        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.resize(self.native_width, self.native_height)

        # Laplacian bar — lives below the image, outside the image area
        self._lap_bar = QProgressBar()
        self._lap_bar.setRange(0, 1000)
        self._lap_bar.setTextVisible(False)
        self._lap_bar.setFixedHeight(10)
        self._lap_label = QLabel("Focus: —")
        self._lap_label.setFixedHeight(16)

        lap_row = QHBoxLayout()
        lap_row.setContentsMargins(0, 0, 0, 0)
        lap_row.addWidget(self._lap_bar)
        lap_row.addWidget(self._lap_label)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.image_label)
        layout.addLayout(lap_row)
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        self.measure_points = []
        self.show_crosshair = False
        self.show_full_crosshair = False
        self.show_hud = False
        self.measure_mode = False
        self._hud_cache = []
        self._hud_tick = 0
        self._mag_flash_text = None
        self._mag_flash_until = 0.0

        self.zoom_window = ZoomWindow()
        self.zoom_window.hide()
        self.zoom_window.double_clicked.connect(self._on_zoom_double_clicked)
        self._zoom_center_frame = (0, 0)  # updated each frame

        self.last_time = time.time()
        self.frame_count = 0
        self.measured_fps = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # Imaging state — mirrored here so the info overlay can read them back.
        self._info_exposure_us = 6000.0
        self._info_gain_db     = 0.0
        self._info_wb_kelvin   = 5300
        self._info_auto_exp    = False
        self._info_binning     = 4

        # controls.py emits exposure in µs (after the µs refactor).
        controller.exposure_changed.connect(self._on_exposure_changed)
        controller.gain_changed.connect(self._on_gain_changed)
        controller.auto_exposure_changed.connect(self._set_auto_exposure)
        controller.wb_temperature_changed.connect(self._set_wb_temperature)
        controller.magnification_changed.connect(self.set_magnification)
        controller.show_scale_bar_changed.connect(self.set_show_scale_bar)
        controller.color_changed.connect(self.set_color)
        controller.measure_mode_changed.connect(self.set_measure_mode)
        controller.crosshair_visible_changed.connect(self.set_crosshair_visible)
        controller.full_crosshair_changed.connect(self.set_full_crosshair)
        controller.hud_changed.connect(self.set_hud)
        controller.native_zoom_toggled.connect(self.set_native_zoom)
        controller.binning_changed.connect(self._on_binning_changed)
        controller.zoom_under_cursor_changed.connect(self.set_zoom_under_cursor)

        self.scale_bar_color = "White"
        self.show_scale_bar = True
        self.magnification = "5x"
        self.native_zoom = False
        self._binning_factor = 1
        self.zoom_under_cursor = False

    def track_mouse(self, event: QMouseEvent):
        self.last_mouse_pos = event.pos()
        
    def get_frame(self):
        """
        Return the latest displayed frame as a numpy array (BGR).
        Falls back to grabbing a fresh frame if needed.
        """
        try:
            if hasattr(self, "last_output_frame") and self.last_output_frame is not None:
                import numpy as np
                return self.last_output_frame.copy()
            # Fallback: grab directly from camera
            if self.cap is not None and self.cap.connected():
                ok, frame = self.cap.read()
                return frame if ok else None
        except Exception:
            pass
        return None

    def get_scale_bar_pixels(self, magnification):
        """Return (bar_px, label_str, ppm) or None if uncalibrated.

        ppm is adjusted for the current binning factor: a 2× binned pixel
        covers 2× the physical area, so there are half as many pixels per µm.
        """
        ppm = self.calibration_table.get(magnification)
        if ppm is None:
            if magnification not in PreviewWindow._warned_calibrations:
                print(f"[WARN] No calibration for {magnification} — set ppm in calibration_table")
                PreviewWindow._warned_calibrations.add(magnification)
            return None

        ppm = ppm / self._binning_factor
        bar_um = {"5x": 200, "10x": 100, "20x": 50, "50x": 20, "100x": 5}.get(magnification, 100)
        return int(ppm * bar_um), f"{bar_um} um", ppm

    def _on_exposure_changed(self, v):
        self.cap.set_exposure_us(v)
        self._info_exposure_us = v
        self._hud_cache = []

    def _on_gain_changed(self, v):
        self.cap.set_gain_db(v)
        self._info_gain_db = v
        self._hud_cache = []

    def set_magnification(self, mag):
        self.magnification = mag
        self._hud_cache = []

    def set_show_scale_bar(self, show):
        self.show_scale_bar = show

    def set_color(self, color):
        self.scale_bar_color = color

    def set_measure_mode(self, mode):
        self.measure_mode = mode
        self._update_zoom_visibility()

    def set_crosshair_visible(self, visible):
        self.show_crosshair = visible
        self._update_zoom_visibility()

    def set_full_crosshair(self, visible):
        self.show_full_crosshair = visible

    def set_hud(self, visible):
        self.show_hud = visible

    def flash_mag(self, mag_str, duration=1.5):
        self._mag_flash_text = mag_str
        self._mag_flash_until = time.time() + duration

    # ------------------------------------------------------------------
    # HUD helpers
    # ------------------------------------------------------------------

    _TICK_INTERVALS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]

    def _draw_full_crosshair(self, draw, disp_w, disp_h, ppm, sx):
        cx, cy = disp_w // 2, disp_h // 2
        line_color = (0, 160, 0)
        tick_color = (0, 255, 0)
        cv2.line(draw, (0, cy), (disp_w - 1, cy), line_color, 1)
        cv2.line(draw, (cx, 0), (cx, disp_h - 1), line_color, 1)

        if ppm is None or sx is None or sx == 0:
            return

        ppm_disp = ppm / sx           # pixels per µm in display space
        half_w_um = cx / ppm_disp
        interval_um = next(
            (v for v in self._TICK_INTERVALS if half_w_um / v <= 10),
            self._TICK_INTERVALS[-1],
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, ft = 0.35, 1

        def _tick(px, py, horiz, major):
            arm = 12 if major else 6
            if horiz:
                cv2.line(draw, (px, cy - arm), (px, cy + arm), tick_color, 1)
                if major:
                    um = abs(int(round((px - cx) / ppm_disp)))
                    lbl = f"{um}um"
                    (tw, th), _ = cv2.getTextSize(lbl, font, fs, ft)
                    cv2.putText(draw, lbl, (px - tw // 2, cy + arm + th + 2),
                                font, fs, tick_color, ft)
            else:
                cv2.line(draw, (cx - arm, py), (cx + arm, py), tick_color, 1)
                if major:
                    um = abs(int(round((py - cy) / ppm_disp)))
                    lbl = f"{um}um"
                    (tw, th), _ = cv2.getTextSize(lbl, font, fs, ft)
                    cv2.putText(draw, lbl, (cx + arm + 3, py + th // 2),
                                font, fs, tick_color, ft)

        n_h = int(half_w_um / interval_um) + 1
        for i in range(-n_h, n_h + 1):
            if i == 0:
                continue
            px = cx + int(i * interval_um * ppm_disp)
            if 0 <= px < disp_w:
                _tick(px, cy, horiz=True, major=(abs(i) % 5 == 0))

        half_h_um = cy / ppm_disp
        n_v = int(half_h_um / interval_um) + 1
        for i in range(-n_v, n_v + 1):
            if i == 0:
                continue
            py = cy + int(i * interval_um * ppm_disp)
            if 0 <= py < disp_h:
                _tick(cx, py, horiz=False, major=(abs(i) % 5 == 0))

    def _draw_hud(self, draw):
        # Refresh cached lines every 10 frames (~3 Hz at 30 fps).
        # Imaging params are cheap; motor queries need the serial bus so we
        # throttle them at the same rate.
        self._hud_tick += 1
        if self._hud_tick % 10 == 1 or not self._hud_cache:
            # --- Exposure: read actual value from camera when auto-exp is on ---
            exp_us = self._info_exposure_us
            if self._info_auto_exp:
                try:
                    ae = self.cap.get_exposure_us()
                    if ae and ae > 0:
                        exp_us = ae
                except Exception:
                    pass
            exp_ms = exp_us / 1000.0
            exp_str = (f"{exp_ms:.2f} ms  AUTO" if self._info_auto_exp
                       else f"{exp_ms:.3f} ms")

            lines = [
                f"Mag: {self.magnification}   Bin: {self._info_binning}x",
                f"Exp: {exp_str}",
                f"Gain: {self._info_gain_db:.1f} dB   WB: {self._info_wb_kelvin} K",
                f"FPS: {self.measured_fps:.1f}",
            ]

            mm = self.motor_manager
            if mm:
                for axis, fmt, unit in [
                    ("X", ".3f", "mm"), ("Y", ".3f", "mm"), ("Z", ".4f", "mm"),
                ]:
                    try:
                        v = mm.get_position_units(axis)
                        lines.append(f"{axis}: {v:{fmt}} {unit}" if v is not None
                                     else f"{axis}: N/A")
                    except Exception:
                        lines.append(f"{axis}: err")

            if self.manip_controls is not None:
                ht = self.manip_controls.height_display.text()
                lines.append(ht)

            self._hud_cache = lines

        font, fs, ft = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        lh, pad = 18, 8
        max_w = max(cv2.getTextSize(l, font, fs, ft)[0][0] for l in self._hud_cache)
        bw = max_w + 2 * pad
        bh = len(self._hud_cache) * lh + 2 * pad
        x0, y0 = 8, 8
        roi = draw[y0:y0 + bh, x0:x0 + bw]
        dark = np.zeros_like(roi)
        cv2.addWeighted(dark, 0.55, roi, 0.45, 0, roi)
        draw[y0:y0 + bh, x0:x0 + bw] = roi
        for i, line in enumerate(self._hud_cache):
            cv2.putText(draw, line, (x0 + pad, y0 + pad + (i + 1) * lh - 2),
                        font, fs, (255, 255, 255), ft)

    def _update_zoom_visibility(self):
        if self.show_crosshair or self.zoom_under_cursor or (self.measure_mode and len(self.measure_points) < 2):
            self.zoom_window.show()
        else:
            self.zoom_window.hide()

    def set_zoom_under_cursor(self, enabled: bool):
        self.zoom_under_cursor = enabled
        self._update_zoom_visibility()

    def set_native_zoom(self, flag):
        self.native_zoom = flag
        if flag:
            # Snap the window to exact 1:1 pixel size; user can resize after.
            self.image_label.resize(self.native_width, self.native_height)
            self.adjustSize()

    def _on_binning_changed(self, factor: int):
        self.cap.set_live_binning(factor)
        # camera_manager always delivers full-FOV output at the binned resolution
        # (hardware binning or software downsample), so each output pixel always
        # covers factor×factor sensor pixels → ppm divides by factor.
        self._binning_factor = factor
        self._info_binning   = factor
        self.native_width  = self.cap.native_width
        self.native_height = self.cap.native_height
        self._hud_cache = []
        # Do not resize the label — it is freely resizable by the user.
        # update_frame always scales the camera frame to fill whatever size
        # the window currently is, so scale-bar and double-click stay correct.

    def _set_auto_exposure(self, enabled):
        self.cap.set_auto_exposure(enabled)
        self._info_auto_exp = enabled
        self._hud_cache = []

    def _set_wb_temperature(self, kelvin):
        if hasattr(self.cap, 'set_white_balance_kelvin'):
            self.cap.set_white_balance_kelvin(kelvin)
        self._info_wb_kelvin = kelvin
        self._hud_cache = []

    def start_color_pick(self):
        """Activate eyedropper: next click samples the colour at that position."""
        self._pick_mode = True
        self.image_label.setCursor(Qt.CrossCursor)

    def handle_click(self, event: QMouseEvent):
        if self._pick_mode:
            self._pick_mode = False
            self.image_label.setCursor(Qt.ArrowCursor)
            if self._last_raw_frame is not None:
                x, y = event.pos().x(), event.pos().y()
                ox, oy = self.display_offset
                sx, sy = self.display_scale
                fx = int(max(0, min((x - ox) * sx, self._last_raw_frame.shape[1] - 1)))
                fy = int(max(0, min((y - oy) * sy, self._last_raw_frame.shape[0] - 1)))
                # Average a 5×5 neighbourhood for noise robustness
                h, w = self._last_raw_frame.shape[:2]
                x1, x2 = max(0, fx - 2), min(w, fx + 3)
                y1, y2 = max(0, fy - 2), min(h, fy + 3)
                patch = self._last_raw_frame[y1:y2, x1:x2]   # BGR
                b, g, r = [int(v) for v in patch.reshape(-1, 3).mean(axis=0)]
                self.color_sampled.emit(r, g, b)
            return
        if not self.measure_mode:
            return
        x, y = event.pos().x(), event.pos().y()
        ox, oy = self.display_offset
        sx, sy = self.display_scale
        if not (ox <= x <= self.image_label.width() - ox and oy <= y <= self.image_label.height() - oy):
            return
        cx, cy = (x - ox) * sx, (y - oy) * sy
        if len(self.measure_points) < 2:
            self.measure_points.append((cx, cy))
        else:
            self.measure_points.clear()

    def handle_double_click(self, event: QMouseEvent):
        """Move stage so the double-clicked feature is centred in the frame."""
        if self.motor_manager is None or event.button() != Qt.LeftButton:
            return
        px, py = event.pos().x(), event.pos().y()
        ox, oy = self.display_offset
        sx, sy = self.display_scale

        # Ignore clicks outside the image area
        lw, lh = self.image_label.width(), self.image_label.height()
        if not (ox <= px < lw - ox and oy <= py < lh - oy):
            return

        # Offset from frame centre in native pixels
        dpx = (px - ox) * sx - self.native_width  / 2.0
        dpy = (py - oy) * sy - self.native_height / 2.0

        result = self.get_scale_bar_pixels(self.magnification)
        if result is None:
            return
        _, _, ppm = result   # pixels per µm

        dx_mm =  dpx / ppm / 1000.0
        dy_mm = -dpy / ppm / 1000.0   # image +Y is down; stage +Y is up

        self.motor_manager.move_relative_xy_units(dx_mm, dy_mm, wait=False)

        if self.stage_controls is not None:
            sc = self.stage_controls
            cfg = getattr(self.motor_manager, 'step_config', {})
            step_x = cfg.get('X', {}).get('step', 0.001)
            step_y = cfg.get('Y', {}).get('step', 0.001)
            sc.stage_x_slider.blockSignals(True)
            sc.stage_y_slider.blockSignals(True)
            sc.stage_x_slider.setValue(sc.stage_x_slider.value() + round(dx_mm / step_x))
            sc.stage_y_slider.setValue(sc.stage_y_slider.value() + round(dy_mm / step_y))
            sc.stage_x_slider.blockSignals(False)
            sc.stage_y_slider.blockSignals(False)

    def _on_zoom_double_clicked(self, px, py):
        """Move stage so the double-clicked point in the zoom window is centred."""
        if self.motor_manager is None:
            return
        result = self.get_scale_bar_pixels(self.magnification)
        if result is None:
            return
        _, _, ppm = result
        # Zoom window is 500×500 showing a ±20 px crop of the original frame.
        # Each display pixel = 40/500 = 0.08 original frame pixels.
        zs = 20
        frame_scale = 500.0 / (2 * zs)
        dpx = (px - 250) / frame_scale   # offset from zoom centre in frame px
        dpy = (py - 250) / frame_scale
        dx_mm =  dpx / ppm / 1000.0
        dy_mm = -dpy / ppm / 1000.0
        self.motor_manager.move_relative_xy_units(dx_mm, dy_mm, wait=False)

    def set_temporal_average(self, n):
        """Set rolling-average depth. n=1 disables averaging."""
        self._avg_n = max(1, int(n))
        self._avg_buf.clear()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Temporal averaging: average last N raw frames to suppress
        # AC-lighting flicker and reduce camera noise.
        if self._avg_n > 1:
            self._avg_buf.append(frame.astype(np.float32))
            while len(self._avg_buf) > self._avg_n:
                self._avg_buf.popleft()
            frame = np.mean(self._avg_buf, axis=0).astype(np.uint8)

        orig_h, orig_w = frame.shape[:2]
        lw, lh = self.image_label.width(), self.image_label.height()

        if self.flat_field_panel is not None:
            frame = self.flat_field_panel.apply_correction(frame)

        if self.native_zoom:
            new_w, new_h = orig_w, orig_h
            ox, oy, sx, sy = 0, 0, 1, 1
            display = frame.copy()
        else:
            aspect = orig_w / orig_h
            if lw / lh > aspect:
                new_h, new_w = lh, int(lh * aspect)
            else:
                new_w, new_h = lw, int(lw / aspect)
            display = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            ox, oy = (lw - new_w)//2, (lh - new_h)//2
            sx, sy = orig_w / new_w, orig_h / new_h
            canvas = np.zeros((lh, lw, 3), dtype=np.uint8)
            canvas[oy:oy+new_h, ox:ox+new_w] = display
            display = canvas

        self.display_offset = (ox, oy)
        self.display_scale = (sx, sy)
        self._last_raw_frame = frame

        if self.layer_panel is not None:
            display = self.layer_panel.apply_overlay(display)
        if self.mark_panel is not None:
            display = self.mark_panel.apply_overlay(display)

        draw = display.copy()
        disp_h, disp_w = draw.shape[:2]

        sb = self.get_scale_bar_pixels(self.magnification)
        if sb and self.show_scale_bar:
            bar_px, label, ppm = sb
            try:
                disp_bar = int(bar_px / sx)
            except:
                disp_bar = bar_px
            offset = 20
            bar_h = 4
            color = (255,255,255) if self.scale_bar_color=='White' else (0,0,0)
            x1 = max(0, disp_w - disp_bar - offset)
            y1 = disp_h - offset
            x2 = disp_w - offset
            y2 = disp_h - offset
            cv2.rectangle(draw, (x1, y1-bar_h), (x2, y2), color, -1)
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(draw, label, (x1 + (disp_bar-tw)//2, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Crosshair — full-screen with ticks, or small marker
        cx_disp = disp_w // 2
        cy_disp = disp_h // 2
        if self.show_full_crosshair:
            sb = self.get_scale_bar_pixels(self.magnification)
            _ppm = sb[2] if sb else None
            self._draw_full_crosshair(draw, disp_w, disp_h, _ppm, sx)
        else:
            cv2.drawMarker(draw, (cx_disp, cy_disp), (0, 255, 0),
                           markerType=cv2.MARKER_CROSS, markerSize=20, thickness=1)

        if self.show_hud:
            self._draw_hud(draw)

        # Mag flash (triggered by gamepad X button)
        if self._mag_flash_text and time.time() < self._mag_flash_until:
            font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 3.0, 6
            (tw, th), bl = cv2.getTextSize(self._mag_flash_text, font, scale, thick)
            tx = (disp_w - tw) // 2
            ty = (disp_h + th) // 2
            cv2.putText(draw, self._mag_flash_text, (tx + 2, ty + 2),
                        font, scale, (0, 0, 0), thick + 2)
            cv2.putText(draw, self._mag_flash_text, (tx, ty),
                        font, scale, (0, 255, 0), thick)


        for pt in self.measure_points:
            dx = int(pt[0]/sx + ox)
            dy = int(pt[1]/sy + oy)
            cv2.drawMarker(draw, (dx, dy), (0,255,255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)

        if len(self.measure_points) == 2:
            p1, p2 = self.measure_points
            dx_n = p2[0] - p1[0]
            dy_n = p2[1] - p1[1]
            dist_px = (dx_n**2 + dy_n**2)**0.5
            mid_x = int((p1[0]+p2[0])/(2*sx) + ox)
            mid_y = int((p1[1]+p2[1])/(2*sy) + oy)
            pt1_disp = (int(p1[0]/sx + ox), int(p1[1]/sy + oy))
            pt2_disp = (int(p2[0]/sx + ox), int(p2[1]/sy + oy))
            cv2.line(draw, pt1_disp, pt2_disp, (0,255,255), 1)
            angle = np.degrees(np.arctan2(dy_n, dx_n))
            # ±1 px positional error → angle uncertainty via error propagation
            angle_err = np.degrees(np.arctan2(1, dist_px)) if dist_px != 0 else 0
            sb = self.get_scale_bar_pixels(self.magnification)
            if sb:
                dist_um = dist_px / sb[2]
                # +/-1 px positional error -> distance uncertainty in um
                dist_err_um = 1.0 / sb[2]
                dist_label = f"{dist_px:.1f} +/- 1 px  /  {dist_um:.1f} +/- {dist_err_um:.1f} um"
            else:
                dist_label = f"{dist_px:.1f} +/- 1 px  (uncalibrated)"
            cv2.putText(draw, dist_label, (mid_x, mid_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(draw, f"{angle:.2f} +/- {angle_err:.2f} deg", (mid_x, mid_y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        self.last_output_frame = draw.copy()
        rgb = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, draw.shape[1], draw.shape[0], 3*draw.shape[1], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))

        center_x, center_y = orig_w//2, orig_h//2
        if self.zoom_under_cursor or (self.measure_mode and len(self.measure_points) < 2):
            mx, my = self.last_mouse_pos.x(), self.last_mouse_pos.y()
            try:
                center_x = int((mx - ox) * sx)
                center_y = int((my - oy) * sy)
            except:
                pass
        self._zoom_center_frame = (center_x, center_y)
        zs = 20
        x1 = max(center_x - zs, 0)
        x2 = min(center_x + zs, orig_w)
        y1 = max(center_y - zs, 0)
        y2 = min(center_y + zs, orig_h)
        zoom_crop = frame[y1:y2, x1:x2]

        if zoom_crop.size == 0 or x2 <= x1 or y2 <= y1:
            return

        zoom_resized = cv2.resize(zoom_crop, (500, 500), interpolation=cv2.INTER_NEAREST)

        # Draw scale bar on zoom window
        sb = self.get_scale_bar_pixels(self.magnification)
        if sb and self.show_scale_bar:
            _, _, ppm = sb
            zoom_display_scale = 500.0 / (2 * zs)   # display px per frame px
            ppm_zoom = ppm * zoom_display_scale       # display px per µm
            # Largest nice bar that fits in ≤200 display px
            _nice = [1, 2, 5, 10, 20, 50, 100, 200]
            bar_um = next((u for u in _nice if ppm_zoom * u >= 30), _nice[0])
            bar_um = next((u for u in reversed(_nice) if ppm_zoom * u <= 200), bar_um)
            bar_px_z = int(ppm_zoom * bar_um)
            bar_color = (255, 255, 255) if self.scale_bar_color == 'White' else (0, 0, 0)
            bx1, by = 10, 488
            cv2.rectangle(zoom_resized, (bx1, by - 4), (bx1 + bar_px_z, by), bar_color, -1)
            cv2.putText(zoom_resized, f"{bar_um} um",
                        (bx1, by - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, bar_color, 1)

        if self.show_crosshair or self.zoom_under_cursor or (self.measure_mode and len(self.measure_points) < 2):
            self.zoom_window.update_image(zoom_resized)
            self.zoom_window.show()
        else:
            self.zoom_window.hide()

        self.frame_count += 1
        elapsed = time.time() - self.last_time
        if elapsed >= 1.0:
            self.measured_fps = self.frame_count / elapsed
            self.last_time = time.time()
            self.frame_count = 0

        # Update Laplacian bar (log10 scale: 1→0%, 5→~44%, 40→100%)
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            bar_val = int(np.log10(max(1.0, lap)) / 1.6 * 1000)
            self._lap_bar.setValue(min(1000, bar_val))
            self._lap_label.setText(f"Focus: {lap:.0f}")
            color = "#dddddd" if lap > 20 else "#ee7733" if lap > 5 else "#0077bb"
            self._lap_bar.setStyleSheet(
                f"QProgressBar::chunk {{ background-color: {color}; }}")
        except Exception:
            pass
            
    def get_clean_frame(self):
        """Return the latest frame without any UI annotations (scale bar, crosshair, etc).
        Flat-field correction is applied but overlays are not.  Full sensor resolution."""
        frame = getattr(self, '_last_raw_frame', None)
        if frame is not None:
            return frame.copy()
        # Fallback: grab directly from camera
        if self.cap is not None and self.cap.connected():
            ok, frame = self.cap.read()
            return frame if ok else None
        return None

    def get_latest_frame(self):
        # Returns a copy of the last output frame, or None if not yet available
        return getattr(self, 'last_output_frame', None)

    def closeEvent(self, event):
        self.cap.close()
        event.accept()

class ZoomWindow(QWidget):
    # Emits (px, py) in the 500×500 zoom display when the user double-clicks
    double_clicked = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zoom View")
        self.setFixedSize(500, 500)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Tool)

        self.label = QLabel(self)
        self.label.setGeometry(0, 0, 500, 500)
        self.label.setScaledContents(True)
        self.label.mouseDoubleClickEvent = self._on_double_click

    def _on_double_click(self, event):
        if event.button() == Qt.LeftButton:
            self.double_clicked.emit(event.pos().x(), event.pos().y())

    def update_image(self, zoom_img):
        cv2.drawMarker(zoom_img, (zoom_img.shape[1]//2, zoom_img.shape[0]//2), (0, 0, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
        rgb = cv2.cvtColor(zoom_img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3*rgb.shape[1], QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimg))

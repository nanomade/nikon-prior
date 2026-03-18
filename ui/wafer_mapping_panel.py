# ui/wafer_mapping_panel.py

import cv2
import numpy as np
import time
import os
import platform
import subprocess
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QTextEdit, QDoubleSpinBox,
    QCheckBox, QProgressBar, QMessageBox,
)
from PyQt5.QtCore import pyqtSignal, QThread, QTimer


class WaferScanWorker(QThread):
    progress      = pyqtSignal(str, int)
    finished      = pyqtSignal(dict)
    image_captured = pyqtSignal(float, float, object)   # x, y, point_data

    def __init__(self, motor_manager, preview, wafer_boundaries,
                 step_x=0.5, step_y=0.5, settling_time=0.5,
                 autofocus_panel=None, focus_map_panel=None, safe_zone_mm=0.5):
        super().__init__()
        self.motor_manager   = motor_manager
        self.preview         = preview
        self.wafer_boundaries = wafer_boundaries
        self.step_x          = step_x
        self.step_y          = step_y
        self.settling_time   = settling_time
        self.autofocus_panel = autofocus_panel
        self.focus_map_panel = focus_map_panel
        self.safe_zone_mm    = safe_zone_mm
        self.should_stop     = False
        self.scan_data       = []
        self.scan_folder     = None

    def stop(self):
        self.should_stop = True

    # ── Stage helpers ──────────────────────────────────────────────────────

    def wait_for_position(self, target_x, target_y, tolerance=0.01, max_wait=10.0):
        t0 = time.time()
        while time.time() - t0 < max_wait:
            if self.should_stop:
                return False
            cx = self.motor_manager.get_position_units('X')
            cy = self.motor_manager.get_position_units('Y')
            if abs(cx - target_x) <= tolerance and abs(cy - target_y) <= tolerance:
                return True
            rem = max_wait - (time.time() - t0)
            self.progress.emit(
                f"Positioning: ΔX={abs(cx-target_x):.3f} ΔY={abs(cy-target_y):.3f} ({rem:.1f}s)", 0)
            time.sleep(0.1)
        return False

    # ── Autofocus helpers ──────────────────────────────────────────────────

    @staticmethod
    def _focus_metric(frame, metric):
        """Compute a focus metric on the full frame (mirrors FocusWorker._metric)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if metric == 'LaplacianVariance':
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        elif metric == 'Tenengrad':
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            return float((gx**2 + gy**2).mean())
        elif metric == 'Brightness':
            return float(gray.mean())
        else:
            return float(gray.var())

    def _run_autofocus_at_point(self):
        """Two-pass autofocus: coarse brightness sweep then fine Laplacian sweep.
        Restores Z if no meaningful improvement. Returns True on success."""
        from ui.autofocus_panel import FocusWorker

        frame = self.preview.get_frame()
        if frame is None:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if float(np.mean(gray)) < 15.0:
            self.progress.emit("Autofocus skipped: dark frame", 0)
            return False

        ap = self.autofocus_panel
        z_cfg     = self.motor_manager.step_config.get('Z', {})
        z_step_mm = z_cfg.get('step', 0.001)
        z_invert  = z_cfg.get('invert', 1)

        pos = self.motor_manager.get_position('Z')
        hw  = getattr(pos, 'Position', 0) if pos is not None else 0
        z0  = hw * z_invert * z_step_mm

        step_mm     = ap.step.value()
        settle_ms   = int(ap.settle.value())
        min_improve = ap.min_improve.value()
        fine_metric = ap.metric.currentText()

        m0 = self._focus_metric(frame, fine_metric)

        def _run_worker(metric, z_min, z_max, step):
            p = {
                'metric':        metric,
                'step_mm':       step,
                'settle_ms':     settle_ms,
                'min_improve':   min_improve,
                'coarse_factor': int(ap.coarse_factor.value()),
                'fine_factor':   ap.fine_factor.value(),
                'n_avg':         int(ap.n_avg.value()),
                'z_min_mm':      z_min,
                'z_max_mm':      z_max,
                'z_step_mm':     z_step_mm,
                'z_invert':      z_invert,
            }
            res = {'z': None, 'metric': None, 'aborted': None}
            w = FocusWorker(self.motor_manager, self.preview.get_frame, p)
            w.finished.connect(lambda z, m: res.update({'z': z, 'metric': m}))
            w.aborted.connect(lambda r: res.update({'aborted': r}))
            w.run()
            return res

        def _restore_z():
            self.motor_manager.move_absolute('Z', int(round(z0 / z_step_mm)))

        # Pass 1 — coarse brightness sweep over wide range
        coarse_step = step_mm * 2
        span = coarse_step * 25
        self.progress.emit("Autofocus pass 1: brightness sweep…", 0)
        res1 = _run_worker('Brightness', z0 - span, z0 + span, coarse_step)
        if res1['aborted']:
            self.progress.emit(f"Autofocus pass 1 failed: {res1['aborted']} — restoring Z", 0)
            _restore_z()
            return False

        z_coarse = res1['z'] if res1['z'] is not None else z0

        # Pass 2 — fine sweep around coarse result
        fine_span = step_mm * 10
        self.progress.emit(f"Autofocus pass 2: {fine_metric} fine sweep…", 0)
        res2 = _run_worker(fine_metric, z_coarse - fine_span, z_coarse + fine_span, step_mm)
        if res2['aborted']:
            self.progress.emit(f"Autofocus pass 2 failed: {res2['aborted']} — restoring Z", 0)
            _restore_z()
            return False

        m_final = res2['metric'] if res2['metric'] is not None else 0.0
        if m_final < m0 * (1.0 + min_improve / 100.0):
            self.progress.emit("Autofocus: no meaningful improvement — restoring Z", 0)
            _restore_z()
            return False

        self.progress.emit(
            f"Autofocus done: z={res2['z']:.4f} mm  {fine_metric}={m_final:.1f}", 0)
        return True

    # ── Focus dispatch ─────────────────────────────────────────────────────

    def _apply_focus_at_point(self, x_pos, y_pos):
        """Apply the best available focus strategy at the current XY position.

        Priority:
          1. Focus map (if available and has data) — instant Z move.
          2. Autofocus refinement (if enabled) — runs after focus map move,
             or standalone if no map data.
        """
        z_cfg     = self.motor_manager.step_config.get('Z', {})
        z_step_mm = z_cfg.get('step', 0.001)

        used_map = False
        if self.focus_map_panel is not None:
            z_pred = self.focus_map_panel.get_focus_z(x_pos, y_pos)
            if z_pred is not None:
                self.motor_manager.move_absolute(
                    'Z', int(round(z_pred / z_step_mm)))
                time.sleep(0.05)   # brief settle after Z move
                used_map = True
                self.progress.emit(
                    f"Focus map Z={z_pred:.4f} mm at "
                    f"({x_pos:.2f}, {y_pos:.2f})", 0)
            else:
                self.progress.emit(
                    f"Focus map: no data at ({x_pos:.2f}, {y_pos:.2f})"
                    " — falling back to autofocus", 0)

        # Run autofocus: either as a refinement pass after the map move,
        # or as the primary focus source if no map data was available.
        if self.autofocus_panel is not None:
            self._run_autofocus_at_point()

    # ── Main scan ──────────────────────────────────────────────────────────

    def run(self):
        try:
            x_min, x_max, y_min, y_max = self.wafer_boundaries

            # Apply safe-zone inset
            sz   = self.safe_zone_mm
            sx_lo = min(x_min, x_max) + sz
            sx_hi = max(x_min, x_max) - sz
            sy_lo = min(y_min, y_max) + sz
            sy_hi = max(y_min, y_max) - sz

            if sx_lo >= sx_hi or sy_lo >= sy_hi:
                self.finished.emit({'status': 'error',
                                    'error': f"Safe zone ({sz:.2f} mm) is larger than the wafer "
                                             f"in at least one axis — reduce it and retry."})
                return

            x_points = np.arange(sx_lo, sx_hi, self.step_x)
            y_points = np.arange(sy_lo, sy_hi, self.step_y)

            if len(x_points) == 0 or len(y_points) == 0:
                self.finished.emit({'status': 'error',
                                    'error': f"Scan grid is empty "
                                             f"(X: {sx_lo:.3f}→{sx_hi:.3f}, "
                                             f"Y: {sy_lo:.3f}→{sy_hi:.3f})"})
                return

            # Create output folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.scan_folder = f"wafer_scan_{timestamp}"
            os.makedirs(self.scan_folder, exist_ok=True)

            total_points  = len(x_points) * len(y_points)
            current_point = 0
            self.scan_data = []

            self.progress.emit(
                f"Scanning: {len(x_points)}×{len(y_points)} = {total_points} points", 5)
            self.progress.emit(f"Output: {self.scan_folder}", 6)

            # Move to first point and stabilise
            self.motor_manager.move_absolute_units('X', x_points[0])
            self.motor_manager.move_absolute_units('Y', y_points[0])
            if not self.wait_for_position(x_points[0], y_points[0]):
                self.progress.emit("Warning: start position not exact, continuing…", 10)

            t0 = time.time()
            while time.time() - t0 < 2.0:
                if self.should_stop:
                    return
                time.sleep(0.1)

            # Serpentine scan
            for y_idx, y_pos in enumerate(y_points):
                if self.should_stop:
                    break
                x_range = x_points if y_idx % 2 == 0 else x_points[::-1]

                for x_pos in x_range:
                    if self.should_stop:
                        break

                    current_point += 1
                    pct = 10 + int((current_point / total_points) * 85)

                    # Move
                    self.motor_manager.move_absolute_units('X', x_pos)
                    self.motor_manager.move_absolute_units('Y', y_pos)
                    if not self.wait_for_position(x_pos, y_pos):
                        self.progress.emit(
                            f"Position not exact at ({x_pos:.2f}, {y_pos:.2f}), continuing…", pct)

                    # Settle
                    t_settle = time.time()
                    while time.time() - t_settle < self.settling_time:
                        if self.should_stop:
                            break
                        time.sleep(0.05)

                    if self.should_stop:
                        break

                    # Focus (map and/or autofocus)
                    if self.focus_map_panel is not None or self.autofocus_panel is not None:
                        self.progress.emit(
                            f"Focusing at ({x_pos:.2f}, {y_pos:.2f})…", pct)
                        self._apply_focus_at_point(x_pos, y_pos)

                    # Capture and save
                    frame = self.preview.get_frame()
                    if frame is not None:
                        filename = f"img_X{x_pos:+.3f}_Y{y_pos:+.3f}.png"
                        cv2.imwrite(os.path.join(self.scan_folder, filename), frame)
                        point_data = {
                            'x': x_pos, 'y': y_pos,
                            'filename': filename,
                            'timestamp': datetime.now().isoformat(),
                        }
                        self.scan_data.append(point_data)
                        self.image_captured.emit(x_pos, y_pos, point_data)

                    self.progress.emit(
                        f"Captured {current_point}/{total_points}", pct)

            # Return to centre
            if not self.should_stop:
                self.progress.emit("Returning to centre…", 97)
                self.motor_manager.move_absolute_units('X', 0)
                self.motor_manager.move_absolute_units('Y', 0)

            status = 'stopped' if self.should_stop else 'success'
            self.progress.emit(
                "Scan complete." if status == 'success' else "Scan stopped.", 100)
            self.finished.emit({
                'status':         status,
                'scan_folder':    self.scan_folder,
                'total_points':   total_points,
                'scanned_points': len(self.scan_data),
            })

        except Exception as e:
            self.finished.emit({'status': 'error', 'error': str(e)})


# ── Panel ──────────────────────────────────────────────────────────────────

class WaferMappingPanel(QWidget):
    def __init__(self, preview_window, motor_manager, stage_controls,
                 autofocus_panel=None, focus_map_panel=None):
        super().__init__()
        self.preview         = preview_window
        self.motor_manager   = motor_manager
        self.stage_controls  = stage_controls
        self.autofocus_panel = autofocus_panel
        self.focus_map_panel = focus_map_panel
        self.worker          = None
        self.scan_data      = []
        self.wafer_boundaries = None
        self.total_scan_points = 0
        self.current_scan_folder = None
        self.scan_start_time = None

        self.setWindowTitle("Wafer Scan")
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("<b>Wafer Scan</b>"))

        self.status_label = QLabel(
            "Waiting for wafer extents — run Find Wafer Extents first.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # ── Parameters ────────────────────────────────────────────────────
        params_group = QGroupBox("Scan Parameters")
        params_layout = QVBoxLayout()

        def _dspin(lo, hi, val, dec, step, suffix=""):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setDecimals(dec)
            s.setSingleStep(step)
            if suffix:
                s.setSuffix(suffix)
            return s

        def _row(label, widget):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(widget)
            row.addStretch()
            params_layout.addLayout(row)

        self.step_x_spin = _dspin(0.01, 10.0, 0.5, 3, 0.05, " mm")
        self.step_y_spin = _dspin(0.01, 10.0, 0.5, 3, 0.05, " mm")
        _row("Step X:", self.step_x_spin)
        _row("Step Y:", self.step_y_spin)

        # FOV calculator
        fov_row = QHBoxLayout()
        calc_btn = QPushButton("Set step = FOV")
        calc_btn.setToolTip(
            "Set Step X/Y to the full field of view at the current\n"
            "magnification and image resolution (no overlap).\n"
            "Reduce step manually if you need tile overlap for stitching.")
        calc_btn.clicked.connect(self._calculate_step_from_fov)
        fov_row.addWidget(calc_btn)
        fov_row.addStretch()
        params_layout.addLayout(fov_row)
        self.fov_label = QLabel("")
        self.fov_label.setWordWrap(True)
        params_layout.addWidget(self.fov_label)

        self.settling_spin = _dspin(0.1, 5.0, 0.5, 1, 0.1, " s")
        self.settling_spin.setToolTip(
            "Time to wait after the stage reaches each position\n"
            "before capturing the frame.")
        _row("Settling time:", self.settling_spin)

        self.tolerance_spin = _dspin(0.001, 0.1, 0.01, 3, 0.005, " mm")
        self.tolerance_spin.setToolTip(
            "Maximum position error before capture proceeds.")
        _row("Position tolerance:", self.tolerance_spin)

        self.safe_zone_spin = _dspin(0.0, 5.0, 0.5, 2, 0.1, " mm")
        self.safe_zone_spin.setToolTip(
            "Inset from each detected wafer edge.\n"
            "The scan grid will not approach within this distance\n"
            "of any boundary, preventing the stage from driving off the wafer.")
        _row("Safe zone:", self.safe_zone_spin)

        focus_map_row = QHBoxLayout()
        self.use_focus_map_check = QCheckBox("Use focus map Z")
        self.use_focus_map_check.setChecked(False)
        self.use_focus_map_check.setEnabled(self.focus_map_panel is not None)
        self.use_focus_map_check.setToolTip(
            "Before each capture, look up the interpolated Z from the Focus Map\n"
            "and move to that position. Requires a Focus Map to have been measured.\n"
            "Much faster than autofocus per-point; can be combined with autofocus\n"
            "refinement below for the best of both.")
        focus_map_row.addWidget(self.use_focus_map_check)
        focus_map_row.addStretch()
        params_layout.addLayout(focus_map_row)

        af_row = QHBoxLayout()
        self.autofocus_check = QCheckBox("Autofocus each point")
        self.autofocus_check.setChecked(False)
        self.autofocus_check.setToolTip(
            "Run autofocus at each scan point.\n"
            "If 'Use focus map Z' is also checked, autofocus runs as a\n"
            "short refinement pass starting from the map-predicted Z.")
        af_row.addWidget(self.autofocus_check)
        af_row.addStretch()
        params_layout.addLayout(af_row)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # ── Buttons ───────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start Scan")
        self.start_btn.clicked.connect(self.start_manual_mapping)
        btn_row.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_mapping)
        self.stop_btn.setEnabled(False)
        btn_row.addWidget(self.stop_btn)

        self.center_btn = QPushButton("Go to Centre")
        self.center_btn.clicked.connect(self.goto_center)
        btn_row.addWidget(self.center_btn)
        layout.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # ── Summary ───────────────────────────────────────────────────────
        summary_group = QGroupBox("Scan Summary")
        summary_layout = QVBoxLayout()

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(140)
        summary_layout.addWidget(self.results_text)

        stats_row = QHBoxLayout()
        self.stats_label = QLabel("Images: 0")
        stats_row.addWidget(self.stats_label)
        self.time_label = QLabel("Time: --:--")
        stats_row.addWidget(self.time_label)
        summary_layout.addLayout(stats_row)

        self.folder_label = QLabel("Scan folder: —")
        self.folder_label.setWordWrap(True)
        summary_layout.addWidget(self.folder_label)

        self.open_folder_btn = QPushButton("Open Scan Folder")
        self.open_folder_btn.clicked.connect(self.open_scan_folder)
        self.open_folder_btn.setEnabled(False)
        summary_layout.addWidget(self.open_folder_btn)

        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)

        layout.addStretch()

        self.scan_timer = QTimer()
        self.scan_timer.timeout.connect(self._update_scan_time)

    # ── FOV calculator ─────────────────────────────────────────────────────

    def _calculate_step_from_fov(self):
        mag = self.preview.magnification
        result = self.preview.get_scale_bar_pixels(mag)
        if result is None:
            self.fov_label.setText(
                f"No calibration for {mag} — enter step manually.")
            return
        _, _, ppm = result      # pixels per µm
        w_px = getattr(self.preview, "native_width",  616)
        h_px = getattr(self.preview, "native_height", 514)
        fov_x_mm = w_px / ppm / 1000.0
        fov_y_mm = h_px / ppm / 1000.0
        self.step_x_spin.setValue(round(fov_x_mm, 3))
        self.step_y_spin.setValue(round(fov_y_mm, 3))
        self.fov_label.setText(
            f"FOV {fov_x_mm:.3f} × {fov_y_mm:.3f} mm  ({mag}, {w_px}×{h_px})")

    # ── Scan control ───────────────────────────────────────────────────────

    def start_auto_mapping_with_boundaries(self, wafer_boundaries):
        if self.worker and self.worker.isRunning():
            return

        self.wafer_boundaries = wafer_boundaries
        x_min, x_max, y_min, y_max = wafer_boundaries

        step_x       = self.step_x_spin.value()
        step_y       = self.step_y_spin.value()
        settling     = self.settling_spin.value()
        safe_zone_mm = self.safe_zone_spin.value()

        # Preview expected grid (with safe zone applied)
        sz   = safe_zone_mm
        sx_lo = min(x_min, x_max) + sz
        sx_hi = max(x_min, x_max) - sz
        sy_lo = min(y_min, y_max) + sz
        sy_hi = max(y_min, y_max) - sz
        expected_cols = len(np.arange(sx_lo, sx_hi, step_x)) if sx_lo < sx_hi else 0
        expected_rows = len(np.arange(sy_lo, sy_hi, step_y)) if sy_lo < sy_hi else 0
        expected_images = expected_cols * expected_rows

        fps = getattr(self.preview, 'measured_fps', 0)
        frame_time = 1.0 / fps if fps > 0 else 0.1
        est_s = expected_images * (settling + 0.3 + frame_time)
        est_m, est_s_rem = int(est_s // 60), int(est_s % 60)

        self.scan_data = []
        self.results_text.clear()
        self.results_text.append("=== WAFER SCAN ===")
        self.results_text.append(
            f"• Boundaries: X[{x_min:.2f}, {x_max:.2f}]  Y[{y_min:.2f}, {y_max:.2f}]")
        self.results_text.append(f"• Safe zone: {safe_zone_mm:.2f} mm per edge")
        self.results_text.append(f"• Step: {step_x:.3f} × {step_y:.3f} mm")
        self.results_text.append(f"• Settling: {settling:.1f} s")
        self.results_text.append(f"• Expected images: {expected_images}")
        self.results_text.append(f"• Estimated time: ~{est_m}:{est_s_rem:02d} min")
        use_focus_map = (self.use_focus_map_check.isChecked()
                         and self.focus_map_panel is not None
                         and self.focus_map_panel.has_map())
        use_autofocus = (self.autofocus_check.isChecked()
                         and self.autofocus_panel is not None)

        focus_mode = []
        if use_focus_map:
            focus_mode.append("focus map")
        if use_autofocus:
            focus_mode.append("autofocus" + (" refinement" if use_focus_map else ""))
        if not focus_mode:
            focus_mode.append("none")

        self.results_text.append(f"• Focus: {' + '.join(focus_mode)}")
        self.results_text.append("• Images saved with X,Y coordinates in filename")

        self.total_scan_points = expected_images

        self.worker = WaferScanWorker(
            motor_manager=self.motor_manager,
            preview=self.preview,
            wafer_boundaries=self.wafer_boundaries,
            step_x=step_x,
            step_y=step_y,
            settling_time=settling,
            autofocus_panel=self.autofocus_panel if use_autofocus else None,
            focus_map_panel=self.focus_map_panel if use_focus_map else None,
            safe_zone_mm=safe_zone_mm,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.image_captured.connect(self._on_image_captured)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.open_folder_btn.setEnabled(False)
        self.scan_start_time = datetime.now()
        self.scan_timer.start(1000)

    def start_manual_mapping(self):
        if not self.wafer_boundaries:
            self.results_text.append(
                "No wafer boundaries — run Find Wafer Extents first.")
            self.status_label.setText("No boundaries — run Find Wafer Extents")
            return
        self.start_auto_mapping_with_boundaries(self.wafer_boundaries)

    def stop_mapping(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1000)

    def goto_center(self):
        self.motor_manager.move_absolute_units('X', 0)
        self.motor_manager.move_absolute_units('Y', 0)
        self.status_label.setText("Moving to centre…")
        QTimer.singleShot(2000, lambda: self.status_label.setText("At wafer centre"))

    # ── Slots ──────────────────────────────────────────────────────────────

    def _on_progress(self, message, value):
        self.status_label.setText(message)
        self.progress_bar.setValue(value)

    def _on_image_captured(self, x, y, point_data):
        self.scan_data.append(point_data)
        self.stats_label.setText(f"Images: {len(self.scan_data)}")

    def _on_finished(self, result):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.scan_timer.stop()

        status = result.get('status', 'error')
        if status == 'success':
            self.status_label.setText("Scan complete.")
            self.results_text.append(
                f"Done. {result.get('scanned_points', 0)} images captured.")
        elif status == 'stopped':
            self.status_label.setText("Scan stopped.")
            self.results_text.append(
                f"Stopped. {result.get('scanned_points', 0)} images captured.")
        else:
            self.status_label.setText("Scan failed.")
            self.results_text.append(f"Error: {result.get('error', 'Unknown')}")

        folder = result.get('scan_folder')
        if folder:
            self.current_scan_folder = folder
            self.folder_label.setText(f"Scan folder: {folder}")
            self.open_folder_btn.setEnabled(True)

    def _update_scan_time(self):
        if self.scan_start_time is None:
            return
        elapsed = (datetime.now() - self.scan_start_time).total_seconds()
        el_m, el_s = int(elapsed // 60), int(elapsed % 60)
        captured = len(self.scan_data)
        total = self.total_scan_points
        if captured > 0 and total > 0 and elapsed > 0:
            remaining = (total - captured) / (captured / elapsed)
            rem_m, rem_s = int(remaining // 60), int(remaining % 60)
            self.time_label.setText(
                f"Elapsed: {el_m:02d}:{el_s:02d}  ETA: {rem_m:02d}:{rem_s:02d}")
        else:
            self.time_label.setText(f"Elapsed: {el_m:02d}:{el_s:02d}")

    def open_scan_folder(self):
        folder = self.current_scan_folder
        if folder and os.path.exists(folder):
            try:
                system = platform.system()
                if system == "Windows":
                    os.startfile(folder)
                elif system == "Darwin":
                    subprocess.run(["open", folder])
                else:
                    subprocess.run(["xdg-open", folder])
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not open folder: {e}")
        else:
            QMessageBox.warning(self, "No Folder", "No scan folder available")

# ui/focus_map_panel.py
#
# The focus map measures autofocus Z at a regular NxM grid of points across the
# wafer surface and builds a 2D RBF (radial-basis-function) interpolant.
# Points with a poor Laplacian metric get high RBF smoothing so they have
# little influence on the surface; well-focused points are treated as exact
# constraints.  The public get_focus_z(x, y) method lets the mosaic scanner
# (or any other caller) look up the predicted focus depth at any XY position.

import json
import threading
import time

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QDoubleSpinBox, QComboBox,
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QGroupBox, QProgressBar, QFileDialog, QMessageBox,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class FocusMapWorker(QThread):
    point_measured         = pyqtSignal(float, float, float, float)  # x, y, z, metric
    point_skipped          = pyqtSignal(float, float)                # x, y
    manual_focus_requested = pyqtSignal(float, float)                # x, y — pauses worker
    progress               = pyqtSignal(str, int)
    finished               = pyqtSignal(dict)

    def __init__(self, motor_manager, preview, autofocus_panel,
                 wafer_boundaries, grid_cols=5, grid_rows=5,
                 min_metric=100.0):
        super().__init__()
        self.motor_manager    = motor_manager
        self.preview          = preview
        self.autofocus_panel  = autofocus_panel
        self.wafer_boundaries = wafer_boundaries
        self.grid_cols        = grid_cols
        self.grid_rows        = grid_rows
        self.min_metric       = min_metric
        self.should_stop      = False
        self._pause_event     = threading.Event()
        self._manual_z        = None

    def stop(self):
        self.should_stop = True
        self._pause_event.set()

    def resume_with_z(self, z):
        self._manual_z = z
        self._pause_event.set()

    def skip_point(self):
        self._manual_z = None
        self._pause_event.set()

    def _autofocus(self):
        """Run FocusWorker synchronously.
        Returns (z_mm, metric) if focus is good, None if failed or below threshold."""
        from ui.autofocus_panel import FocusWorker

        frame = self.preview.get_frame()
        if frame is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if float(np.mean(gray)) < 15.0:
            return None  # off wafer / no illumination

        ap        = self.autofocus_panel
        z_cfg     = self.motor_manager.step_config.get('Z', {})
        z_step_mm = z_cfg.get('step', 0.001)
        z_invert  = z_cfg.get('invert', 1)

        pos = self.motor_manager.get_position('Z')
        hw  = getattr(pos, 'Position', 0) if pos is not None else 0
        z0  = hw * z_invert * z_step_mm

        step_mm = ap.step.value()
        span    = step_mm * 50
        p = {
            'metric':        ap.metric.currentText(),
            'step_mm':       step_mm,
            'settle_ms':     int(ap.settle.value()),
            'min_improve':   ap.min_improve.value(),
            'coarse_factor': int(ap.coarse_factor.value()),
            'fine_factor':   ap.fine_factor.value(),
            'n_avg':         int(ap.n_avg.value()),
            'z_min_mm':      z0 - span,
            'z_max_mm':      z0 + span,
            'z_step_mm':     z_step_mm,
            'z_invert':      z_invert,
        }

        result = {'z': None, 'metric': None, 'aborted': None}
        worker = FocusWorker(self.motor_manager, self.preview.get_frame, p)
        worker.finished.connect(lambda z, m: result.update({'z': z, 'metric': m}))
        worker.aborted.connect(lambda r: result.update({'aborted': r}))
        worker.run()

        if result['aborted']:
            self.motor_manager.move_absolute('Z', int(round(z0 / z_step_mm)))
            return None

        if result['z'] is None:
            return None

        # Reject if metric is below the quality threshold
        if result['metric'] is not None and result['metric'] < self.min_metric:
            return None

        return result['z'], result['metric']

    def _request_manual(self, x_pos, y_pos, pct, reason=""):
        """Pause worker and ask the user to focus manually."""
        self.progress.emit(
            f"({x_pos:.2f}, {y_pos:.2f}) — {reason}waiting for manual focus…", pct)
        self._pause_event.clear()
        self._manual_z = None
        self.manual_focus_requested.emit(x_pos, y_pos)
        self._pause_event.wait()

    def run(self):
        try:
            x_min, x_max, y_min, y_max = self.wafer_boundaries
            x_min, x_max = min(x_min, x_max), max(x_min, x_max)
            y_min, y_max = min(y_min, y_max), max(y_min, y_max)

            # Stay one grid period inset from the edge on all sides so we
            # never land on the wafer rim.  Generate n+2 points over the full
            # span and discard the outermost one on each side.
            x_points = np.linspace(x_min, x_max, self.grid_cols + 2)[1:-1]
            y_points = np.linspace(y_min, y_max, self.grid_rows + 2)[1:-1]
            total = len(x_points) * len(y_points)
            done  = 0

            # Serpentine: reverse X direction on odd rows
            for row_idx, y_pos in enumerate(y_points):
                x_row = x_points if row_idx % 2 == 0 else x_points[::-1]
                for x_pos in x_row:
                    if self.should_stop:
                        break
                    done += 1
                    pct = int(done / total * 100)
                    self.progress.emit(
                        f"Moving to ({x_pos:.2f}, {y_pos:.2f})  [{done}/{total}]", pct)
                    self.motor_manager.move_absolute_units('X', x_pos)
                    self.motor_manager.move_absolute_units('Y', y_pos)
                    time.sleep(0.5)

                    af_result = self._autofocus()
                    if af_result is not None:
                        z_best, metric = af_result
                        self.point_measured.emit(x_pos, y_pos, z_best, metric)
                        self.progress.emit(
                            f"({x_pos:.2f}, {y_pos:.2f}) → Z={z_best:.4f} mm  "
                            f"metric={metric:.0f}", pct)
                    else:
                        self._request_manual(x_pos, y_pos, pct,
                                             reason="poor focus — ")
                        if self.should_stop:
                            break
                        if self._manual_z is not None:
                            self.point_measured.emit(x_pos, y_pos, self._manual_z, 0.0)
                            self.progress.emit(
                                f"({x_pos:.2f}, {y_pos:.2f}) → Z={self._manual_z:.4f} mm"
                                f" (manual)", pct)
                        else:
                            self.point_skipped.emit(x_pos, y_pos)
                            self.progress.emit(
                                f"({x_pos:.2f}, {y_pos:.2f}) skipped", pct)
                if self.should_stop:
                    break

            self.motor_manager.move_absolute_units('X', 0)
            self.motor_manager.move_absolute_units('Y', 0)

            self.finished.emit({
                'status': 'success' if not self.should_stop else 'stopped',
            })
        except Exception as e:
            self.finished.emit({'status': 'error', 'error': str(e)})


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class FocusMapPanel(QWidget):
    def __init__(self, motor_manager, preview, autofocus_panel,
                 wafer_mapping_panel=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Focus Map")
        self.motor_manager       = motor_manager
        self.preview             = preview
        self.autofocus_panel     = autofocus_panel
        self.wafer_mapping_panel = wafer_mapping_panel

        self._measured     = []   # list of (x, y, z_absolute, metric)
        self._skipped      = []   # list of (x, y)
        self._interp       = None   # fitted to Z offsets from _z_ref
        self._interp_error = None
        self._worker       = None

        # Offset model:
        #   _z_ref         — mean Z of measured points at build time
        #   _z_ref_applied — reference Z to add to offsets when querying;
        #                    set to _z_ref after build so the map is usable
        #                    immediately, and updated by the user when
        #                    reloading a saved map in a new session.
        self._z_ref         = None
        self._z_ref_applied = None

        self._init_ui()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_focus_z(self, x_mm, y_mm):
        """Return interpolated Z (mm) for (x_mm, y_mm), or None if no map.

        Returns z_ref_applied + offset(x, y).  z_ref_applied equals the
        absolute Z from the build session initially; call set_focus_reference()
        to re-anchor to a new Z after loading a saved map.
        """
        if self._interp is None or len(self._measured) < 3:
            return None
        if self._z_ref_applied is None:
            return None
        try:
            offset = float(self._interp([[x_mm, y_mm]])[0])
            return self._z_ref_applied + offset
        except Exception:
            return None

    def set_focus_reference(self, z_mm):
        """Anchor the offset map to z_mm at the current stage position.

        Call this after loading a saved map in a new session: focus manually
        at any representative point, then call set_focus_reference(current_Z).
        """
        self._z_ref_applied = float(z_mm)
        self._update_ref_label()

    def has_map(self):
        """True if a usable interpolated map is available."""
        return self._interp is not None and len(self._measured) >= 3 \
               and self._z_ref_applied is not None

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Focus Map</b>"))
        layout.addWidget(QLabel(
            "Measures autofocus Z at a regular grid and builds a 2D RBF surface.\n"
            "Points with poor focus metric are down-weighted in the interpolation."
        ))

        self._boundary_label = QLabel("No wafer boundaries — run Find Wafer Extents first.")
        layout.addWidget(self._boundary_label)

        # --- Grid settings ---
        grid_group  = QGroupBox("Measurement Grid")
        grid_layout = QHBoxLayout()
        grid_layout.addWidget(QLabel("Columns:"))
        self._cols_spin = QSpinBox()
        self._cols_spin.setRange(2, 15)
        self._cols_spin.setValue(5)
        grid_layout.addWidget(self._cols_spin)
        grid_layout.addWidget(QLabel("Rows:"))
        self._rows_spin = QSpinBox()
        self._rows_spin.setRange(2, 15)
        self._rows_spin.setValue(5)
        grid_layout.addWidget(self._rows_spin)
        grid_layout.addWidget(QLabel("  Min metric:"))
        self._min_metric_spin = QDoubleSpinBox()
        self._min_metric_spin.setRange(0, 1e7)
        self._min_metric_spin.setDecimals(0)
        self._min_metric_spin.setValue(100)
        self._min_metric_spin.setSingleStep(50)
        self._min_metric_spin.setToolTip(
            "Minimum Laplacian variance to accept autofocus result.\n"
            "Results below this trigger the manual focus dialog.")
        grid_layout.addWidget(self._min_metric_spin)
        grid_layout.addWidget(QLabel("  Fit degree:"))
        self._degree_combo = QComboBox()
        self._degree_combo.addItems(["1 — plane", "2 — quadratic", "3 — cubic"])
        self._degree_combo.setCurrentIndex(1)
        self._degree_combo.currentIndexChanged.connect(
            lambda: (self._rebuild_interp(), self._redraw()))
        grid_layout.addWidget(self._degree_combo)
        grid_layout.addStretch()
        grid_group.setLayout(grid_layout)
        layout.addWidget(grid_group)

        # --- Measure / stop / clear / apply ---
        btn_row = QHBoxLayout()
        self._measure_btn = QPushButton("Measure Focus Map")
        self._measure_btn.clicked.connect(self._start)
        btn_row.addWidget(self._measure_btn)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.clicked.connect(self._stop)
        self._stop_btn.setEnabled(False)
        btn_row.addWidget(self._stop_btn)
        self._clear_btn = QPushButton("Clear Map")
        self._clear_btn.clicked.connect(self._clear)
        btn_row.addWidget(self._clear_btn)
        self._apply_btn = QPushButton("Apply Focus Here")
        self._apply_btn.setToolTip(
            "Read current stage XY, move Z to the interpolated focus depth.")
        self._apply_btn.clicked.connect(self._apply_focus_at_current_xy)
        btn_row.addWidget(self._apply_btn)
        layout.addLayout(btn_row)

        # --- Save / load / set reference ---
        persist_row = QHBoxLayout()
        self._save_btn = QPushButton("Save Map…")
        self._save_btn.setToolTip(
            "Save focus offsets to a JSON file.\n"
            "The map stores Z offsets relative to the build-time mean Z,\n"
            "so it can be reloaded and reused in later sessions.")
        self._save_btn.clicked.connect(self._save_map)
        persist_row.addWidget(self._save_btn)

        self._load_btn = QPushButton("Load Map…")
        self._load_btn.setToolTip(
            "Load a previously saved focus-map JSON file.\n"
            "After loading, focus manually at any representative point\n"
            "on the wafer and click 'Set Reference Z' to anchor the offsets.")
        self._load_btn.clicked.connect(self._load_map)
        persist_row.addWidget(self._load_btn)

        self._set_ref_btn = QPushButton("Set Reference Z")
        self._set_ref_btn.setToolTip(
            "Records the current stage Z as the reference for the offset map.\n"
            "Use after loading a saved map: focus at a representative point\n"
            "on the wafer, then click this to anchor the offsets to that Z.")
        self._set_ref_btn.clicked.connect(self._set_reference_from_stage)
        persist_row.addWidget(self._set_ref_btn)
        layout.addLayout(persist_row)

        self._ref_label = QLabel("Reference Z: —")
        layout.addWidget(self._ref_label)

        self._progress_bar = QProgressBar()
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("No map measured.")
        layout.addWidget(self._status_label)

        # --- Heatmap ---
        self._fig    = Figure(figsize=(5, 4), tight_layout=True)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setMinimumSize(400, 320)
        layout.addWidget(self._canvas)

        self._stats_label = QLabel("")
        layout.addWidget(self._stats_label)

        layout.addStretch()
        self._redraw()

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_boundary_label()

    def _refresh_boundary_label(self):
        bounds = self._boundaries()
        if bounds:
            x_min, x_max = min(bounds[0], bounds[1]), max(bounds[0], bounds[1])
            y_min, y_max = min(bounds[2], bounds[3]), max(bounds[2], bounds[3])
            self._boundary_label.setText(
                f"Boundaries: X [{x_min:.1f}, {x_max:.1f}]  "
                f"Y [{y_min:.1f}, {y_max:.1f}]"
            )
        else:
            self._boundary_label.setText(
                "No wafer boundaries — run Find Wafer Extents first.")

    def _boundaries(self):
        if self.wafer_mapping_panel is None:
            return None
        return getattr(self.wafer_mapping_panel, 'wafer_boundaries', None)

    def _current_z_mm(self):
        z_cfg     = self.motor_manager.step_config.get('Z', {})
        z_step_mm = z_cfg.get('step', 0.001)
        z_invert  = z_cfg.get('invert', 1)
        pos = self.motor_manager.get_position('Z')
        hw  = getattr(pos, 'Position', 0) if pos is not None else 0
        return hw * z_invert * z_step_mm

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #

    def _start(self):
        self._refresh_boundary_label()
        bounds = self._boundaries()
        if bounds is None:
            self._status_label.setText("No wafer boundaries — run Find Wafer Extents first.")
            return
        if self.autofocus_panel is None:
            self._status_label.setText("No autofocus panel available.")
            return

        self._measured = []
        self._skipped  = []
        self._interp   = None

        self._worker = FocusMapWorker(
            motor_manager=self.motor_manager,
            preview=self.preview,
            autofocus_panel=self.autofocus_panel,
            wafer_boundaries=bounds,
            grid_cols=self._cols_spin.value(),
            grid_rows=self._rows_spin.value(),
            min_metric=self._min_metric_spin.value(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.point_measured.connect(self._on_point_measured)
        self._worker.point_skipped.connect(self._on_point_skipped)
        self._worker.manual_focus_requested.connect(self._on_manual_focus_requested)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

        self._measure_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

    def _stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(1000)

    def _clear(self):
        self._measured      = []
        self._skipped       = []
        self._interp        = None
        self._z_ref         = None
        self._z_ref_applied = None
        self._stats_label.setText("")
        self._status_label.setText("Map cleared.")
        self._update_ref_label()
        self._redraw()

    def _update_ref_label(self):
        if self._z_ref_applied is None:
            self._ref_label.setText("Reference Z: —")
        else:
            self._ref_label.setText(
                f"Reference Z: {self._z_ref_applied:.4f} mm  "
                f"(map mean: {self._z_ref:.4f} mm  "
                f"offset: {self._z_ref_applied - self._z_ref:+.4f} mm)"
                if self._z_ref is not None
                else f"Reference Z: {self._z_ref_applied:.4f} mm"
            )

    def _set_reference_from_stage(self):
        """Anchor the offset map to the current stage Z."""
        z_cfg     = self.motor_manager.step_config.get('Z', {})
        z_step_mm = z_cfg.get('step', 0.001)
        z_invert  = z_cfg.get('invert', 1)
        pos = self.motor_manager.get_position('Z')
        hw  = getattr(pos, 'Position', 0) if pos is not None else 0
        z_now = hw * z_invert * z_step_mm
        self.set_focus_reference(z_now)
        self._status_label.setText(
            f"Reference Z set to {z_now:.4f} mm  "
            f"(map offset: {z_now - self._z_ref:+.4f} mm)"
            if self._z_ref is not None
            else f"Reference Z set to {z_now:.4f} mm"
        )

    def _save_map(self):
        """Save offset map to a JSON file."""
        if not self._measured:
            QMessageBox.warning(self, "No Map", "No measurements to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Focus Map", "focus_map.json",
            "Focus map (*.json);;All files (*)")
        if not path:
            return
        degree = self._degree_combo.currentIndex() + 1
        data = {
            'z_ref':   self._z_ref,
            'degree':  degree,
            'points':  [
                {'x': x, 'y': y,
                 'z_offset': z - self._z_ref,
                 'metric': m}
                for x, y, z, m in self._measured
            ],
            'skipped': [{'x': x, 'y': y} for x, y in self._skipped],
        }
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            self._status_label.setText(
                f"Saved {len(self._measured)} points to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _load_map(self):
        """Load an offset map from a JSON file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Focus Map", "",
            "Focus map (*.json);;All files (*)")
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            z_ref   = float(data['z_ref'])
            degree  = int(data.get('degree', 2))
            points  = data['points']
            skipped = data.get('skipped', [])

            # Restore as absolute Z so _rebuild_interp can re-derive offsets
            self._measured = [
                (p['x'], p['y'], z_ref + p['z_offset'], p['metric'])
                for p in points
            ]
            self._skipped = [(p['x'], p['y']) for p in skipped]

            # Set the combo to the saved degree
            idx = min(max(degree - 1, 0), self._degree_combo.count() - 1)
            self._degree_combo.setCurrentIndex(idx)

            # _rebuild_interp will set _z_ref and _z_ref_applied from the data
            self._z_ref         = None   # force first-build branch
            self._z_ref_applied = None
            self._rebuild_interp()
            self._redraw()
            self._update_stats()

            self._status_label.setText(
                f"Loaded {len(self._measured)} points from {path}\n"
                "Focus manually at a representative point and click "
                "'Set Reference Z' to anchor the offsets to this session's Z.")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _apply_focus_at_current_xy(self):
        """Move Z to the interpolated focus depth at the current stage XY."""
        x = self.motor_manager.get_position_units('X') or 0.0
        y = self.motor_manager.get_position_units('Y') or 0.0
        z = self.get_focus_z(x, y)
        if z is None:
            err = self._interp_error or "unknown error"
            self._status_label.setText(f"No map: {err}")
            return
        z_cfg     = self.motor_manager.step_config.get('Z', {})
        z_step_mm = z_cfg.get('step', 0.001)
        self.motor_manager.move_absolute('Z', int(round(z / z_step_mm)))
        self._status_label.setText(
            f"Applied focus: X={x:.2f} Y={y:.2f} → Z={z:.4f} mm")

    # ------------------------------------------------------------------ #
    # Slots
    # ------------------------------------------------------------------ #

    def _on_progress(self, msg, pct):
        self._status_label.setText(msg)
        if pct > 0:
            self._progress_bar.setValue(pct)

    def _on_point_measured(self, x, y, z, metric):
        self._measured.append((x, y, z, metric))
        self._rebuild_interp()
        self._redraw()

    def _on_point_skipped(self, x, y):
        self._skipped.append((x, y))
        self._redraw()

    def _on_manual_focus_requested(self, x, y):
        dlg = QDialog(self)
        dlg.setWindowTitle("Manual Focus Required")
        dlg.setWindowFlags(dlg.windowFlags() | Qt.WindowStaysOnTopHint)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(
            f"Autofocus quality too low at ({x:.2f}, {y:.2f}) mm.\n\n"
            "Adjust focus manually using the stage Z controls,\n"
            "then click Accept to record the current Z position,\n"
            "or Skip to leave this point unmeasured."
        ))
        btns = QDialogButtonBox()
        accept_btn = btns.addButton("Accept current Z", QDialogButtonBox.AcceptRole)
        accept_btn.setDefault(True)
        btns.addButton("Skip point", QDialogButtonBox.RejectRole)
        lay.addWidget(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        if dlg.exec_() == QDialog.Accepted:
            self._worker.resume_with_z(self._current_z_mm())
        else:
            self._worker.skip_point()

    def _on_finished(self, result):
        self._measure_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        if result['status'] == 'error':
            self._status_label.setText(f"Error: {result.get('error')}")
            return
        n = len(self._measured)
        s = len(self._skipped)
        self._status_label.setText(
            f"{'Complete' if result['status'] == 'success' else 'Stopped'} — "
            f"{n} measured, {s} skipped."
        )
        self._rebuild_interp()
        self._redraw()
        self._update_stats()

    # ------------------------------------------------------------------ #
    # Interpolation — 2D RBF with per-point quality weighting
    # ------------------------------------------------------------------ #

    def _rebuild_interp(self):
        degree  = self._degree_combo.currentIndex() + 1  # 1, 2, or 3
        min_pts = (degree + 1) * (degree + 2) // 2       # coefficients needed
        if len(self._measured) < min_pts:
            self._interp = None
            self._interp_error = (f"Degree-{degree} fit needs ≥{min_pts} points "
                                  f"(have {len(self._measured)})")
            return
        try:
            pts     = np.array([(x, y) for x, y, z, m in self._measured])
            z_abs   = np.array([z       for x, y, z, m in self._measured])
            metrics = np.array([m       for x, y, z, m in self._measured], dtype=float)
            m_max   = metrics.max()
            w       = metrics / m_max if m_max > 0 else np.ones(len(metrics))

            # Fit offsets from mean Z so the surface has zero mean
            z_ref_new = float(z_abs.mean())
            offsets   = z_abs - z_ref_new
            self._interp = _PolySurface(pts, offsets, weights=w, degree=degree)
            self._interp_error = None

            # Update reference: preserve z_ref_applied if it was manually set
            # (i.e. user loaded a map and set a reference), otherwise keep it
            # tracking the build-time absolute Z.
            if self._z_ref is None:
                # First build: anchor applied reference to build absolute Z
                self._z_ref_applied = z_ref_new
            else:
                # Incremental update during an ongoing measurement: shift the
                # applied reference by the change in z_ref so that absolute Z
                # predictions stay consistent.
                if self._z_ref_applied is not None:
                    self._z_ref_applied += (z_ref_new - self._z_ref)
                else:
                    self._z_ref_applied = z_ref_new
            self._z_ref = z_ref_new
            self._update_ref_label()
        except Exception as e:
            self._interp = None
            self._interp_error = str(e)
            print(f"[FocusMap] _rebuild_interp failed: {e}")

    # ------------------------------------------------------------------ #
    # Visualisation
    # ------------------------------------------------------------------ #

    def _redraw(self):
        self._fig.clear()
        ax = self._fig.add_subplot(111)

        bounds = self._boundaries()
        if bounds:
            x_min, x_max = min(bounds[0], bounds[1]), max(bounds[0], bounds[1])
            y_min, y_max = min(bounds[2], bounds[3]), max(bounds[2], bounds[3])
        elif self._measured:
            x_min = min(p[0] for p in self._measured)
            x_max = max(p[0] for p in self._measured)
            y_min = min(p[1] for p in self._measured)
            y_max = max(p[1] for p in self._measured)
        else:
            ax.set_title("Focus map (no data)")
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            self._canvas.draw()
            return

        # Work in offset space so the map looks identical regardless of
        # which session it was built in.
        z_ref = self._z_ref or 0.0

        # Interpolated surface (offsets from z_ref)
        if self._interp is not None and len(self._measured) >= 3:
            res = 100
            xi = np.linspace(x_min, x_max, res)
            yi = np.linspace(y_min, y_max, res)
            xx, yy = np.meshgrid(xi, yi)
            zz = self._interp(
                np.column_stack([xx.ravel(), yy.ravel()])
            ).reshape(res, res)
            im = ax.pcolormesh(xx, yy, zz * 1000, cmap='RdYlGn_r',
                               shading='auto')
            self._fig.colorbar(im, ax=ax, label='ΔZ offset (µm)')

        # Measured points annotated with their offsets
        if self._measured:
            offsets = np.array([z - z_ref for _, _, z, _ in self._measured])
            metrics = np.array([m for _, _, _, m in self._measured], dtype=float)
            m_max   = metrics.max() if metrics.max() > 0 else 1.0
            w       = metrics / m_max
            o_lo, o_hi = offsets.min(), offsets.max()
            for (x, y, z, m), wi, off in zip(self._measured, w, offsets):
                edge = 'black' if wi > 0.3 else 'red'
                size = 30 + 70 * wi
                ax.scatter(x, y, c=[off * 1000], vmin=o_lo * 1000,
                           vmax=o_hi * 1000,
                           cmap='RdYlGn_r', edgecolors=edge,
                           linewidths=1.2, s=size, zorder=5)
                ax.annotate(f"{off*1000:+.0f}µm", (x, y),
                            textcoords='offset points', xytext=(4, 4),
                            fontsize=6, color='white',
                            path_effects=[_black_outline()])

        # Skipped points
        if self._skipped:
            ax.scatter([p[0] for p in self._skipped],
                       [p[1] for p in self._skipped],
                       c='gray', marker='x', s=50, zorder=5, label='skipped')
            ax.legend(fontsize=7)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ref_str = f"  ref={z_ref:.4f} mm" if self._z_ref else ""
        ax.set_title(f"Focus map — {len(self._measured)} points{ref_str}")
        self._canvas.draw()

    def _update_stats(self):
        if not self._measured:
            self._stats_label.setText("")
            return
        z_ref = self._z_ref or 0.0
        offs = [p[2] - z_ref for p in self._measured]
        ms   = [p[3] for p in self._measured]
        poor = sum(1 for m in ms if m < (max(ms) * 0.3)) if ms else 0
        self._stats_label.setText(
            f"ΔZ  {min(offs)*1000:+.1f} → {max(offs)*1000:+.1f} µm  "
            f"peak-to-valley={(max(offs)-min(offs))*1000:.1f} µm  "
            f"σ={np.std(offs)*1000:.1f} µm"
            f"  |  metric {min(ms):.0f} → {max(ms):.0f}"
            + (f"  ({poor} low-quality)" if poor else "")
        )


# ---------------------------------------------------------------------------
# Polynomial surface fit (numpy only, no scipy required)
# ---------------------------------------------------------------------------

class _PolySurface:
    """Weighted least-squares polynomial surface z = f(x, y).

    All monomials x^i * y^j with i+j <= degree are included.
    Quality weights downweight poor-focus measurements so they influence
    the surface less without being discarded entirely.
    Extrapolates naturally beyond the measurement area.
    """

    def __init__(self, pts, vals, weights=None, degree=2):
        pts  = np.asarray(pts,  dtype=float)
        vals = np.asarray(vals, dtype=float)
        w    = (np.asarray(weights, dtype=float)
                if weights is not None else np.ones(len(vals)))
        # Normalise coordinates to [-1, 1] for numerical stability
        self._cx = pts[:, 0].mean(); self._sx = pts[:, 0].std() or 1.0
        self._cy = pts[:, 1].mean(); self._sy = pts[:, 1].std() or 1.0
        xn = (pts[:, 0] - self._cx) / self._sx
        yn = (pts[:, 1] - self._cy) / self._sy
        A  = _poly_matrix(xn, yn, degree)
        # Weighted least squares: scale rows by sqrt(w)
        sw = np.sqrt(np.maximum(w, 1e-6))
        self._coeffs, _, _, _ = np.linalg.lstsq(A * sw[:, None], vals * sw, rcond=None)
        self._degree = degree

    def __call__(self, query_pts):
        query_pts = np.asarray(query_pts, dtype=float)
        xn = (query_pts[:, 0] - self._cx) / self._sx
        yn = (query_pts[:, 1] - self._cy) / self._sy
        A  = _poly_matrix(xn, yn, self._degree)
        return A @ self._coeffs


def _poly_matrix(x, y, degree):
    """Design matrix: all monomials x^i * y^j with i+j <= degree."""
    cols = []
    for total in range(degree + 1):
        for i in range(total + 1):
            cols.append(x ** i * y ** (total - i))
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _black_outline():
    from matplotlib.patheffects import withStroke
    return withStroke(linewidth=2, foreground='black')

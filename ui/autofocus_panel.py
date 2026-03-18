# ui/autofocus_panel.py
import json
import os
import datetime

from PyQt5.QtCore import QThread, pyqtSignal, QObject, Qt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QDoubleSpinBox, QComboBox, QGroupBox, QFormLayout, QDialog)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import cv2
import time

# ── Per-magnification built-in defaults ───────────────────────────────────────
# Keys are objective magnification as float.  Parameters scale with depth of
# field: step and fine_factor shrink as magnification increases.
_BUILTIN_DEFAULTS = {
     5: dict(metric='LaplacianVariance', step_mm=0.010, settle_ms=20,  coarse_factor=4,  fine_factor=0.25, fine_range_factor=1.0,  z_range_mm=4.0,  trend_n=10, n_avg=1),
    10: dict(metric='LaplacianVariance', step_mm=0.003, settle_ms=30,  coarse_factor=6,  fine_factor=0.20, fine_range_factor=0.75, z_range_mm=2.0,  trend_n=10, n_avg=1),
    20: dict(metric='LaplacianVariance', step_mm=0.001, settle_ms=50,  coarse_factor=10, fine_factor=0.10, fine_range_factor=0.5,  z_range_mm=1.0,  trend_n=10, n_avg=1),
    40: dict(metric='LaplacianVariance', step_mm=0.001, settle_ms=80,  coarse_factor=10, fine_factor=0.10, fine_range_factor=0.5,  z_range_mm=0.5,  trend_n=10, n_avg=2),
   100: dict(metric='LaplacianVariance', step_mm=0.001, settle_ms=100, coarse_factor=10, fine_factor=0.10, fine_range_factor=0.5,  z_range_mm=0.3,  trend_n=10, n_avg=3),
}

_DEFAULTS_FILE = os.path.join(os.path.dirname(__file__), '..', 'autofocus_defaults.json')
_LOG_DIR       = os.path.join(os.path.dirname(__file__), '..', 'autofocus_logs')


class FocusWorker(QObject):
    progress = pyqtSignal(float, float, float)  # z_mm, metric, best_z_mm
    finished = pyqtSignal(float, float)         # best_z_mm, best_metric
    aborted  = pyqtSignal(str)

    def __init__(self, mm, get_frame_fn, params):
        super().__init__()
        self.mm        = mm
        self.get_frame = get_frame_fn
        self.p         = params
        self._stop     = False

    def stop(self):
        self._stop = True

    def _save_log(self, z_list, m_list, params, result, **extra):
        """Save sweep data to a timestamped JSON file in autofocus_logs/."""
        try:
            os.makedirs(_LOG_DIR, exist_ok=True)
            ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(_LOG_DIR, f'af_{ts}_{result}.json')
            log  = {
                'timestamp': ts,
                'result':    result,
                'params':    {k: v for k, v in params.items()
                              if isinstance(v, (int, float, str, bool, type(None)))},
                'z_list':    [round(z, 6) for z in z_list],
                'm_list':    [round(m, 4) for m in m_list],
            }
            log.update(extra)
            with open(path, 'w') as f:
                json.dump(log, f, indent=2)
        except Exception:
            pass   # never crash the worker over a logging failure

    def _measure_avg(self, n=1):
        """Measure focus metric, optionally averaged over n frames."""
        if n <= 1:
            return self._measure()
        ms = [self._measure() for _ in range(n)]
        ms = [m for m in ms if m is not None]
        return float(np.mean(ms)) if ms else None

    def _metric(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        m = self.p['metric']
        if m == 'LaplacianVariance':
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        elif m == 'Tenengrad':
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            return float((gx**2 + gy**2).mean())
        elif m == 'Brightness':
            return float(gray.mean())
        else:
            return float(gray.var())

    def _get_z_mm(self):
        try:
            return self.mm.get_position_units('Z')
        except Exception:
            return None

    def _move(self, z_mm):
        z_min, z_max = self.p['z_min_mm'], self.p['z_max_mm']
        z_mm = max(z_min, min(z_mm, z_max))
        self.mm.move_absolute_units('Z', z_mm, wait=False)
        t0 = time.time()
        tol = self.p['z_step_mm'] * 1.5
        while time.time() - t0 < 3.0:
            cur = self._get_z_mm()
            if cur is not None and abs(cur - z_mm) <= tol:
                break
            time.sleep(0.01)
        if self.p['settle_ms'] > 0:
            time.sleep(self.p['settle_ms'] / 1000.0)
        # Return the actual readback position after settling so that z_list
        # records where the motor really is, not just what we commanded.
        actual = self._get_z_mm()
        return actual if actual is not None else z_mm

    def _measure(self):
        # Discard one frame captured before or during the motor move, then
        # wait long enough for the camera to capture a genuinely fresh frame.
        # At typical preview rates (10–30 fps, 33–100 ms/frame) settle_ms is
        # already the right order of magnitude; clamping to ≥ 50 ms covers
        # fast-settle cases so the two get_frame() calls can't both return the
        # same buffered frame.
        self.get_frame()
        time.sleep(max(self.p['settle_ms'], 50) / 1000.0)
        frame = self.get_frame()
        return self._metric(frame) if frame is not None else None

    def run(self):
        """
        Two-phase autofocus based on the derivative of the focus metric.

        Phase 1 (coarse) — derivative / zero-crossing approach:
          1. Choose initial scan direction from last_focus_z hint (if stored
             for this magnification) or fall back to +Z.
          2. Scan exactly trend_n coarse steps, computing delta[i] = m[i]-m[i-1]
             at each step.
          3. If the mean delta over those steps is negative, the metric is
             declining → reverse direction and restart from z0 with a fresh
             delta list.  No configurable threshold: the sign of the mean over
             trend_n (≥ 10) steps is noise-robust on its own.
          4. Continue scanning.  At every step compute the smoothed delta =
             mean of the last trend_n//2 deltas.
             • If smoothed_delta > 0 → set was_positive = True (metric rising,
               heading toward the peak).
             • If was_positive and smoothed_delta < 0 → the smoothed derivative
               just crossed zero from above: the peak is confirmed.  Stop.
          5. If no zero crossing is found before the boundary, return to z0
             and abort — no fine scan on an uncertain peak.

        Phase 2 (fine):
          - Fit a downward parabola to the coarse measurements nearest the
            argmax (by Z distance) for a sub-step accurate peak estimate.
          - Sweep ±fine_range_factor × coarse_step around the vertex.
          - Move to the best Z found and emit finished().
        """
        try:
            step              = self.p['step_mm']
            z_min             = self.p['z_min_mm']   # stage hard limit (clamps _move)
            z_max             = self.p['z_max_mm']   # stage hard limit (clamps _move)
            z_range_mm        = float(self.p.get('z_range_mm', 2.0))
            coarse_factor     = max(1, int(self.p.get('coarse_factor', 4)))
            fine_factor       = max(0.01, float(self.p.get('fine_factor', 0.25)))
            fine_range_factor = max(0.1, float(self.p.get('fine_range_factor', 1.0)))
            n_avg             = max(1, int(self.p.get('n_avg', 1)))
            coarse_step       = step * coarse_factor
            fine_step         = step * fine_factor
            TREND_N           = max(4, int(self.p.get('trend_n', 10)))
            HALF_N            = max(2, TREND_N // 2)
            last_focus_z      = self.p.get('last_focus_z')

            z0 = self._get_z_mm() or 0.0
            self._move(z0)
            m0 = self._measure_avg(n_avg)
            if m0 is None:
                self.aborted.emit("No frame available")
                return

            z_list = [z0]
            m_list = [m0]
            self.progress.emit(z0, m0, z0)

            def probe(z):
                if self._stop:
                    return None
                z_actual = self._move(z)
                m = self._measure_avg(n_avg)
                if m is not None:
                    z_list.append(z_actual)
                    m_list.append(m)
                    best_idx = int(np.argmax(m_list))
                    self.progress.emit(z_actual, m, z_list[best_idx])
                return m

            # ── Phase 1: coarse scan (derivative / zero-crossing) ─────────

            # Initial scan direction from stored focus hint
            if last_focus_z is not None and abs(last_focus_z - z0) > coarse_step:
                direction = +1 if last_focus_z > z0 else -1
            else:
                direction = +1

            # Helper: scan from z_scan in direction until zero crossing or boundary.
            # was_pos seeds whether we have already seen a smoothed delta that
            # exceeded the noise_floor.  noise_floor prevents false zero crossings
            # in flat/noisy data where the metric barely changes at all.
            def scan_for_peak(direction, z_scan, deltas,
                              was_pos=False, noise_floor=0.0):
                # When starting fresh (no prior deltas), probe the starting
                # position first so that the first step's delta is computed
                # relative to m(z_scan), not to whatever was last visited in
                # a previous scan segment.  Without this, the jump from the
                # end of one segment to the start of the next creates a
                # spurious large delta that can falsely set was_pos=True.
                if not deltas:
                    probe(z_scan)
                max_steps = int(z_range_mm / coarse_step) + 5
                for _ in range(max_steps):
                    if self._stop:
                        return False, z_scan
                    z_scan += direction * coarse_step
                    if z_scan < z_min or z_scan > z_max:
                        break
                    probe(z_scan)
                    deltas.append(m_list[-1] - m_list[-2])
                    smoothed = float(np.mean(deltas[-HALF_N:]))
                    # Raw-delta check comes first: a single step that strongly
                    # exceeds the noise floor in the negative direction means the
                    # metric has turned around, even if the smoothed average is
                    # still positive from preceding large positive deltas (common
                    # after a precipitous focus peak).
                    if was_pos and deltas[-1] < -noise_floor:
                        return True, z_scan
                    if smoothed > noise_floor:
                        was_pos = True
                    elif was_pos and smoothed < 0:
                        return True, z_scan
                # Metric was still rising when the boundary was reached —
                # the peak is at or just beyond the scan edge.  Treat as found.
                return was_pos, z_scan

            # Initial trend_n steps with integrated zero-crossing detection.
            # During this phase the noise floor has not yet been estimated so we
            # use 0 as the threshold — this is intentional: the initial scan is
            # also the data source for the noise estimate used afterwards.
            deltas     = []
            z_scan     = z0
            was_pos    = False
            peak_found = False
            for _ in range(TREND_N):
                if self._stop:
                    self.aborted.emit("Stopped by user"); return
                z_scan += direction * coarse_step
                if z_scan < z_min or z_scan > z_max:
                    break
                probe(z_scan)
                deltas.append(m_list[-1] - m_list[-2])
                smoothed = float(np.mean(deltas[-HALF_N:]))
                if smoothed > 0:
                    was_pos = True
                elif was_pos and smoothed < 0:
                    peak_found = True
                    break

            if self._stop:
                self.aborted.emit("Stopped by user"); return

            # Estimate noise floor from the initial scan deltas.
            # The continued scan requires smoothed_delta > noise_floor to set
            # was_pos, preventing a false zero crossing in flat/noisy images where
            # the metric barely changes across the whole Z range.
            # Use the lower half of |deltas| for the MAD estimate so that a
            # focus peak that falls within the initial scan (possible when starting
            # very close to focus) does not inflate the noise floor.  The peak
            # contributes large |deltas| that land in the upper half; the lower
            # half reflects the true background noise level.
            if len(deltas) >= 3:
                sorted_abs  = np.sort(np.abs(deltas))
                noise_floor = float(np.median(
                    sorted_abs[:max(1, len(sorted_abs) // 2)])) * 5
            else:
                noise_floor = 0.0

            # Re-evaluate was_pos using the noise floor.  The initial scan uses
            # threshold=0 so even a tiny noise wiggle can set was_pos=True.
            # Check the total metric rise; if it's smaller than the noise floor
            # there was no real signal and we shouldn't continue in this direction.
            # NOTE: do NOT reset peak_found here — if a genuine zero-crossing was
            # detected (metric rose then fell) in the initial scan it is a real
            # peak regardless of the noise floor estimate.  The noise floor
            # estimate itself can be inflated when the peak falls within the
            # initial scan (all |deltas| are similar in magnitude for a symmetric
            # peak → lower-half median ≈ per-step delta → noise_floor ≈ max_rise).
            if deltas and noise_floor > 0 and was_pos and not peak_found:
                max_rise = max(m_list[1:len(deltas) + 1]) - m_list[0]
                if max_rise <= noise_floor:
                    was_pos = False

            # Direction decision and continuation (only if peak not yet found)
            if not peak_found:
                if not was_pos:
                    # Metric never rose during initial scan → wrong direction
                    direction = -direction
                    z_scan    = z0
                    deltas    = []
                # else: rising but not yet peaked → keep going the same way
                peak_found, z_scan = scan_for_peak(direction, z_scan, deltas,
                                                   was_pos=was_pos,
                                                   noise_floor=noise_floor)

            # Last resort: try the opposite direction from z0
            if not peak_found:
                if self._stop:
                    self.aborted.emit("Stopped by user"); return
                deltas = []
                peak_found, z_scan = scan_for_peak(-direction, z0, deltas,
                                                   noise_floor=noise_floor)

            if self._stop:
                self.aborted.emit("Stopped by user"); return

            if not peak_found:
                self._move(z0)
                self._save_log(z_list, m_list, self.p, result='no_peak')
                self.aborted.emit("No peak found — returned to z0")
                return

            # ── Phase 2: fine sweep around coarse peak ────────────────────
            z_coarse = _parabolic_peak(z_list, m_list, n_neighbors=TREND_N)
            fine_lo  = z_coarse - coarse_step * fine_range_factor
            fine_hi  = z_coarse + coarse_step * fine_range_factor
            n_fine   = max(3, int(round((fine_hi - fine_lo) / fine_step)) + 1)
            for z in np.linspace(fine_lo, fine_hi, n_fine):
                if self._stop:
                    self.aborted.emit("Stopped by user"); return
                probe(z)

            if self._stop:
                self.aborted.emit("Stopped by user"); return

            best_idx = int(np.argmax(m_list))
            z_best   = z_list[best_idx]
            self._move(z_best)
            self._save_log(z_list, m_list, self.p, result='finished',
                           z_best=z_best, m_best=float(m_list[best_idx]))
            self.finished.emit(z_best, float(m_list[best_idx]))
        except Exception as e:
            self.aborted.emit(str(e))


def _parabolic_peak(z_list, m_list, n_neighbors=5):
    """Return a sub-step accurate peak Z by fitting a parabola near the argmax.

    Selects the 2*n_neighbors+1 points spatially closest (in Z) to the
    argmax — NOT by list index — so that a direction reversal during the
    coarse sweep (probe steps in +Z then sweep continues in −Z) cannot
    accidentally pull in spatially distant points and corrupt the fit.

    Fits ax² + bx + c and returns the vertex −b/2a.  Falls back to the
    raw argmax if the fit is degenerate (upward/flat parabola, or too few
    distinct Z values).
    """
    zs = np.asarray(z_list, dtype=float)
    ms = np.asarray(m_list, dtype=float)
    best_z = float(zs[int(np.argmax(ms))])

    # Pick the nearest 2*n_neighbors+1 points by Z distance to the peak
    order  = np.argsort(np.abs(zs - best_z))[:2 * n_neighbors + 1]
    z_win  = zs[order]
    m_win  = ms[order]

    if len(np.unique(z_win)) < 3:
        return best_z
    try:
        a, b, _ = np.polyfit(z_win, m_win, 2)
    except Exception:
        return best_z
    if a >= 0:                  # upward or flat — not a valid focus peak
        return best_z
    vertex = -b / (2.0 * a)
    # Clamp to the fitted window to prevent wild extrapolation
    return float(np.clip(vertex, z_win.min(), z_win.max()))


class AutoFocusProgressDialog(QDialog):
    """Live metric-vs-Z plot shown while autofocus is running.

    Connect to a FocusWorker's signals before calling show():
        worker.progress.connect(dlg.on_progress)
        worker.finished.connect(dlg.on_finished)
        worker.aborted.connect(dlg.on_aborted)
    The dialog closes itself when finished or aborted is received.
    """

    stop_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Autofocus running…")
        self.setWindowModality(Qt.ApplicationModal)
        self.setMinimumSize(480, 480)

        lay = QVBoxLayout(self)

        self._fig = Figure(figsize=(5, 5), tight_layout=True)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setMinimumSize(420, 400)
        lay.addWidget(self._canvas)

        # Two subplots: metric on top, delta (derivative) on bottom
        self._ax_m = self._fig.add_subplot(211)
        self._ax_d = self._fig.add_subplot(212)
        self._ax_m.set_ylabel("Focus metric")
        self._ax_m.set_title("Autofocus sweep")
        self._ax_d.set_xlabel("Z (mm)")
        self._ax_d.set_ylabel("Δ metric / step")
        self._ax_d.axhline(0, color='k', linewidth=0.6, linestyle=':')
        self._canvas.draw()

        self._status = QLabel("Running…")
        lay.addWidget(self._status)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.clicked.connect(self.stop_requested)
        lay.addWidget(self._stop_btn)

        self._z_data  = []
        self._m_data  = []
        self._best_z  = None

    # ── Slots ──────────────────────────────────────────────────────────────

    def on_progress(self, z, m, best_z):
        self._z_data.append(z)
        self._m_data.append(m)
        self._best_z = best_z
        self._redraw()
        self._status.setText(
            f"z = {z:.4f} mm    metric = {m:.1f}    best = {best_z:.4f} mm")

    def on_finished(self, z_best, m_best):
        self._best_z = z_best
        self._redraw()
        self._status.setText(
            f"Done — best z = {z_best:.4f} mm    metric = {m_best:.1f}")
        self._stop_btn.setEnabled(False)
        self.accept()

    def on_aborted(self, reason):
        self._status.setText(f"Aborted: {reason}")
        self._stop_btn.setEnabled(False)
        self.reject()

    # ── Internal ───────────────────────────────────────────────────────────

    def _redraw(self):
        self._ax_m.clear()
        self._ax_d.clear()

        self._ax_m.set_ylabel("Focus metric")
        self._ax_m.set_title("Autofocus sweep")
        self._ax_d.set_xlabel("Z (mm)")
        self._ax_d.set_ylabel("Δ metric / step")
        self._ax_d.axhline(0, color='k', linewidth=0.6, linestyle=':')

        if self._z_data:
            self._ax_m.plot(self._z_data, self._m_data,
                            color='steelblue', marker='.', markersize=4,
                            linewidth=0.8, label='metric')
            if self._best_z is not None:
                self._ax_m.axvline(self._best_z, color='tomato', linestyle='--',
                                   linewidth=1.4,
                                   label=f'best = {self._best_z:.4f} mm')
                self._ax_d.axvline(self._best_z, color='tomato', linestyle='--',
                                   linewidth=1.4)
            self._ax_m.legend(fontsize=8)

        if len(self._z_data) >= 2:
            dz = self._z_data[1:]
            dm = [self._m_data[i + 1] - self._m_data[i]
                  for i in range(len(self._m_data) - 1)]
            self._ax_d.plot(dz, dm,
                            color='seagreen', marker='.', markersize=4,
                            linewidth=0.8, label='delta')
            self._ax_d.legend(fontsize=8)

        self._canvas.draw()

    def closeEvent(self, event):
        # If the user clicks X while still running, treat as Stop
        if self._stop_btn.isEnabled():
            self.stop_requested.emit()
        super().closeEvent(event)


class AutoFocusPanel(QWidget):
    def __init__(self, motor_manager, preview_obj, stage_controls=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Autofocus")
        self.mm             = motor_manager
        self.preview        = preview_obj
        self.stage_controls = stage_controls

        self._user_defaults = self._load_defaults_file()

        lay = QVBoxLayout(self)

        # ── Per-magnification defaults ────────────────────────────────────
        mag_group = QGroupBox("Per-magnification defaults")
        mag_row   = QHBoxLayout()
        self._mag_label = QLabel("Mag: —")
        mag_row.addWidget(self._mag_label)
        apply_btn = QPushButton("Apply for current mag")
        apply_btn.setToolTip(
            "Load the saved (or built-in) defaults for the current objective\n"
            "magnification and apply them to the parameters below.")
        apply_btn.clicked.connect(self._apply_for_current_mag)
        mag_row.addWidget(apply_btn)
        save_btn = QPushButton("Save as default")
        save_btn.setToolTip(
            "Save the current parameter values as the default for the\n"
            "current objective magnification.")
        save_btn.clicked.connect(self._save_for_current_mag)
        mag_row.addWidget(save_btn)
        mag_row.addStretch()
        mag_group.setLayout(mag_row)
        lay.addWidget(mag_group)

        # ── Metric ────────────────────────────────────────────────────────
        metric_row = QHBoxLayout()
        metric_row.addWidget(QLabel("Metric:"))
        self.metric = QComboBox()
        self.metric.addItems(["LaplacianVariance", "Tenengrad", "ImageVariance", "Brightness"])
        metric_row.addWidget(self.metric)
        metric_row.addStretch()
        lay.addLayout(metric_row)

        # ── Parameters ────────────────────────────────────────────────────
        params_box  = QGroupBox("Parameters")
        params_form = QFormLayout()
        self.step          = self._dspin(0.0001, 1.0,  0.005, 4)
        self.settle        = self._dspin(0,      2000,  20,   0)
        self.coarse_factor     = self._dspin(1,    20,    4,    0)
        self.fine_factor       = self._dspin(0.01, 1.0,   0.25, 2)
        self.fine_range_factor = self._dspin(0.1,  5.0,   1.0,  2)
        self.fine_range_factor.setToolTip(
            "Fine sweep half-width = fine_range_factor × coarse_step.\n"
            "Reduce once the coarse sweep reliably finds the peak.\n"
            "e.g. 0.5 → ±½ coarse step around coarse peak.")
        self.trend_n           = self._dspin(4,    30,    10,   0)
        self.trend_n.setToolTip(
            "Controls two things:\n"
            "  • Initial scan window: trend_n coarse steps are taken before\n"
            "    deciding whether to reverse direction (sign of mean delta).\n"
            "  • Smoothing window: trend_n//2 deltas are averaged for the\n"
            "    zero-crossing detector.  Increase to suppress noise.")
        self.n_avg             = self._dspin(1,    8,     1,    0)
        self.z_range           = self._dspin(0.01, 20.0,  0.5,  2)
        self.z_range.setToolTip(
            "Maximum coarse search distance (mm) in each direction from the\n"
            "starting position.  The scan stops as soon as a peak is found,\n"
            "so this is only a runaway-prevention budget, not a window.\n"
            "Stage hard limits (min_mm / max_mm in step_config.json) are the\n"
            "only absolute constraints.")
        params_form.addRow("Step (mm):",     self.step)
        params_form.addRow("Settle (ms):",   self.settle)
        params_form.addRow("Coarse factor:", self.coarse_factor)
        params_form.addRow("Fine factor:",       self.fine_factor)
        params_form.addRow("Fine range factor:", self.fine_range_factor)
        params_form.addRow("Trend N:",           self.trend_n)
        params_form.addRow("Frames avg:",        self.n_avg)
        params_form.addRow("Z range (mm):",      self.z_range)
        params_box.setLayout(params_form)
        lay.addWidget(params_box)

        # ── Run / Stop ────────────────────────────────────────────────────
        btns = QHBoxLayout()
        self.run_btn  = QPushButton("Run autofocus")
        self.stop_btn = QPushButton("Stop")
        btns.addWidget(self.run_btn)
        btns.addWidget(self.stop_btn)
        lay.addLayout(btns)

        self.status = QLabel("Idle.")
        lay.addWidget(self.status)

        self.thread = None
        self.worker = None

        self.run_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop_worker)

    # ── Defaults machinery ────────────────────────────────────────────────

    @staticmethod
    def _load_defaults_file():
        try:
            path = os.path.normpath(_DEFAULTS_FILE)
            if os.path.exists(path):
                with open(path) as f:
                    # JSON keys are strings; convert to float
                    return {float(k): v for k, v in json.load(f).items()}
        except Exception as e:
            print(f"[AutoFocus] Could not load defaults: {e}")
        return {}

    def _save_defaults_file(self):
        try:
            path = os.path.normpath(_DEFAULTS_FILE)
            with open(path, 'w') as f:
                json.dump({str(k): v for k, v in self._user_defaults.items()},
                          f, indent=2)
        except Exception as e:
            print(f"[AutoFocus] Could not save defaults: {e}")

    def _current_mag(self):
        raw = getattr(self.preview, 'magnification', None)
        try:
            return float(str(raw).rstrip('xX '))
        except (ValueError, TypeError):
            return 0.0

    def _defaults_for_mag(self, mag):
        """Return the best available defaults for mag.

        Priority: user-saved > nearest built-in (by magnification).
        """
        if mag in self._user_defaults:
            return self._user_defaults[mag]
        # Nearest built-in
        if not _BUILTIN_DEFAULTS:
            return {}
        nearest = min(_BUILTIN_DEFAULTS, key=lambda m: abs(m - mag))
        return _BUILTIN_DEFAULTS[nearest]

    def _apply_defaults(self, d):
        """Write a defaults dict into the spinboxes."""
        if 'metric' in d:
            idx = self.metric.findText(d['metric'])
            if idx >= 0:
                self.metric.setCurrentIndex(idx)
        if 'step_mm'       in d: self.step.setValue(d['step_mm'])
        if 'settle_ms'     in d: self.settle.setValue(d['settle_ms'])
        if 'coarse_factor' in d: self.coarse_factor.setValue(d['coarse_factor'])
        if 'fine_factor'       in d: self.fine_factor.setValue(d['fine_factor'])
        if 'fine_range_factor' in d: self.fine_range_factor.setValue(d['fine_range_factor'])
        if 'z_range_mm'        in d: self.z_range.setValue(d['z_range_mm'])
        if 'trend_n'  in d: self.trend_n.setValue(d['trend_n'])
        if 'n_avg'    in d: self.n_avg.setValue(d['n_avg'])

    def _apply_for_current_mag(self):
        self.apply_defaults_for_mag(getattr(self.preview, 'magnification', None))

    def apply_defaults_for_mag(self, mag_raw):
        """Apply defaults for *mag_raw* (string like '20x' or numeric).

        Called automatically when the magnification selector changes.
        """
        try:
            mag = float(str(mag_raw).rstrip('xX '))
        except (ValueError, TypeError):
            mag = 0.0
        if mag <= 0:
            self.status.setText("Cannot read magnification from preview.")
            return
        d = self._defaults_for_mag(mag)
        self._apply_defaults(d)
        source = "user" if mag in self._user_defaults else "built-in"
        self.status.setText(f"Applied {source} defaults for {mag:g}×")
        self._mag_label.setText(f"Mag: {mag:g}×")

    def _save_for_current_mag(self):
        mag = self._current_mag()
        if mag <= 0:
            self.status.setText("Cannot read magnification from preview.")
            return
        # Preserve last_focus_z if already stored (not overwritten by manual save)
        existing = self._user_defaults.get(mag, {})
        self._user_defaults[mag] = dict(
            metric            = self.metric.currentText(),
            step_mm           = self.step.value(),
            settle_ms         = int(self.settle.value()),
            coarse_factor     = int(self.coarse_factor.value()),
            fine_factor       = self.fine_factor.value(),
            fine_range_factor = self.fine_range_factor.value(),
            z_range_mm        = self.z_range.value(),
            trend_n           = int(self.trend_n.value()),
            n_avg             = int(self.n_avg.value()),
        )
        if 'last_focus_z' in existing:
            self._user_defaults[mag]['last_focus_z'] = existing['last_focus_z']
        self._save_defaults_file()
        self.status.setText(f"Saved defaults for {mag:g}×")

    def showEvent(self, event):
        super().showEvent(event)
        mag = self._current_mag()
        self._mag_label.setText(f"Mag: {mag:g}×" if mag > 0 else "Mag: —")

    def _dspin(self, mn, mx, val, decs):
        s = QDoubleSpinBox()
        s.setRange(mn, mx)
        s.setDecimals(decs)
        s.setValue(val)
        s.setSingleStep(10 ** -decs if decs > 0 else 1)
        s.setMinimumWidth(100)
        return s

    def _start(self):
        if self.thread is not None:
            return

        z_cfg     = self.mm.step_config.get('Z', {})
        z_step_mm = z_cfg.get('step', 0.000025)

        z0 = self.mm.get_position_units('Z') or 0.0
        self._start_z = z0

        # Z is a relative axis — step_config min/max are meaningless here.
        # Use a generous ±margin so _move() clamping never clips the scan;
        # z_range_mm already limits how far the search actually travels.
        z_range_budget = self.z_range.value() * 3
        z_min = z0 - z_range_budget
        z_max = z0 + z_range_budget

        mag = self._current_mag()
        last_focus_z = self._defaults_for_mag(mag).get('last_focus_z') if mag > 0 else None

        p = {
            'metric':        self.metric.currentText(),
            'step_mm':       self.step.value(),
            'settle_ms':     int(self.settle.value()),
            'coarse_factor':     int(self.coarse_factor.value()),
            'fine_factor':       self.fine_factor.value(),
            'fine_range_factor':   self.fine_range_factor.value(),
            'z_range_mm':    self.z_range.value(),
            'trend_n':  int(self.trend_n.value()),
            'n_avg':    int(self.n_avg.value()),
            'last_focus_z':  last_focus_z,
            'z_min_mm':      z_min,
            'z_max_mm':      z_max,
            'z_step_mm':     z_step_mm,
        }

        get_frame = getattr(self.preview, "get_frame", None)
        if get_frame is None:
            self.status.setText("Preview has no get_frame().")
            return

        self.thread = QThread()
        self.worker = FocusWorker(self.mm, get_frame, p)
        self.worker.moveToThread(self.thread)
        # Let Qt delete the C++ worker object from the correct thread context;
        # avoids "moveToThread: current thread is not the object's thread" warning
        # that fires when Python's GC deletes it from the main thread.
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.aborted.connect(self._on_aborted)

        # Pause Z velocity jogging so it doesn't fight the autofocus sweep
        self._z_vel_timer_was_active = False
        if self.stage_controls is not None:
            t = getattr(self.stage_controls, '_z_vel_timer', None)
            if t and t.isActive():
                t.stop()
                self._z_vel_timer_was_active = True

        # Live-plot dialog
        self._progress_dlg = AutoFocusProgressDialog(self)
        self._progress_dlg.stop_requested.connect(self._stop_worker)
        self.worker.progress.connect(self._progress_dlg.on_progress)
        self.worker.finished.connect(self._progress_dlg.on_finished)
        self.worker.aborted.connect(self._progress_dlg.on_aborted)

        self.thread.start()
        self.status.setText("Running...")
        self._progress_dlg.show()

    def _stop_worker(self):
        if self.worker:
            self.worker.stop()

    def _on_progress(self, z, m, zb):
        self.status.setText(f"z={z:.4f} mm  metric={m:.1f}  best={zb:.4f} mm")

    def _cleanup(self):
        if self.worker:
            self.worker.stop()
        if self.thread:
            self.thread.quit()
            self.thread.wait()
        # Drop Python references; C++ deletion is handled by the
        # thread.finished → worker.deleteLater connection set up in _start.
        self.thread = None
        self.worker = None
        # Resume Z velocity jogging now that autofocus is done
        if self.stage_controls is not None and getattr(self, '_z_vel_timer_was_active', False):
            t = getattr(self.stage_controls, '_z_vel_timer', None)
            if t:
                t.start(50)

    def _on_finished(self, z_best, m_best):
        self.status.setText(f"Done. z={z_best:.4f} mm  metric={m_best:.1f}")
        # Persist the best Z as the starting-direction hint for next time.
        mag = self._current_mag()
        if mag > 0:
            if mag not in self._user_defaults:
                self._user_defaults[mag] = {}
            self._user_defaults[mag]['last_focus_z'] = z_best
            self._save_defaults_file()
        self._cleanup()
        # Sync the Z slider so the next manual jog starts from the correct position.
        if self.stage_controls is not None:
            self.stage_controls._sync_sliders_to_motors()

    def _on_aborted(self, reason):
        start_z = getattr(self, '_start_z', None)
        self._cleanup()
        if "Stopped by user" in reason and start_z is not None:
            try:
                self.mm.move_absolute_units('Z', start_z)
                if self.stage_controls is not None:
                    self.stage_controls._sync_sliders_to_motors()
                reason = f"{reason} — returned to z={start_z:.4f}"
            except Exception as exc:
                print(f"[AutoFocus] Could not restore Z: {exc}")
        self.status.setText(f"Aborted: {reason}")

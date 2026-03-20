# ui/flat_field_panel.py
"""
Flat-field correction panel.

Because no clear area of the wafer exists, the background is assembled
from frames taken at many random stage positions.  The stage moves
automatically, waits 50 ms for vibration to settle, then captures a frame.
At any given pixel, sample features appear in fewer than half the frames,
so the per-pixel median of the stack recovers the clean background —
vignetting, illumination gradient, and fixed dust spots included.

After collection the stage returns to the original position.

Rule of thumb: collect at least 2× as many frames as the fraction of the
field covered by sample.  e.g. 50 % coverage → ≥ 100 frames.
"""

import threading
import time
import datetime
import random

import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QCheckBox, QProgressBar, QSlider, QGroupBox,
)

_SETTLE_S   = 0.05   # seconds to wait after each move before capturing


class FlatFieldPanel(QWidget):
    _progress_sig = pyqtSignal(int, int)   # (current, total)
    _build_done   = pyqtSignal(object)     # raw median float32 BGR, or None

    def __init__(self, preview, motor_manager, parent=None):
        super().__init__(parent)
        self.preview       = preview
        self.motor_manager = motor_manager
        self.setWindowTitle("Flat-Field Correction")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        self._flat_raw  = None
        self._flat      = None
        self._running   = False

        # ── Collection group ──────────────────────────────────────────────
        col_group = QGroupBox("Automatic background collection")
        col_grid  = QGridLayout()

        col_grid.addWidget(QLabel("Positions:"), 0, 0)
        self._n_spin = QSpinBox()
        self._n_spin.setRange(10, 500)
        self._n_spin.setValue(50)
        self._n_spin.setToolTip(
            "Number of random stage positions to visit.\n"
            "Use ≥ 2× the fraction of the field covered by sample\n"
            "(e.g. 50 % coverage → ≥ 100 positions).")
        col_grid.addWidget(self._n_spin, 0, 1)

        col_grid.addWidget(QLabel("Range (mm):"), 1, 0)
        self._range_spin = QDoubleSpinBox()
        self._range_spin.setRange(0.05, 10.0)
        self._range_spin.setSingleStep(0.1)
        self._range_spin.setDecimals(2)
        self._range_spin.setValue(0.5)
        self._range_spin.setToolTip(
            "Stage will move randomly within ±this distance (mm)\n"
            "from the starting position in X and Y.")
        col_grid.addWidget(self._range_spin, 1, 1)

        self._collect_btn = QPushButton("Collect && Build Background")
        self._collect_btn.clicked.connect(self._start_collection)
        col_grid.addWidget(self._collect_btn, 2, 0, 1, 2)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.clicked.connect(self._stop_collection)
        self._stop_btn.setEnabled(False)
        col_grid.addWidget(self._stop_btn, 3, 0, 1, 2)

        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFormat("Idle")
        col_grid.addWidget(self._progress, 4, 0, 1, 2)

        self._status = QLabel("No background built yet.")
        self._status.setWordWrap(True)
        col_grid.addWidget(self._status, 5, 0, 1, 2)

        col_group.setLayout(col_grid)

        # ── Correction group ──────────────────────────────────────────────
        cor_group = QGroupBox("Correction")
        cor_grid  = QGridLayout()

        self._enable_chk = QCheckBox("Apply correction to preview")
        self._enable_chk.setChecked(False)
        self._enable_chk.setEnabled(False)
        cor_grid.addWidget(self._enable_chk, 0, 0, 1, 3)

        cor_grid.addWidget(QLabel("Smoothing σ:"), 1, 0)
        self._sigma_slider = QSlider(Qt.Horizontal)
        self._sigma_slider.setRange(0, 30)
        self._sigma_slider.setValue(3)
        self._sigma_slider.setToolTip(
            "Gaussian blur applied to the flat field before division.\n"
            "σ=0–5: smooths pixel noise, dust cancels cleanly.\n"
            "σ>10: smears dust spots → ring artefacts around them.")
        self._sigma_slider.valueChanged.connect(self._on_sigma_changed)
        cor_grid.addWidget(self._sigma_slider, 1, 1)
        self._sigma_label = QLabel("3")
        self._sigma_label.setFixedWidth(24)
        cor_grid.addWidget(self._sigma_label, 1, 2)

        cor_group.setLayout(cor_grid)

        # ── Layout ────────────────────────────────────────────────────────
        layout = QVBoxLayout()
        layout.addWidget(col_group)
        layout.addWidget(cor_group)
        self.setLayout(layout)

        self._progress_sig.connect(self._on_progress)
        self._build_done.connect(self._on_build_done)
        preview.flat_field_panel = self

    # ── Collection ────────────────────────────────────────────────────────

    def _start_collection(self):
        if self._running:
            return
        self._running = True
        self._collect_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._enable_chk.setChecked(False)
        self._enable_chk.setEnabled(False)
        n     = self._n_spin.value()
        rng   = self._range_spin.value()
        self._progress.setRange(0, n)
        self._progress.setValue(0)
        self._progress.setFormat(f"0 / {n}")
        self._status.setText("Moving to starting position…")
        threading.Thread(
            target=self._collect_worker,
            args=(n, rng),
            daemon=True,
        ).start()

    def _stop_collection(self):
        self._running = False   # worker checks this flag each iteration

    def _collect_worker(self, n, rng_mm):
        mm = self.motor_manager
        frames = []

        # Record start position so we can return afterward.
        # get_position_units may return None if the stage is in mock mode.
        x0 = mm.get_position_units('X') or 0.0
        y0 = mm.get_position_units('Y') or 0.0

        # Track position ourselves — don't rely on get_position_units accuracy
        cur_x, cur_y = x0, y0

        def _move_rel(dx, dy):
            if hasattr(mm, 'move_relative_xy_units'):
                mm.move_relative_xy_units(dx, dy)
            else:
                mm.move_units('X', dx)
                mm.move_units('Y', dy)

        try:
            for i in range(n):
                if not self._running:
                    break

                # Move to a random offset anchored to the start position
                tx = x0 + random.uniform(-rng_mm, rng_mm)
                ty = y0 + random.uniform(-rng_mm, rng_mm)
                _move_rel(tx - cur_x, ty - cur_y)
                cur_x, cur_y = tx, ty

                # Wait for stage to settle, then grab a fresh frame directly
                # from the camera (bypasses the preview timer so the image is
                # guaranteed to be from the new position)
                time.sleep(_SETTLE_S)
                ret, frame = self.preview.cap.read()
                if ret:
                    frames.append(frame)

                self._progress_sig.emit(i + 1, n)

        finally:
            # Always return to the starting position
            _move_rel(x0 - cur_x, y0 - cur_y)

        if not frames:
            self._build_done.emit(None)
            return

        # Compute per-pixel median — sample features appear at different
        # pixel locations in each frame, so the median is the clean background
        self._progress_sig.emit(-1, n)   # signal "computing"
        stack  = np.stack([f.astype(np.float32) for f in frames], axis=0)
        median = np.median(stack, axis=0)
        self._build_done.emit(median)

    def _on_progress(self, current, total):
        if current == -1:
            self._progress.setRange(0, 0)
            self._progress.setFormat("Computing median…")
        else:
            self._progress.setRange(0, total)
            self._progress.setValue(current)
            self._progress.setFormat(f"{current} / {total}")
            self._status.setText(f"Collecting… {current}/{total} frames")

    def _on_build_done(self, median):
        self._running = False
        self._collect_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        if median is None:
            self._status.setText("Collection failed — no frames captured.")
            self._progress.setFormat("Failed")
            return
        self._flat_raw = median
        self._rebuild_flat()
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        n  = int(self._n_spin.value())
        self._status.setText(
            f"Background built at {ts} from {n} frames.\n"
            f"Stage returned to starting position.")
        self._progress.setFormat("Done")
        self._enable_chk.setEnabled(True)
        self._enable_chk.setChecked(True)

    # ── Smoothing ─────────────────────────────────────────────────────────

    def _on_sigma_changed(self, value):
        self._sigma_label.setText(str(value))
        self._rebuild_flat()

    def _rebuild_flat(self):
        if self._flat_raw is None:
            return
        sigma = self._sigma_slider.value()
        if sigma > 0:
            blurred = cv2.GaussianBlur(self._flat_raw, (0, 0), sigma)
        else:
            blurred = self._flat_raw.copy()
        self._flat = np.clip(blurred, 1.0, None)

    # ── Per-frame correction (called from preview.update_frame) ───────────

    def apply_correction(self, frame):
        """
        Divide frame by the flat field and stretch contrast per channel.
        Returns the corrected uint8 BGR frame, or the original if
        correction is disabled or no background has been built.
        """
        if not self._enable_chk.isChecked() or self._flat is None:
            return frame

        flat = self._flat
        if flat.shape[:2] != frame.shape[:2]:
            flat = cv2.resize(flat, (frame.shape[1], frame.shape[0]),
                              interpolation=cv2.INTER_LINEAR)

        mean_val = flat.mean()
        corrected = frame.astype(np.float32) / flat * mean_val

        # Stretch using a single shared scale across all channels so that
        # colour balance is preserved.  Per-channel independent stretching
        # would amplify any residual colour cast in low-signal channels.
        lo = corrected.min()
        hi = corrected.max()
        if hi > lo:
            corrected = (corrected - lo) / (hi - lo) * 255.0

        return np.clip(corrected, 0, 255).astype(np.uint8)

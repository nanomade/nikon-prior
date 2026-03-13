# ui/index_mark_panel.py
"""
Index mark navigator.

Index marks are cross (+) shapes etched into the substrate with XX,YY
coordinate labels.  They are spaced every 500 µm and count up to the
right (+X) and up (+Y).

The panel lets the user:
  1. Set the current grid position (manually, or via OCR if tesseract is
     installed: sudo apt install tesseract-ocr && pip install pytesseract).
  2. Set the rotation angle of the sample grid relative to the stage axes
     (auto-detected from visible cross positions, or entered manually).
  3. Enter a target grid position and move the stage there.

Navigation applies a 2-D rotation so diagonal samples work correctly:
  stage_dx = (dXX·cosθ − dYY·sinθ) × 0.5 mm
  stage_dy = (dXX·sinθ + dYY·cosθ) × 0.5 mm
"""

import math
import re
import threading

import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QCheckBox, QGroupBox, QFrame,
)

try:
    import pytesseract
    _TESSERACT_OK = True
except ImportError:
    _TESSERACT_OK = False

_SPACING_MM = 0.5   # 500 µm grid spacing


# ---------------------------------------------------------------------------
# Vision helpers
# ---------------------------------------------------------------------------

def _mark_binary(frame):
    """
    Return a binary image highlighting index-mark pixels (bright on blue SiO2).
    White marks have high R and high G; the blue substrate has low R.
    Using R channel + CLAHE + Otsu gives the best mark/background separation.
    """
    r = frame[:, :, 2]                      # BGR → R channel
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    r = clahe.apply(r)
    _, thresh = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def _mark_clusters(thresh, img_w, img_h):
    """
    Return connected-component stats for bright clusters that look like
    index-mark digit groups (the 2×2 digit grid etched above each cross).

    Size heuristic: the digit cluster occupies roughly 2–15 % of the smaller
    image dimension in both width and height, and is nearly square.

    Returns list of stat rows for qualifying components.
    """
    # Merge individual digit pixels into one blob per mark
    # Use a kernel large enough to bridge the gap between the two digit rows
    # (top "59" and bottom "55" separated by a thin gap).
    merge_k = max(5, img_w // 60)
    merged = cv2.dilate(thresh, np.ones((merge_k, merge_k), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(merged)

    short = min(img_w, img_h)
    lo = short * 0.02   # component must be at least 2 % of image
    hi = short * 0.25   # and at most 25 %

    good = []
    for i in range(1, n):
        cw  = stats[i, cv2.CC_STAT_WIDTH]
        ch_ = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if cw < lo or ch_ < lo or cw > hi or ch_ > hi:
            continue
        aspect = max(cw, ch_) / max(min(cw, ch_), 1)
        if aspect > 3.5:
            continue       # too elongated — debris streak or flake
        # Require a minimum fill fraction so scattered dust doesn't qualify
        if area < cw * ch_ * 0.05:
            continue
        good.append(stats[i])
    return good


def _find_cross_centers(frame_or_gray, min_arm=12):
    """
    Detect index-mark centres in an image.

    For BGR input, thresholds on the R channel and finds compact, nearly-square
    clusters that match the 2×2 digit label above each cross mark.
    Falls back to a morphological line-intersection approach for grey input.

    Returns a list of (x, y) pixel centres.
    """
    if frame_or_gray.ndim == 3:
        h, w = frame_or_gray.shape[:2]
        thresh = _mark_binary(frame_or_gray)
        clusters = _mark_clusters(thresh, w, h)
        centers = []
        for st in clusters:
            cx = int(st[cv2.CC_STAT_LEFT] + st[cv2.CC_STAT_WIDTH]  / 2)
            cy = int(st[cv2.CC_STAT_TOP]  + st[cv2.CC_STAT_HEIGHT] / 2)
            if all(abs(cx - ex) > 20 or abs(cy - ey) > 20
                   for ex, ey in centers):
                centers.append((cx, cy))
        if centers:
            return centers

    # ── Grey fallback: morphological line-intersection ───────────────────
    gray = frame_or_gray if frame_or_gray.ndim == 2 else \
           cv2.cvtColor(frame_or_gray, cv2.COLOR_BGR2GRAY)
    centers = []
    for inv in (False, True):
        src = cv2.bitwise_not(gray) if inv else gray
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        src = clahe.apply(src)
        _, thresh = cv2.threshold(src, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        h_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (min_arm, 1))
        v_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_arm))
        cross = cv2.bitwise_and(
            cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kern),
            cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kern))
        cross = cv2.dilate(cross, np.ones((7, 7), np.uint8))
        cnts, _ = cv2.findContours(cross, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < 20:
                continue
            M = cv2.moments(c)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                if all(abs(cx - ex) > 20 or abs(cy - ey) > 20
                       for ex, ey in centers):
                    centers.append((cx, cy))
    return centers


def _estimate_rotation_from_centers(centers):
    """
    Given ≥2 cross centres, estimate the grid rotation angle (degrees).

    The marks lie on a square grid, so inter-mark directions cluster at
    θ, θ+90°, θ+180°, θ+270°.  We fold all angles into [0°, 90°) and
    return the median as the rotation estimate.
    Returns None if fewer than 2 centres are provided.
    """
    if len(centers) < 2:
        return None
    angles = []
    pts = np.array(centers, dtype=float)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dx, dy = pts[j] - pts[i]
            # Image Y increases downward; negate dy to get maths convention
            angle = math.degrees(math.atan2(-dy, dx)) % 90.0
            angles.append(angle)
    return float(np.median(angles))


# Char substitutions: visually similar letter → digit (no whitelist mode)
_OCR_CHARMAP = {
    'q': '9', 'g': '9',
    'o': '0', 'O': '0', 'Q': '0', 'D': '0',
    'l': '1', 'I': '1', 'i': '1', '|': '1',
    's': '5', 'S': '5',
    'b': '6', 'B': '8', 'Z': '2', 'z': '2',
}


def _ocr_single_digit(crop_bin):
    """
    OCR one digit crop (bright mark on black background).
    Returns a single digit string or None.
    """
    inv = cv2.bitwise_not(crop_bin)
    scale = max(3, 150 // max(1, crop_bin.shape[0]))
    up = cv2.resize(inv, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_NEAREST)
    up = cv2.copyMakeBorder(up, 20, 20, 20, 20,
                            cv2.BORDER_CONSTANT, value=255)
    for psm in (10, 8):
        cfg = f'--psm {psm} --oem 1'
        text = pytesseract.image_to_string(up, config=cfg).strip().replace(' ', '')
        mapped = ''.join(_OCR_CHARMAP.get(c, c) for c in text)
        digits = re.findall(r'\d', mapped)
        if digits:
            return digits[0]
    return None


def _cross_center_in_rotated(rot_bin):
    """
    Find the cross (+) centre in a rotation-corrected binary crop.
    Uses long-span morphological OPEN (32 % of image dimension) so only
    the cross arms (which span the mark width/height) survive — digit
    segments are too short.
    Returns (col, row) in the rotated frame, or centre of image as fallback.
    """
    h, w = rot_bin.shape
    hs = int(w * 0.32); vs = int(h * 0.32)
    h_arm = cv2.morphologyEx(rot_bin, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (hs, 1)))
    v_arm = cv2.morphologyEx(rot_bin, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (1, vs)))
    cross_px = cv2.dilate(cv2.bitwise_and(h_arm, v_arm),
                          np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(cross_px)
    if n > 1:
        best = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA])
        return (int(stats[best, cv2.CC_STAT_LEFT] + stats[best, cv2.CC_STAT_WIDTH]  // 2),
                int(stats[best, cv2.CC_STAT_TOP]  + stats[best, cv2.CC_STAT_HEIGHT] // 2))
    return w // 2, h // 2


# Fixed box offsets (relative to cross centre, in rotated frame).
# Calibrated at 10× magnification on 850×896 px frames.
# Units: pixels at the calibration resolution.
_BOX_OFFSETS = {
    'TL': (-46.5, -67.2, -8.2,  -7.8),
    'TR': (  8.2, -68.0, 45.8,  -8.8),
    'BL': (-47.2,   9.0, -6.5,  69.5),
    'BR': (  8.5,   8.0, 47.2,  69.0),
}


def _ocr_grid_coords(frame, rotation_deg=0.0):
    """
    Read XX,YY grid coordinates from a frame containing an index mark.

    Strategy:
      1. Threshold on R channel to isolate the bright mark.
      2. Find mark clusters via _mark_clusters().
      3. For each cluster:
           a. Rotate the crop by -rotation_deg around its centre.
           b. Detect the cross (+) centre with long-span morphology.
           c. Extract the 4 digit boxes at fixed offsets from the cross.
           d. OCR each digit individually (PSM 10, char-map substitution).
      4. Return list of (xx, yy) tuples.

    Returns list of (xx, yy) int tuples (may be empty).
    """
    if not _TESSERACT_OK:
        return []

    h, w = frame.shape[:2]
    thresh = _mark_binary(frame)
    clusters = _mark_clusters(thresh, w, h)

    # Scale box offsets to match current frame resolution relative to
    # calibration resolution (850 px wide).
    scale_factor = w / 850.0

    results = []
    pad = 8
    for st in clusters:
        x0 = max(0, st[cv2.CC_STAT_LEFT]  - pad)
        y0 = max(0, st[cv2.CC_STAT_TOP]   - pad)
        x1 = min(w, x0 + st[cv2.CC_STAT_WIDTH]  + pad * 2)
        y1 = min(h, y0 + st[cv2.CC_STAT_HEIGHT] + pad * 2)
        crop = thresh[y0:y1, x0:x1]
        ch, cw = crop.shape

        # Rotate around crop centre
        M = cv2.getRotationMatrix2D((cw / 2, ch / 2), -rotation_deg, 1.0)
        rot = cv2.warpAffine(crop, M, (cw, ch),
                             flags=cv2.INTER_NEAREST, borderValue=0)

        ccx, ccy = _cross_center_in_rotated(rot)

        # Extract and OCR each digit box
        digits = {}
        for name, (ox0, oy0, ox1, oy1) in _BOX_OFFSETS.items():
            bx0 = max(0, int(ccx + ox0 * scale_factor))
            by0 = max(0, int(ccy + oy0 * scale_factor))
            bx1 = min(cw, int(ccx + ox1 * scale_factor))
            by1 = min(ch, int(ccy + oy1 * scale_factor))
            if bx1 <= bx0 or by1 <= by0:
                continue
            d = _ocr_single_digit(rot[by0:by1, bx0:bx1])
            if d:
                digits[name] = d

        if all(k in digits for k in _BOX_OFFSETS):
            xx = int(digits['TL'] + digits['TR'])
            yy = int(digits['BL'] + digits['BR'])
            results.append((xx, yy))

    return results


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class IndexMarkPanel(QWidget):
    _ocr_done = pyqtSignal(list)   # emitted from worker thread with results

    def __init__(self, preview, motor_manager, stage_controls, parent=None):
        super().__init__(parent)
        self.preview        = preview
        self.motor_manager  = motor_manager
        self.stage_controls = stage_controls
        self.setWindowTitle("Index Mark Navigator")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        self._show_marks = False

        layout = QVBoxLayout()

        # ── Current position ─────────────────────────────────────────────
        cur_box = QGroupBox("Current position (read from image)")
        cur_grid = QGridLayout()

        cur_grid.addWidget(QLabel("X:"), 0, 0)
        self._cur_x = QSpinBox(); self._cur_x.setRange(-999, 999)
        cur_grid.addWidget(self._cur_x, 0, 1)

        cur_grid.addWidget(QLabel("Y:"), 0, 2)
        self._cur_y = QSpinBox(); self._cur_y.setRange(-999, 999)
        cur_grid.addWidget(self._cur_y, 0, 3)

        detect_btn = QPushButton("Auto-detect (OCR) — coming soon")
        detect_btn.setToolTip("OCR detection is temporarily disabled.")
        detect_btn.setEnabled(False)
        cur_grid.addWidget(detect_btn, 1, 0, 1, 4)

        self._detect_status = QLabel("")
        self._detect_status.setWordWrap(True)
        cur_grid.addWidget(self._detect_status, 2, 0, 1, 4)

        cur_box.setLayout(cur_grid)
        layout.addWidget(cur_box)

        # ── Sample rotation ──────────────────────────────────────────────
        rot_box = QGroupBox("Sample rotation")
        rot_grid = QGridLayout()

        rot_grid.addWidget(QLabel("Angle (°):"), 0, 0)
        self._rotation = QDoubleSpinBox()
        self._rotation.setRange(-180.0, 180.0)
        self._rotation.setDecimals(2)
        self._rotation.setSingleStep(0.5)
        self._rotation.setValue(0.0)
        self._rotation.setToolTip(
            "Angle of sample XX axis relative to stage X axis, CCW positive.\n"
            "Use 'Detect from marks' to estimate automatically.")
        rot_grid.addWidget(self._rotation, 0, 1)

        detect_rot_btn = QPushButton("Detect from marks")
        detect_rot_btn.setToolTip(
            "Estimates rotation from the geometry of detected cross marks.\n"
            "Enable 'Highlight cross marks' first to confirm marks are found.")
        detect_rot_btn.clicked.connect(self._detect_rotation)
        rot_grid.addWidget(detect_rot_btn, 0, 2)

        self._rot_status = QLabel("")
        rot_grid.addWidget(self._rot_status, 1, 0, 1, 3)

        rot_box.setLayout(rot_grid)
        layout.addWidget(rot_box)

        # ── Target position ──────────────────────────────────────────────
        tgt_box = QGroupBox("Target position")
        tgt_grid = QGridLayout()

        tgt_grid.addWidget(QLabel("X:"), 0, 0)
        self._tgt_x = QSpinBox(); self._tgt_x.setRange(-999, 999)
        tgt_grid.addWidget(self._tgt_x, 0, 1)

        tgt_grid.addWidget(QLabel("Y:"), 0, 2)
        self._tgt_y = QSpinBox(); self._tgt_y.setRange(-999, 999)
        tgt_grid.addWidget(self._tgt_y, 0, 3)

        go_btn = QPushButton("Go To")
        go_btn.clicked.connect(self._go_to)
        tgt_grid.addWidget(go_btn, 1, 0, 1, 4)

        tgt_box.setLayout(tgt_grid)
        layout.addWidget(tgt_box)

        # ── Show marks toggle ────────────────────────────────────────────
        self._marks_check = QCheckBox("Highlight cross marks in preview")
        self._marks_check.setChecked(False)
        self._marks_check.stateChanged.connect(
            lambda s: setattr(self, '_show_marks', s == Qt.Checked))
        layout.addWidget(self._marks_check)

        # ── Status ───────────────────────────────────────────────────────
        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        layout.addWidget(sep)
        self._status = QLabel("Grid spacing: 500 µm")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._ocr_done.connect(self._on_ocr_done)
        self._ocr_running = False

        self.setLayout(layout)
        preview.mark_panel = self

    # ── OCR detection ────────────────────────────────────────────────────

    def _detect_ocr(self):
        if self._ocr_running:
            return
        frame = self.preview._last_raw_frame
        if frame is None:
            self._detect_status.setText("No frame available yet.")
            return
        self._ocr_running = True
        self._detect_status.setText("Running OCR…")
        rotation = self._rotation.value()
        # Snapshot the frame so the worker has its own copy
        frame_copy = frame.copy()
        threading.Thread(
            target=self._ocr_worker,
            args=(frame_copy, rotation),
            daemon=True,
        ).start()

    def _ocr_worker(self, frame, rotation):
        try:
            pairs = _ocr_grid_coords(frame, rotation_deg=rotation)
        except Exception:
            pairs = []
        self._ocr_done.emit(pairs)

    def _on_ocr_done(self, pairs):
        self._ocr_running = False
        if pairs:
            xx, yy = pairs[0]
            self._cur_x.setValue(xx)
            self._cur_y.setValue(yy)
            extra = f" (+{len(pairs)-1} more)" if len(pairs) > 1 else ""
            self._detect_status.setText(f"Detected: {xx}, {yy}{extra}")
        else:
            self._detect_status.setText(
                "No XX,YY pattern found — try adjusting zoom or enter manually.")

    # ── Rotation detection ───────────────────────────────────────────────

    def _detect_rotation(self):
        frame = self.preview._last_raw_frame
        if frame is None:
            self._rot_status.setText("No frame available.")
            return
        min_arm = max(8, frame.shape[1] // 120)
        centers = _find_cross_centers(frame, min_arm=min_arm)
        if len(centers) < 2:
            self._rot_status.setText(
                f"Found {len(centers)} mark(s) — need ≥2 to estimate rotation.")
            return
        angle = _estimate_rotation_from_centers(centers)
        self._rotation.setValue(round(angle, 2))
        self._rot_status.setText(
            f"Estimated {angle:.2f}° from {len(centers)} marks — verify visually.")

    # ── Navigation ───────────────────────────────────────────────────────

    def _go_to(self):
        dx_grid = self._tgt_x.value() - self._cur_x.value()
        dy_grid = self._tgt_y.value() - self._cur_y.value()

        theta  = math.radians(self._rotation.value())
        cos_t  = math.cos(theta)
        sin_t  = math.sin(theta)

        dx_mm = (dx_grid * cos_t - dy_grid * sin_t) * _SPACING_MM
        dy_mm = (dx_grid * sin_t + dy_grid * cos_t) * _SPACING_MM

        self.motor_manager.move_units('X', dx_mm)
        self.motor_manager.move_units('Y', dy_mm)

        # Sync stage sliders
        if self.stage_controls is not None:
            cfg   = self.motor_manager.step_config
            stepx = cfg.get('X', {}).get('step', 0.005)
            stepy = cfg.get('Y', {}).get('step', 0.005)
            sc = self.stage_controls
            sc.stage_x_slider.blockSignals(True)
            sc.stage_y_slider.blockSignals(True)
            sc.stage_x_slider.setValue(
                sc.stage_x_slider.value() + round(dx_mm / stepx))
            sc.stage_y_slider.setValue(
                sc.stage_y_slider.value() + round(dy_mm / stepy))
            sc.stage_x_slider.blockSignals(False)
            sc.stage_y_slider.blockSignals(False)

        self._cur_x.setValue(self._tgt_x.value())
        self._cur_y.setValue(self._tgt_y.value())

        self._status.setText(
            f"Moved  ΔX={dx_mm:+.3f} mm  ΔY={dy_mm:+.3f} mm  "
            f"(grid Δ{dx_grid:+d}, {dy_grid:+d})  θ={self._rotation.value():.1f}°")

    # ── Preview overlay ──────────────────────────────────────────────────

    def apply_overlay(self, display):
        """Highlight detected cross marks in the preview."""
        if not self._show_marks:
            return display
        min_arm = max(8, display.shape[1] // 120)
        centers = _find_cross_centers(display, min_arm=min_arm)
        if not centers:
            return display
        result = display.copy()
        for cx, cy in centers:
            cv2.drawMarker(result, (cx, cy), (0, 200, 255),
                           markerType=cv2.MARKER_CROSS,
                           markerSize=24, thickness=1)
            cv2.circle(result, (cx, cy), 14, (0, 200, 255), 1)
        # Draw a small arrow indicating the detected +X grid direction
        if len(centers) >= 2:
            theta = math.radians(self._rotation.value())
            # Arrow from image centre in the +XX direction
            h, w = result.shape[:2]
            cx0, cy0 = w // 2, h // 2
            length = min(w, h) // 8
            ex = int(cx0 + length * math.cos(theta))
            ey = int(cy0 - length * math.sin(theta))   # screen Y inverted
            cv2.arrowedLine(result, (cx0, cy0), (ex, ey),
                            (0, 200, 255), 1, tipLength=0.2)
            cv2.putText(result, "+XX", (ex + 4, ey - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
        return result

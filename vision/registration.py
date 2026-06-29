"""
Placement registration: match live camera frames against saved corner reference
images to estimate the rigid-body shift between the original scan placement
and the current stage placement.

Match algorithm mirrors the timelapse drift corrector (ORB + PC fallback).
All functions are pure (no Qt) and can be called from a worker thread.
"""

import json
import math
import pathlib

import cv2
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

_ORB_MIN_INLIERS  = 6
_RANSAC_THR_PX    = 3.0
_PC_MIN_CONF      = 0.05
_MAX_ROTATION_DEG = 20.0   # above this → ask user to remount


# ── Corner extraction ─────────────────────────────────────────────────────────

def extract_scan_corners(scan_folder: pathlib.Path,
                         output_dir: pathlib.Path) -> list[dict]:
    """
    Find the 4 extreme tiles of a scan (NW/NE/SW/SE corners of the scan grid)
    and save them as JPEG reference crops in output_dir.

    Returns a list of 4 dicts:
        {'label': 'NW', 'x_mm': float, 'y_mm': float,
         'image_path': str, 'scan_folder': str}

    Raises if scan_metadata.json is missing or has fewer than 4 images.
    """
    meta_path = pathlib.Path(scan_folder) / 'scan_metadata.json'
    meta = json.loads(meta_path.read_text())
    images = meta.get('images', [])
    if len(images) < 4:
        raise ValueError(f"Only {len(images)} images in scan — need at least 4")

    # Find extremes in x_mm / y_mm space
    xs = np.array([im['x_mm'] for im in images])
    ys = np.array([im['y_mm'] for im in images])
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())

    # Score each tile by how close it is to each target corner
    corners_def = [
        ('NW', x_min, y_min),
        ('NE', x_max, y_min),
        ('SW', x_min, y_max),
        ('SE', x_max, y_max),
    ]

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    used_indices: set = set()
    for label, tx, ty in corners_def:
        dists = (xs - tx) ** 2 + (ys - ty) ** 2
        # Pick closest unused tile
        for idx in np.argsort(dists):
            if int(idx) not in used_indices:
                break
        used_indices.add(int(idx))
        im_meta = images[int(idx)]
        src = pathlib.Path(scan_folder) / im_meta['filename']
        dst = output_dir / f'corner_{label}.jpg'

        img = cv2.imread(str(src))
        if img is None:
            raise FileNotFoundError(f"Cannot read tile: {src}")
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 92])

        results.append({
            'label':       label,
            'x_mm':        float(im_meta['x_mm']),
            'y_mm':        float(im_meta['y_mm']),
            'z_mm':        float(im_meta['z_mm']) if 'z_mm' in im_meta else None,
            'image_path':  str(dst),
            'scan_folder': str(scan_folder),
        })

    return results


# ── Frame-to-reference matching ───────────────────────────────────────────────

_orb   = None
_bf    = None
_clahe = None

def _ensure_cv2_objects():
    global _orb, _bf, _clahe
    if _orb is None:
        _orb   = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=4)
        _bf    = cv2.BFMatcher(cv2.NORM_HAMMING)
        _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def match_frame_to_reference(live_frame: np.ndarray,
                              ref_image:  np.ndarray,
                              max_shift_px: float = 400.0
                              ) -> tuple[float, float, float, float, int]:
    """
    Estimate the pixel displacement and rotation of live_frame vs ref_image.

    Returns (dx_px, dy_px, rotation_deg, confidence, n_inliers) where:
        n_inliers > 0  → ORB result; rotation_deg from inlier keypoint angles
        n_inliers == -1 → phase-correlation result; rotation_deg = 0.0
        n_inliers == 0  → no reliable match found

    dx/dy: positive means content shifted right/down in live vs reference.
    rotation_deg: CW rotation of the scene in the live image relative to the
        reference. OpenCV ORB angles are CW-positive in image coordinates
        (0°=right, 90°=down), so angle_live − angle_ref > 0 for a CW image rotation.
        Derived from the median difference of matched ORB keypoint orientations.
    """
    _ensure_cv2_objects()

    # Resize live to match ref if needed
    rh, rw = ref_image.shape[:2]
    lh, lw = live_frame.shape[:2]
    if (lh, lw) != (rh, rw):
        live_frame = cv2.resize(live_frame, (rw, rh), interpolation=cv2.INTER_AREA)

    ref_g  = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY) if ref_image.ndim == 3 else ref_image.copy()
    live_g = cv2.cvtColor(live_frame, cv2.COLOR_BGR2GRAY) if live_frame.ndim == 3 else live_frame.copy()

    eq_ref  = _clahe.apply(ref_g)
    eq_live = _clahe.apply(live_g)

    kp_r, des_r = _orb.detectAndCompute(eq_ref,  None)
    kp_l, des_l = _orb.detectAndCompute(eq_live, None)

    if (des_r is not None and des_l is not None
            and len(kp_r) >= _ORB_MIN_INLIERS and len(kp_l) >= _ORB_MIN_INLIERS):
        raw = _bf.knnMatch(des_r, des_l, k=2)
        matches = [p[0] for p in raw
                   if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if len(matches) >= _ORB_MIN_INLIERS:
            dxs = np.array([kp_l[m.trainIdx].pt[0] - kp_r[m.queryIdx].pt[0] for m in matches])
            dys = np.array([kp_l[m.trainIdx].pt[1] - kp_r[m.queryIdx].pt[1] for m in matches])
            dx_med, dy_med = float(np.median(dxs)), float(np.median(dys))
            inliers = (np.abs(dxs - dx_med) < _RANSAC_THR_PX) & (np.abs(dys - dy_med) < _RANSAC_THR_PX)
            n = int(inliers.sum())
            if n >= _ORB_MIN_INLIERS:
                dx = float(dxs[inliers].mean())
                dy = float(dys[inliers].mean())
                if abs(dx) <= max_shift_px and abs(dy) <= max_shift_px:
                    # Rotation from inlier keypoint angle differences.
                    # ORB angles are CW-positive; median diff = CW rotation of the scene.
                    inlier_matches = [m for m, ok in zip(matches, inliers) if ok]
                    raw_diffs = [kp_l[m.trainIdx].angle - kp_r[m.queryIdx].angle
                                 for m in inlier_matches]
                    norm_diffs = [(d + 180.0) % 360.0 - 180.0 for d in raw_diffs]
                    rotation_deg = float(np.median(norm_diffs)) if norm_diffs else 0.0
                    return dx, dy, rotation_deg, 1.0, n

    # Phase-correlation fallback — no rotation estimate available
    h, w = ref_g.shape
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx_px, dy_px), response = cv2.phaseCorrelate(
        ref_g.astype(np.float32) * win,
        live_g.astype(np.float32) * win)
    if response >= _PC_MIN_CONF and abs(dx_px) <= max_shift_px and abs(dy_px) <= max_shift_px:
        return float(dx_px), float(dy_px), 0.0, float(response), -1

    return 0.0, 0.0, 0.0, 0.0, 0


# ── Transform fitting ─────────────────────────────────────────────────────────

def fit_placement_transform(observations: list[dict]) -> dict:
    """
    Fit a rigid-body placement transform from 2+ corner observations.

    Each observation dict must have:
        'ref_x_mm', 'ref_y_mm'  — where this corner was in the original scan
        'obs_x_mm', 'obs_y_mm'  — where it is now (stage position after matching)

    Returns a dict:
        {'dx_mm', 'dy_mm', 'rotation_deg', 'rms_mm', 'n_points', 'valid'}

    Raises ValueError if rotation > _MAX_ROTATION_DEG (ask user to remount).
    For n_observations == 1, rotation is assumed 0 (translation-only).
    """
    n = len(observations)
    if n < 1:
        raise ValueError("Need at least 1 observation")

    ref = np.array([[o['ref_x_mm'], o['ref_y_mm']] for o in observations])
    obs = np.array([[o['obs_x_mm'], o['obs_y_mm']] for o in observations])

    if n == 1:
        dx = float(obs[0, 0] - ref[0, 0])
        dy = float(obs[0, 1] - ref[0, 1])
        return {'dx_mm': dx, 'dy_mm': dy, 'rotation_deg': 0.0,
                'rms_mm': 0.0, 'n_points': 1, 'valid': True}

    # Procrustes: centre both clouds, SVD to find R
    ref_c = ref.mean(axis=0)
    obs_c = obs.mean(axis=0)
    A = ref - ref_c
    B = obs - obs_c
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Ensure proper rotation (det = +1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    angle_rad = math.atan2(R[1, 0], R[0, 0])
    rotation_deg = math.degrees(angle_rad)

    if abs(rotation_deg) > _MAX_ROTATION_DEG:
        raise ValueError(
            f"Rotation {rotation_deg:.1f}° exceeds {_MAX_ROTATION_DEG}° limit — "
            f"please remount the sample in the correct orientation")

    t = obs_c - R @ ref_c
    predicted = (R @ ref.T).T + t          # full-space predicted obs positions
    residuals = np.linalg.norm(obs - predicted, axis=1)
    rms = float(residuals.mean())

    return {
        'dx_mm':       float(t[0]),
        'dy_mm':       float(t[1]),
        'rotation_deg': rotation_deg,
        'rms_mm':       rms,
        'n_points':     n,
        'valid':        True,
    }


def apply_placement_transform(transform: dict,
                               x_mm: float, y_mm: float) -> tuple[float, float]:
    """
    Convert reference-stage coordinates to current-stage coordinates using the
    stored placement transform (translation + rotation from fit_placement_transform).
    """
    theta = math.radians(transform['rotation_deg'])
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    x_rot = cos_t * x_mm - sin_t * y_mm
    y_rot = sin_t * x_mm + cos_t * y_mm
    return x_rot + transform['dx_mm'], y_rot + transform['dy_mm']


def apply_inverse_placement_transform(transform: dict,
                                       x_mm: float, y_mm: float) -> tuple[float, float]:
    """Inverse of apply_placement_transform: current-stage → reference-stage."""
    theta = math.radians(transform['rotation_deg'])
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    dx, dy = x_mm - transform['dx_mm'], y_mm - transform['dy_mm']
    return cos_t * dx + sin_t * dy, -sin_t * dx + cos_t * dy


# ── Chip-local coordinate system ──────────────────────────────────────────────

def compute_chip_transform(corners: list[dict]) -> dict:
    """
    Derive a chip-local coordinate frame from saved corner stage positions.

    Two modes (auto-selected):

    * Compass (legacy, square chips): if NW/NE/SW/SE labels are present,
      origin = SW → chip (0,0); +X = SW→SE; +Y = SW→NW.  Backward-compatible
      with all existing samples.

    * Index-based (N corners, non-square chips, roadmap #25): origin = first
      corner (C1) → chip (0,0); +X = C1→C2 (normalised); +Y = +X rotated 90°
      CCW (right-handed orthonormal frame).  Any number of corners ≥ 2; extra
      corners (C3…CN) define the boundary (convex hull → extents) but not the
      frame.  width_mm/height_mm are the chip's projected bounding span.

    Returns a dict with 'origin_mm', 'x_axis', 'y_axis', 'width_mm', 'height_mm'.
    Raises ValueError if axes cannot be determined.
    """
    labels = {c.get('label') for c in corners}
    use_compass = ('SW' in labels and
                   ('SE' in labels or ('NE' in labels and 'NW' in labels)))

    if use_compass:
        by = {c['label']: np.array([c['x_mm'], c['y_mm']], float) for c in corners}
        sw = by['SW']
        x_vec = (by['SE'] - sw) if 'SE' in by else (by['NE'] - by['NW'])
        if 'NW' in by:
            y_vec = by['NW'] - sw
        elif 'NE' in by and 'SE' in by:
            y_vec = by['NE'] - by['SE']
        else:
            raise ValueError("Need NW corner (or NE+SE) to define chip Y axis")
        width_mm  = float(np.linalg.norm(x_vec))
        height_mm = float(np.linalg.norm(y_vec))
        if width_mm < 1e-6 or height_mm < 1e-6:
            raise ValueError("Corner positions too close together to define chip axes")
        origin = sw
        x_axis = x_vec / width_mm
        y_axis = y_vec / height_mm
    else:
        # Index-based: C1 = origin, C1→C2 = +X, +Y ⊥ X (90° CCW).
        if len(corners) < 2:
            raise ValueError("Need at least 2 corners (C1 origin, C2 +X direction)")
        pts = [np.array([c['x_mm'], c['y_mm']], float) for c in corners]
        origin = pts[0]
        x_vec = pts[1] - origin
        edge = float(np.linalg.norm(x_vec))
        if edge < 1e-6:
            raise ValueError("First two corners coincide — cannot define chip X axis")
        x_axis = x_vec / edge
        y_axis = np.array([-x_axis[1], x_axis[0]])     # +90° CCW → right-handed
        rel = np.array([p - origin for p in pts])
        xs, ys = rel @ x_axis, rel @ y_axis
        width_mm  = float(xs.max() - xs.min())
        height_mm = float(ys.max() - ys.min())

    return {
        'origin_mm':  [round(float(origin[0]), 6), round(float(origin[1]), 6)],
        'x_axis':     [round(float(x_axis[0]), 8), round(float(x_axis[1]), 8)],
        'y_axis':     [round(float(y_axis[0]), 8), round(float(y_axis[1]), 8)],
        'width_mm':   round(width_mm, 4),
        'height_mm':  round(height_mm, 4),
    }


def chip_to_reference_stage(chip_tf: dict,
                              cx_mm: float, cy_mm: float) -> tuple[float, float]:
    """Convert chip-local (cx, cy) mm to reference-stage (sx, sy) mm."""
    ox, oy = chip_tf['origin_mm']
    xx, xy = chip_tf['x_axis']
    yx, yy = chip_tf['y_axis']
    return ox + cx_mm * xx + cy_mm * yx, oy + cx_mm * xy + cy_mm * yy


def reference_stage_to_chip(chip_tf: dict,
                              sx_mm: float, sy_mm: float) -> tuple[float, float]:
    """Convert reference-stage (sx, sy) mm to chip-local (cx, cy) mm.

    Solves the linear system [dx; dy] = A @ [cx; cy] using the 2×2 matrix
    inverse, where A = [[x_axis[0], y_axis[0]], [x_axis[1], y_axis[1]]].
    This is the exact inverse of chip_to_reference_stage even when x_axis and
    y_axis are not orthogonal (which can happen if the stage XY axes are not
    perfectly perpendicular or the chip was placed at an angle).
    """
    ox, oy = chip_tf['origin_mm']
    xx, xy = chip_tf['x_axis']   # x_axis = [x-component, y-component]
    yx, yy = chip_tf['y_axis']   # y_axis = [x-component, y-component]
    dx, dy = sx_mm - ox, sy_mm - oy
    # A = [[xx, yx], [xy, yy]];  A^{-1} = 1/det * [[yy, -yx], [-xy, xx]]
    det = xx * yy - yx * xy
    return (yy * dx - yx * dy) / det, (-xy * dx + xx * dy) / det

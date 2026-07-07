"""Single accessor for the per-rig calibration.json (plan task L2).

Before this module, calibration.json was independently re-read in six places
(ui/preview, vision/flake_detect_v3, tools/calibrate_ladder,
tools/map_contrast_calibrate, tools/make_map ×2) and the camera-rotation
pixel↔stage math was hand-rolled per site. One loader, one transform,
mtime-cached so runtime writes from the calibration panel are picked up
without explicit reload calls.

Transform convention (inherited verbatim from ui/preview's rig-verified
click-to-move math): at θ=0, image-right = stage +X and image-DOWN = stage
−Y; the camera rotation θ (camera_rotation_deg, ~0.295°) is applied so a
pure stage-X move appears along image X only when θ=0. NOTE this is the
stage-MOVE/mark-position frame used by preview and the index-mark watcher —
the v3 detector's tile frame (image-row-down = stage-y-plus, CLAUDE.md #7
pipeline notes) is a different frame and does not use this function. The
pixel→stage matrix [[c, s], [s, −c]] is involutory (its own inverse), which
the round-trip test exploits.
"""
import json
import math
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
_CAL_PATH = APP_DIR / 'calibration.json'

_cache: dict = {}
_cache_mtime: float | None = None


def load_rig_calibration() -> dict:
    """Return the calibration.json dict ({} if missing/unreadable).

    Cached on file mtime — cheap to call per frame, refreshes automatically
    when the calibration panel writes the file.
    """
    global _cache, _cache_mtime
    try:
        mtime = _CAL_PATH.stat().st_mtime
    except OSError:
        _cache, _cache_mtime = {}, None
        return _cache
    if mtime != _cache_mtime:
        try:
            _cache = json.loads(_CAL_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            _cache = {}
        _cache_mtime = mtime
    return _cache


def camera_rotation_deg() -> float:
    try:
        return float(load_rig_calibration().get('camera_rotation_deg', 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def objective_offsets() -> dict:
    return load_rig_calibration().get('objective_offsets', {}) or {}


def objective_offset(mag: str) -> tuple[float, float]:
    """Per-magnification (dx_mm, dy_mm) paraxial (lateral XY) offset, (0, 0) if unset."""
    off = objective_offsets().get(mag, [0.0, 0.0])
    return float(off[0]), float(off[1])


def resolution_offsets() -> dict:
    """Legacy Arducam per-resolution centre shifts (mm), keyed 'WxH'.

    Empty for Alvium widths — the IMX250 uses the same sensor region at every
    binning mode, so no entries exist (or ever should) for 2464/1232/616.
    """
    return load_rig_calibration().get('resolution_offsets', {}) or {}


def parfocal_z_offset(mag: str):
    """Per-magnification parfocal focus offset (mm, relative to the 100x
    reference feature focus), or None when uncalibrated. Like the XY
    objective offsets, 100x is the reference (0.0)."""
    if mag == '100x':
        return 0.0
    v = (load_rig_calibration().get('parfocal_z_offsets', {}) or {}).get(mag)
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def manip_to_stage() -> dict | None:
    return load_rig_calibration().get('manip_to_stage')


# ── canonical pixel↔stage delta transform ─────────────────────────────────────

def pixel_delta_to_stage_mm(dpx: float, dpy: float, ppm: float,
                            rotation_deg: float | None = None) -> tuple[float, float]:
    """Pixel offset from frame centre (+right/+down) → stage (dX, dY) in mm.

    ppm is pixels per µm. rotation_deg overrides the stored camera rotation
    (pass 0.0 to disable). Matches ui/preview's historical
    _pixel_delta_to_stage_mm exactly.
    """
    t = math.radians(camera_rotation_deg() if rotation_deg is None else rotation_deg)
    c, s = math.cos(t), math.sin(t)
    return ((c * dpx + s * dpy) / ppm / 1000.0,
            (s * dpx - c * dpy) / ppm / 1000.0)


def stage_delta_to_pixel(dx_mm: float, dy_mm: float, ppm: float,
                         rotation_deg: float | None = None) -> tuple[float, float]:
    """Exact inverse of pixel_delta_to_stage_mm (the matrix is involutory)."""
    t = math.radians(camera_rotation_deg() if rotation_deg is None else rotation_deg)
    c, s = math.cos(t), math.sin(t)
    ux, uy = dx_mm * 1000.0 * ppm, dy_mm * 1000.0 * ppm
    return (c * ux + s * uy, s * ux - c * uy)

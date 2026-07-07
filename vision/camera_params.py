"""Shared camera calibration — single source of truth for pixels-per-micron.

Ported from standa_stacker/vision/camera_params.py, with THIS rig's measured
values. Imported by ui/preview.py, core/scan_io.py and the vision detectors so
there is only one place to update after a camera change or new measurement.

NOTE: do NOT copy standa_stacker's numbers here (1.5999 at 10x) — same Alvium
camera, different microscope optics. nikon-257's value below was measured with
a stage micrometer.
"""

# Magnification scaling factors relative to 10×.
MAG_SCALE: dict[str, float] = {
    '5x': 0.5, '10x': 1.0, '20x': 2.0, '50x': 5.0, '100x': 10.0,
}

# ── Alvium 1800 U-508c on the Nikon L200ND (nikon-257) ───────────────────────
# 10x base measured 2026-03-18 with a stage micrometer; binning ratios are
# exact (hardware), so the 2x/4x entries are the base halved per step.
# Keyed by output frame width.
_ALVIUM_10X: dict[int, float] = {
    2464: 2.03,     # native full-res  2464×2056
    1232: 1.015,    # 2× binning       1232×1028
    616:  0.5075,   # 4× binning        616×514
}

_BASE_10X: dict[int, float] = dict(_ALVIUM_10X)


def px_per_um(mag: str, frame_w: int) -> float | None:
    """Pixels-per-micron for (magnification, frame width).

    Returns None if the combination is not in the table.
    """
    ms   = MAG_SCALE.get(mag)
    base = _BASE_10X.get(frame_w)
    if ms is None or base is None:
        return None
    return base * ms

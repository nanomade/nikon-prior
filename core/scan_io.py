"""One home for reading area-scan folders (plan task L1).

Extracted from tools/make_map.py, where the canonical load_scan() lived
trapped in a 4000-line __main__ script while ~18 other sites re-parsed
scan_metadata.json by hand. Everything that reads a scan goes through here;
the schema itself is pinned by tests/test_scan_metadata_schema.py.

Two entry points, deliberately distinct:
  load_metadata(folder)  — the raw scan_metadata.json dict. Tile positions
                           exactly as the scanner recorded them (what the
                           detector uses: candidate stage coords must match
                           the commanded positions).
  load_scan(folder)      — the map-building descriptor: ppm/fov resolved,
                           tile positions NORMALISED by the legacy Arducam
                           resolution_offsets (no-op for Alvium widths).
"""
import json
from pathlib import Path

from vision.camera_params import px_per_um as _camera_ppm, MAG_SCALE
from core.rig_calibration import resolution_offsets


def load_metadata(folder) -> dict:
    """Raw scan_metadata.json from a scan folder (FileNotFoundError if absent)."""
    meta_path = Path(folder) / 'scan_metadata.json'
    if not meta_path.exists():
        raise FileNotFoundError(f"No scan_metadata.json in {folder}")
    return json.loads(meta_path.read_text())


def scan_mag(meta: dict, default: str = '10x') -> str:
    """Magnification string, from either historical home."""
    return (meta.get('scan_params', {}).get('mag')
            or meta.get('imaging', {}).get('magnification')
            or default)


def scan_ppu(meta: dict) -> float:
    """Pixels-per-µm for a scan (ValueError if the resolution is unknown)."""
    mag = scan_mag(meta)
    frame_w = int(meta['imaging']['frame_width'])
    ppu = _camera_ppm(mag, frame_w)
    if ppu is None:
        raise ValueError(f"Unknown camera resolution: mag={mag!r}, frame_w={frame_w}")
    return ppu


def load_scan(folder) -> dict:
    """Read scan_metadata.json and return the map-building scan descriptor:

      ppu        — pixels per µm at this mag/resolution
      ppm        — pixels per mm  (ppu × 1000)
      fov_x_mm   — image field of view in X (mm)
      fov_y_mm   — image field of view in Y (mm)
      frame_w    — image frame width (px)
      frame_h    — image frame height (px)
      mag        — magnification string e.g. "20x"
      folder     — Path to the folder
      images     — list of dicts with x_mm, y_mm, filename, focus_ok
                   (positions normalised to the sensor-centre reference via
                   the legacy Arducam resolution_offsets; Alvium = no-op)
      scan_params — the raw scan_params dict
    """
    folder = Path(folder)
    meta = load_metadata(folder)

    im = meta['imaging']
    mag = im.get('magnification') or meta['scan_params'].get('mag', '10x')
    frame_w = int(im['frame_width'])
    frame_h = int(im['frame_height'])

    ppu = scan_ppu(meta)
    ppm = ppu * 1000.0

    res_off = resolution_offsets().get(f"{frame_w}x{frame_h}", [0.0, 0.0])
    images = meta['images']
    if res_off[0] or res_off[1]:
        images = [dict(img, x_mm=img['x_mm'] - res_off[0],
                               y_mm=img['y_mm'] - res_off[1])
                  for img in images]

    return {
        'ppu':         ppu,
        'ppm':         ppm,
        'fov_x_mm':    frame_w / ppm,
        'fov_y_mm':    frame_h / ppm,
        'frame_w':     frame_w,
        'frame_h':     frame_h,
        'mag':         mag,
        'folder':      folder,
        'images':      images,
        'scan_params': meta.get('scan_params', {}),
    }


def scan_bounds(scan: dict) -> tuple:
    """(x_min, y_min, x_max, y_max) in mm including FOV half-extents."""
    hw = scan['fov_x_mm'] / 2.0
    hh = scan['fov_y_mm'] / 2.0
    xs = [img['x_mm'] for img in scan['images']]
    ys = [img['y_mm'] for img in scan['images']]
    return (min(xs) - hw, min(ys) - hh, max(xs) + hw, max(ys) + hh)


def all_bounds(scans: list) -> tuple:
    """Union of all scan bounds."""
    bounds = [scan_bounds(s) for s in scans if s['images']]
    return (min(b[0] for b in bounds), min(b[1] for b in bounds),
            max(b[2] for b in bounds), max(b[3] for b in bounds))


def discover_scan_folders(root) -> list[Path]:
    """All leaf directories under root that contain scan_metadata.json.

    Handles three layouts:
      root/scan_metadata.json            — root is itself a scan folder
      root/{mag}/scan_.../scan_metadata  — sample-root hierarchy
      root/scan_.../scan_metadata        — mag dir passed directly

    Sorted lowest magnification first (higher mags paint on top of maps).
    """
    root = Path(root)
    if (root / 'scan_metadata.json').exists():
        return [root]

    found = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if (child / 'scan_metadata.json').exists():
            found.append(child)
            continue
        for grandchild in sorted(child.iterdir()):
            if grandchild.is_dir() and (grandchild / 'scan_metadata.json').exists():
                found.append(grandchild)

    def _mag_key(p: Path) -> float:
        try:
            return MAG_SCALE.get(scan_mag(load_metadata(p)), 1.0)
        except Exception:
            return 1.0

    return sorted(found, key=_mag_key)

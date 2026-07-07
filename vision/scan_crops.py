"""Extract a thumbnail crop of a flake from the area-scan tiles.

Used to give map-marked (and any stage-positioned) catalogue flakes the same
thumbnail an app-captured flake has.  Auto-detected flakes get their crop straight
from the detector (tools/calibrate_ladder.py); this is the path for flakes that
only carry a stage position.

Pixel convention matches the map (_mmFromVp) and the detector: image +x -> stage
+x, image +y (down) -> stage +y, linear within a tile (the ~0.3 deg camera tilt is
<2 um across one tile -- irrelevant for a thumbnail).
"""
import math
from pathlib import Path

import cv2

from core.scan_io import load_metadata, scan_mag
from vision.camera_params import px_per_um


def find_scan_folder(sample_dir, mag: str | None = None) -> Path | None:
    """Newest area-scan folder under a sample, optionally preferring a magnification."""
    metas = sorted(Path(sample_dir).glob('mapping/scans/**/scan_metadata.json'),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not metas:
        return None
    if mag:
        for p in metas:
            try:
                if scan_mag(load_metadata(p.parent), default='') == mag:
                    return p.parent
            except Exception:
                continue
    return metas[0].parent


def crop_radius_px(area_um2: float | None, ppu: float) -> int:
    """Crop half-width (source px) for a flake of given area — the single formula
    shared by the detector and the map-crop path.  Equivalent-disc radius, padded."""
    if area_um2:
        return int(max(60, math.sqrt(area_um2 / math.pi) * ppu * 1.7))
    return 110


def crop_at_stage(scan_folder, sx_mm: float, sy_mm: float,
                  area_um2: float | None = None, out_size: int = 220):
    """Return (BGR crop out_size x out_size, crop_um) for the flake at stage (sx, sy),
    or (None, None).  crop_um is the crop's full physical width in microns (for a
    scale bar).  The crop half-width scales with flake area when known, else fixed.
    """
    scan_folder = Path(scan_folder)
    try:
        meta = load_metadata(scan_folder)
    except Exception:
        return None, None
    imgs = [im for im in meta.get('images', [])
            if im.get('x_mm') is not None and im.get('y_mm') is not None]
    if not imgs:
        return None, None
    fw = meta['imaging']['frame_width']
    fh = meta['imaging'].get('frame_height', 2056)
    mag = scan_mag(meta, default='20x')
    ppu = px_per_um(mag, fw)
    if not ppu:
        return None, None
    ppm = ppu * 1000.0
    t = min(imgs, key=lambda im: (im['x_mm'] - sx_mm) ** 2 + (im['y_mm'] - sy_mm) ** 2)
    cx = fw / 2.0 + (sx_mm - t['x_mm']) * ppm        # +pdx
    cy = fh / 2.0 + (sy_mm - t['y_mm']) * ppm        # +pdy (matches map / detector)
    rad = crop_radius_px(area_um2, ppu)
    crop_um = round(2 * rad / ppu, 1)                 # full physical width (µm)
    img = cv2.imread(str(scan_folder / t['filename']))
    if img is None:
        return None, None
    x0, y0 = max(0, int(cx - rad)), max(0, int(cy - rad))
    crop = img[y0:int(cy + rad), x0:int(cx + rad)]
    if crop.size == 0:
        return None, None
    return cv2.resize(crop, (out_size, out_size)), crop_um

"""Tile-position solving + canvas assembly for area-scan maps (plan task L5).

Extracted from tools/make_map.py: the ORB + phase-correlation pair matcher,
position propagation, reference-canvas registration, the rotation model
(camera_rotation_deg extraction + placement), flat/background correction and
blended pasting, and the assemble/assemble_layers canvas builders. make_map
remains the CLI + HTML generator and imports everything from here.

Also home of the lightweight _stage timing used across the map build
(set_timing() switches the printed breakdown on; make_map --timing).
"""
import math
import shutil
import sys
import time as _time
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow not installed — run: pip install Pillow")

try:
    import numpy as np
    import cv2
    _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False

from vision.camera_params import px_per_um as _camera_ppm, MAG_SCALE as _MAG_SCALE
from core.scan_io import scan_bounds, all_bounds  # noqa: F401  (all_bounds used by assemble)
from core.rig_calibration import APP_DIR as _APP_DIR


def set_timing(on: bool) -> None:
    """Enable/disable the per-stage wall-clock breakdown (make_map --timing)."""
    global _TIMING
    _TIMING = bool(on)


_TIMING = False
_STAGE_TIMES: dict = {}


class _stage:
    """Context manager accumulating elapsed wall-time under a label."""
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self._t = _time.perf_counter()
        return self

    def __exit__(self, *exc):
        dt = _time.perf_counter() - self._t
        _STAGE_TIMES[self.name] = _STAGE_TIMES.get(self.name, 0.0) + dt
        if _TIMING:
            print(f"  ⏱ {self.name}: {dt:.2f}s")
        return False


def _print_stage_times() -> None:
    if not _TIMING or not _STAGE_TIMES:
        return
    total = sum(_STAGE_TIMES.values())
    print("\n⏱ make_map timing breakdown (non-overlapping stages):")
    for k, v in sorted(_STAGE_TIMES.items(), key=lambda kv: -kv[1]):
        pct = 100 * v / total if total else 0.0
        print(f"    {k:32s} {v:8.2f}s  ({pct:4.1f}%)")
    print(f"    {'TOTAL (timed)':32s} {total:8.2f}s")


def _px_per_um(mag: str, frame_w: int) -> float:
    """Return pixels-per-µm for the given magnification and frame width."""
    ppm = _camera_ppm(mag, frame_w)
    if ppm is None:
        raise ValueError(f"Unknown camera resolution: mag={mag!r}, frame_w={frame_w}")
    return ppm


MAX_CANVAS_PX = 25000  # auto-scale if either dimension exceeds this


def _choose_canvas_ppm(scans: list, force_scale: float | None) -> float:
    """
    Choose pixels-per-mm for the output canvas.

    Use the highest-mag scan's native ppm as the "ideal" scale, then
    auto-downscale so neither canvas dimension exceeds MAX_CANVAS_PX,
    unless the user forced a scale factor.
    """
    native_ppm = max(s['ppm'] for s in scans)

    if force_scale is not None:
        return native_ppm * force_scale

    x0, y0, x1, y1 = all_bounds(scans)
    canvas_w = (x1 - x0) * native_ppm
    canvas_h = (y1 - y0) * native_ppm
    longest = max(canvas_w, canvas_h)
    if longest > MAX_CANVAS_PX:
        native_ppm *= MAX_CANVAS_PX / longest

    return native_ppm



_ORB_OVERLAP_FRAC = 0.15   # fallback overlap fraction when step_x/y absent from metadata
_ORB_MAX_PX       = 80     # hard clamp: max correction in canvas pixels
_ORB_MIN_MATCHES  = 6      # minimum inlier keypoints to accept a correction
_ORB_RANSAC_THR   = 3.0    # inlier threshold in native pixels (|displacement - median| < thr)
_PC_MIN_RESPONSE  = 0.02   # phase-correlation response threshold for fallback


def _build_grid(images: list) -> dict:
    """
    Map each image to a (row, col) grid cell by clustering y_mm and x_mm.

    When two images land at the same cell (e.g. a focus_ok=True image replacing
    a focus_ok=False one), the focus_ok=True entry wins.
    Returns dict: (row, col) → img_meta.
    """
    ys = sorted({round(im['y_mm'], 3) for im in images})
    xs = sorted({round(im['x_mm'], 3) for im in images})
    grid: dict = {}
    for im in images:
        row = min(range(len(ys)), key=lambda r: abs(ys[r] - im['y_mm']))
        col = min(range(len(xs)), key=lambda c: abs(xs[c] - im['x_mm']))
        prev = grid.get((row, col))
        if prev is None or (im.get('focus_ok', True) and not prev.get('focus_ok', True)):
            grid[(row, col)] = im
    return grid


def _orb_match_pair(img1_g: 'np.ndarray', img2_g: 'np.ndarray',
                    step_native: float, axis: int,
                    ov_size_native: int, max_shift_px: float,
                    orb, bf, clahe=None) -> tuple:
    """
    Strip-based ORB match + median-RANSAC.

    Crops just the overlap strip from each tile so ORB sees only the shared region.
    In strip coordinates the nominal displacement is ≈0, so matching keypoints
    should have nearly identical (x,y); any observed offset is the misalignment.

    axis=0  horizontal: img1=left tile (right strip) ↔ img2=right tile (left strip)
    axis=1  vertical:   img1=top tile (bottom strip) ↔ img2=bottom tile (top strip)

    Returns (dx, dy, n_inliers) in native-pixel units, representing the correction
    to add to the nominal canvas step (positive = img2 is shifted right/down vs stage).
    Returns (0.0, 0.0, 0) on failure.
    """
    h, w = img1_g.shape
    if axis == 0:   # horizontal: right edge of img1 ↔ left edge of img2
        strip1 = img1_g[:, w - ov_size_native:]
        strip2 = img2_g[:, :ov_size_native]
    else:           # vertical: bottom edge of img1 ↔ top edge of img2
        strip1 = img1_g[h - ov_size_native:, :]
        strip2 = img2_g[:ov_size_native, :]

    eq1 = clahe.apply(strip1) if clahe is not None else strip1
    eq2 = clahe.apply(strip2) if clahe is not None else strip2

    kp1, des1 = orb.detectAndCompute(eq1, None)
    kp2, des2 = orb.detectAndCompute(eq2, None)

    orb_ok = False
    if (des1 is not None and des2 is not None
            and len(kp1) >= _ORB_MIN_MATCHES and len(kp2) >= _ORB_MIN_MATCHES):
        raw = bf.knnMatch(des1, des2, k=2)
        matches = [pair[0] for pair in raw
                   if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
        if len(matches) >= _ORB_MIN_MATCHES:
            # In strip coords, at perfect alignment both tiles show the same content,
            # so matched keypoints have equal positions → displacement ≈ 0.
            dxs = np.array([kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0] for m in matches])
            dys = np.array([kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1] for m in matches])
            dx_med = float(np.median(dxs))
            dy_med = float(np.median(dys))
            inliers = ((np.abs(dxs - dx_med) < _ORB_RANSAC_THR) &
                       (np.abs(dys - dy_med) < _ORB_RANSAC_THR))
            n = int(inliers.sum())
            if n >= _ORB_MIN_MATCHES:
                dx = float(dxs[inliers].mean())
                dy = float(dys[inliers].mean())
                if abs(dx) <= max_shift_px and abs(dy) <= max_shift_px:
                    orb_ok = True

    if orb_ok:
        return dx, dy, n

    # Phase-correlation fallback: works on any content (blobs, particles, texture).
    # In strip coordinates the nominal shift is ≈0; the peak offset = misalignment.
    # phaseCorrelate(a, b) returns the shift of b relative to a, so correction = -shift.
    sh, sw = strip1.shape
    hann = cv2.createHanningWindow((sw, sh), cv2.CV_32F)
    (pc_dx, pc_dy), resp = cv2.phaseCorrelate(
        strip1.astype(np.float32), strip2.astype(np.float32), hann)
    pc_corr_x, pc_corr_y = -pc_dx, -pc_dy
    if (resp >= _PC_MIN_RESPONSE
            and abs(pc_corr_x) <= max_shift_px
            and abs(pc_corr_y) <= max_shift_px):
        return pc_corr_x, pc_corr_y, -1   # -1 = phase-correlation result

    return 0.0, 0.0, 0


def _precompute_orb(scan: dict, canvas_ppm: float) -> tuple:
    """
    Load all adjacent tile pairs from disk and compute ORB corrections.

    Horizontal pairs: compare right edge of left tile ↔ left edge of right tile.
    Vertical pairs  : compare bottom edge of top tile ↔ top edge of bottom tile.
    Both dx and dy are estimated from each pair; no axis separation.

    Overlap strip widths are derived from step_x/step_y in scan_params when
    present (exact), falling back to _ORB_OVERLAP_FRAC × frame dimension.

    Returns (grid, h_corr, v_corr, img_scale) where:
        h_corr/v_corr : dict (row,col) → (dx_canvas, dy_canvas, n_inliers)
        img_scale     : canvas_ppm / native_ppm
    """
    folder     = scan['folder']
    frame_w    = scan['frame_w']
    frame_h    = scan['frame_h']
    native_ppm = scan['ppm']
    img_scale  = canvas_ppm / native_ppm
    max_nat    = max(200.0, _ORB_MAX_PX / img_scale)  # clamp in native pixels; floor at 200 so systematic tile offsets aren't rejected at high canvas scales

    sp = scan.get('scan_params', {})
    # Exact overlap from step metadata; fallback to fraction
    step_x_nat = sp['step_x'] * native_ppm if 'step_x' in sp else frame_w * (1 - _ORB_OVERLAP_FRAC)
    step_y_nat = sp['step_y'] * native_ppm if 'step_y' in sp else frame_h * (1 - _ORB_OVERLAP_FRAC)
    ov_w_nat   = max(16, min(round(frame_w - step_x_nat), frame_w // 2))
    ov_h_nat   = max(16, min(round(frame_h - step_y_nat), frame_h // 2))

    grid  = _build_grid(scan['images'])
    orb   = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=4)
    bf    = cv2.BFMatcher(cv2.NORM_HAMMING)   # knnMatch + ratio test in _orb_match_pair
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    h_corr: dict = {}
    v_corr: dict = {}
    n_h_total = n_v_total = 0

    for (row, col) in sorted(grid):
        img_path = folder / grid[(row, col)]['filename']

        if (row, col - 1) in grid:
            n_h_total += 1
            left_path = folder / grid[(row, col - 1)]['filename']
            limg = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
            rimg = cv2.imread(str(img_path),  cv2.IMREAD_GRAYSCALE)
            if limg is not None and rimg is not None:
                dx_n, dy_n, n = _orb_match_pair(
                    limg, rimg, step_x_nat, 0, ov_w_nat, max_nat, orb, bf, clahe)
                if n != 0:
                    h_corr[(row, col)] = (dx_n * img_scale, dy_n * img_scale, n)

        if (row - 1, col) in grid:
            n_v_total += 1
            top_path = folder / grid[(row - 1, col)]['filename']
            timg = cv2.imread(str(top_path),  cv2.IMREAD_GRAYSCALE)
            bimg = cv2.imread(str(img_path),  cv2.IMREAD_GRAYSCALE)
            if timg is not None and bimg is not None:
                dx_n, dy_n, n = _orb_match_pair(
                    timg, bimg, step_y_nat, 1, ov_h_nat, max_nat, orb, bf, clahe)
                if n != 0:
                    v_corr[(row, col)] = (dx_n * img_scale, dy_n * img_scale, n)

    n_h_orb = sum(1 for v in h_corr.values() if v[2] > 0)
    n_h_pc  = sum(1 for v in h_corr.values() if v[2] == -1)
    n_v_orb = sum(1 for v in v_corr.values() if v[2] > 0)
    n_v_pc  = sum(1 for v in v_corr.values() if v[2] == -1)
    print(f"    H-pairs: {n_h_orb+n_h_pc}/{n_h_total} matched "
          f"(ORB={n_h_orb} PC={n_h_pc})  "
          f"V-pairs: {n_v_orb+n_v_pc}/{n_v_total} matched "
          f"(ORB={n_v_orb} PC={n_v_pc})  "
          f"overlap zone: {ov_w_nat}×{frame_h} / {frame_w}×{ov_h_nat} native px")

    # Fill unmatched pairs with the median correction from matched pairs.
    # Cross-axis coupling (dy per H-step, dx per V-step) is a hardware
    # constant — applying the median to unmatched pairs is better than 0.
    def _fill_fallback(corr_dict, total_keys):
        if not corr_dict:
            return
        # Use only directly-measured entries (ORB or phase correlation) for the median
        measured = [v for v in corr_dict.values() if v[2] != 0]
        if not measured:
            return
        dxs = [v[0] for v in measured]
        dys = [v[1] for v in measured]
        med_dx = float(np.median(dxs))
        med_dy = float(np.median(dys))
        n_filled = 0
        for key in total_keys:
            if key not in corr_dict:
                corr_dict[key] = (med_dx, med_dy, 0)
                n_filled += 1
        if n_filled:
            print(f"    Filled {n_filled} unmatched pair(s) with median "
                  f"correction ({med_dx:+.1f}, {med_dy:+.1f}) px")

    h_keys = [(row, col) for (row, col) in sorted(grid) if (row, col-1) in grid]
    v_keys = [(row, col) for (row, col) in sorted(grid) if (row-1, col) in grid]
    _fill_fallback(h_corr, h_keys)
    _fill_fallback(v_corr, v_keys)

    return grid, h_corr, v_corr, img_scale


def _propagate_positions(grid: dict, h_corr: dict, v_corr: dict,
                         canvas_ppm: float, x0: float, y0: float) -> dict:
    """
    BFS outward from the centre tile to build absolute canvas centre positions.

    Centre-out order means no tile is further than half the grid diagonal from
    the reference, minimising worst-case cumulative drift vs. corner-to-corner.

    Each neighbour's centre = current tile's actual centre
                            + stage-coordinate step
                            + ORB correction (or 0 if ORB failed).

    Corrections are inverted for leftward/upward traversal so the BFS can
    walk the grid in all four directions from a single stored h_corr/v_corr
    that was measured right→ and down→.

    Returns dict: (row, col) → (cx_float, cy_float) canvas centre in pixels.
    """
    from collections import deque

    def _stage_centre(im):
        return ((im['x_mm'] - x0) * canvas_ppm,
                (im['y_mm'] - y0) * canvas_ppm)

    all_cells = sorted(grid)
    rows = sorted({r for r, c in all_cells})
    cols = sorted({c for r, c in all_cells})
    mid_row = rows[len(rows) // 2]
    mid_col = cols[len(cols) // 2]
    start = min(all_cells,
                key=lambda rc: abs(rc[0] - mid_row) + abs(rc[1] - mid_col))

    positions: dict = {}
    positions[start] = _stage_centre(grid[start])
    visited = {start}
    queue = deque([start])

    while queue:
        rc = queue.popleft()
        cx, cy = positions[rc]
        cur_im = grid[rc]

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nrc = (rc[0] + dr, rc[1] + dc)
            if nrc not in grid or nrc in visited:
                continue
            n_im = grid[nrc]
            step_cx = (n_im['x_mm'] - cur_im['x_mm']) * canvas_ppm
            step_cy = (n_im['y_mm'] - cur_im['y_mm']) * canvas_ppm
            if dc == +1:    # right: h_corr stored at nrc
                c = h_corr.get(nrc, (0.0, 0.0, 0));  orb_dx, orb_dy = c[0], c[1]
            elif dc == -1:  # left: negate h_corr stored at rc (reverse direction)
                c = h_corr.get(rc,  (0.0, 0.0, 0));  orb_dx, orb_dy = -c[0], -c[1]
            elif dr == +1:  # down: v_corr stored at nrc
                c = v_corr.get(nrc, (0.0, 0.0, 0));  orb_dx, orb_dy = c[0], c[1]
            else:           # up: negate v_corr stored at rc
                c = v_corr.get(rc,  (0.0, 0.0, 0));  orb_dx, orb_dy = -c[0], -c[1]
            positions[nrc] = (cx + step_cx + orb_dx, cy + step_cy + orb_dy)
            visited.add(nrc)
            queue.append(nrc)

    for rc in all_cells:
        if rc not in positions:
            positions[rc] = _stage_centre(grid[rc])
    return positions




_REF_SAMPLE_TILES  = 9      # max tiles sampled per scan for reference registration
_REF_MIN_RESPONSE  = 0.05   # PC response threshold against the reference canvas
_REF_MAX_SHIFT_PX  = 150    # generous clamp — this is an absolute-position correction


def _clahe_f32(arr: 'np.ndarray', clahe) -> 'np.ndarray':
    """Apply CLAHE to a float32 grayscale array; return float32 suitable for phaseCorrelate."""
    u8 = np.clip(arr, 0, 255).astype(np.uint8)
    return clahe.apply(u8).astype(np.float32)


def _register_to_ref(scan: dict, grid: dict, ref_canvas: 'Image.Image',
                     canvas_ppm: float, x0: float, y0: float,
                     positions: dict) -> dict:
    """
    Refine every tile's canvas position by phase-correlating it against the
    already-assembled lower-magnification reference canvas.

    Both the reference crop and the high-mag tile are at canvas resolution
    (canvas_ppm px/mm), so no scale conversion is needed for PC — they see
    the same features at the same pixel density.  This corrects for
    absolute inter-scan drift and works even for isolated single tiles that
    have no neighbours to ORB against.

    Returns an updated positions dict; tiles where the reference region is
    unpainted or the PC response is below threshold keep their input position.
    """
    folder    = scan['folder']
    frame_w   = scan['frame_w']
    frame_h   = scan['frame_h']
    img_scale = canvas_ppm / scan['ppm']
    dest_w    = max(1, round(frame_w  * img_scale))
    dest_h    = max(1, round(frame_h  * img_scale))
    cw, ch    = ref_canvas.size

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    cells = sorted(positions)

    # Prefer focus_ok tiles for sampling; fall back to all tiles if too few.
    good_cells = [rc for rc in cells if grid.get(rc, {}).get('focus_ok', True)]
    pool = good_cells if len(good_cells) >= 3 else cells

    if len(pool) <= _REF_SAMPLE_TILES:
        sample = pool
    else:
        step   = max(1, len(pool) // _REF_SAMPLE_TILES)
        sample = pool[::step][:_REF_SAMPLE_TILES]

    # ── Phase 1: estimate global offset from sampled tiles ─────────────────
    offsets = []
    for rc in sample:
        im = grid.get(rc)
        if im is None:
            continue
        cx_n, cy_n = positions[rc]

        # Canvas-pixel bounds of this tile's nominal region
        nom_x0 = round(cx_n - dest_w / 2)
        nom_y0 = round(cy_n - dest_h / 2)
        rx0 = max(0, nom_x0);  ry0 = max(0, nom_y0)
        rx1 = min(cw, nom_x0 + dest_w)
        ry1 = min(ch, nom_y0 + dest_h)
        rw, rh = rx1 - rx0, ry1 - ry0
        if rw < dest_w // 3 or rh < dest_h // 3:
            continue  # tile mostly outside reference canvas

        ref_arr = np.array(ref_canvas.crop((rx0, ry0, rx1, ry1)).convert('L'),
                           dtype=np.float32)
        if float(ref_arr.max()) < 40:
            continue  # reference region is unpainted background

        img_path = folder / im['filename']
        tile_g   = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if tile_g is None:
            continue

        tile_scaled = cv2.resize(tile_g, (dest_w, dest_h))
        tx0 = rx0 - nom_x0;  ty0 = ry0 - nom_y0
        tile_crop = tile_scaled[ty0:ty0 + rh, tx0:tx0 + rw].astype(np.float32)
        if tile_crop.shape != ref_arr.shape:
            continue

        hann = cv2.createHanningWindow((rw, rh), cv2.CV_32F)
        (pc_dx, pc_dy), resp = cv2.phaseCorrelate(
            _clahe_f32(ref_arr, clahe), _clahe_f32(tile_crop, clahe), hann)
        if (resp >= _REF_MIN_RESPONSE
                and abs(pc_dx) <= _REF_MAX_SHIFT_PX
                and abs(pc_dy) <= _REF_MAX_SHIFT_PX):
            offsets.append((-pc_dx, -pc_dy))

    if not offsets:
        return positions

    glob_dx = float(np.median([o[0] for o in offsets]))
    glob_dy = float(np.median([o[1] for o in offsets]))
    n_good  = len(good_cells)
    print(f"    Reference registration: global offset ({glob_dx:+.1f}, {glob_dy:+.1f}) px"
          f"  from {len(offsets)}/{len(sample)} sampled tiles"
          + (f"  ({n_good} focus_ok)" if n_good < len(cells) else ""))

    # ── Phase 2: apply global offset, then per-tile PC refinement ─────────
    # Per-tile PC is only reliable for focused tiles; out-of-focus tiles get
    # the global offset applied but skip the refinement step.
    new_positions = {}

    for rc, (cx_n, cy_n) in positions.items():
        cx_g = cx_n + glob_dx
        cy_g = cy_n + glob_dy

        im = grid.get(rc)
        if im is None or not im.get('focus_ok', True):
            new_positions[rc] = (cx_g, cy_g)
            continue

        nom_x0 = round(cx_g - dest_w / 2)
        nom_y0 = round(cy_g - dest_h / 2)
        rx0 = max(0, nom_x0);  ry0 = max(0, nom_y0)
        rx1 = min(cw, nom_x0 + dest_w)
        ry1 = min(ch, nom_y0 + dest_h)
        rw, rh = rx1 - rx0, ry1 - ry0
        if rw < dest_w // 3 or rh < dest_h // 3:
            new_positions[rc] = (cx_g, cy_g)
            continue

        ref_arr = np.array(ref_canvas.crop((rx0, ry0, rx1, ry1)).convert('L'),
                           dtype=np.float32)
        if float(ref_arr.max()) < 40:
            new_positions[rc] = (cx_g, cy_g)
            continue

        img_path = folder / im['filename']
        tile_g   = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if tile_g is None:
            new_positions[rc] = (cx_g, cy_g)
            continue

        tile_scaled = cv2.resize(tile_g, (dest_w, dest_h))
        tx0 = rx0 - nom_x0;  ty0 = ry0 - nom_y0
        tile_crop = tile_scaled[ty0:ty0 + rh, tx0:tx0 + rw].astype(np.float32)
        if tile_crop.shape != ref_arr.shape:
            new_positions[rc] = (cx_g, cy_g)
            continue

        hann = cv2.createHanningWindow((rw, rh), cv2.CV_32F)
        (pc_dx, pc_dy), resp = cv2.phaseCorrelate(
            _clahe_f32(ref_arr, clahe), _clahe_f32(tile_crop, clahe), hann)
        if (resp >= _REF_MIN_RESPONSE
                and abs(pc_dx) <= _REF_MAX_SHIFT_PX
                and abs(pc_dy) <= _REF_MAX_SHIFT_PX):
            new_positions[rc] = (cx_g - pc_dx, cy_g - pc_dy)
        else:
            new_positions[rc] = (cx_g, cy_g)

    return new_positions


_FLAT_MIN_TILES = 5    # minimum tiles to attempt a median flat field
_FLAT_MAX_TILES = 100  # subsample beyond this many tiles
_FLAT_DS        = 4    # native→small downsample factor when stacking


def _build_flat_from_tiles(scan: dict, dest_w: int, dest_h: int) -> 'np.ndarray | None':
    """
    Build a BGR float32 flat-field image at canvas resolution (dest_w × dest_h)
    by taking the per-pixel median of all tiles in the scan.

    Each tile is downsampled _FLAT_DS× before stacking so that peak memory use
    stays modest (≈ N_tiles × (frame_w/_FLAT_DS) × (frame_h/_FLAT_DS) × 12 B).
    The median is then upsampled back to canvas resolution.

    The median removes sample features (which appear at different positions in each
    tile) and retains only the fixed background — vignetting gradient, illumination
    falloff, stuck pixels.  Accurate when sample features cover < 50 % of any given
    pixel position across the tile stack.

    Returns None when fewer than _FLAT_MIN_TILES tiles are readable.
    """
    folder = scan['folder']
    images = list(scan['images'])

    if len(images) > _FLAT_MAX_TILES:
        step   = max(1, len(images) // _FLAT_MAX_TILES)
        images = images[::step][:_FLAT_MAX_TILES]

    ds_w = max(8, scan['frame_w'] // _FLAT_DS)
    ds_h = max(8, scan['frame_h'] // _FLAT_DS)

    small_frames = []
    for im in images:
        arr = cv2.imread(str(folder / im['filename']))
        if arr is None:
            continue
        small_frames.append(
            cv2.resize(arr, (ds_w, ds_h), interpolation=cv2.INTER_AREA).astype(np.float32))

    if len(small_frames) < _FLAT_MIN_TILES:
        return None

    flat_small = np.median(np.stack(small_frames, axis=0), axis=0)   # ds_h × ds_w × 3
    return cv2.resize(flat_small, (dest_w, dest_h), interpolation=cv2.INTER_LINEAR)


def _load_saved_flat(mag: str) -> 'np.ndarray | None':
    """
    Load a flat field saved by the live FlatFieldPanel (flat_field_{mag}.npy).
    Searches the project root (parent of tools/) then cwd.
    Returns float32 BGR array or None.
    """
    safe  = mag.replace(' ', '_').replace('/', '_')
    fname = f"flat_field_{safe}.npy"
    for p in [_APP_DIR / fname, Path.cwd() / fname]:
        if p.exists():
            try:
                return np.load(str(p)).astype(np.float32)
            except Exception:
                pass
    return None


def _apply_flat(tile: Image.Image, flat: np.ndarray) -> Image.Image:
    """
    Apply flat-field gain correction using achromatic luminance gain.

    Matches flat_field_panel.apply_correction(): a single gain map derived
    from the flat luminance is applied to all channels, so colour balance is
    preserved.  The brightest point in the flat gets gain≈1; vignetted edges
    are boosted proportionally.
    """
    arr = np.array(tile)[:, :, ::-1].astype(np.float32)   # RGB → BGR float32
    th, tw = arr.shape[:2]
    if flat.shape[:2] != (th, tw):
        # Resize float32 flat channel-by-channel via PIL (mode='F' auto-detected from dtype)
        flat = np.stack([
            np.array(Image.fromarray(flat[:, :, c].astype(np.float32)).resize((tw, th), Image.BILINEAR))
            for c in range(3)
        ], axis=2)
    flat_lum = flat.mean(axis=2, keepdims=True)            # achromatic luminance
    gain     = flat_lum.max() / np.clip(flat_lum, 1.0, None)
    return Image.fromarray(np.clip(arr * gain, 0, 255).astype(np.uint8)[:, :, ::-1])


def _bg_correct_tile(tile: Image.Image) -> Image.Image:
    """
    Remove per-tile vignetting by dividing each pixel by a Gaussian-estimated
    background.  Works on the low-frequency (~quarter-frame scale) spatial
    brightness gradient only — sample features are much smaller and pass through
    unchanged.  Colour balance is preserved: a single luminance scale field is
    derived from the grayscale estimate and applied identically to all channels.

    Pipeline: grayscale → 4× downsample → large Gaussian → upsample → divide.
    Downsampling makes the blur 16× faster without affecting accuracy at this scale.
    """
    arr = np.array(tile, dtype=np.float32)   # H×W×3 RGB
    h, w = arr.shape[:2]

    # Grayscale luminance estimate (BT.601)
    gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2])

    # Downsample 4×, blur, upsample — 16× cheaper than blurring at full res
    ds_w, ds_h = max(4, w // 4), max(4, h // 4)
    small  = cv2.resize(gray, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
    # sigma_frac=0.25 in original pixel space → /4 in downsampled space
    sigma  = max(w, h) * 0.25 / 4.0
    bg_s   = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma)
    bg     = cv2.resize(bg_s, (w, h), interpolation=cv2.INTER_LINEAR)

    # Correction field: scale < 1 where background is bright (vignette centre),
    # scale > 1 where background is dark (vignette edges) — flattens the gradient.
    mean_bg = float(bg.mean())
    scale   = mean_bg / np.clip(bg, 1.0, None)   # H×W

    corrected = arr * scale[:, :, np.newaxis]
    return Image.fromarray(np.clip(corrected, 0, 255).astype(np.uint8))


def _paste_blended(canvas: Image.Image, tile: Image.Image, px: int, py: int) -> None:
    """
    Paste tile onto canvas with 50/50 blend wherever canvas already has content.

    Unpainted canvas pixels (background ≈ (30,30,30)) are fully replaced by the
    tile; already-painted pixels get an equal mix of canvas and tile so both
    tiles are visible through the overlap zone.
    """
    cw, ch = canvas.size
    pw, ph = tile.size
    cx0 = max(0, px);        cy0 = max(0, py)
    cx1 = min(cw, px + pw);  cy1 = min(ch, py + ph)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    canvas_crop = np.array(canvas.crop((cx0, cy0, cx1, cy1)), dtype=np.float32)
    tx0 = cx0 - px;  ty0 = cy0 - py
    tile_crop = np.array(tile)[ty0:ty0+(cy1-cy0), tx0:tx0+(cx1-cx0)].astype(np.float32)
    bg = np.all(canvas_crop < 40, axis=2)                  # unpainted background
    alpha = np.where(bg, 1.0, 0.5)[:, :, np.newaxis]      # tile weight
    result = (tile_crop * alpha + canvas_crop * (1.0 - alpha)).astype(np.uint8)
    canvas.paste(Image.fromarray(result), (cx0, cy0))


def _paste_blended_rgba(canvas: Image.Image, tile: Image.Image, px: int, py: int) -> None:
    """
    Paste RGB tile onto an RGBA canvas.  alpha=0 → unpainted (replace fully);
    alpha>0 → already painted (blend 50/50).  Sets alpha=255 everywhere painted.
    """
    cw, ch = canvas.size
    pw, ph = tile.size
    cx0 = max(0, px);        cy0 = max(0, py)
    cx1 = min(cw, px + pw);  cy1 = min(ch, py + ph)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    canvas_crop = np.array(canvas.crop((cx0, cy0, cx1, cy1)), dtype=np.float32)   # RGBA
    tx0 = cx0 - px;  ty0 = cy0 - py
    tile_crop   = np.array(tile)[ty0:ty0+(cy1-cy0), tx0:tx0+(cx1-cx0)].astype(np.float32)
    unpainted   = canvas_crop[:, :, 3] < 128
    blend       = np.where(unpainted, 1.0, 0.5)[:, :, np.newaxis]
    blended_rgb = (tile_crop * blend + canvas_crop[:, :, :3] * (1.0 - blend)).astype(np.uint8)
    alpha_ch    = np.full(blended_rgb.shape[:2], 255, dtype=np.uint8)
    canvas.paste(Image.fromarray(np.dstack([blended_rgb, alpha_ch]), 'RGBA'), (cx0, cy0))


def assemble(scans: list, canvas_ppm: float,
             refine: bool = True, correct_bg: bool = False) -> Image.Image:
    """
    Paste all scan images onto a single RGB canvas.

    When refine=True (and cv2 is available):
      1. Pre-compute ORB corrections for every adjacent tile pair from raw images
         — both horizontal and vertical, both dx and dy, no axis separation.
         Overlap strip widths come from step_x/step_y in scan metadata.
      2. Propagate absolute canvas positions via BFS from the centre tile
         outward, so no tile is more than half the grid diagonal from the
         reference origin (minimises cumulative drift).
      3. Paint tiles in grid order, blending 50/50 in overlap zones so both
         tiles are visible through the seam.

    When correct_bg=True (and cv2 is available): each tile is passed through
    _bg_correct_tile() before pasting to remove per-tile vignetting.  This
    flattens intra-tile brightness gradients (brighter centre / darker corners)
    without affecting sample features, which are much smaller than the Gaussian
    kernel used for background estimation.
    """
    x0, y0, x1, y1 = all_bounds(scans)
    canvas_w = max(1, round((x1 - x0) * canvas_ppm))
    canvas_h = max(1, round((y1 - y0) * canvas_ppm))
    print(f"Canvas: {canvas_w} × {canvas_h} px  "
          f"({(x1-x0):.2f} × {(y1-y0):.2f} mm)  "
          f"scale={canvas_ppm:.2f} px/mm")

    canvas = Image.new('RGB', (canvas_w, canvas_h), color=(30, 30, 30))

    all_h_corrs: list = []
    all_v_corrs: list = []

    sorted_scans  = sorted(scans, key=lambda s: _MAG_SCALE[s['mag']])
    min_mag_scale = _MAG_SCALE[sorted_scans[0]['mag']]
    ref_canvas: 'Image.Image | None' = None   # set after all lowest-mag scans are painted

    for scan in sorted_scans:
        folder    = scan['folder']
        img_scale = canvas_ppm / scan['ppm']
        dest_w    = max(1, round(scan['frame_w'] * img_scale))
        dest_h    = max(1, round(scan['frame_h'] * img_scale))
        is_ref_mag = _MAG_SCALE[scan['mag']] == min_mag_scale

        # Save the reference canvas just before the first higher-mag scan
        if not is_ref_mag and ref_canvas is None and _HAVE_CV2:
            ref_canvas = canvas.copy()

        if refine and _HAVE_CV2:
            print(f"  ORB pre-computation [{scan['mag']} {folder.name}]…")
            grid, h_corr, v_corr, _ = _precompute_orb(scan, canvas_ppm)
            positions = _propagate_positions(grid, h_corr, v_corr, canvas_ppm, x0, y0)
            for v in h_corr.values():
                if v[2] != 0:
                    all_h_corrs.append((v[0], v[1], v[2]))
            for v in v_corr.values():
                if v[2] != 0:
                    all_v_corrs.append((v[0], v[1], v[2]))
            # Register higher-mag tiles against the assembled low-mag canvas.
            # Works per-tile, so isolated images with no neighbours also benefit.
            if not is_ref_mag and ref_canvas is not None:
                positions = _register_to_ref(scan, grid, ref_canvas,
                                             canvas_ppm, x0, y0, positions)
        else:
            grid = _build_grid(scan['images'])
            positions = None

        # Build flat field for this scan.  Preference order:
        #   1. flat_field_{mag}.npy saved by the live FlatFieldPanel (best: clean
        #      substrate frames from a real random-walk collection)
        #   2. Per-pixel median of the scan tiles themselves (same principle, but
        #      tiles may contain sample features so needs ≥5 tiles to be reliable)
        flat_canvas = None
        if correct_bg:
            # Prefer the flat saved by the live FlatFieldPanel (numpy only, no cv2)
            flat_canvas = _load_saved_flat(scan['mag'])
            if flat_canvas is not None:
                print(f"    Flat [{scan['mag']}]: loaded flat_field_{scan['mag']}.npy")
            elif _HAVE_CV2:
                # Fall back: build from scan tiles (needs cv2.imread)
                n_src  = len(scan['images'])
                n_used = min(n_src, _FLAT_MAX_TILES)
                flat_canvas = _build_flat_from_tiles(scan, dest_w, dest_h)
                if flat_canvas is not None:
                    print(f"    Flat [{scan['mag']}]: median of {n_used}/{n_src} scan tiles")
                else:
                    print(f"    Flat [{scan['mag']}]: only {n_src} tiles, skipping correction")
            else:
                print(f"    Flat [{scan['mag']}]: no saved flat_field_{scan['mag']}.npy found")

        # Paint in row-major grid order
        for i, ((row, col), img_meta) in enumerate(sorted(grid.items())):
            img_path = folder / img_meta['filename']
            if not img_path.exists():
                continue
            try:
                tile = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"  Warning: {img_meta['filename']}: {e}")
                continue

            if tile.width != scan['frame_w'] or tile.height != scan['frame_h']:
                actual_ppu = _px_per_um(scan['mag'], tile.width)
                tile_scale = canvas_ppm / (actual_ppu * 1000.0)
                tile = tile.resize((max(1, round(tile.width  * tile_scale)),
                                    max(1, round(tile.height * tile_scale))),
                                   Image.LANCZOS)
            else:
                tile = tile.resize((dest_w, dest_h), Image.LANCZOS)

            if positions is not None:
                cx, cy = positions[(row, col)]
            else:
                cx = (img_meta['x_mm'] - x0) * canvas_ppm
                cy = (img_meta['y_mm'] - y0) * canvas_ppm

            # Floor/ceil snap: adjacent tiles share pixel edges exactly
            half_w = tile.width  / 2.0
            half_h = tile.height / 2.0
            px = math.floor(cx - half_w);  py = math.floor(cy - half_h)
            pw = math.ceil(cx + half_w) - px
            ph = math.ceil(cy + half_h) - py
            if tile.width != pw or tile.height != ph:
                tile = tile.resize((pw, ph), Image.LANCZOS)

            if flat_canvas is not None:
                tile = _apply_flat(tile, flat_canvas)

            if _HAVE_CV2:
                _paste_blended(canvas, tile, px, py)
            else:
                canvas.paste(tile, (px, py))

            if (i + 1) % 20 == 0 or (i + 1) == len(grid):
                print(f"  [{scan['mag']} {folder.name}]  {i+1}/{len(grid)}", end='\r')
        print()

    # ── Summary statistics ─────────────────────────────────────────────────────
    def _corr_stats(corrs, label):
        if not corrs:
            return
        dxs = [c[0] for c in corrs];  dys = [c[1] for c in corrs]
        n  = len(corrs)
        mx = sum(dxs)/n;  sx = (sum((v-mx)**2 for v in dxs)/n)**0.5
        my = sum(dys)/n;  sy = (sum((v-my)**2 for v in dys)/n)**0.5
        flag = '← systematic' if abs(mx) > sx or abs(my) > sy else '← jitter'
        print(f"  {label} ({n}):  "
              f"dx={mx:+.1f}±{sx:.1f} [{min(dxs):+.0f}…{max(dxs):+.0f}]  "
              f"dy={my:+.1f}±{sy:.1f} [{min(dys):+.0f}…{max(dys):+.0f}]  {flag}")

        def _clip_iqr(vals):
            s = sorted(vals); n = len(s)
            q1 = s[n // 4];  q3 = s[3 * n // 4];  iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            c = [v for v in vals if lo <= v <= hi]
            if not c: return None, None, 0, n
            mc = sum(c) / len(c)
            sc = (sum((v - mc) ** 2 for v in c) / len(c)) ** 0.5
            return mc, sc, len(c), n - len(c)

        mx_c, sx_c, n_c, n_out = _clip_iqr(dxs)
        my_c, sy_c, _,  _     = _clip_iqr(dys)
        if n_out > 0 and mx_c is not None:
            print(f"    clipped ({n_c}/{n}, {n_out} outlier(s) removed):  "
                  f"dx={mx_c:+.1f}±{sx_c:.1f}  dy={my_c:+.1f}±{sy_c:.1f}")

        orb = [c for c in corrs if c[2] > 0]
        if orb and len(orb) < n:
            odxs = [c[0] for c in orb];  odys = [c[1] for c in orb]
            no = len(orb)
            mox = sum(odxs)/no;  sox = (sum((v-mox)**2 for v in odxs)/no)**0.5
            moy = sum(odys)/no;  soy = (sum((v-moy)**2 for v in odys)/no)**0.5
            print(f"    ORB-only ({no}/{n}):  "
                  f"dx={mox:+.1f}±{sox:.1f}  dy={moy:+.1f}±{soy:.1f}")

    if refine and (all_h_corrs or all_v_corrs):
        print("\nORB corrections (px, centre-out BFS from metadata step sizes):")
        _corr_stats(all_h_corrs, "H-pairs")
        _corr_stats(all_v_corrs, "V-pairs")

    return canvas


def assemble_layers(scans: list, canvas_ppm: float,
                    refine: bool = True, correct_bg: bool = False,
                    blend: bool = False) -> dict:
    """
    Like assemble(), but returns one RGBA Image per magnification as a dict
    keyed by mag string (e.g. '10x', '20x').

    Critical design: each mag layer covers only ITS OWN physical bounds, and its
    ppm is capped to MAX_CANVAS_PX based on that layer's own extent — not the
    global union.  A 100x scan over 0.5 mm gets ~16 000 px/mm; a 10x scan over
    6 mm gets ~1 300 px/mm.  No cross-mag downscaling.

    OSD placement: each layer's DZI covers a different pixel extent but the same
    physical region.  write_html passes explicit x/y/width to addTiledImage so
    layers align in world coordinates regardless of pixel count.

    Returns dict mapping mag → RGBA Image.  Also stores per-layer metadata as
    image.info['x0_mm'], 'y0_mm', 'phys_w', 'phys_h', 'layer_ppm' for write_html.
    """
    x0_g, y0_g, x1_g, y1_g = all_bounds(scans)   # global (alignment) bounds

    # Alignment canvas — used only for ORB pixel arithmetic
    canvas_w = max(1, round((x1_g - x0_g) * canvas_ppm))
    canvas_h = max(1, round((y1_g - y0_g) * canvas_ppm))

    # Per-mag physical bounds (union of all scan_bounds at that mag)
    layer_bounds: dict = {}      # mag → [x0, y0, x1, y1]
    layer_native: dict = {}      # mag → max native ppm
    for scan in scans:
        mag = scan['mag']
        sb  = list(scan_bounds(scan))
        if mag not in layer_bounds:
            layer_bounds[mag] = sb
            layer_native[mag] = scan['ppm']
        else:
            b = layer_bounds[mag]
            b[0] = min(b[0], sb[0]);  b[1] = min(b[1], sb[1])
            b[2] = max(b[2], sb[2]);  b[3] = max(b[3], sb[3])
            layer_native[mag] = max(layer_native[mag], scan['ppm'])

    # Per-mag ppm: capped by this mag's OWN physical extent (not global)
    layer_ppms: dict = {}
    for mag, (lx0, ly0, lx1, ly1) in layer_bounds.items():
        ppm     = layer_native[mag]
        longest = max((lx1 - lx0) * ppm, (ly1 - ly0) * ppm)
        if longest > MAX_CANVAS_PX:
            ppm = ppm * MAX_CANVAS_PX / longest
        layer_ppms[mag] = ppm

    print(f"Global bounds: {x1_g-x0_g:.2f} × {y1_g-y0_g:.2f} mm")
    for mag in sorted(layer_ppms, key=lambda m: _MAG_SCALE[m]):
        lx0, ly0, lx1, ly1 = layer_bounds[mag]
        ppm = layer_ppms[mag]
        lw  = max(1, round((lx1 - lx0) * ppm))
        lh  = max(1, round((ly1 - ly0) * ppm))
        print(f"  Layer {mag}: {lw} × {lh} px  ({ppm:.0f} px/mm)  "
              f"extent {lx1-lx0:.2f}×{ly1-ly0:.2f} mm")

    layers: dict = {}   # mag → RGBA Image (with info dict for write_html)

    all_h_corrs: list = []
    all_v_corrs: list = []

    sorted_scans  = sorted(scans, key=lambda s: _MAG_SCALE[s['mag']])
    min_mag_scale = _MAG_SCALE[sorted_scans[0]['mag']]
    ref_canvas: 'Image.Image | None' = None

    for scan in sorted_scans:
        folder    = scan['folder']
        mag       = scan['mag']
        layer_ppm = layer_ppms[mag]
        lx0, ly0, lx1, ly1 = layer_bounds[mag]
        layer_w   = max(1, round((lx1 - lx0) * layer_ppm))
        layer_h   = max(1, round((ly1 - ly0) * layer_ppm))
        img_scale = layer_ppm / scan['ppm']   # ≈ 1.0; no cross-mag downscale
        dest_w    = max(1, round(scan['frame_w'] * img_scale))
        dest_h    = max(1, round(scan['frame_h'] * img_scale))
        is_ref_mag = _MAG_SCALE[mag] == min_mag_scale

        if mag not in layers:
            img = Image.new('RGBA', (layer_w, layer_h), (0, 0, 0, 0))
            img.info['x0_mm']    = lx0
            img.info['y0_mm']    = ly0
            img.info['phys_w']   = lx1 - lx0
            img.info['phys_h']   = ly1 - ly0
            img.info['layer_ppm'] = layer_ppm
            layers[mag] = img

        # Build ref_canvas (low-mag composite at canvas_ppm res) for cross-mag reg.
        if not is_ref_mag and ref_canvas is None and _HAVE_CV2:
            ref_canvas = Image.new('RGB', (canvas_w, canvas_h), (30, 30, 30))
            for m, lyr in layers.items():
                if _MAG_SCALE[m] == min_mag_scale:
                    mlx0, mly0, mlx1, mly1 = layer_bounds[m]
                    px_off = round((mlx0 - x0_g) * canvas_ppm)
                    py_off = round((mly0 - y0_g) * canvas_ppm)
                    tw = round((mlx1 - mlx0) * canvas_ppm)
                    th = round((mly1 - mly0) * canvas_ppm)
                    lyr_c = lyr.resize((tw, th), Image.LANCZOS)
                    ref_canvas.paste(lyr_c.convert('RGB'),
                                     (px_off, py_off), mask=lyr_c.split()[3])

        if refine and _HAVE_CV2:
            print(f"  ORB pre-computation [{mag} {folder.name}]…")
            grid, h_corr, v_corr, _ = _precompute_orb(scan, canvas_ppm)
            # positions are in global canvas_ppm space from (x0_g, y0_g)
            positions = _propagate_positions(grid, h_corr, v_corr, canvas_ppm,
                                             x0_g, y0_g)
            for v in h_corr.values():
                if v[2] != 0:
                    all_h_corrs.append((v[0], v[1], v[2]))
            for v in v_corr.values():
                if v[2] != 0:
                    all_v_corrs.append((v[0], v[1], v[2]))
            if not is_ref_mag and ref_canvas is not None:
                positions = _register_to_ref(scan, grid, ref_canvas,
                                             canvas_ppm, x0_g, y0_g, positions)
        else:
            grid = _build_grid(scan['images'])
            positions = None

        flat_canvas = None
        if correct_bg:
            flat_canvas = _load_saved_flat(mag)
            if flat_canvas is not None:
                print(f"    Flat [{mag}]: loaded flat_field_{mag}.npy")
            elif _HAVE_CV2:
                n_src  = len(scan['images'])
                n_used = min(n_src, _FLAT_MAX_TILES)
                flat_canvas = _build_flat_from_tiles(scan, dest_w, dest_h)
                if flat_canvas is not None:
                    print(f"    Flat [{mag}]: median of {n_used}/{n_src} scan tiles")
                else:
                    print(f"    Flat [{mag}]: only {n_src} tiles, skipping correction")
            else:
                print(f"    Flat [{mag}]: no saved flat_field_{mag}.npy found")

        layer = layers[mag]
        for i, ((row, col), img_meta) in enumerate(sorted(grid.items())):
            img_path = folder / img_meta['filename']
            if not img_path.exists():
                continue
            try:
                tile = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"  Warning: {img_meta['filename']}: {e}")
                continue

            if tile.width != scan['frame_w'] or tile.height != scan['frame_h']:
                actual_ppu = _px_per_um(scan['mag'], tile.width)
                tile_scale = layer_ppm / (actual_ppu * 1000.0)
                tile = tile.resize((max(1, round(tile.width  * tile_scale)),
                                    max(1, round(tile.height * tile_scale))),
                                   Image.LANCZOS)
            else:
                tile = tile.resize((dest_w, dest_h), Image.LANCZOS)

            if positions is not None:
                # Convert global canvas_ppm position → local layer_ppm position.
                # Global physical: x_phys = x0_g + cx_c/canvas_ppm
                # Local layer px:  cx_local = (x_phys - lx0) * layer_ppm
                cx_c, cy_c = positions[(row, col)]
                cx = (x0_g - lx0) * layer_ppm + cx_c * (layer_ppm / canvas_ppm)
                cy = (y0_g - ly0) * layer_ppm + cy_c * (layer_ppm / canvas_ppm)
            else:
                cx = (img_meta['x_mm'] - lx0) * layer_ppm
                cy = (img_meta['y_mm'] - ly0) * layer_ppm

            # Accumulate tile centres for JS coordinate lookup.
            # Store the raw (post-objective-offset, pre-ORB) motor positions as the
            # mm reference so that candidate x_mm/y_mm (same frame) can be subtracted
            # directly without ORB-correction contaminating the intra-tile offset.
            # cx/cy (canvas pixels) already embed the ORB shift for correct placement.
            layer.info.setdefault('tile_centres', []).append(
                (round(cx, 1), round(cy, 1),
                 round(img_meta['x_mm'], 6), round(img_meta['y_mm'], 6))
            )

            half_w = tile.width  / 2.0
            half_h = tile.height / 2.0
            px = math.floor(cx - half_w);  py = math.floor(cy - half_h)
            pw = math.ceil(cx + half_w) - px
            ph = math.ceil(cy + half_h) - py
            if tile.width != pw or tile.height != ph:
                tile = tile.resize((pw, ph), Image.LANCZOS)

            if flat_canvas is not None:
                tile = _apply_flat(tile, flat_canvas)

            if blend and _HAVE_CV2:
                _paste_blended_rgba(layer, tile, px, py)
            else:
                layer.paste(tile.convert('RGBA'), (px, py))

            if (i + 1) % 20 == 0 or (i + 1) == len(grid):
                print(f"  [{mag} {folder.name}]  {i+1}/{len(grid)}", end='\r')
        print()

    def _corr_stats(corrs, label):
        if not corrs:
            return
        dxs = [c[0] for c in corrs];  dys = [c[1] for c in corrs]
        n  = len(corrs)
        mx = sum(dxs)/n;  sx = (sum((v-mx)**2 for v in dxs)/n)**0.5
        my = sum(dys)/n;  sy = (sum((v-my)**2 for v in dys)/n)**0.5
        flag = '← systematic' if abs(mx) > sx or abs(my) > sy else '← jitter'
        print(f"  {label} ({n}):  "
              f"dx={mx:+.1f}±{sx:.1f} [{min(dxs):+.0f}…{max(dxs):+.0f}]  "
              f"dy={my:+.1f}±{sy:.1f} [{min(dys):+.0f}…{max(dys):+.0f}]  {flag}")

        def _clip_iqr(vals):
            s = sorted(vals); n = len(s)
            q1 = s[n // 4];  q3 = s[3 * n // 4];  iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            c = [v for v in vals if lo <= v <= hi]
            if not c: return None, None, 0, n
            mc = sum(c) / len(c)
            sc = (sum((v - mc) ** 2 for v in c) / len(c)) ** 0.5
            return mc, sc, len(c), n - len(c)

        mx_c, sx_c, n_c, n_out = _clip_iqr(dxs)
        my_c, sy_c, _,  _     = _clip_iqr(dys)
        if n_out > 0 and mx_c is not None:
            print(f"    clipped ({n_c}/{n}, {n_out} outlier(s) removed):  "
                  f"dx={mx_c:+.1f}±{sx_c:.1f}  dy={my_c:+.1f}±{sy_c:.1f}")

        orb = [c for c in corrs if c[2] > 0]
        if orb and len(orb) < n:
            odxs = [c[0] for c in orb];  odys = [c[1] for c in orb]
            no = len(orb)
            mox = sum(odxs)/no;  sox = (sum((v-mox)**2 for v in odxs)/no)**0.5
            moy = sum(odys)/no;  soy = (sum((v-moy)**2 for v in odys)/no)**0.5
            print(f"    ORB-only ({no}/{n}):  "
                  f"dx={mox:+.1f}±{sox:.1f}  dy={moy:+.1f}±{soy:.1f}")

    if refine and (all_h_corrs or all_v_corrs):
        print("\nORB corrections (px, centre-out BFS from metadata step sizes):")
        _corr_stats(all_h_corrs, "H-pairs")
        _corr_stats(all_v_corrs, "V-pairs")

    return layers


def _extract_camera_rotation_deg(h_corr: dict, scan: dict, canvas_ppm: float) -> float | None:
    """
    Estimate camera-to-stage rotation angle from H-pair cross-axis (dy) corrections.

    When the camera is rotated by θ relative to the stage X axis, a rightward stage
    move of step_x produces an image displacement of (step_x·cosθ, step_x·sinθ)
    in canvas space.  The cross-axis component step_x·sinθ is exactly what ORB
    measures as H-pair dy.  We use the median of ORB-only pairs (tightest std).

    Returns angle in degrees, or None if insufficient data.
    """
    sp = scan.get('scan_params', {})
    step_x_mm = sp.get('step_x', scan['frame_w'] / scan['ppm'] * (1 - _ORB_OVERLAP_FRAC))
    step_x_canvas = step_x_mm * canvas_ppm

    # Prefer ORB-matched pairs only (n>0); fall back to PC (n==-1)
    orb_pairs = [v for v in h_corr.values() if v[2] > 0]
    if len(orb_pairs) < 5:
        orb_pairs = [v for v in h_corr.values() if v[2] != 0]
    if len(orb_pairs) < 5:
        return None

    import statistics
    dy_median = statistics.median(v[1] for v in orb_pairs)
    theta_rad = math.atan2(dy_median, step_x_canvas)
    return math.degrees(theta_rad)


def _rotation_model_positions(grid: dict, theta_deg: float,
                              canvas_ppm: float, x0: float, y0: float) -> dict:
    """
    Place tiles using a rigid rotation model instead of BFS + per-pair corrections.

    Given camera rotation θ (degrees), a stage displacement (Δx, Δy) produces
    image displacement R(θ) @ (Δx, Δy) in canvas space.  All tile positions are
    computed analytically relative to the central reference tile — no error
    accumulation, no ORB needed.  Outlier pairs and bad ORB matches have zero
    influence.

    Returns dict: (row, col) → (cx_float, cy_float) canvas centre in pixels.
    """
    cos_t = math.cos(math.radians(theta_deg))
    sin_t = math.sin(math.radians(theta_deg))

    all_cells = sorted(grid)
    rows = sorted({r for r, c in all_cells})
    cols = sorted({c for r, c in all_cells})
    mid_row = rows[len(rows) // 2]
    mid_col = cols[len(cols) // 2]
    ref_rc = min(all_cells, key=lambda rc: abs(rc[0] - mid_row) + abs(rc[1] - mid_col))
    ref_im = grid[ref_rc]
    ref_cx = (ref_im['x_mm'] - x0) * canvas_ppm
    ref_cy = (ref_im['y_mm'] - y0) * canvas_ppm

    positions: dict = {}
    for (row, col), img_meta in grid.items():
        dx_mm = img_meta['x_mm'] - ref_im['x_mm']
        dy_mm = img_meta['y_mm'] - ref_im['y_mm']
        cx = ref_cx + (dx_mm * cos_t - dy_mm * sin_t) * canvas_ppm
        cy = ref_cy + (dx_mm * sin_t + dy_mm * cos_t) * canvas_ppm
        positions[(row, col)] = (cx, cy)
    return positions

"""Flake detector v3 — colour-axis projection with spatial noise averaging.

Motivation (see CLAUDE.md #16 and the 2026-06-22 investigation):
  v2 thresholds per-pixel Weber contrast against per-channel target boxes, then
  scores candidates by flatness.  On dark/underexposed tiles the per-pixel
  contrast noise (~9% on R when the R mean is ~40/255) is half the monolayer
  signal (~19%), so plain substrate floods the masks; and rewarding flatness
  floats bare substrate to the top of the ranking.

v3 fixes the architecture:
  1. LOCAL background (downscale → blur → upscale) so residual vignetting does
     not manufacture spurious contrast far from a flake.
  2. Project Weber contrast onto the CALIBRATED graphene colour axis
     (u = monolayer_target / |monolayer_target|).  This collapses BGR to one
     discriminating scalar using the empirical spectral signature (R-dominant
     at ~300 nm SiO2, G-dominant at 90 nm) instead of hand-set per-channel boxes.
  3. SPATIALLY SMOOTH the projection before thresholding.  A flake is hundreds
     of coherent pixels; noise averages down as 1/sqrt(N).  This is the same
     averaging the calibration did with its 3 µm crops.
  4. Emit FEATURE-RICH candidates (projection magnitude, within-region
     uniformity, spectral cosine, brightness, edge gradient, shape) so a
     trained classifier — not hand-tuned gates — can do precision filtering
     once a labelled set exists (tools/flake_label.py + tools/flake_detect_eval.py).

v3 is a candidate GENERATOR.  Its built-in `score` is provisional; the intended
workflow is generate → label → fit precision filter.  Do not treat raw v3
output as a clean flake list.

CLI:
    python -m vision.flake_detect_v3 <scan_folder> [--calibration PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Allow running as a plain script (python3 vision/flake_detect_v3.py …), not just -m.
import sys as _sys
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parent.parent
if str(_REPO) not in _sys.path:
    _sys.path.insert(0, str(_REPO))

from core.rig_calibration import camera_rotation_deg  # noqa: E402
from core.scan_io import load_metadata, scan_mag      # noqa: E402
from vision.flat_field import apply_flat_field, compute_flat_field  # noqa: E402

# Rehomed from the retired v2 detector (plan U2): the only other
# consumers were the v2 panel and its stale tests.
from vision.camera_params import px_per_um as _camera_ppm  # noqa: E402


def _px_per_um(mag: str, frame_w: int) -> float:
    ppm = _camera_ppm(mag, frame_w)
    return ppm if ppm is not None else 1.0


def _make_id(x_mm: float, y_mm: float, area_um2: float) -> str:
    key = f"{x_mm:.4f},{y_mm:.4f},{area_um2:.1f}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def deduplicate(candidates: list, dedup_radius_um: float = 50.0) -> list:
    """Cluster candidates within dedup_radius_um; keep best per cluster.

    Priority: focus_ok=True first, then larger area.
    """
    if not candidates:
        return []

    sorted_c = sorted(candidates, key=lambda c: (not c["focus_ok"], -c["area_um2"]))
    radius_mm = dedup_radius_um / 1000.0
    merged = [False] * len(sorted_c)
    kept = []

    for i, c in enumerate(sorted_c):
        if merged[i]:
            continue
        kept.append(c)
        xi, yi = c["x_mm"], c["y_mm"]
        for j in range(i + 1, len(sorted_c)):
            if merged[j]:
                continue
            if math.hypot(sorted_c[j]["x_mm"] - xi, sorted_c[j]["y_mm"] - yi) < radius_mm:
                merged[j] = True

    return kept


@dataclass
class V3Config:
    """Configuration for detect_v3.  Defaults from the 2026-06-22 Graphene_A study."""

    # Spatial averaging of the projection / contrast maps (full-res px).
    # 0 = OFF (default): with good focus the raw projection is sharp, and any blur
    # bleeds a 1L/2L layer step across its boundary → merges regions and drags the
    # detected outline off the true edge.  Raise only for genuinely noisy scans.
    smooth_sigma_px: float = 0.0

    # Detection thresholds.
    #   proj_thresh: minimum smoothed projection onto the graphene colour axis.
    #     Substrate ≈ 0, monolayer ≈ |t1| (~0.23).  GT 1L core ~0.14, so 0.13
    #     keeps real monolayers while trimming faint marginal detections.
    #   spectral_cos_min: contrast direction must align with the graphene axis.
    proj_thresh: float = 0.13
    spectral_cos_min: float = 0.85
    # Chromaticity gate: a component's mean contrast vector must be closer (by
    # cosine) to the graphene colour axis than to NEUTRAL GREY, by this margin.
    # spectral_cos_min alone is too weak at thin oxide (~100 nm) where the graphene
    # axis is only ~0.95 from grey, so spectrally-flat junk (defocus/vignette
    # shadows, tape halos) with cos≈0.93–0.97 leaks through.  Requiring
    # cos(axis) − cos(grey) > margin rejects achromatic darkening regardless of
    # its cosine.  0.0 = must be at least as graphene-like as grey.
    chroma_margin: float = 0.0

    # Morphology (full-res px).
    k_open: int = 7
    k_close: int = 15

    # Component filters.  Minimum area is in µm² (converted to px per-frame via
    # the magnification's px/µm), so it's mag-independent.
    min_area_um2: float = 100.0
    max_area_frac: float = 0.20
    min_solidity: float = 0.55       # compact; rejects wispy/fragmented shadows
    max_aspect_ratio: float = 5.0

    # Uniformity WITHIN a layer region — a real flake region is a flat crystal at
    # one thickness → low projection variance.  Mottled junk varies more.
    max_proj_cv: float = 0.50

    # Reject components brighter than local background (contamination glints).
    # A real flake is DARKER; require mean brightness ≤ bg × (1 + this).
    max_bright_excess: float = 0.02
    # Mean-brightness misses a dark flake with a small bright inclusion (glint):
    # also reject if more than this FRACTION of the component's pixels are brighter
    # than the substrate.
    max_bright_frac: float = 0.06

    pixel_size_um: Optional[float] = None  # override auto-detect from mag


# Unit direction of a neutral-grey DARKENING (all channels equally darker).
# A defocus/vignette shadow or tape halo darkens achromatically → contrast ∥ this.
_GREY_DARK = (-np.ones(3) / np.sqrt(3.0)).astype(np.float64)


# ── target loading ──────────────────────────────────────────────────────────

def load_targets(calibration_path, layers=None):
    """Load per-layer BGR Weber-fraction targets from contrast_calibration.json.

    Returns (targets, axis, target_proj, proj_max):
      targets:     {N: np.array([C_B, C_G, C_R])}   signed fractions
      axis:        unit vector along the monolayer target (graphene colour axis)
      target_proj: {N: float}  projection of each target onto axis
      proj_max:    float  upper bound of the TOP layer bin — a candidate whose
                   projection exceeds this is thicker than the top target (bulk
                   graphite / multilayer), NOT the top layer.  Without it the
                   nearest-target rule is unbounded above and dumps thick graphite
                   (proj ≫ target) into the top bin (observed on Q/R at 100 nm).
                   = top_proj + ½·(top − prev) spacing; +inf if only one layer.
    contrast_calibration.json stores mean_bgr_pct in PERCENT → divide by 100.
    """
    cal = json.loads(Path(calibration_path).read_text())
    # `targets` = the (thin, monotonic-projection) rungs that drive segmentation; read
    # ALL of them (analytical calibrations now carry the whole resolvable thin band,
    # not just 1/2/3).  `layers`, if given, still filters.
    targets = {}
    for _, t in cal.get("targets", {}).items():
        lc = int(t.get("layer_count"))
        if layers is None or lc in layers:
            targets[lc] = np.array(t["mean_bgr_pct"], dtype=np.float64) / 100.0
    if not targets:
        raise ValueError(f"no targets in {calibration_path}")
    base = targets.get(min(targets))  # monolayer (or thinnest available)
    axis = base / np.linalg.norm(base)
    target_proj = {N: float(targets[N] @ axis) for N in targets}
    ordered = sorted(target_proj)
    if len(ordered) >= 2:
        top, prev = target_proj[ordered[-1]], target_proj[ordered[-2]]
        proj_max = top + 0.5 * (top - prev)
    else:
        proj_max = float("inf")
    # `ladder` = ALL resolvable rungs (thin + thick return branch) in full BGR (percent),
    # for nearest-rung LAYER CLASSIFICATION that de-aliases thick flakes.  None if absent
    # (empirical calibrations) → detector falls back to nearest-projection labelling.
    ladder = None
    ladder_sigma = None
    ladder_resolvable = None
    if cal.get("ladder"):
        ladder = {int(t["layer_count"]): np.array(t["mean_bgr_pct"], float)
                  for t in cal["ladder"].values()}
        ladder_sigma = {int(t["layer_count"]): t.get("layer_sigma")
                        for t in cal["ladder"].values()}
        ladder_resolvable = {int(t["layer_count"]): t.get("resolvable")
                             for t in cal["ladder"].values()}
        # With a ladder, segmentation must NOT reject the darkest flakes: the fold peak
        # (~10-15 L) projects DARKER than the thin band, so a proj_max tied to the thin
        # rungs would throw those flakes away before the classifier sees them (the
        # 8-19 L histogram gap).  Extend the cap to cover the whole ladder's projection
        # range; the ladder classifier + chroma gate + opacity ceiling handle thickness.
        lad_proj = [float((ladder[M] / 100.0) @ axis) for M in ladder]
        proj_max = max(proj_max, max(lad_proj) * 1.1)
    return (targets, axis, target_proj, proj_max,
            ladder, ladder_sigma, ladder_resolvable)


def layer_bins(target_proj, proj_thresh, proj_max):
    """Projection interval [lo, hi) assigned to each layer N — midpoints between
    adjacent target projections (proj_thresh below the thinnest, proj_max above
    the thickest).  Used to report a component's per-pixel layer composition."""
    ns = sorted(target_proj)
    bins = {}
    for i, N in enumerate(ns):
        lo = proj_thresh if i == 0 else (target_proj[ns[i - 1]] + target_proj[N]) / 2
        hi = proj_max if i == len(ns) - 1 else (target_proj[N] + target_proj[ns[i + 1]]) / 2
        bins[N] = (lo, hi)
    return bins


# ── core ──────────────────────────────────────────────────────────────────────

# global_substrate moved to vision/contrast_cal (plan task L4) — the ladder
# calibrator needs it too; re-exported here for existing importers.
from vision.contrast_cal import global_substrate  # noqa: E402,F401


def _edge_gradient(gray: np.ndarray, comp_mask: np.ndarray) -> float:
    """Mean Sobel magnitude on the 1-px perimeter ring of a component."""
    er = cv2.erode(comp_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    ring = (comp_mask > 0) & (er == 0)
    if not ring.any():
        return 0.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return float(mag[ring].mean())


def detect_v3(image_bgr, x_mm, y_mm, z_mm, mag, source_filename, axis, target_proj=None,
              *, proj_max=float("inf"), ladder=None, ladder_sigma=None,
              ladder_resolvable=None, substrate=None,
              focus_ok=True, camera_rotation_deg=0.0,
              config: Optional[V3Config] = None) -> list:
    """Detect flake candidates in one (flat-field-corrected) frame.

    `axis` (and optionally `target_proj`) come from load_targets(); `target_proj`
    may be None for pre-calibration/bootstrap runs (target_N is then None).
    `substrate` (REQUIRED, BGR vector from global_substrate()) is the reference
    the contrast is measured against — the global modal wafer colour, unbiased by
    flake size. Returns a list of feature-rich candidate dicts (see module docstring).
    """
    cfg = config or V3Config()
    if substrate is None:
        raise ValueError("detect_v3 requires a global modal `substrate` (BGR) reference; "
                         "see global_substrate(). A per-frame local background is unusable on "
                         "flat-fielded scans — it's inflated by contamination and self-biased "
                         "by large flakes.")
    f = image_bgr.astype(np.float32)
    h, w = f.shape[:2]
    ppu = cfg.pixel_size_um and (1.0 / cfg.pixel_size_um) or _px_per_um(mag, w)
    min_area_px = cfg.min_area_um2 * ppu ** 2          # µm² → px at this mag

    bg = np.empty_like(f)
    bg[:] = np.asarray(substrate, np.float32)             # global modal substrate
    C = (f - bg) / np.maximum(bg, 1e-3)                    # BGR Weber contrast
    proj = C @ axis                                        # + = darker, graphene-aligned
    if cfg.smooth_sigma_px > 0:
        proj_s = cv2.GaussianBlur(proj, (0, 0), cfg.smooth_sigma_px)
        C_s = cv2.GaussianBlur(C, (0, 0), cfg.smooth_sigma_px)
    else:
        proj_s, C_s = proj, C          # no blur — sharp, focus-limited edges
    cos = np.divide(C_s @ axis, np.maximum(np.linalg.norm(C_s, axis=2), 1e-4))

    gray = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg_gray = cv2.cvtColor(np.clip(bg, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    now = datetime.now().isoformat()

    # Per-LAYER segmentation: one mask per layer's projection interval, so a stepped
    # flake (1L step + 2L body) yields a 1L region AND a 2L region — each labelled by
    # its own layer and landing in the right list.  No blur → the layer step is a real
    # boundary between the two masks.  Pixels above the top bin (proj ≥ proj_max) fall
    # in no bin → thick graphite is excluded for free.  Bootstrap (no calibration) →
    # one bin covering all above-threshold pixels, label None.
    if target_proj:
        bins = layer_bins(target_proj, cfg.proj_thresh, proj_max)
    else:
        bins = {None: (cfg.proj_thresh, float("inf"))}
    chromatic = cos > cfg.spectral_cos_min
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.k_open, cfg.k_open))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.k_close, cfg.k_close))
    a_rot = math.radians(-camera_rotation_deg)
    ca, sa = math.cos(a_rot), math.sin(a_rot)

    out = []
    for N, (lo, hi) in bins.items():
        lmask = (chromatic & (proj_s >= lo) & (proj_s < hi)).astype(np.uint8)
        lmask = cv2.morphologyEx(lmask, cv2.MORPH_OPEN, k_open)
        lmask = cv2.morphologyEx(lmask, cv2.MORPH_CLOSE, k_close)
        nc, lab, st, cen = cv2.connectedComponentsWithStats(lmask)
        for i in range(1, nc):
            area_px = int(st[i, cv2.CC_STAT_AREA])
            if area_px < min_area_px or area_px > cfg.max_area_frac * h * w:
                continue
            comp = (lab == i).astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            outer = max(cnts, key=cv2.contourArea)
            hull_area = cv2.contourArea(cv2.convexHull(outer))
            solidity = area_px / max(hull_area, 1.0)
            if solidity < cfg.min_solidity:
                continue
            bw, bh = int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT])
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if aspect > cfg.max_aspect_ratio:
                continue

            cmask = comp.astype(bool)
            mean_bgr = f[cmask].mean(0)
            bg_bgr = bg[cmask].mean(0)
            brightness = float(gray[cmask].mean())
            bg_bright = float(bg_gray[cmask].mean())
            if brightness > bg_bright * (1 + cfg.max_bright_excess):
                continue  # brighter than substrate → contamination glint, not a flake
            bright_frac = float((gray[cmask] > bg_bright).mean())
            if bright_frac > cfg.max_bright_frac:
                continue  # localized bright inclusion (glint) the mean misses

            # Chromaticity gate: reject achromatic (grey) darkening — a defocus/
            # vignette shadow or tape halo has contrast ∥ neutral grey, which at thin
            # oxide sits only ~0.95 from the graphene axis and slips past
            # spectral_cos_min.  Require the mean contrast more graphene-like than grey.
            cvec = (mean_bgr - bg_bgr) / np.maximum(bg_bgr, 1e-3)   # Weber, matches `axis`
            nv = float(np.linalg.norm(cvec))
            if nv > 1e-6:
                vhat = cvec / nv
                if float(vhat @ axis) - float(vhat @ _GREY_DARK) <= cfg.chroma_margin:
                    continue

            pin = proj_s[cmask]
            proj_mean = float(pin.mean())
            proj_cv = float(pin.std() / max(proj_mean, 1e-3))
            if proj_cv > cfg.max_proj_cv:
                continue  # non-uniform within the layer region → mottled junk
            spectral_cos = float(cos[cmask].mean())
            edge = _edge_gradient(gray, comp)

            cx, cy = float(cen[i][0]), float(cen[i][1])
            # Pixel offset from tile centre → stage-frame mm.  Image axes (verified
            # against map tile placement and a ground-truth flake): +col(right) =
            # stage +x, +row(down) = stage +y.  Rotate by the camera-to-stage angle
            # (CLAUDE.md #7/#47: every stage↔pixel transform MUST apply the angle;
            # stage→canvas uses R(+θ) so pixel→stage is R(−θ)).  Objective offset is
            # left to make_map — common-mode with the tiles.
            dxp = (cx - w / 2.0) / ppu / 1000.0
            dyp = (cy - h / 2.0) / ppu / 1000.0
            flake_x = x_mm + dxp * ca - dyp * sa
            flake_y = y_mm + dxp * sa + dyp * ca
            area_um2 = round(area_px / ppu ** 2, 1)
            score = round(spectral_cos * max(0.0, 1.0 - proj_cv) * min(1.0, edge / 3.0), 4)
            contour_px = [[int(px), int(py)]
                          for px, py in cv2.approxPolyDP(outer, 2.0, True).reshape(-1, 2)]

            # Layer classification: assign the region to its nearest RESOLVABLE ladder
            # rung in full BGR (de-aliases thick flakes that project into a thin bin).
            # Falls back to the projection-bin label N when no ladder is present.
            cbgr = np.array([(mean_bgr[c] - bg_bgr[c]) / max(bg_bgr[c], 1e-3) * 100.0
                             for c in range(3)])
            layer_N = N
            ladder_dist = None
            layer_sigma = None
            layer_resolvable = None
            if ladder:
                # Best guess = nearest rung over the WHOLE ladder (incl. the turn), so a
                # ~12 L flake pins to ~12 with a large layer_sigma, not to a resolvable
                # edge (7/26) with a falsely-small one.
                layer_N = min(ladder, key=lambda M: float(np.linalg.norm(ladder[M] - cbgr)))
                ladder_dist = round(float(np.linalg.norm(ladder[layer_N] - cbgr)), 2)
                if ladder_sigma:
                    layer_sigma = ladder_sigma.get(layer_N)
                if ladder_resolvable:
                    layer_resolvable = ladder_resolvable.get(layer_N)

            out.append({
                "id": _make_id(flake_x, flake_y, area_um2),
                "x_mm": round(flake_x, 4),
                "y_mm": round(flake_y, 4),
                "z_mm": z_mm,
                "mag": mag,
                "target_N": layer_N,
                "layer_sigma": layer_sigma,
                "layer_resolvable": layer_resolvable,
                "seg_N": N,
                "ladder_dist": ladder_dist,
                "proj": round(proj_mean, 4),
                "proj_cv": round(proj_cv, 3),
                "spectral_cos": round(spectral_cos, 3),
                "contrast_bgr_pct": [round(float((mean_bgr[c] - bg_bgr[c]) / bg_bgr[c] * 100), 2)
                                     for c in range(3)],
                "mean_bgr": [round(float(v), 1) for v in mean_bgr],
                "bg_bgr": [round(float(v), 1) for v in bg_bgr],
                "brightness": round(brightness, 1),
                "edge_grad": round(edge, 2),
                "area_um2": area_um2,
                "area_px": area_px,
                "solidity": round(solidity, 3),
                "aspect_ratio": round(aspect, 2),
                "bbox_px": [int(st[i, 0]), int(st[i, 1]), bw, bh],
                "centroid_px": [round(cx, 1), round(cy, 1)],
                "contour_px": contour_px,
                "source_image": Path(source_filename).name,
                "focus_ok": focus_ok,
                "score": score,
                "detected_at": now,
            })
    return out


# ── Parallel per-tile detection (process pool) ──────────────────────────────
# The per-tile work (imread + detect_v3) is independent and largely GIL-bound
# Python compute (region loops, ladder classification), so PROCESSES — not
# threads — give the speedup.  Read-only shared state (flat, substrate, ladder,
# axis…) is sent ONCE per worker via the initializer, not per tile.  Each worker
# is pinned to a single CV/BLAS thread so N processes use N cores cleanly, with
# no oversubscription.  (Measured: detector is ~67% compute / 33% I/O — #58.)
_W: dict = {}


def _worker_init(state: dict) -> None:
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
    _W.update(state)


def _worker_detect(task):
    """Detect one tile in a pool worker.  task = (filename, x_mm, y_mm, z_mm, focus_ok)."""
    fn, x_mm, y_mm, z_mm, focus_ok = task
    try:
        p = _W["folder"] / fn
        frame = cv2.imread(str(p))
        if frame is None:
            return []
        flat = _W["flat"]
        corr = apply_flat_field(frame, flat) if flat is not None else frame
        return detect_v3(
            corr, x_mm, y_mm, z_mm, _W["mag"], fn,
            _W["axis"], _W["target_proj"], proj_max=_W["proj_max"],
            ladder=_W["ladder"], ladder_sigma=_W["ladder_sigma"],
            ladder_resolvable=_W["ladder_resolvable"], substrate=_W["substrate"],
            focus_ok=focus_ok, camera_rotation_deg=_W["cam_rot"], config=_W["config"])
    except Exception as e:  # a bad tile must not kill the whole scan
        print(f"  ! tile {fn} failed: {e}")
        return []


def _resolve_workers(requested, n_tiles: int) -> int:
    """How many processes to use.  requested None/<=0 → auto (cores−2, capped by
    tiles).  Serial (1) below a small-scan threshold where pool setup isn't worth it."""
    if requested and requested > 0:
        return max(1, min(requested, n_tiles))
    # Auto: leave headroom for the GUI/preview that launched this as a subprocess.
    # Capped at 16 — beyond that the win tapers (disk-read + dedup are serial) and
    # a 30-proc grab on a 32-core box makes the live UI sluggish.  Override with
    # --workers for a headless max-throughput batch run.
    auto = max(1, min((os.cpu_count() or 2) - 2, 16))
    if n_tiles < 8:            # tiny scan — pool spin-up costs more than it saves
        return 1
    return max(1, min(auto, n_tiles))


def run_v3(scan_folder, calibration_path=None, *, config: Optional[V3Config] = None,
           flat_field_method="median", dedup_radius_um=50.0, progress_cb=None,
           timing=False, workers=None) -> list:
    """Run v3 over a completed scan; writes <scan>/flake_candidates_v3.json.

    Contrast is measured against the global modal wafer colour (a sample of
    tiles), not a per-frame local background.  A local background is inflated
    next to bright contamination and self-biased by large flakes, so bare
    substrate beside tape residue reads as a spurious dark "flake"; it also
    cannot tell a real oxide-thickness gradient from a flake.  The global modal
    reference keeps the sign honest (real flakes are darker than modal).

    `timing=True` prints a stage breakdown (flat / substrate / I/O read vs
    per-tile compute / dedup+write) — the I/O-vs-compute split tells you whether
    GPU offload could help at all before doing it (#58).
    """
    t0 = time.perf_counter()
    t_io = t_detect = 0.0
    folder = Path(scan_folder)
    meta = load_metadata(folder)
    if calibration_path is None:
        calibration_path = folder / "contrast_calibration.json"
    (targets, axis, target_proj, proj_max,
     ladder, ladder_sigma, ladder_resolvable) = load_targets(calibration_path)

    mag = scan_mag(meta, default="20x")
    images = meta.get("images", [])

    flat = None
    t_flat = time.perf_counter()
    if flat_field_method != "none":
        flat = compute_flat_field(folder, method=flat_field_method,
                                  progress_cb=lambda i, n: None)
    t_flat = time.perf_counter() - t_flat

    t_sub = time.perf_counter()
    substrate = global_substrate(folder, images, flat=flat)
    t_sub = time.perf_counter() - t_sub
    print(f"  global modal substrate BGR = {[round(float(v), 1) for v in substrate]}")

    cam_rot = camera_rotation_deg()
    print(f"  camera rotation = {cam_rot:+.4f}°")

    # Build the task list once (skip missing files up front).
    tasks = []
    for im in images:
        fn = im.get("filename")
        if not fn or not (folder / fn).exists():
            continue
        tasks.append((fn, float(im.get("x_mm", 0)), float(im.get("y_mm", 0)),
                      float(im.get("z_actual_mm") or im.get("z_mm", 0)),
                      bool(im.get("focus_ok", True))))

    n_workers = _resolve_workers(workers, len(tasks))
    all_c = []
    t_tiles = time.perf_counter()
    if n_workers <= 1:
        # Serial path — keeps the fine I/O-vs-compute timing split.
        for idx, (fn, x_mm, y_mm, z_mm, focus_ok) in enumerate(tasks):
            if progress_cb:
                progress_cb(idx, len(tasks))
            _t = time.perf_counter()
            frame = cv2.imread(str(folder / fn))
            if frame is None:
                continue
            corr = apply_flat_field(frame, flat) if flat is not None else frame
            t_io += time.perf_counter() - _t
            _t = time.perf_counter()
            all_c += detect_v3(
                corr, x_mm, y_mm, z_mm, mag, fn, axis, target_proj, proj_max=proj_max,
                ladder=ladder, ladder_sigma=ladder_sigma,
                ladder_resolvable=ladder_resolvable, substrate=substrate,
                focus_ok=focus_ok, camera_rotation_deg=cam_rot, config=config)
            t_detect += time.perf_counter() - _t
    else:
        # Parallel path — one process per core (minus 2).  Shared read-only state
        # goes to workers once via the initializer.
        print(f"  detecting {len(tasks)} tiles on {n_workers} processes…")
        state = dict(folder=folder, flat=flat, mag=mag, axis=axis,
                     target_proj=target_proj, proj_max=proj_max, ladder=ladder,
                     ladder_sigma=ladder_sigma, ladder_resolvable=ladder_resolvable,
                     substrate=substrate, cam_rot=cam_rot, config=config)
        chunksize = max(1, len(tasks) // (n_workers * 8))
        done = 0
        # MUST use 'spawn', not the Linux default 'fork': global_substrate/flat run
        # cv2 + OpenMP/BLAS BEFORE the pool starts, spinning up thread pools; a
        # fork()'d child inherits those locked mutexes and deadlocks on its first
        # cv2 call (seen: 17 procs pinned at ~0% CPU forever).  spawn starts clean
        # interpreters (re-imports the module) so no inherited locks.
        import multiprocessing as _mp
        _ctx = _mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=_ctx,
                                 initializer=_worker_init, initargs=(state,)) as ex:
            for res in ex.map(_worker_detect, tasks, chunksize=chunksize):
                all_c += res
                done += 1
                if progress_cb and (done % 25 == 0 or done == len(tasks)):
                    progress_cb(done, len(tasks))
    t_tiles = time.perf_counter() - t_tiles
    if progress_cb:
        progress_cb(len(tasks), len(tasks))

    # Dedup PER LAYER — a stepped flake's 1L and 2L regions share a location, so
    # deduping across layers would drop one.  Only merge same-target_N duplicates.
    t_post = time.perf_counter()
    candidates = []
    by_layer: dict = {}
    for c in all_c:
        by_layer.setdefault(c.get("target_N"), []).append(c)
    for group in by_layer.values():
        candidates.extend(deduplicate(group, dedup_radius_um=dedup_radius_um))
    # Largest-first by default — area is the trustworthy, interpretable ranking;
    # `score` is provisional (to be replaced by a trained classifier).
    candidates.sort(key=lambda c: c.get("area_um2", 0), reverse=True)
    out_path = folder / "flake_candidates_v3.json"
    out_path.write_text(json.dumps(candidates, indent=2))
    t_post = time.perf_counter() - t_post

    if timing:
        n = len(tasks)
        total = time.perf_counter() - t0
        print("\n⏱ flake_detect_v3 timing breakdown:")
        rows = [("flat-field", t_flat), ("global substrate", t_sub)]
        if n_workers <= 1:
            rows += [(f"tile I/O read ({n} tiles)", t_io),
                     ("per-tile detect (compute)", t_detect)]
        else:
            rows += [(f"tiles ({n}) on {n_workers} procs", t_tiles)]
        rows += [("dedup + write", t_post)]
        for name, v in rows:
            pct = 100 * v / total if total else 0.0
            print(f"    {name:32s} {v:8.2f}s  ({pct:4.1f}%)")
        print(f"    {'TOTAL':32s} {total:8.2f}s")
        if n_workers <= 1:
            print(f"    per-tile: {1000*(t_io+t_detect)/max(n,1):.1f} ms  "
                  f"(I/O {100*t_io/max(t_io+t_detect,1e-9):.0f}% / "
                  f"compute {100*t_detect/max(t_io+t_detect,1e-9):.0f}%)")
        else:
            print(f"    tiles: {n/max(t_tiles,1e-9):.1f} tiles/s "
                  f"(wall {1000*t_tiles/max(n,1):.1f} ms/tile across {n_workers} procs)")
    return candidates


def _main():
    ap = argparse.ArgumentParser(description="Flake detector v3 (colour-axis projection)")
    ap.add_argument("scan_folder")
    ap.add_argument("--calibration", default=None,
                    help="contrast_calibration.json (default: <scan>/contrast_calibration.json)")
    ap.add_argument("--proj-thresh", type=float, default=V3Config.proj_thresh)
    ap.add_argument("--spectral-cos-min", type=float, default=V3Config.spectral_cos_min)
    ap.add_argument("--smooth-sigma-px", type=float, default=V3Config.smooth_sigma_px)
    ap.add_argument("--min-area-um2", type=float, default=V3Config.min_area_um2,
                    help="minimum candidate area in µm² (default 100)")
    ap.add_argument("--min-solidity", type=float, default=V3Config.min_solidity)
    ap.add_argument("--max-proj-cv", type=float, default=V3Config.max_proj_cv,
                    help="max projection CV — uniformity gate (default 0.50)")
    ap.add_argument("--max-bright-frac", type=float, default=V3Config.max_bright_frac,
                    help="max fraction of pixels brighter than substrate — glint gate")
    ap.add_argument("--chroma-margin", type=float, default=V3Config.chroma_margin,
                    help="min cos(axis)−cos(grey) — rejects achromatic junk (default 0.0)")
    ap.add_argument("--dedup-radius-um", type=float, default=50.0)
    ap.add_argument("--timing", action="store_true",
                    help="print a per-stage breakdown (I/O read vs per-tile compute) "
                         "to find the real bottleneck before GPU offload (#58)")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel tile-detect processes (default: auto = cores−2; "
                         "1 = serial). The per-tile loop is compute-bound, so this is "
                         "the main speedup — ~cores× on a big scan (#58)")
    args = ap.parse_args()
    cfg = V3Config(proj_thresh=args.proj_thresh, spectral_cos_min=args.spectral_cos_min,
                   smooth_sigma_px=args.smooth_sigma_px, min_area_um2=args.min_area_um2,
                   min_solidity=args.min_solidity, max_proj_cv=args.max_proj_cv,
                   max_bright_frac=args.max_bright_frac, chroma_margin=args.chroma_margin)
    print(f"v3 detect: {args.scan_folder}")
    res = run_v3(args.scan_folder, args.calibration, config=cfg,
                 dedup_radius_um=args.dedup_radius_um, timing=args.timing,
                 workers=args.workers,
                 progress_cb=lambda i, n: print(f"\r  {i}/{n}", end="", flush=True))
    from collections import Counter
    print(f"\n{len(res)} candidates → flake_candidates_v3.json"
          f"  by N: {dict(Counter(c['target_N'] for c in res))}")


if __name__ == "__main__":
    _main()

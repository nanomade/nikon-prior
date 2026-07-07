"""Contrast measurement + calibration targets — one home (plan task L4).

Consolidates the pieces that previously lived scattered:
  global_substrate       — was vision/flake_detect_v3 (detector + ladder both need it)
  wand_measure/fit_ladder — was tools/calibrate_ladder (the validated #49 method)
  write_contrast_calibration — THE writer for contrast_calibration.json, now
                           provenance-stamped (core.provenance). All producers
                           (wand ladder, analytical oxide targets) route here so
                           every calibration on disk says what made it.

The retired alternatives (histogram-segment and polygon-draw contrast
measurement, physics pipeline classifier) are in attic/ with README notes.

Conventions: contrast is measured against the GLOBAL modal substrate, never a
per-frame local background (docs/FINDINGS.md; local backgrounds are inflated
by tape residue). Contrast fraction c = (I − S) / S per BGR channel.
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np

from vision.flat_field import apply_flat_field


# ── substrate ─────────────────────────────────────────────────────────────────

def global_substrate(folder, images, flat=None, n_tiles: int = 80,
                     downscale: int = 4, seed: int = 0) -> np.ndarray:
    """Modal BGR colour across a sample of (flat-fielded) scan tiles = the wafer.

    The mode of millions of substrate pixels — far more robust and unbiased than a
    per-frame local background (which a large flake biases toward itself). Returns
    a BGR float32 vector."""
    import random
    folder = Path(folder)
    rng = random.Random(seed)
    sample = rng.sample(list(images), min(n_tiles, len(images)))
    px = []
    for im in sample:
        p = folder / im["filename"]
        if not p.exists():
            continue
        a = cv2.imread(str(p))
        if a is None:
            continue
        if flat is not None:
            a = apply_flat_field(a, flat)
        a = cv2.resize(a, None, fx=1.0 / downscale, fy=1.0 / downscale)[::2, ::2]
        px.append(a.reshape(-1, 3))
    px = np.vstack(px)
    q = (px.astype(np.int64) // 8)                  # 32 bins/channel
    key = q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]
    mode = np.bincount(key).argmax()
    return np.median(px[key == mode], axis=0).astype(np.float32)


# ── noise-aware magic-wand measurement (#49, validated 2026-06-24) ────────────

def wand_measure(img, S, axis, *, tol_frac=0.08, smooth=2.0, min_area=120,
                 cos_min=0.965, r_lo=-0.23, r_hi=-0.09):
    """Magic-wand flake measurement on one tile. Returns dicts with the INTERIOR
    contrast of each flat, graphene-aligned region."""
    f = img.astype(np.float32)
    proj = cv2.GaussianBlur(((f - S) / np.maximum(S, 1.0)) @ axis, (0, 0), smooth)
    sig = float(np.std(proj[np.abs(proj) < 0.03])) or 0.01
    h, w = proj.shape
    SC = 510.0
    pu8 = np.clip(proj * SC, 0, 255).astype(np.uint8)
    seed_thr = max(4 * sig, 0.04); tol = int(round(tol_frac * SC))
    seeds = cv2.morphologyEx((proj > seed_thr).astype(np.uint8), cv2.MORPH_OPEN,
                             np.ones((3, 3), np.uint8))
    seeds = cv2.erode(seeds, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, lbl, st, cen = cv2.connectedComponentsWithStats(seeds)
    lab = np.zeros((h, w), np.int32); ff = np.zeros((h + 2, w + 2), np.uint8); cur = 0
    for i in sorted(range(1, n), key=lambda j: -st[j, cv2.CC_STAT_AREA]):
        if st[i, cv2.CC_STAT_AREA] < 40: break
        sx, sy = int(cen[i][0]), int(cen[i][1])
        if lbl[sy, sx] != i or lab[sy, sx] != 0:
            ys, xs = np.where(lbl == i); k = int(np.argmax(proj[ys, xs])); sy, sx = int(ys[k]), int(xs[k])
        if lab[sy, sx] != 0: continue
        ff[:] = 0
        cv2.floodFill(pu8.copy(), ff, (sx, sy), 0, loDiff=tol, upDiff=tol,
                      flags=8 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (1 << 8))
        m = ff[1:-1, 1:-1].astype(bool) & (lab == 0)
        if m.sum() >= min_area: cur += 1; lab[m] = cur

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gmag = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3),
                         cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3))
    out = []
    for i in range(1, cur + 1):
        m = (lab == i).astype(np.uint8); a = int(m.sum())
        if a < min_area or a > 0.30 * proj.size: continue
        er = cv2.erode(m, np.ones((5, 5), np.uint8))
        interior = er.astype(bool) if er.sum() >= 40 else m.astype(bool)
        bgr = np.median(img[interior].reshape(-1, 3), axis=0).astype(float)
        c = (bgr - S) / S                                    # BGR contrast fraction
        if not (c < -0.01).all(): continue                   # darker in ALL channels (kills glints)
        if c @ axis / (np.linalg.norm(c) + 1e-9) < cos_min: continue   # aligned with graphene
        if not (r_lo <= c[2] <= r_hi): continue              # R-contrast band
        ring = (m > 0) & (cv2.erode(m, np.ones((3, 3), np.uint8)) == 0)
        edge = float(gmag[ring].mean()) if ring.any() else 0.0   # crisp flake vs diffuse blob
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perim = max((cv2.arcLength(cc, True) for cc in cnts), default=0.0)
        circ = float(4 * math.pi * a / (perim * perim)) if perim > 0 else 0.0  # 1=disc, →0 ragged
        Mo = cv2.moments(m)
        out.append({'bgr': bgr, 'c': c, 'area_px': a, 'edge': edge, 'circ': circ,
                    'cx': Mo['m10'] / Mo['m00'], 'cy': Mo['m01'] / Mo['m00']})
    return out


def fit_ladder(proj, weights):
    """Fit a regular layer comb: the dominant (area-weighted) peak is 1L; fit ONE
    constant step that makes the rest of the population land on integer multiples
    (folded-phase concentration). Returns (rungs, step, a0=1L position). Robust to
    the dominant-1L peak that confuses plain peak-finding."""
    lo, hi = np.percentile(proj, 1), np.percentile(proj, 99.5)
    h, edges = np.histogram(proj, 80, (lo, hi), weights=weights)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    a0 = float(ctr[h.argmax()])                       # dominant peak = monolayer
    best_s, best = 0.06, -1.0
    for s in np.linspace(0.035, 0.090, 56):
        ph = (((proj - a0) / s + 0.5) % 1.0) - 0.5    # fractional offset from a rung
        conc = float(weights[np.abs(ph) < 0.18].sum() / weights.sum())
        if conc > best: best, best_s = conc, s
    kmin = int(np.floor((lo - a0) / best_s)); kmax = int(np.ceil((hi - a0) / best_s))
    rungs = sorted(a0 + k * best_s for k in range(kmin, kmax + 1)
                   if lo - 0.5 * best_s <= a0 + k * best_s <= hi + 0.5 * best_s)
    return rungs, best_s, a0


# ── the writer ────────────────────────────────────────────────────────────────

def write_contrast_calibration(path, cal: dict, *, inputs: dict | None = None,
                               params: dict | None = None) -> Path:
    """Write a contrast_calibration.json, provenance-stamped.

    `cal` is the calibration payload (targets, substrate/axis or ladder,
    method, ...). A 'provenance' key (git rev, UTC timestamp, input digests,
    params) is added — every calibration on disk should say what made it.
    """
    from core.provenance import provenance_stamp
    cal = dict(cal)
    cal['provenance'] = provenance_stamp(inputs=inputs, params=params)
    path = Path(path)
    path.write_text(json.dumps(cal, indent=2))
    return path

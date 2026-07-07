#!/usr/bin/env python3
"""Per-sample PDF datasheet exporter (roadmap #59).

Renders a Graphene-Industries-style datasheet for a sample: a cover page with
the wafer-extents map and every flake marked (summary table paginated onto
continuation pages when it can't fit legibly), then one page per flake with its
thumbnail, the source scan tile with the detected boundary dashed on it,
layer/material + measurement data, chip-local and stage coordinates, and the
flake's position highlighted on the same wafer map.

Data sources (no new capture needed):
  - flake catalogue + wafer extents in ``sample.json``
  - chip<->stage transforms via ``core.sample_data`` (degrades to stage coords
    when no chip transform is registered)
  - thumbnails from the sample's ``images/`` directory
  - source tiles from the sample's newest area scan (``mapping/scans/``)

Zero new dependencies: composes pages with matplotlib -> PdfPages (matplotlib is
already a project dependency) and reads thumbnails with OpenCV.

CLI:
    python tools/make_datasheet.py <sample.json | sample-dir> [-o out.pdf]
                                   [--all-flakes] [--reserved-only]

Programmatic (e.g. an "Export datasheet (PDF)" button):
    from tools.make_datasheet import make_datasheet
    path = make_datasheet(sample, sample_dir, out_path=None)
"""
import argparse
import json
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")            # headless: no display needed
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D

import cv2
import numpy as np

# Allow running as a script (python tools/make_datasheet.py) or as a module.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from core import sample_data
except Exception:                # pragma: no cover - chip transforms become n/a
    sample_data = None


# ── helpers ──────────────────────────────────────────────────────────────────

def _identity_xy(x, y):
    """Identity coordinate transform (default when no placement is active)."""
    return (x, y)


def _mag_value(mag) -> float:
    """'100x' -> 100.0; unparseable -> 0.0 (so it sorts last when picking best)."""
    if not mag:
        return 0.0
    try:
        return float(str(mag).lower().rstrip("x"))
    except ValueError:
        return 0.0


def _best_image(flake: dict) -> dict | None:
    """Pick the most representative thumbnail: highest-mag 'frame', else any."""
    images = flake.get("images") or []
    if not images:
        return None
    frames = [im for im in images if im.get("type") == "frame"] or images
    return max(frames, key=lambda im: _mag_value(im.get("mag")))


def _imread_rgb(path: str):
    """Read an image file as an RGB uint8 array, or None on any failure."""
    if not path or not os.path.isfile(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.dtype == np.uint16:
        # Camera frames are 12-bit left-justified into 16-bit (max ≈65520), so
        # the high byte is the 8-bit value.  Use >>8, NOT >>4 — >>4 leaves values
        # up to ~4095 which then wrap in the uint8 cast and solarise the image.
        img = (img >> 8).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_thumb(images_dir: str, image_entry: dict | None):
    """Load a flake's image entry as an RGB uint8 array, or None."""
    if not image_entry or not image_entry.get("file"):
        return None
    return _imread_rgb(os.path.join(images_dir, image_entry["file"]))


# Detected-boundary overlay convention: dashed outline, black-haloed white (the
# vision/scale_bar style — legible on any substrate colour, protanopia-safe,
# never red/green-coded).  Images in this module are RGB.
_TILE_MAX_W = 1200               # downscale tiles to keep the PDF small


def _dash_polyline(img, pts, color=(255, 255, 255), thickness=2,
                   dash=9.0, gap=6.0):
    """Draw a closed DASHED polyline through pts (list of (x, y)) in place, as a
    black-haloed stroke.  cv2 has no dashed stroke; the dash/gap walk lives in
    vision.draw.dash_segments (shared with the catalogue-crop and map-thumbnail
    dashing)."""
    try:
        from vision.draw import dash_segments
    except Exception:
        return
    segs = [(a, b) for a, b in dash_segments(pts, dash, gap)]
    for col, t in (((0, 0, 0), thickness + 2), (color, thickness)):
        for (ax, ay), (bx, by) in segs:
            cv2.line(img, (int(round(ax)), int(round(ay))),
                     (int(round(bx)), int(round(by))), col, t, cv2.LINE_AA)


def _outline_thumb(thumb, flake: dict, image_entry: dict | None):
    """Dash the detector contour onto the stored crop thumbnail, in place, when
    the flake record carries one and the crop's tile-space origin is recoverable
    (the import_found_flakes._make_v3_crops crop-window formula).  Skips crops
    that already carry a baked-in outline and any geometry that can't be
    validated (resized or foreign crops)."""
    cnt = flake.get("contour_px")
    bbox = flake.get("bbox_px")
    if not cnt or not bbox or (image_entry or {}).get("outlined"):
        return
    x, y, w, h = bbox
    pad = int(max(w, h) * 0.4) + 20        # matches _make_v3_crops
    x0, y0 = max(0, x - pad), max(0, y - pad)
    th, tw = thumb.shape[:2]
    pts = [(px - x0, py - y0) for px, py in cnt]
    if not all(0 <= px < tw and 0 <= py < th for px, py in pts):
        return                             # crop window doesn't match — skip
    _dash_polyline(thumb, pts)


def _tile_panel(sample_dir: str, flake: dict):
    """Return (tile_rgb, label) — the flake's source scan tile with its detected
    boundary dashed on it — or (None, None) when no tile can be located (the
    flake page then renders exactly as before: no crash, no empty panel).

    Data path (a): the flake record carries the detector contour + source-tile
    reference (contour_px / source_image) — draw the real boundary.
    Path (b): locate the tile by stage position from the sample's newest scan
    (same tile-finding as vision.scan_crops.crop_at_stage) and draw a dashed
    box sized from area_um2.  Path (c): nothing locatable → (None, None).
    """
    try:
        from core.scan_io import load_metadata, scan_mag
        from vision.camera_params import px_per_um
        from vision.scan_crops import find_scan_folder
        scan = find_scan_folder(sample_dir, flake.get("magnification"))
        if scan is None:
            return None, None
        meta = load_metadata(scan)
        fw = int(meta["imaging"]["frame_width"])
        fh = int(meta["imaging"].get("frame_height", 2056))
        ppu = px_per_um(scan_mag(meta, default="20x"), fw)
    except Exception:
        return None, None
    if not ppu:
        return None, None

    cnt, src = flake.get("contour_px"), flake.get("source_image")
    if cnt and src and os.path.isfile(os.path.join(str(scan), src)):
        # (a) detector contour in the source tile's pixel frame.
        fname = src
        pts = [(float(px), float(py)) for px, py in cnt]
    else:
        # (b) nearest tile centre; stage->pixel via ppm and frame centre
        # (image-row-down = stage-y-+, as in scan_crops / the detector).
        sx, sy = flake.get("stage_x_mm"), flake.get("stage_y_mm")
        imgs = [im for im in meta.get("images", [])
                if im.get("x_mm") is not None and im.get("y_mm") is not None]
        if sx is None or sy is None or not imgs:
            return None, None
        t = min(imgs, key=lambda im: (im["x_mm"] - sx) ** 2 + (im["y_mm"] - sy) ** 2)
        ppm = ppu * 1000.0
        cx = fw / 2.0 + (float(sx) - t["x_mm"]) * ppm      # +pdx
        cy = fh / 2.0 + (float(sy) - t["y_mm"]) * ppm      # +pdy
        if not (0 <= cx < fw and 0 <= cy < fh):
            return None, None              # flake isn't on this scan
        area = flake.get("area_um2")
        side = max(40.0, (float(area) ** 0.5) * 1.6 * ppu) if area else 40.0
        r = side / 2.0
        pts = [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
        fname = t["filename"]

    tile = _imread_rgb(os.path.join(str(scan), fname))
    if tile is None:
        return None, None
    if tile.shape[1] > _TILE_MAX_W:        # keep the PDF small
        s = _TILE_MAX_W / tile.shape[1]
        tile = cv2.resize(tile, (_TILE_MAX_W, max(1, round(tile.shape[0] * s))))
        pts = [(px * s, py * s) for px, py in pts]
        ppu *= s
    _dash_polyline(tile, pts)
    try:
        from vision import scale_bar       # unified µm bar (catalogue convention)
        scale_bar.draw_cv2(tile, ppu)
    except Exception:
        pass
    return tile, fname


def _resolve_corner_image(ipath: str, sample_dir: str):
    """Load a corner crop, trying its stored path then sample-relative fallbacks
    (the path may be absolute from another machine)."""
    base = os.path.basename(ipath or "")
    for cand in (ipath,
                 os.path.join(sample_dir, "registration", base),
                 os.path.join(sample_dir, base),
                 os.path.join(sample_dir, "images", base)):
        img = _imread_rgb(cand)
        if img is not None:
            return img
    return None


def _placement_xform(sample: dict):
    """Return a function mapping stored **reference-stage** (x, y) into the
    **current placement stage frame** — the frame the extents polygon lives in.

    Stored flake/corner coordinates are reference-stage; the live map applies the
    placement transform (rotate by rotation_deg, then translate dx/dy) to display
    them — see vision.registration.apply_placement_transform and make_map.py.
    Verified: applying it to corner C1 lands exactly on the extents polygon
    vertex.  Returns the identity when no placement registration is active (e.g.
    index-mark-grid samples, where stored coords are already current-stage).
    """
    if sample_data is None:
        return _identity_xy
    try:
        tf = sample_data.get_placement_transform(sample)
        if not tf or "rotation_deg" not in tf:
            return _identity_xy
        from vision.registration import apply_placement_transform
    except Exception:
        return _identity_xy
    return lambda x, y: apply_placement_transform(tf, x, y)


def _corner_records(sample: dict, sample_dir: str, xform=None) -> list:
    """Return corner registration records as
    ``[{'label', 'x_mm', 'y_mm', 'img'}]`` in the current placement stage frame.

    Positions + crops come from ``placement.registration.corners`` (the live
    N-corner registration, #25), falling back to a top-level ``corners`` list.
    ``xform`` (default identity) maps the stored reference-stage positions into
    the current frame so corners sit on the extents polygon.
    """
    if xform is None:
        xform = _identity_xy
    reg = (sample.get("placement") or {}).get("registration") or {}
    raw = reg.get("corners") or sample.get("corners") or []
    recs = []
    for c in raw:
        x, y = c.get("x_mm"), c.get("y_mm")
        if x is not None and y is not None:
            x, y = xform(float(x), float(y))
        recs.append({
            "label": c.get("label", ""),
            "x_mm":  x,
            "y_mm":  y,
            "img":   _resolve_corner_image(c.get("image_path") or "", sample_dir),
        })
    return recs


def _chip_coords(sample: dict, flake: dict):
    """Return (chip_x_mm, chip_y_mm) for a flake, or (None, None).

    Prefers stored chip coords; otherwise derives them from the stage position
    via the sample's chip transform.  Any failure degrades to (None, None).
    """
    cx, cy = flake.get("chip_x_mm"), flake.get("chip_y_mm")
    if cx is not None and cy is not None:
        return float(cx), float(cy)
    sx, sy = flake.get("stage_x_mm"), flake.get("stage_y_mm")
    if sample_data is None or sx is None or sy is None:
        return None, None
    try:
        res = sample_data.stage_to_chip(sample, float(sx), float(sy))
    except Exception:
        res = None
    if res is None:
        return None, None
    return float(res[0]), float(res[1])


def _extents_polygon(sample: dict):
    """Return (polygon_xy, (xmin, xmax, ymin, ymax)) in stage mm, or (None, None)."""
    ext = sample.get("extents")
    if not ext:
        return None, None
    poly = ext.get("polygon_mm")
    if poly and len(poly) >= 3:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return poly, (min(xs), max(xs), min(ys), max(ys))
    # Fall back to the bounding rectangle when no polygon was recorded.
    try:
        x0, x1 = ext["x_negative_mm"], ext["x_positive_mm"]
        y0, y1 = ext["y_negative_mm"], ext["y_positive_mm"]
    except KeyError:
        return None, None
    rect = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return rect, (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))


def _sample_title(sample: dict) -> str:
    """Sample name with optional process-state suffix (#51, absent until built)."""
    name = sample.get("name", "sample")
    state = sample.get("process_state")
    return f"{name}  ·  {state}" if state else name


def _draw_wafer_map(ax, sample: dict, flakes: list, highlight_idx: int | None,
                    label_all: bool = False, axis_labels: bool = True,
                    title: str | None = "Wafer map", corners: list | None = None,
                    xform=None):
    """Draw the wafer-extents polygon with flakes marked; highlight one (or none).

    label_all annotates every flake with its ID (used on the cover overview);
    otherwise only the highlighted flake is labelled.  corners (if given) are
    drawn as blue squares at their wafer (x_mm, y_mm) with their labels.

    Stage Y is plotted inverted so the map matches the on-screen view (stage +Y
    is up at the scope but image rows increase downward); axes are in stage mm.
    """
    poly, bounds = _extents_polygon(sample)
    if poly is not None:
        ax.add_patch(MplPolygon(poly, closed=True, fill=False,
                                edgecolor="#444", linewidth=1.2))

    for c in (corners or []):
        if c.get("x_mm") is None or c.get("y_mm") is None:
            continue
        ax.plot(c["x_mm"], c["y_mm"], marker="s", markersize=6,
                markerfacecolor="none", markeredgecolor="#1565c0",
                markeredgewidth=1.4, zorder=4)
        ax.annotate(c["label"], (c["x_mm"], c["y_mm"]),
                    textcoords="offset points", xytext=(4, -9),
                    fontsize=6, color="#1565c0", fontweight="bold")

    if xform is None:
        xform = _identity_xy
    for i, fl in enumerate(flakes):
        sx, sy = fl.get("stage_x_mm"), fl.get("stage_y_mm")
        if sx is None or sy is None:
            continue
        sx, sy = xform(float(sx), float(sy))   # reference-stage → current frame
        is_hl = (i == highlight_idx)
        ax.plot(sx, sy, marker="o", markersize=9 if is_hl else 4,
                markerfacecolor="#d32f2f" if is_hl else "#b0b0b0",
                markeredgecolor="#7a0000" if is_hl else "#808080",
                markeredgewidth=1.2 if is_hl else 0.6, zorder=3 if is_hl else 2)
        if is_hl or label_all:
            ax.annotate(fl.get("id", ""), (sx, sy),
                        textcoords="offset points", xytext=(6, 3),
                        fontsize=8 if is_hl else 6,
                        color="#7a0000" if is_hl else "#333",
                        fontweight="bold")

    if bounds is not None:
        xmin, xmax, ymin, ymax = bounds
        mx = max((xmax - xmin) * 0.08, 0.5)
        my = max((ymax - ymin) * 0.08, 0.5)
        ax.set_xlim(xmin - mx, xmax + mx)
        ax.set_ylim(ymin - my, ymax + my)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    if axis_labels:
        ax.set_xlabel("stage X (mm)", fontsize=7)
        ax.set_ylabel("stage Y (mm)", fontsize=7)
    if title:
        ax.set_title(title, fontsize=8)
    # Ticks inside the axes and mirrored on all four sides (labels stay on the
    # left/bottom only).
    ax.tick_params(which="both", direction="in", top=True, right=True,
                   labelsize=6)


def _fmt(v, suffix="", nd=3):
    if v is None or v == "":
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}{suffix}"
    return f"{v}{suffix}"


# ── page builders ────────────────────────────────────────────────────────────

_FOOTER = ("Coordinates: stage mm (XIMC calb-space) and chip-local mm "
           "(origin at registered index-mark grid).")

# Flake-summary pagination: fixed legible row heights, never shrink-to-fit.
# Cover: 12 data rows in the 0.22-page-high strip under the map ≈ 0.20 in/row.
# Continuation: 34 data rows in a 0.86-page-high full-page table ≈ 0.28 in/row.
_SUMMARY_COLS = ["ID", "Name", "Layers", "Mag", "chip (mm)"]
_COVER_ROWS = 12
_CONT_ROWS = 34


def _summary_rows(sample: dict, flakes: list) -> list:
    rows = []
    for fl in flakes:
        cx, cy = _chip_coords(sample, fl)
        rows.append([
            fl.get("id", ""),
            (fl.get("name", "") or "")[:18],
            _fmt(fl.get("layer_count"), "L", 0) if fl.get("layer_count")
                else (fl.get("status", "") or ""),
            fl.get("magnification", ""),
            "n/a" if cx is None else f"({cx:.2f}, {cy:.2f})",
        ])
    return rows


def _summary_table(fig, rect, rows: list, page_rows: int, fontsize=7.0):
    """Render a flake-summary table into ``rect`` (figure coords), anchored to
    the TOP with the row height fixed by ``page_rows`` — a partial last chunk
    keeps the same row height instead of stretching to fill the box."""
    tax = fig.add_axes(rect)
    tax.axis("off")
    frac = min(1.0, (len(rows) + 1) / (page_rows + 1))    # +1 = header row
    tbl = tax.table(cellText=rows, colLabels=_SUMMARY_COLS,
                    cellLoc="left", bbox=[0, 1 - frac, 1, frac])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    for (r, _c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#d0d0d0")
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#eeeeee")
    return tbl


def _summary_continuation_pages(pdf, sample: dict, flakes: list):
    """Full-page tables for the summary rows that didn't fit on the cover.
    Column set identical to the cover table; page k continues its numbering
    (cover = page 1 of the summary)."""
    rest = flakes[_COVER_ROWS:]
    for k, start in enumerate(range(0, len(rest), _CONT_ROWS), start=2):
        chunk = rest[start:start + _CONT_ROWS]
        fig = plt.figure(figsize=(8.27, 11.69))            # A4 portrait
        fig.suptitle(f"{_sample_title(sample)} — Flake summary "
                     f"(continued, page {k})",
                     fontsize=12, fontweight="bold", y=0.965)
        _summary_table(fig, [0.07, 0.06, 0.86, 0.86],
                       _summary_rows(sample, chunk), _CONT_ROWS, fontsize=7.5)
        fig.text(0.5, 0.02, _FOOTER, fontsize=6, color="#666", ha="center")
        pdf.savefig(fig)
        plt.close(fig)


def _place_corner_thumbs_on_map(fig, ax, corners: list):
    """Inset each corner crop near its real wafer position, pushed radially
    outward from the map centre so the thumbnails ring the map in the same
    spatial arrangement as on the chip (rather than a disconnected strip).
    A thin line links each thumbnail back to its marker on the map."""
    placed = [c for c in corners
              if c.get("img") is not None
              and c.get("x_mm") is not None and c.get("y_mm") is not None]
    if not placed:
        return
    fig.canvas.draw()                       # finalise transData for Agg
    inv = fig.transFigure.inverted()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    cen = inv.transform(ax.transData.transform(((x0 + x1) / 2, (y0 + y1) / 2)))
    ax.set_zorder(1)                         # so connector lines can sit above it
    w = h = 0.08
    for c in placed:
        p = inv.transform(ax.transData.transform((c["x_mm"], c["y_mm"])))
        d = p - cen
        n = (d[0] ** 2 + d[1] ** 2) ** 0.5 or 1.0
        ux, uy = d[0] / n, d[1] / n
        # Centre the thumbnail on a point pushed radially outward from the marker.
        tx = min(max(p[0] + ux * 0.11 - w / 2, 0.01), 0.99 - w)
        ty = min(max(p[1] + uy * 0.11 - h / 2, 0.01), 0.99 - h)
        # connector from thumbnail centre back to the marker — drawn above the
        # map (zorder > the map axes) so it isn't hidden behind it.
        line = Line2D(
            [tx + w / 2, p[0]], [ty + h / 2, p[1]],
            transform=fig.transFigure, color="#1565c0", linewidth=0.6,
            zorder=2)
        fig.add_artist(line)
        a = fig.add_axes([tx, ty, w, h])
        a.set_zorder(3)                     # thumbnail above the connector line
        a.imshow(c["img"])
        a.set_xticks([]); a.set_yticks([])
        a.set_title(c["label"], fontsize=6, pad=1, color="#1565c0")
        for sp in a.spines.values():
            sp.set_color("#1565c0")


def _optical_summary(sample: dict, sample_dir: str) -> list:
    """Oxide + contrast-calibration provenance lines for the cover, or [].

    Sources: the sample's measured sio2_nm, and the newest derived
    contrast_calibration.json (material, resolvable bands, 1L contrast, lamp
    source, git rev + timestamp) — so a datasheet says exactly which oxide and
    which calibration produced its layer numbers (plan S1/S8.6 provenance).
    """
    lines = []
    d = sample.get("sio2_nm")
    diel = sample.get("dielectric", "SiO\u2082")
    if d:
        lines.append(f"Oxide: {float(d):.1f} nm {diel} (measured)")
    else:
        lines.append(f"Oxide: not measured ({diel})")

    cal = None
    try:
        cal_path = (sample_data.latest_optical_calibration(sample_dir)
                    if sample_data is not None else None)
        if cal_path and os.path.exists(cal_path):
            with open(cal_path) as fh:
                cal = json.load(fh)
    except Exception:
        cal = None
    if not cal:
        lines.append("Contrast targets: none derived")
        return lines

    mat = cal.get("material", "graphene")
    lines.append(f"Targets: {mat}")
    mlc = cal.get("monolayer_contrast_pct")
    if mlc is not None:
        lines.append(f"  1L contrast: {float(mlc):.1f}%")
    bands = cal.get("resolvable_layer_bands") or []
    if bands:
        band_txt = ", ".join(f"{lo}\u2013{hi}" for lo, hi in bands)
        lines.append(f"  resolvable N: {band_txt}")
    src = cal.get("source")
    if src:
        lines.append(f"  lamp: {src}")
    prov = cal.get("provenance") or {}
    rev = prov.get("git_rev")
    gen = prov.get("generated_at")
    if rev:
        lines.append(f"  derived: {rev}"
                     + (f" @ {gen[:10]}" if gen else ""))
    return lines


def _cover_page(pdf, sample: dict, flakes: list, sample_dir: str,
                corners: list | None = None, xform=None):
    fig = plt.figure(figsize=(8.27, 11.69))            # A4 portrait
    fig.suptitle(_sample_title(sample), fontsize=16, fontweight="bold", y=0.97)

    created = sample.get("created", "")
    meta = [
        f"User: {sample.get('user', 'n/a')}",
        f"Substrate: {sample.get('substrate', 'n/a')}",
        f"Folder: {sample.get('folder', 'n/a')}",
        f"Created: {created}",
        f"Flakes: {len(flakes)}",
        f"Exported: {datetime.now().isoformat(timespec='seconds')}",
    ]
    fig.text(0.10, 0.94, "\n".join(meta), fontsize=10, va="top",
             family="monospace")

    # Oxide + contrast-calibration provenance (which oxide + which calibration
    # produced the layer numbers on this sheet).
    opt = _optical_summary(sample, sample_dir)
    if opt:
        fig.text(0.56, 0.94, "\n".join(opt), fontsize=9, va="top",
                 family="monospace", color="#333")

    # Wafer map with all flakes; registration corners (C1…CN) marked at their
    # real wafer positions and their crops inset radially around the map.  The
    # map is inset from the page edges to leave room for the corner thumbnails.
    if xform is None:
        xform = _placement_xform(sample)
    if corners is None:
        corners = _corner_records(sample, sample_dir, xform)
    ax = fig.add_axes([0.25, 0.43, 0.50, 0.28])
    _draw_wafer_map(ax, sample, flakes, highlight_idx=None, label_all=True,
                    title=None, corners=corners, xform=xform)
    _place_corner_thumbs_on_map(fig, ax, corners)

    # Flake summary table — only the rows that fit legibly under the map; the
    # remainder continues on dedicated full-page tables after the cover
    # (pagination, never shrink-to-illegible fonts).
    if flakes:
        shown = flakes[:_COVER_ROWS]
        _summary_table(fig, [0.07, 0.05, 0.86, 0.22],
                       _summary_rows(sample, shown), _COVER_ROWS)
        if len(flakes) > len(shown):
            fig.text(0.5, 0.035, f"… +{len(flakes) - len(shown)} more flakes — "
                     f"summary continues on the following page(s)",
                     fontsize=7, ha="center", color="#666", style="italic")

    fig.text(0.5, 0.015, _FOOTER, fontsize=6, color="#666", ha="center")
    pdf.savefig(fig)
    plt.close(fig)


def _flake_page(pdf, sample: dict, flakes: list, idx: int, images_dir: str,
                sample_dir: str, corners: list | None = None, xform=None):
    fl = flakes[idx]
    fig = plt.figure(figsize=(8.27, 11.69))            # A4 portrait
    name = (fl.get("name") or "").strip()
    heading = f"{fl.get('id', '')} — {name}" if name else fl.get("id", "")
    fig.suptitle(heading, fontsize=14, fontweight="bold", y=0.97)

    # Source scan tile with the flake's boundary dashed on it (the app-wide
    # detected-flake convention).  None → the page keeps the original layout.
    tile, tile_label = _tile_panel(sample_dir, fl)
    have_tile = tile is not None

    # Thumbnail (top-left); shorter when the tile panel sits below it.
    ax_img = fig.add_axes([0.07, 0.665, 0.52, 0.255] if have_tile
                          else [0.07, 0.55, 0.52, 0.36])
    ax_img.axis("off")
    best = _best_image(fl)
    thumb = _load_thumb(images_dir, best)
    if thumb is not None:
        _outline_thumb(thumb, fl, best)     # dash the contour when available
        ax_img.imshow(thumb)
        ax_img.set_title(f"{best.get('file', '')}  ({best.get('mag', '')})",
                         fontsize=7)
    else:
        ax_img.text(0.5, 0.5, "(no image)", ha="center", va="center",
                    fontsize=11, color="#999", transform=ax_img.transAxes)

    # Wafer map with this flake highlighted + corner markers for context.
    ax_map = fig.add_axes([0.63, 0.665, 0.30, 0.255] if have_tile
                          else [0.63, 0.55, 0.30, 0.36])
    if xform is None:
        xform = _placement_xform(sample)
    _draw_wafer_map(ax_map, sample, flakes, highlight_idx=idx, corners=corners,
                    xform=xform)

    if have_tile:
        ax_tile = fig.add_axes([0.07, 0.40, 0.86, 0.235])
        ax_tile.axis("off")
        ax_tile.imshow(tile)
        ax_tile.set_title(f"Source tile: {tile_label}", fontsize=7)

    # Data block (lower half).
    cx, cy = _chip_coords(sample, fl)
    chip_str = "n/a" if cx is None else f"({cx:.3f}, {cy:.3f}) mm"
    stage_str = (f"({_fmt(fl.get('stage_x_mm'), nd=3)}, "
                 f"{_fmt(fl.get('stage_y_mm'), nd=3)}) mm")
    # Chip-local coords lead — they're the placement-relevant frame; stage XY is
    # secondary (rig-absolute, less meaningful across remounts).
    fields = [
        ("Chip-local coords", chip_str),
        ("Material / layers", _fmt(fl.get("layer_count"), "L", 0)
            if fl.get("layer_count") else "n/a"),
        ("Status", fl.get("status", "n/a")),
        ("Source / confirmed", f"{fl.get('source', 'n/a')} / "
            f"{'yes' if fl.get('confirmed') else 'no'}"),
        ("Stage coords (abs)", stage_str),
        ("Z / R", f"{_fmt(fl.get('z_mm'), ' mm')}  /  "
            f"{_fmt(fl.get('r_deg'), '°', 2)}"),
        ("Magnification", fl.get("magnification", "n/a")),
        ("Area", _fmt(fl.get("area_um2"), " µm²", 1)),
        ("Circularity / aspect / solidity",
            f"{_fmt(fl.get('circularity'), nd=2)} / "
            f"{_fmt(fl.get('aspect_ratio'), nd=2)} / "
            f"{_fmt(fl.get('solidity'), nd=2)}"),
        ("Cleanliness / isolation",
            f"{fl.get('cleanliness') or 'n/a'} / {fl.get('isolation') or 'n/a'}"),
        ("Captured", fl.get("created_at", "n/a")),
        ("Updated", fl.get("updated_at", "n/a")),
    ]
    notes = (fl.get("notes") or "").strip()
    if notes:                               # in-table so both layouts fit it
        fields.append(("Notes", notes[:100] + ("…" if len(notes) > 100 else "")))
    # Compact two-column property table.
    tax = fig.add_axes([0.08, 0.06, 0.84, 0.31] if have_tile
                       else [0.08, 0.18, 0.84, 0.34])
    tax.axis("off")
    tbl = tax.table(cellText=[[lbl, str(val)] for lbl, val in fields],
                    colLabels=["Property", "Value"],
                    colWidths=[0.34, 0.66], cellLoc="left", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#d0d0d0")
        cell.PAD = 0.03
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#eeeeee")
        elif c == 0:
            cell.set_text_props(fontweight="bold")

    fig.text(0.5, 0.02, _FOOTER, fontsize=6, color="#666", ha="center")
    pdf.savefig(fig)
    plt.close(fig)


# ── public API ───────────────────────────────────────────────────────────────

def make_datasheet(sample: dict, sample_dir: str, out_path: str | None = None,
                   flakes: list | None = None) -> str:
    """Write a PDF datasheet for ``sample`` and return the output path.

    sample_dir : directory holding sample.json and the images/ subdir.
    out_path   : destination PDF; defaults to ``<sample_dir>/<name>_datasheet.pdf``.
    flakes     : explicit flake subset (e.g. reserved only); defaults to all.
    """
    if flakes is None:
        flakes = sample.get("flakes", []) or []
    images_dir = os.path.join(sample_dir, "images")

    if out_path is None:
        state = sample.get("process_state")
        stem = sample.get("name", "sample")
        if state:
            stem = f"{stem}__{state}"
        out_path = os.path.join(sample_dir, f"{stem}_datasheet.pdf")

    # Stored flake/corner coords are reference-stage; map them into the current
    # placement frame (the frame the extents polygon is in) so everything is
    # coincident, exactly as the live app map does.
    xform = _placement_xform(sample)
    corners = _corner_records(sample, sample_dir, xform)   # loaded once, reused
    with PdfPages(out_path) as pdf:
        _cover_page(pdf, sample, flakes, sample_dir, corners=corners, xform=xform)
        _summary_continuation_pages(pdf, sample, flakes)
        for idx in range(len(flakes)):
            _flake_page(pdf, sample, flakes, idx, images_dir, sample_dir,
                        corners=corners, xform=xform)
        d = pdf.infodict()
        d["Title"] = f"{_sample_title(sample)} datasheet"
        d["Author"] = sample.get("user", "")
        d["Subject"] = "vdW heterostructure flake datasheet"
    return out_path


_NUM_NEG_INF = float("-inf")


def _rank_key(fl: dict):
    """Composite 'best flake' key: confirmed/reserved first, then larger area,
    then more layers.  Missing values sort last (descending order assumed)."""
    area = fl.get("area_um2")
    layers = fl.get("layer_count")
    return (
        1 if fl.get("confirmed") or fl.get("locked") else 0,
        area if isinstance(area, (int, float)) else _NUM_NEG_INF,
        layers if isinstance(layers, (int, float)) else _NUM_NEG_INF,
    )


def _num_key(field):
    def key(fl):
        v = fl.get(field)
        return v if isinstance(v, (int, float)) else _NUM_NEG_INF
    return key


def _str_key(field):
    return lambda fl: str(fl.get(field) or "")


# Sort field -> (key fn, default-descending?).  Numeric/quality fields default
# to descending (best first); identifier/time fields to ascending.
SORT_KEYS = {
    "catalogue": (None,                 False),  # original order
    "rank":      (_rank_key,            True),
    "area":      (_num_key("area_um2"), True),
    "layers":    (_num_key("layer_count"), True),
    "id":        (_str_key("id"),       False),
    "name":      (_str_key("name"),     False),
    "created":   (_str_key("created_at"), False),
}


def sort_flakes(flakes: list, sort_by: str = "catalogue",
                reverse: bool = False) -> list:
    """Sort flakes by ``sort_by`` (see SORT_KEYS); ``reverse`` flips the field's
    natural direction.  'catalogue' keeps original order (reverse = bottom-up)."""
    keyfn, desc_default = SORT_KEYS.get(sort_by, (None, False))
    descending = desc_default ^ reverse
    if keyfn is None:
        return list(reversed(flakes)) if reverse else list(flakes)
    return sorted(flakes, key=keyfn, reverse=descending)


def top_flakes(flakes: list, n: int) -> list:
    """Return the top ``n`` flakes by the composite rank key (stable for ties)."""
    ranked = sort_flakes(flakes, "rank")
    return ranked[:n] if n and n > 0 else ranked


def _resolve_sample(arg: str):
    """Accept a sample.json file or a directory containing one."""
    if os.path.isdir(arg):
        path = os.path.join(arg, "sample.json")
    else:
        path = arg
    if not os.path.isfile(path):
        raise FileNotFoundError(f"sample.json not found at {path}")
    with open(path) as f:
        sample = json.load(f)
    return sample, os.path.dirname(os.path.abspath(path))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Export a per-sample PDF datasheet.")
    ap.add_argument("sample", help="path to sample.json or its directory")
    ap.add_argument("-o", "--out", default=None, help="output PDF path")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--all-flakes", action="store_true",
                     help="include every catalogued flake (default)")
    grp.add_argument("--reserved-only", action="store_true",
                     help="include only locked/reserved flakes")
    ap.add_argument("--top", type=int, default=None, metavar="N",
                    help="only the top N flakes (after --sort-by)")
    ap.add_argument("--sort-by", choices=sorted(SORT_KEYS), default=None,
                    help="order flakes by this field (default: rank when --top "
                         "is set, else catalogue order)")
    ap.add_argument("--reverse", action="store_true",
                    help="flip the sort direction")
    args = ap.parse_args(argv)

    sample, sample_dir = _resolve_sample(args.sample)
    flakes = sample.get("flakes", []) or []
    if args.reserved_only:
        flakes = [f for f in flakes if f.get("locked")]

    # Default ordering: 'rank' (best-first) when topping, else catalogue order.
    sort_by = args.sort_by or ("rank" if args.top is not None else "catalogue")
    flakes = sort_flakes(flakes, sort_by, args.reverse)
    if args.top is not None and args.top > 0:
        flakes = flakes[:args.top]

    out = make_datasheet(sample, sample_dir, args.out, flakes=flakes)
    n_cont = -(-max(0, len(flakes) - _COVER_ROWS) // _CONT_ROWS)
    print(f"Wrote {out}  ({len(flakes)} flake page(s) + cover"
          f"{f' + {n_cont} summary continuation page(s)' if n_cont else ''}; "
          f"sort={sort_by}{' reversed' if args.reverse else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

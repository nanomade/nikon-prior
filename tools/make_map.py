#!/usr/bin/env python3
"""
make_map.py — assemble area-scan images into a zoomable DZI map.

Usage
-----
  # Single scan folder (contains scan_metadata.json directly):
  python tools/make_map.py PATH/sample/20x/scan_20260512_154414

  # Sample root — discovers all scans under sample/{mag}/scan_* automatically:
  python tools/make_map.py PATH/sample_name

  # Mix magnifications or regions explicitly:
  python tools/make_map.py PATH/sample/5x/scan_A PATH/sample/20x/scan_B

  # Common options:
  python tools/make_map.py PATH/sample [-o OUT_DIR] [--name MAP] [--scale 0.5]
                           [--jpeg-quality Q] [--open]

Folder layout expected from area_wafer_scan_panel:
  {output_dir}/{sample_name}/{mag}/{sample_name}_{mag}_{timestamp}/
    scan_metadata.json
    img_*.png

When a root folder contains no scan_metadata.json but has mag subfolders
(5x / 10x / 20x / 50x / 100x), all scan subdirectories within are discovered
automatically, newest first per magnification.

Output (in -o DIR, default: root of the discovered tree):
  <name>.dzi          — DZI descriptor (XML)
  <name>_files/       — tile pyramid
  <name>.html         — OpenSeadragon viewer with coordinate readout
"""

import argparse
import datetime
import json
import math
import shutil
import sys
import time as _time
import xml.etree.ElementTree as ET
from pathlib import Path


# ---------------------------------------------------------------------------
# Lightweight stage timing (opt-in via --timing).  Stages are non-overlapping so
# the breakdown sums; the goal is to see where map-build wall-clock actually goes
# (stitch/pos-solve vs DZI encoding vs HTML) before reaching for CUDA — see #58.
# ---------------------------------------------------------------------------




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

# ---------------------------------------------------------------------------
# Calibration — single source of truth in vision/camera_params.py
# ---------------------------------------------------------------------------

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
from vision.camera_params import px_per_um as _camera_ppm, MAG_SCALE as _MAG_SCALE
# Scan loading + bounds live in core/scan_io (plan task L1); rig calibration
# access in core/rig_calibration (L2). Imported under their historical names —
# this script was the reference implementation the rest of the repo copied.
from core.scan_io import load_scan, scan_bounds, all_bounds  # noqa: F401
from core import rig_calibration as _rig
# Stitch + DZI machinery extracted to vision/ (plan task L5); imported under
# their historical names so downstream code here is unchanged.
from vision import map_stitch as _map_stitch
from vision.map_stitch import (_stage, _print_stage_times, _px_per_um,   # noqa: F401
                               _choose_canvas_ppm, _build_grid, assemble,
                               assemble_layers, MAX_CANVAS_PX)
from vision.dzi import (TILE_SIZE, TILE_OVERLAP, write_dzi,              # noqa: F401
                        write_per_frame_dzis)
from tools.map_html import write_html      # noqa: F401  (HTML viewer, plan L5)

_APP_DIR = Path(__file__).resolve().parent.parent




# ---------------------------------------------------------------------------
# Canvas assembly
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# DZI output
# ---------------------------------------------------------------------------









# ---------------------------------------------------------------------------
# HTML viewer
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

_KNOWN_MAGS = set(_MAG_SCALE.keys())   # {'5x', '10x', '20x', '50x', '100x'}

from core.scan_io import discover_scan_folders as _discover_scan_folders  # noqa: E402,F401


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _find_sample_root(path: Path) -> 'Path | None':
    """Walk up from path until a directory containing sample.json is found."""
    from core.sample_data import find_sample_dir_upwards
    d = find_sample_dir_upwards(str(path))
    return Path(d) if d else None


def _default_stem(root: Path, scans: list) -> str:
    sample_root = (_find_sample_root(scans[0]['folder'])
                   or _find_sample_root(root))
    if sample_root is not None:
        return sample_root.name + '_map'
    if len(scans) == 1:
        return scans[0]['folder'].name + '_map'
    mags = '_'.join(sorted({s['mag'] for s in scans}))
    return f"map_{mags}"


def _default_out_dir(root: Path, scans: list) -> Path:
    """Output goes into <sample_root>/mapping/maps/."""
    sample_root = (_find_sample_root(scans[0]['folder'])
                   or _find_sample_root(root))
    if sample_root is not None:
        return sample_root / 'mapping' / 'maps'
    # Fallback: mapping/maps/ alongside the common ancestor
    try:
        ancestor = scans[0]['folder'].parent.parent
        if all(s['folder'].is_relative_to(ancestor) for s in scans):
            return ancestor.parent / 'mapping' / 'maps'
    except Exception:
        pass
    return scans[0]['folder'].parent / 'mapping' / 'maps'


def _rebuild_layers_info(stem: str, out_dir: Path, scans: list) -> list:
    """
    Reconstruct layers_info from existing DZI files without re-running assembly.
    Physical extents are recomputed from scan bounds; tile_centres are left empty
    (the JS falls back to linear interpolation, which is fine for overlays).
    """
    layer_bounds: dict = {}
    for scan in scans:
        mag = scan['mag']
        sb = list(scan_bounds(scan))
        if mag not in layer_bounds:
            layer_bounds[mag] = sb
        else:
            b = layer_bounds[mag]
            b[0] = min(b[0], sb[0]); b[1] = min(b[1], sb[1])
            b[2] = max(b[2], sb[2]); b[3] = max(b[3], sb[3])

    global_x0, global_y0, global_x1, global_y1 = all_bounds(scans)
    global_phys_w = global_x1 - global_x0

    info_list = []
    for mag in sorted(layer_bounds, key=lambda m: _MAG_SCALE[m]):
        layer_stem = f"{stem}_{mag}"
        dzi_path = out_dir / f"{layer_stem}.dzi"
        if not dzi_path.exists():
            # Stem may differ from an older run — search for any *_{mag}.dzi
            matches = sorted(out_dir.glob(f"*_{mag}.dzi"))
            if matches:
                dzi_path = matches[0]
                layer_stem = dzi_path.stem
                print(f"  Using existing DZI: {dzi_path.name}")
            else:
                print(f"  Warning: {dzi_path.name} not found — skipping {mag} layer")
                continue
        try:
            tree = ET.parse(str(dzi_path))
            root = tree.getroot()
            ns = 'http://schemas.microsoft.com/deepzoom/2008'
            size_el = root.find(f'{{{ns}}}Size')
            if size_el is None:
                size_el = root.find('Size')
            w = int(size_el.get('Width'))
            h = int(size_el.get('Height'))
            fmt = root.get('Format', 'jpg')
        except Exception as e:
            print(f"  Warning: could not read {dzi_path.name}: {e} — skipping")
            continue
        lx0, ly0, lx1, ly1 = layer_bounds[mag]
        info_list.append({
            'mag':          mag,
            'stem':         layer_stem,
            'w':            w,
            'h':            h,
            'fmt':          fmt,
            'x0_mm':        lx0,
            'y0_mm':        ly0,
            'phys_w':       lx1 - lx0,
            'phys_h':       ly1 - ly0,
            'osd_x':        (lx0 - global_x0) / global_phys_w,
            'osd_y':        (ly0 - global_y0) / global_phys_w,
            'osd_width':    (lx1 - lx0)       / global_phys_w,
            'tile_centres': [],
        })
        print(f"  {mag}: {w}×{h} px  ({fmt})")
    return info_list


def _inverse_placement(reg: dict, points: list) -> list:
    """Convert current-stage mm coords → reference-stage mm coords (inverse of placement transform)."""
    theta = math.radians(reg['rotation_deg'])
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    dx_mm, dy_mm = reg['dx_mm'], reg['dy_mm']
    return [[cos_t * (p[0] - dx_mm) + sin_t * (p[1] - dy_mm),
             -sin_t * (p[0] - dx_mm) + cos_t * (p[1] - dy_mm)]
            for p in points]


def _forward_placement(reg: dict, points: list) -> list:
    """Convert reference-stage mm coords → scan-frame mm coords (forward placement)."""
    theta = math.radians(reg['rotation_deg'])
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    dx_mm, dy_mm = reg['dx_mm'], reg['dy_mm']
    return [[cos_t * p[0] - sin_t * p[1] + dx_mm,
             sin_t * p[0] + cos_t * p[1] + dy_mm]
            for p in points]


def _to_scan_frame(points: list, live_reg: dict | None, scan_reg: dict | None) -> list:
    """Convert live current-stage mm coords → this map's scan-frame mm coords.

    Stored sample geometry (extents, flakes) lives in the current stage frame the
    sample was last registered in (`live_reg`).  The map canvas is in the frame
    the scan was captured in (`scan_reg`, the placement active at scan time).  Go
    current → reference (inverse live) → scan-frame (forward scan).  With
    scan_reg None this reduces to the historical inverse-placement, so pre-stamp
    maps are unchanged; same-mount (live == scan) round-trips to identity.
    """
    pts = _inverse_placement(live_reg, points) if live_reg else points
    if scan_reg and 'dx_mm' in scan_reg:
        pts = _forward_placement(scan_reg, pts)
    return pts


def _select_candidates(cands: list, sort_by: str = 'area', top_per_layer: int = 20) -> list:
    """Sort candidates (largest/highest first) and optionally keep top-N per layer.

    sort_by: 'area' (area_um2) or 'score'.  top_per_layer: 0 keeps all.  Replaces
    the old hand-filtering — selection is now a reproducible, flag-driven step.
    """
    keyf = ((lambda c: c.get('area_um2', 0)) if sort_by == 'area'
            else (lambda c: c.get('score', 0)))
    cands = sorted(cands, key=keyf, reverse=True)
    if top_per_layer and top_per_layer > 0:
        kept, counts = [], {}
        for c in cands:
            n = c.get('target_N', 0)
            if counts.get(n, 0) < top_per_layer:
                counts[n] = counts.get(n, 0) + 1
                kept.append(c)
        cands = kept
    return cands


def main():
    ap = argparse.ArgumentParser(
        description='Assemble area-scan images into a zoomable DZI map.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument('folders', nargs='+', metavar='FOLDER',
                    help='Scan folder(s) or sample root dir (auto-discovers scans)')
    ap.add_argument('-o', '--output', metavar='DIR', default=None,
                    help='Output directory (default: sample root or first scan parent)')
    ap.add_argument('--name', '-n', metavar='NAME', default=None,
                    help='Output file stem (default: derived from sample/mag names)')
    ap.add_argument('--scale', '-s', type=float, default=None, metavar='FACTOR',
                    help='Scale factor relative to highest-mag native resolution '
                         '(default: auto-fit to ≤8000 px)')
    ap.add_argument('--jpeg-quality', '-q', type=int, default=85, metavar='Q',
                    help='JPEG tile quality 1-95 (default: 85)')
    ap.add_argument('--open', action='store_true',
                    help='Open the HTML viewer in a browser when done')
    ap.add_argument('--no-refine', action='store_true',
                    help='Disable ORB tile alignment (use raw stage coordinates)')
    ap.add_argument('--save-rotation', action='store_true',
                    help='After ORB, write extracted camera_rotation_deg to calibration.json')
    ap.add_argument('--rotation-model', action='store_true',
                    help='Use stored camera_rotation_deg from calibration.json for tile '
                         'positioning instead of per-pair ORB/BFS. Skips ORB entirely — '
                         'fast, outlier-free, and safe for filtered scans.')
    ap.add_argument('--rotate-tiles', action='store_true',
                    help='Straighten each tile by +camera_rotation_deg and place tiles on '
                         'an un-rotated stage grid, producing a stage-axis-aligned map '
                         '(requires --rotation-model and cv2). Not needed for contrast '
                         'analysis — the default --rotation-model map already stitches.')
    ap.add_argument('--correct-bg', action='store_true',
                    help='Remove per-tile vignetting via flat-field division. '
                         'Do NOT use when flat-field correction was already applied '
                         'at capture time (the default in standa-stacker).')
    ap.add_argument('--blend', action='store_true',
                    help='50/50 blend tiles in overlap zones (default: last tile wins)')
    ap.add_argument('--sample-json', metavar='PATH', default=None,
                    help='Path to sample.json — overlays flake markers and wafer extents')
    ap.add_argument('--nav-port', type=int, default=0, metavar='PORT',
                    help='HTTP port of the MapNavServer (enables click-to-navigate)')
    ap.add_argument('--overview', action='store_true',
                    help='Compose all frames into one canvas per mag (legacy mode, lower resolution)')
    ap.add_argument('--update-html', action='store_true',
                    help='Regenerate HTML only — reuse existing DZI tiles, skip ORB/assembly')
    ap.add_argument('--cand-sort', choices=['area', 'score'], default='area',
                    help='Sort flake candidates by this key, largest/highest first (default: area)')
    ap.add_argument('--cand-top', type=int, default=20, metavar='N',
                    help='Keep only the top N candidates per layer count for the overlay '
                         '(default: 20; 0 = keep all)')
    ap.add_argument('--timing', action='store_true',
                    help='Print a per-stage wall-clock breakdown (stitch vs DZI '
                         'encoding vs HTML) — use to find the real bottleneck (#58)')
    args = ap.parse_args()

    _map_stitch.set_timing(args.timing)

    # Resolve input paths, auto-discovering scan folders when needed
    scan_folders: list[Path] = []
    roots: list[Path] = []
    for fstr in args.folders:
        p = Path(fstr).resolve()
        if not p.exists():
            sys.exit(f"Path not found: {p}")
        discovered = _discover_scan_folders(p)
        if not discovered:
            sys.exit(f"No scan_metadata.json found under {p}")
        if len(discovered) > 1 and not (p / 'scan_metadata.json').exists():
            print(f"Discovered {len(discovered)} scan(s) under {p.name}:")
            for d in discovered:
                print(f"  {d.relative_to(p)}")
        scan_folders.extend(discovered)
        roots.append(p)

    # Load scans
    scans = []
    for folder in scan_folders:
        print(f"Loading: {folder.name}")
        scan = load_scan(folder)
        print(f"  mag={scan['mag']}  {scan['frame_w']}×{scan['frame_h']}  "
              f"{len(scan['images'])} images  ppu={scan['ppu']:.4f} px/µm")
        scans.append(scan)

    for s in scans:
        if not s['images']:
            print(f"  WARNING: skipping {s['folder'].name} — no images")
    scans = [s for s in scans if s['images']]
    if not scans:
        sys.exit("No scans loaded.")

    # Apply objective_offset corrections so all mags share the 100x physical frame.
    # calibration.json stores offsets for each objective relative to 100x (the reference).
    # During a scan at mag M, the stage position = true_physical_pos + offset[M], so
    # subtracting offset[M] converts to the 100x frame.
    _obj_offsets = _rig.objective_offsets()

    print("Applying objective offsets (100x reference):")
    for scan in scans:
        off = _obj_offsets.get(scan['mag'], [0.0, 0.0])
        if abs(off[0]) > 1e-6 or abs(off[1]) > 1e-6:
            for img in scan['images']:
                img['x_mm'] -= off[0]
                img['y_mm'] -= off[1]
            print(f"  {scan['mag']}: ({off[0]:+.4f}, {off[1]:+.4f}) mm applied to {len(scan['images'])} tiles")
        else:
            print(f"  {scan['mag']}: 100x reference (no offset)")

    def _flake_to_map_frame(f, x, y, chip_tf=None):
        """Map-frame position of a catalogue flake.

        Primary path: chip-local coords through the chip transform — mag-agnostic,
        remount-invariant, and natively in the map/reference frame (verified:
        chip→reference == stored stage for marked flakes, both 'app' and 'map').
        This is the single source of truth and needs NO objective-offset: flakes
        and tiles share the scan frame, so the offset is common-mode (it only ever
        mattered for cross-objective alignment, which the chip transform handles).

        Applying off[mag] here was a double-correction for flakes already stored in
        the reference frame — it shifted scope-marked flakes by off[mag] (≈16 µm at
        20×) off their features.

        Legacy fallback (no chip coords): map-marked are already in the map frame;
        scope-marked were stored commanded → keep the old objective-offset subtract."""
        cx, cy = f.get('chip_x_mm'), f.get('chip_y_mm')
        if cx is not None and cy is not None and chip_tf:
            ox, oy = chip_tf['origin_mm']
            xx, xy = chip_tf['x_axis']
            yx, yy = chip_tf['y_axis']
            rx, ry = ox + cx * xx + cy * yx, oy + cx * xy + cy * yy
            if f.get('source') == 'auto':
                # Auto flakes' chip coords derive (at import) from the DETECTOR's raw
                # stage positions — the same frame as flake_candidates_v3.json, which
                # _load_candidates shifts by −objective_offset to sit on the
                # (offset-corrected) tiles.  Apply the same shift here or the whole
                # auto set lands ~off (≈21 µm at 20×) off its features.  Stored coords
                # stay raw (navigate/datasheet want the commanded position).
                foff = _obj_offsets.get(f.get('magnification') or '', [0.0, 0.0])
                rx, ry = rx - foff[0], ry - foff[1]
            return rx, ry
        if f.get('source') == 'map':
            return x, y
        foff = _obj_offsets.get(f.get('magnification') or '', [0.0, 0.0])
        return x - foff[0], y - foff[1]

    # Output location
    if args.output:
        out_dir = Path(args.output).resolve()
    else:
        out_dir = _default_out_dir(roots[0], scans)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = args.name or _default_stem(roots[0], scans)

    if args.update_html:
        # If no DZI files in the computed out_dir, search sibling directories
        # (handles maps generated with an older layout before maps/ was introduced)
        if not args.output and not any(out_dir.glob('*.dzi')):
            for alt in [out_dir.parent, out_dir.parent / 'scans']:
                if alt.is_dir() and any(alt.glob('*.dzi')):
                    print(f"  DZI files found in …/{alt.name}/ — using that as output directory")
                    out_dir = alt
                    break

        print("\nRebuilding HTML from existing DZI files…")
        layers_info = _rebuild_layers_info(stem, out_dir, scans)
        if not layers_info:
            sys.exit("No DZI layers found — run without --update-html first.")

        all_candidates: list = []
        seen_ids: set = set()
        for scan in scans:
            off = _obj_offsets.get(scan['mag'], [0.0, 0.0])
            cand_path = scan['folder'] / 'flake_candidates_v3.json'
            if not cand_path.exists():
                cand_path = scan['folder'] / 'flake_candidates.json'
            if cand_path.exists():
                try:
                    cands = json.loads(cand_path.read_text())
                    if isinstance(cands, list):
                        for c in cands:
                            cid = c.get('id', '')
                            if cid not in seen_ids:
                                seen_ids.add(cid)
                                c = dict(c, scan_folder=str(scan['folder']))
                                if abs(off[0]) > 1e-6 or abs(off[1]) > 1e-6:
                                    c = dict(c, x_mm=c['x_mm'] - off[0],
                                                y_mm=c['y_mm'] - off[1])
                                all_candidates.append(c)
                except Exception:
                    pass
        _total = len(all_candidates)
        all_candidates = _select_candidates(all_candidates, args.cand_sort, args.cand_top)
        if all_candidates:
            _sel = (f" (top {args.cand_top}/layer by {args.cand_sort})"
                    if args.cand_top else f" (all, by {args.cand_sort})")
            print(f"Loaded {len(all_candidates)} of {_total} flake candidate(s) for overlay{_sel}")

        sample_flakes: list | None = None
        extents_polygon: list | None = None
        chip_transform: dict | None = None
        _sj_path: Path | None = None
        if args.sample_json:
            _sj_path = Path(args.sample_json)
            if _sj_path.is_dir():
                _sj_path = _sj_path / 'sample.json'
        else:
            _sr = _find_sample_root(scans[0]['folder'])
            if _sr:
                _sj_path = _sr / 'sample.json'
        if _sj_path and _sj_path.exists():
            try:
                sample = json.loads(_sj_path.read_text())
                reg = (sample.get('placement') or {}).get('registration')
                _ctf = (reg or {}).get('chip_transform')
                flakes = sample.get('flakes', [])
                if flakes:
                    norm = []
                    for f in flakes:
                        if f.get('deleted'):
                            continue
                        x = f.get('x_mm') if f.get('x_mm') is not None else f.get('stage_x_mm')
                        y = f.get('y_mm') if f.get('y_mm') is not None else f.get('stage_y_mm')
                        if x is not None and y is not None:
                            x, y = _flake_to_map_frame(f, x, y, _ctf)
                            norm.append(dict(f, x_mm=x, y_mm=y))
                    sample_flakes = norm
                # Coords → map frame via _flake_to_map_frame (chip coords when
                # available; objective-offset fallback only for legacy no-chip flakes).
                ext = sample.get('extents', {})
                poly = ext.get('polygon_mm')
                if poly and len(poly) >= 3:
                    extents_polygon = _inverse_placement(reg, poly) if reg else poly
                chip_transform = (sample.get('placement') or {}).get('registration', {}).get('chip_transform')
                print(f"Sample: {len(sample_flakes or [])} flake(s)"
                      + (f", extents polygon ({len(extents_polygon)} pts)" if extents_polygon else "")
                      + (", chip transform" if chip_transform else ""))
            except Exception as e:
                print(f"Warning: could not load sample.json: {e}")

        _sf = sample.get('folder', '') if 'sample' in dir() else ''
        # Re-apply the camera rotation if this map was built with --rotation-model
        # (the DZIs on disk are already rotation-placed; keep the transform matched).
        _upd_rot = 0.0
        if getattr(args, 'rotation_model', False):
            _upd_rot = _rig.camera_rotation_deg()
        html_path = write_html(stem, layers_info, scans, out_dir,
                               candidates=all_candidates or None,
                               sample_flakes=sample_flakes,
                               extents_polygon=extents_polygon,
                               nav_port=args.nav_port,
                               chip_transform=chip_transform,
                               sample_folder=_sf,
                               rotation_deg=_upd_rot,
                               scan_placement=(scans[0].get('scan_params', {})
                                               .get('registration') if scans else None))
        print(f"\nDone.  Open: {html_path}")
        if args.open:
            import socket, subprocess, time
            serve_dir = html_path.parent
            with socket.socket() as _s:
                _s.bind(('', 0))
                port = _s.getsockname()[1]
            subprocess.Popen(
                [sys.executable, '-c',
                 'import socketserver,http.server\n'
                 'socketserver.TCPServer.allow_reuse_address=True\n'
                 f'socketserver.ThreadingTCPServer(("",{port}),'
                 'http.server.SimpleHTTPRequestHandler).serve_forever()'],
                cwd=str(serve_dir),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
            time.sleep(0.4)
            url = f"http://localhost:{port}/{html_path.name}"
            print(f"Serving map at {url}")
            subprocess.Popen(['xdg-open', url], start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return

    # Shared: load flake candidates and sample.json (used by both modes)
    def _load_candidates():
        all_cands: list = []
        seen: set = set()
        for scan in scans:
            off = _obj_offsets.get(scan['mag'], [0.0, 0.0])
            cand_path = scan['folder'] / 'flake_candidates_v3.json'
            if not cand_path.exists():
                cand_path = scan['folder'] / 'flake_candidates.json'
            if cand_path.exists():
                try:
                    cands = json.loads(cand_path.read_text())
                    if isinstance(cands, list):
                        for c in cands:
                            cid = c.get('id', '')
                            if cid not in seen:
                                seen.add(cid)
                                c = dict(c, scan_folder=str(scan['folder']))
                                if abs(off[0]) > 1e-6 or abs(off[1]) > 1e-6:
                                    c = dict(c, x_mm=c['x_mm'] - off[0],
                                                y_mm=c['y_mm'] - off[1])
                                all_cands.append(c)
                except Exception:
                    pass
        _total = len(all_cands)
        all_cands = _select_candidates(all_cands, args.cand_sort, args.cand_top)
        if all_cands:
            _sel = (f" (top {args.cand_top}/layer by {args.cand_sort})"
                    if args.cand_top else f" (all, by {args.cand_sort})")
            print(f"Loaded {len(all_cands)} of {_total} flake candidate(s) for overlay{_sel}")
        return all_cands

    def _load_sample():
        sf: list | None = None
        ep: list | None = None
        ct: dict | None = None
        folder: str = ''
        sj_path: Path | None = None
        if args.sample_json:
            sj_path = Path(args.sample_json)
            if sj_path.is_dir():
                sj_path = sj_path / 'sample.json'
        else:
            sr = _find_sample_root(roots[0])
            if sr:
                sj_path = sr / 'sample.json'
        if sj_path and sj_path.exists():
            try:
                sample = json.loads(sj_path.read_text())
                folder = sample.get('folder', '')
                reg = (sample.get('placement') or {}).get('registration')
                _ctf = (reg or {}).get('chip_transform')
                flakes = sample.get('flakes', [])
                if flakes:
                    norm = []
                    for f in flakes:
                        if f.get('deleted'):
                            continue
                        x = f.get('x_mm') if f.get('x_mm') is not None else f.get('stage_x_mm')
                        y = f.get('y_mm') if f.get('y_mm') is not None else f.get('stage_y_mm')
                        if x is not None and y is not None:
                            x, y = _flake_to_map_frame(f, x, y, _ctf)
                            norm.append(dict(f, x_mm=x, y_mm=y))
                    sf = norm
                poly = sample.get('extents', {}).get('polygon_mm')
                if poly and len(poly) >= 3:
                    # Stored extents are in the live current-stage frame; convert
                    # into THIS map's scan frame (identity when the scan was taken
                    # in the current mount — the register-then-scan case that the
                    # old inverse-only path skewed).
                    scan_reg = (scans[0].get('scan_params', {}).get('registration')
                                if scans else None)
                    ep = _to_scan_frame(poly, reg, scan_reg)
                ct = (sample.get('placement') or {}).get('registration', {}).get('chip_transform')
                print(f"Sample: {len(sf or [])} flake(s)"
                      + (f", extents polygon ({len(ep)} pts)" if ep else "")
                      + (", chip transform" if ct else ""))
            except Exception as e:
                print(f"Warning: could not load sample.json: {e}")
        return sf, ep, ct, folder

    # Choose scale (needed by both modes for ORB / assembly)
    canvas_ppm = _choose_canvas_ppm(scans, args.scale)
    refine = not args.no_refine
    if refine and not _HAVE_CV2:
        print("Note: cv2 not available — ORB tile alignment disabled")
        refine = False
    correct_bg = getattr(args, 'correct_bg', False)

    # Stored camera rotation (written by a previous --save-rotation run).
    # None-when-absent matters here: it gates the --rotation-model error below.
    stored_rotation: float | None = _rig.load_rig_calibration().get('camera_rotation_deg')

    use_rotation_model = getattr(args, 'rotation_model', False)
    rotate_tiles       = getattr(args, 'rotate_tiles', False)
    save_rotation      = getattr(args, 'save_rotation', False)

    if use_rotation_model:
        if stored_rotation is None:
            sys.exit("--rotation-model: no camera_rotation_deg in calibration.json — "
                     "run once with --save-rotation first.")
        if not _HAVE_CV2 and rotate_tiles:
            print("Note: cv2 not available — --rotate-tiles ignored")
            rotate_tiles = False
        refine = False   # rotation model replaces ORB for positioning

    if not args.overview:
        # Per-frame DZI mode (default): each source frame gets its own tile pyramid.
        # Native resolution — no canvas size ceiling.
        global_x0, global_y0, global_x1, global_y1 = all_bounds(scans)
        global_phys_w = global_x1 - global_x0

        if use_rotation_model:
            extras = f'rotation model θ={stored_rotation:+.4f}°'
            if rotate_tiles:
                extras += ', tiles straightened + grid un-rotated'
        else:
            extras = 'with ORB alignment' if refine else 'no ORB (cv2 missing)'
        print(f"\nWriting per-frame DZIs ({extras})…")
        frame_data, corr_vectors, extracted_theta = write_per_frame_dzis(
            scans, canvas_ppm, out_dir, stem,
            global_x0, global_y0, global_phys_w,
            refine=refine,
            correct_bg=correct_bg,
            jpeg_quality=args.jpeg_quality,
            rotation_model=stored_rotation if use_rotation_model else None,
            rotate_tiles=rotate_tiles,
        )

        # Persist extracted rotation angle to calibration.json
        if save_rotation and extracted_theta is not None:
            try:
                _cal_path2 = _APP_DIR / 'calibration.json'
                _cal_w = json.loads(_cal_path2.read_text()) if _cal_path2.exists() else {}
                _cal_w['camera_rotation_deg'] = round(extracted_theta, 6)
                _cal_path2.write_text(json.dumps(_cal_w, indent=2))
                print(f"  Saved camera_rotation_deg={extracted_theta:+.6f}° to calibration.json")
            except Exception as e:
                print(f"  Warning: could not write calibration.json: {e}")

        all_candidates = _load_candidates()
        sample_flakes, extents_polygon, chip_transform, sample_folder = _load_sample()
        with _stage("HTML build"):
            html_path = write_html(stem, [], scans, out_dir,
                                   candidates=all_candidates or None,
                                   sample_flakes=sample_flakes,
                                   extents_polygon=extents_polygon,
                                   nav_port=args.nav_port,
                                   frame_data=frame_data,
                                   chip_transform=chip_transform,
                                   sample_folder=sample_folder,
                                   corr_vectors=corr_vectors,
                                   rotation_deg=(stored_rotation if use_rotation_model else 0.0),
                                   scan_placement=(scans[0].get('scan_params', {})
                                                   .get('registration') if scans else None))
    else:
        # Overview / legacy mode: single composited canvas per mag.
        blend = getattr(args, 'blend', False)
        extras_str = ', '.join(filter(None, ['bg correction' if correct_bg else '',
                                             'blend' if blend else '']))
        if refine:
            print(f"\nAssembling layers (with ORB tile alignment"
                  + (f", {extras_str}" if extras_str else "") + ")…")
        else:
            print(f"\nAssembling layers" + (f" ({extras_str})" if extras_str else "") + "…")
        with _stage("assemble (stitch+composite)"):
            layers = assemble_layers(scans, canvas_ppm, refine=refine,
                                     correct_bg=correct_bg, blend=blend)

        print("\nWriting DZI tiles…")
        global_x0, global_y0, global_x1, global_y1 = all_bounds(scans)
        global_phys_w = global_x1 - global_x0

        layers_info = []
        for mag, layer_img in layers.items():
            layer_stem = f"{stem}_{mag}"
            with _stage("DZI encode+write (canvas)"):
                write_dzi(layer_img, layer_stem, out_dir, jpeg_quality=args.jpeg_quality)
            info = layer_img.info
            layers_info.append({
                'mag':     mag,
                'stem':    layer_stem,
                'w':       layer_img.width,
                'h':       layer_img.height,
                'fmt':     'png',
                'x0_mm':   info['x0_mm'],
                'y0_mm':   info['y0_mm'],
                'phys_w':  info['phys_w'],
                'phys_h':  info['phys_h'],
                'osd_x':     (info['x0_mm'] - global_x0) / global_phys_w,
                'osd_y':     (info['y0_mm'] - global_y0) / global_phys_w,
                'osd_width':  info['phys_w']              / global_phys_w,
                'tile_centres': info.get('tile_centres', []),
            })

        all_candidates = _load_candidates()
        sample_flakes, extents_polygon, chip_transform, sample_folder = _load_sample()
        with _stage("HTML build"):
            html_path = write_html(stem, layers_info, scans, out_dir,
                                   candidates=all_candidates or None,
                                   sample_flakes=sample_flakes,
                                   extents_polygon=extents_polygon,
                                   nav_port=args.nav_port,
                                   chip_transform=chip_transform,
                                   sample_folder=sample_folder,
                                   scan_placement=(scans[0].get('scan_params', {})
                                                   .get('registration') if scans else None))

    _print_stage_times()
    print(f"\nDone.  Open: {html_path}")

    if args.open:
        import socket, subprocess, time
        # Chrome blocks XMLHttpRequest/fetch from file:// origins, so tiles
        # silently fail to load and the viewer shows blank.  Spin up a
        # detached Python HTTP server (start_new_session=True so it outlives
        # this process) and open the browser at http://localhost:PORT/.
        serve_dir = html_path.parent
        with socket.socket() as _s:
            _s.bind(('', 0))
            port = _s.getsockname()[1]
        subprocess.Popen(
            [sys.executable, '-c',
             'import socketserver,http.server\n'
             'socketserver.TCPServer.allow_reuse_address=True\n'
             f'socketserver.ThreadingTCPServer(("",{port}),'
             'http.server.SimpleHTTPRequestHandler).serve_forever()'],
            cwd=str(serve_dir),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
        time.sleep(0.4)   # let the server bind before the browser requests
        url = f"http://localhost:{port}/{html_path.name}"
        print(f"Serving map at {url}")
        subprocess.Popen(['xdg-open', url], start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == '__main__':
    main()

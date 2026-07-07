#!/usr/bin/env python3
"""Import auto-detected flakes (flakes_found.json from calibrate_ladder) into the
sample catalogue, and backfill focus Z for map-marked flakes.

Also reads flake_candidates_v3.json (the analytical/oxide detector, #61); both
schemas normalise to the same catalogue flake.

Each found flake becomes a catalogue flake with:
  - source='auto', confirmed=False (find → navigate → confirm workflow)
  - detect_method: 'analytical' (classified vs the oxide-derived model; carries
    layer_sigma + layer_resolvable) or 'self_cal' (ladder KDE'd from the scan).
    Finer than source='auto'; neither is verified (still confirmed=False).
  - its measured layer_count, area_um2, and BGR contrast (in notes)
  - a focus Z looked up from the scan's per-tile z_actual_mm (so it navigates IN
    FOCUS, unlike a 2-D map mark that carries z=0)
  - chip coords if the sample is registered
Deduped against existing catalogue flakes. Also backfills z_mm for existing
source='map' flakes (which carry the z=0 placeholder) from the same focus surface.

Usage:  python3 tools/import_found_flakes.py <scan_folder> [--dedup-um 15]
The app should be told to reload the sample afterwards (it writes sample.json).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(APP))
import core.sample_data as sd                       # noqa: E402
from core.scan_io import load_metadata, scan_mag   # noqa: E402


def _load_found(scan: Path) -> list:
    """Load auto-detected flakes from whichever detector wrote this scan.

    Two producers, two schemas — normalise both to the `flakes_found.json` shape
    the importer consumes:
      • calibrate_ladder.py  → flakes_found.json  (self-calibrating wand; #49)
      • flake_detect_v3.py   → flake_candidates_v3.json  (analytical/oxide; #61)
    flakes_found.json wins when both exist (it carries pre-made crops).  The v3
    schema has no crop file — the catalogue contact-sheet `--backfill` fills the
    thumbnails afterwards, so we just map the core fields here.
    """
    ff = scan / 'flakes_found.json'
    if ff.exists():
        data = json.loads(ff.read_text())
        for f in data:                       # tag provenance: ladder derived FROM the scan
            f.setdefault('detect_method', 'self_cal')
        return data
    v3 = scan / 'flake_candidates_v3.json'
    if v3.exists():
        cands = json.loads(v3.read_text())
        out = []
        for c in cands:
            if c.get('x_mm') is None or c.get('y_mm') is None:
                continue
            out.append({
                'id':              c.get('id'),      # candidate id (--ids selection)
                'stage_x_mm':      round(float(c['x_mm']), 4),
                'stage_y_mm':      round(float(c['y_mm']), 4),
                'layer_count':     c.get('target_N'),
                'area_um2':        c.get('area_um2'),
                'contrast_bgr_pct': c.get('contrast_bgr_pct'),
                # classified against the oxide-derived ForwardModel targets — carry
                # the layer-number confidence + resolvability so the catalogue can
                # show "3L ±0.4 (model)" rather than a bare "3L (auto)".
                'detect_method':   'analytical',
                'layer_sigma':     c.get('layer_sigma'),
                'layer_resolvable': c.get('layer_resolvable'),
                'source':          'auto',
                'confirmed':       False,
                # v3 writes no crop file; carry the source tile + detected geometry
                # so the importer can cut outlined + plain crops for the catalogue.
                'crop_file':       None,
                'crop_um':         None,
                'source_image':    c.get('source_image'),
                'bbox_px':         c.get('bbox_px'),
                'contour_px':      c.get('contour_px'),
            })
        print(f"loaded {len(out)} candidates from flake_candidates_v3.json (analytical)")
        return out
    sys.exit(f"No flakes_found.json or flake_candidates_v3.json in {scan} — "
             "run detection first.")


def _shortlist(found: list, *, rank='area', per_layer=0, min_area=0.0, max_area=0.0) -> list:
    """Rank + cap the found list so the catalogue isn't flooded.

    calibrate_ladder pre-shortlists before writing flakes_found.json, but
    flake_detect_v3 writes every candidate — so the analytical path relies on
    this.  Defaults are no-ops (per_layer=0, no area bounds) → the self-calibrating
    path is unaffected.
    """
    fs = list(found)
    if min_area > 0:
        fs = [f for f in fs if (f.get('area_um2') or 0) >= min_area]
    if max_area > 0:
        fs = [f for f in fs if (f.get('area_um2') or 0) <= max_area]
    keyf = {'area':     lambda f: -(f.get('area_um2') or 0),
            'contrast': lambda f: -abs((f.get('contrast_bgr_pct') or [0, 0, 0])[2])}
    fs.sort(key=keyf.get(rank, keyf['area']))
    if per_layer and per_layer > 0:
        seen, out = {}, []
        for f in fs:
            n = f.get('layer_count')
            if seen.get(n, 0) < per_layer:
                seen[n] = seen.get(n, 0) + 1
                out.append(f)
        fs = out
    return fs


def _dashed_contour(img, pts, color, thickness=2, dash=9.0, gap=6.0):
    """Draw a closed DASHED polyline through pts (list of (x,y)).  cv2 has no
    dashed stroke; the dash/gap walk lives in vision.draw.dash_segments
    (shared with the make_map thumbnail dashing)."""
    import cv2
    from vision.draw import dash_segments
    for (ax, ay), (bx, by) in dash_segments(pts, dash, gap):
        cv2.line(img, (int(round(ax)), int(round(ay))),
                 (int(round(bx)), int(round(by))),
                 color, thickness, cv2.LINE_AA)


def _make_v3_crops(scan: Path, d: dict, meta: dict, images_dir: Path, fid: str, mag: str) -> list:
    """Cut an OUTLINED and a PLAIN crop from the source tile for an analytical
    (v3) flake, save both into the sample's images/, return their image-dict
    entries.  self_cal flakes already ship a crop; this fills the v3 gap so the
    catalogue Images look the same for both detectors.

    The OUTLINED crop carries the detected boundary (dashed cyan) + a small µm
    scale bar; the PLAIN crop is left completely clean so there's always an
    un-annotated version to fall back on."""
    src = d.get('source_image'); bbox = d.get('bbox_px')
    if not src or not bbox:
        return []
    import cv2
    from vision import scale_bar
    tile = cv2.imread(str(scan / src))
    if tile is None:
        return []
    H, W = tile.shape[:2]
    x, y, w, h = bbox
    pad = int(max(w, h) * 0.4) + 20            # breathing room around the flake
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    plain = tile[y0:y1, x0:x1].copy()
    if plain.size == 0:
        return []
    ppu = None
    try:
        from vision.camera_params import px_per_um
        ppu = px_per_um(mag, meta['imaging']['frame_width'])
    except Exception:
        pass
    crop_um = round((x1 - x0) / ppu, 1) if ppu else None

    outlined = plain.copy()
    cnt = d.get('contour_px')
    if cnt:                                     # detected boundary → dashed cyan (protanopia-safe)
        _dashed_contour(outlined, [(px - x0, py - y0) for px, py in cnt], (255, 255, 0), 2)
    scale_bar.draw_cv2(outlined, ppu)           # small unified bar (outlined crop only)

    imgs = []
    for suffix, arr, outlined_flag in (('outlined', outlined, True), ('plain', plain, False)):
        dest = f"{fid}_auto_{suffix}.png"
        if cv2.imwrite(str(images_dir / dest), arr):
            imgs.append({'file': dest, 'mag': mag, 'type': 'crop',
                         'crop_um': crop_um, 'outlined': outlined_flag})
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scan_folder')
    ap.add_argument('--dedup-um', type=float, default=15.0,
                    help='Skip a found flake within this distance of an existing one')
    ap.add_argument('--per-layer', type=int, default=0,
                    help='Keep at most N per layer count (0 = no cap; the analytical '
                         'v3 detector writes all candidates, so cap here)')
    ap.add_argument('--min-area', type=float, default=0.0, help='µm² floor')
    ap.add_argument('--max-area', type=float, default=0.0, help='µm² ceiling (0 = none)')
    ap.add_argument('--rank', choices=['area', 'contrast'], default='area')
    ap.add_argument('--ids', default=None,
                    help='Comma-separated candidate ids: import exactly these, '
                         'bypassing --per-layer/--rank/area shortlisting '
                         '(dedup-against-existing and crop cutting still apply)')
    args = ap.parse_args()

    scan = Path(args.scan_folder)
    found_all = _load_found(scan)
    if args.ids:
        # Explicit user selection (Flake Results panel "Add checked"): exactly
        # these candidates, no shortlist.  Additive — the idempotent clear of
        # prior auto flakes below is skipped so earlier keepers are preserved.
        want = [t.strip() for t in args.ids.split(',') if t.strip()]
        by_id = {str(f.get('id')): f for f in found_all if f.get('id')}
        missing = [i for i in want if i not in by_id]
        if missing:
            print(f"warning: {len(missing)} id(s) not in candidates: "
                  f"{', '.join(missing)}")
        found = [by_id[i] for i in want if i in by_id]
        if not found:
            sys.exit('none of the requested --ids matched any candidate.')
        print(f"selected {len(found)} candidate(s) by id")
    else:
        found = _shortlist(found_all, rank=args.rank, per_layer=args.per_layer,
                           min_area=args.min_area, max_area=args.max_area)
    meta = load_metadata(scan)
    images = meta['images']
    mag = scan_mag(meta, default='20x')

    sdir = sd.find_sample_dir_upwards(str(scan))
    sjp = Path(sdir) / 'sample.json' if sdir else None
    if not sjp:
        sys.exit('No sample.json found above the scan folder.')
    sample = json.loads(sjp.read_text())
    flakes = sample.setdefault('flakes', [])
    shutil.copy2(sjp, str(sjp) + '.bak_import')

    # Idempotent re-import: drop the previous auto-detected flakes so a re-run
    # cleanly REPLACES them instead of duplicating or silently deduping against
    # its own stale copies.  Hand-marked (source 'map'/'app') flakes are kept and
    # still dedup-protected below, so a confirmed flake is never duplicated.
    # SKIPPED for --ids: an explicit selection is additive — clearing would
    # delete previously imported keepers; dedup below still prevents doubles.
    if not args.ids:
        n_auto_old = sum(1 for f in flakes if f.get('source') == 'auto')
        flakes[:] = [f for f in flakes if f.get('source') != 'auto']
        if n_auto_old:
            print(f"cleared {n_auto_old} prior auto flakes (idempotent re-import)")

    images_dir = sjp.parent / 'images'; images_dir.mkdir(exist_ok=True)
    existing = [(f['stage_x_mm'], f['stage_y_mm']) for f in flakes
                if f.get('stage_x_mm') is not None]
    r2 = (args.dedup_um / 1000.0) ** 2
    added = 0
    for d in found:
        sx, sy = d['stage_x_mm'], d['stage_y_mm']
        if any((sx - ex) ** 2 + (sy - ey) ** 2 < r2 for ex, ey in existing):
            continue
        z = sd.focus_z_at(images, sx, sy) or 0.0
        fid = sd.next_flake_id(flakes)
        fl = sd.new_flake(fid, '', sx, sy, z, mag, layer_count=d.get('layer_count'),
                          source='auto', confirmed=False)
        fl['area_um2'] = d.get('area_um2')
        # Provenance beyond the coarse source='auto': how the layer count was
        # arrived at.  'analytical' = classified against the oxide-derived model
        # (carries a layer-number ±σ and resolvability); 'self_cal' = ladder KDE'd
        # from this scan (relative).  Neither is verified — still confirmed=False.
        method = d.get('detect_method', 'self_cal')
        fl['detect_method'] = method
        sigma = d.get('layer_sigma')
        if sigma is not None:
            fl['layer_sigma'] = sigma
        if d.get('layer_resolvable') is not None:
            fl['layer_resolvable'] = d.get('layer_resolvable')
        tag = {'analytical': 'model', 'self_cal': 'self-cal'}.get(method, 'auto')
        n = d.get('layer_count')
        layer_txt = (f"{n}L" if n is not None else "?L") + (f" ±{sigma}" if sigma is not None else "")
        c = d.get('contrast_bgr_pct')
        note_bits = [f"{tag}: {layer_txt}"]
        if c:
            note_bits.append(f"BGR {c}% (R {c[2]}%)")
        fl['notes'] = "  ".join(note_bits)
        # Persist the detector's evidence so downstream renderers (datasheet
        # tile panels, future map layers) can draw the TRUE dashed contour
        # instead of a position-derived box (datasheet feedback 2026-07-03).
        if d.get('source_image'):
            fl['source_image'] = d['source_image']
        if d.get('contour_px'):
            fl['contour_px'] = d['contour_px']
        if d.get('bbox_px'):
            fl['bbox_px'] = d['bbox_px']
        # Auto-flake stage coords are REFERENCE-stage: the scan defines the
        # reference frame (docs/COORDINATE_SYSTEMS.md).  Map straight to chip via
        # the chip_transform — do NOT use stage_to_chip(), which strips the
        # *current* placement (right only for a current-stage input).  Stripping
        # a placement the scan never had makes chip coords round-trip back to raw
        # reference, so a later remount's placement is silently dropped and
        # flake-table navigate misses by exactly that placement.
        chip_tf = sd.get_chip_transform(sample)
        if chip_tf:
            from vision.registration import reference_stage_to_chip
            cx_chip, cy_chip = reference_stage_to_chip(chip_tf, sx, sy)
            fl['chip_x_mm'], fl['chip_y_mm'] = round(cx_chip, 6), round(cy_chip, 6)
        # Attach the detector's crop as a catalogue thumbnail (copied into images/,
        # so it shows exactly like an app-captured flake and survives scan deletion).
        crop_rel = d.get('crop_file')
        if crop_rel and (scan / crop_rel).exists():
            dest = f"{fid}_auto_crop.png"
            shutil.copy2(str(scan / crop_rel), str(images_dir / dest))
            crop_um = d.get('crop_um')
            if crop_um is None:                     # older flakes_found.json: recompute
                from vision.camera_params import px_per_um
                from vision.scan_crops import crop_radius_px
                _ppu = px_per_um(mag, meta['imaging']['frame_width'])
                if _ppu:
                    crop_um = round(2 * crop_radius_px(d.get('area_um2'), _ppu) / _ppu, 1)
            fl['images'].append({'file': dest, 'mag': mag, 'type': 'crop', 'crop_um': crop_um})
        elif d.get('source_image'):                 # analytical (v3): cut crops now
            fl['images'].extend(_make_v3_crops(scan, d, meta, images_dir, fid, mag))
        flakes.append(fl); existing.append((sx, sy)); added += 1

    # Backfill focus Z for map-marked flakes that carry the z=0 placeholder.
    backfilled = 0
    for f in flakes:
        if f.get('source') == 'map' and not f.get('z_mm'):
            z = sd.focus_z_at(images, f.get('stage_x_mm'), f.get('stage_y_mm'))
            if z:
                f['z_mm'] = round(z, 5); backfilled += 1

    sjp.write_text(json.dumps(sample, indent=1))
    methods = {}
    for f in flakes:
        if f.get('source') == 'auto':
            methods[f.get('detect_method', 'auto')] = methods.get(f.get('detect_method', 'auto'), 0) + 1
    method_txt = ", ".join(f"{k}={v}" for k, v in sorted(methods.items())) or "none"
    print(f"imported {added} auto flakes (source=auto, confirmed=false, focus Z stamped; "
          f"by method: {method_txt})")
    print(f"backfilled focus Z for {backfilled} map-marked flakes")
    print(f"catalogue now {len(flakes)} flakes — reload the sample in the app")
    print(f"backup: {sjp}.bak_import")


if __name__ == '__main__':
    main()

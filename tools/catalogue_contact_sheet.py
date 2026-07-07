#!/usr/bin/env python3
"""Contact sheet of a sample's CATALOGUE flakes (F-numbered, from sample.json).

Unlike calibrate_ladder's layer_crops.png (the raw detector shortlist), this draws
the actual catalogue entries — auto, map and app flakes alike — using each flake's
registered crop in images/.  A title banner with the sample/map name sits at the top.

Usage:
    python tools/catalogue_contact_sheet.py <sample dir | sample.json> [--output PATH]
                                            [--cols N] [--cell PX]
"""
import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np

# Allow `python tools/catalogue_contact_sheet.py …` to import vision.* (for --backfill).
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_CELL = 200          # crop size (px)
_LABEL = 30          # per-cell label strip (px)
_PAD = 6
_TITLE_H = 56        # title banner height (px)


def _label_for(f):
    lc = f.get('layer_count')
    layer = f"{int(lc)}L" if lc else '?'
    area = f.get('area_um2')
    area_s = f"{area:.0f}um2" if area else ''
    return (f"{f.get('id','')}  {layer}  {area_s}".strip(),
            f"{f.get('source','')}  @{f.get('stage_x_mm',0):+.2f},{f.get('stage_y_mm',0):+.2f}")


def _crop_um(f, image):
    """Full physical width (µm) of a flake's crop: stored value, else recomputed
    from area + magnification (so it works on crops made before crop_um existed)."""
    um = (image or {}).get('crop_um')
    if um:
        return um
    try:
        from vision.camera_params import px_per_um
        from vision.scan_crops import crop_radius_px
        ppu = px_per_um(f.get('magnification') or '20x', 2464) or px_per_um('20x', 2464)
        return 2 * crop_radius_px(f.get('area_um2'), ppu) / ppu
    except Exception:
        return None


def _draw_scale_bar(cell, crop_um):
    """Unified µm scale bar (see vision.scale_bar), bottom-left of the crop."""
    if not crop_um or crop_um <= 0:
        return
    from vision import scale_bar
    scale_bar.draw_cv2(cell, _CELL / crop_um)


def _cell_image(f, images_dir):
    """Load a flake's crop (prefer type 'crop'), resized to _CELL, or a placeholder."""
    imgs = f.get('images') or []
    pick = next((im for im in imgs if im.get('type') == 'crop'), imgs[0] if imgs else None)
    cell = None
    if pick:
        p = os.path.join(images_dir, pick['file'])
        img = cv2.imread(p)
        if img is not None:
            cell = cv2.resize(img, (_CELL, _CELL))
    if cell is None:
        cell = np.full((_CELL, _CELL, 3), 40, np.uint8)
        cv2.putText(cell, 'no image', (_CELL // 2 - 42, _CELL // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)
    else:
        _draw_scale_bar(cell, _crop_um(f, pick))
    # label strip on top
    out = cv2.copyMakeBorder(cell, _LABEL, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    l1, l2 = _label_for(f)
    cv2.putText(out, l1, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    cv2.putText(out, l2, (4, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1)
    return out


def backfill_crops(sample, sample_dir, mag=None):
    """Crop+register a thumbnail for any flake missing an image, from the scan.
    Mutates `sample` in place; returns the number backfilled."""
    from vision.scan_crops import find_scan_folder, crop_at_stage
    scan = find_scan_folder(sample_dir, mag)
    if not scan:
        return 0
    images_dir = os.path.join(sample_dir, 'images'); os.makedirs(images_dir, exist_ok=True)
    n = 0
    for f in sample.get('flakes', []):
        if f.get('images') or f.get('stage_x_mm') is None:
            continue
        crop, crop_um = crop_at_stage(scan, f['stage_x_mm'], f['stage_y_mm'], f.get('area_um2'))
        if crop is None:
            continue
        fn = f"{f['id']}_crop.png"
        cv2.imwrite(os.path.join(images_dir, fn), crop)
        f.setdefault('images', []).append(
            {'file': fn, 'mag': f.get('magnification', ''), 'type': 'crop', 'crop_um': crop_um})
        n += 1
    return n


def build(sample_path, output=None, cols=6, cell=_CELL, backfill=False):
    sample_path = os.path.abspath(sample_path)
    sample_dir = sample_path if os.path.isdir(sample_path) else os.path.dirname(sample_path)
    sjp = sample_path if sample_path.endswith('.json') else os.path.join(sample_dir, 'sample.json')
    sample = json.loads(open(sjp).read())
    images_dir = os.path.join(sample_dir, 'images')

    if backfill:
        n = backfill_crops(sample, sample_dir)
        if n:
            json.dump(sample, open(sjp, 'w'), indent=1)
            print(f"backfilled crops for {n} flake(s)")

    flakes = sample.get('flakes', [])
    # Sort: by layer (unknown last), then area desc, then id.
    flakes = sorted(flakes, key=lambda f: (f.get('layer_count') or 99,
                                           -(f.get('area_um2') or 0), f.get('id', '')))
    if not flakes:
        raise SystemExit('No flakes in catalogue.')

    cells = [_cell_image(f, images_dir) for f in flakes]
    cw, ch = cells[0].shape[1], cells[0].shape[0]
    rows = (len(cells) + cols - 1) // cols
    grid_w = cols * cw + (cols + 1) * _PAD
    grid_h = rows * ch + (rows + 1) * _PAD

    sheet = np.full((_TITLE_H + grid_h, grid_w, 3), 25, np.uint8)
    for i, c in enumerate(cells):
        r, k = divmod(i, cols)
        y = _TITLE_H + _PAD + r * (ch + _PAD)
        x = _PAD + k * (cw + _PAD)
        sheet[y:y + ch, x:x + cw] = c

    # ── title banner ──────────────────────────────────────────────────────────
    name = sample.get('name', '') or os.path.basename(sample_dir)
    folder = sample.get('folder', os.path.basename(sample_dir))
    state = sample.get('process_state', '')
    n_by_src = {}
    for f in flakes:
        n_by_src[f.get('source', '?')] = n_by_src.get(f.get('source', '?'), 0) + 1
    title = folder + (f"  [{state}]" if state else '')
    sub = f"{len(flakes)} flakes  (" + ', '.join(f"{k}:{v}" for k, v in sorted(n_by_src.items())) + \
          f")   {date.today().isoformat()}"
    cv2.putText(sheet, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(sheet, sub, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1)

    out = output or os.path.join(sample_dir, f"{folder}_contact_sheet.png")
    cv2.imwrite(out, sheet)
    print(f"wrote {out}  ({len(flakes)} flakes, {rows}x{cols})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sample', help='Sample directory or sample.json path')
    ap.add_argument('--output', default=None)
    ap.add_argument('--cols', type=int, default=6)
    ap.add_argument('--cell', type=int, default=_CELL)
    ap.add_argument('--backfill', action='store_true',
                    help='Crop+register a thumbnail from the scan for any flake missing one '
                         '(writes sample.json)')
    args = ap.parse_args()
    build(args.sample, args.output, args.cols, args.cell, args.backfill)


if __name__ == '__main__':
    main()

"""DZI pyramid encoding for area-scan maps (plan task L5).

Extracted from tools/make_map.py: write_dzi (single canvas -> DZI descriptor +
tile pyramid) and write_per_frame_dzis (per-tile DZIs placed by the rotation
model, threaded). Position solving lives in vision.map_stitch; make_map is the
CLI + HTML generator.
"""
import math
import shutil
import sys
import xml.etree.ElementTree as ET
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

from vision.map_stitch import (_stage, _px_per_um, _extract_camera_rotation_deg,
                               _rotation_model_positions, _build_grid,
                               _precompute_orb, _propagate_positions,
                               _load_saved_flat, _apply_flat, _bg_correct_tile)
from core.scan_io import scan_bounds, all_bounds  # noqa: F401
from vision.camera_params import MAG_SCALE as _MAG_SCALE


TILE_SIZE    = 256
TILE_OVERLAP = 1


def write_dzi(image: Image.Image, stem: str, out_dir: Path,
              jpeg_quality: int = 85, verbose: bool = True) -> Path:
    """
    Write a Deep Zoom Image tile pyramid.  RGBA images use PNG tiles; RGB uses JPEG.

    Returns path to the .dzi descriptor file.
    """
    is_rgba  = image.mode == 'RGBA'
    tile_fmt = 'PNG'  if is_rgba else 'JPEG'
    tile_ext = 'png'  if is_rgba else 'jpg'

    tiles_dir = out_dir / f"{stem}_files"
    if tiles_dir.exists():
        shutil.rmtree(tiles_dir)
    tiles_dir.mkdir(parents=True)

    w, h = image.size
    max_level = math.ceil(math.log2(max(w, h)))

    for level in range(max_level, -1, -1):
        level_dir = tiles_dir / str(level)
        level_dir.mkdir()

        scale = 2 ** (level - max_level)
        lw = max(1, round(w * scale))
        lh = max(1, round(h * scale))
        if level == max_level:
            level_img = image
        else:
            level_img = image.resize((lw, lh), Image.LANCZOS)

        cols = math.ceil(lw / TILE_SIZE)
        rows = math.ceil(lh / TILE_SIZE)

        for col in range(cols):
            for row in range(rows):
                x0 = max(0, col * TILE_SIZE - TILE_OVERLAP)
                y0 = max(0, row * TILE_SIZE - TILE_OVERLAP)
                x1 = min(lw, (col + 1) * TILE_SIZE + TILE_OVERLAP)
                y1 = min(lh, (row + 1) * TILE_SIZE + TILE_OVERLAP)
                tile = level_img.crop((x0, y0, x1, y1))
                if is_rgba:
                    # Skip tiles that are entirely transparent — OSD shows nothing
                    # there anyway, and skipping saves significant write time for
                    # sparse high-mag layers.
                    tile_arr = np.array(tile)
                    if tile_arr[:, :, 3].max() == 0:
                        continue
                    tile.save(level_dir / f"{col}_{row}.{tile_ext}", tile_fmt,
                              compress_level=1)   # fast write; alpha compresses well
                else:
                    tile.save(level_dir / f"{col}_{row}.{tile_ext}", tile_fmt,
                              quality=jpeg_quality, optimize=True)

    # Write DZI XML descriptor
    dzi_path = out_dir / f"{stem}.dzi"
    root = ET.Element('Image',
                       xmlns='http://schemas.microsoft.com/deepzoom/2008',
                       TileSize=str(TILE_SIZE),
                       Overlap=str(TILE_OVERLAP),
                       Format=tile_ext)
    ET.SubElement(root, 'Size', Width=str(w), Height=str(h))
    tree = ET.ElementTree(root)
    ET.indent(tree, space='  ')
    tree.write(str(dzi_path), xml_declaration=True, encoding='utf-8')

    if verbose:
        print(f"DZI written: {dzi_path}  ({max_level+1} levels, {w}×{h} px, {tile_fmt})")
    return dzi_path


def _dzi_worker(job: tuple) -> tuple:
    """Thread worker: load, flat-correct, optionally de-rotate, and write one frame DZI."""
    img_path, flat_np, frames_dir, idx, jpeg_quality, rotate_deg = job
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        return idx, str(e)
    if flat_np is not None:
        img = _apply_flat(img, flat_np)
    if rotate_deg and _HAVE_CV2:
        import numpy as np
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        h, w = arr.shape[:2]
        # +rotate_deg straightens the camera tilt out of the content: a +θ camera
        # rotation puts a +θ tilt in the frame, so rotating the image by +θ here
        # cancels it (verified against tile-overlap phase correlation: the H-pair
        # cross-axis offset goes to ~0; the old -θ doubled it).
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rotate_deg, 1.0)
        arr = cv2.warpAffine(arr, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    write_dzi(img, str(idx), frames_dir, jpeg_quality=jpeg_quality, verbose=False)
    return idx, None


def write_per_frame_dzis(scans: list, canvas_ppm: float,
                         out_dir: Path, stem: str,
                         global_x0: float, global_y0: float,
                         global_phys_w: float,
                         refine: bool = True,
                         correct_bg: bool = False,
                         jpeg_quality: int = 85,
                         rotation_model: float | None = None,
                         rotate_tiles: bool = False) -> tuple:
    """
    Write one DZI per source frame at native resolution.

    ORB corrections are computed at canvas_ppm scale for sub-pixel accuracy
    but applied only to each frame's world-coordinate placement — the tile
    image itself is always written at full native resolution.

    rotation_model: if a float (degrees), use _rotation_model_positions instead of
        BFS + per-pair ORB corrections.  Pass the camera_rotation_deg from
        calibration.json.  Skips ORB entirely for positioning (much faster, no
        outliers, works for filtered scans where ORB/PC fail).

    rotate_tiles: if True, de-rotate each tile image by +rotation_model degrees
        (straightening the camera tilt out of the content) AND place tiles at
        plain stage positions instead of R(θ)·Δstage.  The two must go together:
        de-rotated content has no inter-tile tilt, so it stitches against an
        un-rotated grid.  Result is a stage-axis-aligned map.  (Requires
        rotation_model to be set and cv2 available.)

    Frame DZIs are written in parallel using a thread pool (Pillow releases
    the GIL for JPEG compression and resize, giving real concurrency).

    Returns (frame_data, corr_vectors).
    """
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    all_frame_data: list = []
    sorted_scans = sorted(scans, key=lambda s: _MAG_SCALE[s['mag']])
    all_h_corrs: list = []
    all_v_corrs: list = []
    all_corr_vectors: list = []   # [[x_mm, y_mm, dx_mm, dy_mm, method]] per tile
    all_h_corrs_theta: list = []  # rotation angles extracted per scan (degrees)
    rotate_deg = rotation_model if (rotate_tiles and rotation_model is not None) else 0.0

    n_workers = min(os.cpu_count() or 4, 8)

    for scan in sorted_scans:
        folder     = scan['folder']
        mag        = scan['mag']
        native_ppm = scan['ppm']
        frame_w    = scan['frame_w']
        frame_h    = scan['frame_h']
        fov_x_mm   = frame_w / native_ppm
        fov_y_mm   = frame_h / native_ppm
        frame_world_w = fov_x_mm / global_phys_w

        layer_id   = f"{mag}_{folder.name}"   # unique per scan, not just per mag
        layer_stem = f"{stem}_{layer_id}"
        frames_dir = out_dir / f"{layer_stem}_frames"
        frames_dir.mkdir(exist_ok=True)

        if rotation_model is not None:
            # Rotation-model mode: use stored camera angle, skip ORB entirely.
            # When rotate_tiles is on, the tile content is straightened to the
            # stage frame (de-rotated by +θ in _dzi_worker), so it must be placed
            # on an un-rotated grid (pos_theta=0); otherwise positions carry the
            # +θ tilt to match the still-tilted content.
            grid = _build_grid(scan['images'])
            pos_theta = 0.0 if rotate_tiles else rotation_model
            positions = _rotation_model_positions(
                grid, pos_theta, canvas_ppm, global_x0, global_y0)
            print(f"  Rotation model [{mag} {folder.name}]: θ={rotation_model:+.4f}°"
                  + ("  (tiles straightened, grid un-rotated)" if rotate_tiles else ""))
            # Populate corr vectors vs. nominal so the arrow overlay still works
            for (row, col), img_meta in grid.items():
                cx_m, cy_m = positions[(row, col)]
                cx_nom = (img_meta['x_mm'] - global_x0) * canvas_ppm
                cy_nom = (img_meta['y_mm'] - global_y0) * canvas_ppm
                all_corr_vectors.append([
                    round(img_meta['x_mm'], 5), round(img_meta['y_mm'], 5),
                    round((cx_m - cx_nom) / canvas_ppm, 6),
                    round((cy_m - cy_nom) / canvas_ppm, 6), 2])
        elif refine and _HAVE_CV2:
            print(f"  ORB [{mag} {folder.name}]…")
            with _stage("stitch: ORB/phase-correlate"):
                grid, h_corr, v_corr, _ = _precompute_orb(scan, canvas_ppm)
            positions = _propagate_positions(grid, h_corr, v_corr, canvas_ppm,
                                             global_x0, global_y0)
            for v in h_corr.values():
                if v[2] != 0:
                    all_h_corrs.append((v[0], v[1], v[2]))
            for v in v_corr.values():
                if v[2] != 0:
                    all_v_corrs.append((v[0], v[1], v[2]))
            # Extract camera rotation from H-pair cross-axis term
            theta = _extract_camera_rotation_deg(h_corr, scan, canvas_ppm)
            if theta is not None:
                all_h_corrs_theta.append(theta)
                print(f"    Camera rotation: {theta:+.4f}°  "
                      f"(use --save-rotation to persist to calibration.json)")
            # Collect per-tile total position corrections for the HTML overlay
            for (row, col), img_meta in grid.items():
                if (row, col) not in positions:
                    continue
                cx_orb, cy_orb = positions[(row, col)]
                cx_nom = (img_meta['x_mm'] - global_x0) * canvas_ppm
                cy_nom = (img_meta['y_mm'] - global_y0) * canvas_ppm
                h_n = h_corr.get((row, col), (0, 0, 0))[2]
                v_n = v_corr.get((row, col), (0, 0, 0))[2]
                method = 0 if (h_n > 0 or v_n > 0) else (1 if (h_n == -1 or v_n == -1) else 2)
                all_corr_vectors.append([
                    round(img_meta['x_mm'], 5), round(img_meta['y_mm'], 5),
                    round((cx_orb - cx_nom) / canvas_ppm, 6),
                    round((cy_orb - cy_nom) / canvas_ppm, 6), method])
        else:
            grid = _build_grid(scan['images'])
            positions = None

        flat_np = None
        if correct_bg:
            flat_np = _load_saved_flat(mag)
            if flat_np is not None:
                print(f"    Flat [{mag}]: loaded flat_field_{mag}.npy")
            else:
                print(f"    Flat [{mag}]: no saved flat_field_{mag}.npy found")

        # Pass 1: compute per-frame world positions (fast, serial)
        frame_meta: list = []   # [(i, img_path, world_x, world_y, x_mm, y_mm)]
        for i, ((row, col), img_meta) in enumerate(sorted(grid.items())):
            img_path = folder / img_meta['filename']
            if not img_path.exists():
                continue
            if positions is not None and (row, col) in positions:
                cx_c, cy_c = positions[(row, col)]
                cx_mm = global_x0 + cx_c / canvas_ppm
                cy_mm = global_y0 + cy_c / canvas_ppm
            else:
                cx_mm = img_meta['x_mm']
                cy_mm = img_meta['y_mm']
            world_x = (cx_mm - fov_x_mm / 2 - global_x0) / global_phys_w
            world_y = (cy_mm - fov_y_mm / 2 - global_y0) / global_phys_w
            frame_meta.append((i, img_path, world_x, world_y,
                                img_meta['x_mm'], img_meta['y_mm']))

        # Pass 2: write DZIs in parallel
        n = len(frame_meta)
        jobs = [(img_path, flat_np, frames_dir, i, jpeg_quality, rotate_deg)
                for (i, img_path, _, _, _, _) in frame_meta]
        completed = 0
        print(f"  Writing {n} frame DZIs [{mag} {folder.name}] "
              f"({n_workers} threads)…")
        with _stage("DZI encode+write (parallel)"), \
                ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_dzi_worker, job): job[2] for job in jobs}
            for fut in as_completed(futs):
                idx, err = fut.result()
                if err:
                    print(f"\n  Warning: frame {idx}: {err}")
                completed += 1
                if completed % 50 == 0 or completed == n:
                    print(f"  [{mag} {folder.name}]  {completed}/{n}", end='\r')
        print()

        # Accumulate frame descriptors in stable grid order
        for (i, img_path, world_x, world_y, x_mm, y_mm) in frame_meta:
            fd = {
                'dzi':   f"{layer_stem}_frames/{i}.dzi",
                'x':     round(world_x, 8),
                'y':     round(world_y, 8),
                'w':     round(frame_world_w, 8),
                'layer': layer_id,
                'mag':   mag,
                'x_mm':  round(x_mm, 5),
                'y_mm':  round(y_mm, 5),
                'fw':    frame_w,
                'fh':    frame_h,
            }
            all_frame_data.append(fd)

    def _corr_stats(corrs, label):
        if not corrs:
            return
        dxs = [c[0] for c in corrs]; dys = [c[1] for c in corrs]
        n_  = len(corrs)
        mx  = sum(dxs)/n_; sx = (sum((v-mx)**2 for v in dxs)/n_)**0.5
        my  = sum(dys)/n_; sy = (sum((v-my)**2 for v in dys)/n_)**0.5
        flag = '← systematic' if abs(mx) > sx or abs(my) > sy else '← jitter'
        print(f"  {label} ({n_}):  "
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
            print(f"    clipped ({n_c}/{n_}, {n_out} outlier(s) removed):  "
                  f"dx={mx_c:+.1f}±{sx_c:.1f}  dy={my_c:+.1f}±{sy_c:.1f}")

        orb = [c for c in corrs if c[2] > 0]
        if orb and len(orb) < n_:
            odxs = [c[0] for c in orb];  odys = [c[1] for c in orb]
            no = len(orb)
            mox = sum(odxs)/no;  sox = (sum((v-mox)**2 for v in odxs)/no)**0.5
            moy = sum(odys)/no;  soy = (sum((v-moy)**2 for v in odys)/no)**0.5
            print(f"    ORB-only ({no}/{n_}):  "
                  f"dx={mox:+.1f}±{sox:.1f}  dy={moy:+.1f}±{soy:.1f}")

    if refine and (all_h_corrs or all_v_corrs):
        print("\nORB corrections (canvas_ppm space, centre-out BFS):")
        _corr_stats(all_h_corrs, "H-pairs")
        _corr_stats(all_v_corrs, "V-pairs")

    extracted_theta: float | None = None
    if all_h_corrs_theta:
        import statistics
        extracted_theta = statistics.mean(all_h_corrs_theta)
        print(f"\nExtracted camera rotation: {extracted_theta:+.4f}°"
              + (f"  (mean of {len(all_h_corrs_theta)} scans)" if len(all_h_corrs_theta) > 1 else ""))
        if rotate_tiles:
            print(f"  Tile content de-rotated by {-extracted_theta:.4f}°")

    return all_frame_data, all_corr_vectors, extracted_theta

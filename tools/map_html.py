"""OpenSeadragon HTML viewer generation for area-scan maps (plan task L5).

Extracted from tools/make_map.py: write_html — the full viewer page including
the embedded JavaScript navigate math. The JS placement/chip/viewport
formulas in here MUST stay in lockstep with vision/registration.py;
tests/test_map_js_parity.py extracts them from THIS file's source and
evaluates them under node against the Python originals.

Position solving: vision/map_stitch. DZI encoding: vision/dzi.
CLI + overlay data preparation: tools/make_map.
"""
import datetime
import json
import sys as _sys, os as _os
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None   # thumbnails degrade gracefully; viewer HTML still builds

try:
    import numpy as np
    import cv2
    _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False

_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))
from vision.map_stitch import _build_grid                    # noqa: E402
from vision.dzi import TILE_SIZE, TILE_OVERLAP               # noqa: E402
from core.scan_io import all_bounds                          # noqa: E402


_OSD_CDN = ("https://cdn.jsdelivr.net/npm/openseadragon@5.0/build/openseadragon/"
            "openseadragon.min.js")

# Accent colours for the per-mag layer controls in the HTML viewer
_MAG_COLORS = {
    '5x': '#4fc', '10x': '#8f4', '20x': '#fe4', '50x': '#f84', '100x': '#f44',
}


def write_html(stem: str, layers_info: list, scans: list,
               out_dir: Path,
               candidates: list | None = None,
               sample_flakes: list | None = None,
               extents_polygon: list | None = None,
               nav_port: int = 0,
               frame_data: list | None = None,
               chip_transform: dict | None = None,
               sample_folder: str = '',
               corr_vectors: list | None = None,
               rotation_deg: float = 0.0,
               scan_placement: dict | None = None) -> Path:
    """
    Write an OpenSeadragon HTML viewer with per-mag layer opacity/visibility controls,
    flake candidate overlays, and click-to-annotate mode.

    layers_info: list of dicts ordered low-mag → high-mag, each with keys:
      'mag'  — mag string ('10x' etc.)
      'stem' — DZI file stem (no extension, relative to out_dir)
      'w'    — canvas width in pixels
      'h'    — canvas height in pixels
      'fmt'  — tile format ('jpg' or 'png')

    candidates: optional list of flake candidate dicts from flake_candidates.json.
      When provided, each candidate is shown as a coloured dot on the map.
      Dot colour: yellow-green gradient by |contrast_pct| (dimmer = fainter flake).
    """
    x0, y0, x1, y1 = all_bounds(scans)
    global_phys_w = x1 - x0

    # Rotation reference for navigate/marker/dot transforms — must match the
    # centre tile that _rotation_model_positions rotated the grid about, else
    # the pixel↔stage transform won't invert the tile placement (navigate
    # offset that grows toward the map edges).  Single-scan maps: exact;
    # multi-scan: uses the highest-ppm scan's centre.
    rot_ref_x = rot_ref_y = 0.0
    if rotation_deg:
        _rscan = max(scans, key=lambda s: s['ppm'])
        _rgrid = _build_grid(_rscan['images'])
        _cells = sorted(_rgrid)
        _rows = sorted({r for r, c in _cells})
        _cols = sorted({c for r, c in _cells})
        _mr, _mc = _rows[len(_rows) // 2], _cols[len(_cols) // 2]
        _ref = min(_cells, key=lambda rc: abs(rc[0] - _mr) + abs(rc[1] - _mc))
        rot_ref_x, rot_ref_y = _rgrid[_ref]['x_mm'], _rgrid[_ref]['y_mm']

    html_path = out_dir / f"{stem}.html"

    # ── Inline tileSource JS objects (file://-safe, no XHR) ──────────────────
    ts_parts = []
    for li in layers_info:
        ts_parts.append(
            '  {\n'
            '    Image: {\n'
            '      xmlns: "http://schemas.microsoft.com/deepzoom/2008",\n'
            f'      Url: "{li["stem"]}_files/",\n'
            f'      Format: "{li["fmt"]}",\n'
            f'      Overlap: "{TILE_OVERLAP}",\n'
            f'      TileSize: "{TILE_SIZE}",\n'
            f'      Size: {{ Width: "{li["w"]}", Height: "{li["h"]}" }}\n'
            '    }\n'
            '  }'
        )
    tile_sources_js = ',\n'.join(ts_parts)

    # ── Per-layer OSD placement + coordinate info ─────────────────────────────
    lb_parts = []
    for li in layers_info:
        # Embed ORB-corrected tile centres as a compact JS array:
        # [[img_x, img_y, stage_x_mm, stage_y_mm], ...]
        # JS uses nearest-tile + pixel offset for accurate mm coordinates.
        tc = li.get('tile_centres', [])
        tc_js = json.dumps([[round(c[0],1), round(c[1],1), c[2], c[3]] for c in tc],
                           separators=(',', ':'))
        lb_parts.append(
            f'  {{ x:{li["osd_x"]:.6f}, y:{li["osd_y"]:.6f}, '
            f'width:{li["osd_width"]:.6f}, '
            f'x0mm:{li["x0_mm"]:.6f}, y0mm:{li["y0_mm"]:.6f}, '
            f'phys_w:{li["phys_w"]:.6f}, phys_h:{li["phys_h"]:.6f}, '
            f'w:{li["w"]}, h:{li["h"]}, '
            f'tileCentres:{tc_js} }}'
        )
    layer_bounds_js = ',\n'.join(lb_parts)

    # ── Per-mag layer control rows ────────────────────────────────────────────
    if frame_data:
        _seen_layers: list = []   # [(layer_id, mag, label), ...]
        _seen_layer_set: set = set()
        for _fd in frame_data:
            lid = _fd['layer']
            if lid not in _seen_layer_set:
                _seen_layer_set.add(lid)
                _mag = _fd['mag']
                # Strip sample+mag prefix from folder name to get a short timestamp label
                _folder = lid[len(_mag) + 1:]          # e.g. "Graphene_A_20x_20260618_090208"
                _short = _folder.split(_mag + '_', 1)[-1] if f'_{_mag}_' in _folder else _folder
                _label = f'{_mag}  {_short}' if _short != _folder else _mag
                _seen_layers.append((lid, _mag, _label))
        layer_rows = []
        for lid, _mag, _label in _seen_layers:
            color = _MAG_COLORS.get(_mag, '#ddd')
            safe = lid.replace(' ', '_')
            layer_rows.append(
                f'<div class="lrow">'
                f'<span class="mag-label" style="color:{color}">{_label}</span>'
                f'<input type="checkbox" id="vis_{safe}" checked '
                f'onchange="toggleVis(\'{safe}\')">'
                f'<input type="range" class="oslider" id="opac_{safe}" '
                f'min="0" max="100" value="100" '
                f'oninput="setOpac(\'{safe}\',this.value)">'
                f'<span class="oval" id="oval_{safe}">100%</span>'
                f'</div>'
            )
    else:
        layer_rows = []
        for i, li in enumerate(layers_info):
            color = _MAG_COLORS.get(li['mag'], '#ddd')
            layer_rows.append(
                f'<div class="lrow">'
                f'<span class="mag-label" style="color:{color}">{li["mag"]}</span>'
                f'<input type="checkbox" id="vis{i}" checked '
                f'onchange="toggleVis({i})">'
                f'<input type="range" class="oslider" id="opac{i}" '
                f'min="0" max="100" value="100" '
                f'oninput="setOpac({i},this.value)">'
                f'<span class="oval" id="oval{i}">100%</span>'
                f'</div>'
            )
    layer_controls_html = '\n    '.join(layer_rows)

    # ── Summary table rows ────────────────────────────────────────────────────
    seen: set = set()
    table_rows = []
    for s in scans:
        key = (s['folder'].name, s['mag'])
        if key not in seen:
            seen.add(key)
            table_rows.append(
                f'<tr><td>{s["folder"].name}</td>'
                f'<td>{s["mag"]}</td>'
                f'<td>{len(s["images"])}</td></tr>'
            )
    rows_html = '\n    '.join(table_rows)

    # ── Embed flake candidates as JS array ───────────────────────────────────
    _N_COLORS = {1: '#F0E442', 2: '#E69F00', 3: '#CC79A7'}
    if candidates:
        # One small crop thumbnail per candidate for the sidebar list.
        _thumb_dir = out_dir / 'candidate_thumbs'
        _thumb_dir.mkdir(exist_ok=True)
        _img_cache: dict = {}

        def _make_thumb(c):
            cid = c.get('id', '')
            sf, si = c.get('scan_folder', ''), c.get('source_image', '')
            cen, bb = c.get('centroid_px'), c.get('bbox_px')
            if not (cid and sf and si and cen):
                return ''
            rel = f'candidate_thumbs/{cid}.jpg'
            outp = out_dir / rel
            if outp.exists():
                return rel
            try:
                from PIL import ImageDraw
                key = (sf, si)
                im = _img_cache.get(key)
                if im is None:
                    im = Image.open(Path(sf) / si).convert('RGB')
                    _img_cache[key] = im
                cx, cy = float(cen[0]), float(cen[1])
                half = 70.0
                if bb and len(bb) == 4:
                    half = max(half, max(bb[2], bb[3]) * 0.7)
                TS = 96
                thumb = im.crop((int(cx - half), int(cy - half),
                                 int(cx + half), int(cy + half))).resize((TS, TS))
                # Scale bar: ppu (px/µm) from the candidate's own area (mag-agnostic).
                # Unified geometry/style via vision.scale_bar (shared with the
                # catalogue crops + contact sheet).
                area_px, area_um2 = c.get('area_px', 0), c.get('area_um2', 0)
                if area_px and area_um2:
                    from vision import scale_bar as _sb
                    ppu = (area_px / area_um2) ** 0.5
                    thumb_ppu = ppu * (TS / (2.0 * half))      # px/µm in the thumbnail
                    _sb.draw_pil(ImageDraw.Draw(thumb), TS, TS, thumb_ppu)
                # Unobtrusive dotted outline of the EXACT detected region so the
                # measured area is visible (not just a bbox crop).  contour_px is in
                # source-tile px; map into the thumbnail's crop→resize frame.
                contour = c.get('contour_px')
                if contour and len(contour) >= 3:
                    dd = ImageDraw.Draw(thumb)
                    sc = TS / (2.0 * half)
                    pts = [((px - (cx - half)) * sc, (py - (cy - half)) * sc)
                           for px, py in contour]
                    from vision.draw import dash_segments
                    # 1 px-wide dashed line, ~3 on / 2 off
                    for a, b in dash_segments(pts, dash=3.0, gap=2.0):
                        dd.line([a, b], fill=(255, 255, 255), width=1)
                thumb.save(outp, 'JPEG', quality=80)
                return rel
            except Exception:
                return ''

        def _fmt_c(c):
            # v2 emits mean_contrast_bgr as a signed Weber FRACTION (×100 → %);
            # v3 emits contrast_bgr_pct already in PERCENT (no scaling).
            if 'mean_contrast_bgr' in c:
                bgr = [round(v * 100, 1) if v is not None else None
                       for v in c['mean_contrast_bgr']]
            else:
                bgr = [round(v, 1) if v is not None else None
                       for v in c.get('contrast_bgr_pct', [None, None, None])]
            return {
                'id':           c.get('id', ''),
                'scan_folder':  c.get('scan_folder', ''),
                'source_image': c.get('source_image', ''),
                'x_mm':         c['x_mm'],
                'y_mm':         c['y_mm'],
                'n':            c.get('target_N', 0),
                'contrast_pct': round(c.get('contrast_pct', 0), 1),
                'area_um2':     round(c.get('area_um2', 0)),
                'score':        round(c.get('score', 0), 3),
                'solidity':     round(c.get('solidity', 0), 2),
                'bgr':          bgr,
                'thumb':        _make_thumb(c),
            }
        _fmtd = [_fmt_c(c) for c in candidates]   # build once; index = JS array index
        cands_js = json.dumps(_fmtd, separators=(',', ':'))
        n_cands = len(candidates)
        # N-group sidebar rows (visibility checkboxes)
        from collections import Counter as _Counter
        _n_counts = _Counter(fc['n'] for fc in _fmtd)
        _n_rows = []
        for _nv in sorted(_n_counts):
            _label = f'{_nv}L' if _nv else '?L'
            _color = _N_COLORS.get(_nv, '#888')
            _n_rows.append(
                f'<div class="lrow">'
                f'<span class="mag-label" style="color:{_color}">{_label}</span>'
                f'<input type="checkbox" id="showN{_nv}" checked '
                f'onchange="toggleN({_nv})">'
                f'<span style="font-size:10px;color:#888;grid-column:3/5">'
                f'{_n_counts[_nv]}</span>'
                f'</div>'
            )
        n_layer_html = '\n    '.join(_n_rows)
        # Clickable thumbnail list, grouped by layer count.
        _by_n: dict = {}
        for _i, _fc in enumerate(_fmtd):
            _by_n.setdefault(_fc['n'], []).append((_i, _fc))
        _grp_html = []
        for _nv in sorted(_by_n):
            _label = f'{_nv}L' if _nv else '?L'
            _color = _N_COLORS.get(_nv, '#888')
            _thumbs = ''.join(
                (f'<img class="cand-thumb" data-cand-idx="{_i}" src="{_fc["thumb"]}" '
                 f'loading="lazy" onclick="_gotoCand({_i})" '
                 f'title="{_label} · {_fc["area_um2"]} um2 · R{_fc["bgr"][2]}%">')
                for _i, _fc in _by_n[_nv] if _fc['thumb'])
            _grp_html.append(
                f'<div class="cand-grp" id="candGrp{_nv}">'
                f'<div class="cand-grp-hdr" style="color:{_color}">{_label} '
                f'<span style="color:#888">({len(_by_n[_nv])})</span></div>'
                f'<div class="cand-grp-thumbs">{_thumbs}</div></div>')
        cand_thumb_html = ('<div id="candThumbList">' + '\n'.join(_grp_html) + '</div>') if _grp_html else ''
    else:
        cands_js = '[]'
        n_cands = 0
        n_layer_html = ''
        cand_thumb_html = ''

    # Detection stepper + label-export controls (only when candidates exist).
    # Static HTML (no f-string braces); inserted via {detect_tools_html}.
    detect_tools_html = ('''
    <div style="padding:5px 0 4px 0;">
      <div style="display:flex;gap:4px;align-items:center;margin-bottom:4px;">
        <button onclick="stepCand(-1)" title="Previous detection in active layer" style="font-size:12px;padding:1px 8px;">&#9664;</button>
        <span id="candStepInfo" style="font-size:11px;color:#9cf;flex:1;text-align:center;">step –/–</span>
        <button onclick="stepCand(1)" title="Next detection in active layer" style="font-size:12px;padding:1px 8px;">&#9654;</button>
      </div>
      <label style="display:flex;align-items:center;gap:6px;font-size:11px;cursor:pointer;color:#aaa;">
        <input type="checkbox" id="candStepNav"> move stage to each (click/step)
      </label>
      <button onclick="exportLabels()" style="font-size:11px;margin-top:5px;width:100%;cursor:pointer;">
        Export labels (<span id="labelCount">0</span>)</button>
    </div>''' if n_cands else '')

    # ── Embed sample flakes and wafer extents ─────────────────────────────────
    if sample_flakes:
        flakes_js = json.dumps([
            {'id': f['id'], 'name': f.get('name', ''),
             'x_mm': f['x_mm'], 'y_mm': f['y_mm'],
             'chip_x_mm': f.get('chip_x_mm'), 'chip_y_mm': f.get('chip_y_mm'),
             'status': f.get('status', 'Candidate'),
             'layer_count': f.get('layer_count'),
             'area_um2': f.get('area_um2'),
             'magnification': f.get('magnification') or '',
             'cleanliness': f.get('cleanliness') or '',
             'isolation': f.get('isolation') or '',
             'notes': f.get('notes') or '',
             'n_images': len(f.get('images') or []),
             'source': f.get('source') or '',
             'created_at': (f.get('created_at') or '')[:10]}
            for f in sample_flakes
        ], separators=(',', ':'))
        n_flakes = len(sample_flakes)
    else:
        flakes_js = '[]'
        n_flakes = 0

    extents_js = json.dumps(extents_polygon or [], separators=(',', ':'))

    # ── Mode-dependent JavaScript blocks ─────────────────────────────────────
    _OSD_CFG = (
        'var viewer = OpenSeadragon({\n'
        '  id: "viewer",\n'
        '  prefixUrl: "https://cdn.jsdelivr.net/npm/openseadragon@5.0/build/openseadragon/images/",\n'
        '  showNavigator: true,\n'
        '  navigatorPosition: "TOP_RIGHT",\n'
        '  minZoomLevel: 0.1,\n'
        '  maxZoomLevel: 40,\n'
        '  visibilityRatio: 0.3,\n'
        '  zoomPerScroll: 1.4,\n'
        '  animationTime: 0.3,\n'
        '});'
    )
    _CANDS_FLAKES = (
        f'// Detected flake candidates from the last detection run\n'
        f'var detectedCandidates = {cands_js};\n\n'
        f'// Manually-recorded sample flakes (from sample.json)\n'
        f'var sampleFlakes = {flakes_js};\n\n'
        f'// Wafer extents polygon [[x_mm, y_mm], ...]\n'
        f'var waferExtents = {extents_js};'
    )

    chip_tf_js = json.dumps(chip_transform) if chip_transform else 'null'
    corr_vectors_js = json.dumps(corr_vectors or [], separators=(',', ':'))
    # Placement transform active when this map's scan was captured.  Sent to the
    # app on navigate/import so it can compose scan-frame→current-stage; also used
    # browser-side to compose with the live placement for flake-position display.
    scan_placement_js = (json.dumps(scan_placement)
                         if (scan_placement and 'dx_mm' in scan_placement) else 'null')

    if frame_data:
        frames_js = json.dumps(frame_data, separators=(',', ':'))
        viewer_data_js = (
            f'var STEM = {json.dumps(stem)};\n'
            f'var NAV_PORT = {nav_port};\n'
            f'var SCAN_PLACEMENT = {scan_placement_js};\n'
            f'var MAP_SAMPLE_FOLDER = {json.dumps(sample_folder)};\n'
            f'var MAP_BAKED_DATE = {json.dumps(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))};\n'
            f'var GLOBAL_X0 = {x0};\n'
            f'var GLOBAL_Y0 = {y0};\n'
            f'var GLOBAL_PHYS_W = {global_phys_w};\n'
            f'var ROT_DEG = {rotation_deg};\n'
            f'var ROT_REF_X = {rot_ref_x};\n'
            f'var ROT_REF_Y = {rot_ref_y};\n'
            f'var CHIP_TRANSFORM = {chip_tf_js};\n\n'
            f'var frames = {frames_js};\n\n'
            f'// ORB/PC correction vectors [[x_mm, y_mm, dx_mm, dy_mm, method]] method: 0=ORB 1=PC 2=filled\n'
            f'var corrVectors = {corr_vectors_js};\n\n'
            + _CANDS_FLAKES + '\n\n'
            + _OSD_CFG + '\n\n'
            'var layerItems = {};\n'
            'frames.forEach(function(f) {\n'
            '  var lid = f.layer || f.mag;\n'
            '  if (!layerItems[lid]) layerItems[lid] = [];\n'
            '  var ts = {Image:{xmlns:"http://schemas.microsoft.com/deepzoom/2008",\n'
            '    Url:f.dzi.replace(/\\.dzi$/,"_files/"),Format:"jpg",Overlap:"1",TileSize:"256",\n'
            '    Size:{Width:f.fw||2464,Height:f.fh||2056}}};\n'
            '  viewer.addTiledImage({\n'
            '    tileSource: ts, x: f.x, y: f.y, width: f.w, opacity: 1.0,\n'
            '    success: function(ev) { layerItems[lid].push(ev.item); }\n'
            '  });\n'
            '});\n\n'
            'function toggleVis(lid) {\n'
            '  var items = layerItems[lid] || [];\n'
            "  var cb = document.getElementById('vis_' + lid);\n"
            "  var sl = document.getElementById('opac_' + lid);\n"
            '  var opac = (cb && cb.checked) ? parseInt(sl ? sl.value : 100) / 100 : 0;\n'
            '  items.forEach(function(it) { it.setOpacity(opac); });\n'
            '  viewer.forceRedraw();\n'
            '}\n\n'
            'function setOpac(lid, val) {\n'
            "  document.getElementById('oval_' + lid).textContent = val + '%';\n"
            "  var cb = document.getElementById('vis_' + lid);\n"
            '  if (cb && cb.checked) {\n'
            '    (layerItems[lid] || []).forEach(function(it) { it.setOpacity(parseInt(val) / 100); });\n'
            '  }\n'
            '  viewer.forceRedraw();\n'
            '}'
        )
        coord_helpers_js = (
            '// ── Coordinate helpers (per-frame DZI) ─────────────────────────────────────\n'
            '// Tiles are placed at R(ROT_DEG)·(stage-ref) about the centre tile (camera\n'
            '// rotation model). The pixel↔stage transforms must apply the same rotation,\n'
            '// else navigate/markers/dots drift from features (grows toward map edges).\n\n'
            '// stage → pseudo-mm (the rotated frame the tiles live in)\n'
            'function _fwdRot(x, y) {\n'
            '  if (!ROT_DEG) return {x: x, y: y};\n'
            '  var a = ROT_DEG * Math.PI / 180, c = Math.cos(a), s = Math.sin(a);\n'
            '  var dx = x - ROT_REF_X, dy = y - ROT_REF_Y;\n'
            '  return {x: ROT_REF_X + c*dx - s*dy, y: ROT_REF_Y + s*dx + c*dy};\n'
            '}\n'
            '// pseudo-mm → stage  (inverse rotation R(-ROT_DEG))\n'
            'function _invRot(x, y) {\n'
            '  if (!ROT_DEG) return {x: x, y: y};\n'
            '  var a = ROT_DEG * Math.PI / 180, c = Math.cos(a), s = Math.sin(a);\n'
            '  var dx = x - ROT_REF_X, dy = y - ROT_REF_Y;\n'
            '  return {x: ROT_REF_X + c*dx + s*dy, y: ROT_REF_Y - s*dx + c*dy};\n'
            '}\n\n'
            'function _vpFromMm(xMm, yMm) {\n'
            '  var p = _fwdRot(xMm, yMm);\n'
            '  return new OpenSeadragon.Point(\n'
            '    (p.x - GLOBAL_X0) / GLOBAL_PHYS_W,\n'
            '    (p.y - GLOBAL_Y0) / GLOBAL_PHYS_W\n'
            '  );\n'
            '}\n\n'
            'function _mmFromVp(vp) {\n'
            '  var px = GLOBAL_X0 + vp.x * GLOBAL_PHYS_W;\n'
            '  var py = GLOBAL_Y0 + vp.y * GLOBAL_PHYS_W;\n'
            '  return _invRot(px, py);\n'
            '}\n\n'
            'function _chipFromMm(mm) {\n'
            '  if (!CHIP_TRANSFORM) return null;\n'
            '  var t = CHIP_TRANSFORM;\n'
            '  var ox = t.origin_mm[0], oy = t.origin_mm[1];\n'
            '  var xx = t.x_axis[0],   xy = t.x_axis[1];\n'
            '  var yx = t.y_axis[0],   yy = t.y_axis[1];\n'
            '  var dx = mm.x - ox, dy = mm.y - oy;\n'
            '  var det = xx * yy - yx * xy;\n'
            '  return {x: (yy * dx - yx * dy) / det, y: (xx * dy - xy * dx) / det};\n'
            '}'
        )
    else:
        viewer_data_js = (
            f'var STEM = {json.dumps(stem)};\n'
            f'var NAV_PORT = {nav_port};\n'
            f'var SCAN_PLACEMENT = {scan_placement_js};\n'
            f'var MAP_SAMPLE_FOLDER = {json.dumps(sample_folder)};\n'
            f'var MAP_BAKED_DATE = {json.dumps(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))};\n'
            f'var CHIP_TRANSFORM = {chip_tf_js};\n'
            f'var corrVectors = {corr_vectors_js};\n'
            f'var tileSources = [\n{tile_sources_js}\n];\n\n'
            '// Per-layer physical + OSD placement info\n'
            f'var layerBounds = [\n{layer_bounds_js}\n];\n\n'
            + _CANDS_FLAKES + '\n\n'
            + _OSD_CFG + '\n\n'
            'tileSources.forEach(function(ts, i) {\n'
            '  var lb = layerBounds[i];\n'
            '  viewer.addTiledImage({\n'
            '    tileSource: ts,\n'
            '    x: lb.x, y: lb.y, width: lb.width,\n'
            '    opacity: 1.0,\n'
            '  });\n'
            '});\n\n'
            'function _item(i) { return viewer.world.getItemAt(i); }\n\n'
            'function toggleVis(i) {\n'
            '  var item = _item(i);\n'
            '  if (!item) return;\n'
            "  var opac = parseInt(document.getElementById('opac' + i).value) / 100;\n"
            "  item.setOpacity(document.getElementById('vis' + i).checked ? opac : 0);\n"
            '  viewer.forceRedraw();\n'
            '}\n\n'
            'function setOpac(i, val) {\n'
            "  document.getElementById('oval' + i).textContent = val + '%';\n"
            '  var item = _item(i);\n'
            '  if (!item) return;\n'
            "  if (document.getElementById('vis' + i).checked)\n"
            '    item.setOpacity(parseInt(val) / 100);\n'
            '  viewer.forceRedraw();\n'
            '}'
        )
        coord_helpers_js = (
            '// ── Coordinate helpers ────────────────────────────────────────────────────────\n\n'
            'function _vpFromMm(xMm, yMm) {\n'
            '  var item0 = _item(0);\n'
            '  if (!item0) return null;\n'
            '  var lb0 = layerBounds[0];\n'
            '  // Use nearest tile centre + offset for accurate inverse mapping.\n'
            '  var tc = lb0.tileCentres;\n'
            '  var ppm = lb0.w / lb0.phys_w;  // px per mm\n'
            '  if (tc && tc.length) {\n'
            '    var best = tc[0], bestD = 1e18;\n'
            '    for (var i = 0; i < tc.length; i++) {\n'
            '      var dx = tc[i][2] - xMm, dy = tc[i][3] - yMm;\n'
            '      var d = dx*dx + dy*dy;\n'
            '      if (d < bestD) { bestD = d; best = tc[i]; }\n'
            '    }\n'
            '    var imgX = best[0] + (xMm - best[2]) * ppm;\n'
            '    var imgY = best[1] + (yMm - best[3]) * ppm;\n'
            '    return item0.imageToViewportCoordinates(new OpenSeadragon.Point(imgX, imgY));\n'
            '  }\n'
            '  // Fallback: linear (no ORB correction data)\n'
            '  var imgX = (xMm - lb0.x0mm) * ppm;\n'
            '  var imgY = (yMm - lb0.y0mm) * ppm;\n'
            '  return item0.imageToViewportCoordinates(new OpenSeadragon.Point(imgX, imgY));\n'
            '}\n\n'
            'function _mmFromVp(vp) {\n'
            '  var item0 = _item(0);\n'
            '  if (!item0) return null;\n'
            '  var img = item0.viewportToImageCoordinates(vp);\n'
            '  var lb0 = layerBounds[0];\n'
            '  var ppm = lb0.w / lb0.phys_w;\n'
            '  // Use nearest tile centre + pixel offset — corrects for ORB tile alignment.\n'
            '  var tc = lb0.tileCentres;\n'
            '  if (tc && tc.length) {\n'
            '    var best = tc[0], bestD = 1e18;\n'
            '    for (var i = 0; i < tc.length; i++) {\n'
            '      var dx = img.x - tc[i][0], dy = img.y - tc[i][1];\n'
            '      var d = dx*dx + dy*dy;\n'
            '      if (d < bestD) { bestD = d; best = tc[i]; }\n'
            '    }\n'
            '    return {\n'
            '      x: best[2] + (img.x - best[0]) / ppm,\n'
            '      y: best[3] + (img.y - best[1]) / ppm,\n'
            '    };\n'
            '  }\n'
            '  // Fallback: linear (no ORB correction data)\n'
            '  return {\n'
            '    x: lb0.x0mm + img.x / ppm,\n'
            '    y: lb0.y0mm + img.y / ppm,\n'
            '  };\n'
            '}\n\n'
            'function _chipFromMm(mm) {\n'
            '  if (!CHIP_TRANSFORM) return null;\n'
            '  var t = CHIP_TRANSFORM;\n'
            '  var ox = t.origin_mm[0], oy = t.origin_mm[1];\n'
            '  var xx = t.x_axis[0],   xy = t.x_axis[1];\n'
            '  var yx = t.y_axis[0],   yy = t.y_axis[1];\n'
            '  var dx = mm.x - ox, dy = mm.y - oy;\n'
            '  var det = xx * yy - yx * xy;\n'
            '  return {x: (yy * dx - yx * dy) / det, y: (xx * dy - xy * dx) / det};\n'
            '}'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{stem}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{ margin:0; background:#111; color:#ddd; font-family:sans-serif; display:flex; flex-direction:column; height:100vh; }}
  #toolbar {{
    flex:0 0 auto; background:#222; display:flex; align-items:center; flex-wrap:wrap;
    padding:0 12px; gap:12px; font-size:13px; z-index:100; min-height:36px;
  }}
  #toolbar strong {{ color:#fff; }}
  #coords {{ font-family:monospace; font-size:12px; color:#aef; }}
  #main {{ display:flex; flex:1; overflow:hidden; }}
  #sidebar {{
    flex:0 0 190px; background:#1a1a1a; border-right:1px solid #333;
    display:flex; flex-direction:column; overflow-y:auto; padding:8px 0;
  }}
  #sidebar h3 {{ margin:0 8px 6px; font-size:11px; color:#888; text-transform:uppercase; letter-spacing:.06em; }}
  .lrow {{
    display:grid; grid-template-columns:36px 18px 1fr 32px;
    align-items:center; gap:4px; padding:4px 8px;
    border-bottom:1px solid #222;
  }}
  .mag-label {{ font-size:13px; font-weight:bold; }}
  .oslider {{ width:100%; accent-color:#4a4; }}
  .oval {{ font-size:10px; color:#888; text-align:right; }}
  #info {{
    padding:8px; font-size:11px; border-top:1px solid #333;
  }}
  #info b {{ color:#ccc; }}
  #info table {{ border-collapse:collapse; width:100%; margin-top:4px; }}
  #info td {{ padding:2px 4px; }}
  #info tr:nth-child(even) {{ background:rgba(255,255,255,.05); }}
  #viewer {{ flex:1; position:relative; }}

  /* Annotation controls */
  .ann-btn {{
    padding:3px 8px; border:1px solid #555; border-radius:3px; background:#333;
    color:#ddd; cursor:pointer; font-size:12px;
  }}
  .ann-btn:hover {{ background:#444; }}
  .ann-btn.active {{ background:#264; border-color:#4a4; color:#8f8; }}

  /* Candidate overlay dots */
  .cand-dot {{
    width:10px; height:10px; border-radius:50%;
    border:1.5px solid rgba(0,0,0,0.6);
    transform:translate(-50%,-50%);
    pointer-events:auto; cursor:default;
    position:absolute;
  }}
  /* OSD manages style.display for off-screen culling — use visibility instead */
  .cand-hidden {{ visibility:hidden !important; pointer-events:none !important; }}
  .cand-dot {{ cursor:pointer; }}
  .cand-dot[data-labeled="1L"], .cand-dot[data-labeled="2L"],
  .cand-dot[data-labeled="3L"] {{ border-color:#fff; box-shadow:0 0 0 2px rgba(255,255,255,0.55); }}
  .cand-dot[data-labeled="junk"] {{ border-color:#f55; opacity:0.45; }}
  .cand-dot.cand-current {{ box-shadow:0 0 0 3px #ff0, 0 0 9px 3px rgba(255,255,0,0.6); z-index:50; }}
  #candThumbList {{ max-height:46vh; overflow-y:auto; margin:4px 0; }}
  .cand-grp {{ margin-bottom:6px; }}
  .cand-grp-hdr {{ font-size:11px; font-weight:bold; margin:3px 0 2px; position:sticky; top:0;
                   background:#1b1b1b; padding:2px 0; z-index:1; }}
  .cand-grp-thumbs {{ display:flex; flex-wrap:wrap; gap:3px; }}
  .cand-thumb {{ width:46px; height:46px; object-fit:cover; border:2px solid #444;
                 border-radius:3px; cursor:pointer; }}
  .cand-thumb:hover {{ border-color:#9cf; }}
  .cand-thumb.thumb-current {{ border-color:#ff0; box-shadow:0 0 5px 1px rgba(255,255,0,0.6); }}
  #candPopup {{
    display:none; position:fixed; background:#252525; border:1px solid #555;
    padding:10px; border-radius:6px; z-index:2000; min-width:210px;
    box-shadow:0 4px 16px rgba(0,0,0,0.7);
  }}
  #candPopup .cl-btn {{ font-size:12px; padding:4px 10px; margin-right:4px; cursor:pointer; }}
  #candPopup .cl-btn.sel {{ background:#4af; color:#000; border-color:#4af; }}

  /* Unified flake markers (catalogue + user-placed) */
  .flake-dot {{
    width:13px; height:13px; border-radius:50%;
    border:2px solid rgba(0,0,0,0.7);
    transform:translate(-50%,-50%);
    pointer-events:auto; cursor:default;
    position:absolute;
    transition:background 0.15s;
  }}
  .flake-dot[data-source="marker"],
  .flake-dot[data-source="catalogue"] {{ cursor:pointer; }}
  .flake-dot[data-source="marker"]:hover {{ opacity:0.75; }}
  .ann-marker-label {{
    position:absolute; bottom:14px; left:50%; transform:translateX(-50%);
    font-size:9px; font-weight:bold; white-space:nowrap;
    pointer-events:none; text-shadow:0 1px 3px #000;
  }}

  /* Marker metadata popup */
  #markerPopup {{
    display:none; position:fixed; background:#252525; border:1px solid #555;
    padding:12px; border-radius:6px; z-index:2000; min-width:240px; max-width:320px;
    box-shadow:0 4px 16px rgba(0,0,0,0.7);
  }}
  #flakeMeta {{
    display:none; grid-template-columns:auto 1fr; gap:3px 10px;
    font-size:11px; border-top:1px solid #383838; margin-bottom:10px; padding-top:8px;
    align-items:center;
  }}
  #flakeMeta .mk {{ color:#666; white-space:nowrap; }}
  #flakeMeta .mv {{ color:#ccc; word-break:break-word; }}
  #flakeMeta .mi {{
    background:#1a1a1a; border:1px solid #3a3a3a; color:#ddd;
    padding:2px 5px; border-radius:2px; font-size:11px; font-family:inherit;
    width:100%; box-sizing:border-box; outline:none;
  }}
  #flakeMeta .mi:focus {{ border-color:#666; }}
  #flakeMeta .mi-wrap {{ display:flex; align-items:center; gap:4px; }}
  #flakeMeta .mi-wrap .mi {{ flex:1; min-width:0; }}
  .layer-pick-btn {{
    padding:4px 10px; border-radius:4px; border:1.5px solid #555;
    cursor:pointer; font-size:13px; font-weight:bold;
    background:#1a1a1a; color:#aaa;
  }}
  .layer-pick-btn:hover {{ border-color:#aaa; }}
  .layer-pick-btn.sel {{ border-color:#fff; background:#3a3a3a; }}
  .layer-pick-btn[data-l="1"] {{ color:#4c4; }}
  .layer-pick-btn[data-l="2"] {{ color:#f84; }}
  .layer-pick-btn[data-l="3"] {{ color:#a4f; }}
  .layer-pick-btn[data-l="0"] {{ color:#888; }}

  .layer-filter-btn {{
    background:#1a1a1a; border:1px solid #333; color:#555;
    padding:2px 7px; border-radius:3px; font-size:11px;
    cursor:pointer; transition:border-color .12s, opacity .12s;
  }}
  .layer-filter-btn.on  {{ opacity:1; border-color:#555; }}
  .layer-filter-btn.off {{ opacity:0.28; border-color:#2a2a2a; }}
  .layer-filter-btn[data-lf="1"].on  {{ color:#4c4; }}
  .layer-filter-btn[data-lf="2"].on  {{ color:#f84; }}
  .layer-filter-btn[data-lf="3"].on  {{ color:#a4f; }}
  .layer-filter-btn[data-lf="0"].on  {{ color:#888; }}
  .layer-filter-btn[data-lf=""].on   {{ color:#ccc; }}

  /* Wafer extents canvas — overlaid on viewer */
  #extentsCanvas {{
    position:absolute; top:0; left:0;
    pointer-events:none; z-index:10;
  }}
</style>
</head>
<body>
<div id="toolbar">
  <strong>{stem}</strong>
  <span id="coords">hover for coordinates</span>
  <span style="color:#555">|</span>
  <span class="ann-btn" style="cursor:default;opacity:0.6" title="Ctrl+click to place a marker  ·  Alt+click a dot to remove it">✚ Ctrl+click: mark</span>
  <button class="ann-btn" id="navBtn"{' disabled title="Navigate: app not running"' if not nav_port else ' title="Shift+click to move stage here"'} style="cursor:{'default' if nav_port else 'not-allowed'}">⊕ Shift+click: navigate</button>
  <button class="ann-btn" onclick="downloadMarkers()" title="Save markers as JSON">↓ Save JSON</button>
  <button class="ann-btn" onclick="loadMarkersFromFile()" title="Load markers from JSON{'; also imports to sample catalogue' if nav_port else ''}">↑ Load JSON</button>
  <span id="markCount" style="font-size:12px;color:#8f8"></span>
</div>
<div id="markerPopup">
  <div id="markerPopupTitle" style="font-size:12px;color:#ddd;font-weight:bold;margin-bottom:6px;">Layer count</div>
  <div id="flakeMeta"></div>
  <div id="layerPickLabel" style="font-size:10px;color:#777;margin-bottom:6px;">Layer count</div>
  <div style="display:flex;gap:5px;margin-bottom:10px;">
    <button class="layer-pick-btn" data-l="1" onclick="_pickLayer(1)">1L</button>
    <button class="layer-pick-btn" data-l="2" onclick="_pickLayer(2)">2L</button>
    <button class="layer-pick-btn" data-l="3" onclick="_pickLayer(3)">3L</button>
    <button class="layer-pick-btn" data-l="0" onclick="_pickLayer(0)">?L</button>
  </div>
  <input id="markerNoteInput" type="text" placeholder="Note (optional)"
    style="width:100%;box-sizing:border-box;background:#1a1a1a;border:1px solid #444;
           color:#ddd;padding:5px;border-radius:3px;font-size:12px;outline:none;">
  <div style="display:flex;gap:5px;margin-top:10px;justify-content:flex-end;">
    <button id="markerPopupConfirm" onclick="_confirmMarker()" style="font-size:11px;padding:3px 9px;">Place</button>
    <button onclick="_cancelMarkerPopup()" style="font-size:11px;padding:3px 9px;border-color:#666;">Cancel</button>
  </div>
</div>
<div id="candPopup">
  <div id="candPopupTitle" style="font-size:11px;color:#ddd;font-weight:bold;margin-bottom:7px;"></div>
  <div style="display:flex;gap:4px;margin-bottom:8px;">
    <button class="cl-btn" data-cl="1L" onclick="labelCand('1L')">1L</button>
    <button class="cl-btn" data-cl="2L" onclick="labelCand('2L')">2L</button>
    <button class="cl-btn" data-cl="3L" onclick="labelCand('3L')">3L</button>
    <button class="cl-btn" data-cl="junk" onclick="labelCand('junk')" style="border-color:#a55;">junk</button>
  </div>
  <div style="display:flex;gap:5px;justify-content:flex-end;">
    <button onclick="_clearCandLabel()" style="font-size:10px;padding:2px 7px;border-color:#666;">clear</button>
    <button onclick="_cancelCandPopup()" style="font-size:10px;padding:2px 7px;border-color:#666;">close</button>
  </div>
</div>
<div id="main">
  <div id="sidebar">
    <h3>Layers</h3>
    {layer_controls_html}
    <h3>Overlays</h3>
    <div style="padding:2px 0 6px 0;">
      <label style="display:flex;align-items:center;gap:6px;font-size:12px;cursor:pointer;padding:3px 0;">
        <input type="checkbox" id="showFlakes" checked onchange="toggleFlakes()">
        Flakes (<span id="flakeCountSidebar">{n_flakes}</span>)
      </label>
      <div id="layerFilterBar" style="padding:2px 0 4px 18px;display:flex;gap:4px;flex-wrap:wrap;">
        <button class="layer-filter-btn on" data-lf="1" onclick="_toggleLayerFilter('1')" title="Toggle 1-layer flakes">1L</button>
        <button class="layer-filter-btn on" data-lf="2" onclick="_toggleLayerFilter('2')" title="Toggle 2-layer flakes">2L</button>
        <button class="layer-filter-btn on" data-lf="3" onclick="_toggleLayerFilter('3')" title="Toggle 3-layer flakes">3L</button>
        <button class="layer-filter-btn on" data-lf="0" onclick="_toggleLayerFilter('0')" title="Toggle unknown-layer flakes">?L</button>
        <button class="layer-filter-btn on" data-lf=""  onclick="_toggleLayerFilter('')"  title="Toggle unclassified flakes">—</button>
      </div>
      <label style="display:flex;align-items:center;gap:6px;font-size:12px;cursor:pointer;padding:3px 0;">
        <input type="checkbox" id="showExtents" checked onchange="drawExtents()">
        Chip extents
      </label>
      <label style="display:flex;align-items:center;gap:6px;font-size:12px;cursor:pointer;padding:3px 0;">
        <input type="checkbox" id="showCorrs" onchange="drawCorrs()">
        Tile corrections
        <span style="font-size:10px;color:#888">(ORB/PC shift vectors)</span>
      </label>
      <div id="corrLegend" style="padding:1px 0 4px 18px;display:none;font-size:10px;color:#aaa;line-height:1.6;">
        <span style="color:#4af">&#x25A0;</span> ORB &nbsp;
        <span style="color:#4f8">&#x25A0;</span> PC &nbsp;
        <span style="color:#888">&#x25A0;</span> filled
      </div>
    </div>
    {'<h3>Detections</h3>' + chr(10) + '    ' + n_layer_html if n_layer_html else ''}
    {detect_tools_html}
    {cand_thumb_html}
    <div id="info">
      <b>Scans</b>
      <table>
        <tr><th>Folder</th><th>Mag</th><th>Imgs</th></tr>
        {rows_html}
      </table>
      <div style="margin-top:6px;color:#666;font-size:10px">
        X [{x0:.3f}, {x1:.3f}] mm<br>
        Y [{y0:.3f}, {y1:.3f}] mm
      </div>
    </div>
  </div>
  <div id="viewer"><canvas id="extentsCanvas"></canvas><canvas id="corrsCanvas" style="position:absolute;top:0;left:0;pointer-events:none;z-index:10;"></canvas></div>
</div>
<script src="{_OSD_CDN}"></script>
<script>
{viewer_data_js}

{coord_helpers_js}

// ── Coordinate readout ────────────────────────────────────────────────────────

new OpenSeadragon.MouseTracker({{
  element: viewer.element,
  moveHandler: function(e) {{
    var mm = _mmFromVp(viewer.viewport.pointFromPixel(e.position));
    if (!mm) return;
    var chip = _chipFromMm(mm);
    var txt = chip
      ? 'Chip X ' + chip.x.toFixed(3) + '  Y ' + chip.y.toFixed(3) + ' mm'
        + '   Stage X ' + mm.x.toFixed(3) + '  Y ' + mm.y.toFixed(3) + ' mm'
      : 'Stage X ' + mm.x.toFixed(3) + '  Y ' + mm.y.toFixed(3) + ' mm';
    document.getElementById('coords').textContent = txt;
  }}
}}).setTracking(true);

// ── Candidate dot overlays ────────────────────────────────────────────────────

var candEls = [];
var _N_COLOR = {{1:'#F0E442', 2:'#E69F00', 3:'#CC79A7'}};

function _candColor(c) {{
  return _N_COLOR[c.n] || '#888';
}}

function _candTitle(c) {{
  var nLabel = c.n ? c.n + 'L' : '?L';
  var lines = [
    nLabel + '  score ' + c.score.toFixed(3) + '  sol ' + c.solidity.toFixed(2),
    c.contrast_pct.toFixed(1) + '% Δlum  ' + c.area_um2 + ' µm²',
  ];
  if (c.bgr && c.bgr[0] !== null) {{
    lines.push('B ' + c.bgr[0].toFixed(1) + '%  G ' + c.bgr[1].toFixed(1) +
               '%  R ' + c.bgr[2].toFixed(1) + '%');
  }}
  return lines.join('\\n');
}}

function _nVisible(n) {{
  var cb = document.getElementById('showN' + n);
  return cb ? cb.checked : true;
}}

function toggleN(n) {{
  var show = _nVisible(n);
  candEls.forEach(function(el) {{
    if (parseInt(el.dataset.n || 0) === n)
      el.classList.toggle('cand-hidden', !show);
  }});
  var grp = document.getElementById('candGrp' + n);   // hide the thumbnail group too
  if (grp) grp.style.display = show ? '' : 'none';
  if (typeof _updateCandStepInfo === 'function') _updateCandStepInfo();
}}

function _placeCandidates() {{
  candEls.forEach(function(el) {{ viewer.removeOverlay(el); }});
  candEls = [];
  detectedCandidates.forEach(function(c, i) {{
    var vp = _vpFromMm(c.x_mm, c.y_mm);
    if (!vp) return;
    var el = document.createElement('div');
    el.className = 'cand-dot';
    el.dataset.n = String(c.n || 0);
    el.dataset.candIdx = String(i);
    el.style.background = _candColor(c);
    if (!_nVisible(c.n || 0)) el.classList.add('cand-hidden');
    el.title = _candTitle(c);
    var lab = candLabels[c.id];
    if (lab) el.dataset.labeled = lab;
    viewer.addOverlay({{ element: el, location: vp, placement: 'CENTER' }});
    candEls.push(el);
  }});
}}

function toggleCandidates() {{ _placeCandidates(); }}

// ── Candidate labelling (flake/junk → precision ground truth) ─────────────────
var candLabels = {{}};                       // {{candidateId: '1L'|'2L'|'3L'|'junk'}}
var _CANDLABEL_KEY = 'candLabels_' + STEM;
var _pendingCandIdx = -1;

try {{ candLabels = JSON.parse(localStorage.getItem(_CANDLABEL_KEY) || '{{}}'); }} catch(e) {{ candLabels = {{}}; }}

function _candElByIdx(i) {{
  for (var k = 0; k < candEls.length; k++)
    if (parseInt(candEls[k].dataset.candIdx) === i) return candEls[k];
  return null;
}}

function _updateLabelCount() {{
  var n = Object.keys(candLabels).length;
  var el = document.getElementById('labelCount');
  if (el) el.textContent = n;
}}

function _hitCandDot(clientX, clientY) {{
  var dots = viewer.element.querySelectorAll('.cand-dot:not(.cand-hidden)');
  for (var i = 0; i < dots.length; i++) {{
    var r = dots[i].getBoundingClientRect();
    if (clientX >= r.left && clientX <= r.right &&
        clientY >= r.top  && clientY <= r.bottom) return dots[i];
  }}
  return null;
}}

function _showCandPopup(idx, clientX, clientY) {{
  _pendingCandIdx = idx;
  var c = detectedCandidates[idx];
  var bgr = (c.bgr && c.bgr[0] !== null)
    ? ('  B' + c.bgr[0] + ' G' + c.bgr[1] + ' R' + c.bgr[2] + '%') : '';
  document.getElementById('candPopupTitle').innerHTML =
    (c.n ? c.n + 'L?' : '?L') + '  score ' + c.score.toFixed(2) +
    '  ' + c.area_um2 + 'µm²' + bgr;
  var cur = candLabels[c.id] || '';
  document.querySelectorAll('#candPopup .cl-btn').forEach(function(b) {{
    b.classList.toggle('sel', b.dataset.cl === cur);
  }});
  var pop = document.getElementById('candPopup');
  pop.style.left = Math.min(clientX + 10, window.innerWidth  - pop.offsetWidth  - 10) + 'px';
  pop.style.top  = Math.min(clientY + 10, window.innerHeight - pop.offsetHeight - 10) + 'px';
  pop.style.display = 'block';
}}

function _cancelCandPopup() {{
  document.getElementById('candPopup').style.display = 'none';
  _pendingCandIdx = -1;
}}

function labelCand(label) {{
  if (_pendingCandIdx < 0) return;
  var c = detectedCandidates[_pendingCandIdx];
  candLabels[c.id] = label;
  try {{ localStorage.setItem(_CANDLABEL_KEY, JSON.stringify(candLabels)); }} catch(e) {{}}
  var el = _candElByIdx(_pendingCandIdx);
  if (el) el.dataset.labeled = label;
  _updateLabelCount();
  if (NAV_PORT) {{
    fetch('http://127.0.0.1:' + NAV_PORT + '/label_candidate', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{scan_folder: c.scan_folder, id: c.id, label: label,
        x_mm: c.x_mm, y_mm: c.y_mm, source_image: c.source_image, target_N: c.n}})
    }}).catch(function(e) {{ console.log('label_candidate failed:', e); }});
  }}
  _cancelCandPopup();
}}

function _clearCandLabel() {{
  if (_pendingCandIdx < 0) return;
  var c = detectedCandidates[_pendingCandIdx];
  delete candLabels[c.id];
  try {{ localStorage.setItem(_CANDLABEL_KEY, JSON.stringify(candLabels)); }} catch(e) {{}}
  var el = _candElByIdx(_pendingCandIdx);
  if (el) delete el.dataset.labeled;
  _updateLabelCount();
  if (NAV_PORT) {{
    fetch('http://127.0.0.1:' + NAV_PORT + '/label_candidate', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{scan_folder: c.scan_folder, id: c.id, label: 'unsure',
        x_mm: c.x_mm, y_mm: c.y_mm, source_image: c.source_image, target_N: c.n}})
    }}).catch(function(e) {{}});
  }}
  _cancelCandPopup();
}}

function exportLabels() {{
  var labels = {{}};
  detectedCandidates.forEach(function(c) {{
    if (candLabels[c.id]) labels[c.id] = {{
      label: candLabels[c.id], x_mm: c.x_mm, y_mm: c.y_mm,
      source_image: c.source_image, target_N: c.n}};
  }});
  var sf = (detectedCandidates[0] && detectedCandidates[0].scan_folder) || '';
  var out = {{scan_folder: sf, candidates_file: 'flake_candidates_v3.json', labels: labels}};
  var blob = new Blob([JSON.stringify(out, null, 2)], {{type: 'application/json'}});
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'flake_labels.json';
  a.click();
  URL.revokeObjectURL(a.href);
}}

// ── Detection stepper (walk visible candidates in the active layer) ───────────
var _candStepIdx = -1;

function _visibleCandIdxs() {{
  var out = [];
  detectedCandidates.forEach(function(c, i) {{ if (_nVisible(c.n || 0)) out.push(i); }});
  return out;
}}

function _updateCandStepInfo() {{
  var vis = _visibleCandIdxs();
  var pos = vis.indexOf(_candStepIdx);
  var el = document.getElementById('candStepInfo');
  if (el) el.textContent = 'step ' + (pos >= 0 ? (pos + 1) : '–') + '/' + vis.length;
}}

// Centre map (and optionally stage) on a candidate by index. Shared by the
// step buttons and the sidebar thumbnail clicks.
function _goToCandidate(idx) {{
  var c = detectedCandidates[idx];
  if (!c) return;
  _candStepIdx = idx;
  candEls.forEach(function(el) {{ el.classList.remove('cand-current'); }});
  var el = _candElByIdx(idx);
  var vp = _vpFromMm(c.x_mm, c.y_mm);
  if (vp) viewer.viewport.panTo(vp, false);
  if (el) el.classList.add('cand-current');
  // highlight the matching thumbnail and scroll it into view
  var prev = document.querySelector('.cand-thumb.thumb-current');
  if (prev) prev.classList.remove('thumb-current');
  var th = document.querySelector('.cand-thumb[data-cand-idx="' + idx + '"]');
  if (th) {{ th.classList.add('thumb-current');
            th.scrollIntoView({{block: 'nearest'}}); }}
  _updateCandStepInfo();
  var nav = document.getElementById('candStepNav');
  if (nav && nav.checked && NAV_PORT) _navigateTo(c.x_mm, c.y_mm);
  // open the label popup near screen centre so you can label as you go
  var rect = viewer.element.getBoundingClientRect();
  _showCandPopup(idx, rect.left + rect.width / 2, rect.top + 60);
}}

// Sidebar thumbnail click handler.
function _gotoCand(idx) {{ _goToCandidate(idx); }}

function stepCand(dir) {{
  var vis = _visibleCandIdxs();
  if (!vis.length) return;
  var pos = vis.indexOf(_candStepIdx);
  pos = (pos < 0) ? (dir > 0 ? 0 : vis.length - 1) : (pos + dir);
  pos = ((pos % vis.length) + vis.length) % vis.length;   // wrap
  _goToCandidate(vis[pos]);
}}

// Plain click on a candidate dot → label popup (cand-dots are disjoint from
// flake-dots, so this coexists with the marker/catalogue click handlers).
viewer.element.addEventListener('click', function(e) {{
  if (e.ctrlKey || e.shiftKey || e.altKey) return;
  var hit = _hitCandDot(e.clientX, e.clientY);
  if (hit && hit.dataset.candIdx !== undefined) {{
    _showCandPopup(parseInt(hit.dataset.candIdx), e.clientX, e.clientY);
    e.stopPropagation();
  }}
}});
viewer.addHandler('animation-start', _cancelCandPopup);

// ── Unified flake markers (catalogue + user-placed) ───────────────────────────

var allFlakeEls = [];   // {{x_mm, y_mm, note, layer, source, el}}
var _L_COLOR = {{1:'#F0E442', 2:'#E69F00', 3:'#CC79A7', 0:'#888'}};
var _STATUS_COLOR = {{'Candidate':'#888','Approved':'#4af','In Use':'#fa4','Rejected':'#f44'}};

function _addFlake(xMm, yMm, opts) {{
  var vp = _vpFromMm(xMm, yMm);
  if (!vp) return;
  opts = opts || {{}};
  var source  = opts.source || 'marker';
  var layer   = (opts.layer !== undefined && opts.layer !== null) ? parseInt(opts.layer) : null;
  var lc      = (layer !== null) ? layer : opts.layer_count;
  var color   = (lc != null ? _L_COLOR[lc] : null) || (source === 'catalogue' ? (_STATUS_COLOR[opts.status] || '#888') : '#ff0');
  var el = document.createElement('div');
  el.className = 'flake-dot';
  el.dataset.source = source;
  el.dataset.layer  = (lc !== null && lc !== undefined) ? String(lc) : '';
  if (opts.id) el.dataset.flakeId = opts.id;
  el.style.background = color;
  // Label
  var labelText = null;
  if (source === 'catalogue') {{
    labelText = opts.id + (opts.name ? ' ' + opts.name : '');
  }} else if (lc !== null) {{
    labelText = lc > 0 ? lc + 'L' : '?L';
  }}
  if (labelText) {{
    var lbl = document.createElement('span');
    lbl.className = 'ann-marker-label';
    lbl.style.color = color;
    lbl.textContent = labelText;
    el.appendChild(lbl);
  }}
  // Tooltip
  var tip = [];
  if (opts.id) tip.push(opts.id + (opts.name ? ': ' + opts.name : ''));
  tip.push(xMm.toFixed(4) + ', ' + yMm.toFixed(4) + ' mm');
  if (lc !== null) tip.push(lc > 0 ? lc + 'L' : '?L');
  if (opts.note) tip.push(opts.note);
  if (opts.status && source === 'catalogue') tip.push(opts.status);
  if (opts.source_tag) tip.push('[' + opts.source_tag + ']');
  if (opts.area_um2) tip.push(Math.round(opts.area_um2) + ' µm²');
  if (source === 'marker' || source === 'catalogue') tip.push('Alt+click to remove');
  el.title = tip.join('  ');
  var show = document.getElementById('showFlakes');
  if ((show && !show.checked) || _hiddenLayers.has(el.dataset.layer))
    el.style.visibility = 'hidden';
  viewer.addOverlay({{ element: el, location: vp, placement: 'CENTER' }});
  allFlakeEls.push({{ x_mm: parseFloat(xMm.toFixed(4)), y_mm: parseFloat(yMm.toFixed(4)),
                      note: opts.note || '', layer: layer, source: source, el: el }});
  _updateFlakeCount();
}}

function _placeFlakes() {{
  allFlakeEls.forEach(function(f) {{ viewer.removeOverlay(f.el); }});
  allFlakeEls = [];
  sampleFlakes.forEach(function(f) {{
    _addFlake(f.x_mm, f.y_mm, {{
      source: 'catalogue', id: f.id, name: f.name,
      layer: f.layer_count, layer_count: f.layer_count,
      note: f.notes || '',
      status: f.status, area_um2: f.area_um2,
      source_tag: f.source === 'map' ? 'map' : null
    }});
  }});
}}

var _hiddenLayers = new Set();

function _toggleLayerFilter(lf) {{
  if (_hiddenLayers.has(lf)) {{
    _hiddenLayers.delete(lf);
  }} else {{
    _hiddenLayers.add(lf);
  }}
  document.querySelectorAll('.layer-filter-btn[data-lf="' + lf + '"]').forEach(function(b) {{
    b.classList.toggle('on',  !_hiddenLayers.has(lf));
    b.classList.toggle('off',  _hiddenLayers.has(lf));
  }});
  _applyLayerFilters();
}}

function _applyLayerFilters() {{
  var globalShow = document.getElementById('showFlakes').checked;
  allFlakeEls.forEach(function(f) {{
    var visible = globalShow && !_hiddenLayers.has(f.el.dataset.layer);
    f.el.style.visibility = visible ? '' : 'hidden';
  }});
}}

function toggleFlakes() {{
  _applyLayerFilters();
}}

function _updateFlakeCount() {{
  var n = allFlakeEls.length;
  document.getElementById('flakeCountSidebar').textContent = n;
  var nMarkers = allFlakeEls.filter(function(f) {{ return f.source === 'marker'; }}).length;
  var det = detectedCandidates.length;
  document.getElementById('markCount').textContent =
    (det ? det + ' detected' : '') + (det && nMarkers ? '  ' : '') + (nMarkers ? nMarkers + ' marked' : '');
}}

// ── Wafer extents polygon ─────────────────────────────────────────────────────

function drawExtents() {{
  var canvas = document.getElementById('extentsCanvas');
  var vdiv = viewer.element;
  canvas.width  = vdiv.clientWidth;
  canvas.height = vdiv.clientHeight;
  var ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!document.getElementById('showExtents').checked) return;
  if (!waferExtents || waferExtents.length < 3) return;

  var rect = vdiv.getBoundingClientRect();
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(255,255,255,0.55)';
  ctx.lineWidth = 2;
  ctx.setLineDash([10, 5]);
  var first = true;
  for (var i = 0; i < waferExtents.length; i++) {{
    var pt = waferExtents[i];
    var vp = _vpFromMm(pt[0], pt[1]);
    if (!vp) continue;
    var sp = viewer.viewport.viewportToWindowCoordinates(vp);
    var x = sp.x - rect.left, y = sp.y - rect.top;
    if (first) {{ ctx.moveTo(x, y); first = false; }}
    else        {{ ctx.lineTo(x, y); }}
  }}
  ctx.closePath();
  ctx.stroke();
}}

// ── ORB/PC correction vector overlay ─────────────────────────────────────────
// corrVectors: [[x_mm, y_mm, dx_mm, dy_mm, method], ...]  method: 0=ORB 1=PC 2=filled
// Arrow tip offset in world space = correction_mm × CORR_SCALE / 1000.
// At CORR_SCALE=2000 a 100µm accumulated drift → ~14 screen px on a 20mm-wide scan at full zoom.
var CORR_SCALE = 2000;
var REF_CORR_UM = 100;   // legend reference value in µm
function drawCorrs() {{
  var cb = document.getElementById('showCorrs');
  var legend = document.getElementById('corrLegend');
  if (legend) legend.style.display = (cb && cb.checked) ? '' : 'none';
  var canvas = document.getElementById('corrsCanvas');
  var vdiv = viewer.element;
  canvas.width  = vdiv.clientWidth;
  canvas.height = vdiv.clientHeight;
  var ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!cb || !cb.checked || !corrVectors || corrVectors.length === 0) return;
  var rect = vdiv.getBoundingClientRect();
  var colours = ['rgba(68,170,255,0.85)', 'rgba(68,255,136,0.85)', 'rgba(160,160,160,0.7)'];
  for (var i = 0; i < corrVectors.length; i++) {{
    var v = corrVectors[i];
    var vp0 = _vpFromMm(v[0], v[1]);
    if (!vp0) continue;
    var sp0 = viewer.viewport.viewportToWindowCoordinates(vp0);
    var x0c = sp0.x - rect.left, y0c = sp0.y - rect.top;
    var vp1 = _vpFromMm(v[0] + v[2] * CORR_SCALE / 1000, v[1] + v[3] * CORR_SCALE / 1000);
    var sp1 = viewer.viewport.viewportToWindowCoordinates(vp1);
    var x1c = sp1.x - rect.left, y1c = sp1.y - rect.top;
    var col = colours[Math.min(v[4], 2)];
    ctx.strokeStyle = col;
    ctx.fillStyle   = col;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(x0c, y0c);
    ctx.lineTo(x1c, y1c);
    ctx.stroke();
    var angle = Math.atan2(y1c - y0c, x1c - x0c);
    var hs = 4;
    ctx.beginPath();
    ctx.moveTo(x1c, y1c);
    ctx.lineTo(x1c - hs * Math.cos(angle - 0.5), y1c - hs * Math.sin(angle - 0.5));
    ctx.lineTo(x1c - hs * Math.cos(angle + 0.5), y1c - hs * Math.sin(angle + 0.5));
    ctx.closePath();
    ctx.fill();
  }}

  // ── Scale legend (bottom-left) ───────────────────────────────────────────
  // Use the same _vpFromMm pipeline so the legend arrow scales with zoom.
  var refMm = REF_CORR_UM / 1000;
  var midX = GLOBAL_X0 + GLOBAL_PHYS_W / 2, midY = GLOBAL_Y0;
  var vpL0 = _vpFromMm(midX, midY);
  var vpL1 = _vpFromMm(midX + refMm * CORR_SCALE / 1000, midY);
  if (vpL0 && vpL1) {{
    var spL0 = viewer.viewport.viewportToWindowCoordinates(vpL0);
    var spL1 = viewer.viewport.viewportToWindowCoordinates(vpL1);
    var arrowPx = Math.abs(spL1.x - spL0.x);
    var lx = 16, ly = canvas.height - 16;
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    ctx.fillRect(lx - 4, ly - 18, arrowPx + 72, 26);
    ctx.strokeStyle = 'rgba(255,255,180,0.95)';
    ctx.fillStyle   = 'rgba(255,255,180,0.95)';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + arrowPx, ly); ctx.stroke();
    var hs2 = 5;
    ctx.beginPath();
    ctx.moveTo(lx + arrowPx, ly);
    ctx.lineTo(lx + arrowPx - hs2 * Math.cos(-0.5), ly - hs2 * Math.sin(-0.5));
    ctx.lineTo(lx + arrowPx - hs2 * Math.cos(+0.5), ly - hs2 * Math.sin(+0.5));
    ctx.closePath(); ctx.fill();
    ctx.font = '11px monospace';
    ctx.fillStyle = 'rgba(255,255,180,0.95)';
    ctx.fillText('= ' + REF_CORR_UM + 'µm', lx + arrowPx + 6, ly + 4);
  }}
}}

// ── Navigate ──────────────────────────────────────────────────────────────────
// Shift+click moves the stage — no mode toggle needed; Shift is the intent signal.

// On load, ping the nav port.
// • Unreachable → disable Navigate button + show stale badge.
// • Reachable, wrong/no sample → warn but keep enabled (stage coords are still valid).
// • Reachable, matching sample → fetch live flakes, refresh overlay silently.
function _showStaleBadge(reason) {{
  var badge = document.createElement('span');
  badge.style.cssText = 'margin-left:8px;padding:2px 7px;border-radius:3px;' +
    'background:#5a4500;color:#fdb;font-size:11px;cursor:default;';
  badge.title = reason;
  badge.textContent = '⏱ Data as of ' + MAP_BAKED_DATE;
  var tb = document.getElementById('toolbar');
  if (tb) tb.appendChild(badge);
}}

function _loadLiveFlakes() {{
  fetch('http://127.0.0.1:' + NAV_PORT + '/flakes', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: '{{}}'
  }}).then(function(r) {{ return r.json(); }})
    .then(function(live) {{
      if (!Array.isArray(live) || !live.length) return;
      sampleFlakes = live;
      _placeFlakes();
    }}).catch(function() {{}});
}}

// Live reference→current placement from the app (refreshed on every /ping).
// Map coords are in this scan's frame; current = live_placement ∘
// inverse(SCAN_PLACEMENT), mirroring vision.registration.map_to_current_stage.
// SCAN_PLACEMENT null (pre-stamp map) → the inverse step is identity, so this
// reduces to the historical apply_placement(reference).
var PLACEMENT = null;
function _applyInverseScan(x, y) {{
  if (!SCAN_PLACEMENT) return {{x: x, y: y}};
  var a = SCAN_PLACEMENT.rotation_deg * Math.PI / 180, c = Math.cos(a), s = Math.sin(a);
  var dx = x - SCAN_PLACEMENT.dx_mm, dy = y - SCAN_PLACEMENT.dy_mm;
  return {{x: c * dx + s * dy, y: -s * dx + c * dy}};
}}
function _applyPlacement(x, y) {{
  var r = _applyInverseScan(x, y);
  if (!PLACEMENT) return r;
  var a = PLACEMENT.rotation_deg * Math.PI / 180, c = Math.cos(a), s = Math.sin(a);
  return {{x: c * r.x - s * r.y + PLACEMENT.dx_mm, y: s * r.x + c * r.y + PLACEMENT.dy_mm}};
}}
if (NAV_PORT) {{
  fetch('http://127.0.0.1:' + NAV_PORT + '/ping', {{method:'POST',
      headers:{{'Content-Type':'application/json'}}, body:'{{}}'}})
    .then(function(r) {{ return r.json(); }})
    .then(function(d) {{
      PLACEMENT = d.placement || null;
      var btn = document.getElementById('navBtn');
      var appFolder = d.sample_folder || '';
      if (!appFolder) {{
        // App is reachable but no sample open
        if (btn) btn.title = 'Navigate active — no sample loaded in app';
        _showNavWarning('No sample open in app — navigating moves the stage to map coordinates.');
        _showStaleBadge('No sample is open in the app — flake data is from map generation time.');
      }} else if (MAP_SAMPLE_FOLDER && appFolder !== MAP_SAMPLE_FOLDER) {{
        // Different sample open
        if (btn) btn.title = 'Navigate active — different sample open: ' + appFolder;
        _showNavWarning('App has “' + appFolder + '” open, but this map shows “' + MAP_SAMPLE_FOLDER + '”.');
        _showStaleBadge('A different sample is open in the app — flake data is from map generation time.');
      }} else {{
        // Matching sample — fetch live flakes and refresh overlay silently
        if (btn) btn.classList.add('active');
        _loadLiveFlakes();
      }}
    }})
    .catch(function() {{
      var btn = document.getElementById('navBtn');
      if (btn) {{ btn.disabled = true; btn.title = 'Navigate: app not reachable from this browser'; }}
      _showStaleBadge('App is not reachable — flake data is from map generation time. Open standa-stacker to enable live updates.');
    }});
}} else {{
  _showStaleBadge('No app connection configured — flake data is from map generation time.');
}}

function _showNavWarning(msg) {{
  var bar = document.createElement('div');
  bar.style.cssText = 'position:fixed;top:36px;left:0;right:0;z-index:3000;' +
    'background:#7a3800;color:#fdd;font-size:12px;padding:5px 12px;' +
    'display:flex;justify-content:space-between;align-items:center;';
  bar.innerHTML = '<span>⚠ ' + msg + '</span>' +
    '<button onclick="this.parentElement.remove()" style="background:none;border:none;' +
    'color:#fdd;cursor:pointer;font-size:14px;padding:0 4px;">✕</button>';
  document.body.appendChild(bar);
}}

function _navigateTo(x_mm, y_mm) {{
  // Brief feedback in the coords display — overwritten naturally on next mousemove
  var cs = document.getElementById('coords');
  if (cs) cs.textContent = '→ ' + x_mm.toFixed(3) + ', ' + y_mm.toFixed(3) + ' mm';
  fetch('http://127.0.0.1:' + NAV_PORT + '/navigate', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{x_mm: x_mm, y_mm: y_mm, scan_placement: SCAN_PLACEMENT}})
  }}).catch(function(e) {{ console.log('Navigate failed:', e); }});
}}

// Place overlays once the first tiled image is in the world.
// Cannot use 'open' here — items are added via addTiledImage() after the viewer
// initialises, so world.getItemAt(0) returns null during the 'open' event.
var _overlaysPlaced = false;
viewer.world.addHandler('add-item', function _onFirstItem() {{
  if (_overlaysPlaced) return;
  _overlaysPlaced = true;
  viewer.world.removeHandler('add-item', _onFirstItem);
  _placeCandidates();
  _placeFlakes();
  _restoreMarkersFromStorage();
  drawExtents();
  drawCorrs();
  _updateLabelCount();
  _updateCandStepInfo();
}});
viewer.addHandler('update-viewport', function() {{ drawExtents(); drawCorrs(); }});
viewer.addHandler('resize', function() {{ drawExtents(); drawCorrs(); }});

// ── User annotation markers ───────────────────────────────────────────────────
// Ctrl+click places a marker — no mode toggle needed.

// ── Marker persistence (localStorage + app) ──────────────────────────────────
var _MARKER_KEY = 'markerDots_' + STEM;

function _saveMarkersToStorage() {{
  var m = allFlakeEls
    .filter(function(f) {{ return f.source === 'marker'; }})
    .map(function(f) {{ return {{x_mm: f.x_mm, y_mm: f.y_mm, note: f.note, layer: f.layer}}; }});
  try {{ localStorage.setItem(_MARKER_KEY, JSON.stringify(m)); }} catch(e) {{}}
}}

function _restoreMarkersFromStorage() {{
  try {{
    var raw = localStorage.getItem(_MARKER_KEY);
    if (!raw) return;
    JSON.parse(raw).forEach(function(m) {{
      _addFlake(m.x_mm, m.y_mm, {{source: 'marker', note: m.note || '', layer: m.layer != null ? m.layer : null}});
    }});
  }} catch(e) {{}}
}}

function _addUserMarker(xMm, yMm, note, layer) {{
  _addFlake(xMm, yMm, {{source: 'marker', note: note || '', layer: layer}});
  _saveMarkersToStorage();
  // Also persist to sample catalogue if app is connected
  if (NAV_PORT) {{
    fetch('http://127.0.0.1:' + NAV_PORT + '/import_flakes', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{markers: [{{x_mm: xMm, y_mm: yMm, note: note || '',
                              layer: layer != null ? layer : null}}],
                             scan_placement: SCAN_PLACEMENT}})
    }}).then(function() {{
      var cs = document.getElementById('coords');
      if (cs) cs.textContent = '✓ Saved to catalogue';
    }}).catch(function(e) {{ console.log('import_flakes failed:', e); }});
  }}
}}

// Escape HTML attribute values
function _esc(s) {{
  return String(s == null ? '' : s)
    .replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}}

function _applyCatalogueUpdate(flakeId, updates) {{
  var sf = sampleFlakes.find(function(f) {{ return f.id === flakeId; }});
  if (sf) Object.assign(sf, updates);
  // Track all changed fields for Save JSON
  if (!_catalogueUpdates[flakeId]) _catalogueUpdates[flakeId] = {{}};
  Object.assign(_catalogueUpdates[flakeId], updates);
  // Re-render the dot (picks up updated sf.name / sf.layer_count / sf.area_um2)
  var idx = allFlakeEls.findIndex(function(f) {{ return f.el.dataset.flakeId === flakeId; }});
  if (idx !== -1) {{
    viewer.removeOverlay(allFlakeEls[idx].el);
    allFlakeEls.splice(idx, 1);
    if (sf) _addFlake(sf.x_mm, sf.y_mm, {{
      source: 'catalogue', id: sf.id, name: sf.name,
      layer: sf.layer_count, layer_count: sf.layer_count,
      note: sf.notes || '',
      status: sf.status, area_um2: sf.area_um2,
      source_tag: sf.source === 'map' ? 'map' : null
    }});
  }}
  // Persist via nav port if app is running
  if (NAV_PORT) {{
    fetch('http://127.0.0.1:' + NAV_PORT + '/update_flake', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify(Object.assign({{id: flakeId}}, updates))
    }}).catch(function(e) {{ console.log('update_flake failed:', e); }});
  }}
}}

// Alt+click / plain-click on viewer: remove or edit flake dots.
// Use getBoundingClientRect hit-test instead of e.target walk-up — OSD may
// redirect pointer-capture so e.target is the OSD canvas rather than the
// overlay element, making walk-up miss sibling overlay nodes.
function _hitFlakeDot(clientX, clientY, sourceFilter) {{
  var sel = sourceFilter
    ? '.flake-dot[data-source="' + sourceFilter + '"]'
    : '.flake-dot[data-source="marker"], .flake-dot[data-source="catalogue"]';
  var dots = viewer.element.querySelectorAll(sel);
  for (var _i = 0; _i < dots.length; _i++) {{
    var r = dots[_i].getBoundingClientRect();
    if (clientX >= r.left && clientX <= r.right &&
        clientY >= r.top  && clientY <= r.bottom) return dots[_i];
  }}
  return null;
}}

viewer.element.addEventListener('click', function(e) {{
  // Alt+click: remove marker or catalogue flake
  if (e.altKey) {{
    var hit = _hitFlakeDot(e.clientX, e.clientY, null);
    if (hit) {{
      var src = hit.dataset.source;
      viewer.removeOverlay(hit);
      allFlakeEls = allFlakeEls.filter(function(f) {{ return f.el !== hit; }});
      if (src === 'marker') _saveMarkersToStorage();
      if (src === 'catalogue' && hit.dataset.flakeId) {{
        var fid = hit.dataset.flakeId;
        var si = sampleFlakes.findIndex(function(f) {{ return f.id === fid; }});
        if (si !== -1) sampleFlakes.splice(si, 1);
        if (NAV_PORT) {{
          fetch('http://127.0.0.1:' + NAV_PORT + '/delete_flake', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{id: fid}})
          }}).catch(function(e) {{ console.log('delete_flake failed:', e); }});
        }}
      }}
      _updateFlakeCount();
      e.stopPropagation();
      e.preventDefault();
    }}
    return;
  }}
  // Plain click on catalogue or user-placed marker: open edit popup
  if (!e.ctrlKey && !e.shiftKey) {{
    var hit2 = _hitFlakeDot(e.clientX, e.clientY, 'catalogue');
    if (hit2 && hit2.dataset.flakeId) {{
      _showCatalogueFlakePopup(hit2.dataset.flakeId, e.clientX, e.clientY);
      e.stopPropagation();
      return;
    }}
    var hit3 = _hitFlakeDot(e.clientX, e.clientY, 'marker');
    if (hit3) {{
      var entry = allFlakeEls.find(function(f) {{ return f.el === hit3; }});
      if (entry) {{ _showExistingMarkerPopup(entry, e.clientX, e.clientY); e.stopPropagation(); }}
    }}
  }}
}});
// Close popup when viewer pans or zooms
viewer.addHandler('animation-start', _cancelMarkerPopup);

// ── Marker popup (new markers + catalogue layer editing) ──────────────────────
var _pendingMarkerPos       = null;
var _pendingLayer           = null;
var _pendingCatalogueFlakeId = null;
var _pendingMarkerEntry     = null;  // allFlakeEls entry for existing marker edits
var _catalogueUpdates       = {{}};   // {{flakeId: {{...}}}}

function _popupPosition(clientX, clientY) {{
  var pop = document.getElementById('markerPopup');
  pop.style.left = Math.min(clientX + 10, window.innerWidth  - pop.offsetWidth  - 10) + 'px';
  pop.style.top  = Math.min(clientY + 10, window.innerHeight - pop.offsetHeight - 10) + 'px';
  pop.style.display = 'block';
}}

function _showMarkerPopup(xMm, yMm, clientX, clientY) {{
  _pendingMarkerPos        = {{x: xMm, y: yMm}};
  _pendingCatalogueFlakeId = null;
  _pendingMarkerEntry      = null;
  _pendingLayer = null;
  document.getElementById('markerPopupTitle').textContent =
    xMm.toFixed(3) + ', ' + yMm.toFixed(3) + ' mm';
  document.getElementById('flakeMeta').style.display = 'none';
  document.getElementById('layerPickLabel').style.display = '';
  document.getElementById('markerNoteInput').style.display = '';
  document.getElementById('markerNoteInput').value = '';
  document.getElementById('markerPopupConfirm').textContent = 'Place';
  document.querySelectorAll('.layer-pick-btn').forEach(function(b) {{ b.classList.remove('sel'); }});
  _popupPosition(clientX, clientY);
  setTimeout(function() {{ document.getElementById('markerNoteInput').focus(); }}, 50);
}}

function _showExistingMarkerPopup(entry, clientX, clientY) {{
  _pendingMarkerEntry      = entry;
  _pendingMarkerPos        = null;
  _pendingCatalogueFlakeId = null;
  _pendingLayer = entry.layer;
  document.getElementById('markerPopupTitle').textContent =
    entry.x_mm.toFixed(3) + ', ' + entry.y_mm.toFixed(3) + ' mm';
  var meta = document.getElementById('flakeMeta');
  meta.innerHTML =
    '<span class="mk">Note</span>' +
    '<input class="mi" data-field="note" value="' + _esc(entry.note || '') + '">';
  meta.style.display = 'grid';
  document.getElementById('layerPickLabel').style.display = '';
  document.getElementById('markerNoteInput').style.display = 'none';
  document.getElementById('markerPopupConfirm').textContent = 'Save';
  document.querySelectorAll('.layer-pick-btn').forEach(function(b) {{
    b.classList.toggle('sel', _pendingLayer !== null && parseInt(b.dataset.l) === _pendingLayer);
  }});
  _popupPosition(clientX, clientY);
}}

function _buildMetaGrid(flake) {{
  var html = '';
  // Read-only position rows
  if (flake.chip_x_mm != null && flake.chip_y_mm != null)
    html += '<span class="mk">Chip</span><span class="mv">' +
            flake.chip_x_mm.toFixed(3) + ', ' + flake.chip_y_mm.toFixed(3) + ' mm</span>';
  // Stored flake.x_mm/y_mm is REFERENCE-stage; show CURRENT-stage so the number
  // matches the live readout after a remount (= apply_placement(reference)).
  var _cur = _applyPlacement(flake.x_mm, flake.y_mm);
  html += '<span class="mk">Stage</span><span class="mv">' +
          _cur.x.toFixed(3) + ', ' + _cur.y.toFixed(3) + ' mm' +
          (PLACEMENT ? '' : ' <i style="opacity:.55">(reference — app offline)</i>') +
          '</span>';
  // Editable fields
  html += '<span class="mk">Name</span><input class="mi" data-field="name" value="' + _esc(flake.name) + '">';
  html += '<span class="mk">Area</span>' +
          '<span class="mi-wrap"><input class="mi" data-field="area_um2" type="number" min="0" step="1" value="' +
          _esc(flake.area_um2 != null ? Math.round(flake.area_um2) : '') + '"> µm²</span>';
  html += '<span class="mk">Mag</span><input class="mi" data-field="magnification" value="' + _esc(flake.magnification) + '">';
  html += '<span class="mk">Cleanliness</span><input class="mi" data-field="cleanliness" value="' + _esc(flake.cleanliness) + '">';
  html += '<span class="mk">Isolation</span><input class="mi" data-field="isolation" value="' + _esc(flake.isolation) + '">';
  html += '<span class="mk">Notes</span><input class="mi" data-field="notes" value="' + _esc(flake.notes) + '">';
  // Read-only footer
  var footer = [];
  if (flake.source) footer.push('Source: ' + flake.source);
  if (flake.created_at) footer.push('Added: ' + flake.created_at);
  if (flake.n_images) footer.push(flake.n_images + ' img');
  if (footer.length)
    html += '<span style="grid-column:1/-1;color:#555;font-size:10px;margin-top:4px;">' +
            footer.join('  ·  ') + '</span>';
  return html;
}}

function _showCatalogueFlakePopup(flakeId, clientX, clientY) {{
  _pendingCatalogueFlakeId = flakeId;
  _pendingMarkerPos        = null;
  _pendingMarkerEntry      = null;
  var flake = sampleFlakes.find(function(f) {{ return f.id === flakeId; }});
  var lc = flake ? flake.layer_count : null;
  _pendingLayer = lc;
  var title = flakeId;
  if (flake && flake.name) title += ' — ' + flake.name;
  document.getElementById('markerPopupTitle').textContent = title;
  var meta = document.getElementById('flakeMeta');
  if (flake) {{
    meta.innerHTML = _buildMetaGrid(flake);
    meta.style.display = 'grid';
  }} else {{
    meta.style.display = 'none';
  }}
  document.getElementById('layerPickLabel').style.display = '';
  document.getElementById('markerNoteInput').style.display = 'none';
  document.getElementById('markerPopupConfirm').textContent = 'Save';
  document.querySelectorAll('.layer-pick-btn').forEach(function(b) {{
    b.classList.toggle('sel', lc !== null && parseInt(b.dataset.l) === lc);
  }});
  _popupPosition(clientX, clientY);
}}

function _pickLayer(n) {{
  _pendingLayer = n;
  document.querySelectorAll('.layer-pick-btn').forEach(function(b) {{
    b.classList.toggle('sel', parseInt(b.dataset.l) === n);
  }});
  _confirmMarker();
}}

function _collectMetaInputs() {{
  var updates = {{}};
  document.querySelectorAll('#flakeMeta .mi').forEach(function(inp) {{
    var field = inp.dataset.field;
    var val = inp.value.trim();
    if (field === 'area_um2') updates[field] = val !== '' ? parseFloat(val) : null;
    else updates[field] = val || null;
  }});
  return updates;
}}

function _confirmMarker() {{
  if (_pendingCatalogueFlakeId) {{
    var updates = Object.assign({{layer_count: _pendingLayer}}, _collectMetaInputs());
    _applyCatalogueUpdate(_pendingCatalogueFlakeId, updates);
    _cancelMarkerPopup();
    return;
  }}
  if (_pendingMarkerEntry) {{
    var mi = _collectMetaInputs();
    _pendingMarkerEntry.note  = mi.note  != null ? mi.note  : (_pendingMarkerEntry.note || '');
    _pendingMarkerEntry.layer = _pendingLayer;
    // Refresh visual dot
    var lbl = _pendingMarkerEntry.el.querySelector('.ann-marker-label');
    var color = (_pendingLayer != null ? _L_COLOR[_pendingLayer] : null) || '#ff0';
    _pendingMarkerEntry.el.style.background = color;
    if (lbl) {{ lbl.textContent = _pendingLayer != null ? (_pendingLayer > 0 ? _pendingLayer + 'L' : '?L') : ''; lbl.style.color = color; }}
    // Rebuild tooltip with updated note/layer
    var tip = [_pendingMarkerEntry.x_mm.toFixed(4) + ', ' + _pendingMarkerEntry.y_mm.toFixed(4) + ' mm'];
    if (_pendingLayer !== null) tip.push(_pendingLayer > 0 ? _pendingLayer + 'L' : '?L');
    if (_pendingMarkerEntry.note) tip.push(_pendingMarkerEntry.note);
    tip.push('Alt+click to remove');
    _pendingMarkerEntry.el.title = tip.join('  ');
    _saveMarkersToStorage();
    _cancelMarkerPopup();
    return;
  }}
  if (!_pendingMarkerPos) return;
  var note = document.getElementById('markerNoteInput').value.trim();
  _addUserMarker(_pendingMarkerPos.x, _pendingMarkerPos.y, note, _pendingLayer);
  _cancelMarkerPopup();
}}

function _cancelMarkerPopup() {{
  document.getElementById('markerPopup').style.display = 'none';
  _pendingMarkerPos   = null;
  _pendingMarkerEntry = null;
  _pendingLayer       = null;
  _pendingCatalogueFlakeId = null;
}}

document.addEventListener('keydown', function(e) {{
  if (e.key === 'Escape') _cancelMarkerPopup();
}});
document.getElementById('markerNoteInput').addEventListener('keydown', function(e) {{
  if (e.key === 'Enter') _confirmMarker();
}});

// Ctrl+click → place marker popup; Shift+click → navigate (no mode toggle required)
viewer.element.addEventListener('click', function(e) {{
  var rect = viewer.element.getBoundingClientRect();
  var px   = new OpenSeadragon.Point(e.clientX - rect.left, e.clientY - rect.top);
  if (e.ctrlKey && !e.shiftKey && !e.altKey) {{
    e.preventDefault();
    var mm = _mmFromVp(viewer.viewport.pointFromPixel(px));
    if (mm) _showMarkerPopup(mm.x, mm.y, e.clientX, e.clientY);
  }} else if (e.shiftKey && !e.ctrlKey && !e.altKey && NAV_PORT) {{
    var navBtn = document.getElementById('navBtn');
    if (navBtn && navBtn.disabled) return;   // app not reachable
    e.preventDefault();
    var mm2 = _mmFromVp(viewer.viewport.pointFromPixel(px));
    if (mm2) _navigateTo(mm2.x, mm2.y);
  }}
}});

function clearUserMarkers() {{
  var toRemove = allFlakeEls.filter(function(f) {{ return f.source === 'marker'; }});
  toRemove.forEach(function(f) {{ viewer.removeOverlay(f.el); }});
  allFlakeEls = allFlakeEls.filter(function(f) {{ return f.source !== 'marker'; }});
  _saveMarkersToStorage();
  _updateFlakeCount();
}}

function downloadMarkers() {{
  var markers = allFlakeEls.filter(function(f) {{ return f.source === 'marker'; }}).map(function(m) {{
    var d = {{x_mm: m.x_mm, y_mm: m.y_mm, note: m.note}};
    if (m.layer !== null && m.layer !== undefined) d.layer = m.layer;
    return d;
  }});
  var hasUpdates = Object.keys(_catalogueUpdates).length > 0;
  var out = hasUpdates ? {{markers: markers, catalogue_updates: _catalogueUpdates}} : markers;
  var blob = new Blob([JSON.stringify(out, null, 2)], {{type: 'application/json'}});
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = STEM + '_markers.json';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}}

function loadMarkersFromFile() {{
  var input = document.createElement('input');
  input.type = 'file';
  input.accept = '.json';
  input.onchange = function(e) {{
    var file = e.target.files[0];
    if (!file) return;
    var reader = new FileReader();
    reader.onload = function(ev) {{
      try {{
        var data = JSON.parse(ev.target.result);
        var markers = Array.isArray(data) ? data : (data.markers || []);
        var updates = Array.isArray(data) ? {{}} : (data.catalogue_updates || {{}});
        if (!Array.isArray(markers)) throw new Error('Expected markers array');
        markers.forEach(function(m) {{
          _addUserMarker(m.x_mm, m.y_mm, m.note || '',
                         (m.layer !== undefined) ? m.layer : null);
        }});
        Object.keys(updates).forEach(function(fid) {{
          _applyCatalogueUpdate(fid, updates[fid]);
        }});
        _updateFlakeCount();
        if (NAV_PORT && data.length) {{
          fetch('http://127.0.0.1:' + NAV_PORT + '/import_flakes', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{markers: data, scan_placement: SCAN_PLACEMENT}})
          }}).then(function(r) {{
            if (r.ok) console.log('Imported ' + data.length + ' flake(s) to sample catalogue');
            else console.log('Import returned status ' + r.status);
          }}).catch(function(e) {{ console.log('Import failed:', e); }});
        }}
      }} catch(err) {{ alert('Could not load markers: ' + err); }}
    }};
    reader.readAsText(file);
  }};
  input.click();
}}
</script>
</body>
</html>
"""

    html_path.write_text(html)
    print(f"Viewer written: {html_path}")
    return html_path

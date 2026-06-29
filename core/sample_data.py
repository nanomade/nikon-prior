"""
core/sample_data.py
-------------------
Data model for a sample and its flakes.  Pure Python / JSON — no Qt.

Directory layout
----------------
users/
  <user>/
    <name>_YYYY-MM-DD/
      sample.json
      images/
        F01_50x_104002.png
        ...
"""

import json
import math
import os
import re
from datetime import date, datetime
from pathlib import Path

_APP_DIR = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Users list
# ---------------------------------------------------------------------------

_USERS_FILE = _APP_DIR / 'users.json'


_USERS_DIR = _APP_DIR / 'users'


def load_users() -> list[str]:
    try:
        with open(_USERS_FILE) as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            return [str(u) for u in data]
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    # Fall back: discover from users/ subdirectories
    if _USERS_DIR.is_dir():
        return sorted(p.name for p in _USERS_DIR.iterdir() if p.is_dir())
    return []


def save_users(users: list[str]):
    with open(_USERS_FILE, 'w') as f:
        json.dump(sorted(set(users)), f, indent=2)


def add_user(name: str) -> list[str]:
    users = load_users()
    name = name.strip()
    if name and name not in users:
        users.append(name)
        save_users(users)
    return load_users()


# ---------------------------------------------------------------------------
# Flake
# ---------------------------------------------------------------------------

_STATUSES    = ['Candidate', 'Approved', 'In Use', 'Rejected']
_CLEANLINESS = ['', 'Clean', 'Bubbles', 'Contaminated']
_ISOLATION   = ['', 'Isolated', 'Clustered', 'Touching']
_SUBSTRATES  = ['SiO2/Si 285nm', 'SiO2/Si 90nm', 'hBN', 'Sapphire', 'Quartz', 'Other']


def new_flake(flake_id: str, name: str,
              stage_x_mm: float, stage_y_mm: float, z_mm: float,
              magnification: str, r_deg: float = 0.0,
              chip_x_mm: float | None = None,
              chip_y_mm: float | None = None,
              layer_count: int | None = None,
              source: str = 'app', confirmed: bool = False) -> dict:
    now = datetime.now().isoformat(timespec='seconds')
    return {
        'id':           flake_id,
        'name':         name,
        'layer_count':  layer_count,
        'stage_x_mm':   stage_x_mm,
        'stage_y_mm':   stage_y_mm,
        'chip_x_mm':    chip_x_mm,
        'chip_y_mm':    chip_y_mm,
        'z_mm':         z_mm,
        'r_deg':        r_deg,
        'magnification': magnification,
        'source':       source,        # provenance: app (scope) | map (browser) | auto (detector)
        'confirmed':    confirmed,      # user-verified at the scope (find → navigate → confirm)
        'status':       'Candidate',
        'locked':       False,
        'area_um2':     None,
        'circularity':  None,    # 4π·area/perimeter² (shape metric from detector)
        'aspect_ratio': None,    # long/short bbox side
        'solidity':     None,    # area / convex-hull area
        'cleanliness':  '',
        'isolation':    '',
        'notes':        '',
        'images':       [],
        'created_at':   now,
        'updated_at':   now,
    }


def focus_z_at(scan_images: list[dict], x_mm: float, y_mm: float) -> float | None:
    """Focus Z at a stage XY from a scan's per-tile measured focus.

    Returns the `z_actual_mm` (fallback `z_plane_mm`) of the nearest scan tile —
    so a flake derived from a 2-D map or detector navigates IN FOCUS instead of
    to a Z=0 placeholder. Returns None if no tile has a usable Z."""
    best_d, best_z = float('inf'), None
    for im in scan_images:
        ix, iy = im.get('x_mm'), im.get('y_mm')
        if ix is None or iy is None:
            continue
        d = (ix - x_mm) ** 2 + (iy - y_mm) ** 2
        if d < best_d:
            z = im.get('z_actual_mm') or im.get('z_plane_mm')
            if z:
                best_d, best_z = d, float(z)
    return best_z


def next_flake_id(flakes: list[dict]) -> str:
    """Return the next F-number not already in the list."""
    used = set()
    for f in flakes:
        m = re.match(r'F(\d+)', f.get('id', ''))
        if m:
            used.add(int(m.group(1)))
    n = 1
    while n in used:
        n += 1
    return f'F{n:02d}'


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------

def _sample_dir(users_root, user: str, folder: str) -> str:
    return os.path.join(str(users_root), user, folder)


def _json_path(sample_dir: str) -> str:
    return os.path.join(sample_dir, 'sample.json')


def _images_dir(sample_dir: str) -> str:
    return os.path.join(sample_dir, 'images')


def _scans_dir(sample_dir: str) -> str:
    return os.path.join(sample_dir, 'scans')


def _timelapse_dir(sample_dir: str) -> str:
    return os.path.join(sample_dir, 'timelapse')


def _frames_dir(sample_dir: str) -> str:
    return os.path.join(sample_dir, 'frames')


def new_sample(users_root: str, user: str, short_name: str,
               substrate: str = 'SiO2/Si 285nm') -> dict:
    """Create directory structure and return a fresh sample dict."""
    folder = f"{short_name}_{date.today().isoformat()}"
    sdir   = _sample_dir(users_root, user, folder)
    os.makedirs(_images_dir(sdir), exist_ok=True)
    sample = {
        'name':       short_name,
        'folder':     folder,
        'user':       user,
        'substrate':  substrate,
        'created':    datetime.now().isoformat(timespec='seconds'),
        'notes':      '',
        'transform':  None,   # index-mark grid ↔ stage (Phase 2)
        'placement':  {       # physical load state and coordinate registration
            'loaded':       False,
            'loaded_at':    None,
            'registration': None,   # set by mark_registered()
        },
        'flakes':     [],
    }
    with open(_json_path(sdir), 'w') as f:
        json.dump(sample, f, indent=2)
    return sample


def load_sample(users_root: str, user: str, folder: str) -> dict:
    path = _json_path(_sample_dir(users_root, user, folder))
    with open(path) as f:
        return json.load(f)


def save_sample(users_root: str, sample: dict):
    sdir = _sample_dir(users_root, sample['user'], sample['folder'])
    with open(_json_path(sdir), 'w') as f:
        json.dump(sample, f, indent=2)


# ── Placement / registration helpers ─────────────────────────────────────────

def _placement(sample: dict) -> dict:
    """Return placement sub-dict, creating it if the sample predates this field."""
    if 'placement' not in sample:
        sample['placement'] = {'loaded': False, 'loaded_at': None, 'registration': None}
    return sample['placement']


def registration_state(sample: dict) -> str:
    """
    Return the current registration state as a string:
        'none'       — no corner references captured yet
        'extents'    — corner refs saved but not yet confirmed for this placement
        'registered' — transform confirmed for current placement
        'stale'      — was registered but sample has been reloaded since
    """
    p = _placement(sample)
    reg = p.get('registration')
    if reg is None:
        return 'none'
    return reg.get('state', 'none')


def mark_loaded(sample: dict) -> bool:
    """
    Record that the sample has been physically placed on the stage.

    If the sample was previously registered, the registration transitions to
    'stale' (needs re-confirmation for the new placement).

    Returns True if the caller should prompt for re-registration
    (i.e. corners exist but placement is now unconfirmed).
    """
    p = _placement(sample)
    p['loaded']    = True
    p['loaded_at'] = datetime.now().isoformat(timespec='seconds')
    reg = p.get('registration') or {}
    state = reg.get('state', 'none')
    if state in ('registered', 'extents'):
        reg['state'] = 'stale'
        p['registration'] = reg
        return True
    return False


def mark_unloaded(sample: dict):
    """Record that the sample has been removed from the stage."""
    p = _placement(sample)
    p['loaded']    = False
    p['loaded_at'] = None
    reg = p.get('registration') or {}
    if reg.get('state') in ('registered', 'extents'):
        reg['state'] = 'stale'
        p['registration'] = reg


def mark_corners_saved(sample: dict, corners: list[dict]):
    """
    Record that corner reference images have been extracted (from Wafer Extents
    or a scan).  Transitions state to 'extents' — corners are known but the
    current placement hasn't been confirmed yet.

    corners: list from vision.registration.extract_scan_corners()
    """
    p = _placement(sample)
    p['registration'] = {
        'state':     'extents',
        'corners':   corners,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
    }


def mark_registered(sample: dict, dx_mm: float, dy_mm: float,
                    rotation_deg: float, rms_mm: float,
                    n_points: int, method: str = 'corners'):
    """Record a confirmed placement registration."""
    p = _placement(sample)
    reg = p.get('registration') or {}
    reg.update({
        'state':        'registered',
        'method':       method,
        'dx_mm':        round(dx_mm, 6),
        'dy_mm':        round(dy_mm, 6),
        'rotation_deg': round(rotation_deg, 4),
        'rms_mm':       round(rms_mm, 4),
        'n_points':     n_points,
        'timestamp':    datetime.now().isoformat(timespec='seconds'),
    })
    p['registration'] = reg


def get_placement_transform(sample: dict) -> dict | None:
    """
    Return the current placement transform dict if registration is valid,
    otherwise None.  Callers should treat None as 'identity / unregistered'.
    """
    state = registration_state(sample)
    if state != 'registered':
        return None
    return sample.get('placement', {}).get('registration')


def get_placement_transform_hint(sample: dict) -> dict | None:
    """Return the stored placement transform as a best-effort hint.

    Unlike get_placement_transform, also returns the transform when state is
    'stale'.  Use for navigation where an approximate position is better than
    the raw (unregistered) reference coordinates.  Returns None only when no
    transform has ever been computed for this sample.
    """
    reg = (_placement(sample).get('registration') or {})
    if 'dx_mm' in reg and 'dy_mm' in reg and 'rotation_deg' in reg:
        return reg
    return None


def save_chip_transform(sample: dict, chip_tf: dict):
    """Store the chip-local coordinate frame derived from corner positions.

    This is computed once from the reference corner positions and does not
    change between mounts (unlike the placement transform).
    """
    reg = _placement(sample).setdefault('registration', {})
    reg['chip_transform'] = chip_tf


def get_chip_transform(sample: dict) -> dict | None:
    """Return the chip-local coordinate frame, or None if not yet computed."""
    return (_placement(sample).get('registration') or {}).get('chip_transform')


def chip_to_stage(sample: dict, cx_mm: float, cy_mm: float) -> tuple[float, float] | None:
    """Convert chip-local (cx, cy) mm to current stage (sx, sy) mm.

    Uses the corner-registration path (chip_transform + placement_transform)
    unless chip_tf has an explicit 'grid_origin_xx' linking it to the
    index-mark coordinate system.
    Returns None if no chip transform is available.
    """
    chip_tf  = get_chip_transform(sample)
    index_tf = sample.get('transform')

    # Index-mark path only when chip_tf has been explicitly linked to the grid
    # (grid_origin_xx set).  Without it, (0, 0) would be assumed as the chip
    # origin in grid space, which is wrong for any real chip placement.
    if (index_tf is not None and chip_tf is not None
            and 'grid_origin_xx' in chip_tf):
        gox = chip_tf['grid_origin_xx']
        goy = chip_tf['grid_origin_yy']
        gs  = index_tf['grid_spacing_mm']
        eff_xx = gox + cx_mm / gs
        eff_yy = goy + cy_mm / gs
        return grid_to_stage(index_tf, eff_xx, eff_yy)

    # Corner-registration path: chip_local → reference_stage → current_stage
    if chip_tf is not None:
        from vision.registration import (chip_to_reference_stage,
                                          apply_placement_transform)
        ref_x, ref_y = chip_to_reference_stage(chip_tf, cx_mm, cy_mm)
        tf = get_placement_transform_hint(sample)
        if tf is None:
            return ref_x, ref_y
        return apply_placement_transform(tf, ref_x, ref_y)

    return None


def stage_to_chip(sample: dict, sx_mm: float, sy_mm: float) -> tuple[float, float] | None:
    """Convert current stage (sx, sy) mm to chip-local (cx, cy) mm.

    Uses the corner-registration path unless chip_tf has an explicit
    'grid_origin_xx' linking it to the index-mark coordinate system.
    Returns None if no chip transform is available.
    """
    chip_tf  = get_chip_transform(sample)
    index_tf = sample.get('transform')

    if (index_tf is not None and chip_tf is not None
            and 'grid_origin_xx' in chip_tf):
        eff_xx, eff_yy = stage_to_grid(index_tf, sx_mm, sy_mm)
        gox = chip_tf['grid_origin_xx']
        goy = chip_tf['grid_origin_yy']
        gs  = index_tf['grid_spacing_mm']
        return (eff_xx - gox) * gs, (eff_yy - goy) * gs

    if chip_tf is not None:
        from vision.registration import (reference_stage_to_chip,
                                          apply_inverse_placement_transform)
        tf = get_placement_transform_hint(sample)
        if tf is not None:
            sx_mm, sy_mm = apply_inverse_placement_transform(tf, sx_mm, sy_mm)
        return reference_stage_to_chip(chip_tf, sx_mm, sy_mm)

    return None


# ── Index-mark ↔ chip-local bridge ───────────────────────────────────────────
# Both systems route through stage coordinates as the common currency.
# chip_local → stage (via chip_transform + placement_transform)
# stage → grid  (via index-mark transform)
# — and vice versa.  No additional maths required; just compose the two chains.


def chip_to_grid(sample: dict,
                 cx_mm: float, cy_mm: float) -> tuple[float, float] | None:
    """Convert chip-local (cx, cy) mm to effective index-mark grid coordinates.

    Returns (eff_xx, eff_yy) in grid steps (500 µm per step), or None if
    either transform is unavailable.  To get local XX/YY + quadrant, divide
    eff by 100 and inspect the quotient.
    """
    result = chip_to_stage(sample, cx_mm, cy_mm)
    if result is None:
        return None
    sx, sy = result
    index_tf = sample.get('transform')
    if index_tf is None:
        return None
    return stage_to_grid(index_tf, sx, sy)


def grid_to_chip(sample: dict,
                 eff_xx: float, eff_yy: float) -> tuple[float, float] | None:
    """Convert effective index-mark grid coordinates to chip-local (cx, cy) mm.

    Returns None if either transform is unavailable.
    """
    index_tf = sample.get('transform')
    if index_tf is None:
        return None
    sx, sy = grid_to_stage(index_tf, eff_xx, eff_yy)
    return stage_to_chip(sample, sx, sy)


def index_mark_chip_position(sample: dict,
                              grid_xx: int, grid_yy: int,
                              quadrant: str | None = None
                              ) -> tuple[float, float] | None:
    """Return the chip-local position of a specific index mark.

    Useful for displaying 'you are near mark XX=6, YY=9' in chip-local coords,
    or for seeding the chip transform from a known mark observation.
    Returns (cx_mm, cy_mm) or None.
    """
    index_tf = sample.get('transform')
    if index_tf is None:
        return None
    sx, sy = grid_to_stage(index_tf, grid_xx, grid_yy, quadrant=quadrant)
    return stage_to_chip(sample, sx, sy)


def list_samples(users_root: str, user: str) -> list[str]:
    """Return folder names for all samples belonging to user, newest first."""
    base = os.path.join(users_root, user)
    if not os.path.isdir(base):
        return []
    folders = [
        d for d in os.listdir(base)
        if os.path.isfile(_json_path(os.path.join(base, d)))
    ]
    return sorted(folders, reverse=True)


def add_image_to_flake(users_root: str, sample: dict,
                       flake_id: str, frame, magnification: str,
                       exposure_ms: float | None = None) -> str | None:
    """Save *frame* (BGR numpy array) as a PNG in the sample images dir.

    Returns the filename (not full path), or None on failure.
    """
    import cv2
    flake = next((f for f in sample['flakes'] if f['id'] == flake_id), None)
    if flake is None:
        return None
    sdir  = _sample_dir(users_root, sample['user'], sample['folder'])
    ts    = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_str = f'_{exposure_ms:.0f}ms' if exposure_ms is not None else ''
    gx = flake.get('grid_x_eff')
    gy = flake.get('grid_y_eff')
    uv_str = f'_U{gx:+.0f}V{gy:+.0f}' if (gx is not None and gy is not None) else ''
    fname = f"{flake_id}_{magnification}_{ts}{exp_str}{uv_str}.png"
    fpath = os.path.join(_images_dir(sdir), fname)
    import numpy as np
    save = (frame << 4) if frame.dtype == np.uint16 else frame
    cv2.imwrite(fpath, save)
    flake['images'].append({'file': fname, 'mag': magnification, 'type': 'frame'})
    flake['updated_at'] = datetime.now().isoformat(timespec='seconds')
    save_sample(users_root, sample)
    return fname


def register_image_for_flake(users_root: str, sample: dict,
                              flake_id: str, filename: str,
                              magnification: str,
                              image_type: str = 'frame') -> bool:
    """Register an already-saved image file with a flake (no copy performed).

    Use this when the image has already been written to the images directory
    by other means (e.g. a QPixmap.save() call for an annotated view).
    Returns True on success.
    """
    flake = next((f for f in sample['flakes'] if f['id'] == flake_id), None)
    if flake is None:
        return False
    flake['images'].append({'file': filename, 'mag': magnification,
                             'type': image_type})
    flake['updated_at'] = datetime.now().isoformat(timespec='seconds')
    save_sample(users_root, sample)
    return True


def images_dir_for_sample(users_root, sample: dict) -> str:
    """Return the absolute path to the sample's images directory."""
    return _images_dir(_sample_dir(users_root, sample['user'], sample['folder']))


def scans_dir_for_sample(users_root, sample: dict) -> str:
    """Return the absolute path to the sample's scans directory (area scan output)."""
    return _scans_dir(_sample_dir(users_root, sample['user'], sample['folder']))


def timelapse_dir_for_sample(users_root, sample: dict) -> str:
    """Return the absolute path to the sample's timelapse directory."""
    return _timelapse_dir(_sample_dir(users_root, sample['user'], sample['folder']))


def frames_dir_for_sample(users_root, sample: dict) -> str:
    """Return the absolute path to the sample's frames directory (individual captures)."""
    return _frames_dir(_sample_dir(users_root, sample['user'], sample['folder']))


def image_path(users_root, sample: dict, filename: str) -> str:
    sdir = _sample_dir(users_root, sample['user'], sample['folder'])
    return os.path.join(_images_dir(sdir), filename)


# ---------------------------------------------------------------------------
# Coordinate transform  (index-mark grid ↔ stage)
# ---------------------------------------------------------------------------

_GRID_SPACING_MM = 0.5   # 500 µm index mark grid pitch

# Each quadrant step offsets the XX/YY coordinates by ±100 grid steps (±50 mm).
# The no-dot first quadrant is the reference (offset 0, 0).
# Intercardinal tokens are tried before cardinal so that e.g. "NW" in "NNW"
# is consumed as one token rather than two separate letters.
_COMPASS_STEPS = {
    'NE': (+100, +100), 'NW': (-100, +100),
    'SE': (+100, -100), 'SW': (-100, -100),
    'N':  (   0, +100), 'S':  (   0, -100),
    'E':  (+100,    0), 'W':  (-100,    0),
}
# Parsing order: 2-letter intercardinals first so they are not split.
_COMPASS_ORDER = ['NE', 'NW', 'SE', 'SW', 'N', 'S', 'E', 'W']


def _quadrant_offset(quadrant: str | None) -> tuple[int, int]:
    """Return (dx, dy) in grid steps for a quadrant string such as 'NW' or 'NNW'.

    Parses the string left-to-right, consuming the longest matching token at
    each position (intercardinals before cardinals).  Each token adds its
    directional step to the running total::

        'N'   → (  0, +100)      one step north
        'NW'  → (−100, +100)     one step north-west  (diagonal)
        'NNW' → (−100, +200)     N + NW
        'WNW' → (−200, +100)     W + NW
    """
    if not quadrant:
        return 0, 0
    dx = dy = 0
    s = quadrant.upper()
    while s:
        for token in _COMPASS_ORDER:
            if s.startswith(token):
                ddx, ddy = _COMPASS_STEPS[token]
                dx += ddx
                dy += ddy
                s = s[len(token):]
                break
        else:
            s = s[1:]   # skip unrecognised character
    return dx, dy


def compute_transform(ref1: dict, ref2: dict,
                      grid_spacing_mm: float = _GRID_SPACING_MM) -> dict:
    """Compute a rigid-body transform from two reference mark observations.

    Each *ref* dict must contain::

        {'grid_xx': int, 'grid_yy': int, 'quadrant': str | None,
         'stage_x_mm': float, 'stage_y_mm': float}

    The ``quadrant`` field (e.g. ``'NW'``, ``'NNE'``, ``None``) encodes the
    repeating-tile offset: each step adds ±100 to the effective XX or YY.

    Returns a transform dict that can be stored as ``sample['transform']``.
    Raises ``ValueError`` if the two marks occupy the same effective position.
    """
    def _eff(ref):
        qdx, qdy = _quadrant_offset(ref.get('quadrant'))
        return ref['grid_xx'] + qdx, ref['grid_yy'] + qdy

    exx1, eyy1 = _eff(ref1)
    exx2, eyy2 = _eff(ref2)
    dx_grid  =  (exx2 - exx1) * grid_spacing_mm
    # Y motor has invert=-1: moving North (increasing chip YY) decreases stage Y.
    # Negate dy_grid so it matches the sign convention of dy_stage.
    dy_grid  = -(eyy2 - eyy1) * grid_spacing_mm
    dx_stage = ref2['stage_x_mm'] - ref1['stage_x_mm']
    dy_stage = ref2['stage_y_mm'] - ref1['stage_y_mm']

    grid_len  = math.hypot(dx_grid,  dy_grid)
    stage_len = math.hypot(dx_stage, dy_stage)
    if grid_len < 1e-9:
        raise ValueError("Reference marks must be at different grid positions.")

    # Rotation: angle of stage displacement minus angle of grid displacement
    theta = math.atan2(dy_stage, dx_stage) - math.atan2(dy_grid, dx_grid)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # Scale (should be ~1.0 if calibration is correct)
    scale = stage_len / grid_len

    # Stage position of grid origin (0, 0):  origin = stage1 - R @ grid1
    gx1 =  exx1 * grid_spacing_mm
    gy1 = -eyy1 * grid_spacing_mm   # physical Y = -chip_YY (Y motor inverted)
    tx = ref1['stage_x_mm'] - (gx1 * cos_t - gy1 * sin_t)
    ty = ref1['stage_y_mm'] - (gx1 * sin_t + gy1 * cos_t)

    return {
        'ref_marks':        [ref1, ref2],
        'rotation_deg':     math.degrees(theta),
        'scale':            scale,
        'grid_spacing_mm':  grid_spacing_mm,
        'origin_stage_x_mm': tx,
        'origin_stage_y_mm': ty,
    }


def grid_to_stage(transform: dict,
                  grid_xx: float, grid_yy: float,
                  quadrant: str | None = None,
                  r_current_deg: float | None = None) -> tuple[float, float]:
    """Convert index-mark coordinates to stage XY in mm.

    Parameters
    ----------
    grid_xx, grid_yy : float
        Mark coordinates within the local quadrant (0–99).
    quadrant : str or None
        Quadrant indicator from the cross dot markings, e.g. ``'NW'``,
        ``'NNE'``, or ``None`` for the no-dot first quadrant.
        Each letter token adds ±100 effective grid steps (±50 mm).
    r_current_deg : float or None
        Current R-axis position in degrees.  If provided, the transform
        rotation is adjusted by the delta from ``transform['r_deg_at_calibration']``
        so the result is valid at the current stage rotation.
    """
    qdx, qdy = _quadrant_offset(quadrant)
    eff_xx = grid_xx + qdx
    eff_yy = grid_yy + qdy
    base_rotation = transform['rotation_deg']
    if r_current_deg is not None:
        r_cal = transform.get('r_deg_at_calibration', 0.0)
        base_rotation += r_current_deg - r_cal
    theta = math.radians(base_rotation)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    gs    = transform['grid_spacing_mm']
    tx    = transform['origin_stage_x_mm']
    ty    = transform['origin_stage_y_mm']
    gx =  eff_xx * gs
    gy = -eff_yy * gs   # physical Y = -chip_YY (Y motor inverted)
    return (gx * cos_t - gy * sin_t + tx,
            gx * sin_t + gy * cos_t + ty)


def stage_to_grid(transform: dict,
                  stage_x_mm: float, stage_y_mm: float) -> tuple[float, float]:
    """Convert stage XY (mm) to effective grid coordinates (fractional).

    Returns the effective (xx, yy) including quadrant offset — divide by 100
    and take the remainder to get the local XX/YY, quotient gives the
    quadrant step count in each axis.
    """
    theta = math.radians(transform['rotation_deg'])
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    gs    = transform['grid_spacing_mm']
    dx = stage_x_mm - transform['origin_stage_x_mm']
    dy = stage_y_mm - transform['origin_stage_y_mm']
    # Inverse rotation (transpose of R); negate gy to convert physical Y back to chip YY
    gx =  ( dx * cos_t + dy * sin_t) / gs
    gy = -(-dx * sin_t + dy * cos_t) / gs
    return gx, gy

# TODO — nikon-prior

Created: 2026-03-13

Imaging and stage control software for a Nikon microscope with a
Prior Scientific ProScan III XYZ motorised stage.

---

## Immediate setup tasks

- [ ] **Verify ProScan III serial connection** — run `python main.py`
      with the stage connected.  If it falls back to mock motors, check:
      - USB/RS232 adapter is recognised (`ls /dev/ttyUSB*` or `ls /dev/ttyACM*`)
      - Baud rate matches the controller setting (default 9600)
      - User is in the `dialout` group: `sudo usermod -aG dialout $USER`

- [ ] **Verify ProScan III command syntax** — the driver in
      `motors/prior_motor_manager.py` uses the standard ProScan III
      ASCII protocol (`G x,y,z` / `GR dx,dy,dz` / `P` / `VS n`).
      Check against your firmware version; some older firmware uses
      `GXY x y` (space-separated) or separate `T z` for Z axis.
      Edit `_move_absolute_counts` / `_move_relative_counts` if needed.

- [ ] **Calibrate step size** — `motors/step_config.json` defaults to
      0.1 µm/count (Prior standard).  Verify by commanding a 1 mm move
      and measuring with a stage micrometer.  Also check invert flags
      so that positive X/Y move in the expected directions.

- [ ] **Calibrate stage travel limits** — update `min_mm` / `max_mm`
      in `step_config.json` to match the actual stage travel (varies
      between Prior H101, H117, etc.)

- [ ] **Calibrate scale bar** — `ui/preview.py` contains a lookup table
      of pixels-per-mm per magnification per resolution.  The values
      from the stacker will be wrong for this camera/tube lens
      combination.  Measure with a stage micrometer graticule and
      update the `SCALE_BAR_TABLE` entries.

- [ ] **Set exposure presets** — use the imaging controls to find good
      exposure/gain values per objective, then save them as presets in
      `focus_presets.json`.

- [ ] **Set focus presets** — use the focus panel to store the Z
      height for each objective and save to `focus_presets.json`.

---

## Porting from standa-stacker

Active workstream on branch `claude/standa-stacker-nikon-port-ooga7z`.

**Strategy — batch cherry-pick + capability flags.** Keep nikon-prior as the
app with its own clean Prior motor API (mm-based, combined XY). Port standa's
hardware-agnostic subsystems in batches; absorb the rig differences with
*shims on the Nikon side* so standa modules run near-verbatim, rather than
emulating Standa's slider/calb/invert motor conventions (the danger zone) or
hand-editing each module. Manipulator / rotation / heater features are gated
off via stage `capabilities`, not ported (that hardware doesn't exist here).

**Capability flags.** Both motor managers expose
`capabilities = {manipulator, rotation, heater}` (all `False`). Ported standa
code queries these to gate absent-hardware features.

**Seam shims (Nikon side) — extend as more modules land:**
- `ui/__init__.py` → `APP_DIR` (stable paths regardless of CWD).
- motor managers → `get_position_units_cached(axis)` (returns `None` for axes
  the rig lacks, e.g. `R`, so rotation polling degrades to 0°).
- `StageControlWindow` → `set_flake_markers` / `update_edge_positions` /
  `clear_edge_positions` + `_on_extents_updated` hook; renders flake markers +
  wafer-extents polygon on the stage minimap.

### Shipped (this port)
- [x] Three-column `AppShell` (SCOPE | preview | PROCESS accordion).
- [x] Preview fills the central pane (no centering wrapper).
- [x] Keyboard: in-preview jog + app-level `GlobalKeyRouter` (arrows pan FOV,
      numpad XY jog, numpad±/PageUp-Down Z, Enter autofocus, X cycle mag,
      Escape panic-stop); mouse-wheel Z focus; spinbox wheel-guard.
- [x] Launcher hardening (`scripts/launch.sh`: BLAS/OpenMP thread caps +
      rolling `crash_logs/launcher.log`).
- [x] Zoom View close now deactivates measure / centre-zoom / zoom-under-cursor
      (was inconsistent with unchecking the box).
- [x] Backbone: `core/sample_data.py`, `core/logbook.py`,
      `vision/registration.py` + capability flags.
- [x] `SampleManagerPanel` (Users → Samples → Flakes catalogue) in PROCESS.
- [x] Live BGR histogram (#40) under Imaging.

### Rig verification needed (do at the scope)
- [ ] **Capture & Add** and **Navigate** command real stage moves — confirm
      they land on the right feature on an actual chip.
- [ ] Flake markers on the stage minimap sit at the right positions.
- [ ] Minimap click-to-move and arrow-key FOV pan go the right direction/distance.
- [ ] Keyboard jog step sizes feel right (XY µm/press in `ui/preview.py`
      `_KB_STAGE_UM`; Z uses the mag-scaled focus step).
- [ ] Escape panic-stop actually halts a long ProScan III move (`motor.stop()`).
- [ ] (See also "Immediate setup tasks": step size, scale bar, presets.)

### Z focus recall — resolved: no Z recall, autofocus instead
**Root cause (rig):** the Nikon coarse-focus knob is *manual and unsensed*, so
the ProScan Z counter is decoupled from actual focus. A saved absolute Z is
therefore meaningless for focus — and commanding it is unsafe (could drive the
objective into the sample). So "Z recall" does not make sense on this rig.

**Done:** catalogue Navigate now restores **XY only** (single combined,
non-blocking move) and never commands a saved Z. The "(rel)" Z label is correct
and stays. Z control remains relative (velocity slider + jog), which is right
for a manual-knob focus.

**Remaining (optional):** autofocus is the only sensible focus mechanism, and
only when the stage is already *roughly* in focus (within capture range).
- [ ] Add an opt-in "Autofocus after Navigate" (checkbox/button) that runs
      `AutoFocusPanel` once the XY move settles — off by default (running it
      wildly out of focus just wastes a sweep).
- [ ] Consider the same XY-only treatment for `PositionManagerWindow` recall
      (line ~628 still commands absolute Z; fine within a session, stale after
      the knob moves).

### Pending batches (agnostic — keep pulling)
- [ ] **Flake detection** — `vision/flake_detect_v3.py`, `flake_classify.py`,
      `optical_contrast.py`, `camera_params.py` + `ui/flake_candidates_panel.py`.
      Completes the catalogue's "Automatic Flake Detection" hook
      (`SampleManagerPanel.import_detected_candidates` already present).
      ~2200 lines of CV — **verify by looking at real chip images** (debug
      vision by looking, not by score), so do it as a rig session.
- [ ] **Corner-finding Find Wafer Extents** — replace the deprecated edge-walk
      with "move to each expected corner → detect the corner in-frame", feeding
      `vision/registration.compute_chip_transform`'s corner list. The button
      currently just opens the existing edge-detection panel.
- [ ] **Registration panel** (corner-based chip transform + extents) — couples
      to the ghost overlay (manual align) and ORB matching vs reference scans;
      needs the ghost subsystem ported or the capture flow reworked first.
      Overlaps with corner-finding above.
- [ ] **Wafer map browser** — `ui/map_nav_server.py` + `tools/make_map.py`
      (this repo's Planned feature #3): HTML zoomable/pannable map from scan data.
- [ ] **Logbook event wiring** — `core/logbook.py` is ported but not yet emitting
      events; wire sample-open / flake-add / scan / navigate.
- [ ] **Window-geometry persistence** — standa's `ui/window_geometry.py` saves
      window + splitter sizes across runs; not ported.
- [ ] **App-wide button icons (#57)** — bundled Lucide SVGs + `_btn_icon` helper.

### Deferred / not applicable (no such hardware on this rig)
- Index marks (skipped for now, by request) — `index_mark_panel`, watcher.
- Manipulator-dependent: ghost / reference overlay, slide lock, lateral
  wiggler, slow dZ approach, virtual-annotations slide layer, compound moves.
- Rotation-dependent: eucentric calibration, R-rotation overlays.
- Heater / recipe Heat step / load-cell contact sensing.
  (All gated via `capabilities`; revisit only if the hardware is added.)

---

## Planned features

### 1. Slow Z approach  *(safety-critical)*
Prior ProScan III Z axis can be speed-controlled via `VS n`.  Add a
`SlowApproachPanel` with velocity selector and live metric monitor
for safe surface approach.  Same design as planned for standa-stacker.

### 2. Per-magnification flat-field
`flat_field_panel.py` works; needs auto-apply at capture time keyed
to the current objective.

### 3. Full wafer map browser
Interactive zoomable/pannable viewer on top of the existing wafer
mapping output (Qt `QGraphicsView` + `QGraphicsScene`, tile-based).

### 4. Fix mark detection
Replace Tesseract OCR in `index_mark_panel.py` with a purely
CV-based cross/digit detector for robustness on low-contrast images.

### 5. ProScan III joystick passthrough
Prior controllers have a built-in joystick port.  Could optionally
disable the hardware joystick (send `J 0`) when the software gamepad
is active, and re-enable it on exit.

---

## Known issues / inherited from standa-stacker

- Measurement tool label offset (~20% right displacement)
- Preview overlay scaling with resized windows
- No unit tests; mock manager enables offline development

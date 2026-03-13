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

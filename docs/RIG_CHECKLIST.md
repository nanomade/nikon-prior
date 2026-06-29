# Nikon-Prior rig verification checklist

Run at the scope after pulling branch `claude/standa-stacker-nikon-port-ooga7z`.
Tests the standa→Nikon port against real hardware. Work top-to-bottom — later
sections assume earlier ones pass. Note anything off in the "Notes" lines;
those become fix tickets.

> Safety: keep a hand near the controller. If anything moves unexpectedly,
> press **Escape** (panic stop) or kill the window.

```
git pull origin claude/standa-stacker-nikon-port-ooga7z
python main.py        # or scripts/launch.sh
```

---

## 0. Launch
- [ ] App opens as the three-column shell (SCOPE | preview | PROCESS).
- [ ] **Real ProScan III connected** (log says "ProScan III connected on …",
      not "using mock motors"). If mock, see TODO "Immediate setup tasks".
- [ ] **Real Alvium camera** live in the preview (not the synthetic mock).
- [ ] Preview fills the central pane and resizes with the window.

Notes:

## 1. Stage basics (XYZ)
- [ ] Readout (Motion Controls → Stage) shows live X/Y/Z and tracks the joystick.
- [ ] Minimap **click-to-move**: click a point → stage goes there; green dot
      lands where you clicked (right direction, right distance).
- [ ] Jog pad ↑↓←→ and Z± move the expected way at ×1/×10/×100 steps.
- [ ] Goto X/Y/Z entry + Go drives to the typed mm position.
- [ ] "Stage Controls…" opens the full window; its map agrees with the minimap.

Notes (esp. any inverted axis or wrong distance):

## 2. Keyboard (cursor over the preview; also works with focus elsewhere)
- [ ] **Arrows** pan ~90% of the field of view in the correct direction.
- [ ] **Numpad 1–9** jog XY 8-ways; **Shift** = 1 µm fine, **Ctrl** = 5× turbo.
- [ ] **Numpad +/−** and **PageUp/PageDown** move Z focus (mag-scaled step).
- [ ] **Enter** triggers autofocus.
- [ ] **X** cycles magnification (and the scale bar / ppm update with it).
- [ ] **Escape** halts a long move immediately (start a big goto, then Escape).
- [ ] Typing in a text field does NOT jog the stage (router is suppressed).

Notes (step sizes too coarse/fine? wrong pan direction?):

## 3. Mouse
- [ ] **Wheel** over the preview moves Z focus; Shift = 1 µm fine.
- [ ] **Double-click** a feature → it centres in the frame (this was "bang on"
      before — confirm still true).
- [ ] Wheel over a side-panel spin box scrolls the panel, doesn't change the value.
- [ ] Measure tool: enable, click two points, read distance/angle.
- [ ] **Zoom View close consistency**: enable Measure (or Show Centre Zoom, or
      Zoom Under Cursor) → Zoom View opens → click its **X** → the matching
      checkbox unticks and the window stays closed (doesn't pop back).

Notes:

## 4. Sample catalogue  ← most important (foundation for later batches)
- [ ] Add a user (PROCESS → Sample → User → Add…).
- [ ] Create a sample; pick a substrate.
- [ ] **Capture & Add** a flake at the current position → row appears in the
      Flake Catalogue (Find section) with a thumbnail.
- [ ] A **marker appears on the stage minimap** at the flake's position.
- [ ] Move the stage away, then **Navigate** to that flake → stage returns to
      the same spot, flake re-centred (the round-trip — verify it lands).
- [ ] Add 2–3 flakes; markers + names all sit in the right places.
- [ ] Close and re-open the sample → flakes + markers restore from JSON.

Notes (Navigate accuracy in µm? marker offset?):

## 5. Histogram
- [ ] Imaging → Histogram rollout: expand → live B/G/R curves update.
- [ ] Means/clip readout reflect exposure changes; CLIP flags at the top end.
- [ ] Collapse the rollout → it stops updating (no perf cost when hidden).

Notes:

## 6. Calibration (from "Immediate setup tasks" — do once, persist)
- [ ] Step size: command a 1 mm move, measure with a stage micrometer; fix
      `motors/step_config.json` `step` / `invert` if off.
- [ ] Travel limits: set `min_mm` / `max_mm` to the actual stage range.
- [ ] Scale bar / ppm: verify against a graticule; update the table in
      `ui/preview.py` (`calibration_table`).
- [ ] Exposure + focus presets per objective saved.

Notes:

---

## Result
- Blocking issues (must fix before more batches):
- Tuning requests (step sizes, directions, thresholds):
- Ready for next batch? (flake detection needs real images on hand)
```

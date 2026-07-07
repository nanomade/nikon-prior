"""
Flake detection results viewer (v3) — supersedes FlakeCandidatesPanel (v2)
per plan U2.

Views a scan folder's flake_candidates_v3.json (written by the Survey
pipeline / vision/flake_detect_v3.py). Deliberately NO in-process detection
and NO second import implementation: detection runs in the pipeline, and
"Add checked to Catalogue" shells out to the ONE import path
(tools/import_found_flakes.py --ids via core.pipeline_cmds.import_cmd).

Capture Tour (manual-first, rig feedback 2026-07-03): 'Start Tour' walks
the checked candidates navigation-only — ◀ Previous / Next ▶ move the stage,
the user focuses by hand, 'Capture' saves a map tile for the current stop,
and 'End Tour' finalises the metadata (partial captures fine).  An
'Automatic' checkbox restores the machine-driven flow (advance → robust
arrival detect → optional autofocus → capture → next).  Tour captures are
just sparse high-mag map tiles: they land in the sample's standard scan
hierarchy ({scans_dir}/{mag}/{name}_{mag}_tour_{ts}/ with standard tile
names + scan_metadata.json) so make_map.py layers them onto the mother map
at the correct scale automatically.
"""
import json
from pathlib import Path

from core.sample_data import placement_dz_mm

from PyQt5.QtCore import QObject, QProcess, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QProgressBar,
    QPushButton, QSizePolicy, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

_CANDIDATES_FILE = 'flake_candidates_v3.json'

# Okabe-Ito blue #0072B2 in BGR — contour outline on the crop preview
# (protanopia rule: never red/green).
_CONTOUR_BGR = (178, 114, 0)
_CROP_CACHE_MAX = 20     # last-N crop pixmaps kept (keyed by candidate id)


def _ro(text):
    it = QTableWidgetItem(str(text) if text is not None else '')
    it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
    return it


class _SortItem(QTableWidgetItem):
    """Read-only item displaying `text` but sorting by a numeric key
    (e.g. "2 ±0.30" sorts by 2, so "17 ±…" doesn't sort between 1 and 2)."""

    def __init__(self, text, sort_key):
        super().__init__(str(text))
        self._key = float(sort_key)
        self.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)

    def __lt__(self, other):
        if isinstance(other, _SortItem):
            return self._key < other._key
        return super().__lt__(other)


_POLL_MS           = 150    # arrival poll interval
_ARRIVE_TOL_MM     = 0.002  # arrived when X AND Y within 2 µm of target…
_STILL_TOL_MM      = 0.0002 # …AND change < 0.2 µm between consecutive polls…
_STILL_POLLS       = 3      # …for this many consecutive polls
_ARRIVE_TIMEOUT_MS = 20000  # give up waiting; pause — NEVER capture blind
_DWELL_MS          = 200    # vibration damp after arrival before capture
_AF_TIMEOUT_MS     = 30000  # autofocus completion fallback (per stop)


class _TourRunner(QObject):
    """
    QTimer-based tour state machine over the checked candidates.

    Manual mode (default, rig feedback 2026-07-03 — 'automatic image taking
    annoying without flow control'): navigation only.  next_stop() /
    prev_stop() move the stage, the user focuses by hand, capture_now()
    saves the shot for the current stop (replace-by-candidate-id on
    re-capture), end_tour() writes the manifest (partial captures fine).

    Automatic mode: advance → robust arrival detect → optional autofocus
    (AutoFocusPanel.run_programmatic + focus_finished/focus_aborted, same
    pattern as the recipe runner; 30 s fallback) → capture → next.

    Arrival detection (the rig bug: hardware moves are non-blocking and the
    old is_moving_cached poll fired early, so autofocus/capture ran
    mid-travel): poll get_position_units_cached for X and Y every 150 ms and
    require BOTH (a) within 2 µm of the commanded target and (b) change
    < 0.2 µm across 3 consecutive polls.  20 s timeout → status reports
    "arrival timeout" and the tour pauses without capturing.  Manual
    next/prev need no gate (the user is watching) but the same cheap poll
    updates the status line with "settled".

    Stops with a truthy z_mm also command Z = z_mm + dz_mm (the mount's
    rigid-body placement shift) — scan-era focus heights commanded verbatim
    after a remount caused 'complete defocussing'.

    Captures are just sparse high-mag map tiles: each lands in the sample's
    standard scan hierarchy ({scans_dir}/{mag}/{name}_{mag}_tour_{ts}/) with
    standard tile names + a full scan_metadata.json, grouped per magnification
    (change objective mid-tour → one folder per mag, shared timestamp), so
    make_map.py layers them onto the mother map at the correct scale.  Metadata
    is rewritten on every capture, so a crash never loses recorded tiles.
    """
    progress     = pyqtSignal(int, int, str)   # current, total, message
    stop_changed = pyqtSignal(int, int)        # 0-based stop idx, total
    finished     = pyqtSignal(str)             # output folder path

    def __init__(self, candidates, scans_dir, sample_name, tour_ts, mm, preview,
                 autofocus_panel=None, automatic=False, autofocus=False,
                 dz_mm=0.0, parent=None):
        super().__init__(parent)
        self._candidates = list(candidates)
        self._scans_dir = Path(scans_dir)
        self._sample_name = sample_name or 'sample'
        self._tour_ts = tour_ts
        self._mm = mm
        self._preview = preview
        self._af_panel = autofocus_panel
        self._automatic = bool(automatic)
        self._autofocus = (bool(autofocus) and self._automatic
                           and autofocus_panel is not None)
        self._dz_mm = float(dz_mm or 0.0)
        self._idx = 0
        # per-mag capture folders: mag -> {'dir', 'by_cid': {cid: entry}, 'w', 'h'}
        self._folders = {}
        self._captured_cids = set()
        self._move_start_ms = 0
        self._target = (0.0, 0.0)   # commanded (x_mm, y_mm) of current stop
        self._last_pos = None       # previous poll's (x, y) for stillness test
        self._still_count = 0       # consecutive polls with change < 0.2 µm
        self._paused = False        # automatic mode only (arrival timeout)
        self._af_waiting = False
        # arrival poll: fires every _POLL_MS until arrived or timed out
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(_POLL_MS)
        self._poll_timer.timeout.connect(self._poll_arrival)
        # dwell timer: fires once after arrival (automatic mode)
        self._dwell_timer = QTimer(self)
        self._dwell_timer.setSingleShot(True)
        self._dwell_timer.timeout.connect(self._on_dwell_done)
        # autofocus fallback: capture anyway if focus never reports back
        self._af_timer = QTimer(self)
        self._af_timer.setSingleShot(True)
        self._af_timer.setInterval(_AF_TIMEOUT_MS)
        self._af_timer.timeout.connect(self._on_af_settled)

    @property
    def is_paused(self):
        return self._paused

    @property
    def automatic(self):
        return self._automatic

    @property
    def manual_controls_active(self):
        """True when Previous / Next / Capture should respond (manual mode,
        or an automatic tour paused on arrival timeout)."""
        return (not self._automatic) or self._paused

    def start(self):
        self._idx = 0
        self._navigate_current()

    @property
    def captured_count(self):
        return len(self._captured_cids)

    @property
    def captured_mags(self):
        return sorted(self._folders.keys())

    def end_tour(self):
        """Finish (or abort) the tour: write the manifest with whatever was
        captured so far — partial captures are fine."""
        self._poll_timer.stop()
        self._dwell_timer.stop()
        self._af_timer.stop()
        self._paused = False
        if self._af_waiting:
            self._af_waiting = False
            self._disconnect_af()
        self._finalise_metadata()
        self.finished.emit(str(self._scans_dir))

    # ── Manual controls (manual mode, or automatic paused on timeout) ───────

    def next_stop(self):
        if not self.manual_controls_active:
            return
        if self._idx + 1 >= len(self._candidates):
            return
        self._paused = False
        self._idx += 1
        self._navigate_current()

    def prev_stop(self):
        if not self.manual_controls_active or self._idx == 0:
            return
        self._paused = False
        self._idx -= 1
        self._navigate_current()

    def capture_now(self):
        """Save the map tile for the current stop (user has focused)."""
        if not self.manual_controls_active:
            return
        self._capture_current()
        n = len(self._candidates)
        self.progress.emit(self._idx + 1, n, self._stop_status())

    # ── Navigation + arrival detection ───────────────────────────────────────

    def _current_cid(self):
        return self._candidates[self._idx].get('id', f'{self._idx:04d}')

    def _stop_status(self, extra=''):
        """'Stop 3/7 — F-id — captured ✓/not captured' (+ optional suffix)."""
        n = len(self._candidates)
        cid = self._current_cid()
        cap = ('captured ✓' if cid in self._captured_cids
               else 'not captured')
        return f"Stop {self._idx + 1}/{n} — {str(cid)[:12]} — {cap}{extra}"

    def _navigate_current(self):
        import time
        c = self._candidates[self._idx]
        n = len(self._candidates)
        self._mm.move_absolute_units('X', c['x_mm'])
        self._mm.move_absolute_units('Y', c['y_mm'])
        z = c.get('z_mm')
        if z:   # scan-era focus + rigid-body mount shift; skip if no real focus
            self._mm.move_absolute_units('Z', z + self._dz_mm)
        self._target = (float(c['x_mm']), float(c['y_mm']))
        self._last_pos = None
        self._still_count = 0
        self._move_start_ms = int(time.monotonic() * 1000)
        if self._automatic:
            self.progress.emit(self._idx, n,
                f"({self._idx + 1}/{n}) Moving to {str(c.get('id', '?'))[:12]}…")
        else:
            self.progress.emit(self._idx + 1, n,
                               self._stop_status(' — moving…'))
        self.stop_changed.emit(self._idx, n)
        self._poll_timer.start()

    def _poll_arrival(self):
        import time
        elapsed = int(time.monotonic() * 1000) - self._move_start_ms
        x = y = None
        try:
            x = self._mm.get_position_units_cached('X')
            y = self._mm.get_position_units_cached('Y')
        except Exception:
            pass
        if x is not None and y is not None:
            if self._last_pos is not None:
                lx, ly = self._last_pos
                if abs(x - lx) < _STILL_TOL_MM and abs(y - ly) < _STILL_TOL_MM:
                    self._still_count += 1
                else:
                    self._still_count = 0
            self._last_pos = (x, y)
            tx, ty = self._target
            on_target = (abs(x - tx) <= _ARRIVE_TOL_MM
                         and abs(y - ty) <= _ARRIVE_TOL_MM)
            if on_target and self._still_count >= _STILL_POLLS:
                self._poll_timer.stop()
                self._on_arrived()
                return
        if elapsed >= _ARRIVE_TIMEOUT_MS:
            self._poll_timer.stop()
            self._on_arrival_timeout()

    def _on_arrived(self):
        if self._automatic:
            self._dwell_timer.start(_DWELL_MS)
        else:
            # No gate needed in manual mode — but the settle info is cheap.
            n = len(self._candidates)
            self.progress.emit(self._idx + 1, n,
                               self._stop_status(' — settled'))

    def _on_arrival_timeout(self):
        n = len(self._candidates)
        if self._automatic:
            # Pause — never capture blind mid-travel.
            self._paused = True
            self.progress.emit(self._idx, n, self._stop_status(
                ' — arrival timeout — paused (not captured)'))
            self.stop_changed.emit(self._idx, n)
        # Manual mode: the user is watching; stop polling quietly.

    # ── Automatic mode: after arrival — optional autofocus, then capture ────

    def _on_dwell_done(self):
        if self._autofocus:
            self._start_autofocus()
        else:
            self._capture_current()
            self._advance_auto()

    def _start_autofocus(self):
        c = self._candidates[self._idx]
        n = len(self._candidates)
        self.progress.emit(self._idx, n,
            f"({self._idx + 1}/{n}) Autofocus at {c.get('id', '?')[:12]}…")
        self._af_waiting = True
        self._af_panel.focus_finished.connect(self._on_af_finished)
        self._af_panel.focus_aborted.connect(self._on_af_aborted)
        self._af_timer.start()
        # Size a centred metric ROI from the candidate: an isolated
        # monolayer's few soft edges drown in whole-frame noise, so grade
        # only around the (centred, just-navigated-to) flake.
        roi = None
        try:
            from vision.camera_params import px_per_um
            area = float(c.get('area_um2') or 0)
            frame = self._preview.get_frame()
            if area > 0 and frame is not None:
                h, w = frame.shape[:2]
                mag = getattr(self._preview, 'magnification', None) or '20x'
                ppu = px_per_um(mag, w)
                if ppu:
                    side_px = (area ** 0.5) * 1.8 * ppu + 120
                    roi = min(0.6, max(0.15, side_px / w))
        except Exception:
            roi = None
        self._af_panel.run_programmatic(roi_frac=roi)

    def _on_af_finished(self, _z_best, _m_best):
        self._on_af_settled()

    def _on_af_aborted(self, _reason):
        self._on_af_settled()

    def _on_af_settled(self):
        """Common landing point for focus_finished / focus_aborted / timeout."""
        if not self._af_waiting:
            return
        self._af_waiting = False
        self._af_timer.stop()
        self._disconnect_af()
        self._capture_current()
        self._advance_auto()

    def _disconnect_af(self):
        try:
            self._af_panel.focus_finished.disconnect(self._on_af_finished)
            self._af_panel.focus_aborted.disconnect(self._on_af_aborted)
        except TypeError:
            pass   # already disconnected

    def _flat_field_applied(self):
        """Whether the live preview flat-fields frames (get_clean_frame does
        when the panel is active) — recorded honestly so make_map never
        double-corrects."""
        ffp = getattr(self._preview, 'flat_field_panel', None)
        return bool(ffp is not None
                    and getattr(getattr(ffp, '_enable_chk', None),
                                'isChecked', lambda: False)()
                    and getattr(ffp, '_flat', None) is not None)

    def _folder_for_mag(self, mag, w, h):
        f = self._folders.get(mag)
        if f is None:
            d = (self._scans_dir / mag /
                 f"{self._sample_name}_{mag}_tour_{self._tour_ts}")
            d.mkdir(parents=True, exist_ok=True)
            f = {'dir': d, 'by_cid': {}, 'w': int(w), 'h': int(h)}
            self._folders[mag] = f
        return f

    def _capture_current(self):
        from datetime import datetime
        cid = self._current_cid()
        c = self._candidates[self._idx]
        frame = self._preview.get_frame()
        if frame is None:
            return
        try:
            import cv2
            import numpy as np
            if frame.dtype != np.uint8:
                frame = (frame >> 4).astype(np.uint8)
            h, w = frame.shape[:2]
            mag = getattr(self._preview, 'magnification', None) or '20x'
            # Recorded stage position (cache is fine — recording, not commanding).
            def _pos(ax, fallback):
                try:
                    v = self._mm.get_position_units_cached(ax)
                    return float(v) if v is not None else float(fallback)
                except Exception:
                    return float(fallback)
            x = _pos('X', c['x_mm'])
            y = _pos('Y', c['y_mm'])
            z = _pos('Z', 0.0)
            folder = self._folder_for_mag(mag, w, h)
            fname = f"img_X{x:+.3f}_Y{y:+.3f}_Z{z:.4f}_{mag}.png"
            # Re-capture of the same stop replaces its tile (delete stale file
            # if the position — hence the name — changed).
            prev = folder['by_cid'].get(cid)
            if prev and prev['filename'] != fname:
                try:
                    (folder['dir'] / prev['filename']).unlink()
                except OSError:
                    pass
            cv2.imwrite(str(folder['dir'] / fname), frame)
            folder['by_cid'][cid] = {
                'filename': fname,
                'x_mm': x, 'y_mm': y,
                'z_actual_mm': z, 'z_plane_mm': z,
                'focus_ok': True,
                'timestamp': datetime.now().isoformat(timespec='seconds'),
            }
            self._captured_cids.add(cid)
            self._write_metadata(folder, mag)
        except Exception:
            pass

    def _advance_auto(self):
        """Automatic mode: move on after a capture (or finish)."""
        n = len(self._candidates)
        self._idx += 1
        self.progress.emit(self._idx, n,
            f"({self._idx}/{n}) Captured — {self._idx} done, "
            f"{n - self._idx} remaining")
        if self._idx < n:
            self._navigate_current()
        else:
            self._idx = n - 1   # keep index valid for status helpers
            self._finalise_metadata()
            self.finished.emit(str(self._scans_dir))

    def _write_metadata(self, folder, mag):
        """Write a full scan_metadata.json for one per-mag folder — the exact
        schema make_map/load_scan consume (frame_width/height from the ACTUAL
        captured frame, so the tiles place at the right scale)."""
        imgs = list(folder['by_cid'].values())
        ff = self._flat_field_applied()
        meta = {
            'imaging': {
                'magnification': mag,
                'frame_width': folder['w'],
                'frame_height': folder['h'],
            },
            'scan_params': {
                'mag': mag,
                'sample_name': self._sample_name,
                'flat_field_applied': ff,
            },
            'grid': {'nx': 0, 'ny': 0, 'n_total': len(imgs)},
            'status': 'finished',
            'total_images': len(imgs),
            'flat_field_applied': ff,
            'images': imgs,
            'source': 'tour',
        }
        (folder['dir'] / 'scan_metadata.json').write_text(
            json.dumps(meta, indent=2))

    def _finalise_metadata(self):
        for mag, folder in self._folders.items():
            self._write_metadata(folder, mag)
        total = sum(len(f['by_cid']) for f in self._folders.values())
        print(f"[tour] {total} tile(s) in {len(self._folders)} mag layer(s) "
              f"→ {self._scans_dir}")


class FlakeResultsPanel(QWidget):
    """Floating panel: view v3 detection results for a scan, tour candidates.

    View-only by design (plan D1 lesson): the Survey pipeline runs the
    detector and owns import; this panel just reads flake_candidates_v3.json.
    """

    # table column indices
    _COL_CHK, _COL_ID, _COL_N, _COL_RES, _COL_AREA, _COL_SCORE, \
        _COL_X, _COL_Y, _COL_FOCUS = range(9)

    def __init__(self, sample_manager=None, motor_manager=None, preview=None,
                 autofocus_panel=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Flake Results (v3)")
        self.setMinimumSize(720, 640)
        self._sample_manager = sample_manager
        self._motor_manager = motor_manager
        self._preview = preview
        self._autofocus_panel = autofocus_panel
        self._candidates = []
        self._scan_folder = None     # explicit override via set_scan_folder()
        self._resolved_folder = None
        self._tour_runner = None
        self._crop_cache = {}        # candidate id → QPixmap (cap _CROP_CACHE_MAX)
        self._import_proc = None     # QProcess for 'Add checked to Catalogue'
        self._build_ui()

    def set_sample_manager(self, sm):
        self._sample_manager = sm

    def set_autofocus_panel(self, panel):
        """Wire the AutoFocusPanel after construction (parent session)."""
        self._autofocus_panel = panel
        self._update_af_enabled()

    def set_scan_folder(self, path):
        """Explicitly pin the scan folder to view (overrides auto-resolution)."""
        self._scan_folder = Path(path) if path else None
        self.refresh()

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        # Header row: resolved scan folder + refresh
        head = QHBoxLayout()
        head.addWidget(QLabel("Scan:"))
        self._folder_label = QLabel("—")
        self._folder_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        head.addWidget(self._folder_label)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedWidth(70)
        refresh_btn.setToolTip(
            "Reload results from the newest flake_candidates_v3.json under "
            "the active sample (or the explicitly set scan folder).")
        refresh_btn.clicked.connect(self.refresh)
        hint = QLabel("Double-click a row to navigate")
        hint.setStyleSheet("color: #888;")
        head.addWidget(hint)
        head.addWidget(refresh_btn)
        root.addLayout(head)

        # Division of labour: this panel shows raw detector output; the
        # Catalogue is the curated record.
        caption = QLabel(
            "Raw detector output for this scan — add keepers to the "
            "Catalogue (the sample's curated record).")
        caption.setStyleSheet("color: #888; font-size: 11px;")
        caption.setWordWrap(True)
        root.addWidget(caption)

        # Results table
        cols = ["✓", "ID", "N", "Res.", "Area µm²", "Score",
                "x mm", "y mm", "Focus"]
        self._table = QTableWidget(0, len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        self._table.cellDoubleClicked.connect(self._on_double_click)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(self._COL_CHK, QHeaderView.Fixed)
        self._table.setColumnWidth(self._COL_CHK, 30)
        for c in range(1, len(cols)):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        root.addWidget(self._table)

        # Crop preview: candidate cut from its source tile, contour outlined
        self._crop_preview = QLabel("Select a row to preview its crop")
        self._crop_preview.setAlignment(Qt.AlignCenter)
        self._crop_preview.setFixedHeight(220)
        self._crop_preview.setStyleSheet(
            "background: #1c1c1c; color: #888; border: 1px solid #444;")
        root.addWidget(self._crop_preview)

        # Status line
        self._status = QLabel("")
        self._status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        root.addWidget(self._status)
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        # Bottom row: select all/none + capture tour
        bot = QHBoxLayout()
        all_btn = QPushButton("All")
        all_btn.setFixedWidth(36)
        all_btn.clicked.connect(lambda: self._set_all(True))
        bot.addWidget(all_btn)
        none_btn = QPushButton("None")
        none_btn.setFixedWidth(44)
        none_btn.clicked.connect(lambda: self._set_all(False))
        bot.addWidget(none_btn)

        self._import_btn = QPushButton("Add checked to Catalogue")
        self._import_btn.setEnabled(False)
        self._import_btn.setToolTip(
            "Import exactly the checked candidates into the sample catalogue "
            "via tools/import_found_flakes.py --ids (the one import path — "
            "dedup + crop cutting included).")
        self._import_btn.clicked.connect(self._add_checked_to_catalogue)
        bot.addWidget(self._import_btn)
        bot.addStretch()

        self._auto_check = QCheckBox("Automatic")
        self._auto_check.setToolTip(
            "Machine-driven tour: advance → wait for arrival → optional "
            "autofocus → capture → next.  Unchecked (default): navigation "
            "only — you focus by hand and press Capture at each stop.")
        self._auto_check.toggled.connect(self._update_af_enabled)
        bot.addWidget(self._auto_check)

        self._af_check = QCheckBox("Autofocus at each stop")
        self._af_check.setToolTip(
            "Run a single-shot autofocus (AutoFocusPanel.run_programmatic) "
            "after arrival, before each capture.  Automatic mode only.")
        bot.addWidget(self._af_check)
        self._update_af_enabled()

        self._prev_btn = QPushButton("◀ Previous")
        self._prev_btn.setEnabled(False)
        self._prev_btn.setToolTip("Move the stage back to the previous stop.")
        self._prev_btn.clicked.connect(self._tour_prev)
        bot.addWidget(self._prev_btn)

        self._next_btn = QPushButton("Next ▶")
        self._next_btn.setEnabled(False)
        self._next_btn.setToolTip("Move the stage to the next stop.")
        self._next_btn.clicked.connect(self._tour_next)
        bot.addWidget(self._next_btn)

        self._capture_btn = QPushButton("Capture")
        self._capture_btn.setEnabled(False)
        self._capture_btn.setToolTip(
            "Save a map tile for the current stop at the live magnification "
            "(re-capturing a stop overwrites its tile).")
        self._capture_btn.clicked.connect(self._tour_capture)
        bot.addWidget(self._capture_btn)

        self._tour_btn = QPushButton("Start Tour")
        self._tour_btn.setEnabled(False)
        self._tour_btn.setToolTip(
            "Walk the checked candidates: Previous/Next move the stage, you "
            "focus by hand and press Capture.  Captures are saved as sparse "
            "high-mag scan tiles under the sample's scans/ — re-run Generate "
            "Map to layer them onto the mother map.  Check 'Automatic' for "
            "the machine-driven flow.")
        self._tour_btn.clicked.connect(self._toggle_tour)
        bot.addWidget(self._tour_btn)
        root.addLayout(bot)

    def _update_af_enabled(self, _checked=False):
        """'Autofocus at each stop' is meaningful only in Automatic mode."""
        self._af_check.setEnabled(
            self._auto_check.isChecked() and self._autofocus_panel is not None)

    # ── Folder resolution + load ────────────────────────────────────────────

    def _resolve_folder(self):
        """Explicit set_scan_folder() wins; else newest flake_candidates_v3.json
        under the active sample's scans dir."""
        if self._scan_folder is not None:
            return self._scan_folder
        if self._sample_manager is None:
            return None
        out_dir, _sample_name = self._sample_manager.scan_output_info()
        if out_dir is None:
            return None
        hits = sorted(
            Path(out_dir).rglob(_CANDIDATES_FILE),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return hits[0].parent if hits else None

    def refresh(self):
        folder = self._resolve_folder()
        self._resolved_folder = folder
        self._crop_cache.clear()
        self._crop_preview.setText("Select a row to preview its crop")
        if folder is None:
            self._folder_label.setText("—")
            self._status.setText(
                "No results — no flake_candidates_v3.json found "
                "(run a Survey detection first).")
            self._candidates = []
            self._populate_table([])
            self._tour_btn.setEnabled(False)
            self._import_btn.setEnabled(False)
            return
        self._folder_label.setText(folder.name)
        cand_path = folder / _CANDIDATES_FILE
        try:
            candidates = json.loads(cand_path.read_text())
        except Exception as exc:
            self._candidates = []
            self._populate_table([])
            self._status.setText(f"Failed to read {cand_path.name}: {exc}")
            self._tour_btn.setEnabled(False)
            self._import_btn.setEnabled(False)
            return
        self._candidates = candidates
        self._populate_table(candidates)
        n_res = sum(1 for c in candidates if c.get('layer_resolvable'))
        self._status.setText(
            f"{len(candidates)} candidate(s)  —  {n_res} layer-resolvable")
        self._tour_btn.setEnabled(
            bool(candidates) and self._motor_manager is not None
            and self._preview is not None)
        self._import_btn.setEnabled(
            bool(candidates) and self._import_proc is None)

    # ── Table ───────────────────────────────────────────────────────────────

    def _populate_table(self, candidates):
        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)
        for i, c in enumerate(candidates):
            row = self._table.rowCount()
            self._table.insertRow(row)

            chk = QCheckBox()
            chk.setChecked(True)
            chk.setStyleSheet("margin-left: 6px;")
            self._table.setCellWidget(row, self._COL_CHK, chk)

            def _num(v, fmt="{:.3f}"):
                it = QTableWidgetItem()
                it.setData(Qt.DisplayRole, float(fmt.format(v).replace(",", "")))
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                return it

            cid = str(c.get('id', ''))
            id_item = _ro(cid[:8])
            # Stash the candidate index so sorted rows still map back
            id_item.setData(Qt.UserRole, i)
            self._table.setItem(row, self._COL_ID, id_item)

            n = c.get('target_N', 0)
            sigma = c.get('layer_sigma', 0)
            self._table.setItem(row, self._COL_N,
                                _SortItem(f"{n} ±{sigma:.2f}", n))
            # ✓/— text, not colour (protanopia rule: never red/green-only)
            self._table.setItem(row, self._COL_RES,
                                _ro("✓" if c.get('layer_resolvable') else "—"))
            self._table.setItem(row, self._COL_AREA,
                                _num(c.get('area_um2', 0), "{:.0f}"))
            self._table.setItem(row, self._COL_SCORE, _num(c.get('score', 0)))
            self._table.setItem(row, self._COL_X, _num(c.get('x_mm', 0), "{:.4f}"))
            self._table.setItem(row, self._COL_Y, _num(c.get('y_mm', 0), "{:.4f}"))
            self._table.setItem(row, self._COL_FOCUS,
                                _ro("✓" if c.get('focus_ok') else "—"))

        self._table.setSortingEnabled(True)
        self._table.sortItems(self._COL_AREA, Qt.DescendingOrder)

    def _set_all(self, checked: bool):
        for row in range(self._table.rowCount()):
            w = self._table.cellWidget(row, self._COL_CHK)
            if w:
                w.setChecked(checked)

    def _candidate_for_row(self, row):
        """Return the candidate dict for the given visible table row, or None."""
        if row < 0 or row >= self._table.rowCount():
            return None
        # When sorting is enabled the visual row ≠ insertion order —
        # recover via the index stashed on the ID item.
        id_item = self._table.item(row, self._COL_ID)
        if id_item is None:
            return None
        idx = id_item.data(Qt.UserRole)
        if idx is None or not (0 <= idx < len(self._candidates)):
            return None
        return self._candidates[idx]

    # ── Navigation ──────────────────────────────────────────────────────────

    def _placement_dz(self):
        """Rigid-body focus shift of the current mount (mm; 0.0 unregistered)."""
        sample = getattr(self._sample_manager, '_sample', None) or {}
        return placement_dz_mm(sample)

    def _navigate_to(self, candidate):
        if self._motor_manager is None:
            return
        self._motor_manager.move_absolute_units('X', candidate['x_mm'])
        self._motor_manager.move_absolute_units('Y', candidate['y_mm'])
        z = candidate.get('z_mm')
        if z:   # scan-era focus + mount shift; falsy = no real focus recorded
            self._motor_manager.move_absolute_units('Z', z + self._placement_dz())

    def _on_double_click(self, row, _col):
        c = self._candidate_for_row(row)
        if c is not None:
            self._navigate_to(c)

    # ── Crop preview ─────────────────────────────────────────────────────────

    def _on_selection_changed(self):
        c = self._candidate_for_row(self._table.currentRow())
        if c is None:
            self._crop_preview.setText("Select a row to preview its crop")
            return
        pm = self._crop_pixmap(c)
        if pm is None:
            self._crop_preview.setText("no image")
            return
        self._crop_preview.setPixmap(pm.scaled(
            max(64, self._crop_preview.width() - 4),
            max(64, self._crop_preview.height() - 4),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _crop_pixmap(self, candidate):
        """Cached crop pixmap for a candidate (keyed by id, cap ~20)."""
        cid = str(candidate.get('id', ''))
        if cid and cid in self._crop_cache:
            return self._crop_cache[cid]
        pm = self._cut_crop(candidate)
        if pm is not None and cid:
            while len(self._crop_cache) >= _CROP_CACHE_MAX:
                self._crop_cache.pop(next(iter(self._crop_cache)))
            self._crop_cache[cid] = pm
        return pm

    def _cut_crop(self, candidate):
        """Cut a square crop around the candidate's bbox from its source tile
        (~25% margin, clamped to the frame), outline the detected contour in
        Okabe-Ito blue (protanopia rule: no red/green).  Returns a QPixmap or
        None (missing tile / fields) — never raises."""
        folder = self._resolved_folder
        src = candidate.get('source_image')
        bbox = candidate.get('bbox_px')
        if folder is None or not src or not bbox or len(bbox) != 4:
            return None
        try:
            import cv2
            import numpy as np
            tile = cv2.imread(str(Path(folder) / src))
            if tile is None:
                return None
            fh, fw = tile.shape[:2]
            x, y, w, h = (int(v) for v in bbox)
            side = max(int(max(w, h) * 1.5), 48)   # bbox + ~25% margin each side
            cx, cy = x + w // 2, y + h // 2
            x1 = min(fw, max(0, cx - side // 2) + side)
            y1 = min(fh, max(0, cy - side // 2) + side)
            x0 = max(0, x1 - side)
            y0 = max(0, y1 - side)
            crop = tile[y0:y1, x0:x1].copy()
            if crop.size == 0:
                return None
            cnt = candidate.get('contour_px')
            if cnt:
                pts = np.array([[int(px) - x0, int(py) - y0] for px, py in cnt],
                               dtype=np.int32)
                cv2.polylines(crop, [pts], True, _CONTOUR_BGR, 2, cv2.LINE_AA)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            ch, cw = rgb.shape[:2]
            qimg = QImage(rgb.data, cw, ch, 3 * cw, QImage.Format_RGB888)
            return QPixmap.fromImage(qimg.copy())   # copy: rgb buffer is local
        except Exception:
            return None

    # ── Add checked to Catalogue (the ONE import path — plan D1) ────────────

    def _add_checked_to_catalogue(self):
        """Import exactly the checked candidates via import_found_flakes --ids."""
        if self._import_proc is not None:
            return
        if self._resolved_folder is None:
            return
        ids = [str(c['id']) for c in self._checked_candidates() if c.get('id')]
        if not ids:
            QMessageBox.information(self, "Nothing checked",
                                    "Check at least one candidate to import.")
            return
        from core.pipeline_cmds import import_cmd
        cmd = import_cmd(self._resolved_folder, dedup_um=15.0, ids=ids,
                         unbuffered=True)
        p = QProcess(self)
        p.setProcessChannelMode(QProcess.MergedChannels)
        p.finished.connect(self._on_import_finished)
        self._import_proc = p
        self._import_btn.setEnabled(False)
        self._status.setText(
            f"Importing {len(ids)} checked candidate(s) to catalogue…")
        p.start(cmd[0], cmd[1:])

    def _on_import_finished(self, code, _status):
        out = bytes(self._import_proc.readAllStandardOutput()).decode(
            'utf-8', 'replace')
        self._import_proc = None
        self._import_btn.setEnabled(bool(self._candidates))
        if code != 0:
            print(out)
            self._status.setText(f"Import failed (exit {code}). See terminal.")
            return
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        summary = next((ln for ln in reversed(lines)
                        if ln.startswith('imported ')),
                       lines[-1] if lines else "ok")
        self._status.setText(f"Import done — {summary}"[:200])
        # Refresh the catalogue view so the imported flakes appear
        fn = getattr(self._sample_manager, '_open_sample', None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass

    # ── Capture Tour (sparse high-mag map layers) ──────────────────────────────

    def _checked_candidates(self):
        result = []
        for row in range(self._table.rowCount()):
            w = self._table.cellWidget(row, self._COL_CHK)
            if w and w.isChecked():
                c = self._candidate_for_row(row)
                if c is not None:
                    result.append(c)
        return result

    def _tour_output(self):
        """(scans_dir Path, sample_name) for tour capture layers, or (None, None).

        Captures go into the sample's standard scan hierarchy so make_map
        layers them onto the mother map — not an isolated map."""
        if self._sample_manager is not None:
            out_dir, name = self._sample_manager.scan_output_info()
            if out_dir:
                return Path(out_dir), (name or 'sample')
        if self._resolved_folder is not None:
            # .../scans/{mag}/{scan}/ → scans dir is two up; name three up.
            scans = self._resolved_folder.parent.parent
            return scans, scans.parent.name
        return None, None

    def _toggle_tour(self):
        if self._tour_runner is not None:
            self._tour_runner.end_tour()
            return

        checked = self._checked_candidates()
        if not checked:
            QMessageBox.information(self, "Nothing selected",
                                    "Check at least one candidate to visit.")
            return

        scans_dir, sample_name = self._tour_output()
        if scans_dir is None:
            QMessageBox.warning(self, "No output folder",
                                "Cannot determine where to save tour tiles.\n"
                                "Set the scan folder or load a sample first.")
            return

        from datetime import datetime
        tour_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._tour_runner = _TourRunner(
            checked, scans_dir, sample_name, tour_ts,
            self._motor_manager, self._preview,
            autofocus_panel=self._autofocus_panel,
            automatic=self._auto_check.isChecked(),
            autofocus=self._af_check.isChecked(),
            dz_mm=self._placement_dz(),
            parent=self,
        )
        self._tour_runner.progress.connect(self._on_tour_progress)
        self._tour_runner.stop_changed.connect(self._on_tour_stop_changed)
        self._tour_runner.finished.connect(self._on_tour_finished)

        self._tour_btn.setText("End Tour")
        self._status.setText(f"Tour: 0/{len(checked)} — starting…")
        self._tour_runner.start()

    def _on_tour_progress(self, current, total, msg):
        self._status.setText(f"Tour: {msg}")
        self._progress.setMaximum(total)
        self._progress.setValue(current)
        self._progress.setVisible(True)

    def _on_tour_stop_changed(self, idx, total):
        """Keep Previous / Next / Capture matched to the runner's state:
        live in manual mode (and when an automatic tour pauses on arrival
        timeout), dark while the machine is driving."""
        r = self._tour_runner
        if r is None:
            return
        manual = r.manual_controls_active
        self._prev_btn.setEnabled(manual and idx > 0)
        self._next_btn.setEnabled(manual and idx + 1 < total)
        self._capture_btn.setEnabled(manual)

    def _tour_next(self):
        if self._tour_runner is not None:
            self._tour_runner.next_stop()

    def _tour_prev(self):
        if self._tour_runner is not None:
            self._tour_runner.prev_stop()

    def _tour_capture(self):
        if self._tour_runner is not None:
            self._tour_runner.capture_now()

    def _on_tour_finished(self, _scans_dir):
        r = self._tour_runner
        n = r.captured_count if r is not None else 0
        mags = r.captured_mags if r is not None else []
        self._tour_runner = None
        self._progress.setVisible(False)
        self._next_btn.setEnabled(False)
        self._prev_btn.setEnabled(False)
        self._capture_btn.setEnabled(False)
        self._tour_btn.setText("Start Tour")
        layers = ", ".join(mags) if mags else "no"
        self._status.setText(
            f"Tour done — {n} tile(s) saved as {layers} scan layer(s).  "
            f"Re-run Generate Map to layer them onto the mother map.")

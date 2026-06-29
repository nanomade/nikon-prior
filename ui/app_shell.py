# ui/app_shell.py
"""
AppShell — three-column main window for nikon-prior.

  [ SCOPE panel ] [ Preview ] [ PROCESS panel ]

Both side panels are scrollable accordions of collapsible sections.
Sections contain either inline controls or buttons that open existing
floating panels.  All existing panels remain fully functional; this
shell is strictly additive.

Ported from standa-stacker's ui/app_shell.py and adapted to the Prior
ProScan III stage: there is no manipulator (dX/dY/dZ), no rotation (R)
axis, no eucentric/heater/slide hardware, so those sections and the
manipulator inline widgets are dropped.  The stage inline widgets are
rewritten against StageControlWindow's Nikon API (µm sliders,
get_position_units, move_absolute_xy_units, jog_axis).
"""

import time

from PyQt5.QtCore import Qt, QCoreApplication, QTimer
from PyQt5.QtWidgets import (
    QApplication, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QPushButton, QScrollArea, QSplitter, QVBoxLayout, QWidget,
)

_UM_PER_MM = 1000.0


# ── Accordion primitives ──────────────────────────────────────────────────────

class _Section(QWidget):
    """A labelled collapsible section with a toggle-button header."""

    def __init__(self, title, content, expanded=True, sub=False, parent=None):
        super().__init__(parent)
        self._title = title
        # sub may be bool (False/True → depth 0/1) or int depth (0–3)
        self._depth = int(sub)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._btn = QPushButton()
        self._btn.setCheckable(True)
        self._btn.setChecked(expanded)
        indent = 6 + self._depth * 12
        self._btn.setStyleSheet(f"text-align: left; padding-left: {indent}px;")
        self._btn.clicked.connect(self._toggle)
        lay.addWidget(self._btn)

        self._content = content
        lay.addWidget(content)
        content.setVisible(expanded)
        self._refresh()

    def _refresh(self):
        arrow = "▼" if self._btn.isChecked() else "▶"
        label = self._title if self._depth > 0 else self._title.upper()
        self._btn.setText(f"  {arrow}  {label}")

    def _toggle(self, checked):
        self._content.setVisible(checked)
        self._refresh()


class _SidePanel(QScrollArea):
    """Scrollable column of _Section widgets."""

    def __init__(self, heading=None):
        super().__init__()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMinimumWidth(160)
        self.setFrameShape(QFrame.NoFrame)

        self._w = QWidget()
        self._lay = QVBoxLayout(self._w)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(0)

        if heading:
            lbl = QLabel(heading)
            lbl.setContentsMargins(6, 4, 0, 4)
            self._lay.addWidget(lbl)
        self._lay.addStretch(1)
        self.setWidget(self._w)

    def add_section(self, title, widget, expanded=True):
        section = _Section(title, widget, expanded=expanded)
        self._lay.insertWidget(self._lay.count() - 1, section)
        return section


# ── Content helpers ───────────────────────────────────────────────────────────

def _show_on_screen(panel):
    """Show a floating panel, clamping its position to the available screen area."""
    panel.show()
    panel.raise_()
    panel.activateWindow()
    fg = panel.frameGeometry()
    screen = QApplication.screenAt(fg.center()) or QApplication.primaryScreen()
    avail = screen.availableGeometry()
    x = max(avail.left(), min(fg.left(), avail.right() - fg.width()))
    y = max(avail.top(), min(fg.top(), avail.bottom() - fg.height()))
    if x != fg.left() or y != fg.top():
        panel.move(x, y)


def _btn_panel(*pairs):
    """QWidget with one Open button per (label, panel) pair; None pairs skipped."""
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(6, 4, 6, 6)
    lay.setSpacing(3)
    for label, panel in pairs:
        if panel is None:
            continue
        b = QPushButton(label)
        b.clicked.connect(lambda _, p=panel: _show_on_screen(p))
        lay.addWidget(b)
    lay.addStretch(1)
    return w


def _open_btn(label, panel):
    b = QPushButton(label)
    b.clicked.connect(lambda _, p=panel: _show_on_screen(p))
    return b


# ── Stage inline widgets (Nikon ProScan III) ───────────────────────────────────

class _MiniStageMap(QLabel):
    """Scaled mirror of StageControlWindow.stage_display; click/drag to move.

    Grabs the already-rendered pixmap every 250 ms and scales it to fit.
    Click-to-move recomputes the target in mm from the click fraction using
    the same travel limits as InteractiveStageDisplay, so the mapping is
    scale-invariant: clicking the same relative position on the minimap and
    the full map commands the same stage coordinate.
    """

    def __init__(self, stage_ctrl, size=320, parent=None):
        super().__init__(parent)
        self._sc = stage_ctrl
        self.setFixedSize(size, size)
        self.setMouseTracking(True)
        self._dragging = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._sync)
        self._timer.start(250)

    def _sync(self):
        pm = self._sc.stage_display.pixmap()
        if pm and not pm.isNull():
            self.setPixmap(pm.scaled(
                self.width(), self.height(),
                Qt.IgnoreAspectRatio, Qt.SmoothTransformation,
            ))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._move_to(event.pos())

    def mouseMoveEvent(self, event):
        if self._dragging and (event.buttons() & Qt.LeftButton):
            self._move_to(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False

    def _move_to(self, pos):
        sc = self._sc
        w, h = self.width(), self.height()
        if not w or not h:
            return
        x_travel = sc._x_max_um - sc._x_min_um
        y_travel = sc._y_max_um - sc._y_min_um
        x_um = pos.x() / w * x_travel + sc._x_min_um
        y_um = (1.0 - pos.y() / h) * y_travel + sc._y_min_um   # Y-flip: top = +Y
        x_mm = max(sc._x_min_um, min(sc._x_max_um, x_um)) / _UM_PER_MM
        y_mm = max(sc._y_min_um, min(sc._y_max_um, y_um)) / _UM_PER_MM

        sc.stage_x_slider.blockSignals(True)
        sc.stage_y_slider.blockSignals(True)
        sc.stage_x_slider.setValue(int(round(x_mm * _UM_PER_MM)))
        sc.stage_y_slider.setValue(int(round(y_mm * _UM_PER_MM)))
        sc.stage_x_slider.blockSignals(False)
        sc.stage_y_slider.blockSignals(False)
        sc._move_xy(x_mm, y_mm)


class _InlineStageReadout(QWidget):
    """Compact X/Y/Z position readout + goto text entry for the stage."""

    def __init__(self, stage_ctrl, parent=None):
        super().__init__(parent)
        self._sc = stage_ctrl

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(2)

        pos_row = QHBoxLayout()
        self._lbl = {}
        for ax in ("X", "Y", "Z"):
            lbl = QLabel(f"{ax}: —")
            lbl.setMinimumWidth(78)
            self._lbl[ax] = lbl
            pos_row.addWidget(lbl)
            pos_row.addStretch(1)
        lay.addLayout(pos_row)

        goto_row = QHBoxLayout()
        goto_row.addWidget(QLabel("→"))
        self._edits = {}
        for ax in ("X", "Y", "Z"):
            goto_row.addWidget(QLabel(f"{ax}:"))
            ed = QLineEdit()
            ed.setFixedWidth(54)
            ed.setPlaceholderText("mm")
            ed.returnPressed.connect(self._goto_all)
            goto_row.addWidget(ed)
            self._edits[ax] = ed
        go_btn = QPushButton("Go")
        go_btn.setFixedWidth(32)
        go_btn.clicked.connect(self._goto_all)
        goto_row.addWidget(go_btn)
        goto_row.addStretch(1)
        lay.addLayout(goto_row)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update)
        self._timer.start(200)
        self._update()

    def _update(self):
        mm = self._sc.motor_manager
        for ax in ("X", "Y", "Z"):
            try:
                v = mm.get_position_units(ax)
                if v is not None:
                    self._lbl[ax].setText(f"{ax}: {v:+.3f} mm")
            except Exception:
                pass

    def _goto_all(self):
        sc = self._sc
        mm = sc.motor_manager

        def _val(ax):
            t = self._edits[ax].text().strip()
            if not t:
                return None
            try:
                return float(t)
            except ValueError:
                return None

        x, y, z = _val("X"), _val("Y"), _val("Z")
        try:
            if x is not None or y is not None:
                # Fill the unspecified axis with its current position so the
                # combined XY command never moves an axis the user left blank.
                if x is None:
                    x = mm.get_position_units("X") or 0.0
                if y is None:
                    y = mm.get_position_units("Y") or 0.0
                sc.stage_x_slider.blockSignals(True)
                sc.stage_y_slider.blockSignals(True)
                sc.stage_x_slider.setValue(int(round(x * _UM_PER_MM)))
                sc.stage_y_slider.setValue(int(round(y * _UM_PER_MM)))
                sc.stage_x_slider.blockSignals(False)
                sc.stage_y_slider.blockSignals(False)
                sc._move_xy(x, y)
            if z is not None:
                mm.move_absolute_units("Z", z, wait=False)
            sc._last_cmd_time = time.time()
            sc.update_all_displays()
        except Exception as exc:
            print(f"[inline stage goto] {exc}")


class _InlineStageJog(QWidget):
    """Compact XY directional pad + Z column with independent step selectors."""

    def __init__(self, stage_ctrl, parent=None):
        super().__init__(parent)
        self._sc = stage_ctrl
        self._xy_step = 10     # µm per click
        self._z_step = 10      # µm per click

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 6)
        lay.setSpacing(4)

        def _step_selector(label, default, options, setter):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            btns = {}
            for lbl, val in options:
                b = QPushButton(lbl)
                b.setCheckable(True)
                b.setChecked(val == default)
                b.setFixedWidth(46)
                b.clicked.connect(lambda _, v=val, s=setter, bs=btns: (
                    s(v),
                    [bb.setChecked(k == v) for k, bb in bs.items()]
                ))
                row.addWidget(b)
                btns[val] = b
            row.addStretch(1)
            return row

        lay.addLayout(_step_selector(
            "XY:", 10, (("1µm", 1), ("10µm", 10), ("100µm", 100)),
            lambda v: setattr(self, '_xy_step', v)
        ))
        lay.addLayout(_step_selector(
            "Z:", 10, (("1µm", 1), ("10µm", 10), ("100µm", 100)),
            lambda v: setattr(self, '_z_step', v)
        ))

        pad_row = QHBoxLayout()
        pad_row.setSpacing(10)

        xy_grid = QGridLayout()
        xy_grid.setSpacing(2)
        for row, col, label, axis, sign in (
            (0, 1, "↑", "Y", +1),
            (1, 0, "←", "X", -1),
            (1, 2, "→", "X", +1),
            (2, 1, "↓", "Y", -1),
        ):
            b = QPushButton(label)
            b.setFixedSize(34, 30)
            b.clicked.connect(
                lambda _, a=axis, s=sign: self._sc.jog_axis(a, s * self._xy_step))
            xy_grid.addWidget(b, row, col)
        pad_row.addLayout(xy_grid)

        z_col = QVBoxLayout()
        z_col.setSpacing(2)
        z_col.addWidget(QLabel("Z"))
        z_up = QPushButton("Z+")
        z_up.setFixedWidth(46)
        z_up.clicked.connect(lambda: self._sc.jog_axis("Z", +self._z_step))
        z_dn = QPushButton("Z−")
        z_dn.setFixedWidth(46)
        z_dn.clicked.connect(lambda: self._sc.jog_axis("Z", -self._z_step))
        z_col.addWidget(z_up)
        z_col.addWidget(z_dn)
        z_col.addStretch(1)
        pad_row.addLayout(z_col)
        pad_row.addStretch(1)
        lay.addLayout(pad_row)


def _motion_controls_panel(stage_controls, position_manager):
    """Motion Controls: a single Stage sub-rollout (no manipulator on Prior)."""
    outer = QWidget()
    lay = QVBoxLayout(outer)
    lay.setContentsMargins(8, 0, 0, 0)
    lay.setSpacing(0)

    if stage_controls is not None:
        stage_w = QWidget()
        stage_v = QVBoxLayout(stage_w)
        stage_v.setContentsMargins(6, 2, 6, 4)
        stage_v.setSpacing(3)

        stage_v.addWidget(_MiniStageMap(stage_controls, size=320))
        stage_v.addWidget(_InlineStageReadout(stage_controls))
        stage_v.addWidget(_Section("Jog", _InlineStageJog(stage_controls),
                                   expanded=False, sub=2))
        stage_v.addWidget(_open_btn("Stage Controls…", stage_controls))
        if position_manager is not None:
            stage_v.addWidget(_Section("Positions", position_manager,
                                       expanded=False, sub=True))
        stage_v.addStretch(1)
        lay.addWidget(_Section("Stage", stage_w, expanded=True, sub=True))

    lay.addStretch(1)
    return outer


# ── Shell ─────────────────────────────────────────────────────────────────────

class AppShell(QMainWindow):
    """
    Main application window.  The preview is embedded in the centre of a
    QSplitter; scope-config sections live on the left, process sections on
    the right.  All existing floating panels remain accessible.
    """

    def __init__(self, preview, launcher, controls,
                 stage_controls=None, position_manager=None,
                 autofocus_panel=None, focus_panel=None, focus_map_panel=None,
                 flat_field_panel=None, layer_contrast_panel=None,
                 index_mark_panel=None, edge_detection_panel=None,
                 wafer_mapping_panel=None, file_save_panel=None,
                 gamepad_panel=None, pixel_panel=None):
        super().__init__()
        self.setWindowTitle("Nikon / Prior ProScan III")
        self._wafer_mapping_panel = wafer_mapping_panel

        # ── Left: SCOPE ───────────────────────────────────────────────────────
        left = _SidePanel("SCOPE")

        def _imaging_panel():
            w = QWidget()
            v = QVBoxLayout(w)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(0)
            v.addWidget(controls)
            if file_save_panel is not None:
                v.addWidget(_Section("File Save", file_save_panel,
                                     expanded=False, sub=True))
            return w

        left.add_section("Imaging", _imaging_panel(), expanded=True)

        left.add_section(
            "Motion Controls",
            _motion_controls_panel(stage_controls, position_manager),
            expanded=True,
        )

        if autofocus_panel is not None:
            left.add_section("Autofocus", autofocus_panel, expanded=False)

        def _focus_panel():
            w = QWidget()
            v = QVBoxLayout(w)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(0)
            if focus_panel is not None:
                v.addWidget(_Section("Focus Presets", focus_panel,
                                     expanded=True, sub=True))
            if focus_map_panel is not None:
                v.addWidget(_Section("Focus Map", focus_map_panel,
                                     expanded=False, sub=True))
            return w

        if focus_panel is not None or focus_map_panel is not None:
            left.add_section("Focus", _focus_panel(), expanded=False)

        left.add_section(
            "Calibration",
            _btn_panel(
                ("Flat-Field Correction", flat_field_panel),
            ),
            expanded=False,
        )

        left.add_section(
            "Advanced",
            _btn_panel(
                ("Launcher (all panels)", launcher),
                ("Gamepad", gamepad_panel),
                ("Pixel Intensity", pixel_panel),
            ),
            expanded=False,
        )

        # ── Right: PROCESS ────────────────────────────────────────────────────
        right = _SidePanel("PROCESS")

        right.add_section(
            "Sample",
            _btn_panel(("Find Wafer Extents", edge_detection_panel)),
            expanded=True,
        )

        def _find_panel():
            w = QWidget()
            v = QVBoxLayout(w)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(0)
            if index_mark_panel is not None:
                v.addWidget(_Section("Index Marks",
                                     _btn_panel(("Index Mark Navigator…",
                                                 index_mark_panel)),
                                     expanded=False, sub=True))
            if layer_contrast_panel is not None:
                v.addWidget(_Section("Layer Contrast", layer_contrast_panel,
                                     expanded=False, sub=True))
            return w

        right.add_section("Find", _find_panel(), expanded=False)

        def _record_panel():
            w = QWidget()
            v = QVBoxLayout(w)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(0)
            if wafer_mapping_panel is not None:
                v.addWidget(_Section("Wafer Mapping", wafer_mapping_panel,
                                     expanded=True, sub=True))
            return w

        right.add_section("Record", _record_panel(), expanded=False)

        # ── Splitter ──────────────────────────────────────────────────────────
        # The preview is added directly so it fills the central pane: its
        # image_label has an Expanding size policy and update_frame scales each
        # frame to the label's live size.  (No centering wrapper — that is for
        # standa's fixed-size preview and would pin this one to its minimum.)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(preview)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([400, 900, 400])

        self.setCentralWidget(splitter)

    def closeEvent(self, event):
        # Join any running wafer-scan worker before teardown — destroying a
        # live QThread aborts the process.
        if self._wafer_mapping_panel is not None:
            for attr in ("worker", "_worker", "scan_worker"):
                wk = getattr(self._wafer_mapping_panel, attr, None)
                if wk is not None:
                    try:
                        wk.requestInterruption()
                        wk.wait(2000)
                    except Exception:
                        pass
        event.accept()
        QCoreApplication.quit()

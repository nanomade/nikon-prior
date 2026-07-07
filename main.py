"""Nikon / Prior ProScan III microscope control application.

Entry point.  Run with:
    python main.py

Note on Qt / cv2 conflict
--------------------------
opencv-python bundles its own Qt and registers cv2/qt/plugins as a Qt
platform-plugin search path at import time.  If cv2 is imported before
QApplication is created, its bundled (often ABI-incompatible) xcb plugin
can win the search and crash the app with:
  "Could not load the Qt platform plugin xcb … Aborted (core dumped)"
The moveToThread warning at startup is a related symptom (cv2 creates Qt
objects internally during import when no QApplication exists yet).

Fix applied here: create QApplication before importing any module that
imports cv2 at module level (stage_controls, autofocus_panel, etc.).
Once QApplication is constructed, PyQt5's platform plugin is already
loaded and cv2's path registration has no effect.

Permanent alternative: pip install opencv-python-headless
(headless OpenCV has no bundled Qt, so the conflict never arises).
"""

import logging
import os
import sys

from PyQt5.QtWidgets import QApplication, QSplashScreen

# ── Create QApplication NOW, before any cv2-importing module is loaded ──────
# This must come before all other local imports.
_qapp = QApplication(sys.argv)

# Link the running window to its launcher so GNOME shows the app icon instead
# of the generic gear. applicationName sets the window's WM_CLASS class part
# (default would be "main.py", which StartupWMClass can't match) and
# desktopFileName is the direct hint, keyed to nikon-prior.desktop's basename.
_qapp.setApplicationName("nikon-prior")
_qapp.setDesktopFileName("nikon-prior")

from PyQt5.QtCore import QPoint, Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QPainter, QPixmap

from controller import Controller
from motors.factory import create_motor_manager
from pixel_intensity_panel import PixelIntensityPanel
from ui.app_shell import AppShell
from ui.autofocus_panel import AutoFocusPanel
from ui.controls import ControlWindow
from ui.edge_detection_panel import EdgeDetectionPanel
from ui.file_save_panel import FileSavePanel
from ui.flat_field_panel import FlatFieldPanel
from ui.focus_map_panel import FocusMapPanel
from ui.focus_panel import FocusPanel
from ui.gamepad_panel import GamepadPanel
from ui.index_mark_panel import IndexMarkPanel
from ui.launcher import LauncherWindow
from ui.layer_contrast_panel import LayerContrastPanel
from ui.preview import PreviewWindow
from ui.stage_controls import PositionManagerWindow, StageControlWindow
from ui.wafer_mapping_panel import WaferMappingPanel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Splash screen (ported from standa_stacker) ──────────────────────────────
# Optional artwork: drop a PNG at this path and it will be used automatically
# (scaled to _SPLASH_W, with a darkened title banner). Until then the rendered
# no-picture panel below is the splash.
_SPLASH_IMAGE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets", "nikon-prior-splash.png")
_SPLASH_W = 560   # on-screen splash width; height follows the image aspect ratio


def _render_fallback_splash() -> QPixmap:
    """Self-contained splash pixmap (no background picture)."""
    w, h = 480, 280
    pix = QPixmap(w, h)
    pix.fill(QColor("#15181d"))
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QColor("#3a4250"))
    p.drawRect(0, 0, w - 1, h - 1)
    p.setPen(QColor("#e8eef6"))
    p.setFont(QFont("Sans", 26, QFont.Bold))
    p.drawText(0, 70, w, 50, Qt.AlignHCenter, "Nikon-Prior")
    p.setPen(QColor("#8aa0bf"))
    p.setFont(QFont("Sans", 11))
    p.drawText(0, 120, w, 24, Qt.AlignHCenter,
               "Microscope control — Nikon L200ND · Prior ProScan III")
    p.end()
    return pix


def _make_splash() -> QSplashScreen:
    """Build the splash screen: assets image if present, else the rendered panel.

    Shown before the slow start-up work (serial stage connect, camera init,
    panel construction) so the user gets immediate feedback that the app is
    coming up instead of a blank screen for several seconds.
    """
    pix = QPixmap(_SPLASH_IMAGE)
    if pix.isNull():
        # Expected for now — no artwork yet; the rendered panel is the splash.
        pix = _render_fallback_splash()
    else:
        pix = pix.scaledToWidth(_SPLASH_W, Qt.SmoothTransformation)
        # Darken a banner along the bottom so the overlaid title and the live
        # status message stay legible regardless of the underlying image.
        w, h = pix.width(), pix.height()
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        banner_h = 104
        banner_top = h - banner_h
        p.fillRect(0, banner_top, w, banner_h, QColor(0, 0, 0, 160))
        p.setPen(QColor("#e8eef6"))
        p.setFont(QFont("Sans", 24, QFont.Bold))
        p.drawText(0, banner_top + 8, w, 44, Qt.AlignHCenter, "Nikon-Prior")
        p.setPen(QColor("#8aa0bf"))
        p.setFont(QFont("Sans", 11))
        p.drawText(0, banner_top + 52, w, 22, Qt.AlignHCenter,
                   "Microscope control — Nikon L200ND · Prior ProScan III")
        p.end()
    return QSplashScreen(pix, Qt.WindowStaysOnTopHint)


def _tile_windows(widgets, start=QPoint(40, 40), gap=20):
    """Lay out widgets left-to-right, wrapping within screen bounds."""
    screen_rect = QApplication.primaryScreen().availableGeometry()
    x, y, row_h = start.x(), start.y(), 0
    for w in widgets:
        hint = w.sizeHint()
        if not hint.isEmpty():
            w.resize(hint)
        if x + w.width() > screen_rect.right() - gap:
            x = start.x()
            y += row_h + gap
            row_h = 0
        w.move(x, y)
        x += w.width() + gap
        row_h = max(row_h, w.height())


class Application:
    def __init__(self):
        self.app = QApplication.instance()  # reuse the one created at module level

        # Splash up front: start-up (serial stage connect, camera init, panels)
        # takes several seconds, so show feedback immediately. processEvents()
        # forces a repaint before each blocking step (the message can't animate
        # *during* a step — the event loop is busy — but updating it before
        # each step is enough).
        self._splash = _make_splash()
        self._splash.show()
        self.app.processEvents()

        self._splash_msg("Connecting ProScan III stage…")
        self.motor_manager = create_motor_manager()
        self.controller = Controller()
        self._splash_msg("Opening Alvium camera & building interface…")
        self._init_windows()

    def _splash_msg(self, text: str):
        sp = getattr(self, "_splash", None)
        if sp is None:
            return
        # The animator owns the trailing dots, so strip any ellipsis/dots the
        # caller supplied and let _tick_splash_dots() cycle them.
        self._splash_base = text.rstrip(" .…")
        self._splash_dot = 0
        # Lazily start a GUI-thread timer that cycles "" → "." → ".." → "..."
        # on the current message.  It only visibly advances while the event
        # loop spins (i.e. between blocking startup steps), but it animates
        # the gaps for free.  Parented to the splash so it's stopped and
        # destroyed on the GUI thread with it.
        if getattr(self, "_splash_timer", None) is None:
            self._splash_timer = QTimer(sp)
            self._splash_timer.timeout.connect(self._tick_splash_dots)
            self._splash_timer.start(350)
        self._tick_splash_dots()
        self.app.processEvents()

    def _tick_splash_dots(self):
        sp = getattr(self, "_splash", None)
        if sp is None:
            return
        self._splash_dot = (getattr(self, "_splash_dot", 0) + 1) % 4
        sp.showMessage("   " + self._splash_base + "." * self._splash_dot,
                       Qt.AlignBottom | Qt.AlignLeft, QColor("#cfe0f5"))

    def _init_windows(self):
        mm = self.motor_manager

        self.preview = PreviewWindow(self.controller)
        self.preview.motor_manager = mm
        self.controls = ControlWindow(self.controller, self.preview)
        self._splash_msg("Building panels…")

        self.stage_controls = StageControlWindow(self.preview, mm)
        self.preview.stage_controls = self.stage_controls
        self.position_manager = PositionManagerWindow(self.stage_controls)
        self.stage_controls.position_manager = self.position_manager

        self.focus_panel = FocusPanel(self.stage_controls)
        self.stage_controls.focus_panel = self.focus_panel

        self.autofocus_panel = AutoFocusPanel(
            motor_manager=mm,
            preview_obj=self.preview,
            stage_controls=self.stage_controls,
        )
        self.controller.magnification_changed.connect(
            self.autofocus_panel.apply_defaults_for_mag
        )
        # Enter key over the preview runs autofocus (preview.keyPressEvent)
        self.preview.autofocus_panel = self.autofocus_panel

        self.focus_map_panel = FocusMapPanel(
            motor_manager=mm,
            preview=self.preview,
            autofocus_panel=self.autofocus_panel,
            wafer_mapping_panel=None,
        )

        self.wafer_mapping_panel = WaferMappingPanel(
            self.preview, mm, self.stage_controls,
            autofocus_panel=self.autofocus_panel,
            focus_map_panel=self.focus_map_panel,
            controls=self.controls,
        )
        self.focus_map_panel.wafer_mapping_panel = self.wafer_mapping_panel

        self.layer_contrast_panel = LayerContrastPanel(self.preview)
        self.flat_field_panel = FlatFieldPanel(self.preview, mm)
        self.pixel_panel = PixelIntensityPanel(mm, self.preview)
        self.index_mark_panel = IndexMarkPanel(
            self.preview, mm, self.stage_controls
        )

        self.edge_detection_panel = EdgeDetectionPanel(
            self.preview, mm,
            wafer_mapping_panel=self.wafer_mapping_panel,
        )

        self.file_save_panel = FileSavePanel(
            self.preview, mm, self.controls,
        )

        self.gamepad_panel = GamepadPanel(
            self.stage_controls, self.controller,
            autofocus_panel=self.autofocus_panel,
            preview=self.preview,
            controls=self.controls,
        )

        self.launcher = LauncherWindow(
            self.preview,
            self.controls,
            self.stage_controls,
            self.position_manager,
            focus_panel=self.focus_panel,
            autofocus_panel=self.autofocus_panel,
            focus_map_panel=self.focus_map_panel,
            wafer_mapping_panel=self.wafer_mapping_panel,
            index_mark_panel=self.index_mark_panel,
            gamepad_panel=self.gamepad_panel,
            pixel_panel=self.pixel_panel,
            layer_contrast_panel=self.layer_contrast_panel,
            flat_field_panel=self.flat_field_panel,
            edge_detection_panel=self.edge_detection_panel,
            file_save_panel=self.file_save_panel,
        )

        # Three-column shell is the primary window.  The Launcher and all
        # floating panels remain available (opened on demand from the shell).
        self.shell = AppShell(
            preview=self.preview,
            launcher=self.launcher,
            controls=self.controls,
            stage_controls=self.stage_controls,
            position_manager=self.position_manager,
            autofocus_panel=self.autofocus_panel,
            focus_panel=self.focus_panel,
            focus_map_panel=self.focus_map_panel,
            flat_field_panel=self.flat_field_panel,
            layer_contrast_panel=self.layer_contrast_panel,
            index_mark_panel=self.index_mark_panel,
            edge_detection_panel=self.edge_detection_panel,
            wafer_mapping_panel=self.wafer_mapping_panel,
            file_save_panel=self.file_save_panel,
            gamepad_panel=self.gamepad_panel,
            pixel_panel=self.pixel_panel,
        )

    def run(self):
        self.app.setQuitOnLastWindowClosed(False)
        self.shell.destroyed.connect(self.app.quit)
        screen = QApplication.primaryScreen().availableGeometry()
        self.shell.move(screen.left(), screen.top())
        self.shell.resize(min(1700, screen.width()), min(1000, screen.height()))
        self._splash_msg("Opening main window…")
        self.shell.show()

        # Tear down the splash once the main window is up.  Stop the dot
        # animator on the GUI thread first (it's parented to the splash).
        if getattr(self, "_splash_timer", None) is not None:
            self._splash_timer.stop()
            self._splash_timer = None
        if getattr(self, "_splash", None) is not None:
            self._splash.finish(self.shell)
            self._splash = None

        sys.exit(self.app.exec_())


if __name__ == "__main__":
    Application().run()
